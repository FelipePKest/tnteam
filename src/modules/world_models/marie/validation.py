"""Batched held-out MARIE predictions, preserving per-window metric weights."""
import torch as th


def validation_windows(batches, max_horizon=15, history_frames=1):
    for batch in batches:
        for episode in range(batch.batch_size):
            length = int(batch['filled'][episode].sum().item()) - 1
            if length < 1:
                continue
            width = min(max_horizon, length)
            for start in sorted(set((0, (length-width)//2, length-width))):
                window = {
                    name: batch[name][episode, start:start+width+extra]
                    for name, extra in [('obs',1), ('actions',0), ('reward',0),
                                        ('terminated',0), ('avail_actions',1)]
                }
                if 'trainable_agents' in batch.scheme:
                    window['trainable_agents']=batch['trainable_agents'][episode,start:start+width]
                if history_frames > 1:
                    left = max(0, start-history_frames+1)
                    past = start-left
                    window['context_obs'] = batch['obs'].new_zeros(
                        history_frames, *batch['obs'].shape[2:])
                    window['context_actions'] = batch['actions'].new_zeros(
                        history_frames-1, *batch['actions'].shape[2:])
                    window['context_valid'] = batch['filled'].new_zeros(history_frames, 1)
                    window['context_obs'][-past-1:] = batch['obs'][episode, left:start+1]
                    window['context_valid'][-past-1:] = batch['filled'][episode, left:start+1]
                    if past:
                        window['context_actions'][-past:] = batch['actions'][episode, left:start]
                yield window


def window_groups(batches, batch_size, max_horizon=15, history_frames=1):
    pending = []
    for window in validation_windows(batches, max_horizon, history_frames):
        if pending and (len(pending) == batch_size or
                        window['obs'].shape[0] != pending[0]['obs'].shape[0]):
            yield pending
            pending = []
        pending.append(window)
    if pending:
        yield pending


@th.no_grad()
def collect_validation(world, batches, n_agents, batch_size, horizons=(1, 5, 15)):
    """Caller owns eval mode and fork_rng. Batched draws retain their distribution.

    RNG realizations differ from serial validation when batch_size > 1. Restore
    batch_size=1 to recover serial draw order; never mix batch sizes within a
    convergence history without resetting that history.
    """
    if batch_size < 1:
        raise ValueError('marie_validation_batch_size must be positive')
    if not horizons or any(h < 1 for h in horizons):
        raise ValueError('Validation horizons must be positive')
    records = {h: [] for h in horizons}
    reconstruction_errors = []
    device = next(world.parameters()).device
    recurrent = getattr(world, 'recurrent_world_model', False)
    history_frames = world.recurrent_history_frames if recurrent else 1
    for windows in window_groups(batches, batch_size, max(horizons), history_frames):
        data = {name: th.stack([w[name] for w in windows]).to(device)
                for name in windows[0]}
        count, length = data['obs'].shape[:2]
        width = length - 1
        obs = data['obs'].transpose(1, 2).reshape(count*n_agents, length, -1)
        actions = data['actions'].transpose(1, 2).reshape(count*n_agents, width, -1)
        avail = data['avail_actions'].transpose(1, 2).reshape(count*n_agents, length, -1)
        target, _ = world.encode(obs, sample=False)
        errors = (world.decode(target)-obs).abs().reshape(count, -1).mean(-1)
        reconstruction_errors.extend(errors.cpu().tolist())
        latent = target[:, :1]
        focal = th.arange(n_agents, device=device).repeat(count)
        memory_options = {}
        if recurrent and history_frames > 1:
            context = data['context_obs'].transpose(1, 2).reshape(count*n_agents, history_frames, -1)
            past_actions = data['context_actions'].transpose(1, 2).reshape(count*n_agents, history_frames-1, 1)
            valid = data['context_valid'][:, :-1, 0].repeat_interleave(n_agents, dim=0).bool()
            encoded_context, _ = world.encode(context[:, :-1], sample=False)
            memory_options['recurrent_state'] = world.burn_in_memory(
                encoded_context, past_actions, focal, valid=valid)
        cache = None
        predicted_return = th.zeros(count*n_agents, 1, device=device)
        real_return = th.zeros_like(predicted_return)
        def per_window(value):
            return value.reshape(count, -1).float().mean(-1)
        def per_window_sum(value):
            return value.reshape(count, -1).sum(-1)
        for step in range(width):
            action = actions[:, step:step+1]
            if cache is None:
                hidden, cache = world.init_dynamics_cache(latent, action, focal, **memory_options)
                hidden = hidden[:, -1]
            else:
                hidden, cache = world.append_action_cache(latent, action, focal, cache)
            heads = world.prediction_auxiliary_heads(hidden, latent[:, -1])
            reward = world.reward_value(heads['reward_ensemble_logits'][0])
            reward_target = data['reward'][:, step, None].expand(-1,n_agents,-1).reshape(-1,1)
            terminal_target = data['terminated'][:, step, None].expand(-1,n_agents,-1).reshape(-1,1).bool()
            predicted_return += reward
            real_return += reward_target
            terminal = heads['continuation_logits'].sigmoid() < .5
            availability = world.predicted_availability_from_hidden(hidden)
            next_latent, _, cache = world.sample_next_latent_cached(hidden, cache)
            horizon = step+1
            if horizon in records:
                metrics = {
                    'observation_mae': per_window((world.decode(next_latent)-obs[:,horizon]).abs()),
                    'token_accuracy': per_window(next_latent.argmax(-1)==target[:,horizon].argmax(-1)),
                    'reward_mae': per_window((reward-reward_target).abs()),
                    'return_mae': per_window((predicted_return-real_return).abs()),
                    'availability_accuracy': per_window(availability.bool()==avail[:,horizon].bool()),
                    'terminal_tp': per_window_sum(terminal & terminal_target),
                    'terminal_predicted': per_window_sum(terminal),
                    'terminal_actual': per_window_sum(terminal_target),
                }
                if getattr(world,'full_team_world_model',False):
                    foreign=(~data['trainable_agents'][:,step,:,0].bool()).reshape(-1)
                    predicted=world.foreign_action_logits(latent[:,-1]).argmax(-1)
                    correct=(predicted==action.reshape(-1)) & foreign
                    metrics['foreign_action_accuracy']=(per_window_sum(correct.float()) /
                        per_window_sum(foreign.float()).clamp_min(1))
                rows = th.stack(list(metrics.values()),1).cpu().tolist()
                records[horizon].extend(dict(zip(metrics, row)) for row in rows)
            latent = next_latent[:, None]
    return records, reconstruction_errors
