"""Batched held-out MARIE predictions, preserving per-window metric weights."""
import torch as th


def validation_windows(batches):
    for batch in batches:
        for episode in range(batch.batch_size):
            length = int(batch['filled'][episode].sum().item()) - 1
            if length < 1:
                continue
            width = min(15, length)
            for start in sorted(set((0, (length-width)//2, length-width))):
                yield {
                    name: batch[name][episode, start:start+width+extra]
                    for name, extra in [('obs',1), ('actions',0), ('reward',0),
                                        ('terminated',0), ('avail_actions',1)]
                }


def window_groups(batches, batch_size):
    pending = []
    for window in validation_windows(batches):
        if pending and (len(pending) == batch_size or
                        window['obs'].shape[0] != pending[0]['obs'].shape[0]):
            yield pending
            pending = []
        pending.append(window)
    if pending:
        yield pending


@th.no_grad()
def collect_validation(world, batches, n_agents, batch_size):
    """Caller owns eval mode and fork_rng. Batched draws retain their distribution.

    RNG realizations differ from serial validation when batch_size > 1. Restore
    batch_size=1 to recover serial draw order; never mix batch sizes within a
    convergence history without resetting that history.
    """
    if batch_size < 1:
        raise ValueError('marie_validation_batch_size must be positive')
    records = {h: [] for h in (1, 5, 15)}
    reconstruction_errors = []
    device = next(world.parameters()).device
    for windows in window_groups(batches, batch_size):
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
                hidden, cache = world.init_dynamics_cache(latent, action, focal)
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
                rows = th.stack(list(metrics.values()),1).cpu().tolist()
                records[horizon].extend(dict(zip(metrics, row)) for row in rows)
            latent = next_latent[:, None]
    return records, reconstruction_errors
