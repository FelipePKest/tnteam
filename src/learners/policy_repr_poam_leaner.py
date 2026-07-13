import torch as th
import torch.nn.functional as F
from torch.optim import Adam

from learners.poam_learner import POAMLearner
from utils.encoder_decoder import (
    build_decoder_inputs,
    build_decoder_targets,
    build_encoder_inputs,
)
from utils.policy_repr import (
    PolicyRepresentationHead,
    masked_accuracy,
    masked_cross_entropy,
    supervised_contrastive_loss,
)


class PolicyRepresentationPOAMLearner(POAMLearner):
    """
    POAM-style learner with a policy-representation auxiliary objective.

    Original POAM auxiliary objective:
        embedding -> decoder -> reconstruct/predict other agents' obs/actions

    This learner's auxiliary objective:
        embedding + current obs -> predict the represented agent's action

    Optionally, if the EpisodeBatch contains a "policy_type" field, it also
    learns a supervised type-prediction head.

    This keeps the actor path unchanged:
        RNNPOAMAgent still receives encoder embeddings as additional input.
    """

    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)

        obs_dim = scheme["obs"]["vshape"]
        n_policy_types = getattr(args, "n_policy_types", 0)

        # Replace POAM's decoder with a policy-representation head.
        self.repr_head = PolicyRepresentationHead(
            args=args,
            embed_dim=args.embed_dim,
            obs_dim=obs_dim,
            n_actions=args.n_actions,
            n_policy_types=n_policy_types,
        )

        # Optimize the shared encoder, POAM decoder, and policy-representation head together.
        self.repr_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.repr_head.parameters())
        )

        self.repr_optimiser = Adam(
            params=self.repr_params,
            lr=getattr(args, "repr_lr", args.ed_lr),
            eps=args.optim_eps,
        )

        # POAMLearner's train() expects ed_* names. Keep ed_obs_loss and
        # ed_act_loss as the original POAM decoder components, and add
        # repr_* stats for the policy-representation supervision.

    def encoder_decoder_update(self, batch, mask, encoder_decoder_train_stats, t_env):
        """
        Override POAMLearner.encoder_decoder_update.

        This method is called by POAMLearner.train(), so we keep the same
        method name and the same stats dictionary interface.
        """

        max_t = batch.max_seq_length - 1

        for key in [
            "repr_action_loss",
            "repr_type_loss",
            "repr_type_acc",
            "repr_contrastive_loss",
            "decoder_loss",
        ]:
            encoder_decoder_train_stats.setdefault(key, [])

        # Encoder input matches POAM:
        #   [obs_t, previous_action_{t-1}]
        encoder_input = build_encoder_inputs(
            n_agents=self.n_agents,
            batch=batch,
            t=None,
            concat_obs_act=True,
        )
        encoder_input = encoder_input[:, :-1]

        obs = batch["obs"][:, :-1]
        actions = batch["actions"][:, :-1].squeeze(-1)

        # The base POAM learner pre-filters ED masks for open training. Build
        # decoder/repr masks from the raw episode mask here so the decoder scope
        # and representation scope can be controlled independently.
        terminated = batch["terminated"][:, :max_t].float()
        episode_mask = batch["filled"][:, :max_t].float()
        episode_mask[:, 1:] = episode_mask[:, 1:] * (1 - terminated[:, :-1])
        episode_mask = episode_mask.repeat(1, 1, self.n_agents)

        decoder_mask = episode_mask
        if self.args.open_train_or_eval:
            trainable_mask = batch["trainable_agents"][:, :max_t].squeeze(-1).float()
            decoder_scope = getattr(self.args, "decoder_agent_scope", "all")
            if decoder_scope == "controlled":
                decoder_mask = decoder_mask * trainable_mask
            elif decoder_scope == "uncontrolled":
                decoder_mask = decoder_mask * (1.0 - trainable_mask)
            elif decoder_scope != "all":
                raise ValueError(f"Unknown decoder_agent_scope: {decoder_scope}")

        repr_mask = episode_mask
        if self.args.open_train_or_eval and getattr(self.args, "repr_uncontrolled_only", False):
            repr_mask = repr_mask * (~batch["trainable_agents"][:, :max_t].squeeze(-1)).float()

        obs_targ, act_targ_onehot, mask_targ = build_decoder_targets(
            n_agents=self.n_agents,
            batch=batch,
            mask=decoder_mask,
            t=None,
            concat_agents=False,
            concat_obs_act=False,
        )
        obs_targ = obs_targ[:, :-1].detach()
        act_targ_onehot = act_targ_onehot[:, :-1].detach()

        # Optional labels. This requires adding "policy_type" to the batch scheme
        # and filling it in the open runner/controller.
        has_policy_type = False
        policy_type = None
        try:
            policy_type = batch["policy_type"][:, :max_t].squeeze(-1).long()
            has_policy_type = True
        except Exception:
            has_policy_type = False

        repr_epochs = getattr(self.args, "repr_epochs", self.args.ed_epochs)
        n_repr_minibatch = getattr(self.args, "n_repr_minibatch", self.args.n_ed_minibatch)

        for _ in range(repr_epochs):
            mb_rand = th.randperm(batch.batch_size).cpu().numpy()
            mb_size = max(1, batch.batch_size // n_repr_minibatch)

            sampler = [
                mb_rand[i * mb_size:(i + 1) * mb_size]
                for i in range(n_repr_minibatch)
            ]

            for indices in sampler:
                if len(indices) == 0:
                    continue

                encoder_input_mb = encoder_input[indices]
                obs_mb = obs[indices]
                actions_mb = actions[indices]
                repr_mask_mb = repr_mask[indices]
                obs_targ_mb = obs_targ[indices]
                act_targ_onehot_mb = act_targ_onehot[indices]
                mask_targ_mb = mask_targ[indices][:, :max_t]
                decoder_mask_mb = decoder_mask[indices]

                embeddings = self.encoder.forward_all(encoder_input_mb)

                decoder_inputs = build_decoder_inputs(embeddings)
                decoded_obs, decoded_act_logits = self.decoder(decoder_inputs)
                mb_size = len(indices)
                decoded_act_logits = decoded_act_logits.view(
                    mb_size,
                    -1,
                    self.n_agents,
                    self.n_agents - 1,
                    self.n_actions,
                )
                act_targ_onehot_mb = act_targ_onehot_mb.view(
                    mb_size,
                    -1,
                    self.n_agents,
                    self.n_agents - 1,
                    self.n_actions,
                )
                act_targ_mb = th.argmax(act_targ_onehot_mb, dim=-1)

                log_prob, _ = self.mac.action_selector.eval_action(
                    agent_inputs=decoded_act_logits,
                    actions=act_targ_mb,
                )
                log_prob = log_prob.unsqueeze(-1)

                if self.args.ed_model_uncontrolled_only:
                    obs_mask_mb = mask_targ_mb
                    act_mask_mb = mask_targ_mb
                else:
                    obs_mask_mb = decoder_mask_mb.unsqueeze(-1).unsqueeze(-1).repeat(
                        1,
                        1,
                        1,
                        self.n_agents - 1,
                        1,
                    )
                    act_mask_mb = mask_targ_mb

                if self.args.ed_bce_loss:
                    decoder_obs_loss = (
                        F.binary_cross_entropy(decoded_obs, obs_targ_mb, reduction="none")
                        * obs_mask_mb
                    ).sum() / (obs_mask_mb.sum() + 1e-8)
                else:
                    decoder_obs_loss = (
                        F.mse_loss(decoded_obs, obs_targ_mb, reduction="none")
                        * obs_mask_mb
                    ).sum() / (obs_mask_mb.sum() + 1e-8)

                decoder_act_loss = (
                    -log_prob * act_mask_mb
                ).sum() / (act_mask_mb.sum() + 1e-8)

                action_logits = self.repr_head.forward_action(
                    obs=obs_mb,
                    embedding=embeddings,
                )

                action_loss = masked_cross_entropy(
                    logits=action_logits,
                    targets=actions_mb,
                    mask=repr_mask_mb,
                    n_classes=self.n_actions,
                )

                if has_policy_type and self.repr_head.type_head is not None:
                    policy_type_mb = policy_type[indices]

                    type_logits = self.repr_head.forward_type(embeddings)

                    type_loss = masked_cross_entropy(
                        logits=type_logits,
                        targets=policy_type_mb,
                        mask=repr_mask_mb,
                        n_classes=self.args.n_policy_types,
                        ignore_index=-1,
                    )
                    type_acc = masked_accuracy(
                        logits=type_logits,
                        targets=policy_type_mb,
                        mask=repr_mask_mb,
                        ignore_index=-1,
                    )
                    contrastive_loss = supervised_contrastive_loss(
                        embeddings=embeddings,
                        labels=policy_type_mb,
                        mask=repr_mask_mb,
                        temperature=getattr(self.args, "repr_contrastive_temperature", 0.2),
                        max_samples=getattr(self.args, "repr_contrastive_max_samples", 2048),
                    )
                else:
                    type_loss = th.zeros_like(action_loss)
                    type_acc = th.zeros_like(action_loss)
                    contrastive_loss = th.zeros_like(action_loss)

                action_coef = getattr(self.args, "repr_action_coef", 1.0)
                type_coef = getattr(self.args, "repr_type_coef", 0.0)
                contrastive_coef = getattr(self.args, "repr_contrastive_coef", 0.0)
                decoder_obs_coef = getattr(self.args, "decoder_obs_coef", 1.0)
                decoder_act_coef = getattr(self.args, "decoder_act_coef", 1.0)

                decoder_loss = decoder_obs_coef * decoder_obs_loss + decoder_act_coef * decoder_act_loss
                loss = (
                    decoder_loss
                    + action_coef * action_loss
                    + type_coef * type_loss
                    + contrastive_coef * contrastive_loss
                )

                self.repr_optimiser.zero_grad()
                loss.backward()

                grad_norm = th.nn.utils.clip_grad_norm_(
                    self.repr_params,
                    self.args.ed_grad_norm_clip,
                )

                self.repr_optimiser.step()

                encoder_decoder_train_stats["ed_obs_loss"].append(decoder_obs_loss.item())
                encoder_decoder_train_stats["ed_act_loss"].append(decoder_act_loss.item())
                encoder_decoder_train_stats["ed_loss"].append(loss.item())
                encoder_decoder_train_stats["ed_grad_norm"].append(grad_norm.item())
                encoder_decoder_train_stats["decoder_loss"].append(decoder_loss.item())
                encoder_decoder_train_stats["repr_action_loss"].append(action_loss.item())
                encoder_decoder_train_stats["repr_type_loss"].append(type_loss.item())
                encoder_decoder_train_stats["repr_type_acc"].append(type_acc.item())
                encoder_decoder_train_stats["repr_contrastive_loss"].append(contrastive_loss.item())

        return encoder_decoder_train_stats

    def cuda(self):
        super().cuda()
        self.decoder.cuda()
        self.repr_head.cuda()
