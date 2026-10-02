"""MARIE temporal dynamics and imagined rollout API."""

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.matwm import MATWMWorldModel
from .attention import CachedCausalTransformer
from .tokenizer import MARIEVQTokenizer
from .aggregation import PerceiverAggregator

class MARIEWorldModel(MATWMWorldModel):
    """Decentralized temporal dynamics with centralized aggregation."""

    def __init__(self, obs_dim, n_agents, n_actions, args):
        super().__init__(obs_dim, n_agents, n_actions, args)
        # MATWM owns additional dynamics and teammate-prediction modules that
        # are not part of MARIE. Remove them before constructing the MARIE
        # model so they cannot enter checkpoints or optimizer groups.
        for name in (
            "action_mixer", "dynamics", "action_mask", "teammate_input",
            "teammate_position", "teammate_model", "teammate_head",
        ):
            delattr(self, name)
        self.sequence_model = CachedCausalTransformer(
            self.hidden_dim,
            getattr(args, "matwm_attention_heads", 8),
            getattr(args, "matwm_ff_dim", self.hidden_dim * 4),
            getattr(args, "matwm_transformer_layers", 2),
            getattr(args, "matwm_dropout", 0.0),
        )
        for layer in self.sequence_model.layers:
            layer.manual_inference_attention = getattr(
                args, "marie_manual_inference_attention", True
            )
        token_embed_dim = getattr(args, "marie_token_embed_dim", 128)
        tokenizer_hidden = getattr(args, "marie_tokenizer_hidden_dim", 512)
        self.encoder = MARIEVQTokenizer(
            obs_dim, self.n_latents, self.n_categories,
            token_embed_dim, tokenizer_hidden,
            ema_decay=getattr(args, "marie_vq_ema_decay", 0.8),
            commitment_weight=getattr(args, "marie_vq_commitment_weight", 10.0),
        )
        self.token_feature_dim = self.n_latents * token_embed_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.token_feature_dim, tokenizer_hidden), nn.GELU(),
            nn.Linear(tokenizer_hidden, tokenizer_hidden), nn.GELU(),
            nn.Linear(tokenizer_hidden, obs_dim),
        )
        # Dynamics uses a free token embedding, independent of VQ codebook
        # geometry, as in upstream MARIE.
        self.world_token_embedding = nn.Embedding(
            self.n_categories, self.hidden_dim
        )
        # MARIE shares one action embedding table across decentralized agent
        # streams; agent identity is supplied only to the Perceiver context.
        self.action_token_embedding = nn.Embedding(
            n_actions, self.hidden_dim
        )
        self.block_size = self.n_latents + 2  # observation, action, aggregate
        self.position = nn.Parameter(th.zeros(
            1,
            self.max_seq_length * self.block_size,
            self.hidden_dim,
        ))
        nn.init.normal_(self.position, std=0.02)
        self.next_token_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_categories),
        )
        layers = getattr(args, "marie_perceiver_layers", 2)
        heads = getattr(args, "marie_perceiver_heads", 4)
        latent_heads = getattr(args, "marie_perceiver_latent_heads", 8)
        dropout = getattr(args, "marie_perceiver_dropout", 0.0)
        self.aggregator = PerceiverAggregator(
            self.hidden_dim, n_agents, heads, latent_heads, layers, dropout
        )
        position = th.arange(30, dtype=th.float32)[:, None]
        dimension = th.arange(self.hidden_dim, dtype=th.float32)[None]
        angle = position / th.pow(
            10000.0, 2 * th.div(dimension, 2, rounding_mode="floor")
            / self.hidden_dim,
        )
        agent_position = th.empty(30, self.hidden_dim)
        agent_position[:, 0::2] = th.sin(angle[:, 0::2])
        agent_position[:, 1::2] = th.cos(angle[:, 1::2])
        self.register_buffer(
            "perceiver_agent_position", agent_position, persistent=False
        )
        def auxiliary_head(output_dim):
            return nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, output_dim),
            )
        self.reward = auxiliary_head(1)
        self.reward_ensemble = nn.ModuleList()
        self.continuation = auxiliary_head(1)
        # Reference MARIE's ``--ce_for_av`` path predicts a two-class
        # categorical distribution (unavailable/available) for every action.
        # A single Bernoulli logit is not checkpoint- or loss-equivalent and
        # tends to hide errors on the comparatively rare attack actions.
        self.next_availability = auxiliary_head(n_actions * 2)
        self.reward_uses_mlp = False
        self._initialize_reference_world_model()
        self.encoder.eval()

    def _initialize_reference_world_model(self):
        """Apply MARIE's N(0, .02) model initialization, excluding the VQ AE."""
        for name, module in self.named_modules():
            if name == "encoder" or name.startswith("encoder."):
                continue
            if name == "decoder" or name.startswith("decoder."):
                continue
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def teammate_logits(self, latent, detach_encoder=True):
        """Compatibility placeholder; MARIE has no teammate prediction head."""
        del detach_encoder
        return latent.new_zeros(
            *latent.shape[:-2], self.n_agents, self.n_actions
        )

    def token_features(self, latent):
        return self.encoder.embed(latent).flatten(-2)

    def decode(self, latent):
        return self.decoder(self.token_features(latent))

    def embed_world_tokens(self, latent):
        if latent.shape[-1] == self.n_categories:
            return th.matmul(latent, self.world_token_embedding.weight)
        return self.world_token_embedding(latent.long())

    def _dynamics_blocks(self, latent, actions, focal_ids):
        length = latent.shape[1]
        if actions.dim() >= 4 and actions.shape[-2:] == (
            self.n_agents, self.n_actions
        ):
            agent_index = focal_ids[:, None, None, None].expand(
                -1, length, 1, self.n_actions
            )
            action_ids = actions.gather(-2, agent_index).squeeze(-2).argmax(-1)
        elif actions.dim() == 3 and actions.shape[-1] == self.n_agents:
            agent_index = focal_ids[:, None, None].expand(-1, length, 1)
            action_ids = actions.gather(-1, agent_index).squeeze(-1).long()
        else:
            action_ids = actions.long().squeeze(-1)
        observation_tokens = self.embed_world_tokens(latent)
        action_token = self.action_token_embedding(action_ids)

        # Central aggregation is computed before temporal prediction from all
        # agents' local observation/action encodings, then inserted as the last
        # token of every causal block, matching upstream MARIE's layout.
        group_size = getattr(self, "context_n_agents", self.n_agents)
        streams = observation_tokens.shape[0]
        complete_teams = False
        if streams % group_size == 0:
            ids = focal_ids.view(-1, group_size)
            expected = th.arange(group_size, device=ids.device)[None]
            complete_teams = th.equal(ids, expected.expand_as(ids))
        if complete_teams:
            batch = streams // group_size
            # Preserve every local VQ token. Upstream MARIE's Perceiver sees
            # N * (M observation tokens + one action token), not an average.
            team_tokens = th.cat(
                (observation_tokens, action_token.unsqueeze(-2)), dim=-2
            ).view(
                batch, group_size, length, self.n_latents + 1,
                self.hidden_dim,
            ).permute(0, 2, 1, 3, 4).reshape(
                batch * length,
                group_size * (self.n_latents + 1),
                self.hidden_dim,
            )
            agent_positions = self.perceiver_agent_position[
                :group_size
            ].repeat_interleave(self.n_latents + 1, dim=0)
            team_tokens = team_tokens + agent_positions[None]
            aggregate = (self.aggregator(team_tokens) if group_size == self.n_agents
                         else self.aggregator(team_tokens, n_agents=group_size))
            aggregate = aggregate.view(
                batch, length, group_size, self.hidden_dim
            ).permute(0, 2, 1, 3).reshape(
                streams, length, self.hidden_dim
            )
        else:
            aggregate = observation_tokens.mean(-2)

        return th.cat((
            observation_tokens,
            action_token.unsqueeze(-2),
            aggregate.unsqueeze(-2),
        ), dim=-2)

    def teacher_forced_dynamics(
        self, latent, actions, focal_ids, attention_mask=None
    ):
        """Train next codes on the same uninterrupted causal token stream.

        The aggregation token at t predicts the first code of observation
        t+1. Each teacher-forced code then predicts the following code.
        """
        transitions = actions.shape[1]
        blocks = self._dynamics_blocks(
            latent[:, :transitions], actions, focal_ids
        )
        streams = blocks.shape[0]
        sequence_tokens = blocks.reshape(
            streams, transitions * self.block_size, self.hidden_dim
        )
        final_observation = self.embed_world_tokens(
            latent[:, transitions]
        )
        sequence_tokens = th.cat((sequence_tokens, final_observation), dim=1)
        sequence_tokens = sequence_tokens + self.position[
            :, :sequence_tokens.shape[1]
        ]
        sequence = self.sequence_model(sequence_tokens, mask=attention_mask)
        hidden = sequence[:, self.block_size - 1:
                          transitions * self.block_size:self.block_size]
        logits = []
        for step in range(transitions):
            first_source = sequence[:, step * self.block_size + self.block_size - 1]
            if step + 1 < transitions:
                next_block = (step + 1) * self.block_size
            else:
                next_block = transitions * self.block_size
            sources = th.cat((
                first_source[:, None],
                sequence[:, next_block:next_block + self.n_latents - 1],
            ), dim=1)
            logits.append(self.next_token_head(sources))
        return hidden, th.stack(logits, dim=1)

    def reference_teacher_forced_dynamics(
        self, latent, actions, focal_ids, attention_mask=None
    ):
        """Run the exact token shift used by ``MAWorldModel.compute_loss``.

        Each replay record contributes one complete observation/action/team
        block. Observation outputs are the first M-1 local token positions and
        the aggregate position; shifting that stream by one predicts every
        observation token except the first token of the sampled sequence.
        """
        length = actions.shape[1]
        blocks = self._dynamics_blocks(latent[:, :length], actions, focal_ids)
        streams = blocks.shape[0]
        sequence_tokens = blocks.reshape(
            streams, length * self.block_size, self.hidden_dim
        )
        sequence_tokens = sequence_tokens + self.position[
            :, :sequence_tokens.shape[1]
        ]
        sequence = self.sequence_model(
            sequence_tokens, mask=attention_mask
        ).view(streams, length, self.block_size, self.hidden_dim)
        observation_sources = th.cat((
            sequence[:, :, :self.n_latents - 1],
            sequence[:, :, -1:],
        ), dim=2).reshape(
            streams, length * self.n_latents, self.hidden_dim
        )[:, :-1]
        return sequence[:, :, -1], self.next_token_head(observation_sources)

    def dynamics_sequence(self, latent, actions, focal_ids):
        """Autoregress over VQ observation tokens and decentralized actions."""
        length = latent.shape[1]
        if length > self.max_seq_length:
            latent = latent[:, -self.max_seq_length:]
            actions = actions[:, -self.max_seq_length:]
            length = self.max_seq_length
        blocks = self._dynamics_blocks(latent, actions, focal_ids)
        streams = blocks.shape[0]
        token_sequence = blocks.reshape(
            streams, length * self.block_size, self.hidden_dim
        )
        token_sequence = token_sequence + self.position[:, :token_sequence.shape[1]]
        sequence = self.sequence_model(
            token_sequence,
            mask=th.triu(
                th.full(
                    (token_sequence.shape[1], token_sequence.shape[1]),
                    float("-inf"), device=token_sequence.device
                ),
                diagonal=1,
            ),
        )
        # Prediction heads consume the aggregation-token state of each block.
        return sequence[:, self.block_size - 1::self.block_size]

    def init_dynamics_cache(self, latent, actions, focal_ids):
        """Build a KV cache from an initial context and return its hidden states."""
        if latent.shape[1] > self.max_seq_length:
            latent = latent[:, -self.max_seq_length:]
            actions = actions[:, -self.max_seq_length:]
        blocks = self._dynamics_blocks(latent, actions, focal_ids)
        tokens = blocks.flatten(1, 2)
        tokens = tokens + self.position[:, :tokens.shape[1]]
        sequence, cache = self.sequence_model.forward_cached(
            tokens, max_cache_tokens=self.max_seq_length * self.block_size
        )
        return sequence[:, self.block_size - 1::self.block_size], cache

    def append_dynamics_cache(self, latent, actions, focal_ids, cache):
        """Append one or more complete MARIE blocks without replaying history."""
        blocks = self._dynamics_blocks(latent, actions, focal_ids)
        tokens = blocks.flatten(1, 2)
        cached_length = 0 if not cache else cache[0][0].shape[2]
        max_tokens = self.max_seq_length * self.block_size
        retained_length = min(cached_length, max_tokens - tokens.shape[1])
        position_end = retained_length + tokens.shape[1]
        tokens = tokens + self.position[:, retained_length:position_end]
        sequence, cache = self.sequence_model.forward_cached(
            tokens, cache, max_cache_tokens=max_tokens
        )
        return sequence[:, self.block_size - 1::self.block_size], cache

    def append_action_cache(self, latent, actions, focal_ids, cache):
        """Append only action and Perceiver tokens after generated obs codes.

        The observation codes are already present in the shared KV cache. This
        is the exact token flow used by ``MAWorldModelEnv.step_ar``.
        """
        blocks = self._dynamics_blocks(latent, actions, focal_ids)
        tokens = blocks[:, :, -2:].flatten(1, 2)
        cached_length = cache[0][0].shape[2]
        max_tokens = self.max_seq_length * self.block_size
        retained_length = min(cached_length, max_tokens - tokens.shape[1])
        tokens = tokens + self.position[
            :, retained_length:retained_length + tokens.shape[1]
        ]
        sequence, cache = self.sequence_model.forward_cached(
            tokens, cache, max_cache_tokens=max_tokens
        )
        return sequence[:, -1], cache

    def autoregressive_dynamics_logits(self, hidden, target=None):
        """Predict observation codes with the shared causal Transformer.

        The aggregation state predicts token zero. Each following position is
        conditioned on teacher-forced or generated earlier observation tokens,
        matching MAWorldModelEnv's token-by-token autoregressive procedure.
        """
        original_shape = hidden.shape[:-1]
        prefix = hidden.reshape(-1, self.hidden_dim)
        flat_target = None if target is None else target.reshape(
            -1, self.n_latents, self.n_categories
        )
        logits = []
        output, cache = self.sequence_model.forward_cached(
            prefix.unsqueeze(1) + self.position[:, :1]
        )
        for token_index in range(self.n_latents):
            token_logits = self.next_token_head(output[:, -1])
            logits.append(token_logits)
            if flat_target is None:
                token = F.one_hot(
                    token_logits.argmax(-1), self.n_categories
                ).to(token_logits.dtype)
            else:
                token = flat_target[:, token_index]
            if token_index + 1 < self.n_latents:
                embedded = self.embed_world_tokens(token).unsqueeze(1)
                embedded = embedded + self.position[
                    :, token_index + 1:token_index + 2
                ]
                output, cache = self.sequence_model.forward_cached(
                    embedded, cache
                )
        return th.stack(logits, 1).view(
            *original_shape, self.n_latents, self.n_categories
        )

    def prediction_auxiliary_heads(self, hidden, latent):
        del latent
        reward_logits = self.reward(hidden)
        heads = {
            "reward_logits": reward_logits,
            "reward_ensemble_logits": [reward_logits],
            "continuation_logits": self.continuation(hidden),
        }
        heads["availability_logits"] = self.next_availability(hidden).view(
            *hidden.shape[:-1], self.n_actions, 2
        )
        return heads

    def availability_from_logits(self, logits):
        """Decode reference-style per-action categorical availability."""
        available = logits.argmax(-1).bool()
        # SMAC always exposes at least one action. Keep imagined rollouts
        # valid if an uncertain model predicts an empty set.
        confidence = logits[..., 1] - logits[..., 0]
        fallback = F.one_hot(
            confidence.argmax(-1), self.n_actions
        ).bool()
        return th.where(~available.any(-1, keepdim=True), fallback, available)

    def prediction_heads(self, hidden, latent, next_latent=None):
        heads = self.prediction_auxiliary_heads(hidden, latent)
        heads["dynamics_logits"] = self.autoregressive_dynamics_logits(
            hidden, next_latent
        )
        return heads

    def predicted_availability_from_hidden(self, hidden):
        logits = self.next_availability(hidden).view(
            *hidden.shape[:-1], self.n_actions, 2
        )
        return self.availability_from_logits(logits)

    def sample_next_latent(self, hidden):
        original_shape = hidden.shape[:-1]
        prefix = hidden.reshape(-1, self.hidden_dim)
        tokens = []
        output, cache = self.sequence_model.forward_cached(
            prefix.unsqueeze(1) + self.position[:, :1]
        )
        for token_index in range(self.n_latents):
            index = th.distributions.Categorical(
                logits=self.next_token_head(output[:, -1])
            ).sample()
            token = F.one_hot(
                index, self.n_categories
            ).to(hidden.dtype)
            tokens.append(token)
            if token_index + 1 < self.n_latents:
                embedded = self.embed_world_tokens(token).unsqueeze(1)
                embedded = embedded + self.position[
                    :, token_index + 1:token_index + 2
                ]
                output, cache = self.sequence_model.forward_cached(
                    embedded, cache
                )
        return th.stack(tokens, 1).view(
            *original_shape, self.n_latents, self.n_categories
        )

    def sample_next_latent_cached(self, hidden, cache):
        """Generate next observation codes in the trajectory's existing KV cache."""
        original_shape = hidden.shape[:-1]
        output = hidden.reshape(-1, self.hidden_dim)
        tokens = []
        max_tokens = self.max_seq_length * self.block_size
        for token_index in range(self.n_latents):
            index = th.distributions.Categorical(
                logits=self.next_token_head(output)
            ).sample()
            token = F.one_hot(index, self.n_categories).to(hidden.dtype)
            tokens.append(token)
            embedded = self.world_token_embedding(index).unsqueeze(1)
            cached_length = cache[0][0].shape[2]
            retained_length = min(cached_length, max_tokens - 1)
            embedded = embedded + self.position[
                :, retained_length:retained_length + 1
            ]
            sequence, cache = self.sequence_model.forward_cached(
                embedded, cache, max_cache_tokens=max_tokens
            )
            output = sequence[:, -1]
        latent = th.stack(tokens, 1).view(
            *original_shape, self.n_latents, self.n_categories
        )
        return latent, output.view(*original_shape, self.hidden_dim), cache
