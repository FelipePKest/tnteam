"""MARIE world model adapted to EPyMARL's episode-major replay.

The implementation follows Zhang et al. (2024): one shared autoregressive
Transformer models every agent's local token stream and a Perceiver-style
cross-attention block injects a centralized team representation.  The actor
remains decentralized and consumes only the focal agent's model features.
"""

import copy

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.matwm import MATWMPolicy, MATWMWorldModel
from modules.vector_quantizer import EMAVectorQuantizer


class _FixedKVCache:
    """Reference-style fixed-capacity inference cache without cat peaks."""

    def __init__(self, key, value, capacity):
        shape = (*key.shape[:2], capacity, key.shape[-1])
        self.key = key.new_empty(shape)
        self.value = value.new_empty(shape)
        self.size = 0
        self.append(key, value)

    def append(self, key, value):
        length = key.shape[2]
        overflow = max(0, self.size + length - self.key.shape[2])
        if overflow:
            retained = self.size - overflow
            self.key[:, :, :retained].copy_(
                self.key[:, :, overflow:self.size].clone()
            )
            self.value[:, :, :retained].copy_(
                self.value[:, :, overflow:self.size].clone()
            )
            self.size = retained
        self.key[:, :, self.size:self.size + length].copy_(key)
        self.value[:, :, self.size:self.size + length].copy_(value)
        self.size += length

    def get(self):
        return self.key[:, :, :self.size], self.value[:, :, :self.size]

    def __getitem__(self, index):
        return self.get()[index]


class CachedCausalTransformerLayer(nn.Module):
    """Pre-norm causal self-attention with an append-only KV cache."""

    def __init__(self, hidden_dim, heads, ff_dim, dropout):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("Transformer hidden dimension must divide attention heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.scale = self.head_dim ** -0.5
        self.attention_norm = nn.LayerNorm(hidden_dim)
        # Keep separate projections to match the reference parameterization
        # and initialization exactly.
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.attention_output = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim), nn.Dropout(dropout),
        )

    def forward_cached(
        self, inputs, cache=None, max_cache_tokens=None, attention_mask=None
    ):
        batch, length, hidden_dim = inputs.shape
        normalized = self.attention_norm(inputs)
        query = self.query(normalized).view(
            batch, length, self.heads, self.head_dim
        ).transpose(1, 2)
        key = self.key(normalized).view(
            batch, length, self.heads, self.head_dim
        ).transpose(1, 2)
        value = self.value(normalized).view(
            batch, length, self.heads, self.head_dim
        ).transpose(1, 2)
        if isinstance(cache, _FixedKVCache):
            cache.append(key, value)
            key, value = cache.get()
        elif cache is not None:
            past_key, past_value = cache
            if max_cache_tokens is not None:
                keep = max(0, max_cache_tokens - length)
                past_key = past_key[:, :, -keep:] if keep else past_key[:, :, :0]
                past_value = past_value[:, :, -keep:] if keep else past_value[:, :, :0]
            key = th.cat((past_key, key), dim=2)
            value = th.cat((past_value, value), dim=2)
        elif max_cache_tokens is not None:
            cache = _FixedKVCache(key, value, max_cache_tokens)
            key, value = cache.get()
        past_length = key.shape[2] - length
        scores = th.matmul(query, key.transpose(-2, -1)) * self.scale
        query_index = th.arange(length, device=inputs.device)[:, None]
        key_index = th.arange(key.shape[2], device=inputs.device)[None, :]
        causal_mask = key_index > past_length + query_index
        scores = scores.masked_fill(causal_mask, float("-inf"))
        if attention_mask is not None and cache is None:
            if attention_mask.dim() == 2:
                allowed = th.isfinite(attention_mask)[
                    None, None, :length, :key.shape[2]
                ]
            else:
                allowed = attention_mask[
                    :, None, :length, :key.shape[2]
                ].bool()
            scores = scores.masked_fill(~allowed, float("-inf"))
        attention = self.attention_dropout(scores.softmax(-1))
        update = th.matmul(attention, value).transpose(1, 2).reshape(
            batch, length, hidden_dim
        )
        output = inputs + self.residual_dropout(self.attention_output(update))
        output = output + self.feed_forward(output)
        return output, cache if isinstance(cache, _FixedKVCache) else (key, value)


class CachedCausalTransformer(nn.Module):
    """Causal Transformer supporting both training and incremental inference."""

    def __init__(self, hidden_dim, heads, ff_dim, layers, dropout):
        super().__init__()
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            CachedCausalTransformerLayer(hidden_dim, heads, ff_dim, dropout)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs, mask=None):
        output = self.input_dropout(inputs)
        for layer in self.layers:
            output, _ = layer.forward_cached(
                output, attention_mask=mask
            )
        return self.norm(output)

    def forward_cached(
        self, inputs, cache=None, max_cache_tokens=None, attention_mask=None
    ):
        if cache is None:
            cache = [None] * len(self.layers)
        output = self.input_dropout(inputs)
        new_cache = []
        for layer, layer_cache in zip(self.layers, cache):
            if attention_mask is None:
                output, layer_cache = layer.forward_cached(
                    output, layer_cache, max_cache_tokens
                )
            else:
                output, layer_cache = layer.forward_cached(
                    output, layer_cache, max_cache_tokens,
                    attention_mask=attention_mask,
                )
            new_cache.append(layer_cache)
        return self.norm(output), new_cache


class MARIEVQTokenizer(nn.Module):
    """Vector-quantize each observation into MARIE's discrete token stream."""

    def __init__(
        self, obs_dim, n_tokens, vocab_size, embed_dim, hidden_dim,
        ema_decay=0.8, commitment_weight=10.0,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_tokens * embed_dim),
        )
        self.quantizer = EMAVectorQuantizer(
            dim=embed_dim, codebook_size=vocab_size, decay=ema_decay
        )
        self.commitment_weight = commitment_weight
        self._last_commitment_loss = None
        self._last_tokens = None
        self._last_quantized = None

    @property
    def codebook(self):
        return self.quantizer.codebook

    def encode_vectors(self, observation):
        return self.encoder(observation).view(
            *observation.shape[:-1], self.n_tokens, self.embed_dim
        )

    def logits(self, observation):
        encoded = self.encode_vectors(observation)
        codebook = self.quantizer.codebook.detach()
        encoded_squared = encoded.pow(2).sum(-1, keepdim=True)
        codebook_squared = codebook.pow(2).sum(-1)
        cross = th.matmul(encoded, codebook.t())
        return -(encoded_squared + codebook_squared - 2.0 * cross)

    def forward(self, observation, sample=True):
        del sample
        encoded = self.encode_vectors(observation)
        shape = encoded.shape
        quantized, indices, commitment = self.quantizer(
            encoded.reshape(-1, self.n_tokens, self.embed_dim)
        )
        indices = indices.reshape(*shape[:-1])
        self._last_commitment_loss = commitment.mean()
        logits = self.logits(observation)
        hard = F.one_hot(indices, self.vocab_size).to(logits.dtype)
        self._last_tokens = hard
        self._last_quantized = quantized.reshape(*shape)
        return hard, logits

    def embed(self, tokens):
        if tokens is self._last_tokens and self._last_quantized is not None:
            return self._last_quantized
        indices = tokens.argmax(-1)
        return self.codebook[indices]

    def commitment_loss(self, observation):
        if self._last_commitment_loss is None:
            self.forward(observation)
        return self.commitment_weight * self._last_commitment_loss

    @th.no_grad()
    def update_codebook(self, observation):
        del observation  # VectorQuantize performs its EMA update in forward.


class _GEGLU(nn.Module):
    def forward(self, value):
        value, gate = value.chunk(2, dim=-1)
        return value * F.gelu(gate)


class _PerceiverAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads, dim_head, dropout):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, inner, bias=True)
        self.to_kv = nn.Linear(context_dim, inner * 2, bias=True)
        self.to_out = nn.Linear(inner, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context):
        batch, query_length = query.shape[:2]
        context_length = context.shape[1]
        q = self.to_q(query).view(
            batch, query_length, self.heads, -1
        ).transpose(1, 2)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        k = k.view(batch, context_length, self.heads, -1).transpose(1, 2)
        v = v.view(batch, context_length, self.heads, -1).transpose(1, 2)
        attention = (q @ k.transpose(-2, -1) * self.scale).softmax(-1)
        output = self.dropout(attention) @ v
        output = output.transpose(1, 2).reshape(batch, query_length, -1)
        return self.to_out(output)


class _PerceiverFeedForward(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 8), _GEGLU(),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def forward(self, value):
        return self.net(value)


class PerceiverAggregator(nn.Module):
    """Produce one centralized latent for every decentralized agent stream.

    Official MARIE uses ``NUM_AGENTS`` Perceiver latents.  Keeping one query
    per agent is important: a single pooled token makes every agent receive
    exactly the same centralized feature and is not the architecture described
    in the paper or implemented upstream.
    """

    def __init__(
        self, hidden_dim, n_agents, cross_heads, latent_heads, layers, dropout
    ):
        super().__init__()
        self.query = nn.Parameter(th.randn(n_agents, hidden_dim))
        self.cross_query_norm = nn.LayerNorm(hidden_dim)
        self.cross_context_norm = nn.LayerNorm(hidden_dim)
        # Reference configuration uses dim_head=64 independently of model dim.
        self.cross_attention = _PerceiverAttention(
            hidden_dim, hidden_dim, cross_heads, 64, dropout
        )
        self.cross_ff_norm = nn.LayerNorm(hidden_dim)
        self.cross_ff = _PerceiverFeedForward(hidden_dim, dropout)
        self.latent_attention = nn.ModuleList([
            _PerceiverAttention(
                hidden_dim, hidden_dim, latent_heads, 64, dropout
            ) for _ in range(layers)
        ])
        self.latent_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(layers)
        ])
        self.latent_ff = nn.ModuleList([
            _PerceiverFeedForward(hidden_dim, dropout) for _ in range(layers)
        ])
        self.latent_ff_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(layers)
        ])

    def forward(self, agent_features):
        # agent_features: [batch*time, agents, hidden]
        token = self.query.unsqueeze(0).expand(agent_features.shape[0], -1, -1)
        update = self.cross_attention(
            self.cross_query_norm(token),
            self.cross_context_norm(agent_features),
        )
        token = token + update
        token = token + self.cross_ff(self.cross_ff_norm(token))
        for attention, norm, feed_forward, ff_norm in zip(
            self.latent_attention, self.latent_norms, self.latent_ff,
            self.latent_ff_norms,
        ):
            normalized = norm(token)
            update = attention(normalized, normalized)
            token = token + update
            token = token + feed_forward(ff_norm(token))
        return token


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
        streams = observation_tokens.shape[0]
        complete_teams = False
        if streams % self.n_agents == 0:
            ids = focal_ids.view(-1, self.n_agents)
            expected = th.arange(self.n_agents, device=ids.device)[None]
            complete_teams = th.equal(ids, expected.expand_as(ids))
        if complete_teams:
            batch = streams // self.n_agents
            # Preserve every local VQ token. Upstream MARIE's Perceiver sees
            # N * (M observation tokens + one action token), not an average.
            team_tokens = th.cat(
                (observation_tokens, action_token.unsqueeze(-2)), dim=-2
            ).view(
                batch, self.n_agents, length, self.n_latents + 1,
                self.hidden_dim,
            ).permute(0, 2, 1, 3, 4).reshape(
                batch * length,
                self.n_agents * (self.n_latents + 1),
                self.hidden_dim,
            )
            agent_positions = self.perceiver_agent_position[
                :self.n_agents
            ].repeat_interleave(self.n_latents + 1, dim=0)
            team_tokens = team_tokens + agent_positions[None]
            aggregate = self.aggregator(team_tokens).view(
                batch, length, self.n_agents, self.hidden_dim
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
            embedded = self.embed_world_tokens(token).unsqueeze(1)
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


class OriginalMARIEActor(nn.Module):
    """Two-hidden-layer ReLU actor used by upstream discrete MARIE."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features):
        return self.net(features)


class OriginalMARIECritic(nn.Module):
    """Team-attention value model equivalent to upstream AugmentedCritic."""

    def __init__(self, input_dim, hidden_dim, heads):
        super().__init__()
        del heads  # Kept in the constructor for checkpoint/config compatibility.
        self.embed = nn.Linear(input_dim, hidden_dim)
        position = th.arange(30, dtype=th.float32)[:, None]
        dimension = th.arange(hidden_dim, dtype=th.float32)[None]
        angle = position / th.pow(
            10000.0, 2 * th.div(dimension, 2, rounding_mode="floor")
            / hidden_dim,
        )
        positional = th.empty(30, hidden_dim)
        positional[:, 0::2] = th.sin(angle[:, 0::2])
        positional[:, 1::2] = th.cos(angle[:, 1::2])
        self.register_buffer("positional", positional[None])
        # This is the exact AttentionEncoder configuration used by upstream
        # AugmentedCritic: one post-norm layer, eight heads, FF width=hidden.
        self.attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=0.0,
                batch_first=True,
                norm_first=False,
            ),
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, team_features):
        embedded = F.relu(self.embed(team_features))
        embedded = embedded + self.positional[:, :embedded.shape[1]]
        attended = self.attention(embedded)
        return self.value(F.relu(attended))

    def independent(self, features):
        return self.value(F.relu(self.embed(features)))


class MARIEPolicy(MATWMPolicy):
    """MARIE world model with the established per-agent actor/critic API."""

    def __init__(self, obs_dim, n_agents, n_actions, args):
        nn.Module.__init__(self)
        self.args = args
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.use_raw_obs_skip = False
        self.policy_obs_norm = None
        self.world_model = MARIEWorldModel(obs_dim, n_agents, n_actions, args)
        self.stack_obs = getattr(args, "marie_stack_obs", 5)
        self.actor_feature_dim = obs_dim * self.stack_obs
        self.critic_feature_dim = self.actor_feature_dim
        agent_hidden = getattr(args, "matwm_agent_hidden_dim", 256)
        # Upstream MARIE feeds the same reconstructed local observation stack
        # to actor and critic. The Perceiver state belongs to the environment.
        self.actors = nn.ModuleList([
            OriginalMARIEActor(self.actor_feature_dim, agent_hidden, n_actions)
        ])
        self.critics = nn.ModuleList([
            OriginalMARIECritic(
                self.critic_feature_dim, agent_hidden,
                getattr(args, "marie_critic_heads", 4),
            )
        ])
        self.ema_critics = copy.deepcopy(self.critics)
        for parameter in self.ema_critics.parameters():
            parameter.requires_grad_(False)
        # Match DreamerLearner's explicit StarCraft initialization: actor
        # orthogonal, critic Xavier (biases remain at their zero defaults).
        for parameter in self.actors.parameters():
            if parameter.ndim >= 2:
                nn.init.orthogonal_(parameter)
        for parameter in self.critics.parameters():
            if parameter.ndim >= 2:
                nn.init.xavier_uniform_(parameter)
        self.ema_critics.load_state_dict(self.critics.state_dict())

    def agent_forward(self, modules, state, focal_ids):
        return modules[0](state)

    def actor_logits(self, state, focal_ids):
        return self.actors[0](state[..., :self.actor_feature_dim])

    def values(self, state, focal_ids, ema=False):
        modules = self.ema_critics if ema else self.critics
        features = state[..., :self.actor_feature_dim]
        if features.dim() == 2 and features.shape[0] % self.n_agents == 0:
            ids = focal_ids.reshape(-1, self.n_agents)
            expected = th.arange(self.n_agents, device=ids.device)[None]
            if th.equal(ids, expected.expand_as(ids)):
                values = modules[0](
                    features.view(-1, self.n_agents, features.shape[-1])
                )
                return values.reshape(features.shape[0], 1)
        return modules[0].independent(features)

    def build_state(
        self, latent, hidden, teammate_logits, focal_ids, observation=None
    ):
        if observation is None:
            observation = self.world_model.decode(latent)
        else:
            # Execution and imagined learning must consume the same tokenizer
            # reconstruction distribution used by upstream MARIE.
            with th.no_grad():
                obs_latent, _ = self.world_model.encode(
                    observation, sample=False
                )
                observation = self.world_model.decode(obs_latent)
        # MAWorldModelEnv clamps decoded SMAC observations before both acting
        # and critic evaluation.
        observation = observation.clamp(-1.0, 1.0)
        if observation.dim() == 2:
            observation = observation[:, None]
        observation = observation[:, -self.stack_obs:]
        if observation.shape[1] < self.stack_obs:
            padding = observation.new_zeros(
                observation.shape[0],
                self.stack_obs - observation.shape[1],
                observation.shape[-1],
            )
            observation = th.cat((padding, observation), dim=1)
        actor_features = observation.reshape(observation.shape[0], -1)
        return actor_features
