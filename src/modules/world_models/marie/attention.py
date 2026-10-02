"""Cached causal attention and inference buffers."""

import torch as th
import torch.nn as nn
import torch.nn.functional as F


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
        self.manual_inference_attention = True
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
        # A single cached query can attend to every retained key. SDPA's
        # is_causal aligns to the upper left, so cached chunks need an offset.
        allowed = None
        causal = past_length == 0
        if length > 1 and past_length > 0:
            query_index = th.arange(length, device=inputs.device)[:, None]
            key_index = th.arange(key.shape[2], device=inputs.device)[None, :]
            allowed = key_index <= past_length + query_index
        if attention_mask is not None and cache is None:
            if attention_mask.dim() == 2:
                allowed = th.isfinite(attention_mask)[
                    None, None, :length, :key.shape[2]
                ]
            else:
                allowed = attention_mask[
                    :, None, :length, :key.shape[2]
                ].bool()
            allowed = allowed & th.ones(
                length, key.shape[2], dtype=th.bool, device=inputs.device
            ).tril(diagonal=past_length)
            causal = False
        # PyTorch 2.5's FP32 efficient-SDPA backend is substantially slower
        # for MARIE's incremental inference shapes. Keep SDPA for gradients
        # and mixed precision, where the fused kernels have different costs.
        manual_inference = (
            self.manual_inference_attention and not self.training
            and not th.is_grad_enabled() and query.dtype == th.float32
        )
        if hasattr(F, "scaled_dot_product_attention") and not manual_inference:
            update = F.scaled_dot_product_attention(
                query, key, value, attn_mask=allowed,
                dropout_p=self.attention_dropout.p if self.training else 0.0,
                is_causal=causal,
            )
        else:
            scores = th.matmul(query, key.transpose(-2, -1)) * self.scale
            if causal:
                allowed = th.ones(
                    length, key.shape[2], dtype=th.bool, device=inputs.device
                ).tril()
            if allowed is not None:
                scores = scores.masked_fill(~allowed, float("-inf"))
            update = th.matmul(self.attention_dropout(scores.softmax(-1)), value)
        update = update.transpose(1, 2).reshape(
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
