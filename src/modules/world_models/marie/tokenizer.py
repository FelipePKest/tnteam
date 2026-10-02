"""Vector-quantized local observation tokenizer."""

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.vector_quantizer import EMAVectorQuantizer

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
