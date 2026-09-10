"""Small, dependency-free EMA vector quantizer used by tnteam's MARIE port.

Only the single-codebook, channel-last mode required by MARIE is implemented.
Keeping it here makes the model portable without adding another repository to
``sys.path`` or relying on a separately installed VQ package.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class _EMACodebook(nn.Module):
    """Buffer layout compatible with checkpoints made by the earlier backend."""

    def __init__(self, dim, codebook_size):
        super().__init__()
        codebook = th.empty(1, codebook_size, dim)
        nn.init.kaiming_uniform_(codebook)
        self.register_buffer("initted", th.Tensor([True]))
        self.register_buffer("cluster_size", th.zeros(1, codebook_size))
        self.register_buffer("embed_avg", codebook.clone())
        self.register_buffer("embed", codebook)


class EMAVectorQuantizer(nn.Module):
    """Euclidean nearest-neighbour VQ with exponential-moving-average codes."""

    def __init__(self, dim, codebook_size, decay=0.8, eps=1e-5):
        super().__init__()
        if dim <= 0 or codebook_size <= 0:
            raise ValueError("dim and codebook_size must be positive")
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.dim = dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.eps = eps

        self._codebook = _EMACodebook(dim, codebook_size)

    @property
    def codebook(self):
        return self._codebook.embed[0]

    @th.no_grad()
    def _ema_update(self, vectors, indices):
        counts = th.bincount(indices, minlength=self.codebook_size).to(
            vectors.dtype
        )
        sums = vectors.new_zeros(self.codebook_size, self.dim)
        sums.index_add_(0, indices, vectors)

        cluster_size = self._codebook.cluster_size[0]
        embed_avg = self._codebook.embed_avg[0]
        cluster_size.lerp_(counts, 1.0 - self.decay)
        embed_avg.lerp_(sums, 1.0 - self.decay)

        total = cluster_size.sum()
        smoothed = (
            (cluster_size + self.eps)
            / (total + self.codebook_size * self.eps)
            * total
        )
        self._codebook.embed[0].copy_(embed_avg / smoothed.unsqueeze(-1))

    def forward(self, inputs):
        if inputs.shape[-1] != self.dim:
            raise ValueError(
                f"Expected vectors of dimension {self.dim}, got "
                f"{inputs.shape[-1]}"
            )
        shape = inputs.shape
        vectors = inputs.reshape(-1, self.dim).float()
        codebook = self.codebook.detach()
        distances = (
            vectors.pow(2).sum(-1, keepdim=True)
            + codebook.pow(2).sum(-1).unsqueeze(0)
            - 2.0 * vectors @ codebook.t()
        )
        indices = distances.argmin(-1)
        quantized = F.embedding(indices, codebook)

        # Match the reference quantizer: assignments use the pre-update
        # codebook, and the EMA mutation happens during a training forward.
        if self.training:
            self._ema_update(vectors.detach(), indices)
            commitment = F.mse_loss(quantized.detach(), vectors)
            quantized = vectors + (quantized - vectors).detach()
        else:
            commitment = vectors.new_zeros(1)

        return (
            quantized.reshape(shape),
            indices.reshape(shape[:-1]),
            commitment.reshape(1),
        )
