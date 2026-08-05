"""Masked-input corruption -- step 1 of the Graph AutoEncoder Classifier spec
(Notion page "Graph AutoEncoder Classifier", 3b2d7b0592128092b307fef34d65d0b4).

For a skeleton's node embeddings X in R^(N,D), select 30% of nodes as M. Each
selected node i is replaced with:
  - a learned mask embedding m,               w.p. 0.8
  - a replacement embedding x_r ~ D_train,     w.p. 0.1
  - left unchanged (x_i),                      w.p. 0.1

The reconstruction loss (see losses.py) is computed over ALL of M, including
the unchanged 10% -- the model still has to reconstruct those, it just isn't
told they're masked. Never touches a CLS query: there is no CLS node in the
graph here, q_CLS lives in readout.py and only reads finished node embeddings,
so nothing in this module can accidentally corrupt it.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class NodeMasker(nn.Module):
    """Samples M and builds X-tilde for one forward pass.

    Resampled on every call, which is at least as fresh as the spec's "resample
    the corruption pattern every epoch" -- call once per batch (the normal
    training-loop shape) or once per epoch if batches are graphs one at a time;
    either satisfies "not fixed for the whole run".
    """

    def __init__(
        self,
        dim: int,
        mask_prob: float = 0.3,
        mask_token_prob: float = 0.8,
        replace_prob: float = 0.1,
        keep_prob: float = 0.1,
    ):
        super().__init__()
        total = mask_token_prob + replace_prob + keep_prob
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"mask_token_prob + replace_prob + keep_prob must sum to 1, got {total}")
        self.mask_prob = mask_prob
        self.mask_token_prob = mask_token_prob
        self.replace_prob = replace_prob
        self.keep_prob = keep_prob
        self.mask_embedding = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.mask_embedding, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        replacement_source: Callable[[int], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (N, D) raw embeddings for a batch of one or more graphs (PyG batches
            concatenate nodes across graphs along dim 0, which this treats
            uniformly -- masking is per-node, not per-graph).
        replacement_source: callable(n) -> (n, D) tensor of x_r replacement
            embeddings, already sampled from the training split by the caller
            (see gnn/dataset.py::ReplacementPool) per whichever of the three
            strategies the spec lists (random other neuron / different class /
            same class). Required whenever replace_prob > 0.

        Returns (x_tilde, mask) where mask (N,) bool is the full selected set
        M -- callers compute the reconstruction loss only where mask is True.
        """
        n = x.shape[0]
        device = x.device
        mask = torch.rand(n, device=device) < self.mask_prob
        choice = torch.rand(n, device=device)
        use_token = mask & (choice < self.mask_token_prob)
        use_replace = mask & (choice >= self.mask_token_prob) & (
            choice < self.mask_token_prob + self.replace_prob
        )
        # The remaining mask_prob * keep_prob fraction is selected but left as
        # x_i -- no action needed, x_tilde already equals x there.

        x_tilde = x.clone()
        x_tilde[use_token] = self.mask_embedding.to(x.dtype)

        n_replace = int(use_replace.sum().item())
        if n_replace:
            if replacement_source is None:
                raise ValueError("replace_prob > 0 but no replacement_source was given")
            x_tilde[use_replace] = replacement_source(n_replace).to(x.dtype)

        return x_tilde, mask
