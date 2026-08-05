"""Loss functions from the Graph AutoEncoder Classifier spec.

Reconstruction (pretraining): computed ONLY over the selected set M (all three
corruption groups from masking.py, including the 10% left unchanged) -- never
over unmasked nodes. Cosine loss is the default; combined cosine+SmoothL1 is
gated behind the embedding-norm diagnostic (scripts/norm_diagnostic.py) per
the spec -- don't enable use_smooth_l1 without having run it and seen norms
carry information that ablating them would lose.

Classification: plain or class-weighted cross-entropy, with weights computed
from the training split only (see compute_class_weights, and dataset.py which
is responsible for only ever passing it training-split labels).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_reconstruction_loss(x_hat: torch.Tensor, x_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L_cos = mean_i (1 - <x_hat_i, x_i> / (||x_hat_i|| ||x_i|| + eps))."""
    cos_sim = F.cosine_similarity(x_hat, x_true, dim=-1, eps=eps)
    return (1.0 - cos_sim).mean()


def masked_reconstruction_loss(
    x_hat: torch.Tensor,
    x_true: torch.Tensor,
    use_smooth_l1: bool = False,
    lambda_mag: float = 1.0,
    smooth_l1_beta: float = 1.0,
) -> torch.Tensor:
    """L_mask, optionally L_cos + lambda_mag * SmoothL1 if the norm diagnostic
    says magnitude carries information cosine similarity alone would discard.

    x_hat/x_true must already be restricted to the masked set M -- pass
    x_hat[mask] / x[mask] in, not the full (N, D) tensors.
    """
    loss = cosine_reconstruction_loss(x_hat, x_true)
    if use_smooth_l1:
        loss = loss + lambda_mag * F.smooth_l1_loss(x_hat, x_true, beta=smooth_l1_beta)
    return loss


def classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """L_cls, class-weighted iff class_weights is given."""
    return F.cross_entropy(logits, targets, weight=class_weights)


def joint_loss(
    cls_loss: torch.Tensor,
    rec_loss: torch.Tensor,
    lambda_rec: float = 1.0,
) -> torch.Tensor:
    """L_joint = L_cls + lambda_rec * L_mask."""
    return cls_loss + lambda_rec * rec_loss


def compute_class_weights(
    train_labels: torch.Tensor, num_classes: int, eps: float = 1.0
) -> torch.Tensor:
    """Inverse-frequency class weights from TRAINING labels only.

    Caller is responsible for only passing labels from the training split --
    this function has no way to enforce that itself. weight_c = N / (C * n_c),
    with counts floored at `eps` so an unseen class doesn't divide by zero.
    """
    counts = torch.bincount(train_labels, minlength=num_classes).float().clamp(min=eps)
    return counts.sum() / (num_classes * counts)
