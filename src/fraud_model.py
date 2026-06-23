"""PyTorch model definition for credit-card fraud detection.

This is a **fraud-dedicated clone** of ``src/model.py`` (the Titanic model).
It is duplicated on purpose: the fraud pipeline must never break if the Titanic
architecture is tweaked, and vice-versa. The two files may diverge over time as
each task is tuned independently — do not refactor them back into one shared
module without a deliberate decision.

Architecture rationale: under a 578:1 class imbalance with 28 PCA features plus
engineered amount/time signals, a plain MLP under-fits the sparse fraud
structure. We use a small **ResNet-style tabular network** — pre-activation
residual blocks with skip connections, BatchNorm and dropout — fronted by a
TabNet-style **feature-attention gate** that learns to scale raw features
(automated feature crossing) and exposes its mask for explainability. Outputs
are **raw logits** (no sigmoid) to pair with ``BCEWithLogitsLoss`` / focal loss
for numerical stability.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Pre-activation residual block for tabular features.

    Computes ``x + f(x)`` where ``f`` is a two-layer BN -> ReLU -> Linear ->
    dropout stack. Width is preserved so the identity skip needs no projection.

    Args:
        dim: Feature width flowing through the block.
        dropout: Dropout probability applied inside ``f``.
    """

    def __init__(self, dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block.

        Args:
            x: Input tensor of shape ``(batch, dim)``.

        Returns:
            Tensor of shape ``(batch, dim)`` (input + transformed).
        """
        return x + self.block(x)


class FeatureAttentionGate(nn.Module):
    """Dynamic feature selection gate inspired by TabNet.

    Learns to output a mask [0, 1] that dynamically scales the raw input features
    based on their interactions, acting as automated Feature Crossing.
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        # Small bottleneck to learn cross-column relationships.
        self.gate = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, in_features),
            nn.Sigmoid(),  # squeeze the attention weights into [0, 1]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Gated features, and the attention mask itself (for explainability).
        """
        mask = self.gate(x)
        return x * mask, mask


class TabularResNet(nn.Module):
    """Small ResNet-style classifier for the fraud feature matrix.

    Projects the input features to a hidden width, passes them through a stack
    of residual blocks, then maps to a single logit. Outputs **raw logits**
    (no sigmoid) so it pairs with ``BCEWithLogitsLoss`` / focal loss for
    numerical stability; callers apply ``torch.sigmoid`` for probabilities.

    Args:
        in_features: Number of input features (32 from the fraud preprocessor).
        hidden_dim: Width of the hidden / residual representation.
        n_blocks: Number of residual blocks.
        dropout: Dropout probability used in the stem and residual blocks.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 32,
        n_blocks: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_features = in_features

        self.feature_attention = FeatureAttentionGate(in_features)

        self.stem = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass; stores the attention mask for explainability."""
        x, self.attention_weights = self.feature_attention(x)

        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return fraud probabilities for input features.

        Sets eval mode so BatchNorm/Dropout behave deterministically.

        Args:
            x: Feature tensor of shape ``(batch, in_features)``.

        Returns:
            Probability tensor of shape ``(batch,)`` in ``[0, 1]``.
        """
        self.eval()
        return torch.sigmoid(self.forward(x))


if __name__ == "__main__":
    # Forward-pass smoke test on a dummy batch (32 fraud features).
    torch.manual_seed(0)
    model = TabularResNet(in_features=32, hidden_dim=64, n_blocks=2)
    dummy = torch.randn(8, 32)
    logits = model(dummy)
    proba = model.predict_proba(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ok] logits shape: {tuple(logits.shape)}")
    print(f"[ok] proba shape:  {tuple(proba.shape)}, range [{proba.min():.3f}, {proba.max():.3f}]")
    print(f"[ok] trainable params: {n_params:,}")
