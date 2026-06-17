"""PyTorch model definition for Titanic survival prediction.

Kept deliberately separate from the training script (``train.py``) so the
architecture can be imported, unit-tested and reloaded for inference without
pulling in any training machinery.

Architecture rationale (from ``notebooks/eda.ipynb``): the dominant signal is
the **Sex x Pclass interaction** plus a **non-linear child effect** (the
``Master`` title / young-age survival bump). A plain logistic baseline or flat
MLP under-fits these interactions, so we use a small **ResNet-style tabular
network**: pre-activation residual blocks with skip connections, BatchNorm and
dropout. The skip connections let the network learn interaction terms on top of
an identity path, keeping gradients healthy at depth while staying tiny enough
for the ~900-row dataset (regularised hard to avoid overfitting).
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


class TabularResNet(nn.Module):
    """Small ResNet-style classifier for the Titanic feature matrix.

    Projects the input features to a hidden width, passes them through a stack
    of residual blocks, then maps to a single logit. Outputs **raw logits**
    (no sigmoid) so it pairs with ``BCEWithLogitsLoss`` for numerical stability;
    callers apply ``torch.sigmoid`` for probabilities.

    Args:
        in_features: Number of input features (e.g. 18 from the preprocessor).
        hidden_dim: Width of the hidden / residual representation.
        n_blocks: Number of residual blocks.
        dropout: Dropout probability used in the stem and residual blocks.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        n_blocks: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_features = in_features
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
        """Forward pass.

        Args:
            x: Feature tensor of shape ``(batch, in_features)``.

        Returns:
            Logit tensor of shape ``(batch,)`` (squeezed).
        """
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return survival probabilities for input features.

        Sets eval mode so BatchNorm/Dropout behave deterministically.

        Args:
            x: Feature tensor of shape ``(batch, in_features)``.

        Returns:
            Probability tensor of shape ``(batch,)`` in ``[0, 1]``.
        """
        self.eval()
        return torch.sigmoid(self.forward(x))


if __name__ == "__main__":
    # Brick 4 verification: forward pass on a dummy batch.
    torch.manual_seed(0)
    model = TabularResNet(in_features=18)
    dummy = torch.randn(8, 18)
    logits = model(dummy)
    proba = model.predict_proba(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ok] logits shape: {tuple(logits.shape)}")
    print(f"[ok] proba shape:  {tuple(proba.shape)}, range [{proba.min():.3f}, {proba.max():.3f}]")
    print(f"[ok] trainable params: {n_params:,}")
