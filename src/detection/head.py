from torch import nn


class DetectionHead(nn.Module):
    """Return logits [B], for BCEWithLogitsLoss (no sigmoid in forward)."""

    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, features):
        return self.classifier(features).squeeze(-1)
