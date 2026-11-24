from torch import nn


class ConsistencyHead(nn.Module):
    """Return logits [B,3 relations,3 classes]; NA is an ignore mask."""

    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 9)

    def forward(self, features):
        return self.classifier(features).reshape(-1, 3, 3)
