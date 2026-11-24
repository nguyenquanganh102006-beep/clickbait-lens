from torch import nn


class TypeHead(nn.Module):
    """Return logits [B,5], independent sigmoids, never softmax."""

    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 5)

    def forward(self, features):
        return self.classifier(features)
