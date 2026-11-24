from torch import nn
from src.detection.head import DetectionHead
from src.type_classification.head import TypeHead
from src.consistency.head import ConsistencyHead


class MultiTaskHeads(nn.Module):
    """Members supply shared features [B,D] from their encoders/fusion models."""

    def __init__(self, input_dim):
        super().__init__()
        self.detection = DetectionHead(input_dim)
        self.types = TypeHead(input_dim)
        self.consistency = ConsistencyHead(input_dim)

    def forward(self, features):
        return {
            "detection": self.detection(features),
            "types": self.types(features),
            "consistency": self.consistency(features),
        }


class SmokeModel(nn.Module):
    """Small offline MLP for integration checks, not a research model."""

    def __init__(self, input_dim=8, hidden_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.heads = MultiTaskHeads(hidden_dim)

    def forward(self, features):
        return self.heads(self.encoder(features))
