import torch
import torch.nn as nn
import timm
from src.config import Config


class DriverClassifier(nn.Module):
    """Transfer Learning Classifier for Distracted Driver Detection."""

    def __init__(self, model_name: str = "efficientnet_b0", pretrained: bool = True):
        super(DriverClassifier, self).__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=Config.NUM_CLASSES,
        )

    def forward(self, x):
        return self.model(x)


def build_model(model_name: str = "efficientnet_b0", pretrained: bool = True):
    """Instantiates model and transfers it to the target device (CPU/CUDA)."""
    model = DriverClassifier(model_name=model_name, pretrained=pretrained)
    return model.to(Config.DEVICE)