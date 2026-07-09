import torch.nn as nn

from rfsensing.models import register


@register("lenet")
class LeNet(nn.Module):
    """Small CNN in the spirit of SenseFi's LeNet baselines, made
    shape-agnostic with an adaptive pooling head."""

    def __init__(self, in_shape, num_classes, dropout=0.5):
        super().__init__()
        in_channels = in_shape[0]
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))