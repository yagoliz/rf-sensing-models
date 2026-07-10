import torch.nn as nn

from rfsensing.models import register


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


@register("resnet18")
class ResNet18(nn.Module):
    """ResNet-18 with an input stem adapted to CSI channel counts."""

    def __init__(self, in_shape, num_classes, base_width=64):
        super().__init__()
        w = base_width
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[0], w, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layers = nn.Sequential(
            BasicBlock(w, w),
            BasicBlock(w, w),
            BasicBlock(w, 2 * w, stride=2),
            BasicBlock(2 * w, 2 * w),
            BasicBlock(2 * w, 4 * w, stride=2),
            BasicBlock(4 * w, 4 * w),
            BasicBlock(4 * w, 8 * w, stride=2),
            BasicBlock(8 * w, 8 * w),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8 * w, num_classes)

    def embed(self, x):
        x = self.layers(self.stem(x))
        return self.pool(x).flatten(1)

    @property
    def head(self):
        return self.fc

    def forward(self, x):
        return self.head(self.embed(x))