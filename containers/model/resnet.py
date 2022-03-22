import torch
from torch import nn
from torchvision.models import resnet18


class ResNet(nn.Module):
    def __init__(self, pretrained: bool=True, num_classes: int=10) -> None:
        super().__init__()
        self.model = resnet18(pretrained=pretrained)
        self.m = nn.LogSoftmax(dim=1)

        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=self.model.conv1.kernel_size,
            stride=self.model.conv1.stride,
            padding=self.model.conv1.padding,
            bias=False
        )
        self.model.fc = nn.Linear(
            in_features=self.model.fc.in_features,
            out_features=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return self.m(x)
