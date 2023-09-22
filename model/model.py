from typing import Union
import torch
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        self.conv1 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.bn1 = BatchNorm2d(num_features=out_channels)

        self.conv2 = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = BatchNorm2d(num_features=out_channels)

        if stride != 1:
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                ),
                BatchNorm2d(num_features=out_channels),
            )
        else:
            self.downsample = None

        self.act = ReLU(inplace=True)

    def forward(self, x) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample != None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        self.conv1 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
        )
        self.bn1 = BatchNorm2d(num_features=out_channels)

        self.conv2 = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = BatchNorm2d(num_features=out_channels)

        self.conv3 = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * self.expansion,
            kernel_size=1,
        )
        self.bn3 = BatchNorm2d(out_channels * self.expansion)

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                BatchNorm2d(num_features=out_channels * self.expansion),
            )
        else:
            self.downsample = None

        self.act = ReLU(inplace=True)

    def forward(self, x) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample != None:
            identity = self.downsample(identity)

        out += identity
        out = self.act(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Union[BasicBlock, Bottleneck],
        layers: list[int],
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        self.in_channels = 64

        self.conv1 = Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = BatchNorm2d(num_features=64)
        self.act = ReLU(inplace=True)
        self.maxpool = MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = self._make_layer(
            block, channels=64, num_layers=layers[0], stride=1
        )
        self.layer2 = self._make_layer(
            block, channels=128, num_layers=layers[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, channels=256, num_layers=layers[2], stride=2
        )
        self.layer4 = self._make_layer(
            block, channels=512, num_layers=layers[3], stride=2
        )

        self.avgpool = AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = Linear(in_features=512 * block.expansion, out_features=num_classes)

    def forward(self, x) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)

        return out

    def _make_layer(
        self,
        block: Union[BasicBlock, Bottleneck],
        channels: int,
        num_layers: int,
        stride: int,
    ) -> nn.Sequential:
        stride = [stride] + [1] * (num_layers - 1)

        layers = []
        for i in range(num_layers):
            layers.append(
                block(
                    in_channels=self.in_channels,
                    out_channels=channels,
                    stride=stride[i],
                )
            )
            self.in_channels = channels * block.expansion

        return nn.Sequential(*layers)


def resnet18() -> ResNet:
    return ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=10)


def resnet34() -> ResNet:
    return ResNet(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=10)


def resnet50() -> ResNet:
    return ResNet(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=10)


def resnet101() -> ResNet:
    return ResNet(block=Bottleneck, layers=[3, 4, 23, 3], num_classes=10)


def resnet152() -> ResNet:
    return ResNet(block=Bottleneck, layers=[3, 8, 36, 3], num_classes=10)


# test code
if __name__ == "__main__":
    model = ResNet(3, 64, 10)

    img = torch.randn(16, 3, 224, 224)
    x = model(img)
    print(x.shape)

    print(sum([p.numel() for p in model.parameters() if p.requires_grad]))
