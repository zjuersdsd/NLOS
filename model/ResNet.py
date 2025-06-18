import torch
import torch.nn as nn
import torch.nn.functional as F
from model.feature_extractor import FeatureExtractor_STFRFT, FeatureExtractor_STFT, FeatureExtractor_spec


class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet2D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2, num_channels=2, order_is_trainable=None, order=0.5, transform=None, only_use_real=None,):
        super(ResNet2D, self).__init__()
        if  transform == "frft":
            self.feature_extractor = FeatureExtractor_STFRFT(256, order, 128, order_is_trainable, only_use_real)
            if only_use_real:
                self.conv1 = nn.Conv2d(num_channels*1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            else:
                self.conv1 = nn.Conv2d(num_channels*2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif transform == "stft":
            self.feature_extractor = FeatureExtractor_STFT(256, 128)
            self.conv1 = nn.Conv2d(num_channels*2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif transform == "spectrograms":
            self.feature_extractor = FeatureExtractor_spec(256, 128)
            self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.in_channels = 64

        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0] )
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def ResNet18_2D(num_classes=2,
                num_channels=2,
                transform=None,
                order_is_trainable=None,
                order=None,
                only_use_real=None,
                ):
    return ResNet2D(BasicBlock2D, [2, 2, 2, 2],
                    num_classes=num_classes,
                    num_channels=num_channels,
                    transform=transform,
                    order_is_trainable=order_is_trainable,
                    order=order,
                    only_use_real=only_use_real,
                    )
def ResNet34_2D(num_classes=2,
                num_channels=2,
                transform=None,
                order_is_trainable=None,
                order=None,
                only_use_real=None,
                ):
    return ResNet2D(
        BasicBlock2D,
        [3, 4, 6, 3],  # ResNet34 block counts for each layer
        num_classes=num_classes,
        num_channels=num_channels,
        transform=transform,
        order_is_trainable=order_is_trainable,
        order=order,
        only_use_real=only_use_real,
        )



# Test the model
if __name__ == "__main__":
    model = ResNet18_2D(num_classes=2, num_channels=2)
    x = torch.randn(16, 2, 4800)  # Example input for 2D data
    output = model(x)
    print(output.shape)
