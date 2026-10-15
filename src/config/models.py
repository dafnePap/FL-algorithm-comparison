from torch import nn
import torch
import torch.nn.functional as F

# (input channels, number of classes)
ARGS = {
    "mnist": (1, 10),
    "emnist": (1, 62),
    "fmnist": (1, 10),
    "cifar10": (3, 10),
    "cifar100": (3, 100),
}


class LeNet5(nn.Module):
    def __init__(self, dataset) -> None:
        super().__init__()
        in_channels, num_classes = ARGS[dataset]

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5, 1, 2),  # 28x28 -> 28x28
            nn.Tanh(),
            nn.AvgPool2d(2, 2),                  # 28x28 -> 14x14
            nn.Conv2d(6, 16, 5, 1),             # 14x14 -> 10x10
            nn.Tanh(),
            nn.AvgPool2d(2, 2),                  # 10x10 -> 5x5
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.feature(x))


class LogisticRegression(nn.Module):
    def __init__(self, dataset: str):
        super().__init__()
        if dataset not in ["mnist", "emnist", "fmnist"]:
            raise ValueError(f"Unsupported dataset: {dataset}")
        self.linear = nn.Linear(28 * 28, ARGS[dataset][1])

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))


def gn(num_channels, max_groups=8):
    # GroupNorm is more suitable than BatchNorm in federated settings
    num_groups = min(max_groups, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = gn(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = gn(planes)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                gn(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64

        # CIFAR-style ResNet stem
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = gn(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out).flatten(1)
        return self.fc(out)


def ResNet18(dataset: str):
    return ResNet(BasicBlock, [2, 2, 2, 2], ARGS[dataset][1])


class CNN(nn.Module):
    def __init__(self, dataset: str = "emnist"):
        super().__init__()
        in_channels, num_classes = ARGS[dataset]

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # 28 -> 14
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # 14 -> 7
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class VGG16(nn.Module):
    def __init__(self, dataset: str = "emnist"):
        super().__init__()
        in_channels, num_classes = ARGS[dataset]

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # 28 -> 14

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # 14 -> 7

            nn.Conv2d(128, 192, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # 7 -> 3

            nn.Conv2d(192, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Xavier initialization for convolutional and linear layers
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x))