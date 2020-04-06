import torch as t
from torch import nn
from torch.nn import functional as F


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()

    def load(self, path):
        self.load_state_dict(t.load(path))

    def save(self, path):
        t.save(self.state_dict(), path)


class ResidualBlockBasic(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResidualBlockBasic, self).__init__()
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel))
        self.shortcut_branch = shortcut

    def forward(self, x):
        out = self.main_branch(x)
        residual = x if self.shortcut_branch is None else self.shortcut_branch(x)
        out += residual
        return F.relu(out)


class ResidualBlockBottleneck(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResidualBlockBottleneck, self).__init__()
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel * 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel * 4),
        )
        self.shortcut_branch = shortcut

    def forward(self, x):
        out = self.main_branch(x)
        residual = x if self.shortcut_branch is None else self.shortcut_branch(x)
        out += residual
        return F.relu(out)


class ResNet34(BasicModule):

    def __init__(self, num_classes=10):
        super(ResNet34, self).__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.present_channel = 64
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, channel, block_num, stride=1):

        if stride != 1 or self.present_channel != channel:
            shortcut = nn.Sequential(
                nn.Conv2d(self.present_channel, channel, 1, stride, bias=False),
                nn.BatchNorm2d(channel)
            )
        else:
            shortcut = None

        layers = []
        layers.append(ResidualBlockBasic(self.present_channel, channel, stride, shortcut))
        self.present_channel = channel
        for i in range(1, block_num):
            layers.append(ResidualBlockBasic(self.present_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = t.flatten(x, 1)

        logits = self.fc(x)
        output = F.softmax(logits)

        return logits, output


class ResNet50(BasicModule):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.present_channel = 64
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(self, channel, block_num, stride=1):

        if stride != 1 or self.present_channel != channel * 4:
            shortcut = nn.Sequential(
                nn.Conv2d(self.present_channel, channel * 4, 1, stride, bias=False),
                nn.BatchNorm2d(channel * 4)
            )
        else:
            shortcut = None

        layers = []
        layers.append(ResidualBlockBottleneck(self.present_channel, channel, stride, shortcut))
        self.present_channel = channel * 4
        for i in range(1, block_num):
            layers.append(ResidualBlockBottleneck(self.present_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = t.flatten(x, 1)

        logits = self.fc(x)
        output = F.softmax(logits)

        return logits, output

