import torch.nn as nn

__all__ = ["SNetCIFAR", "SNetMNIST"]


class SNetCIFAR(nn.Module):
    """ Small custom convolutional cifar model. Has ~149K parameters. """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 128, 5, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

        self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(128, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.avgpool2d(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SNetMNIST(nn.Module):
    """ Small custom convolutional mnist model"""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(32, num_classes)

    def forward(self, x):
        if len(x.size()) == 3:
            x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.avgpool2d(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
