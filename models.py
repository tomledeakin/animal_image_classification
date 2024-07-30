import torch
import torch.nn as nn
from torchvision.models import ResNet


class SimpleCNN(nn.Module):  # for CIFAR10 dataset
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding="same")
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding="same")
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(in_features=2048, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class AdvancedCNN(nn.Module):  # for Animals dataset
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = self.make_block(3, 16)
        self.conv2 = self.make_block(16, 32)
        self.conv3 = self.make_block(32, 64)
        self.conv4 = self.make_block(64, 128)
        self.conv5 = self.make_block(128, 128)

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=6272, out_features=2048),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

    def make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

if __name__ == '__main__':
    model = AdvancedCNN()
    sample_input = torch.rand(8, 3, 224, 224)
    output = model(sample_input)
    print(output.shape)
