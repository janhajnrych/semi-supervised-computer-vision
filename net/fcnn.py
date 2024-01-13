import torch.nn as nn


class FullyCnn(nn.Module):
    def __init__(self, input_channels=3):
        super(FullyCnn, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, input_tensor):
        return self.cnn(input_tensor)
