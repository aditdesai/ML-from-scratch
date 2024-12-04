import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z):
        super().__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(in_features=z, out_features=7*7*64),
            nn.ReLU(inplace=True)
        )

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.linear1(x)
        x = x.reshape(-1, 64, 7, 7)
        x = self.conv_layers(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # (7, 7, 64)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=7*7*64, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)

        return x
