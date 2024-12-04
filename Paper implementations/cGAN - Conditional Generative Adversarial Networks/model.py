import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, num_classes: int = 10, img_dim: int = 28):
        super().__init__()

        self.img_dim = img_dim

        self.label_emb = nn.Embedding(num_classes, img_dim * img_dim)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # (7, 7, 64)
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, stride=1, padding=0)
        )

    def forward(self, x, label):
        label_embed = self.label_emb(label)
        label_embed = label_embed.reshape(-1, 1, self.img_dim, self.img_dim)

        x = torch.cat([x, label_embed], dim=1)

        return self.conv_layers(x)
    

class Generator(nn.Module):
    def __init__(self, num_classes: int = 10, z_dim: int = 100, emb_size: int = 10):
        super().__init__()

        self.emb_size = emb_size

        self.label_emb = nn.Embedding(num_classes, emb_size)

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=z_dim+emb_size, out_channels=64, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x, label):
        label_embed = self.label_emb(label)
        label_embed = label_embed.reshape(-1, self.emb_size, 1, 1)

        x = torch.cat([x, label_embed], dim=1)

        return self.conv_layers(x)