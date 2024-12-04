import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, z_dim : int = 100):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
            # (7, 7, 64)
        )

        self.mu = nn.Linear(in_features=7*7*64, out_features=z_dim)
        self.logvar = nn.Linear(in_features=7*7*64, out_features=z_dim)

        self.decoder_fc = nn.Linear(in_features=z_dim, out_features=7*7*64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x):
        enc = self.encoder(x)
        enc = enc.reshape(enc.size(0), 7*7*64) # (batch_size, 7*7*64)
        mu, logvar = self.mu(enc), self.logvar(enc) # (batch_size, z_dim)

        z = self.reparameterize(mu, logvar)

        dec = self.decoder_fc(z) # (batch_size, 7*7*64)
        dec = dec.reshape(dec.size(0), 64, 7, 7) # (batch_size, 64, 7, 7)

        return self.decoder(dec), mu, logvar
