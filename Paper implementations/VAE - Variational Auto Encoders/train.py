import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
import wandb
import os
from argparse import ArgumentParser
from dotenv import load_dotenv
from model import VAE
from utils import save_generated_images

load_dotenv()
wandb.login(key=os.getenv("WANDB_KEY"))

'''
For two Gaussian distributions:

P: Learned latent distribution N(μ, σ²)
Q: Standard normal distribution N(0, 1)

For a multivariate Gaussian, the KL divergence has a closed-form solution:
KL(N(μ, Σ) || N(0, I)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
'''
def vae_loss(recon_imgs, imgs, mu, logvar):
    # Reconstruction loss - forces reconstructed image to be as close to original image as possible
    BCE = nn.functional.binary_cross_entropy(recon_imgs, imgs, reduction='sum')

    # KL Divergence loss - regularization term that forces latent distribution to follow a standard normal distribution
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(config):
    transform = tfs.Compose([
        tfs.ToTensor()
    ])

    train_ds = torchvision.datasets.MNIST(root="./data", download=True, train=True, transform=transform)
    train_dataloader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae = VAE(config.z_dim).to(device)
    optim = torch.optim.Adam(params=vae.parameters(), lr=config.lr)

    wandb.init(
        name='vae-mnist-test-run',
        project='VAE MNIST Project',
        notes='Training VAE on MNIST dataset',
        tags=['VAE', 'MNIST'],
        entity='adit-desai-52'
    )

    wandb.watch(vae)

    for epoch in range(config.num_epochs):
        vae.train()
        train_loss = 0

        for imgs, _ in train_dataloader:
            imgs = imgs.to(device)

            recon_imgs, mu, logvar = vae(imgs)

            loss = vae_loss(recon_imgs, imgs, mu, logvar)
            train_loss += loss.item()

            optim.zero_grad()
            loss.backward()
            optim.step()

        train_loss /= len(train_dataloader)
        print(f"Epoch: {epoch+1}/{config.num_epochs} | Train Loss: {train_loss: .4f}")

        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss
        })

        if epoch == 0 or (epoch + 1) % 25 == 0:
            save_generated_images(vae, epoch+1, device, config.z_dim)


if __name__ == "__main__":
    parser = ArgumentParser(description="VAE MNIST")

    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--z-dim', type=int, default=100, help='Size of latent space')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for VAE')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')

    config = parser.parse_args()

    train(config)