import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
import wandb
import os
from dotenv import load_dotenv
import argparse
from model import Generator, Discriminator
from utils import save_generated_images, weights_init


load_dotenv()
wandb.login(key=os.getenv("WANDB_KEY"))


def train(config):
    transform = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize((0.5,), (0.5,))
    ])

    train_ds = torchvision.datasets.MNIST(root="./data", download=True, train=True, transform=transform)
    train_dataloader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    wandb.init(
        name='dcgan_mnist_test_run',
        project='DCGAN MNIST Project',
        notes='Training DCGAN on MNIST dataset',
        tags=['MNIST', 'DCGAN'],
        entity='adit-desai-52'
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    loss_fn = nn.BCELoss()

    gen = Generator(config.z).to(device)
    disc = Discriminator().to(device)

    gen.apply(weights_init)
    disc.apply(weights_init)

    g_optim = torch.optim.Adam(params=gen.parameters(), lr=config.g_lr, betas=(0.5, 0.999))
    d_optim = torch.optim.Adam(params=disc.parameters(), lr=config.d_lr, betas=(0.5, 0.999))

    wandb.watch(gen)
    wandb.watch(disc)
    
    for epoch in range(config.num_epochs):
        gen.train()
        disc.train()

        for real_images, _ in train_dataloader:
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            d_optim.zero_grad()

            # Real images
            real_outputs = disc(real_images)
            d_real_loss = loss_fn(real_outputs, real_labels)

            # Fake images
            noise = torch.randn(batch_size, config.z).to(device)
            fake_images = gen(noise)
            fake_outputs = disc(fake_images.detach())
            d_fake_loss = loss_fn(fake_outputs ,fake_labels)

            # Combine losses
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optim.step()


            # Train Generator
            g_optim.zero_grad()

            fake_outputs = disc(fake_images)
            g_loss = loss_fn(fake_outputs, real_labels)
            g_loss.backward()
            g_optim.step()

        print(f"Epoch: {epoch+1}/{config.num_epochs} | Disc Loss: {d_loss.item(): .4f} | Gen Loss: {g_loss.item(): .4f}")

        wandb.log({
            "epoch": epoch + 1,
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item()
        })

        if (epoch + 1) % 25 == 0:
            save_generated_images(gen, epoch, device, config.z)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCGAN")

    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--z', type=int, default=100, help='Size of latent space')
    parser.add_argument('--d-lr', type=float, default=0.0002, help='Learning rate for discriminator')
    parser.add_argument('--g-lr', type=float, default=0.0002, help='Learning rate for generator')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')

    config = parser.parse_args()

    train(config)