import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
import wandb
import os
from argparse import ArgumentParser
from dotenv import load_dotenv
from model import Generator, Discriminator
from utils import save_generated_images

load_dotenv()
wandb.login(key=os.getenv("WANDB_KEY"))


def train(config):
    transform = tfs.Compose([
        tfs.Resize(config.img_dim),
        tfs.ToTensor(),
        tfs.Normalize((0.5,), (0.5,))
    ])

    train_ds = torchvision.datasets.MNIST(root='./data', download=True, train=True, transform=transform)
    train_dataloader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(
        name='cgan-mnist-test-run',
        project='CGAN MNIST Project',
        notes='Training cGAN on MNIST dataset',
        tags=['cGAN', 'MNIST'],
        entity='adit-desai-52'
    )

    gen = Generator().to(device)
    disc = Discriminator().to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    g_optim = torch.optim.Adam(params=gen.parameters(), lr=config.g_lr, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)
    d_optim = torch.optim.Adam(params=disc.parameters(), lr=config.d_lr, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)

    wandb.watch(gen)
    wandb.watch(disc)

    for epoch in range(config.num_epochs):
        gen.train()
        disc.train()

        for real_images, real_labels in train_dataloader:
            real_images, real_labels = real_images.to(device), real_labels.to(device)
            batch_size = real_images.size(0)

            # Train discriminator

            # Real images
            real_outputs = disc(real_images.detach(), real_labels)
            d_real_loss = loss_fn(real_outputs, torch.ones_like(real_outputs))

            # Fake images
            noise = torch.randn(batch_size, config.z, 1, 1).to(device)
            fake_images = gen(noise, real_labels)
            fake_outputs = disc(fake_images.detach(), real_labels)
            d_fake_loss = loss_fn(fake_outputs, torch.zeros_like(fake_outputs))

            # Combine losses
            d_loss = (d_real_loss + d_fake_loss) / 2

            # Backprop 
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()


            # Train Generator
            fake_outputs = disc(fake_images, real_labels)
            g_loss = loss_fn(fake_outputs, torch.ones_like(fake_outputs))

            # Backprop
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

        print(f"Epoch: {epoch+1}/{config.num_epochs} | Disc Loss: {d_loss: .4f} | Gen Loss: {g_loss: .4f}")

        wandb.log({
            "epoch": epoch+1,
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item()
        })

        if epoch == 0 or (epoch+1) % 25 == 0:
            save_generated_images(gen, epoch+1, device, config.z)



if __name__ == "__main__":
    parser = ArgumentParser(description="cGAN")

    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--z', type=int, default=100, help='Size of latent space')
    parser.add_argument('--emb-size', type=int, default=10, help='Label embedding size in generator')
    parser.add_argument('--img-dim', type=int, default=28, help='Image dimension')
    parser.add_argument('--d-lr', type=float, default=0.0002, help='Learning rate for discriminator')
    parser.add_argument('--g-lr', type=float, default=0.0002, help='Learning rate for generator')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')

    config = parser.parse_args()

    train(config)