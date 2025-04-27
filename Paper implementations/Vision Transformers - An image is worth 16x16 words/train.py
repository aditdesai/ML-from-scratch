from typing import Tuple
import warnings
import torch
import torch.nn as nn
import torchvision.transforms as tfs
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
from model import VisionTransformer
from config import get_config


def create_dataloaders(img_size: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    transform = tfs.Compose([
        tfs.Resize(img_size),
        tfs.ToTensor()
    ])

    train_ds = MNIST(root="./dataset", train=True, download=True, transform=transform)
    test_ds = MNIST(root="./dataset", train=False, download=True, transform=transform)

    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_ds, shuffle=False, batch_size=batch_size)

    return train_dataloader, test_dataloader


def train(config: dict) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_dataloader, test_dataloader = create_dataloaders(config['img_size'], config['batch_size'])

    for batch in train_dataloader:
        print(batch)
        break

    model = VisionTransformer(config['d_model'], config['n_classes'], config['img_size'], config['patch_size'], config['n_channels'], config['n_heads'], config['n_layers'], config['dropout'])
    model = model.to(device)

    optim = torch.optim.Adam(params=model.parameters(), lr=config['lr'])
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(config['epochs']):

        model.train()
        train_loss, train_acc = 0, 0
        total_train_samples = 0
        for _, (img, label) in enumerate(train_dataloader):
            img, label = img.to(device), label.to(device)

            # Forward prop
            y_logits = model(img)

            # Convert logits to labels
            y_pred = torch.argmax(torch.softmax(y_logits, dim=-1), dim=-1)

            # Calculate loss
            loss = loss_fn(y_logits, label)
            train_loss += loss.item()

            # Calculate accuracy
            train_acc += (y_pred == label).sum().item()
            total_train_samples += label.size(0)

            # Back prop
            optim.zero_grad()
            loss.backward()
            optim.step()

        train_loss /= len(train_dataloader)
        train_acc = train_acc * 100.0 / total_train_samples

        model.eval()

        test_loss, test_acc = 0, 0
        total_test_samples = 0
        with torch.inference_mode():
            for _, (img, label) in enumerate(test_dataloader):
                img, label = img.to(device), label.to(device)

                # Forward prop
                y_logits = model(img)

                # Calculate loss
                loss = loss_fn(y_logits, label)
                test_loss += loss.item()

                # Convert logits to labels
                y_pred = torch.argmax(torch.softmax(y_logits, dim=-1), dim=-1)
                test_acc += (y_pred == label).sum().item()
                total_test_samples += label.size(0)

            test_loss /= len(test_dataloader)
            test_acc = test_acc * 100.0 / total_test_samples

        print(f"Epoch {epoch+1} / {config['epochs']} | Train Loss: {train_loss: .3f} | Train Accuracy: {train_acc: .2f}% | Test Loss: {test_loss: .3f} | Test Accuracy: {test_acc: .1f}%")



if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train(config)