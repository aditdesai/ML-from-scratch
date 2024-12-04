import torch
import torch.nn as nn
from argparse import ArgumentParser
from dataset import FashionMNIST
from torch.utils.data import DataLoader
from model import CLIP


def train(config):
    train_ds = FashionMNIST(train=True)
    train_dataloader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIP(config.emb_dim, config.vit_d_model, (28, 28), (14, 14), 1, config.vit_layers, config.vit_heads, config.vocab_size, config.text_d_model, config.max_seq_length, config.text_heads, config.text_layers, 0.1)
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)

    model.train()
    best_loss = torch.inf
    for epoch in range(config.num_epochs):
        train_loss = 0
        for i, data in enumerate(train_dataloader):
            img, cap, mask = data["image"].to(device), data["caption"].to(device), data["mask"].to(device)

            loss = model(img, cap, mask)
            train_loss += loss.item()

            optim.zero_grad()
            loss.backward()
            optim.step()

        train_loss /= len(train_dataloader)
        print(f"Epoch: {epoch+1}/{config.num_epochs} | Train Loss: {train_loss: .3f}")

        if train_loss <= best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), "clip.pth")
            print("Model saved.")



if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--emb-dim", type=int, default=32, help="Joint multimodal embedding dimension")
    parser.add_argument("--vit-d-model", type=int, default=9, help="d_model of image encoder (ViT)")
    parser.add_argument("--vit-layers", type=int, default=3, help="Number of layers in image encoder (ViT)")
    parser.add_argument("--vit-heads", type=int, default=3, help="Number of heads in image encoder (ViT)")
    parser.add_argument("--vocab-size", type=int, default=256, help="Vocabulary size")
    parser.add_argument("--text-d-model", type=int, default=32, help="d_model of text encoder")
    parser.add_argument("--max-seq-length", type=int, default=32, help="Maximum sequence length")
    parser.add_argument("--text_d_model", type=int, default=32, help="d_model of text encoder")
    parser.add_argument("--text-layers", type=int, default=4, help="Number of layers in text encoder")
    parser.add_argument("--text-heads", type=int, default=8, help="Number of heads in text encoder")
    parser.add_argument('--num-epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')

    config = parser.parse_args()

    train(config)