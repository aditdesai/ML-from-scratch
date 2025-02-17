import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, label_embed_size=10):
        super().__init__()

        z_dim = 100
        num_classes = 5

        self.label_embedding = nn.Embedding(num_classes, label_embed_size)

        self.gen = nn.Sequential(
            nn.ConvTranspose2d(in_channels=z_dim+label_embed_size, out_channels=512, kernel_size=16, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x, label):
        label_embed = self.label_embedding(label)
        label_embed = label_embed.view(label_embed.shape[0], -1, 1, 1)  # (batch_size, z_dim, 1, 1), concat along dim=1
        x = torch.cat((x, label_embed), dim=1)

        return self.gen(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Generator().to(device)
model.load_state_dict(torch.load("cgan_weights.pth", weights_only=True))

model.eval()

noise = torch.randn(1, 100, 1, 1).to(device)
labels = torch.full((1,), 2, dtype=torch.long).to(device)
# fake = model(noise, labels)

# fake_image = TF.to_pil_image((fake[0] * 0.5) + 0.5)

# plt.imshow(fake_image)
# plt.show()

torch.onnx.export(
    model,
    (noise, labels),
    "cgan.onnx",
    export_params=True,
    input_names=['noise', 'label']
)