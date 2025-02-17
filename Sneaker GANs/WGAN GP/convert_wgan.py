import torch
from torch import nn
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.gen = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=16, stride=1, padding=0),
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

    def forward(self, x):
        return self.gen(x)
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Generator()
model.load_state_dict(torch.load("wgan_weights.pth", weights_only=True))

model.eval()

noise = torch.randn(1, 100, 1, 1).to(device)
# fake = model(noise)

# fake_image = TF.to_pil_image((fake[0] * 0.5) + 0.5)

# plt.imshow(fake_image)
# plt.show()

torch.onnx.export(
    model,
    noise,
    "wgan.onnx",
    export_params=True,
    input_names=['noise']
)