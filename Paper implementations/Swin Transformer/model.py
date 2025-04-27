import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=4, d_model=96):
        super().__init__()

        self.linear_embedding = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.rearrange = Rearrange('b c h w -> b (h w) x')

    def forward(self, x):
        x = self.linear_embedding(x)
        x = self.rearrange(x) # (batch_size, num_patches, d_model)

        return x
    

class PatchMerging(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.lin = nn.Linear(4*d_model, 2*d_model)

    def forward(self, x):
        batch_size, num_patches, d_model = x.shape

        H = W = int(np.sqrt(num_patches) / 2)
        x = rearrange(x, 'b (h s1 w s2) x -> b (h w) (s1 s2 c)', s1=2, s2=2, h=H, w=W)
        x = self.lin(x)

        return x
    

class ShiftedWindowMSA(nn.Module):
    def __init__(self, d_model, n_heads, window_size=4, shifted=True):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.shifted = shifted
        self.lin = nn.Linear(d_model, 3*d_model)

        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

        self.head_dim = self.d_model // self.n_heads

    def forward(self, x):
        h = w = int(np.sqrt(x.shape[1]))
        x = self.lin(x)