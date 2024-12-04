import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    '''
    Splits input image into patches and creates a sequence of linear embeddings of these patches.

    Attributes:
        d_model: An integer representing the dimensionality of the model.
        image_size: A tuple of the image size.
        patch_size: A tuple of the patch size.
        n_channels: An integers representing the number of channels in input image.
        linear_project: A Conv2d layer responsible for creating the patches.
    '''

    def __init__(self, d_model: int, image_size: tuple, patch_size: tuple, n_channels: int) -> None:
        super().__init__()

        self.d_model = d_model
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_channels = n_channels

        self.linear_project = nn.Conv2d(in_channels=self.n_channels, out_channels=d_model, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        x = self.linear_project(x) # (b, c, h, w) --> (b, d_model, patch_col, patch_row)
        x = x.flatten(2) # collapses dimensions starting from 2 --> (b, d_model, P)
        x = x.transpose(1, 2) # swaps 1 and 2 --> (b, P, d_model)

        return x
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Learnable classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding
        pe = torch.zeros(seq_len, d_model)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1) # (b, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1) # concatenate class tokens with patch embeddings
        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)
    

class AttentionHead(nn.Module):
    def __init__(self, d_model: int, d_k: int, dropout: float) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_k = d_k

        self.query = nn.Linear(d_model, d_k)
        self.key = nn.Linear(d_model, d_k)
        self.value = nn.Linear(d_model, d_k)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention = torch.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        attention = torch.matmul(attention, V)

        return attention
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_k = self.d_model // self.n_heads

        self.w_o = nn.Linear(d_model, d_model)
        self.heads = nn.ModuleList([AttentionHead(self.d_model, self.d_k, dropout) for _ in range(self.n_heads)])

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.w_o(out)

        return out


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float, r_mlp: int = 4) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_model * r_mlp)
        self.gelu = nn.GELU() # doesn’t have RELU’s limitation of being non-differentiable at zero.
        self.linear_2 = nn.Linear(d_model * r_mlp, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear_2(self.dropout(self.gelu(self.linear_1(x))))
    

class Encoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, r_mlp: int = 4) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        # Layer Norm 1
        self.ln1 = nn.LayerNorm(d_model)

        # Multihead Attention
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)

        # Layer Norm 2
        self.ln2 = nn.LayerNorm(d_model)

        # Multilayer Perceptron
        self.mlp = FeedForwardBlock(d_model, dropout, r_mlp)

    def forward(self, x):
        # Residual connection 1
        out = x + self.mha(self.ln1(x))

        # Residual connection 2
        out = out + self.mlp(self.ln2(out))

        return out
    

class VisionTransformer(nn.Module):
    def __init__(self, d_model: int, n_classes: int, image_size: tuple, patch_size: tuple, n_channels: int, n_heads: int, n_layers: int, dropout: float) -> None:
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0, "image_size dimensions must be divisble by patch_size dimensions"

        self.d_model = d_model
        self.n_classes = n_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout

        self.n_patches = (self.image_size[0] * self.image_size[1]) // (self.patch_size[0] * self.patch_size[1])
        self.seq_len = self.n_patches + 1

        self.patch_embedding = PatchEmbedding(self.d_model, self.image_size, self.patch_size, self.n_channels)
        self.positional_encoding = PositionalEncoding(self.d_model, self.seq_len, self.dropout)
        self.vit_encoder = nn.Sequential(*[Encoder(self.d_model, self.n_heads, self.dropout) for _ in range(self.n_layers)])

        # Classification MLP
        self.classifier = nn.Linear(self.d_model, self.n_classes)

    def forward(self, images):
        x = self.patch_embedding(images) # (b, c, h, w) --> (b, n_patches, d_model)
        x = self.positional_encoding(x) # (b, n_patches, d_model) --> (b, n_patches + 1, d_model)
        x = self.vit_encoder(x) # (b, n_patches + 1, d_model)

        # x[:, 0] because learned class token is at index 0
        x = self.classifier(x[:, 0]) # (b, n_classes)

        return x

