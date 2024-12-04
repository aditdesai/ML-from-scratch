import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    def __init__(self, d_model: int, image_size: int, patch_size: int, n_channels: int):
        super().__init__()

        self.d_model = d_model
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_channels = n_channels

        self.linear_project = nn.Conv2d(in_channels=self.n_channels, out_channels=self.d_model, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        x = self.linear_project(x) # (b, c, h, w) --> (b, d_model, patch_row, patch_col)
        x = x.flatten(2) # (b, d_model, patch_row, patch_col) --> (b, d_model, P)
        x = x.transpose(1, 2) # (b, d_model, P) --> (b, P, d_model)

        return x
    

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)

        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    pe[pos][i] = torch.sin(pos / (10000 ** (i / d_model)))
                else:
                    pe[pos][i] = torch.cos(pos / (10000 ** ((i-1) / d_model)))

        self.register_buffer('pe', pe.unsqueeze(0))
            

    def forward(self, x):
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1) # (b, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1) # concatenate class tokens with patch embeddings
        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)
    

class AttentionHead(nn.Module):
    def __init__(self, d_model: int, d_k: int, dropout: float):
        super().__init__()

        self.d_k = d_k

        self.key = nn.Linear(d_model, d_k)
        self.query = nn.Linear(d_model, d_k)
        self.value = nn.Linear(d_model, d_k)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # dimensions of Q, K, V --> (b, seq_len, d_k) : projection to lower dimensional space

        attention = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) # (b, seq_len, seq_len)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(attention, dim=-1) # softmax along column
        attention = self.dropout(attention)
        attention = torch.matmul(attention, V) # (b, seq_len, d_k)

        return attention
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()

        self.d_k = d_model // n_heads

        self.W_o = nn.Linear(d_model, d_model)
        self.heads = nn.ModuleList([AttentionHead(d_model, self.d_k, dropout) for _ in range(n_heads)])

    def forward(self, x, mask=None):
        # Combine attention heads
        out = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        out = self.W_o(out)

        return out
    

class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, r_mlp: int = 4):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        # 1st Layer Normalization
        self.ln1 = nn.LayerNorm(d_model)

        # Multi-Head Attention
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)

        # 2nd Layer Normalization
        self.ln2 = nn.LayerNorm(d_model)

        # Multilayer Perceptron
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * r_mlp),
            nn.GELU(), # doesn’t have RELU’s limitation of being non-differentiable at zero.
            nn.Linear(d_model * r_mlp, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # Residual connection 1
        x = x + self.mha(self.ln1(x), mask=mask)

        # Residual connection 2
        x = x + self.mlp(self.ln2(x))

        return x
    

class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_seq_length: int, n_heads: int, n_layers: int, emb_dim: int, dropout: float):
        super().__init__()

        self.max_seq_length = max_seq_length

        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(d_model, max_seq_length)
        self.encoder = nn.ModuleList([TransformerEncoder(d_model, n_heads, dropout) for _ in range(n_layers)])

        # Projects the final encoder output to a joint multimodal embedding space
        self.projection = nn.Parameter(torch.randn(d_model, emb_dim))

    def forward(self, text, mask=None):
        # shape(text) --> (batch_size, seq_length)
        # shape(mask) --> (batch_size, seq_length, seq_length)

        x = self.encoder_embedding(text) # (batch_size, seq_length) --> (batch_size, seq_length, d_model)
        x = self.positional_embedding(x)
        
        for encoder_layer in self.encoder:
            x = encoder_layer(x, mask=mask)

        # Retrieves the embedding of the last valid token - EOS
        # torch.sum() gives total number of valid tokens in a sequence
        # Subtracting 1 gives index of EOS
        x = x[torch.arange(text.shape[0]), torch.sum(mask[:, 0], dim=1) - 1]

        # Joint multimodal embedding
        if self.projection is not None:
            x = torch.matmul(x, self.projection)

        x = x / torch.norm(x, dim=-1, keepdim=True)

        # (batch_size, emb_dim)
        return x 


class ImageEncoder(nn.Module):
    def __init__(self, d_model: int, img_size: tuple, patch_size: tuple, n_channels: int, n_layers: int, n_heads: int, emb_dim: int, dropout: float):
        super().__init__()

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "img_size dimensions must be divisible by patch_size dimensions"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.max_seq_len = self.n_patches + 1
        self.patch_embedding = PatchEmbedding(d_model, img_size, patch_size, n_channels)
        self.position_embedding = PositionalEmbedding(d_model, self.max_seq_len, dropout)
        self.encoder = nn.ModuleList([TransformerEncoder(d_model, n_heads, dropout) for _ in range(n_layers)])

        # Projects the final encoder output to a joint multimodal embedding space
        self.projection = nn.Parameter(torch.randn(d_model, emb_dim))

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.position_embedding(x)

        for encoder_layer in self.encoder:
            x = encoder_layer(x)

        # Getting class tokens
        x = x[:, 0, :]

        # Joint multimodal embedding
        if self.projection is not None:
            x = torch.matmul(x, self.projection)

        x = x / torch.norm(x, dim=-1, keepdim=True)

        return x


class CLIP(nn.Module):
    def __init__(self, emb_dim: int, vit_d_model: int, img_size: tuple, patch_size: tuple, n_channels: int, vit_layers: int, vit_heads: int, vocab_size: int, text_d_model: int, max_seq_length: int, text_heads: int, text_layers: int, dropout: float):
        self.image_encoder = ImageEncoder(vit_d_model, img_size, patch_size, n_channels, vit_layers, vit_heads, emb_dim, dropout)
        self.text_encoder = TextEncoder(vocab_size, text_d_model, max_seq_length, text_heads, text_layers, emb_dim, dropout)
        self.temperature = nn.Parameter(torch.ones([]) * torch.log(1 / 0.07))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, image, text, mask=None):
        I_e = self.image_encoder(image)
        T_e = self.text_encoder(text, mask=mask)
        
        # scaled pairwise cosine similarity scores
        # logits[i,j] represents the similarity between the i-th image and j-th text
        logits = torch.matmul(I_e, T_e.transpose(-2, -1)) * torch.exp(self.temperature) # (batch_size, batch_size)

        labels = torch.arange(logits.shape[0]).to(self.device) # [0, 1, 2, …, batch_size−1]

        # Contrastive Loss
        # image-to-text loss - how well each image embedding matches its corresponding text embedding
        loss_i = nn.functional.cross_entropy(logits.transpose(-2, -1), labels)
        
        # text-to-image loss - how well each text embedding matches its corresponding image embedding
        loss_t = nn.functional.cross_entropy(logits, labels)

        loss = (loss_i + loss_t) / 2

        return loss
