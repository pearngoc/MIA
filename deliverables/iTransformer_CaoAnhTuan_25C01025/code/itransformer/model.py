"""iTransformer (Liu et al., ICLR 2024): attention over variate tokens.
iTransformer: attention trên token biến (variate), theo Liu et al. ICLR 2024.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DataEmbeddingInverted(nn.Module):
    """Embed each variate's full series into one token / Nhúng mỗi lead thành một token."""

    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, L]  (N = n_variates)
        return self.dropout(self.value_embedding(x))


class FullAttention(nn.Module):
    def __init__(self, attention_dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, q, k, v):
        # q,k,v: [B, H, N, D]
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = self.dropout(torch.softmax(scores, dim=-1))
        return torch.matmul(attn, v)


class AttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.inner = FullAttention(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = self.inner(q, k, v).transpose(1, 2).contiguous().reshape(b, n, -1)
        return self.out(out)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = AttentionLayer(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.drop(self.attn(x)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class ITransformer(nn.Module):
    """Classification or forecasting head on inverted Transformer encoder.
    Head phân loại hoặc dự báo trên encoder Transformer đảo.
    """

    def __init__(
        self,
        task: str = 'classification',
        seq_len: int = 512,
        pred_len: int = 96,
        n_variates: int = 12,
        n_class: int = 5,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert task in ('classification', 'forecasting')
        self.task = task
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_variates = n_variates
        self.enc_embedding = DataEmbeddingInverted(seq_len, d_model, dropout)
        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        if task == 'classification':
            self.head = nn.Linear(d_model * n_variates, n_class)
        else:
            self.head = nn.Linear(d_model, pred_len)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        h = self.enc_embedding(x)
        for layer in self.encoder:
            h = layer(h)
        return self.norm(h)  # [B, C, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Chạy forward: cls -> logits [B, n_class]; fcst -> [B, C, H].
        """
        h = self.encode(x)
        if self.task == 'classification':
            return self.head(h.reshape(h.shape[0], -1))
        # Forecasting projects each variate token to horizon / Dự báo: mỗi token lead -> H bước
        return self.head(h)
