import torch
import torch.nn as nn
import math


class GPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # Single projection for Q, K, V 
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)     # Output projection

        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer(                                     # Causal mask
            "mask",
            torch.tril(torch.ones(config.n_positions, config.n_positions))
        )

    def forward(self, x):
        B, T, C = x.size()

        # Project to Q, K, V
        qkv = self.c_attn(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)

        # Reshape for multi-head
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        mask = self.mask[:T, :T]
        att = att.masked_fill(mask == 0, float("-inf"))

        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Merge heads
        return self.c_proj(y)