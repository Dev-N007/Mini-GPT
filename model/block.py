import torch.nn as nn
from .attention import GPT2Attention
from .mlp import GPT2MLP

class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = GPT2Attention(config)

        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = GPT2MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x