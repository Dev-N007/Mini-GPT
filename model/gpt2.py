import torch.nn as nn
from .embeddings import GPT2Embeddings
from .block import GPT2Block
class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed = GPT2Embeddings(config)

        self.blocks = nn.ModuleList(
            [GPT2Block(config) for _ in range(config.n_layer)]
        )
        
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.lm_head.weight = self.embed.wte.weight # Weight tying

    def forward(self, input_ids):
        x = self.embed(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits
