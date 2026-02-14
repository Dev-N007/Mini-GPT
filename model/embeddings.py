import torch
import torch.nn as nn

class GPT2Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # Token embedding matrix (vocab_size × n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd) # Positional embedding matrix (n_positions × n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        B, T = input_ids.size()
        position_ids = torch.arange(0, T, device=input_ids.device) # Create position indices [0, 1, 2, ..., T-1]
        position_ids = position_ids.unsqueeze(0)

        token_embeddings = self.wte(input_ids)
        position_embeddings = self.wpe(position_ids)

        x = token_embeddings + position_embeddings
        return self.dropout(x)
