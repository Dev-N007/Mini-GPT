class GPT2Config:
    def __init__(
        self,
        vocab_size=50257,
        n_positions=512,
        n_embd=512,
        n_layer=6,
        n_head=8,
        dropout=0.1,
        initializer_range=0.02,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.initializer_range = initializer_range
