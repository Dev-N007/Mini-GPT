import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import GPT2Config
from model.gpt2 import GPT2
from tokenizer import get_tokenizer
from dataset import TextDataset
from utils.init_weights import init_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = GPT2Config()
model = GPT2(config).to(device)

model.apply(lambda m: init_weights(m, config.initializer_range))

tokenizer = get_tokenizer()
text = open("data/input.txt", encoding="utf-8").read()

dataset = TextDataset(text, tokenizer, block_size=128)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

model.train()

checkpoint = torch.load("checkpoints/gpt2_epoch_1.pt")

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

start_epoch = checkpoint["epoch"] + 1

for epoch in range(start_epoch, 3):
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = F.cross_entropy(
            logits.view(-1, config.vocab_size),
            y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch:", epoch, "Loss:", loss.item())
    import os

    os.makedirs("checkpoints", exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss.item()
    }, f"checkpoints/gpt2_epoch_{epoch}.pt")

torch.save(model.state_dict(), "checkpoints/gpt2_final.pt")
