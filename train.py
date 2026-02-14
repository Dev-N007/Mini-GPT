import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

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

os.makedirs("checkpoints", exist_ok=True)

loss_history = []

model.train()

epochs = 2

for epoch in range(epochs):
    epoch_loss = 0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch}")

    for x, y in progress_bar:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = F.cross_entropy(
            logits.view(-1, config.vocab_size),
            y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        loss_value = loss.item()
        loss_history.append(loss_value)
        epoch_loss += loss_value

        progress_bar.set_postfix(loss=loss_value)

    avg_epoch_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch} Average Loss: {avg_epoch_loss:.4f}")

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_epoch_loss,
        "loss_history": loss_history
    }, f"checkpoints/gpt2_epoch_{epoch}.pt")

torch.save(model.state_dict(), "checkpoints/gpt2_final.pt")

plt.figure(figsize=(8,5))
plt.plot(loss_history)
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()
