import torch
from config import GPT2Config
from model.gpt2 import GPT2
from tokenizer import get_tokenizer
from utils.sampling import generate_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = GPT2Config()
model = GPT2(config).to(device)

checkpoint = torch.load("checkpoints/gpt2_epoch_1.pt")
model.load_state_dict(torch.load("checkpoints/model.pt"))
model.eval()

tokenizer = get_tokenizer()

prompt = "Machine learning is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

output = generate_text(model, input_ids, max_new_tokens=50)

print(tokenizer.decode(output[0]))