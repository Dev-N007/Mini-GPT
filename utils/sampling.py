import torch

def generate_text(model, input_ids, max_new_tokens=50):
    model.eval()

    for _ in range(max_new_tokens):
        logits = model(input_ids)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids