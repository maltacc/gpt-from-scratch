import requests
import tiktoken
import torch
from models.gpt import GPTModel
from consts import *

response = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")

# read input text file
with open("input.txt", "r", encoding='utf-8') as f:
    text = f.read()

# tokenizer 
encoding = tiktoken.get_encoding("cl100k_base")
vocab_size = encoding.n_vocab
tokens = encoding.encode(text)
# print(tokens[:100])

# transform into tensor
data = torch.tensor(tokens, dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:100])

# train, validation split 
split = 0.8
n = int(split * len(data))
training = data[:n]
validation = data[n:]

def get_batch(split):
    dataset = training if split == 'train' else validation
    ix = torch.randint(len(dataset) - block_size, (batch_size,)) # random offsets in training set
    x = torch.stack([dataset[i:i+block_size] for i in ix])
    y = torch.stack([dataset[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    """
    Reduce noise by getting average loss
    """

    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# TO-DO: implement kv caching
model = GPTModel(vocab_size=vocab_size)
m = model.to(device)

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters): # TO-DO: increase number of steps
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    # eval loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # print(f"Loss: {loss.item()}")

idx = torch.zeros((1, 1), dtype=torch.long)
generated_idx = model.generate(idx, max_new_tokens=500)
generated_text = encoding.decode(generated_idx[0].tolist())

print(generated_text)

