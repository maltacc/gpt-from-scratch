import torch

block_size = 8
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iters = 300
eval_interval = 300
eval_iters = 200
learning_rate = 1e-3
head_size = 16
n_embd = 32  # introduce smaller embedding dimension to prevent memory explosion - cl100k has a vocab_size of 100k
n_head = 6
n_layer = 6
dropout = 0.2
