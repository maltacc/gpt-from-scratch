import torch 
import torch.nn as nn
import torch.nn.functional as F
from consts import n_embd

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(8, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx)  # (B, T, C) -> (4, 8, vocab_size)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        logits = self.lm_head(token_embeddings + position_embeddings)

        # cross entropy loss
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            next_idx = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_idx), dim=1)  # (B, T+1)
        return idx