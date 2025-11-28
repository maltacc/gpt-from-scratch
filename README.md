# gpt-from-scratch

## Design Choices

### Tokenizer

- Used OpenAI's TikToken as a tokenizer. The model is trained on a small vocabulary, which suits subwords over word-level or character-level tokenization. It also provides better generalization to unknown words and memory efficiency, so we used BPE.

### Split

- Basic 80-20 training-validation for small datasets

### KV Caching
