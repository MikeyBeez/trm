import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
from torch.utils.data import Dataset, DataLoader

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

print(f"Using device: {device}")

# Load tiny shakespeare dataset
import urllib.request

# Check for the dataset file
if os.path.exists('tiny_shakespeare.txt'):
    dataset_file = 'tiny_shakespeare.txt'
elif os.path.exists('input.txt'):
    dataset_file = 'input.txt'
else:
    print("Downloading tiny shakespeare dataset...")
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    urllib.request.urlretrieve(url, 'input.txt')
    dataset_file = 'input.txt'
    print("Download complete!")

print(f"Loading dataset from {dataset_file}...")
with open(dataset_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Character-level tokenization
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train/val split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.penultimate_activations = []
        self.ground_truths = []

    def forward(self, idx, targets=None, save_activations=False):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Save penultimate layer activations if requested
        if save_activations and targets is not None:
            self.penultimate_activations.append(x.detach().cpu())
            self.ground_truths.append(targets.detach().cpu())
        
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ========== STAGE 1: Normal Training with Activation Saving ==========
print("\n" + "="*60)
print("STAGE 1: Training model and saving penultimate activations")
print("="*60 + "\n")

model = GPTLanguageModel()
model = model.to(device)
print(f"Model parameters on device: {next(model.parameters()).device}")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Test a forward pass first
print("Testing forward pass...")
test_x, test_y = get_batch('train')
print(f"Test batch on device: {test_x.device}")
test_logits, test_loss = model(test_x, test_y, save_activations=False)
print(f"Forward pass successful, loss: {test_loss.item():.4f}")
print("Starting training...\n")

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    
    # Don't save activations during training - only after
    logits, loss = model(xb, yb, save_activations=False)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Now collect activations after training is complete
print("\nCollecting activations from trained model...")
model.eval()
with torch.no_grad():
    for i in range(500):  # Collect 500 batches worth
        xb, yb = get_batch('train')
        _ = model(xb, yb, save_activations=True)
        if i % 100 == 0:
            print(f"Collected {i} batches...")

print("\nSaving penultimate activations and ground truths...")
activations_array = torch.cat(model.penultimate_activations, dim=0).numpy()
ground_truths_array = torch.cat(model.ground_truths, dim=0).numpy()

with open('activations_and_targets.pkl', 'wb') as f:
    pickle.dump({
        'activations': activations_array,
        'ground_truths': ground_truths_array
    }, f)

print(f"Saved {activations_array.shape[0]} activation samples")
print(f"Activation shape: {activations_array.shape}")
print(f"Ground truth shape: {ground_truths_array.shape}")

# Save the trained model
torch.save(model.state_dict(), 'stage1_model.pth')
print("Stage 1 model saved")

# Generate sample text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\nSample generation from Stage 1 model:")
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

# ========== STAGE 2: Training on Penultimate Activations ==========
print("\n" + "="*60)
print("STAGE 2: Training classifier on penultimate activations")
print("="*60 + "\n")

class ActivationDataset(Dataset):
    def __init__(self, activations, targets):
        self.activations = torch.from_numpy(activations).float()
        self.targets = torch.from_numpy(targets).long()
        
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return self.activations[idx], self.targets[idx]

# Load saved activations
with open('activations_and_targets.pkl', 'rb') as f:
    data = pickle.load(f)

dataset = ActivationDataset(data['activations'], data['ground_truths'])
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class ActivationClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(n_embd, vocab_size)
    
    def forward(self, x, targets=None):
        # x shape: (B, T, n_embd)
        B, T, C = x.shape
        logits = self.classifier(x)
        
        if targets is None:
            loss = None
        else:
            logits = logits.view(B*T, vocab_size)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

classifier = ActivationClassifier().to(device)
optimizer2 = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)

max_iters_stage2 = 2000
eval_interval_stage2 = 200

@torch.no_grad()
def estimate_loss_stage2():
    out = {}
    classifier.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = []
        for acts, targets in loader:
            acts, targets = acts.to(device), targets.to(device)
            logits, loss = classifier(acts, targets)
            losses.append(loss.item())
            if len(losses) >= eval_iters:
                break
        out[split] = np.mean(losses)
    classifier.train()
    return out

print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples\n")

iter_count = 0
for epoch in range(100):  # Large number, will break early
    for acts, targets in train_loader:
        if iter_count >= max_iters_stage2:
            break
            
        if iter_count % eval_interval_stage2 == 0:
            losses = estimate_loss_stage2()
            print(f"step {iter_count}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        acts, targets = acts.to(device), targets.to(device)
        logits, loss = classifier(acts, targets)
        
        optimizer2.zero_grad(set_to_none=True)
        loss.backward()
        optimizer2.step()
        
        iter_count += 1
    
    if iter_count >= max_iters_stage2:
        break

# Final evaluation
losses = estimate_loss_stage2()
print(f"\nFinal Stage 2: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

# Save stage 2 model
torch.save(classifier.state_dict(), 'stage2_classifier.pth')
print("Stage 2 classifier saved")

print("\n" + "="*60)
print("Training complete!")
print("="*60)
print("\nFiles created:")
print("- activations_and_targets.pkl: Saved penultimate activations and ground truths")
print("- stage1_model.pth: Full trained model from Stage 1")
print("- stage2_classifier.pth: Classifier trained on activations from Stage 2")
