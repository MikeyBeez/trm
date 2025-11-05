"""
12-Layer Model Baseline Experiment
====================================

Train a 12-layer model from scratch to compare with:
- 6-layer baseline: ~1.48 loss
- 6-layer + iterative refinement: ~1.12 loss

Question: Does doubling the layers match or beat iterative refinement?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from datetime import datetime
import urllib.request

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

batch_size = 64
block_size = 256
n_embd = 384
n_head = 6
n_layer = 12  # DOUBLED from 6
dropout = 0.2
vocab_size = None

learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
eval_interval = 100

patience = 15
min_delta = 0.0005
max_iters = 50000

results_dir = './model_12layer_results'
os.makedirs(results_dir, exist_ok=True)

print(f"Using device: {device}")
print(f"Results will be saved to: {results_dir}")
print(f"\n{'='*70}")
print(f"TESTING: {n_layer}-LAYER MODEL")
print(f"Comparing to: 6-layer baseline (~1.48) and 6-layer + refinement (~1.12)")
print(f"{'='*70}\n")

# ============================================================================
# DATA LOADING
# ============================================================================

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

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(f"Vocabulary size: {vocab_size}")
print(f"Dataset size: {len(text)} characters")

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# ============================================================================
# MODEL ARCHITECTURE (12 LAYERS)
# ============================================================================

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
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
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
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ============================================================================
# TRAINING
# ============================================================================

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    
    model.train()
    return out

def train_to_convergence(model, optimizer):
    print(f"\n{'='*70}")
    print(f"Training {n_layer}-layer model to convergence")
    print(f"{'='*70}\n")
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_losses': [],
        'val_losses': [],
        'steps': [],
        'converged_at_step': None,
        'best_val_loss': None,
        'final_train_loss': None,
    }
    
    for iter_num in range(max_iters):
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            losses = estimate_loss(model)
            history['steps'].append(iter_num)
            history['train_losses'].append(losses['train'])
            history['val_losses'].append(losses['val'])
            
            print(f"step {iter_num:5d}: train {losses['train']:.4f}, val {losses['val']:.4f}", end='')
            
            if losses['val'] < best_val_loss - min_delta:
                improvement = best_val_loss - losses['val']
                best_val_loss = losses['val']
                patience_counter = 0
                print(f" ✓ (improved by {improvement:.4f})")
                torch.save(model.state_dict(), f"{results_dir}/model_best.pth")
            else:
                patience_counter += 1
                print(f" (patience: {patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    print(f"\n{'='*70}")
                    print(f"Converged at step {iter_num}")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    print(f"{'='*70}\n")
                    history['converged_at_step'] = iter_num
                    history['best_val_loss'] = best_val_loss
                    history['final_train_loss'] = losses['train']
                    break
        
        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    model.load_state_dict(torch.load(f"{results_dir}/model_best.pth"))
    return history

# ============================================================================
# MAIN
# ============================================================================

def main():
    experiment_start = datetime.now()
    print(f"\n{'='*70}")
    print(f"12-LAYER MODEL EXPERIMENT")
    print(f"Started: {experiment_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # Count parameters
    model = GPTLanguageModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"(6-layer model has ~10.7M parameters)")
    print()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Train
    history = train_to_convergence(model, optimizer)
    
    # Save final model
    torch.save(model.state_dict(), f"{results_dir}/model_final.pth")
    
    # Generate sample
    print("\nGenerating sample text:")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = model.generate(context, max_new_tokens=300)
    sample_text = decode(sample[0].tolist())
    print(sample_text)
    print()
    
    with open(f"{results_dir}/sample.txt", 'w') as f:
        f.write(sample_text)
    
    # Save results
    experiment_end = datetime.now()
    duration = (experiment_end - experiment_start).total_seconds()
    
    results = {
        'start_time': experiment_start.isoformat(),
        'end_time': experiment_end.isoformat(),
        'duration_seconds': duration,
        'n_layers': n_layer,
        'n_parameters': n_params,
        'history': history,
    }
    
    with open(f"{results_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Comparison
    print(f"{'='*70}")
    print("COMPARISON TO OTHER MODELS")
    print(f"{'='*70}\n")
    
    baseline_6layer = 1.4811  # From your earlier run
    refinement_6layer = 1.1193  # From your earlier run (though this was wrong approach)
    this_model = history['best_val_loss']
    
    print(f"6-layer baseline:              {baseline_6layer:.4f}")
    print(f"6-layer + iterative refinement: {refinement_6layer:.4f}")
    print(f"12-layer (this run):            {this_model:.4f}")
    print()
    
    if this_model < refinement_6layer:
        diff = refinement_6layer - this_model
        pct = (diff / refinement_6layer) * 100
        print(f"✓ 12-layer BEATS refinement by {diff:.4f} ({pct:.2f}%)")
        print(f"  Conclusion: More layers is simply better")
    elif this_model < baseline_6layer:
        diff1 = baseline_6layer - this_model
        pct1 = (diff1 / baseline_6layer) * 100
        diff2 = this_model - refinement_6layer
        pct2 = (diff2 / refinement_6layer) * 100
        print(f"✓ 12-layer BEATS 6-layer baseline by {diff1:.4f} ({pct1:.2f}%)")
        print(f"✗ But LOSES to 6-layer refinement by {diff2:.4f} ({pct2:.2f}%)")
        print(f"  Conclusion: Iterative refinement extracts something joint training misses")
    else:
        diff = this_model - baseline_6layer
        pct = (diff / baseline_6layer) * 100
        print(f"✗ 12-layer is WORSE than 6-layer by {diff:.4f} ({pct:.2f}%)")
        print(f"  Conclusion: More layers hurt (overfitting or training issues)")
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
