"""
12-Layer Model with Iterative Refinement
=========================================

The critical missing test!

We know:
- 12-layer baseline: 1.4846 (no better than 6-layer)
- 7-layer + refinement: 1.0688 (NEW RECORD)

Question: Does 12-layer + refinement beat 7-layer + refinement?

This will tell us if:
A) Deeper models learn better representations (12-layer refinement < 1.0688)
B) 7 layers is optimal for this dataset size (12-layer refinement ~1.12)

This is potentially the most important experiment yet!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import gc
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import urllib.request

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

batch_size = 64
block_size = 256
n_embd = 384
n_head = 6
n_layer = 12  # THE KEY TEST
dropout = 0.2
vocab_size = None

learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
eval_interval = 100

patience = 15
min_delta = 0.0005
max_iters = 50000

num_activation_batches = 300
refinement_patience = 10

results_dir = './12layer_refinement_results'
os.makedirs(results_dir, exist_ok=True)

print(f"Using device: {device}")
print(f"Results will be saved to: {results_dir}")
print(f"\n{'='*70}")
print(f"12-LAYER MODEL + ITERATIVE REFINEMENT")
print(f"The critical missing test!")
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
# MODEL ARCHITECTURE
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
    
    def get_penultimate_activations(self, idx):
        """Get activations before final classifier for refinement."""
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        return x
    
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
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
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

class ActivationClassifier(nn.Module):
    """Simple classifier for refinement stage."""
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(n_embd, vocab_size)
    
    def forward(self, x, targets=None):
        logits = self.classifier(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss

# ============================================================================
# TRAINING UTILITIES
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

def train_to_convergence(model, optimizer, stage_name):
    """Train a model to convergence."""
    print(f"\n{'='*70}")
    print(f"Training {stage_name}")
    print(f"{'='*70}\n")
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_losses': [],
        'val_losses': [],
        'steps': [],
        'converged_at_step': None,
        'best_val_loss': None,
    }
    
    for iter_num in range(max_iters):
        if iter_num % eval_interval == 0:
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
                torch.save(model.state_dict(), f"{results_dir}/{stage_name}_best.pth")
            else:
                patience_counter += 1
                print(f" (patience: {patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    print(f"\n{'='*70}")
                    print(f"Converged at step {iter_num}, loss {best_val_loss:.4f}")
                    print(f"{'='*70}\n")
                    history['converged_at_step'] = iter_num
                    history['best_val_loss'] = best_val_loss
                    break
        
        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    model.load_state_dict(torch.load(f"{results_dir}/{stage_name}_best.pth"))
    return history

# ============================================================================
# REFINEMENT UTILITIES
# ============================================================================

class ActivationDataset(Dataset):
    def __init__(self, activations, targets):
        self.activations = activations
        self.targets = targets
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return self.activations[idx], self.targets[idx]

def collect_activations(model, num_batches):
    """Collect penultimate activations from frozen model."""
    print(f"\nCollecting {num_batches} batches of activations...")
    model.eval()
    
    all_activations = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(num_batches):
            xb, yb = get_batch('train')
            activations = model.get_penultimate_activations(xb)
            all_activations.append(activations.cpu())
            all_targets.append(yb.cpu())
            
            if (i + 1) % 50 == 0:
                print(f"  Collected {i+1}/{num_batches} batches...")
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
    
    print("Concatenating activations...")
    activations_tensor = torch.cat(all_activations, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    
    del all_activations, all_targets
    gc.collect()
    
    print(f"Total samples: {len(activations_tensor)}")
    return activations_tensor, targets_tensor

def train_refinement_classifier(base_model):
    """Train a fresh classifier on frozen base model representations."""
    print(f"\n{'='*70}")
    print(f"REFINEMENT: Training classifier on 12-layer frozen representations")
    print(f"{'='*70}")
    
    # Freeze base model
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Collect activations
    activations, targets = collect_activations(base_model, num_activation_batches)
    
    # Create dataset
    dataset = ActivationDataset(activations, targets)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Train classifier
    classifier = ActivationClassifier().to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'val_losses': [], 'steps': [], 'best_val_loss': None}
    
    iter_num = 0
    max_iter = 10000
    eval_int = 50
    
    print("\nTraining classifier:")
    while iter_num < max_iter:
        if iter_num % eval_int == 0:
            # Evaluate
            classifier.eval()
            val_losses = []
            for acts, targs in val_loader:
                acts, targs = acts.to(device), targs.to(device)
                _, loss = classifier(acts, targs)
                val_losses.append(loss.item())
                if len(val_losses) >= eval_iters:
                    break
            val_loss = np.mean(val_losses)
            classifier.train()
            
            history['steps'].append(iter_num)
            history['val_losses'].append(val_loss)
            
            print(f"step {iter_num:5d}: val {val_loss:.4f}", end='')
            
            if val_loss < best_val_loss - min_delta:
                improvement = best_val_loss - val_loss
                best_val_loss = val_loss
                patience_counter = 0
                print(f" ✓ (improved by {improvement:.4f})")
                torch.save(classifier.state_dict(), f"{results_dir}/refinement_best.pth")
            else:
                patience_counter += 1
                print(f" (patience: {patience_counter}/{refinement_patience})")
                
                if patience_counter >= refinement_patience:
                    print(f"\nConverged at loss: {best_val_loss:.4f}\n")
                    history['best_val_loss'] = best_val_loss
                    break
        
        # Training step
        for acts, targs in train_loader:
            acts, targs = acts.to(device), targs.to(device)
            _, loss = classifier(acts, targs)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            iter_num += 1
            if iter_num % eval_int == 0:
                break
    
    classifier.load_state_dict(torch.load(f"{results_dir}/refinement_best.pth"))
    
    # Clean up
    del activations, targets, dataset, train_dataset, val_dataset, train_loader, val_loader
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return classifier, history

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    experiment_start = datetime.now()
    print(f"\n{'='*70}")
    print("12-LAYER MODEL + ITERATIVE REFINEMENT")
    print(f"Started: {experiment_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    results = {
        'start_time': experiment_start.isoformat(),
        'n_layers': n_layer,
    }
    
    # ========================================================================
    # BASELINE: Train 12-layer model
    # ========================================================================
    
    print(f"\nPhase 1: Baseline 12-layer model")
    
    model = GPTLanguageModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    results['n_parameters'] = n_params
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    baseline_history = train_to_convergence(model, optimizer, '12layer_baseline')
    results['baseline'] = baseline_history
    
    baseline_loss = baseline_history['best_val_loss']
    print(f"\n12-layer baseline: {baseline_loss:.4f}")
    
    # Generate sample
    print(f"\nGenerating sample from 12-layer baseline:")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = model.generate(context, max_new_tokens=200)
    sample_text = decode(sample[0].tolist())
    print(sample_text[:300])
    with open(f"{results_dir}/baseline_sample.txt", 'w') as f:
        f.write(sample_text)
    
    # ========================================================================
    # REFINEMENT: Train fresh classifier
    # ========================================================================
    
    print(f"\nPhase 2: Iterative refinement")
    
    classifier, refinement_history = train_refinement_classifier(model)
    results['refinement'] = refinement_history
    
    refinement_loss = refinement_history['best_val_loss']
    improvement = baseline_loss - refinement_loss
    improvement_pct = (improvement / baseline_loss) * 100
    
    print(f"\n12-layer + refinement: {refinement_loss:.4f}")
    print(f"Improvement: {improvement:.4f} ({improvement_pct:.2f}%)")
    
    # ========================================================================
    # FINAL COMPARISON
    # ========================================================================
    
    print(f"\n{'='*70}")
    print(f"{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"{'='*70}\n")
    
    print("ALL RESULTS:")
    print(f"  3-layer baseline: 1.4859")
    print(f"  3-layer + refinement: 1.1140 (25.03%)")
    print()
    print(f"  5-layer baseline: 1.4744")
    print(f"  5-layer + refinement: 1.1194 (24.08%)")
    print()
    print(f"  6-layer baseline: 1.4811")
    print(f"  6-layer + refinement: 1.1193 (24.08%)")
    print()
    print(f"  7-layer baseline: 1.4744")
    print(f"  7-layer + refinement: 1.0688 (27.51%) ← PREVIOUS BEST")
    print()
    print(f"  12-layer baseline: {baseline_loss:.4f}")
    print(f"  12-layer + refinement: {refinement_loss:.4f} ({improvement_pct:.2f}%)")
    print()
    
    # Comparison to previous best
    previous_best = 1.0688
    if refinement_loss < previous_best:
        diff = previous_best - refinement_loss
        pct = (diff / previous_best) * 100
        print(f"{'='*70}")
        print(f"✓✓✓ NEW RECORD! ✓✓✓")
        print(f"{'='*70}")
        print(f"12-layer + refinement BEATS 7-layer + refinement!")
        print(f"Improvement: {diff:.4f} ({pct:.2f}% better)")
        print()
        print("CONCLUSION: Deeper models learn BETTER representations!")
        print("The problem is the joint-trained classifier, not model depth.")
        print(f"{'='*70}")
    elif refinement_loss < 1.12:
        diff = refinement_loss - previous_best
        pct = (diff / previous_best) * 100
        print(f"12-layer + refinement is close but doesn't beat 7-layer")
        print(f"Difference: +{diff:.4f} ({pct:.2f}% worse)")
        print()
        print("CONCLUSION: 7 layers may be optimal for this dataset size")
    else:
        print(f"12-layer + refinement (~1.12) similar to 3/5/6 layer results")
        print()
        print("CONCLUSION: 12 layers is too deep, representations overfit")
    
    # Save results
    experiment_end = datetime.now()
    results['end_time'] = experiment_end.isoformat()
    results['duration_seconds'] = (experiment_end - experiment_start).total_seconds()
    results['previous_best'] = previous_best
    results['beats_previous_best'] = refinement_loss < previous_best
    
    with open(f"{results_dir}/experiment_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"Duration: {results['duration_seconds']/60:.1f} minutes")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
