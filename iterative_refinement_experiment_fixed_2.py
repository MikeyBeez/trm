"""
Recursive Depth Training Experiment - CORRECTED VERSION
========================================================

This implements the CORRECT architecture where:
- Stage 1: Train model normally (input -> transformer -> output)
- Stage 2: Feed penultimate representations back as input (penultimate -> transformer -> output)
- Stage 3+: Continue feeding representations back recursively

This effectively increases the model's depth without adding parameters.
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
n_layer = 6
dropout = 0.2
vocab_size = None

learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
eval_interval = 100

patience = 15
min_delta = 0.0005
max_iters = 50000

max_recursive_stages = 10
num_activation_batches = 300

results_dir = './recursive_depth_results'
os.makedirs(results_dir, exist_ok=True)

print(f"Using device: {device}")
print(f"Results will be saved to: {results_dir}")

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

class RecursiveGPT(nn.Module):
    """
    GPT that can process its own penultimate representations recursively.
    """
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None, use_embeddings=True):
        """
        Args:
            idx: Either token indices (if use_embeddings=True) or pre-computed embeddings (if use_embeddings=False)
            targets: Ground truth for loss computation
            use_embeddings: If True, idx contains token indices. If False, idx contains embeddings.
        """
        B, T = idx.shape[0], idx.shape[1]
        
        if use_embeddings:
            # Normal forward pass: embed tokens
            tok_emb = self.token_embedding_table(idx)
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))
            x = tok_emb + pos_emb
        else:
            # Recursive pass: idx already contains embeddings from previous pass
            # Still add positional embeddings
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))
            x = idx + pos_emb
        
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
        
        return logits, loss, x  # Return x (penultimate activations) for recursive passes
    
    def generate(self, idx, max_new_tokens, num_passes=1):
        """Generate with optional multiple recursive passes per token."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            
            # First pass
            logits, _, penultimate = self(idx_cond, use_embeddings=True)
            
            # Additional recursive passes
            for _ in range(num_passes - 1):
                logits, _, penultimate = self(penultimate, use_embeddings=False)
            
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

# ============================================================================
# DATASET FOR RECURSIVE TRAINING
# ============================================================================

class RecursiveActivationDataset(Dataset):
    """Dataset that provides penultimate activations as input."""
    def __init__(self, activations, targets):
        self.activations = activations
        self.targets = targets
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return self.activations[idx], self.targets[idx]

def collect_penultimate_activations(model, num_batches):
    """Collect penultimate layer activations from current model state."""
    print(f"\nCollecting {num_batches} batches of penultimate activations...")
    model.eval()
    
    all_activations = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(num_batches):
            xb, yb = get_batch('train')
            _, _, penultimate = model(xb, use_embeddings=True)
            
            all_activations.append(penultimate.cpu())
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
    print(f"Activation shape: {activations_tensor.shape}")
    
    return activations_tensor, targets_tensor

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

@torch.no_grad()
def estimate_loss(model, stage_num=1):
    """Estimate loss with appropriate number of recursive passes."""
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            
            # First pass
            logits, loss, penultimate = model(X, Y, use_embeddings=True)
            
            # Additional recursive passes for stages > 1
            for _ in range(stage_num - 1):
                logits, loss, penultimate = model(penultimate, Y, use_embeddings=False)
            
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    
    model.train()
    return out

def train_stage_1(model, optimizer):
    """Train model normally (single pass)."""
    print(f"\n{'='*70}")
    print(f"STAGE 1: Normal training to convergence")
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
            losses = estimate_loss(model, stage_num=1)
            history['steps'].append(iter_num)
            history['train_losses'].append(losses['train'])
            history['val_losses'].append(losses['val'])
            
            print(f"step {iter_num:5d}: train {losses['train']:.4f}, val {losses['val']:.4f}", end='')
            
            if losses['val'] < best_val_loss - min_delta:
                improvement = best_val_loss - losses['val']
                best_val_loss = losses['val']
                patience_counter = 0
                print(f" ✓ (improved by {improvement:.4f})")
                torch.save(model.state_dict(), f"{results_dir}/stage1_best.pth")
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
        
        # Training step (single pass)
        xb, yb = get_batch('train')
        logits, loss, _ = model(xb, yb, use_embeddings=True)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    model.load_state_dict(torch.load(f"{results_dir}/stage1_best.pth"))
    return history

def train_recursive_stage(model, optimizer, stage_num, train_loader, val_loader):
    """
    Train model recursively: penultimate activations fed back as input.
    The model continues learning, but now on a harder task.
    """
    print(f"\n{'='*70}")
    print(f"STAGE {stage_num}: Recursive training (effective depth: {stage_num * n_layer} layers)")
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
    
    iter_num = 0
    max_iterations = 10000
    eval_interval_recursive = 50
    patience_recursive = 10
    
    while iter_num < max_iterations:
        if iter_num % eval_interval_recursive == 0:
            # Evaluate with proper number of recursive passes
            losses = estimate_loss(model, stage_num=stage_num)
            
            history['steps'].append(iter_num)
            history['train_losses'].append(losses['train'])
            history['val_losses'].append(losses['val'])
            
            print(f"step {iter_num:5d}: train {losses['train']:.4f}, val {losses['val']:.4f}", end='')
            
            if losses['val'] < best_val_loss - min_delta:
                improvement = best_val_loss - losses['val']
                best_val_loss = losses['val']
                patience_counter = 0
                print(f" ✓ (improved by {improvement:.4f})")
                torch.save(model.state_dict(), f"{results_dir}/stage{stage_num}_best.pth")
            else:
                patience_counter += 1
                print(f" (patience: {patience_counter}/{patience_recursive})")
                
                if patience_counter >= patience_recursive:
                    print(f"\n{'='*70}")
                    print(f"Converged at step {iter_num}, loss {best_val_loss:.4f}")
                    print(f"{'='*70}\n")
                    history['converged_at_step'] = iter_num
                    history['best_val_loss'] = best_val_loss
                    break
        
        # Training step: feed activations recursively
        for acts, targets in train_loader:
            acts, targets = acts.to(device), targets.to(device)
            
            # Forward through model treating acts as embeddings
            logits, loss, _ = model(acts, targets, use_embeddings=False)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            iter_num += 1
            if iter_num % eval_interval_recursive == 0:
                break
        
        if iter_num >= max_iterations:
            break
    
    model.load_state_dict(torch.load(f"{results_dir}/stage{stage_num}_best.pth"))
    return history

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    experiment_start = datetime.now()
    print(f"\n{'='*70}")
    print("RECURSIVE DEPTH TRAINING EXPERIMENT")
    print(f"Started: {experiment_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    experiment_log = {
        'start_time': experiment_start.isoformat(),
        'architecture': f'{n_layer} layers, becomes {n_layer}x through recursion',
        'stages': {},
    }
    
    # Initialize model
    model = RecursiveGPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # ========================================================================
    # STAGE 1: Normal training
    # ========================================================================
    
    stage1_history = train_stage_1(model, optimizer)
    experiment_log['stages']['stage1'] = stage1_history
    
    baseline_loss = stage1_history['best_val_loss']
    print(f"\nBaseline (Stage 1) validation loss: {baseline_loss:.4f}")
    
    # Generate sample
    print("\nGenerating sample from Stage 1 (single pass):")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = model.generate(context, max_new_tokens=200, num_passes=1)
    sample_text = decode(sample[0].tolist())
    print(sample_text)
    with open(f"{results_dir}/stage1_sample.txt", 'w') as f:
        f.write(sample_text)
    
    # ========================================================================
    # STAGE 2+: Recursive training
    # ========================================================================
    
    previous_best_loss = baseline_loss
    
    for stage_num in range(2, max_recursive_stages + 2):
        print(f"\n{'='*70}")
        print(f"Preparing Stage {stage_num}")
        print(f"{'='*70}")
        
        # Collect activations from current model
        activations, targets = collect_penultimate_activations(model, num_activation_batches)
        
        # Create dataset
        dataset = RecursiveActivationDataset(activations, targets)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Train this recursive stage
        stage_history = train_recursive_stage(
            model, optimizer, stage_num, train_loader, val_loader
        )
        
        experiment_log['stages'][f'stage{stage_num}'] = stage_history
        
        current_best_loss = stage_history['best_val_loss']
        improvement = previous_best_loss - current_best_loss
        improvement_pct = (improvement / previous_best_loss) * 100
        
        print(f"\nStage {stage_num} vs Stage {stage_num-1}:")
        print(f"  Previous: {previous_best_loss:.4f}")
        print(f"  Current: {current_best_loss:.4f}")
        print(f"  Improvement: {improvement:.4f} ({improvement_pct:.2f}%)")
        
        print(f"\nStage {stage_num} vs Baseline:")
        total_improvement = baseline_loss - current_best_loss
        total_pct = (total_improvement / baseline_loss) * 100
        print(f"  Baseline: {baseline_loss:.4f}")
        print(f"  Current: {current_best_loss:.4f}")
        print(f"  Total improvement: {total_improvement:.4f} ({total_pct:.2f}%)")
        
        # Check if we should continue
        if improvement < min_delta:
            print(f"\n{'='*70}")
            print(f"CONVERGENCE: No significant improvement at stage {stage_num}")
            print(f"{'='*70}\n")
            experiment_log['final_stage'] = stage_num
            experiment_log['convergence_reason'] = 'no_improvement'
            break
        
        previous_best_loss = current_best_loss
        
        # Clean up for memory
        del activations, targets, dataset, train_dataset, val_dataset, train_loader, val_loader
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        if stage_num >= max_recursive_stages + 1:
            print(f"\n{'='*70}")
            print(f"Reached maximum stages ({max_recursive_stages})")
            print(f"{'='*70}\n")
            experiment_log['final_stage'] = stage_num
            experiment_log['convergence_reason'] = 'max_stages'
            break
    
    # ========================================================================
    # Save results
    # ========================================================================
    
    experiment_end = datetime.now()
    experiment_log['end_time'] = experiment_end.isoformat()
    experiment_log['duration_seconds'] = (experiment_end - experiment_start).total_seconds()
    
    with open(f"{results_dir}/experiment_log.json", 'w') as f:
        json.dump(experiment_log, f, indent=2)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"Duration: {experiment_log['duration_seconds']/60:.1f} minutes")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*70}\n")
    
    # Generate sample at final depth
    final_stage = experiment_log['final_stage']
    print(f"\nGenerating sample at Stage {final_stage} ({final_stage} recursive passes):")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = model.generate(context, max_new_tokens=200, num_passes=final_stage)
    sample_text = decode(sample[0].tolist())
    print(sample_text)
    with open(f"{results_dir}/stage{final_stage}_sample.txt", 'w') as f:
        f.write(sample_text)

if __name__ == "__main__":
    main()
