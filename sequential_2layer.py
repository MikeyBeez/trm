"""
Sequential 2-Layer Models Experiment
====================================

Hypothesis: Joint training causes layers to "step on each other" and interfere.
Solution: Train multiple 2-layer models sequentially, each on frozen representations.

Architecture:
- Model 1: 2 layers (input → output)
- Model 2: 2 layers (Model 1's penultimate → output)
- Model 3: 2 layers (Model 2's penultimate → output)
- Model 4: 2 layers (Model 3's penultimate → output)

At inference: 8 effective layers without coordination problems

Comparison:
- 6-layer joint training: ~1.48
- 6-layer + iterative refinement: ~1.12
- 12-layer joint training: ~1.48
- 8-layer sequential (this test): ???
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
n_layer_per_model = 2  # Each model has 2 layers
num_sequential_models = 4  # Train 4 models sequentially = 8 total layers
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

results_dir = './sequential_models_results'
os.makedirs(results_dir, exist_ok=True)

print(f"Using device: {device}")
print(f"Results will be saved to: {results_dir}")
print(f"\n{'='*70}")
print(f"SEQUENTIAL 2-LAYER MODELS EXPERIMENT")
print(f"Training {num_sequential_models} models × {n_layer_per_model} layers = {num_sequential_models * n_layer_per_model} effective layers")
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

class SequentialModel(nn.Module):
    """
    A model with n_layer_per_model transformer layers.
    Can accept either token indices OR pre-computed embeddings as input.
    """
    def __init__(self, is_first_model=False):
        super().__init__()
        self.is_first_model = is_first_model
        
        # Only first model has token/position embeddings
        if is_first_model:
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer_per_model)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, x, targets=None, return_penultimate=False):
        """
        Args:
            x: Either token indices (if first model) or embeddings (if later model)
            targets: Ground truth for loss
            return_penultimate: If True, return penultimate activations for next model
        """
        if self.is_first_model:
            # Embed tokens
            B, T = x.shape
            tok_emb = self.token_embedding_table(x)
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))
            x = tok_emb + pos_emb
        # else: x already contains embeddings from previous model
        
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Store penultimate for next model
        penultimate = x
        
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        if return_penultimate:
            return logits, loss, penultimate
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """Only for first model - normal generation."""
        if not self.is_first_model:
            raise ValueError("Only first model can generate from tokens")
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

@torch.no_grad()
def estimate_loss_single_model(model, is_first=False):
    """Estimate loss for a single model."""
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

@torch.no_grad()
def estimate_loss_cascade(models):
    """Estimate loss for the full cascade of models."""
    out = {}
    for m in models:
        m.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            
            # Forward through cascade
            current_input = X
            for i, model in enumerate(models):
                if i == 0:
                    # First model: tokens → embeddings
                    _, loss, penultimate = model(current_input, Y, return_penultimate=True)
                else:
                    # Later models: embeddings → embeddings
                    _, loss, penultimate = model(current_input, Y, return_penultimate=True)
                
                current_input = penultimate.detach()
            
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    
    for m in models:
        m.train()
    return out

def train_to_convergence(model, optimizer, stage_name, is_first_model=False):
    """Train a single model to convergence."""
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
            losses = estimate_loss_single_model(model, is_first_model)
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
        
        # Training step
        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    model.load_state_dict(torch.load(f"{results_dir}/{stage_name}_best.pth"))
    return history

# ============================================================================
# ACTIVATION COLLECTION
# ============================================================================

class ActivationDataset(Dataset):
    def __init__(self, activations, targets):
        self.activations = activations
        self.targets = targets
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return self.activations[idx], self.targets[idx]

def collect_penultimate_from_model(model, num_batches, is_first_model=False):
    """Collect penultimate activations from a model."""
    print(f"\nCollecting {num_batches} batches of activations...")
    model.eval()
    
    all_activations = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(num_batches):
            xb, yb = get_batch('train')
            _, _, penultimate = model(xb, yb, return_penultimate=True)
            
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
    return activations_tensor, targets_tensor

def collect_cascade_activations(models, num_batches):
    """
    Collect activations by running through cascade of models.
    Returns activations from the last model's penultimate layer.
    """
    print(f"\nCollecting {num_batches} batches through {len(models)}-model cascade...")
    for m in models:
        m.eval()
    
    all_activations = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(num_batches):
            xb, yb = get_batch('train')
            
            # Forward through cascade
            current_input = xb
            for j, model in enumerate(models):
                _, _, penultimate = model(current_input, return_penultimate=True)
                current_input = penultimate
            
            # Save final penultimate
            all_activations.append(penultimate.cpu())
            all_targets.append(yb.cpu())
            
            if (i + 1) % 50 == 0:
                print(f"  Collected {i+1}/{num_batches} batches...")
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
    
    activations_tensor = torch.cat(all_activations, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    
    del all_activations, all_targets
    gc.collect()
    
    print(f"Total samples: {len(activations_tensor)}")
    return activations_tensor, targets_tensor

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    experiment_start = datetime.now()
    print(f"\n{'='*70}")
    print("SEQUENTIAL 2-LAYER MODELS EXPERIMENT")
    print(f"Started: {experiment_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    experiment_log = {
        'start_time': experiment_start.isoformat(),
        'n_layer_per_model': n_layer_per_model,
        'num_models': num_sequential_models,
        'total_effective_layers': num_sequential_models * n_layer_per_model,
        'models': {},
    }
    
    trained_models = []
    
    # ========================================================================
    # MODEL 1: Train on raw tokens
    # ========================================================================
    
    print(f"\n{'='*70}")
    print(f"MODEL 1: Training first {n_layer_per_model}-layer model on tokens")
    print(f"{'='*70}")
    
    model1 = SequentialModel(is_first_model=True).to(device)
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=learning_rate)
    
    history1 = train_to_convergence(model1, optimizer1, 'model1', is_first_model=True)
    experiment_log['models']['model1'] = history1
    trained_models.append(model1)
    
    print(f"\nModel 1 converged at loss: {history1['best_val_loss']:.4f}")
    
    # Generate sample from Model 1
    print("\nGenerating sample from Model 1:")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = model1.generate(context, max_new_tokens=200)
    sample_text = decode(sample[0].tolist())
    print(sample_text[:300])
    with open(f"{results_dir}/model1_sample.txt", 'w') as f:
        f.write(sample_text)
    
    # ========================================================================
    # MODEL 2+: Train on previous model's activations
    # ========================================================================
    
    for model_num in range(2, num_sequential_models + 1):
        print(f"\n{'='*70}")
        print(f"MODEL {model_num}: Training on Model {model_num-1}'s representations")
        print(f"Effective depth so far: {model_num * n_layer_per_model} layers")
        print(f"{'='*70}")
        
        # Collect activations from cascade up to previous model
        if model_num == 2:
            # Just from model 1
            activations, targets = collect_penultimate_from_model(
                trained_models[0], num_activation_batches, is_first_model=True
            )
        else:
            # From cascade of all previous models
            activations, targets = collect_cascade_activations(
                trained_models, num_activation_batches
            )
        
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
        
        # Train new model
        new_model = SequentialModel(is_first_model=False).to(device)
        optimizer = torch.optim.AdamW(new_model.parameters(), lr=learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_losses': [], 'val_losses': [], 'steps': [], 'best_val_loss': None}
        
        iter_num = 0
        max_iter = 10000
        eval_int = 50
        
        while iter_num < max_iter:
            if iter_num % eval_int == 0:
                # Evaluate
                new_model.eval()
                val_losses = []
                for acts, targs in val_loader:
                    acts, targs = acts.to(device), targs.to(device)
                    _, loss = new_model(acts, targs)
                    val_losses.append(loss.item())
                    if len(val_losses) >= eval_iters:
                        break
                val_loss = np.mean(val_losses)
                new_model.train()
                
                print(f"step {iter_num:5d}: val {val_loss:.4f}", end='')
                
                if val_loss < best_val_loss - min_delta:
                    improvement = best_val_loss - val_loss
                    best_val_loss = val_loss
                    patience_counter = 0
                    print(f" ✓ (improved by {improvement:.4f})")
                    torch.save(new_model.state_dict(), f"{results_dir}/model{model_num}_best.pth")
                else:
                    patience_counter += 1
                    print(f" (patience: {patience_counter}/10)")
                    
                    if patience_counter >= 10:
                        print(f"\nConverged at loss: {best_val_loss:.4f}\n")
                        history['best_val_loss'] = best_val_loss
                        break
            
            # Training step
            for acts, targs in train_loader:
                acts, targs = acts.to(device), targs.to(device)
                _, loss = new_model(acts, targs)
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                
                iter_num += 1
                if iter_num % eval_int == 0:
                    break
        
        new_model.load_state_dict(torch.load(f"{results_dir}/model{model_num}_best.pth"))
        experiment_log['models'][f'model{model_num}'] = history
        trained_models.append(new_model)
        
        # Evaluate full cascade
        print(f"\nEvaluating full cascade of {len(trained_models)} models...")
        cascade_loss = estimate_loss_cascade(trained_models)
        print(f"Cascade validation loss: {cascade_loss['val']:.4f}")
        experiment_log['models'][f'model{model_num}']['cascade_val_loss'] = cascade_loss['val']
        
        # Clean up
        del activations, targets, dataset, train_dataset, val_dataset, train_loader, val_loader
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # ========================================================================
    # FINAL EVALUATION
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}\n")
    
    final_cascade_loss = estimate_loss_cascade(trained_models)
    
    print("Results:")
    print(f"  Model 1 alone ({n_layer_per_model} layers): {history1['best_val_loss']:.4f}")
    print(f"  Full cascade ({len(trained_models) * n_layer_per_model} layers): {final_cascade_loss['val']:.4f}")
    print()
    
    print("Comparison to other experiments:")
    print(f"  6-layer baseline: 1.4811")
    print(f"  6-layer + iterative refinement: 1.1193")
    print(f"  12-layer end-to-end: 1.4846")
    print(f"  {len(trained_models) * n_layer_per_model}-layer sequential (this): {final_cascade_loss['val']:.4f}")
    print()
    
    improvement_vs_baseline = 1.4811 - final_cascade_loss['val']
    improvement_pct = (improvement_vs_baseline / 1.4811) * 100
    
    if final_cascade_loss['val'] < 1.1193:
        print(f"✓ BEATS iterative refinement by {1.1193 - final_cascade_loss['val']:.4f}")
        print("  Conclusion: Sequential training is even better!")
    elif final_cascade_loss['val'] < 1.4811:
        print(f"✓ BEATS baseline by {improvement_vs_baseline:.4f} ({improvement_pct:.2f}%)")
        print(f"✗ But loses to iterative refinement by {final_cascade_loss['val'] - 1.1193:.4f}")
        print("  Conclusion: Sequential helps, but not as much as refinement")
    else:
        print(f"✗ WORSE than baseline by {final_cascade_loss['val'] - 1.4811:.4f}")
        print("  Conclusion: Sequential training didn't help")
    
    # Save results
    experiment_end = datetime.now()
    experiment_log['end_time'] = experiment_end.isoformat()
    experiment_log['duration_seconds'] = (experiment_end - experiment_start).total_seconds()
    experiment_log['final_cascade_loss'] = final_cascade_loss['val']
    
    with open(f"{results_dir}/experiment_log.json", 'w') as f:
        json.dump(experiment_log, f, indent=2)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"Duration: {experiment_log['duration_seconds']/60:.1f} minutes")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
