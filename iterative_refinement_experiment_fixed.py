"""
Iterative Refinement Training Experiment - Memory Optimized Version
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
# HYPERPARAMETERS - MEMORY OPTIMIZED
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

max_refinement_stages = 10
refinement_patience = 10
refinement_eval_interval = 50

# REDUCED for memory efficiency
num_activation_batches = 300  # Reduced from 1000

results_dir = './iterative_refinement_results'
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

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def get_penultimate_activations(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        return x
    
    def forward(self, idx, targets=None):
        x = self.get_penultimate_activations(idx)
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

class ActivationClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(n_embd, vocab_size)
    
    def forward(self, x, targets=None):
        B, T, C = x.shape
        logits = self.classifier(x)
        
        if targets is None:
            loss = None
        else:
            logits_flat = logits.view(B*T, vocab_size)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

@torch.no_grad()
def estimate_loss(model, stage='stage1'):
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

def train_to_convergence(model, optimizer, stage_name, patience_limit, eval_int, max_iterations=max_iters):
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
        'final_train_loss': None,
    }
    
    for iter_num in range(max_iterations):
        if iter_num % eval_int == 0 or iter_num == max_iterations - 1:
            losses = estimate_loss(model, stage_name)
            history['steps'].append(iter_num)
            history['train_losses'].append(losses['train'])
            history['val_losses'].append(losses['val'])
            
            print(f"step {iter_num:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", end='')
            
            if losses['val'] < best_val_loss - min_delta:
                improvement = best_val_loss - losses['val']
                best_val_loss = losses['val']
                patience_counter = 0
                print(f" ✓ (improved by {improvement:.4f})")
                torch.save(model.state_dict(), f"{results_dir}/{stage_name}_best.pth")
            else:
                patience_counter += 1
                print(f" (patience: {patience_counter}/{patience_limit})")
                
                if patience_counter >= patience_limit:
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
    
    model.load_state_dict(torch.load(f"{results_dir}/{stage_name}_best.pth"))
    return history

# ============================================================================
# ACTIVATION DATASET - MEMORY EFFICIENT
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
    print(f"\nCollecting {num_batches} batches of activations...")
    model.eval()
    
    all_activations = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(num_batches):
            xb, yb = get_batch('train')
            activations = model.get_penultimate_activations(xb)
            
            # Keep on CPU to save GPU memory
            all_activations.append(activations.cpu())
            all_targets.append(yb.cpu())
            
            if (i + 1) % 50 == 0:
                print(f"  Collected {i+1}/{num_batches} batches...")
                # Force garbage collection
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
    
    # Concatenate all at once
    print("Concatenating activations...")
    activations_tensor = torch.cat(all_activations, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    
    # Clear intermediate lists
    del all_activations, all_targets
    gc.collect()
    
    print(f"Total samples: {len(activations_tensor)}")
    print(f"Activation shape: {activations_tensor.shape}")
    
    return activations_tensor, targets_tensor

@torch.no_grad()
def estimate_loss_on_activations(classifier, data_loader, stage_name):
    classifier.eval()
    
    losses = []
    for acts, targets in data_loader:
        acts, targets = acts.to(device), targets.to(device)
        _, loss = classifier(acts, targets)
        losses.append(loss.item())
        
        if len(losses) >= eval_iters:
            break
    
    classifier.train()
    return np.mean(losses)

def train_classifier_to_convergence(classifier, train_loader, val_loader, stage_name, patience_limit):
    print(f"\n{'='*70}")
    print(f"Training {stage_name}")
    print(f"{'='*70}\n")
    
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)
    
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
    
    iter_num = 0
    max_iterations = 10000
    
    while iter_num < max_iterations:
        if iter_num % refinement_eval_interval == 0:
            train_loss = estimate_loss_on_activations(classifier, train_loader, stage_name)
            val_loss = estimate_loss_on_activations(classifier, val_loader, stage_name)
            
            history['steps'].append(iter_num)
            history['train_losses'].append(train_loss)
            history['val_losses'].append(val_loss)
            
            print(f"step {iter_num:5d}: train loss {train_loss:.4f}, val loss {val_loss:.4f}", end='')
            
            if val_loss < best_val_loss - min_delta:
                improvement = best_val_loss - val_loss
                best_val_loss = val_loss
                patience_counter = 0
                print(f" ✓ (improved by {improvement:.4f})")
                torch.save(classifier.state_dict(), f"{results_dir}/{stage_name}_best.pth")
            else:
                patience_counter += 1
                print(f" (patience: {patience_counter}/{patience_limit})")
                
                if patience_counter >= patience_limit:
                    print(f"\n{'='*70}")
                    print(f"Converged at step {iter_num}")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    print(f"{'='*70}\n")
                    history['converged_at_step'] = iter_num
                    history['best_val_loss'] = best_val_loss
                    history['final_train_loss'] = train_loss
                    break
        
        for acts, targets in train_loader:
            acts, targets = acts.to(device), targets.to(device)
            _, loss = classifier(acts, targets)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            iter_num += 1
            if iter_num % refinement_eval_interval == 0:
                break
    
    classifier.load_state_dict(torch.load(f"{results_dir}/{stage_name}_best.pth"))
    return history

# ============================================================================
# ANALYSIS UTILITIES
# ============================================================================

@torch.no_grad()
def compare_stages(base_model, classifiers, val_batches=50):  # Reduced from 100
    print(f"\n{'='*70}")
    print("COMPREHENSIVE STAGE COMPARISON")
    print(f"{'='*70}\n")
    
    base_model.eval()
    for _, clf in classifiers:
        clf.eval()
    
    print(f"Collecting {val_batches} validation batches...")
    val_activations = []
    val_targets = []
    
    with torch.no_grad():
        for _ in range(val_batches):
            xb, yb = get_batch('val')
            activations = base_model.get_penultimate_activations(xb)
            val_activations.append(activations)
            val_targets.append(yb)
    
    val_acts = torch.cat(val_activations, dim=0)
    val_targs = torch.cat(val_targets, dim=0)
    
    print(f"Validation samples: {val_acts.shape[0]}")
    print(f"Total predictions: {val_acts.shape[0] * val_acts.shape[1]}")
    
    results = {}
    all_predictions = []
    
    for stage_name, classifier in classifiers:
        with torch.no_grad():
            logits, loss = classifier(val_acts, val_targs)
            B, T, C = logits.shape
            
            preds = logits.argmax(dim=-1)
            acc = (preds == val_targs).float().mean().item()
            
            results[stage_name] = {
                'loss': loss.item(),
                'accuracy': acc,
                'predictions': preds,
            }
            all_predictions.append((stage_name, preds))
            
            print(f"\n{stage_name}:")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    print(f"\n{'='*70}")
    print("PAIRWISE DISAGREEMENT ANALYSIS")
    print(f"{'='*70}\n")
    
    for i in range(len(all_predictions) - 1):
        name1, preds1 = all_predictions[i]
        name2, preds2 = all_predictions[i + 1]
        
        disagreements = (preds1 != preds2)
        disagreement_count = disagreements.sum().item()
        total_preds = preds1.numel()
        disagreement_rate = disagreement_count / total_preds
        
        print(f"\n{name1} vs {name2}:")
        print(f"  Disagreements: {disagreement_count:,} / {total_preds:,} ({disagreement_rate*100:.2f}%)")
        
        if disagreement_count > 0:
            stage1_correct = ((preds1 == val_targs) & disagreements).sum().item()
            stage2_correct = ((preds2 == val_targs) & disagreements).sum().item()
            
            print(f"  {name1} correct on disagreements: {stage1_correct:,}")
            print(f"  {name2} correct on disagreements: {stage2_correct:,}")
            
            if stage2_correct > stage1_correct:
                advantage = stage2_correct - stage1_correct
                print(f"  → {name2} is better by {advantage:,} predictions ({advantage/disagreement_count*100:.1f}%)")
            elif stage1_correct > stage2_correct:
                advantage = stage1_correct - stage2_correct
                print(f"  → {name1} is better by {advantage:,} predictions ({advantage/disagreement_count*100:.1f}%)")
            else:
                print(f"  → Tie")
    
    print(f"\n{'='*70}")
    print("OVERALL IMPROVEMENT SUMMARY")
    print(f"{'='*70}\n")
    
    baseline_loss = results[classifiers[0][0]]['loss']
    baseline_acc = results[classifiers[0][0]]['accuracy']
    
    for stage_name, _ in classifiers[1:]:
        loss_improvement = baseline_loss - results[stage_name]['loss']
        acc_improvement = results[stage_name]['accuracy'] - baseline_acc
        loss_pct = (loss_improvement / baseline_loss) * 100
        
        print(f"\n{stage_name} vs {classifiers[0][0]}:")
        print(f"  Loss improvement: {loss_improvement:.4f} ({loss_pct:.2f}%)")
        print(f"  Accuracy improvement: {acc_improvement:.4f} ({acc_improvement*100:.2f}%)")
    
    return results

def generate_sample_text(model, context_str="", max_tokens=300):  # Reduced tokens
    model.eval()
    
    if context_str:
        context = torch.tensor([encode(context_str)], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    generated = model.generate(context, max_new_tokens=max_tokens)
    return decode(generated[0].tolist())

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    experiment_start_time = datetime.now()
    print(f"\n{'='*70}")
    print("ITERATIVE REFINEMENT EXPERIMENT")
    print(f"Started at: {experiment_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    experiment_log = {
        'start_time': experiment_start_time.isoformat(),
        'hyperparameters': {
            'batch_size': batch_size,
            'block_size': block_size,
            'n_embd': n_embd,
            'n_head': n_head,
            'n_layer': n_layer,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'patience': patience,
            'min_delta': min_delta,
            'num_activation_batches': num_activation_batches,
        },
        'stages': {},
    }
    
    # Stage 1
    print("\n" + "="*70)
    print("STAGE 1: Training base model to convergence")
    print("="*70 + "\n")
    
    base_model = GPTLanguageModel().to(device)
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=learning_rate)
    
    stage1_history = train_to_convergence(base_model, optimizer, 'stage1', patience, eval_interval)
    experiment_log['stages']['stage1'] = stage1_history
    
    torch.save(base_model.state_dict(), f"{results_dir}/stage1_final.pth")
    
    print("\nSample generation from Stage 1:")
    sample = generate_sample_text(base_model, max_tokens=200)
    print(sample)
    print("\n")
    
    with open(f"{results_dir}/stage1_sample.txt", 'w') as f:
        f.write(sample)
    
    # Collect activations
    print("\n" + "="*70)
    print("COLLECTING ACTIVATIONS FROM CONVERGED MODEL")
    print("="*70 + "\n")
    
    activations, targets = collect_activations(base_model, num_activation_batches)
    
    torch.save({
        'activations': activations,
        'targets': targets,
    }, f"{results_dir}/activations.pt")
    
    print(f"Activations saved")
    
    # Create datasets
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
    
    # Iterative refinement
    all_classifiers = [('stage1', base_model.lm_head)]
    previous_best_loss = stage1_history['best_val_loss']
    
    for stage_num in range(2, max_refinement_stages + 2):
        stage_name = f'stage{stage_num}'
        
        print("\n" + "="*70)
        print(f"STAGE {stage_num}: Iterative refinement (iteration {stage_num - 1})")
        print("="*70 + "\n")
        
        classifier = ActivationClassifier().to(device)
        
        stage_history = train_classifier_to_convergence(
            classifier, train_loader, val_loader, stage_name, refinement_patience
        )
        
        experiment_log['stages'][stage_name] = stage_history
        torch.save(classifier.state_dict(), f"{results_dir}/{stage_name}_final.pth")
        all_classifiers.append((stage_name, classifier))
        
        current_best_loss = stage_history['best_val_loss']
        improvement = previous_best_loss - current_best_loss
        improvement_pct = (improvement / previous_best_loss) * 100
        
        print(f"\nImprovement over previous stage:")
        print(f"  Previous best: {previous_best_loss:.4f}")
        print(f"  Current best: {current_best_loss:.4f}")
        print(f"  Improvement: {improvement:.4f} ({improvement_pct:.2f}%)")
        
        if improvement < min_delta:
            print(f"\n{'='*70}")
            print(f"CONVERGENCE REACHED: No significant improvement at {stage_name}")
            print(f"{'='*70}\n")
            experiment_log['final_stage'] = stage_name
            experiment_log['convergence_reason'] = 'no_improvement'
            break
        
        previous_best_loss = current_best_loss
        
        if stage_num >= max_refinement_stages + 1:
            print(f"\n{'='*70}")
            print(f"Reached maximum refinement stages")
            print(f"{'='*70}\n")
            experiment_log['final_stage'] = stage_name
            experiment_log['convergence_reason'] = 'max_stages_reached'
            break
    
    # Final comparison
    comparison_results = compare_stages(base_model, all_classifiers, val_batches=50)
    experiment_log['final_comparison'] = {
        stage: {'loss': results['loss'], 'accuracy': results['accuracy']}
        for stage, results in comparison_results.items()
    }
    
    # Save log
    experiment_end_time = datetime.now()
    experiment_log['end_time'] = experiment_end_time.isoformat()
    experiment_log['duration_seconds'] = (experiment_end_time - experiment_start_time).total_seconds()
    
    with open(f"{results_dir}/experiment_log.json", 'w') as f:
        json.dump(experiment_log, f, indent=2)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"Duration: {experiment_log['duration_seconds'] / 60:.1f} minutes")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
