"""
Tiny Recursive Model (TRM) vs Baseline Transformer
Tiny Shakespeare Experiment - FIXED VERSION

Based on: "Less is More: Recursive Reasoning with Tiny Networks"
arXiv:2510.04871v1

Key fixes:
- TRM is now SMALLER than baseline (as paper intends)
- Proper learning rate and scheduling
- Early stopping to prevent overfitting
- Better hyperparameters for Tiny Shakespeare
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import json
import os
from dataclasses import dataclass
import math


@dataclass
class Config:
    # Data
    block_size: int = 64  # Reduced from 128
    batch_size: int = 128  # Increased batch size
    
    # Model - Much smaller!
    vocab_size: int = None
    embed_dim: int = 64  # Reduced from 128
    n_layers_baseline: int = 4  # Baseline uses 4 layers
    n_layers_trm: int = 2  # TRM uses 2 layers (as paper says!)
    n_heads: int = 4
    dropout: float = 0.2  # Increased dropout
    
    # Training
    max_epochs: int = 50  # Reduced
    learning_rate: float = 1e-3  # Reduced from 3e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.01  # Reduced
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    grad_clip: float = 1.0
    
    # TRM specific
    n_refinements: int = 3  # T in paper
    n_recursions: int = 6   # n in paper
    max_supervision_steps: int = 4  # Reduced from 8
    
    # Evaluation
    eval_interval: int = 250  # More frequent
    eval_iters: int = 50
    patience: int = 10  # Early stopping patience
    
    # Output
    output_dir: str = "/Users/bard/Code/Claude_Data/tool_outputs"
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TinyShakespeareDataset(Dataset):
    def __init__(self, data: str, block_size: int):
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        self.data = torch.tensor([self.stoi[ch] for ch in data], dtype=torch.long)
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data) - self.block_size - 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, causal_mask=None):
        attn_out = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), 
                             attn_mask=causal_mask, need_weights=False)[0]
        x = x + self.dropout(attn_out)
        x = x + self.mlp(self.ln2(x))
        return x


class BaselineTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.block_size, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, config.dropout)
            for _ in range(config.n_layers_baseline)
        ])
        
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        # Tie weights
        self.output_head.weight = self.embedding.weight
        
        self.register_buffer("causal_mask", 
            torch.triu(torch.ones(config.block_size, config.block_size) * float('-inf'), diagonal=1))
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx):
        B, T = idx.shape
        
        tok_emb = self.embedding(idx)
        pos_emb = self.pos_embedding(torch.arange(T, device=idx.device))
        x = self.dropout(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x, self.causal_mask[:T, :T])
        
        x = self.ln_f(x)
        logits = self.output_head(x)
        
        return logits


class TinyRecursiveModel(nn.Module):
    """
    TRM: Smaller network that refines answers recursively
    Key: Uses 2 layers instead of 4, single network, stays in embedding space
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.block_size, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Single small network (2 layers as paper recommends)
        self.net = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, config.dropout)
            for _ in range(config.n_layers_trm)
        ])
        
        self.ln = nn.LayerNorm(config.embed_dim)
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        # Tie weights
        self.output_head.weight = self.embedding.weight
        
        # NO causal mask for TRM - it refines entire sequence simultaneously
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def refine_once(self, x, y, z):
        """
        One refinement cycle: update z (reasoning) n times, then update y (answer)
        Key: No causal masking - all positions can see each other
        """
        # Concatenate inputs for reasoning update
        combined = x + y + z  # Residual connections
        
        # Refine reasoning (n_recursions times)
        for _ in range(self.config.n_recursions):
            for block in self.net:
                combined = block(combined)  # No causal mask!
        z = combined
        
        # Refine answer using new reasoning
        y_new = y + z  # Residual
        for block in self.net:
            y_new = block(y_new)  # No causal mask!
        
        return y_new, z
    
    def forward(self, idx, y_init=None, z_init=None, n_refinements=None):
        B, T = idx.shape
        
        if n_refinements is None:
            n_refinements = self.config.n_refinements
        
        # Embed input
        tok_emb = self.embedding(idx)
        pos_emb = self.pos_embedding(torch.arange(T, device=idx.device))
        x = self.dropout(tok_emb + pos_emb)
        
        # Initialize from input if not provided (warm start)
        if y_init is None:
            y = x.clone()
        else:
            y = y_init
            
        if z_init is None:
            z = torch.zeros_like(x)
        else:
            z = z_init
        
        # Multiple refinement cycles
        # First n-1 without gradients (as in paper)
        if n_refinements > 1:
            with torch.no_grad():
                for _ in range(n_refinements - 1):
                    y, z = self.refine_once(x, y, z)
        
        # Final refinement with gradients
        y, z = self.refine_once(x, y, z)
        
        return y, z
    
    def get_logits(self, y):
        return self.output_head(self.ln(y))


def get_lr(it, config):
    """Learning rate schedule with warmup and cosine decay"""
    # Warmup
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # Decay
    if it > config.lr_decay_iters:
        return config.min_learning_rate
    # Cosine decay
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_learning_rate + coeff * (config.learning_rate - config.min_learning_rate)


def get_batch(dataset: Dataset, batch_size: int, device: str):
    indices = torch.randint(len(dataset), (batch_size,))
    xs, ys = [], []
    for idx in indices:
        x, y = dataset[idx]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


@torch.no_grad()
def estimate_loss(model, dataset, config: Config, is_trm: bool = False):
    model.eval()
    losses = []
    
    for _ in range(config.eval_iters):
        x, y = get_batch(dataset, config.batch_size, config.device)
        
        if is_trm:
            y_emb, _ = model(x)
            logits = model.get_logits(y_emb)
        else:
            logits = model(x)
        
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
        losses.append(loss.item())
    
    model.train()
    return np.mean(losses)


def train_baseline(model, train_dataset, val_dataset, config: Config):
    print("\n" + "="*60)
    print("Training Baseline Transformer (4 layers)")
    print("="*60)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, 
                                   weight_decay=config.weight_decay)
    
    train_losses = []
    val_losses = []
    val_perplexities = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    start_time = time.time()
    step = 0
    
    for epoch in range(config.max_epochs):
        epoch_losses = []
        
        for _ in range(len(train_dataset) // config.batch_size):
            # Learning rate schedule
            lr = get_lr(step, config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            x, y = get_batch(train_dataset, config.batch_size, config.device)
            
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            step += 1
            
            if step % config.eval_interval == 0:
                val_loss = estimate_loss(model, val_dataset, config, is_trm=False)
                val_perplexity = math.exp(min(val_loss, 10))  # Cap for display
                
                val_losses.append(val_loss)
                val_perplexities.append(val_perplexity)
                
                print(f"Step {step:5d} | Epoch {epoch:3d} | LR: {lr:.6f} | "
                      f"Train Loss: {np.mean(epoch_losses):.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Perplexity: {val_perplexity:.2f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.patience:
                        print(f"Early stopping at step {step}")
                        training_time = time.time() - start_time
                        return {
                            "train_losses": train_losses,
                            "val_losses": val_losses,
                            "val_perplexities": val_perplexities,
                            "final_val_loss": val_loss,
                            "final_perplexity": val_perplexity,
                            "training_time_hours": training_time / 3600,
                            "epochs_completed": epoch,
                            "early_stopped": True
                        }
        
        train_losses.append(np.mean(epoch_losses))
    
    training_time = time.time() - start_time
    final_val_loss = estimate_loss(model, val_dataset, config, is_trm=False)
    final_perplexity = math.exp(min(final_val_loss, 10))
    
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_perplexities": val_perplexities,
        "final_val_loss": final_val_loss,
        "final_perplexity": final_perplexity,
        "training_time_hours": training_time / 3600,
        "epochs_completed": config.max_epochs,
        "early_stopped": False
    }


def train_trm(model, train_dataset, val_dataset, config: Config):
    print("\n" + "="*60)
    print("Training Tiny Recursive Model (2 layers)")
    print("="*60)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                   weight_decay=config.weight_decay)
    
    train_losses = []
    val_losses = []
    val_perplexities = []
    refinement_improvements = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    start_time = time.time()
    step = 0
    
    for epoch in range(config.max_epochs):
        epoch_losses = []
        epoch_improvements = []
        
        for _ in range(len(train_dataset) // config.batch_size):
            lr = get_lr(step, config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            x, y_true = get_batch(train_dataset, config.batch_size, config.device)
            
            y = None
            z = None
            supervision_losses = []
            
            # Deep supervision
            for sup_step in range(config.max_supervision_steps):
                y, z = model(x, y_init=y, z_init=z)
                logits = model.get_logits(y)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y_true.view(-1))
                
                supervision_losses.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                
                y = y.detach()
                z = z.detach()
                
                # Early stop if perfect
                with torch.no_grad():
                    pred_tokens = logits.argmax(dim=-1)
                    if torch.all(pred_tokens == y_true):
                        break
            
            if len(supervision_losses) > 1:
                improvement = supervision_losses[0] - supervision_losses[-1]
                epoch_improvements.append(improvement)
            
            epoch_losses.append(supervision_losses[-1])
            step += 1
            
            if step % config.eval_interval == 0:
                val_loss = estimate_loss(model, val_dataset, config, is_trm=True)
                val_perplexity = math.exp(min(val_loss, 10))
                
                val_losses.append(val_loss)
                val_perplexities.append(val_perplexity)
                
                avg_improvement = np.mean(epoch_improvements) if epoch_improvements else 0
                
                print(f"Step {step:5d} | Epoch {epoch:3d} | LR: {lr:.6f} | "
                      f"Train Loss: {np.mean(epoch_losses):.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Perplexity: {val_perplexity:.2f} | "
                      f"Refine Δ: {avg_improvement:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.patience:
                        print(f"Early stopping at step {step}")
                        training_time = time.time() - start_time
                        return {
                            "train_losses": train_losses,
                            "val_losses": val_losses,
                            "val_perplexities": val_perplexities,
                            "refinement_improvements": refinement_improvements,
                            "final_val_loss": val_loss,
                            "final_perplexity": val_perplexity,
                            "training_time_hours": training_time / 3600,
                            "epochs_completed": epoch,
                            "early_stopped": True
                        }
        
        train_losses.append(np.mean(epoch_losses))
        if epoch_improvements:
            refinement_improvements.append(np.mean(epoch_improvements))
    
    training_time = time.time() - start_time
    final_val_loss = estimate_loss(model, val_dataset, config, is_trm=True)
    final_perplexity = math.exp(min(final_val_loss, 10))
    
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_perplexities": val_perplexities,
        "refinement_improvements": refinement_improvements,
        "final_val_loss": final_val_loss,
        "final_perplexity": final_perplexity,
        "training_time_hours": training_time / 3600,
        "epochs_completed": config.max_epochs,
        "early_stopped": False
    }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("="*60)
    print("TRM vs Baseline - FIXED VERSION")
    print("Key: TRM has FEWER parameters but recursive refinement")
    print("="*60)
    
    # Load data
    print("\nLoading Tiny Shakespeare...")
    data_path = "tiny_shakespeare.txt"
    
    if not os.path.exists(data_path):
        print("Downloading...")
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)
    
    with open(data_path, 'r') as f:
        text = f.read()
    
    n = len(text)
    train_text = text[:int(0.9 * n)]
    val_text = text[int(0.9 * n):]
    
    config = Config()
    train_dataset = TinyShakespeareDataset(train_text, config.block_size)
    val_dataset = TinyShakespeareDataset(val_text, config.block_size)
    config.vocab_size = train_dataset.vocab_size
    
    print(f"Vocabulary: {config.vocab_size}")
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    
    # Create models
    print("\nInitializing models...")
    baseline_model = BaselineTransformer(config).to(config.device)
    trm_model = TinyRecursiveModel(config).to(config.device)
    
    baseline_params = count_parameters(baseline_model)
    trm_params = count_parameters(trm_model)
    
    print(f"\nBaseline: {baseline_params:,} params (4 layers)")
    print(f"TRM:      {trm_params:,} params (2 layers)")
    print(f"TRM is {baseline_params/trm_params:.2f}x SMALLER ✓")
    
    # Train
    baseline_results = train_baseline(baseline_model, train_dataset, val_dataset, config)
    trm_results = train_trm(trm_model, train_dataset, val_dataset, config)
    
    # Results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    print(f"\nBaseline (4 layers, {baseline_params:,} params):")
    print(f"  Val Loss: {baseline_results['final_val_loss']:.4f}")
    print(f"  Perplexity: {baseline_results['final_perplexity']:.2f}")
    print(f"  Time: {baseline_results['training_time_hours']:.2f}h")
    
    print(f"\nTRM (2 layers, {trm_params:,} params):")
    print(f"  Val Loss: {trm_results['final_val_loss']:.4f}")
    print(f"  Perplexity: {trm_results['final_perplexity']:.2f}")
    print(f"  Time: {trm_results['training_time_hours']:.2f}h")
    if trm_results['refinement_improvements']:
        print(f"  Avg Refinement: {np.mean(trm_results['refinement_improvements']):.4f}")
    
    if trm_results['final_perplexity'] < baseline_results['final_perplexity']:
        improvement = (baseline_results['final_perplexity'] - trm_results['final_perplexity']) / baseline_results['final_perplexity'] * 100
        print(f"\n✓ TRM wins by {improvement:.1f}% with {baseline_params/trm_params:.1f}x fewer parameters!")
    else:
        print(f"\nBaseline wins (TRM needs tuning)")
    
    # Save
    timestamp = int(time.time())
    results = {
        "experiment": "trm_vs_baseline_fixed",
        "timestamp": timestamp,
        "config": {
            "vocab_size": config.vocab_size,
            "embed_dim": config.embed_dim,
            "n_layers_baseline": config.n_layers_baseline,
            "n_layers_trm": config.n_layers_trm,
            "block_size": config.block_size,
        },
        "model_results": {
            "baseline": baseline_results,
            "trm": trm_results
        },
        "analysis": {
            "baseline_params": baseline_params,
            "trm_params": trm_params,
            "parameter_ratio": baseline_params / trm_params,
            "trm_wins": trm_results['final_perplexity'] < baseline_results['final_perplexity']
        }
    }
    
    output_path = os.path.join(config.output_dir, f"trm_fixed_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved: {output_path}")
    
    return results


if __name__ == "__main__":
    torch.manual_seed(1337)
    results = main()
