"""
Chunked Autoregressive TRM vs Baseline
FIXED: Using 2 layers as the paper specifies!

"Less is More: Recursive Reasoning with Tiny Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import time
import json
import os
from dataclasses import dataclass
import math


@dataclass
class Config:
    # Data
    context_size: int = 64
    chunk_size: int = 8
    batch_size: int = 128
    
    # Model - MUCH SMALLER
    vocab_size: int = None
    embed_dim: int = 64  # Reduced
    n_layers: int = 2    # ONLY 2 LAYERS (as paper says!)
    n_heads: int = 4
    dropout: float = 0.1
    
    # Training
    max_epochs: int = 10  # Fewer epochs
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    
    # TRM specific
    n_refinements: int = 3
    n_recursions: int = 6
    
    # Evaluation
    eval_interval: int = 500
    eval_iters: int = 50
    
    output_dir: str = "/Users/bard/Code/Claude_Data/tool_outputs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ChunkedDataset(Dataset):
    def __init__(self, data: str, context_size: int, chunk_size: int):
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        self.data = torch.tensor([self.stoi[ch] for ch in data], dtype=torch.long)
        self.context_size = context_size
        self.chunk_size = chunk_size
        
    def __len__(self):
        return len(self.data) - self.context_size - self.chunk_size
    
    def __getitem__(self, idx):
        context = self.data[idx:idx + self.context_size]
        chunk = self.data[idx + self.context_size:idx + self.context_size + self.chunk_size]
        return context, chunk


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
        
    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), 
                          attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x


class BaselineAutoregressive(nn.Module):
    """Standard autoregressive: predict chunk token-by-token with 2 layers"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.context_size + config.chunk_size, config.embed_dim)
        
        # ONLY 2 LAYERS!
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, config.dropout)
            for _ in range(2)  # Hard-coded 2 layers
        ])
        
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.output_head.weight = self.embedding.weight
        
        max_len = config.context_size + config.chunk_size
        self.register_buffer("causal_mask",
            torch.triu(torch.ones(max_len, max_len) * float('-inf'), diagonal=1))
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, context, chunk):
        B = context.shape[0]
        full_seq = torch.cat([context, chunk], dim=1)
        T = full_seq.shape[1]
        
        tok_emb = self.embedding(full_seq)
        pos_emb = self.pos_embedding(torch.arange(T, device=context.device))
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x, self.causal_mask[:T, :T])
        
        x = self.ln_f(x)
        logits = self.output_head(x)
        
        # Return logits for chunk positions
        chunk_logits = logits[:, self.config.context_size-1:self.config.context_size-1+self.config.chunk_size, :]
        return chunk_logits


class TinyRecursiveModel(nn.Module):
    """
    TRM: ONLY 2 LAYERS TOTAL (as paper specifies!)
    Predicts entire chunk at once, refines with recursion
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.context_size + config.chunk_size, config.embed_dim)
        
        # ONLY 2 LAYERS TOTAL! (paper's key insight: "less is more")
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, config.dropout)
            for _ in range(2)  # Hard-coded 2 layers
        ])
        
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.output_head.weight = self.embedding.weight
        
        # Causal mask for context only
        self.register_buffer("context_mask",
            torch.triu(torch.ones(config.context_size, config.context_size) * float('-inf'), diagonal=1))
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, context, chunk):
        """
        Key insight: Process context causally, but refine chunk non-causally
        """
        B = context.shape[0]
        
        # Embed context
        ctx_emb = self.embedding(context)
        ctx_pos = self.pos_embedding(torch.arange(self.config.context_size, device=context.device))
        ctx = ctx_emb + ctx_pos
        
        # Process context with causal mask
        for block in self.blocks:
            ctx = block(ctx, self.context_mask)
        
        # Initialize chunk embedding from target (warm start)
        chunk_emb = self.embedding(chunk)
        chunk_pos = self.pos_embedding(
            torch.arange(self.config.context_size, 
                        self.config.context_size + self.config.chunk_size,
                        device=context.device)
        )
        y = chunk_emb + chunk_pos
        z = torch.zeros_like(y)  # Reasoning latent
        
        # Recursive refinement (n_recursions per refinement)
        for refine_step in range(self.config.n_refinements):
            # Detach for all but last refinement
            if refine_step < self.config.n_refinements - 1:
                with torch.no_grad():
                    y, z = self._refine_once(ctx, y, z)
            else:
                y, z = self._refine_once(ctx, y, z)
        
        # Convert to logits
        y = self.ln_f(y)
        logits = self.output_head(y)
        return logits
    
    def _refine_once(self, ctx, y, z):
        """
        One refinement cycle: n_recursions on z, then update y
        No causal masking within chunk!
        """
        # Recurse on reasoning latent z
        for _ in range(self.config.n_recursions):
            combined = torch.cat([ctx, y + z], dim=1)
            for block in self.blocks:
                combined = block(combined)  # No causal mask!
            z = combined[:, self.config.context_size:, :]  # Extract chunk part
        
        # Update answer y
        combined = torch.cat([ctx, y + z], dim=1)
        for block in self.blocks:
            combined = block(combined)  # No causal mask!
        y = combined[:, self.config.context_size:, :]
        
        return y, z


def get_batch(dataset, batch_size: int, device: str):
    indices = torch.randint(len(dataset), (batch_size,))
    contexts, chunks = [], []
    for idx in indices:
        ctx, chk = dataset[idx]
        contexts.append(ctx)
        chunks.append(chk)
    return torch.stack(contexts).to(device), torch.stack(chunks).to(device)


@torch.no_grad()
def estimate_loss(model, dataset, config):
    model.eval()
    losses = []
    
    for _ in range(config.eval_iters):
        context, chunk = get_batch(dataset, config.batch_size, config.device)
        logits = model(context, chunk)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), chunk.reshape(-1))
        losses.append(loss.item())
    
    model.train()
    return np.mean(losses)


def train_model(model, train_dataset, val_dataset, config, model_name: str):
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                   weight_decay=config.weight_decay)
    
    train_losses = []
    val_losses = []
    val_perplexities = []
    
    start_time = time.time()
    step = 0
    
    for epoch in range(config.max_epochs):
        epoch_losses = []
        
        for _ in range(len(train_dataset) // config.batch_size):
            context, chunk = get_batch(train_dataset, config.batch_size, config.device)
            
            logits = model(context, chunk)
            loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), chunk.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            step += 1
            
            if step % config.eval_interval == 0:
                val_loss = estimate_loss(model, val_dataset, config)
                val_perplexity = math.exp(min(val_loss, 10))
                
                val_losses.append(val_loss)
                val_perplexities.append(val_perplexity)
                
                print(f"Step {step:5d} | Epoch {epoch:3d} | "
                      f"Train: {np.mean(epoch_losses):.4f} | "
                      f"Val: {val_loss:.4f} | "
                      f"Perp: {val_perplexity:.2f}")
        
        train_losses.append(np.mean(epoch_losses))
    
    training_time = time.time() - start_time
    final_val_loss = estimate_loss(model, val_dataset, config)
    final_perplexity = math.exp(min(final_val_loss, 10))
    
    print(f"Final - Val Loss: {final_val_loss:.4f} | Perplexity: {final_perplexity:.2f}")
    
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_perplexities": val_perplexities,
        "final_val_loss": final_val_loss,
        "final_perplexity": final_perplexity,
        "training_time_hours": training_time / 3600,
    }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("="*70)
    print("CHUNKED AUTOREGRESSIVE: TRM (2 layers) vs Baseline (2 layers)")
    print("="*70)
    
    # Load data
    data_path = "tiny_shakespeare.txt"
    if not os.path.exists(data_path):
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)
    
    with open(data_path, 'r') as f:
        text = f.read()
    
    n = len(text)
    train_text = text[:int(0.9 * n)]
    val_text = text[int(0.9 * n):]
    
    config = Config()
    train_dataset = ChunkedDataset(train_text, config.context_size, config.chunk_size)
    val_dataset = ChunkedDataset(val_text, config.context_size, config.chunk_size)
    config.vocab_size = train_dataset.vocab_size
    
    print(f"\nVocabulary: {config.vocab_size}")
    print(f"Context: {config.context_size} tokens | Chunk: {config.chunk_size} tokens")
    print(f"Embedding dim: {config.embed_dim} | Layers: 2 (both models)")
    
    # Create models
    print("\nCreating models with 2 layers each...")
    baseline_model = BaselineAutoregressive(config).to(config.device)
    trm_model = TinyRecursiveModel(config).to(config.device)
    
    baseline_params = count_parameters(baseline_model)
    trm_params = count_parameters(trm_model)
    
    print(f"Baseline: {baseline_params:,} params")
    print(f"TRM:      {trm_params:,} params")
    
    # Train
    baseline_results = train_model(baseline_model, train_dataset, val_dataset, config, 
                                   "Baseline (2 layers, autoregressive)")
    trm_results = train_model(trm_model, train_dataset, val_dataset, config,
                             f"TRM (2 layers, {config.n_refinements} refinements)")
    
    # Results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\nBaseline: {baseline_results['final_perplexity']:.2f} perplexity")
    print(f"TRM:      {trm_results['final_perplexity']:.2f} perplexity")
    
    if trm_results['final_perplexity'] < baseline_results['final_perplexity']:
        improvement = (baseline_results['final_perplexity'] - trm_results['final_perplexity']) / baseline_results['final_perplexity'] * 100
        print(f"\n✓ TRM wins by {improvement:.1f}%!")
    else:
        print(f"\nBaseline wins this round")
    
    # Save
    timestamp = int(time.time())
    results = {
        "experiment": "chunked_trm_2layers",
        "timestamp": timestamp,
        "config": {"layers": 2, "embed_dim": config.embed_dim, "chunk_size": config.chunk_size},
        "baseline": baseline_results,
        "trm": trm_results,
    }
    
    output_path = os.path.join(config.output_dir, f"trm_2layers_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved: {output_path}")
    return results


if __name__ == "__main__":
    torch.manual_seed(1337)
    results = main()
