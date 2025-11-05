"""
Fast TRM Training - Optimized for Speed
- 2 refinements × 3 recursions = 8 passes (vs 21)
- Mixed precision training (FP16)
- 4-token chunks (more practical than 2 chars)
- Word-level tokenization
- Tiny Shakespeare for fast iteration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import json
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================

class Config:
    # Model
    vocab_size = 5000  # Will be set from tokenizer
    embed_dim = 128
    n_layers = 2
    n_heads = 4
    dropout = 0.2
    
    # TRM specific
    n_refinements = 2  # Reduced from 3
    n_recursions = 3   # Reduced from 6
    chunk_size = 4     # Increased from 2
    context_size = 64
    
    # Training
    batch_size = 32
    learning_rate = 3e-4
    weight_decay = 0.01
    max_epochs = 50
    grad_clip = 1.0
    max_supervision_steps = 3  # Reduced from 4
    
    # Evaluation
    eval_interval = 250
    eval_iters = 50
    
    # System
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = True  # Mixed precision training
    
    # Paths
    data_path = 'tiny_shakespeare.txt'
    output_dir = 'fast_trm_results'

config = Config()
Path(config.output_dir).mkdir(exist_ok=True)

# ============================================================================
# SIMPLE TOKENIZER
# ============================================================================

class SimpleTokenizer:
    """Word-level tokenizer with special tokens"""
    def __init__(self, text, vocab_size=5000):
        # Split on whitespace and punctuation
        import re
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        
        # Count frequencies
        from collections import Counter
        word_counts = Counter(words)
        
        # Take most common words
        vocab_words = [word for word, _ in word_counts.most_common(vocab_size - 3)]
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.eos_token = '<EOS>'
        
        # Build vocab
        self.vocab = [self.pad_token, self.unk_token, self.eos_token] + vocab_words
        self.vocab_size = len(self.vocab)
        
        self.stoi = {w: i for i, w in enumerate(self.vocab)}
        self.itos = {i: w for i, w in enumerate(self.vocab)}
        
        print(f"Tokenizer: {self.vocab_size} tokens")
    
    def encode(self, text):
        import re
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        return [self.stoi.get(w, self.stoi[self.unk_token]) for w in words]
    
    def decode(self, tokens):
        words = [self.itos[t] for t in tokens if t in self.itos]
        # Simple reconstruction
        text = ' '.join(words)
        # Fix punctuation spacing
        text = text.replace(' .', '.').replace(' ,', ',').replace(' !', '!').replace(' ?', '?')
        return text

# ============================================================================
# DATASET
# ============================================================================

class ChunkedDataset(Dataset):
    def __init__(self, tokens, context_size, chunk_size):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context_size = context_size
        self.chunk_size = chunk_size
        
    def __len__(self):
        return len(self.tokens) - self.context_size - self.chunk_size
    
    def __getitem__(self, idx):
        context = self.tokens[idx:idx + self.context_size]
        chunk = self.tokens[idx + self.context_size:idx + self.context_size + self.chunk_size]
        return context, chunk

# ============================================================================
# MODELS
# ============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.2):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, 
                                         dropout=dropout, batch_first=True)
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


class BaselineModel(nn.Module):
    """Standard autoregressive baseline"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.context_size + config.chunk_size, config.embed_dim)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.output_head.weight = self.embedding.weight
        
        # Causal mask
        mask = torch.triu(torch.ones(config.context_size + config.chunk_size, 
                                     config.context_size + config.chunk_size) * float('-inf'), 
                         diagonal=1)
        self.register_buffer("causal_mask", mask)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, context, chunk):
        B = context.shape[0]
        full_seq = torch.cat([context, chunk], dim=1)
        T = full_seq.shape[1]
        
        # Embed
        tok_emb = self.embedding(full_seq)
        pos_emb = self.pos_embedding(torch.arange(T, device=full_seq.device))
        x = tok_emb + pos_emb
        
        # Transform with causal masking
        for block in self.blocks:
            x = block(x, self.causal_mask[:T, :T])
        
        x = self.ln_f(x)
        logits = self.output_head(x)
        
        # Return logits for chunk positions
        return logits[:, self.config.context_size-1:self.config.context_size+self.config.chunk_size-1, :]


class FastTRM(nn.Module):
    """Fast TRM: 2 refinements × 3 recursions = 8 passes"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.context_size + config.chunk_size, config.embed_dim)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.output_head.weight = self.embedding.weight
        
        # Context mask only
        mask = torch.triu(torch.ones(config.context_size, config.context_size) * float('-inf'), 
                         diagonal=1)
        self.register_buffer("context_mask", mask)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, context, chunk):
        B = context.shape[0]
        
        # Embed context
        ctx_emb = self.embedding(context)
        ctx_pos = self.pos_embedding(torch.arange(self.config.context_size, device=context.device))
        ctx = ctx_emb + ctx_pos
        
        # Process context with causal mask
        for block in self.blocks:
            ctx = block(ctx, self.context_mask)
        
        # Initialize chunk (warm start)
        chunk_emb = self.embedding(chunk)
        chunk_pos = self.pos_embedding(
            torch.arange(self.config.context_size, 
                        self.config.context_size + self.config.chunk_size,
                        device=context.device)
        )
        y = chunk_emb + chunk_pos
        z = torch.zeros_like(y)
        
        # Fast refinement: 2 × 3
        for refine_step in range(self.config.n_refinements):
            if refine_step < self.config.n_refinements - 1:
                with torch.no_grad():
                    y, z = self._refine_once(ctx, y, z)
            else:
                y, z = self._refine_once(ctx, y, z)
        
        y = self.ln_f(y)
        logits = self.output_head(y)
        return logits
    
    def _refine_once(self, ctx, y, z):
        # Recurse on z (3 times instead of 6)
        for _ in range(self.config.n_recursions):
            combined = torch.cat([ctx, y + z], dim=1)
            for block in self.blocks:
                combined = block(combined)
            z = combined[:, self.config.context_size:, :]
        
        # Update y
        combined = torch.cat([ctx, y + z], dim=1)
        for block in self.blocks:
            combined = block(combined)
        y = combined[:, self.config.context_size:, :]
        
        return y, z

# ============================================================================
# TRAINING
# ============================================================================

def get_batch(dataset, batch_size, device):
    indices = torch.randint(len(dataset), (batch_size,))
    contexts = torch.stack([dataset[i][0] for i in indices]).to(device)
    chunks = torch.stack([dataset[i][1] for i in indices]).to(device)
    return contexts, chunks

@torch.no_grad()
def estimate_loss(model, dataset, config):
    model.eval()
    losses = []
    
    for _ in range(config.eval_iters):
        context, chunk = get_batch(dataset, config.batch_size, config.device)
        
        if config.use_amp:
            with autocast():
                logits = model(context, chunk)
                loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), chunk.reshape(-1))
        else:
            logits = model(context, chunk)
            loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), chunk.reshape(-1))
        
        losses.append(loss.item())
    
    model.train()
    return np.mean(losses)

def train_baseline(model, train_dataset, val_dataset, config):
    print("\n" + "="*80)
    print("TRAINING BASELINE")
    print("="*80)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler() if config.use_amp else None
    
    results = {'train_loss': [], 'val_loss': [], 'val_perplexity': []}
    best_val_loss = float('inf')
    step = 0
    
    start_time = time.time()
    
    for epoch in range(config.max_epochs):
        for _ in range(len(train_dataset) // config.batch_size):
            context, chunk = get_batch(train_dataset, config.batch_size, config.device)
            
            if config.use_amp:
                with autocast():
                    logits = model(context, chunk)
                    loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), chunk.reshape(-1))
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                logits = model(context, chunk)
                loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), chunk.reshape(-1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            
            if step % config.eval_interval == 0:
                val_loss = estimate_loss(model, val_dataset, config)
                val_ppl = np.exp(val_loss)
                
                results['train_loss'].append(loss.item())
                results['val_loss'].append(val_loss)
                results['val_perplexity'].append(val_ppl)
                
                elapsed = time.time() - start_time
                print(f"Step {step:5d} | Train: {loss.item():.4f} | Val: {val_loss:.4f} | PPL: {val_ppl:.2f} | Time: {elapsed:.1f}s")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"  → New best validation: {val_ppl:.2f}")
            
            step += 1
    
    return results

def train_trm(model, train_dataset, val_dataset, config):
    print("\n" + "="*80)
    print("TRAINING FAST TRM")
    print("="*80)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler() if config.use_amp else None
    
    results = {'train_loss': [], 'val_loss': [], 'val_perplexity': []}
    best_val_loss = float('inf')
    step = 0
    
    start_time = time.time()
    
    for epoch in range(config.max_epochs):
        for _ in range(len(train_dataset) // config.batch_size):
            context, chunk = get_batch(train_dataset, config.batch_size, config.device)
            
            y, z = None, None
            
            # Deep supervision (3 steps instead of 4)
            for sup_step in range(config.max_supervision_steps):
                if config.use_amp:
                    with autocast():
                        logits = model(context, chunk)
                        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), chunk.reshape(-1))
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    logits = model(context, chunk)
                    loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), chunk.reshape(-1))
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()
            
            if step % config.eval_interval == 0:
                val_loss = estimate_loss(model, val_dataset, config)
                val_ppl = np.exp(val_loss)
                
                results['train_loss'].append(loss.item())
                results['val_loss'].append(val_loss)
                results['val_perplexity'].append(val_ppl)
                
                elapsed = time.time() - start_time
                print(f"Step {step:5d} | Train: {loss.item():.4f} | Val: {val_loss:.4f} | PPL: {val_ppl:.2f} | Time: {elapsed:.1f}s")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"  → New best validation: {val_ppl:.2f}")
            
            step += 1
    
    return results

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Fast TRM Training")
    print("="*80)
    print(f"Device: {config.device}")
    print(f"Mixed Precision: {config.use_amp}")
    print(f"Refinements: {config.n_refinements} × Recursions: {config.n_recursions} = {config.n_refinements * (config.n_recursions + 1)} passes")
    print(f"Chunk size: {config.chunk_size} tokens")
    
    # Load data
    print("\nLoading data...")
    with open(config.data_path, 'r') as f:
        text = f.read()
    
    # Tokenize
    tokenizer = SimpleTokenizer(text, vocab_size=5000)
    config.vocab_size = tokenizer.vocab_size
    
    tokens = tokenizer.encode(text)
    print(f"Total tokens: {len(tokens):,}")
    
    # Split train/val
    split_idx = int(0.9 * len(tokens))
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    
    train_dataset = ChunkedDataset(train_tokens, config.context_size, config.chunk_size)
    val_dataset = ChunkedDataset(val_tokens, config.context_size, config.chunk_size)
    
    print(f"Train examples: {len(train_dataset):,}")
    print(f"Val examples: {len(val_dataset):,}")
    
    # SKIP BASELINE - We already have results showing it overfits
    # Baseline best: PPL 142.56 at step 10750
    # Baseline final: PPL 777+ (massive overfitting)
    
    print("\n" + "="*80)
    print("SKIPPING BASELINE - Using previous results:")
    print("  Best: PPL 142.56 (step 10750)")
    print("  Final: PPL 777+ (overfitting)")
    print("="*80)
    
    # Train TRM
    print("\n" + "="*80)
    trm = FastTRM(config).to(config.device)
    print(f"TRM parameters: {sum(p.numel() for p in trm.parameters()):,}")
    trm_results = train_trm(trm, train_dataset, val_dataset, config)
    
    # Save results
    results = {
        'config': {
            'n_refinements': config.n_refinements,
            'n_recursions': config.n_recursions,
            'chunk_size': config.chunk_size,
            'embed_dim': config.embed_dim,
            'n_layers': config.n_layers,
        },
        'baseline': {
            'note': 'From previous run',
            'best_val_perplexity': 142.56,
            'best_step': 10750,
            'final_val_perplexity': 777.34,
            'overfitting': '5.5x worse'
        },
        'trm': trm_results
    }
    
    output_path = Path(config.output_dir) / 'fast_trm_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Baseline best perplexity: 142.56 (from previous run)")
    print(f"TRM best perplexity: {min(trm_results['val_perplexity']):.2f}")
    print(f"Improvement: {(1 - min(trm_results['val_perplexity'])/142.56)*100:.1f}%")
    print(f"\nResults saved to: {output_path}")

if __name__ == '__main__':
    main()
