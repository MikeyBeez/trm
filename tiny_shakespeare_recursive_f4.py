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

# Now collect activations after training is complete - save incrementally to disk
print("\nCollecting activations from trained model...")
model.eval()
num_batches = 500

# Save to file incrementally
activations_file = 'activations.npy'
targets_file = 'targets.npy'

# Remove old files if they exist
if os.path.exists(activations_file):
    os.remove(activations_file)
if os.path.exists(targets_file):
    os.remove(targets_file)

with torch.no_grad():
    for i in range(num_batches):
        xb, yb = get_batch('train')
        tok_emb = model.token_embedding_table(xb)
        pos_emb = model.position_embedding_table(torch.arange(xb.shape[1], device=device))
        x = tok_emb + pos_emb
        x = model.blocks(x)
        x = model.ln_f(x)
        
        # Save this batch immediately
        batch_acts = x.detach().cpu().numpy()
        batch_targets = yb.detach().cpu().numpy()
        
        # Append to files
        if i == 0:
            # Create new files
            with open(activations_file, 'wb') as f:
                np.save(f, batch_acts)
            with open(targets_file, 'wb') as f:
                np.save(f, batch_targets)
        else:
            # Append to existing files
            with open(activations_file, 'ab') as f:
                np.save(f, batch_acts)
            with open(targets_file, 'ab') as f:
                np.save(f, batch_targets)
        
        if (i + 1) % 100 == 0:
            print(f"Saved {i+1} batches...")

print(f"\nSaved {num_batches} batches to disk")
print(f"Files: {activations_file}, {targets_file}")

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

# Load activations from disk in chunks
class ActivationDataset(Dataset):
    def __init__(self, acts_file, targets_file):
        # Load all the arrays from the files
        acts_list = []
        targets_list = []
        
        print("Loading activations from disk...")
        with open(acts_file, 'rb') as f:
            try:
                while True:
                    acts_list.append(np.load(f))
            except:
                pass
        
        with open(targets_file, 'rb') as f:
            try:
                while True:
                    targets_list.append(np.load(f))
            except:
                pass
        
        self.activations = torch.from_numpy(np.concatenate(acts_list, axis=0)).float()
        self.targets = torch.from_numpy(np.concatenate(targets_list, axis=0)).long()
        print(f"Loaded {len(self.activations)} samples")
        
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return self.activations[idx], self.targets[idx]

dataset = ActivationDataset('activations.npy', 'targets.npy')
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

# ========== ANALYSIS: Did Stage 2 Learn Anything New? ==========
print("\n" + "="*60)
print("ANALYSIS: Comparing Stage 1 vs Stage 2")
print("="*60 + "\n")

# Load Stage 1 model
stage1_model = GPTLanguageModel().to(device)
stage1_model.load_state_dict(torch.load('stage1_model.pth'))
stage1_model.eval()

# Stage 2 classifier is already loaded
classifier.eval()

# Test on the same validation data
print("Evaluating both models on validation set...")

# Collect validation activations
val_activations = []
val_targets = []
val_batches = 50

with torch.no_grad():
    for _ in range(val_batches):
        xb, yb = get_batch('val')
        
        # Get Stage 1 predictions
        tok_emb = stage1_model.token_embedding_table(xb)
        pos_emb = stage1_model.position_embedding_table(torch.arange(xb.shape[1], device=device))
        x = tok_emb + pos_emb
        x = stage1_model.blocks(x)
        x = stage1_model.ln_f(x)
        
        val_activations.append(x)
        val_targets.append(yb)

val_acts = torch.cat(val_activations, dim=0)
val_targs = torch.cat(val_targets, dim=0)

# Get Stage 1 predictions through original head
with torch.no_grad():
    stage1_logits = stage1_model.lm_head(val_acts)
    B, T, C = stage1_logits.shape
    stage1_loss = F.cross_entropy(stage1_logits.view(B*T, C), val_targs.view(B*T))
    
    stage1_preds = stage1_logits.argmax(dim=-1)
    stage1_acc = (stage1_preds == val_targs).float().mean()

# Get Stage 2 predictions through new classifier
with torch.no_grad():
    stage2_logits, stage2_loss = classifier(val_acts, val_targs)
    stage2_logits = stage2_logits.view(B, T, C)
    
    stage2_preds = stage2_logits.argmax(dim=-1)
    stage2_acc = (stage2_preds == val_targs).float().mean()

print(f"\nStage 1 (original head):")
print(f"  Validation Loss: {stage1_loss.item():.4f}")
print(f"  Accuracy: {stage1_acc.item():.4f}")

print(f"\nStage 2 (retrained classifier):")
print(f"  Validation Loss: {stage2_loss.item():.4f}")
print(f"  Accuracy: {stage2_acc.item():.4f}")

loss_improvement = stage1_loss.item() - stage2_loss.item()
acc_improvement = stage2_acc.item() - stage1_acc.item()

print(f"\nImprovement:")
print(f"  Loss: {loss_improvement:.4f} ({'better' if loss_improvement > 0 else 'worse'})")
print(f"  Accuracy: {acc_improvement:.4f} ({acc_improvement*100:.2f}%)")

# Disagreement analysis
disagreements = (stage1_preds != stage2_preds).float().sum().item()
total_predictions = stage1_preds.numel()
disagreement_rate = disagreements / total_predictions

print(f"\nDisagreement Rate: {disagreement_rate:.4f} ({disagreement_rate*100:.2f}% of predictions differ)")

# Among disagreements, which model is more often correct?
disagreement_mask = (stage1_preds != stage2_preds)
if disagreement_mask.sum() > 0:
    stage1_correct_on_disagreements = ((stage1_preds == val_targs) & disagreement_mask).sum().item()
    stage2_correct_on_disagreements = ((stage2_preds == val_targs) & disagreement_mask).sum().item()
    
    print(f"\nWhen models disagree:")
    print(f"  Stage 1 correct: {stage1_correct_on_disagreements} times")
    print(f"  Stage 2 correct: {stage2_correct_on_disagreements} times")
    
    if stage2_correct_on_disagreements > stage1_correct_on_disagreements:
        print(f"  → Stage 2 makes better predictions on disagreements!")
    elif stage1_correct_on_disagreements > stage2_correct_on_disagreements:
        print(f"  → Stage 1 was already better")
    else:
        print(f"  → Tie")

print("\n" + "="*60)
print("CONCLUSION:")
if loss_improvement > 0.01 or acc_improvement > 0.01:
    print("YES! Stage 2 learned something new from the representations!")
    print("The penultimate layer contained information that the original")
    print("head didn't fully utilize.")
elif abs(loss_improvement) < 0.01:
    print("Stage 2 performed similarly to Stage 1.")
    print("The original head already extracted most available information.")
else:
    print("Stage 2 performed worse - the original head was better.")
print("="*60)

print("\nFiles created:")
print("- activations.npy, targets.npy: Saved penultimate activations and ground truths")
print("- stage1_model.pth: Full trained model from Stage 1")
print("- stage2_classifier.pth: Classifier trained on activations from Stage 2")
