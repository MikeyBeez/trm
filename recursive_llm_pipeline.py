#!/usr/bin/env python3
"""
recursive_lm_pipeline.py

Full pipeline:
 - Download / tokenize WikiText-103
 - Train a base causal LM (GPT-like configurable)
 - Save base checkpoint
 - Train a second-stage Refiner that maps base penultimate hidden states -> logits
   (computes hidden states on-the-fly using frozen base)
 - Evaluate and optionally run recursive inference (k refinement passes)

Usage example (quick):
 python recursive_lm_pipeline.py --stage all --model_size small --device cuda --epochs_base 1 --epochs_refine 1

Adjust hyperparameters as needed.
"""

import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from datasets import load_dataset
from tqdm import tqdm


# -----------------------
# Utilities
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def exists(x):
    return x is not None


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


# -----------------------
# Dataset / Tokenization
# -----------------------
class TextDataset(torch.utils.data.Dataset):
    """
    Simple contiguous chunk dataset: concatenates tokenized dataset and chunks into sequences.
    """
    def __init__(self, ids: torch.LongTensor, seq_len: int):
        self.ids = ids
        self.seq_len = seq_len
        if len(self.ids) < seq_len:
            raise ValueError("Token stream shorter than seq_len")
        # number of full chunks (non-overlapping)
        self.num_chunks = len(self.ids) // seq_len

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.ids[start : start + self.seq_len]
        input_ids = torch.from_numpy(chunk).long()
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


def build_contiguous_dataset(tokenizer, dataset_name="wikitext", subset="wikitext-103-v1",
                             split="train", seq_len=256, max_samples: Optional[int] = None):
    """
    Loads dataset via datasets.load_dataset and produces a contiguous token stream.
    Returns Torch LongTensor ids.
    """
    ds = load_dataset(dataset_name, subset, split=split)
    texts = []
    for ex in tqdm(ds, desc=f"loading {split}"):
        # data format differs slightly; prefer 'text' key
        if "text" in ex:
            texts.append(ex["text"])
        elif "article" in ex:
            texts.append(ex["article"])
        else:
            # fallback: join all values
            texts.append(" ".join([str(v) for v in ex.values()]))

    # optionally trim to a subset of examples for quick experiments
    if max_samples:
        texts = texts[:max_samples]

    # join into one big text (note: spaces preserved)
    big_text = "\n\n".join(texts)
    # tokenize
    enc = tokenizer(big_text, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
    # truncate to integer multiples of seq_len
    total_len = (enc.size(0) // seq_len) * seq_len
    enc = enc[:total_len]
    return enc.numpy()


# -----------------------
# Model builder
# -----------------------
def make_gpt_config(model_size: str, vocab_size: int):
    if model_size == "small":
        # ~30-50M params
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=1024,
            n_ctx=1024,
            n_embd=512,
            n_layer=6,
            n_head=8,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
        )
    elif model_size == "medium":
        # ~110-150M params (GPT-2 small like)
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
        )
    elif model_size == "tiny":
        # very small quick dev
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=512,
            n_ctx=512,
            n_embd=256,
            n_layer=4,
            n_head=4,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
        )
    else:
        raise ValueError("Unknown model_size. choose tiny/small/medium")
    return config


# -----------------------
# Refiner module
# -----------------------
class Refiner(nn.Module):
    """
    Refiner: takes penultimate hidden states (B, T, d) and produces logits (B, T, vocab).
    Also returns refined hidden states if needed for recursive passes.
    Architecture:
      LayerNorm(d) -> Linear(d, d//2) -> GELU -> Linear(d//2, d) -> GELU -> residual scale -> add to h -> output_head
    """
    def __init__(self, d_model: int, vocab_size: int, hidden_factor: int = 2, init_scale: float = 0.1, tie_to_base_lm_head: Optional[nn.Linear] = None):
        super().__init__()
        inner = max(d_model // hidden_factor, 4)
        self.ln = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, inner),
            nn.GELU(),
            nn.Linear(inner, d_model),
            nn.GELU(),
        )
        # small learnable scale for residual (start near init_scale)
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        # output head
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        # optional tie to base lm_head weights (transpose)
        if tie_to_base_lm_head is not None:
            # tie weights: output_head.weight = base_lm_head.weight (vocab x d)
            # we will make output_head a view into the base lm_head weights
            # but PyTorch can't simply assign parameter to another tensor easily, so do manual tying:
            # set weight data and register hook to keep them in sync during optimizer updates - simplest: register reference
            with torch.no_grad():
                self.output_head.weight.copy_(tie_to_base_lm_head.weight.data)
            self._tie_to = tie_to_base_lm_head
        else:
            self._tie_to = None

    def forward(self, h):
        """
        h: [B, T, d_model]
        returns logits [B, T, vocab], h_refined [B, T, d_model]
        """
        # compute delta
        x = self.ln(h)
        delta = self.mlp(x)
        h2 = h + delta * self.scale
        logits = self.output_head(h2)
        # if tied, optionally keep weights synced (we rely on weight copying at init; user can control tying)
        return logits, h2


# -----------------------
# Training / Evaluation Loops
# -----------------------
@dataclass
class TrainState:
    device: torch.device
    scaler: Optional[torch.cuda.amp.GradScaler] = None


def train_base(model: GPT2LMHeadModel, tokenizer, train_ids, val_ids, args, state: TrainState):
    """
    Train base model on contiguous token dataset.
    """
    seq_len = args.seq_len
    train_ds = TextDataset(train_ids, seq_len=seq_len)
    val_ds = TextDataset(val_ids, seq_len=seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2)

    model.to(state.device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.lr_base, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs_base
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    loss_f = nn.CrossEntropyLoss()

    print(f"[Base train] steps per epoch: {len(train_loader)}, total steps: {total_steps}")

    global_step = 0
    for epoch in range(args.epochs_base):
        pbar = tqdm(train_loader, desc=f"Base train epoch {epoch+1}/{args.epochs_base}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(state.device)
            labels = batch["labels"].to(state.device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=state.scaler is not None):
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss
            if state.scaler:
                state.scaler.scale(loss).backward()
                state.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                state.scaler.step(optimizer)
                state.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            scheduler.step()
            global_step += 1
            pbar.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

        # evaluation at end of epoch
        val_ppl = evaluate_ppl_from_model(model, val_loader, state, args)
        print(f"[Base train] epoch {epoch+1} val PPL: {val_ppl:.4f}")
        # optionally early stop criteria here

    return model


def evaluate_ppl_from_model(model: GPT2LMHeadModel, dataloader: DataLoader, state: TrainState, args):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating PPL"):
            input_ids = batch["input_ids"].to(state.device)
            labels = batch["labels"].to(state.device)
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            # loss is average over tokens in batch (per HF design)
            bsz, seqlen = labels.size()
            total_loss += loss.item() * (bsz * seqlen)
            total_tokens += bsz * seqlen
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    model.train()
    return ppl


def train_refiner(base_model: GPT2LMHeadModel, refiner: Refiner, tokenizer, train_ids, val_ids, args, state: TrainState):
    """
    Train refiner: on-the-fly compute base penultimate hidden states and train refiner to predict same labels.
    base_model is by default frozen (or optionally low-LR).
    """
    seq_len = args.seq_len
    train_ds = TextDataset(train_ids, seq_len=seq_len)
    val_ds = TextDataset(val_ids, seq_len=seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2)

    # freeze or set requires_grad appropriately for base
    if args.freeze_base:
        for p in base_model.parameters():
            p.requires_grad = False
    else:
        for p in base_model.parameters():
            p.requires_grad = True  # user may set lr_base small
    base_model.to(state.device)
    base_model.eval()  # we'll run base in eval to keep LN stats stable when frozen

    refiner.to(state.device)
    refiner.train()

    # optimizer: optionally include base params with small lr
    param_groups = [{"params": refiner.parameters(), "lr": args.lr_refiner}]
    if not args.freeze_base and args.lr_base_for_refine > 0.0:
        param_groups.append({"params": [p for p in base_model.parameters() if p.requires_grad], "lr": args.lr_base_for_refine})
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs_refine
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps_refiner, num_training_steps=total_steps)

    ce_loss = nn.CrossEntropyLoss(ignore_index=None)

    print(f"[Refiner train] steps per epoch: {len(train_loader)}, total steps: {total_steps}")

    global_step = 0
    for epoch in range(args.epochs_refine):
        pbar = tqdm(train_loader, desc=f"Refiner train epoch {epoch+1}/{args.epochs_refine}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(state.device)
            labels = batch["labels"].to(state.device)

            # 1) forward through base to get hidden states
            with torch.no_grad():
                # ensure we get hidden states
                outputs = base_model(input_ids, output_hidden_states=True)
                # last hidden states (the one before lm_head) is outputs.hidden_states[-1]
                # outputs.hidden_states is tuple: (embeddings, layer1, ..., last)
                h = outputs.hidden_states[-1]  # [B, T, d_model]

                # optionally get base logits for distillation
                if args.distill_lambda > 0.0:
                    base_logits = base_model.lm_head(h) if hasattr(base_model, "lm_head") else base_model.head(h)
                else:
                    base_logits = None

            # 2) pass h to refiner
            logits, h2 = refiner(h)  # logits [B, T, V]

            # compute loss: CE against labels
            bsz, tlen, vocab = logits.size()
            loss_ce = ce_loss(logits.view(-1, vocab), labels.view(-1))

            # optional distillation / anchor loss (KL between refiner logits and base logits)
            loss_kld = 0.0
            if args.distill_lambda > 0.0 and base_logits is not None:
                # compute KL(base || refiner) or refiner||base? we'll use KL(refiner || base)
                # base_logits might be detached (we don't want to update base)
                base_logprob = F.log_softmax(base_logits.detach(), dim=-1)
                ref_logprob = F.log_softmax(logits, dim=-1)
                # KLD per token average
                kld = F.kl_div(ref_logprob, base_logprob, reduction="batchmean")
                loss_kld = kld

            loss = loss_ce + args.distill_lambda * loss_kld

            optimizer.zero_grad()
            if state.scaler:
                state.scaler.scale(loss).backward()
                state.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(refiner.parameters()) + ([p for p in base_model.parameters() if p.requires_grad] if not args.freeze_base else []), args.max_grad_norm)
                state.scaler.step(optimizer)
                state.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(refiner.parameters()) + ([p for p in base_model.parameters() if p.requires_grad] if not args.freeze_base else []), args.max_grad_norm)
                optimizer.step()
            scheduler.step()
            global_step += 1
            pbar.set_postfix({"loss": loss.item(), "ce": loss_ce.item(), "kld": loss_kld if isinstance(loss_kld, float) else loss_kld.item() if loss_kld is not None else 0.0})

        # eval at epoch end
        val_ppl, diag = evaluate_refiner(base_model, refiner, val_loader, state, args)
        print(f"[Refiner train] epoch {epoch+1} val PPL (refiner logits): {val_ppl:.4f}")
        if args.log_diag:
            print(f"[Refiner diag] mean cos sim h->h2: {diag['mean_cos_sim']:.4f}, mean KL(ref||base): {diag['mean_kld']:.6f}")

    return refiner


def evaluate_refiner(base_model: GPT2LMHeadModel, refiner: Refiner, dataloader: DataLoader, state: TrainState, args):
    """
    Evaluate refiner: compute perplexity using refiner logits and return diagnostics.
    """
    base_model.to(state.device)
    base_model.eval()
    refiner.to(state.device)
    refiner.eval()

    total_nll = 0.0
    total_tokens = 0
    cos_sims = []
    klds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Refiner Eval"):
            input_ids = batch["input_ids"].to(state.device)
            labels = batch["labels"].to(state.device)
            outputs = base_model(input_ids, output_hidden_states=True)
            h = outputs.hidden_states[-1]  # [B, T, d]
            base_logits = base_model.lm_head(h) if hasattr(base_model, "lm_head") else base_model.head(h)
            ref_logits, h2 = refiner(h)
            # CE
            bsz, tlen, vocab = ref_logits.size()
            logprobs = F.log_softmax(ref_logits, dim=-1)
            nll = F.nll_loss(logprobs.view(-1, vocab), labels.view(-1), reduction="sum")
            total_nll += nll.item()
            total_tokens += bsz * tlen

            # diagnostics: cosine similarity between h and h2
            # flatten batch*time into N x d
            h_flat = h.view(-1, h.size(-1))
            h2_flat = h2.view(-1, h2.size(-1))
            # normalize
            h_flat_n = F.normalize(h_flat, dim=-1)
            h2_flat_n = F.normalize(h2_flat, dim=-1)
            cos = (h_flat_n * h2_flat_n).sum(dim=-1).mean().item()
            cos_sims.append(cos)

            # diagnostic KLD per token (refiner || base)
            ref_lp = F.log_softmax(ref_logits, dim=-1)
            base_lp = F.log_softmax(base_logits.detach(), dim=-1)
            kld = F.kl_div(ref_lp, base_lp, reduction="batchmean").item()
            klds.append(kld)

    avg_loss = total_nll / total_tokens
    ppl = math.exp(avg_loss)
    mean_cos = sum(cos_sims) / len(cos_sims) if len(cos_sims) > 0 else 0.0
    mean_kld = sum(klds) / len(klds) if len(klds) > 0 else 0.0
    refiner.train()
    return ppl, {"mean_cos_sim": mean_cos, "mean_kld": mean_kld}


# -----------------------
# Recursive inference util
# -----------------------
def recursive_refine_logits(base_model: GPT2LMHeadModel, refiner: Refiner, input_ids: torch.LongTensor, k: int, state: TrainState):
    """
    Given input_ids [1, T], compute base h, then apply refiner k times recursively.
    Return final logits [1, T, V].
    """
    base_model.to(state.device)
    base_model.eval()
    refiner.to(state.device)
    refiner.eval()
    with torch.no_grad():
        outputs = base_model(input_ids.to(state.device), output_hidden_states=True)
        h = outputs.hidden_states[-1]
        logits = None
        for i in range(k):
            logits, h = refiner(h)
    return logits  # caller can softmax/sample


# -----------------------
# Main
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["base", "refine", "all"], default="all", help="Which stage to run")
    p.add_argument("--model_size", choices=["tiny", "small", "medium"], default="small")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs_base", type=int, default=3)
    p.add_argument("--epochs_refine", type=int, default=3)
    p.add_argument("--lr_base", type=float, default=2e-4)
    p.add_argument("--lr_base_for_refine", type=float, default=1e-5, help="LR for base if un-frozen during refine")
    p.add_argument("--lr_refiner", type=float, default=5e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--warmup_steps_refiner", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--freeze_base", action="store_true", help="Freeze base model during refiner training")
    p.add_argument("--distill_lambda", type=float, default=0.0, help="Weight for KL distillation loss (refiner||base)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="./outputs")
    p.add_argument("--dataset_name", type=str, default="wikitext")
    p.add_argument("--dataset_config", type=str, default="wikitext-103-v1")
    p.add_argument("--train_max_samples", type=int, default=None, help="Optional small subset for quick debugging (num docs)")
    p.add_argument("--val_max_samples", type=int, default=None)
    p.add_argument("--epochs_quick", action="store_true", help="Use small epochs for quick tests")
    p.add_argument("--log_diag", action="store_true", help="Log diagnostics during refine eval")
    p.add_argument("--use_amp", action="store_true", help="Use mixed precision")
    p.add_argument("--tie_refiner_to_base_head", action="store_true", help="Initialize refiner output head to base lm_head weights")
    p.add_argument("--save_every_epoch", action="store_true", help="Save checkpoints each epoch")
    p.add_argument("--num_recursive_passes", type=int, default=1, help="Number of recursive inference passes for sampling")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    state = TrainState(device=device, scaler=torch.cuda.amp.GradScaler() if args.use_amp and device.type == "cuda" else None)

    safe_mkdir(args.out_dir)

    # load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})  # ensure pad exists
    vocab_size = len(tokenizer)

    # prepare data (train/validation)
    print("Preparing tokenized datasets (this may take a minute)...")
    train_ids = build_contiguous_dataset(tokenizer, dataset_name=args.dataset_name, subset=args.dataset_config, split="train", seq_len=args.seq_len, max_samples=args.train_max_samples)
    val_ids = build_contiguous_dataset(tokenizer, dataset_name=args.dataset_name, subset=args.dataset_config, split="validation", seq_len=args.seq_len, max_samples=args.val_max_samples)
    print(f"Train tokens: {train_ids.shape[0]}, Val tokens: {val_ids.shape[0]}")

    # Build / train base model
    base_ckpt_path = os.path.join(args.out_dir, "base_model.pt")
    if args.stage in ("base", "all"):
        print("Building base model config...")
        config = make_gpt_config(args.model_size, vocab_size=vocab_size)
        base_model = GPT2LMHeadModel(config)
        # set pad token id
        base_model.resize_token_embeddings(vocab_size)
        base_model.config.pad_token_id = tokenizer.pad_token_id
        print("Training base model...")
        # short-circuit quick epochs
        if args.epochs_quick:
            args.epochs_base = min(args.epochs_base, 1)
            args.epochs_refine = min(args.epochs_refine, 1)
        base_model = train_base(base_model, tokenizer, train_ids, val_ids, args, state)
        print(f"Saving base model to {base_ckpt_path}")
        torch.save(base_model.state_dict(), base_ckpt_path)
    else:
        print("Loading base model from checkpoint...")
        config = make_gpt_config(args.model_size, vocab_size=vocab_size)
        base_model = GPT2LMHeadModel(config)
        base_model.resize_token_embeddings(vocab_size)
        base_model.config.pad_token_id = tokenizer.pad_token_id
        base_model.load_state_dict(torch.load(base_ckpt_path, map_location="cpu"))

    # Stage 2: build refiner and train
    refiner_ckpt_path = os.path.join(args.out_dir, "refiner.pt")
    if args.stage in ("refine", "all"):
        print("Constructing refiner...")
        # optional tie to base lm_head (initialize)
        tie_head = base_model.lm_head if hasattr(base_model, "lm_head") else None
        refiner = Refiner(base_model.config.n_embd, vocab_size, init_scale=0.1, tie_to_base_lm_head=tie_head if args.tie_refiner_to_base_head else None)
        print("Training refiner (on-the-fly base forward)...")
        refiner = train_refiner(base_model, refiner, tokenizer, train_ids, val_ids, args, state)
        print(f"Saving refiner to {refiner_ckpt_path}")
        torch.save(refiner.state_dict(), refiner_ckpt_path)
    else:
        print("Loading refiner from checkpoint...")
        refiner = Refiner(base_model.config.n_embd, vocab_size, init_scale=0.1)
        refiner.load_state_dict(torch.load(refiner_ckpt_path, map_location="cpu"))

    # Final evaluation
    print("Final evaluation of base model on validation set (PPL):")
    # small val loader
    val_ds = TextDataset(val_ids, seq_len=args.seq_len)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    base_ppl = evaluate_ppl_from_model(base_model, val_loader, state, args)
    print(f"Base model val PPL: {base_ppl:.4f}")

    print("Final evaluation of refiner (PPL on refiner logits):")
    refiner_ppl, diag = evaluate_refiner(base_model, refiner, val_loader, state, args)
    print(f"Refiner val PPL: {refiner_ppl:.4f}")
    if args.log_diag:
        print(f"Diagnostics: {diag}")

    # Demonstrate recursive inference on a short context
    demo_text = "In recent years, artificial intelligence research has"
    enc = tokenizer(demo_text, return_tensors="pt")["input_ids"]
    enc = enc.to(state.device)
    k = args.num_recursive_passes
    print(f"Running recursive inference for k={k} passes on sample: \"{demo_text}\"")
    logits = recursive_refine_logits(base_model, refiner, enc, k=k, state=state)
    probs = F.softmax(logits[:, -1, :], dim=-1)  # probs for next token
    topk = torch.topk(probs, k=10, dim=-1)
    top_tokens = [tokenizer.decode(int(t)) for t in topk.indices[0]]
    print("Top next-token candidates (after refinement):", top_tokens)

    print("Done. Outputs and checkpoints saved to:", args.out_dir)


if __name__ == "__main__":
    main()

