#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import math

# -----------------------
# Configurations
# -----------------------
SEQ_LEN = 256
BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 5e-4
USE_FP16 = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Load dataset
# -----------------------
print("Loading WikiText-2...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
dataset["train"] = dataset["train"].select(range(2000))
dataset["validation"] = dataset["validation"].select(range(500))

# -----------------------
# Tokenizer
# -----------------------
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=SEQ_LEN, padding="max_length")

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# -----------------------
# Data collator
# -----------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# -----------------------
# Model (small GPT-style)
# -----------------------
config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=SEQ_LEN,
    n_ctx=SEQ_LEN,
    n_embd=512,
    n_layer=6,
    n_head=8,
)
model = GPT2LMHeadModel(config).to(DEVICE)

# -----------------------
# Training arguments
# -----------------------
training_args = TrainingArguments(
    output_dir="./wikitext2_gpt",
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_steps=100,
    save_steps=500,
    save_strategy="epoch",  # avoid intermediate safetensors
    learning_rate=LEARNING_RATE,
    fp16=USE_FP16,
    report_to="none",
    # Use PyTorch save instead of safetensors to avoid tied weights issue
    save_safetensors=False,
)

# -----------------------
# Trainer for base model
# -----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# -----------------------
# Train base model
# -----------------------
print("Training base model...")
trainer.train()

# -----------------------
# Evaluate base model
# -----------------------
eval_results = trainer.evaluate()
ppl = math.exp(eval_results["eval_loss"])
print(f"\nBase model perplexity: {ppl:.2f}")

# -----------------------
# Recursive/refiner training
# -----------------------
print("Starting recursive/refiner training...")

# Freeze base model parameters
for param in model.parameters():
    param.requires_grad = False

class Refiner(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base = base_model
        self.refiner = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, labels=None, **kwargs):
        outputs = self.base.transformer(input_ids, **kwargs)
        last_hidden = outputs.last_hidden_state  # penultimate layer
        refined = self.refiner(last_hidden)
        logits = self.base.lm_head(refined)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_labels = shift_labels.clamp(0, logits.size(-1) - 1)
            loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"loss": loss, "logits": logits}

refiner_model = Refiner(model, config.n_embd).to(DEVICE)

# Trainer for refiner
refiner_trainer = Trainer(
    model=refiner_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train refiner
refiner_trainer.train()

# Evaluate refiner
eval_results = refiner_trainer.evaluate()
ppl_refined = math.exp(eval_results["eval_loss"])
print(f"\nRefiner model perplexity: {ppl_refined:.2f}")

# Save final model manually
print("Saving refiner model manually...")
torch.save(refiner_model.state_dict(), "refiner_model.pt")
print("Done.")

