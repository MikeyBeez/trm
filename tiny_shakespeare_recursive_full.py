# tiny_shakespeare_recursive_full.py
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments
from datasets import Dataset
import os

# Paths
DATA_PATH = "tiny_shakespeare.txt"
MODEL_OUTPUT_DIR = "tiny_shakespeare_recursive_model"

# Load dataset
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Initialize tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token exists

# Tokenize text
tokens = tokenizer(text)["input_ids"]

# Chunking function
def chunk_text(token_list, block_size=128):
    chunks = []
    for i in range(0, len(token_list) - block_size + 1, block_size):
        chunks.append(token_list[i:i+block_size])
    return chunks

chunks = chunk_text(tokens, block_size=128)
dataset = Dataset.from_dict({"input_ids": chunks})

# Add attention mask
def add_attention_mask(batch):
    batch["attention_mask"] = [ [1]*len(x) for x in batch["input_ids"] ]
    return batch

dataset = dataset.map(add_attention_mask, batched=True)

# Initialize model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))  # In case pad token was added

# Training arguments
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    learning_rate=5e-4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Data collator
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train model
trainer.train()

# Save model
trainer.save_model(MODEL_OUTPUT_DIR)

# Generation example
model.eval()
prompt = "ROMEO:\n"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
attention_mask = input_ids.ne(tokenizer.pad_token_id)
output_ids = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=200,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
)
print("\n=== Generated Text ===\n")
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

