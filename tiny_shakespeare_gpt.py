# tiny_shakespeare_gpt_fixed2.py
import os
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Download dataset
dataset_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
dataset_file = "tiny_shakespeare.txt"
if not os.path.exists(dataset_file):
    os.system(f"wget {dataset_url} -O {dataset_file}")

# Load tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Helper: tokenize and split into chunks
def load_dataset(file_path, tokenizer, block_size=128):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    examples = []
    for i in range(0, len(tokens) - block_size, block_size):
        examples.append(tokens[i:i + block_size])
    return examples

train_dataset = load_dataset(dataset_file, tokenizer)
eval_dataset = load_dataset(dataset_file, tokenizer)

# Convert to torch Dataset
class GPT2Dataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        ids = torch.tensor(self.examples[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids}

train_dataset = GPT2Dataset(train_dataset)
eval_dataset = GPT2Dataset(eval_dataset)

# Load GPT2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./tiny_gpt",
    overwrite_output_dir=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    learning_rate=3e-4,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    fp16=torch.cuda.is_available(),
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Train
print("Training GPT2 on Tiny Shakespeare dataset...")
trainer.train()

# Save model
print("Saving trained model...")
trainer.save_model("./tiny_gpt_final")
tokenizer.save_pretrained("./tiny_gpt_final")

# Quick generation test
model.eval()
prompt = "ROMEO:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)  # <-- MOVE TO DEVICE

outputs = model.generate(
    **inputs,
    max_length=200,
    do_sample=True,
    top_k=30,
    top_p=0.9
)

print("\n=== Generated Text ===\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

