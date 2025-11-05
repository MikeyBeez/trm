import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments
from datasets import Dataset

# Load tiny Shakespeare text
with open("tiny_shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Create a Hugging Face Dataset
dataset = Dataset.from_dict({"text": [text]})

# Load tokenizer and model
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set EOS token as PAD

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.config.pad_token_id = model.config.eos_token_id  # Ensure model knows padding

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Group texts into sequences for training
block_size = 128  # sequence length

def group_texts(examples):
    concatenated = sum(examples["input_ids"], [])
    total_length = (len(concatenated) // block_size) * block_size
    input_ids = [concatenated[i:i+block_size] for i in range(0, total_length, block_size)]
    return {
        "input_ids": input_ids,
        "attention_mask": [[1]*block_size for _ in range(len(input_ids))],
        "labels": input_ids  # <-- labels set for causal LM
    }

lm_dataset = tokenized_dataset.map(group_texts, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./tiny_shakespeare_recursive_model",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    save_steps=500,
    save_total_limit=2,
    learning_rate=5e-4,
    logging_steps=50,
    report_to="none",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset,
    tokenizer=tokenizer,
)

# Train the model
print("Training GPT2 on Tiny Shakespeare dataset recursively...")
trainer.train()

# Save model
trainer.save_model("./tiny_shakespeare_recursive_model")

# Generate text
model.eval()
prompt = "ROMEO:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
output_ids = model.generate(
    input_ids,
    max_length=200,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
)
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("\n=== Generated Text ===\n")
print(generated_text)

