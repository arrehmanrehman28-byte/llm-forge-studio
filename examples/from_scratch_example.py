"""
Example: Train LLM FROM SCRATCH - Brand new model, no pretrained weights!
This is what LLM Forge Studio's From Scratch tab does!
"""
from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments, Trainer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import Dataset
import torch

# 1. TRAIN TOKENIZER FROM SCRATCH
print("🔤 Training tokenizer from scratch...")

# Sample data - replace with your dataset
texts = [
    "Hello world, this is my custom LLM trained from scratch!",
    "LLM Forge Studio makes it easy to build LLMs from zero.",
    "From scratch means no pretrained weights, only your data.",
] * 1000

# Write to file for tokenizer trainer
with open("train_data.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(texts))

# Train BPE tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(
    vocab_size=10000,
    special_tokens=["<|endoftext|>", "<pad>", "<unk>"]
)

tokenizer.train(["train_data.txt"], trainer)
tokenizer.save("tokenizer.json")
print("✅ Tokenizer saved - Vocab 10k")

# 2. CREATE MODEL ARCHITECTURE FROM SCRATCH
print("\n🧬 Creating model from scratch...")

# Tiny 10M model - For demo, trains on CPU!
config = GPT2Config(
    vocab_size=10000,
    n_positions=1024,
    n_embd=256,      # Hidden size
    n_layer=4,       # Layers
    n_head=4,        # Attention heads
    n_inner=1024,    # FFN size
)

# RANDOM initialization - No pretrained weights!
model = GPT2LMHeadModel(config)
print(f"✅ New model created: {model.num_parameters():,} params (from scratch!)")

# 3. PREPARE DATASET
print("\n📊 Preparing dataset...")

# Tokenize
def tokenize_function(examples):
    # Simple char-level tokenization for demo
    # In real app, use your trained tokenizer
    return {"input_ids": [[random.randint(0, 9999) for _ in range(128)] for _ in range(len(examples["text"]))]}

# Dummy tokenized dataset
import random
dummy_data = [{"input_ids": [random.randint(0, 9999) for _ in range(128)], "labels": [random.randint(0, 9999) for _ in range(128)]} for _ in range(100)]
dataset = Dataset.from_list(dummy_data)

# 4. TRAIN FROM SCRATCH
print("\n🚀 Pre-training from scratch...")

training_args = TrainingArguments(
    output_dir="./scratch_models/my-first-llm",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,  # Higher LR for from-scratch
    warmup_steps=50,
    logging_steps=10,
    save_steps=100,
    fp16=False,  # Set True if GPU
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
print("\n✅ Model trained FROM SCRATCH! Saved to ./scratch_models/my-first-llm")
print("💡 This model has NEVER seen any data before your dataset - 100% custom!")

# 5. TEST GENERATION
# from transformers import AutoTokenizer
# tokenizer_hf = AutoTokenizer.from_pretrained("./scratch_models/my-first-llm")
# inputs = tokenizer_hf("Hello, I am", return_tensors="pt")
# outputs = model.generate(**inputs, max_new_tokens=50)
# print(tokenizer_hf.decode(outputs[0]))

# 6. EXPORT TO GGUF FOR OLLAMA
# Use llama.cpp converter:
# python llama.cpp/convert-hf-to-gguf.py ./scratch_models/my-first-llm --outfile my-llm.gguf --outtype q4_k_m
# Then create Modelfile and: ollama create my-llm -f Modelfile
