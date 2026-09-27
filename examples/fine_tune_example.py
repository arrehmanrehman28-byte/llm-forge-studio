"""
Example: Fine-tune Qwen 0.5B with QLoRA - No-Code equivalent in Python
This is what LLM Forge Studio does behind the scenes!
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import torch

# 1. Load model with QLoRA 4-bit
model_id = "Qwen/Qwen2-0.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 2. Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# 3. LoRA config - Only 1% params!
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# 4. Load dataset
# dataset = load_dataset("yahma/alpaca-cleaned", split="train")
# For demo, create dummy dataset
from datasets import Dataset
data = [
    {"text": "### Instruction: Explain QLoRA\n### Response: QLoRA is efficient fine-tuning with 4-bit quantization and LoRA adapters."},
    {"text": "### Instruction: What is Flash Attention?\n### Response: Flash Attention is 2x faster attention with 50% less memory."}
] * 100
dataset = Dataset.from_list(data)

# 5. Training args - Optimized for speed
training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_steps=100,
    gradient_checkpointing=True,
    report_to="none",
)

# 6. Train with SFTTrainer - Super fast!
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    args=training_args,
    max_seq_length=512,
)

print("🚀 Starting training...")
trainer.train()
print("✅ Done! Model saved to ./checkpoints")

# 7. Test
# model = trainer.model
# inputs = tokenizer("Explain QLoRA", return_tensors="pt").to("cuda")
# outputs = model.generate(**inputs, max_new_tokens=100)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# 8. Export to GGUF for Ollama
# model = model.merge_and_unload()
# model.save_pretrained("./final-model", safe_serialization=True)
# Then: python llama.cpp/convert-hf-to-gguf.py ./final-model --outfile model.gguf --outtype q4_k_m
# ollama create my-model -f Modelfile
