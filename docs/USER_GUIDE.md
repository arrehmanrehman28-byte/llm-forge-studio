# 📖 LLM Forge Studio - User Guide

## 🎯 3-Minute Quick Start

### For Complete Beginners (No Code!)

1. **Open App**: Go to `http://localhost:7860` after running `python app.py`

2. **Choose Mode**:
   - **Fine-Tuning** (Maker tab): You have existing model, want to customize it - FAST
   - **From Scratch** (From Scratch tab): You want brand new LLM from zero - CUSTOM

3. **Fine-Tuning Flow** (Recommended for beginners):
   - **Maker**: Select `Qwen2-0.5B-Instruct` (fastest, works on CPU)
   - **Data**: Upload CSV with `text` column or `instruction,input,output` - Or use sample_data.csv
   - **Train**: Click `START TRAINING` - Watch loss go down
   - **Test**: Chat with your model
   - **Export**: Click `GGUF Q4_K_M + Ollama` → Download → Run with Ollama

4. **From Scratch Flow** (Advanced):
   - **Data**: Upload large TXT file (10k+ lines)
   - **From Scratch**: Name `my-llm`, Arch `tiny-10M`, Vocab 16000, Check `Train Tokenizer`
   - **Train**: Click `START FROM SCRATCH` - Watch tokenizer + model training
   - **Test & Export**: Same as above

## 📊 Data Format Guide

### CSV Formats Supported:

**Format 1: Simple Text (for pre-training)**
```csv
text
"Hello world, this is my data"
"Second line of training data"
```

**Format 2: Instruction Tuning (for chatbots)**
```csv
instruction,input,output
"Explain QLoRA","What is QLoRA?","QLoRA is..."
"Write code","Sort list","def sort_list(arr):..."
```

**Format 3: Prompt/Completion**
```csv
prompt,completion
"What is AI?","AI is artificial intelligence..."
```

**Format 4: JSONL**
```jsonl
{"text": "Your training text here"}
{"instruction": "Explain", "output": "Explanation"}
```

### Tips:
- **From Scratch**: Needs 10k+ lines, TXT or CSV with text column
- **Fine-Tuning**: Works with 100+ examples
- Use UTF-8 encoding
- No empty rows

## ⚙️ Understanding Settings

### LoRA / QLoRA (Fine-Tuning)
- **LoRA Rank (r)**: Higher = More learning capacity but more VRAM. Start with 16.
- **QLoRA 4-bit**: Enable to train 7B models on 8GB VRAM - Always enable!
- **Alpha**: Usually 2x rank (rank 16 → alpha 32)

### Training Hyperparameters
- **Epochs**: How many times model sees data. 1-3 for fine-tuning, 1 for from-scratch demo.
- **Batch Size**: How many examples at once. Lower if OOM.
- **Grad Accum**: Effective batch = Batch × Accum. Use 4 to simulate larger batch.
- **LR**: Learning rate. 2e-4 for fine-tuning, 3e-4 for from-scratch.
- **Mixed Precision**: fp16 (fast, NVIDIA) or bf16 (new GPUs) - Saves VRAM
- **Flash Attention**: 2x faster, 50% less VRAM - Always enable if available

### From Scratch Architecture
- **tiny-10M**: 4 layers, 256 hidden - For testing, trains on CPU in minutes
- **small-50M**: 8 layers, 512 hidden - Good for small chatbots
- **base-124M**: 12 layers, 768 hidden - GPT-2 size, needs GPU
- **custom**: Define your own - For research

## 🧪 Testing Your Model

### Chat Playground
- **Temperature**: 0.1 = Focused, deterministic | 1.0+ = Creative, random
- **Top-P**: 0.9 = Balanced | 1.0 = All tokens considered
- **Max Tokens**: How long response should be
- **Model Type**: Compare Base vs Fine-Tuned vs From Scratch

### Batch Testing
Upload CSV with `prompt` column → Get responses for all → See metrics

### Metrics Explained
- **Perplexity**: Lower = Better (how surprised model is). <20 good.
- **ROUGE**: For summarization, higher = better
- **BLEU**: For translation, higher = better
- **Tokens/sec**: Speed - Higher = Faster inference

## 📤 Export Guide

### Which Format to Choose?

| Format | Size | Use Case | Quality |
|--------|------|----------|---------|
| SafeTensors | 2.1GB | HuggingFace, Python | Full |
| GGUF Q4_K_M | 0.6GB | Ollama, LM Studio, Mobile | Good (Recommended) |
| GGUF Q8_0 | 1.1GB | Ollama, Higher quality | Better |
| GGUF F16 | 2.1GB | Full quality | Best |
| ONNX | 1.9GB | CPU inference, FastAPI | Full |
| Ollama Modelfile | 2KB | Ollama | - |

**Recommendation**: Export `GGUF Q4_K_M + Ollama Modelfile` for best size/quality.

### How to Use Exports

**Ollama (Easiest):**
```bash
# After export, you get model.gguf + Modelfile
ollama create my-custom-model -f ./exports/Modelfile
ollama run my-custom-model "Hello, how are you?"
```

**Python:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
model = PeftModel.from_pretrained(model, "./checkpoints")
model = model.merge_and_unload()
model.save_pretrained("./final-model")
```

**HuggingFace Hub:**
```bash
# In Export tab, check "Push to Hub" and enter username/model-name
# Or use code:
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path="./exports/final-model", repo_id="username/my-model")
```

## ❓ FAQ

**Q: Do I need GPU?**
A: No! Tiny models (360M, 0.5B, 10M scratch) train on CPU. GPU makes it faster. QLoRA allows 7B on 8GB VRAM.

**Q: Fine-tuning vs From Scratch?**
A: Fine-tuning = Customize existing model (fast, little data). From Scratch = New model from zero (slow, lots of data, 100% custom).

**Q: How much data?**
A: Fine-tuning: 100+ examples. From Scratch: 10k+ lines minimum, 1M+ for good model.

**Q: My training is slow?**
A: Enable QLoRA, Flash Attention, use smaller model (Qwen 0.5B), reduce batch size, use fp16.

**Q: OOM Error?**
A: Reduce batch size to 1, enable QLoRA 4-bit, gradient checkpointing, use smaller model.

**Q: Can I train Llama 3 70B?**
A: With QLoRA on 24GB VRAM, yes! But slow. Use 0.5B for testing, then scale.

## 🆘 Need Help?

- Check logs in Train tab
- See Python code in accordion - Run locally for more control
- Open GitHub issue with logs
- Join Discord (coming soon)

Happy LLM building! 🚀
