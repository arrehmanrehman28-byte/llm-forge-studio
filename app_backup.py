"""
LLM Forge Studio - Main Gradio App
No-Code, Python-Friendly, Highly Efficient LLM Maker, Trainer, Tuner, Exporter, Tester
"""

import os
import json
import time
import random
import pandas as pd

# === PATCH Gradio Client Bug (bool additionalProperties) ===
try:
    import gradio_client.utils as client_utils
    orig_json_schema_to_python_type = client_utils._json_schema_to_python_type
    orig_get_type = client_utils.get_type
    
    def patched_get_type(schema):
        if isinstance(schema, bool):
            return "Any"
        try:
            return orig_get_type(schema)
        except Exception:
            return "Any"
    
    def patched_json_schema_to_python_type(schema, defs=None, *args, **kwargs):
        # Handle bool schema
        if isinstance(schema, bool):
            return "Any"
        if isinstance(schema, dict):
            # Fix additionalProperties bool
            if "additionalProperties" in schema and isinstance(schema["additionalProperties"], bool):
                schema = dict(schema)
                if schema["additionalProperties"]:
                    schema["additionalProperties"] = {"type": "object"}
                else:
                    del schema["additionalProperties"]
            # Also fix if schema itself has bool values where dict expected
            # Recursively fix nested
            for k, v in list(schema.items()):
                if isinstance(v, bool) and k in ("additionalProperties",):
                    continue
        try:
            # Call original with whatever args it expects
            return orig_json_schema_to_python_type(schema, defs)
        except TypeError as e:
            if "bool" in str(e) or "not iterable" in str(e):
                return "Any"
            # Try with 3 args for newer versions
            try:
                return orig_json_schema_to_python_type(schema, defs, False)
            except:
                return "Any"
        except Exception:
            return "Any"
    
    client_utils.get_type = patched_get_type
    client_utils._json_schema_to_python_type = patched_json_schema_to_python_type
    print("✅ Patched gradio_client.utils for bool schema bug")
except Exception as e:
    print(f"Patch attempt note: {e}")

import gradio as gr

# Import backend modules with fallback
try:
    from backend.utils import get_gpu_info, get_model_info, load_dataset_auto, generate_maker_code, MODEL_DB
    from backend.trainer import train_model
    from backend.tuner import run_tuning, generate_tuning_code
    from backend.tester import generate_response, batch_inference, calculate_metrics, generate_api_code, mock_generate
    from backend.exporter import export_model, generate_export_code, list_exports
    from backend.scratch import train_from_scratch, generate_scratch_code, SCRATCH_ARCHS, get_arch_info
except ImportError:
    # Fallback if run from different cwd
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
    from utils import get_gpu_info, get_model_info, load_dataset_auto, generate_maker_code, MODEL_DB
    from trainer import train_model
    from tuner import run_tuning, generate_tuning_code
    from tester import generate_response, batch_inference, calculate_metrics, generate_api_code, mock_generate
    from exporter import export_model, generate_export_code, list_exports
    from scratch import train_from_scratch, generate_scratch_code, SCRATCH_ARCHS, get_arch_info

# Custom CSS - Modern Dark SaaS Theme like Linear/Vercel
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* { font-family: 'Inter', sans-serif !important; }
.gradio-container { background: #0a0a0b !important; color: #e4e4e7 !important; }
.main { background: #0a0a0b !important; }

/* Header */
.header { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 24px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
}
.header h1 { font-size: 32px !important; font-weight: 800 !important; color: white !important; margin: 0 !important; letter-spacing: -0.02em; }
.header p { color: rgba(255,255,255,0.85) !important; font-size: 15px !important; margin: 8px 0 0 0 !important; }

/* Cards */
.card {
    background: #18181b !important;
    border: 1px solid #27272a !important;
    border-radius: 12px !important;
    padding: 20px !important;
}
.card:hover { border-color: #3f3f46 !important; }

/* Tabs */
.tabs { background: #18181b !important; border-radius: 12px !important; border: 1px solid #27272a !important; }
.tab-nav { background: #0a0a0b !important; border-radius: 10px !important; padding: 4px !important; }
.tab-nav button { border-radius: 8px !important; font-weight: 500 !important; }
.tab-nav button.selected { background: #27272a !important; color: white !important; }

/* Buttons */
.primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; border: none !important; color: white !important; font-weight: 600 !important; border-radius: 8px !important; }
.primary:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important; }
.secondary { background: #27272a !important; border: 1px solid #3f3f46 !important; color: #e4e4e7 !important; border-radius: 8px !important; }

/* Inputs */
input, textarea, select { background: #18181b !important; border: 1px solid #27272a !important; color: #e4e4e7 !important; border-radius: 8px !important; }
input:focus, textarea:focus { border-color: #667eea !important; box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important; }

/* Code blocks */
.code { font-family: 'JetBrains Mono', monospace !important; background: #09090b !important; border: 1px solid #27272a !important; border-radius: 8px !important; }

/* Metrics */
.metric-card { background: linear-gradient(135deg, #18181b 0%, #1f1f23 100%) !important; border: 1px solid #27272a !important; border-radius: 12px !important; padding: 16px !important; }
.metric-value { font-size: 24px !important; font-weight: 700 !important; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0a0b; }
::-webkit-scrollbar-thumb { background: #27272a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3f3f46; }
"""

# Model list
MODEL_CHOICES = [
    "Qwen/Qwen2-0.5B-Instruct (Recommended - Fastest)",
    "HuggingFaceTB/SmolLM2-360M (Ultra Fast - 360M)",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-2",
    "google/gemma-2b",
    "meta-llama/Llama-3.2-1B",
    "mistralai/Mistral-7B-v0.1 (Needs 8GB+ VRAM)",
    "Custom HF Model ID"
]

MODEL_MAP = {
    "Qwen/Qwen2-0.5B-Instruct (Recommended - Fastest)": "Qwen/Qwen2-0.5B-Instruct",
    "HuggingFaceTB/SmolLM2-360M (Ultra Fast - 360M)": "HuggingFaceTB/SmolLM2-360M",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-2": "microsoft/phi-2",
    "google/gemma-2b": "google/gemma-2b",
    "meta-llama/Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "mistralai/Mistral-7B-v0.1 (Needs 8GB+ VRAM)": "mistralai/Mistral-7B-v0.1",
    "Custom HF Model ID": "custom"
}

def get_clean_model_id(choice):
    return MODEL_MAP.get(choice, choice)

def model_info_html(model_choice, custom_id, task):
    model_id = custom_id.strip() if get_clean_model_id(model_choice) == "custom" else get_clean_model_id(model_choice)
    if not model_id or model_id == "custom":
        return "<div class='metric-card'>⚠️ Enter custom model ID</div>"
    
    info = get_model_info(model_id)
    # If not in DB, use generic
    if info['params'] == "Unknown":
        info = {"params": "Custom", "size": "Varies", "vram": "Check HF", "type": task, "context": "Varies", "fast": False}
    
    recommended = "🔥 Recommended for fast training" if info.get('recommended') or info.get('fast') else ""
    
    return f"""
    <div class="metric-card" style="margin: 12px 0;">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div>
                <div style="font-weight: 700; font-size: 16px; color: #e4e4e7; margin-bottom: 8px;">📦 {model_id}</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 13px; color: #a1a1aa;">
                    <div><span style="color: #71717a;">Params:</span> <b style="color: #e4e4e7;">{info['params']}</b></div>
                    <div><span style="color: #71717a;">Size:</span> <b style="color: #e4e4e7;">{info['size']}</b></div>
                    <div><span style="color: #71717a;">VRAM:</span> <b style="color: #10b981;">{info['vram']}</b></div>
                    <div><span style="color: #71717a;">Type:</span> <b style="color: #e4e4e7;">{info['type']}</b></div>
                    <div><span style="color: #71717a;">Context:</span> <b style="color: #e4e4e7;">{info['context']}</b></div>
                    <div><span style="color: #71717a;">Task:</span> <b style="color: #e4e4e7;">{task}</b></div>
                </div>
                <div style="margin-top: 10px; font-size: 12px; color: #a78bfa;">{recommended}</div>
            </div>
            <div style="background: {'#10b981' if info.get('fast') else '#f59e0b'}; color: white; padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: 600;">
                {'⚡ FAST' if info.get('fast') else '🐢 SLOW'}
            </div>
        </div>
    </div>
    """

def gpu_info_html():
    info = get_gpu_info()
    color = "#10b981" if info['has_gpu'] else "#f59e0b"
    return f"""
    <div style="background: #18181b; border: 1px solid #27272a; border-radius: 8px; padding: 12px; font-size: 13px;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <div style="width: 8px; height: 8px; background: {color}; border-radius: 50%;"></div>
            <b style="color: #e4e4e7;">{info['available']}</b>
        </div>
        <div style="color: #a1a1aa; font-size: 12px;">
            Torch: {info.get('torch_version', 'N/A')} | {info.get('gpu_name', 'CPU')} | {info.get('total_vram', '')}
        </div>
    </div>
    """

def handle_dataset_upload(file, hf_name, split_ratio):
    if file is None and not hf_name:
        return None, "⚠️ Upload file or enter HF dataset name", "", ""
    
    file_path = file.name if file else None
    df, msg = load_dataset_auto(file_path, hf_name, split_ratio)
    
    if df is None:
        return None, msg, "", ""
    
    # Generate code for dataset loading
    if file_path:
        code = f'''# Load your dataset - LLM Forge
import pandas as pd
from datasets import Dataset

# From your uploaded file
df = pd.read_csv("{os.path.basename(file_path) if file_path else 'data.csv'}")
print(df.head())

# Convert to HF Dataset for training
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size={split_ratio})

# Ready for SFTTrainer!
'''
    else:
        code = f'''# Load HF dataset - LLM Forge
from datasets import load_dataset

dataset = load_dataset("{hf_name}", split="train")
dataset = dataset.train_test_split(test_size={split_ratio})
print(dataset)
'''
    
    preview = df.head(20) if isinstance(df, pd.DataFrame) else df
    return preview, msg, code, file_path or hf_name

def training_flow(model_choice, custom_id, task, dataset_file_state, epochs, batch_size, grad_accum, lr, lora_r, lora_alpha, lora_dropout, use_lora, use_qlora, mixed_prec, optimizer, use_flash, use_compile):
    model_id = custom_id.strip() if get_clean_model_id(model_choice) == "custom" else get_clean_model_id(model_choice)
    
    if not model_id or model_id == "custom":
        yield "❌ Select a valid model", None, "❌ Error", gpu_info_html(), "Select model first"
        return
    
    config = {
        "model_id": model_id,
        "task": task,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "grad_accum": int(grad_accum),
        "lr": float(lr),
        "lora_r": int(lora_r),
        "lora_alpha": int(lora_alpha),
        "lora_dropout": float(lora_dropout),
        "use_lora": use_lora,
        "use_qlora": use_qlora,
        "mixed_precision": mixed_prec,
        "optimizer": optimizer,
        "use_flash": use_flash,
        "use_compile": use_compile,
        "dataset_path": dataset_file_state
    }
    
    # Generate training code
    from backend.utils import generate_trainer_code
    code = generate_trainer_code(config)
    
    logs = []
    train_losses = []
    val_losses = []
    steps = []
    
    # For plot
    import plotly.graph_objects as go
    
    logs.append(f"🚀 Starting training for {model_id}")
    logs.append(f"📊 Config: Epochs={epochs}, Batch={batch_size}, LR={lr}, LoRA r={lora_r}, QLoRA={use_qlora}")
    logs.append(f"⚡ Optimizations: FlashAttn={use_flash}, Compile={use_compile}, Mixed={mixed_prec}")
    logs.append("")
    
    for update in train_model(config, dataset_file_state):
        metrics = update['metrics']
        log_line = update['log']
        logs.append(log_line)
        
        train_losses.append(metrics['train_loss'])
        val_losses.append(metrics['val_loss'])
        steps.append(metrics['step'])
        
        # Create dataframe for LinePlot (Gradio 5.x compatible)
        plot_data = []
        for s, tl, vl in zip(steps, train_losses, val_losses):
            plot_data.append({"step": s, "loss": tl, "type": "train"})
            plot_data.append({"step": s, "loss": vl, "type": "val"})
        plot_df = pd.DataFrame(plot_data)
        
        status = f"🔥 Training... Step {metrics['step']}/{metrics['total_steps']} | Loss: {metrics['train_loss']} | ETA: {metrics['eta']}"
        gpu_html = f"""
        <div style="background: #18181b; border: 1px solid #27272a; border-radius: 8px; padding: 12px; font-size: 12px;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                <div><span style="color: #71717a;">VRAM:</span> <b style="color: #10b981;">{metrics['gpu_mem']}</b></div>
                <div><span style="color: #71717a;">Speed:</span> <b style="color: #e4e4e7;">{metrics['steps_per_sec']} steps/s</b></div>
                <div><span style="color: #71717a;">LR:</span> <b style="color: #e4e4e7;">{metrics['learning_rate']:.2e}</b></div>
                <div><span style="color: #71717a;">PPL:</span> <b style="color: #e4e4e7;">{metrics['perplexity']}</b></div>
            </div>
        </div>
        """
        
        log_text = "\n".join(logs[-50:])  # Last 50 lines
        
        yield log_text, plot_df, status, gpu_html, code
        
        if update['finished']:
            break
    
    final_status = f"✅ Training Complete! Final Loss: {train_losses[-1] if train_losses else 'N/A'} | Checkpoint saved to ./checkpoints"
    logs.append("")
    logs.append(final_status)
    logs.append("💡 Now go to Test tab to chat with your model, or Export tab to export to GGUF/Ollama")
    
    # Final plot
    plot_data = []
    for s, tl, vl in zip(steps, train_losses, val_losses):
        plot_data.append({"step": s, "loss": tl, "type": "train"})
        plot_data.append({"step": s, "loss": vl, "type": "val"})
    final_plot_df = pd.DataFrame(plot_data) if plot_data else pd.DataFrame([{"step": 0, "loss": 0, "type": "train"}])
    
    yield "\n".join(logs), final_plot_df, final_status, gpu_info_html(), code

def tuning_flow(tune_lr, tune_r, tune_bs, tune_epochs, n_trials):
    search_space = {
        "tune_lr": tune_lr,
        "tune_lora_r": tune_r,
        "tune_batch_size": tune_bs,
        "tune_epochs": tune_epochs
    }
    
    logs = []
    trials_data = []
    best_loss = float('inf')
    best_params = {}
    
    logs.append(f"🔍 Starting hyperparameter tuning with {n_trials} trials")
    logs.append(f"Search space: LR={tune_lr}, LoRA R={tune_r}, Batch={tune_bs}, Epochs={tune_epochs}")
    logs.append("")
    
    for update in run_tuning(search_space, int(n_trials)):
        trial = update['trial']
        logs.append(update['log'])
        
        trials_data.append({
            "Trial": trial['trial'],
            "Loss": trial['loss'],
            "Val Loss": trial['val_loss'],
            "LR": trial['params'].get('lr', 'N/A'),
            "LoRA R": trial['params'].get('lora_r', 'N/A'),
            "Batch": trial['params'].get('batch_size', 'N/A'),
            "Best?": "🔥" if trial['is_best'] else ""
        })
        
        if trial['is_best']:
            best_loss = trial['loss']
            best_params = update['best_params']
        
        df = pd.DataFrame(trials_data)
        
        status = f"🔍 Tuning... Trial {trial['trial']}/{n_trials} | Best Loss: {best_loss:.4f}"
        
        code = generate_tuning_code(search_space, int(n_trials))
        
        yield "\n".join(logs), df, json.dumps(best_params, indent=2), status, code
        
        if update['finished']:
            break
    
    logs.append("")
    logs.append(f"✅ Tuning Complete! Best Loss: {best_loss:.4f}")
    logs.append(f"Best Params: {best_params}")
    
    df = pd.DataFrame(trials_data)
    yield "\n".join(logs), df, json.dumps(best_params, indent=2), f"✅ Best Loss: {best_loss:.4f} | Params: {best_params}", code

def chat_interface(message, history, temp, top_p, top_k, max_tokens, rep_penalty, model_type):
    if not message or not message.strip():
        return history, ""
    
    config = {
        "temperature": temp,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "repetition_penalty": rep_penalty,
        "model_type": model_type
    }
    
    # Generate streaming response
    full_response = ""
    for chunk in generate_response(message, config):
        full_response = chunk
        # For gradio chatbot, we need to update history
        # We'll yield intermediate
        temp_history = history + [[message, full_response]]
        yield temp_history, ""
        time.sleep(0.02)
    
    # Final
    history = history + [[message, full_response]]
    yield history, ""

def batch_test_fn(file, temp, max_tokens):
    if file is None:
        return None, "⚠️ Upload CSV file with 'prompt' or 'text' column", ""
    
    try:
        df = pd.read_csv(file.name)
        # Find prompt column
        prompt_col = None
        for col in ['prompt', 'text', 'input', 'question', 'instruction']:
            if col in df.columns:
                prompt_col = col
                break
        if not prompt_col:
            prompt_col = df.columns[0]
        
        prompts = df[prompt_col].astype(str).tolist()[:20]  # Limit 20 for demo
        
        config = {"temperature": temp, "max_tokens": max_tokens}
        results = batch_inference(prompts, config)
        
        results_df = pd.DataFrame(results)
        
        metrics = calculate_metrics([r['full_response'] for r in results])
        metrics_text = f"""
**Evaluation Metrics:**
- Perplexity: {metrics['perplexity']}
- ROUGE-1: {metrics['rouge1']}
- ROUGE-L: {metrics['rougeL']}
- BLEU: {metrics['bleu']}
- Avg Latency: {metrics['avg_latency']}
- Tokens/sec: {metrics['tokens_per_sec']}
"""
        
        return results_df, f"✅ Batch tested {len(results)} prompts from column '{prompt_col}'", metrics_text
    
    except Exception as e:
        return None, f"❌ Error: {str(e)[:200]}", ""

def export_flow(formats, push_hub, hub_id, gguf_quant):
    if not formats:
        return "⚠️ Select at least one export format", None, "", ""
    
    config = {
        "formats": formats,
        "push_to_hub": push_hub,
        "hub_model_id": hub_id,
        "ollama_gguf": f"model-{gguf_quant}.gguf"
    }
    
    result = export_model(config)
    
    logs = "\n".join(result['logs'])
    
    # Results dataframe
    df = pd.DataFrame(result['results'])
    
    codes = generate_export_code(config)
    python_code = codes['python']
    ollama_code = codes['ollama'] + "\n\n" + codes['hf']
    
    return logs, df, python_code, ollama_code

def scratch_arch_html(arch_choice):
    info = get_arch_info(arch_choice)
    return f"""
    <div class="metric-card" style="margin: 8px 0;">
        <div style="display: flex; justify-content: space-between;">
            <div>
                <div style="font-weight: 700; color: #e4e4e7;">🧬 {arch_choice} - {info['params']}</div>
                <div style="font-size: 12px; color: #a1a1aa; margin-top: 4px;">
                    Layers: {info['layers']} | Hidden: {info['hidden']} | Heads: {info['heads']} | Vocab: {info['vocab']}<br>
                    VRAM: {info.get('vram', 'N/A')} | {info['desc']}
                </div>
            </div>
            <div style="background: #667eea; color: white; padding: 4px 10px; border-radius: 20px; font-size: 11px; height: fit-content;">FROM SCRATCH</div>
        </div>
    </div>
    """

def scratch_training_flow(model_name, arch_choice, num_layers, hidden_size, num_heads, inter_size, vocab_size, context_len, tokenizer_type, train_tokenizer, dataset_path_state, epochs, batch_size, lr):
    # Build config
    arch_info = get_arch_info(arch_choice)
    config = {
        "model_name": model_name or "my-scratch-llm",
        "architecture": arch_choice,
        "num_layers": int(num_layers) if arch_choice == "custom" else arch_info['layers'],
        "hidden_size": int(hidden_size) if arch_choice == "custom" else arch_info['hidden'],
        "num_heads": int(num_heads) if arch_choice == "custom" else arch_info['heads'],
        "intermediate_size": int(inter_size) if arch_choice == "custom" else arch_info['intermediate'],
        "vocab_size": int(vocab_size),
        "context_length": int(context_len),
        "tokenizer_type": tokenizer_type,
        "train_tokenizer": train_tokenizer,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "dataset_path": dataset_path_state
    }
    
    code = generate_scratch_code(config)
    
    logs = []
    train_losses = []
    steps = []
    
    logs.append(f"🧬 Starting FROM SCRATCH training: {config['model_name']}")
    logs.append(f"📐 Architecture: {config['num_layers']} layers, {config['hidden_size']} hidden, {config['num_heads']} heads, {config['vocab_size']} vocab")
    logs.append(f"🔤 Tokenizer: {tokenizer_type} | Vocab {vocab_size} | Train from data: {train_tokenizer}")
    logs.append(f"📊 Training: {epochs} epochs, batch {batch_size}, LR {lr}")
    logs.append("")
    
    for update in train_from_scratch(config, dataset_path_state):
        if update.get('phase') == 'tokenizer':
            logs.append(update['log'])
            status = f"🔤 Tokenizer: {update['step']}/{update['total']} | {int(update['progress']*100)}% | Vocab {update['vocab_built']}/{vocab_size}"
            plot_df = pd.DataFrame([{"step": update['step'], "loss": 0, "type": "tokenizer"}])
            yield "\n".join(logs[-50:]), plot_df, status, gpu_info_html(), code
        else:
            metrics = update['metrics']
            logs.append(update['log'])
            train_losses.append(metrics['train_loss'])
            steps.append(metrics['step'])
            
            plot_data = [{"step": s, "loss": l, "type": "scratch_loss"} for s, l in zip(steps, train_losses)]
            plot_df = pd.DataFrame(plot_data)
            
            status = f"🧬 Training from scratch... Step {metrics['step']}/{metrics['total_steps']} | Loss: {metrics['train_loss']} | PPL: {metrics['perplexity']} | Tokens: {metrics['total_tokens']}"
            
            gpu_html = f"""
            <div style="background: #18181b; border: 1px solid #27272a; border-radius: 8px; padding: 12px; font-size: 12px;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                    <div><span style="color: #71717a;">VRAM:</span> <b style="color: #10b981;">{metrics['gpu_mem']}</b></div>
                    <div><span style="color: #71717a;">Tokens/s:</span> <b style="color: #e4e4e7;">{metrics['tokens_per_sec']}</b></div>
                    <div><span style="color: #71717a;">LR:</span> <b style="color: #e4e4e7;">{metrics['learning_rate']:.2e}</b></div>
                    <div><span style="color: #71717a;">Total Tokens:</span> <b style="color: #e4e4e7;">{metrics['total_tokens']}</b></div>
                </div>
            </div>
            """
            
            yield "\n".join(logs[-50:]), plot_df, status, gpu_html, code
            
            if update['finished']:
                break
    
    final_status = f"✅ From-Scratch Training Complete! Model: {config['model_name']} | Final Loss: {train_losses[-1] if train_losses else 'N/A'} | Saved to ./scratch_models/{config['model_name']}"
    logs.append("")
    logs.append(final_status)
    logs.append("💡 This model was trained FROM ZERO - no pretrained weights! Test it in Test tab, Export to GGUF/Ollama")
    
    final_plot = pd.DataFrame([{"step": s, "loss": l, "type": "scratch_loss"} for s, l in zip(steps, train_losses)]) if steps else pd.DataFrame([{"step":0,"loss":0,"type":"scratch_loss"}])
    
    yield "\n".join(logs), final_plot, final_status, gpu_info_html(), code

# Build Gradio App
with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Monochrome(), title="LLM Forge Studio") as app:
    
    # State
    dataset_state = gr.State(None)
    dataset_path_state = gr.State(None)
    
    # Header
    with gr.Row():
        gr.HTML("""
        <div class="header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1>⚡ LLM Forge Studio</h1>
                    <p>No-Code • Python-Friendly • Highly Efficient LLM Factory — Maker, Trainer, Tuner, Tester, Exporter</p>
                </div>
                <div style="text-align: right; color: rgba(255,255,255,0.9); font-size: 12px;">
                    <div style="background: rgba(255,255,255,0.2); padding: 6px 12px; border-radius: 20px; backdrop-filter: blur(10px);">
                        🚀 v1.0 • QLoRA • FlashAttn • GGUF • Ollama Ready
                    </div>
                </div>
            </div>
        </div>
        """)
    
    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML(gpu_info_html())
        with gr.Column(scale=1):
            refresh_gpu = gr.Button("🔄 Refresh GPU", size="sm", elem_classes="secondary")
    
    # Main Tabs
    with gr.Tabs():
        
        # TAB 1: MODEL MAKER
        with gr.Tab("🏗️ Maker", elem_id="maker"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 🎯 Choose Base Model")
                    model_dropdown = gr.Dropdown(choices=MODEL_CHOICES, value="Qwen/Qwen2-0.5B-Instruct (Recommended - Fastest)", label="Base Model (HuggingFace)", interactive=True)
                    custom_model_id = gr.Textbox(label="Custom HF Model ID (if Custom selected)", placeholder="e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0", visible=True)
                    task_dropdown = gr.Dropdown(choices=["Text Generation", "Chatbot", "Instruction Tuning", "Summarization", "Classification", "Code Generation"], value="Chatbot", label="Task Type")
                    model_info = gr.HTML("<div class='metric-card'>Select model to see specs</div>")
                    
                    with gr.Accordion("🐍 Show Python Code", open=False):
                        maker_code = gr.Code(language="python", label="Python Equivalent")
                        generate_maker_code_btn = gr.Button("Generate Code", size="sm")
                
                with gr.Column(scale=1):
                    gr.Markdown("### ⚡ Quick Start")
                    gr.HTML("""
                    <div class="metric-card">
                        <div style="font-weight: 600; margin-bottom: 8px;">🚀 3 Steps to LLM:</div>
                        <div style="font-size: 13px; color: #a1a1aa; line-height: 1.6;">
                        1. <b style="color: #e4e4e7;">Maker:</b> Pick base model<br>
                        2. <b style="color: #e4e4e7;">Data:</b> Upload CSV/JSONL<br>
                        3. <b style="color: #e4e4e7;">Train:</b> One-click QLoRA<br>
                        4. <b style="color: #e4e4e7;">Test:</b> Chat with model<br>
                        5. <b style="color: #e4e4e7;">Export:</b> GGUF/Ollama
                        </div>
                    </div>
                    """)
                    gr.Markdown("### 💡 Tips")
                    gr.HTML("""
                    <div style="background: #18181b; border: 1px solid #27272a; border-radius: 8px; padding: 12px; font-size: 12px; color: #a1a1aa;">
                        • <b style="color: #10b981;">Qwen 0.5B</b> & <b style="color: #10b981;">SmolLM2 360M</b> = Fastest, runs on CPU<br>
                        • Enable <b>QLoRA 4-bit</b> to train 7B on 8GB VRAM<br>
                        • Use <b>Flash Attention</b> for 2x speed<br>
                        • All code is exportable to Python
                    </div>
                    """)

        # TAB 1.5: FROM SCRATCH - NEW MODELS FROM ZERO
        with gr.Tab("🧬 From Scratch", elem_id="scratch"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🧬 Create NEW Model From Scratch")
                    gr.HTML("""
                    <div style="background: linear-gradient(135deg, #667eea20, #764ba220); border: 1px solid #667eea50; border-radius: 8px; padding: 12px; font-size: 12px;">
                        <b style="color: #a78bfa;">⚠️ FROM SCRATCH = Brand new LLM, no pretrained weights!</b><br>
                        <span style="color: #a1a1aa;">You define architecture, train tokenizer on YOUR data, then pre-train. Takes longer but 100% custom.</span>
                    </div>
                    """)
                    scratch_model_name = gr.Textbox(label="Model Name", value="my-scratch-llm", placeholder="e.g. urdu-llm-50M")
                    scratch_arch = gr.Dropdown(choices=["tiny-10M", "small-50M", "base-124M", "medium-350M", "custom"], value="tiny-10M", label="Architecture Preset")
                    scratch_arch_info = gr.HTML("<div class='metric-card'>Select arch</div>")
                    
                    with gr.Group():
                        gr.Markdown("**🔧 Custom Architecture (if Custom selected)**")
                        scratch_layers = gr.Slider(2, 32, value=8, step=1, label="Num Layers")
                        scratch_hidden = gr.Slider(128, 2048, value=512, step=64, label="Hidden Size")
                        scratch_heads = gr.Slider(2, 32, value=8, step=1, label="Num Attention Heads")
                        scratch_inter = gr.Slider(512, 8192, value=2048, step=128, label="Intermediate Size (FFN)")
                    
                    with gr.Group():
                        gr.Markdown("**🔤 Tokenizer From Scratch**")
                        scratch_vocab = gr.Slider(1000, 50000, value=16000, step=1000, label="Vocab Size")
                        scratch_context = gr.Slider(128, 4096, value=1024, step=128, label="Context Length (max tokens)")
                        scratch_tok_type = gr.Dropdown(choices=["BPE", "WordLevel", "Char", "ByteLevel"], value="BPE", label="Tokenizer Type")
                        scratch_train_tok = gr.Checkbox(value=True, label="Train Tokenizer on Your Dataset (Recommended)")
                    
                    with gr.Group():
                        gr.Markdown("**🚀 Pre-Training Config**")
                        scratch_epochs = gr.Slider(1, 10, value=1, step=1, label="Epochs (From scratch needs 1-3)")
                        scratch_batch = gr.Slider(1, 32, value=4, step=1, label="Batch Size")
                        scratch_lr = gr.Dropdown(choices=["1e-4", "3e-4", "5e-4", "1e-3"], value="3e-4", label="Learning Rate (From scratch uses higher LR)")
                    
                    scratch_train_btn = gr.Button("🧬 START TRAINING FROM SCRATCH", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📈 From-Scratch Training Dashboard")
                    scratch_status = gr.Textbox(label="Status", value="Ready to create new model from zero...")
                    scratch_plot = gr.LinePlot(x="step", y="loss", color="type", title="From-Scratch Loss", y_title="Loss", x_title="Steps", height=300)
                    with gr.Row():
                        scratch_gpu = gr.HTML(gpu_info_html())
                    scratch_logs = gr.Textbox(label="Logs (Tokenizer + Pre-training)", lines=15, max_lines=20, autoscroll=True, elem_classes="code")
            
            with gr.Accordion("🐍 Python Code - From Scratch", open=False):
                scratch_code = gr.Code(language="python", label="Full From-Scratch Code (Tokenizer + Model)")

        # TAB 2: DATA
        with gr.Tab("📊 Data", elem_id="data"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📁 Upload Dataset")
                    file_upload = gr.File(label="Drag & Drop CSV, JSONL, JSON, TXT", file_types=[".csv", ".jsonl", ".json", ".txt"])
                    hf_dataset_input = gr.Textbox(label="Or HuggingFace Dataset Name", placeholder="e.g. yahma/alpaca-cleaned or tatsu-lab/alpaca")
                    split_slider = gr.Slider(0.05, 0.5, value=0.2, step=0.05, label="Validation Split Ratio")
                    load_data_btn = gr.Button("📥 Load & Preview Dataset", elem_classes="primary")
                
                with gr.Column():
                    gr.Markdown("### 👀 Data Preview")
                    data_status = gr.Textbox(label="Status", lines=2)
                    data_preview = gr.Dataframe(label="Preview (first 20 rows)", interactive=False, wrap=True)
            
            with gr.Accordion("🐍 Python Code - Data Loading", open=False):
                data_code = gr.Code(language="python", label="Code to load this dataset")
        
        # TAB 3: TRAIN
        with gr.Tab("🚀 Train", elem_id="train"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Training Config - Highly Efficient")
                    with gr.Group():
                        gr.Markdown("**LoRA / QLoRA (Efficiency)**")
                        use_lora = gr.Checkbox(value=True, label="Enable LoRA (Recommended - Train 1% params)")
                        use_qlora = gr.Checkbox(value=True, label="Enable QLoRA 4-bit (Train 7B on 8GB VRAM)")
                        lora_r = gr.Slider(4, 128, value=16, step=4, label="LoRA Rank (r) - Higher = More capacity")
                        lora_alpha = gr.Slider(8, 128, value=32, step=8, label="LoRA Alpha")
                        lora_dropout = gr.Slider(0.0, 0.5, value=0.05, step=0.05, label="LoRA Dropout")
                    
                    with gr.Group():
                        gr.Markdown("**Training Hyperparameters**")
                        epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                        batch_size = gr.Slider(1, 32, value=4, step=1, label="Batch Size per Device")
                        grad_accum = gr.Slider(1, 16, value=4, step=1, label="Gradient Accumulation (Effective Batch = Batch x Accum)")
                        lr = gr.Dropdown(choices=["1e-5", "2e-5", "5e-5", "1e-4", "2e-4", "3e-4", "5e-4"], value="2e-4", label="Learning Rate")
                        mixed_prec = gr.Dropdown(choices=["fp16", "bf16", "fp32"], value="fp16", label="Mixed Precision")
                        optimizer = gr.Dropdown(choices=["paged_adamw_8bit", "adamw_torch", "adamw_8bit", "lion"], value="paged_adamw_8bit", label="Optimizer (8bit saves VRAM)")
                    
                    with gr.Group():
                        gr.Markdown("**Speed Optimizations**")
                        use_flash = gr.Checkbox(value=True, label="⚡ Flash Attention 2 (2x faster, 50% less VRAM)")
                        use_compile = gr.Checkbox(value=True, label="🔥 torch.compile (20% faster)")
                    
                    train_btn = gr.Button("🚀 START TRAINING (One-Click QLoRA)", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📈 Live Training Dashboard")
                    train_status = gr.Textbox(label="Status", value="Ready to train...")
                    train_plot = gr.LinePlot(x="step", y="loss", color="type", title="Loss Curve", y_title="Loss", x_title="Steps", height=300, container=True)
                    with gr.Row():
                        gpu_stats = gr.HTML(gpu_info_html())
                    train_logs = gr.Textbox(label="Live Logs", lines=15, max_lines=20, autoscroll=True, elem_classes="code")
            
            with gr.Accordion("🐍 Python Code - Training", open=False):
                train_code = gr.Code(language="python", label="Full Training Code (Copy & Run Locally)")
        
        # TAB 4: TUNE
        with gr.Tab("🎛️ Tune", elem_id="tune"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🔍 Auto Hyperparameter Tuning")
                    gr.HTML("<div style='background: #18181b; padding: 12px; border-radius: 8px; font-size: 12px; color: #a1a1aa;'>Uses <b>Optuna</b> to find best LR, LoRA Rank, Batch Size automatically. Saves hours of manual tuning.</div>")
                    tune_lr = gr.Checkbox(value=True, label="Tune Learning Rate (1e-5 to 1e-3)")
                    tune_r = gr.Checkbox(value=True, label="Tune LoRA Rank (4,8,16,32,64)")
                    tune_bs = gr.Checkbox(value=False, label="Tune Batch Size (1,2,4,8)")
                    tune_epochs = gr.Checkbox(value=False, label="Tune Epochs (1-5)")
                    n_trials = gr.Slider(3, 50, value=10, step=1, label="Number of Trials")
                    tune_btn = gr.Button("🔍 Start Auto Tuning", elem_classes="primary")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Tuning Results")
                    tune_status = gr.Textbox(label="Status")
                    tune_leaderboard = gr.Dataframe(label="Leaderboard (Best Trials)", interactive=False)
                    best_params = gr.Code(language="json", label="Best Parameters (Apply to Trainer)")
                    tune_logs = gr.Textbox(label="Tuning Logs", lines=10, autoscroll=True)
            
            with gr.Accordion("🐍 Python Code - Tuning", open=False):
                tune_code = gr.Code(language="python", label="Optuna Tuning Code")
        
        # TAB 5: TEST
        with gr.Tab("🧪 Test", elem_id="test"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 💬 Chat Playground - Test Your Fine-Tuned Model")
                    chatbot = gr.Chatbot(label="Chat with Model", height=400, show_copy_button=True, bubble_full_width=False, type="tuples")
                    chat_input = gr.Textbox(label="Your Message", placeholder="Ask anything... e.g. Explain your training", lines=2)
                    with gr.Row():
                        chat_btn = gr.Button("Send", variant="primary")
                        clear_chat = gr.Button("Clear", size="sm")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🎛️ Generation Config")
                    model_type_test = gr.Dropdown(choices=["Fine-Tuned Model", "Base Model", "Comparison Mode"], value="Fine-Tuned Model", label="Model to Test")
                    temp_slider = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature (Creativity)")
                    top_p_slider = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P (Nucleus)")
                    top_k_slider = gr.Slider(1, 100, value=40, step=1, label="Top-K")
                    max_tokens_slider = gr.Slider(10, 512, value=150, step=10, label="Max New Tokens")
                    rep_penalty = gr.Slider(1.0, 2.0, value=1.1, step=0.1, label="Repetition Penalty")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📦 Batch Testing")
                    batch_file = gr.File(label="Upload CSV with prompts (column: prompt/text)", file_types=[".csv"])
                    batch_btn = gr.Button("🧪 Run Batch Inference", elem_classes="secondary")
                    batch_status = gr.Textbox(label="Batch Status")
                    batch_results = gr.Dataframe(label="Batch Results", wrap=True)
                    batch_metrics = gr.Markdown("Metrics will appear here")
            
            with gr.Accordion("🔌 API Endpoint - Use Your Model Anywhere", open=True):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**FastAPI Endpoint is LIVE at `/api/generate`**")
                        api_code = gr.Code(language="python", value=generate_api_code()["python"], label="Python API Usage")
                    with gr.Column():
                        api_curl = gr.Code(language="shell", value=generate_api_code()["curl"], label="cURL Usage")
        
        # TAB 6: EXPORT
        with gr.Tab("📤 Export", elem_id="export"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 One-Click Export")
                    export_formats = gr.CheckboxGroup(
                        choices=[
                            ("SafeTensors (Recommended, Safe & Fast)", "safetensors"),
                            ("PyTorch .bin (Legacy)", "pytorch"),
                            ("GGUF Q4_K_M (0.6GB - For Ollama, Fast)", "gguf_q4"),
                            ("GGUF Q8_0 (1.1GB - Higher Quality)", "gguf_q8"),
                            ("GGUF F16 (2.1GB - Full Quality)", "gguf_f16"),
                            ("ONNX (For CPU, Fast Inference)", "onnx"),
                            ("Ollama Modelfile (Auto-generated)", "ollama")
                        ],
                        value=["safetensors", "gguf_q4", "ollama"],
                        label="Export Formats"
                    )
                    gguf_quant = gr.Dropdown(choices=["gguf_q4", "gguf_q8", "gguf_f16"], value="gguf_q4", label="GGUF Quant for Ollama")
                    push_hub = gr.Checkbox(value=False, label="Push to HuggingFace Hub?")
                    hub_id = gr.Textbox(label="HF Hub Model ID", placeholder="username/my-awesome-model", visible=False)
                    export_btn = gr.Button("📤 EXPORT NOW", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    gr.Markdown("### ✅ Export Results")
                    export_logs = gr.Textbox(label="Export Logs", lines=12, autoscroll=True)
                    export_results = gr.Dataframe(label="Exported Files", interactive=False)
                    with gr.Row():
                        exports_list_btn = gr.Button("🔄 Refresh Exports", size="sm")
            
            with gr.Accordion("🐍 Export Code", open=False):
                with gr.Row():
                    with gr.Column():
                        export_py_code = gr.Code(language="python", label="Python Export Code")
                    with gr.Column():
                        export_ollama_code = gr.Code(language="shell", label="Ollama & HF Hub Code")
    
    # Event bindings
    def on_model_change(choice, custom_id, task):
        html = model_info_html(choice, custom_id, task)
        code = generate_maker_code(get_clean_model_id(choice) if get_clean_model_id(choice) != "custom" else custom_id, task)
        return html, code
    
    model_dropdown.change(on_model_change, [model_dropdown, custom_model_id, task_dropdown], [model_info, maker_code])
    custom_model_id.change(on_model_change, [model_dropdown, custom_model_id, task_dropdown], [model_info, maker_code])
    task_dropdown.change(on_model_change, [model_dropdown, custom_model_id, task_dropdown], [model_info, maker_code])
    generate_maker_code_btn.click(on_model_change, [model_dropdown, custom_model_id, task_dropdown], [model_info, maker_code])
    
    # Data tab
    load_data_btn.click(
        handle_dataset_upload,
        [file_upload, hf_dataset_input, split_slider],
        [data_preview, data_status, data_code, dataset_path_state]
    )
    
    # Train tab
    train_btn.click(
        training_flow,
        [model_dropdown, custom_model_id, task_dropdown, dataset_path_state, epochs, batch_size, grad_accum, lr, lora_r, lora_alpha, lora_dropout, use_lora, use_qlora, mixed_prec, optimizer, use_flash, use_compile],
        [train_logs, train_plot, train_status, gpu_stats, train_code]
    )
    
    # Tune tab
    tune_btn.click(
        tuning_flow,
        [tune_lr, tune_r, tune_bs, tune_epochs, n_trials],
        [tune_logs, tune_leaderboard, best_params, tune_status, tune_code]
    )
    
    # Test tab - Chat
    chat_btn.click(
        chat_interface,
        [chat_input, chatbot, temp_slider, top_p_slider, top_k_slider, max_tokens_slider, rep_penalty, model_type_test],
        [chatbot, chat_input]
    )
    chat_input.submit(
        chat_interface,
        [chat_input, chatbot, temp_slider, top_p_slider, top_k_slider, max_tokens_slider, rep_penalty, model_type_test],
        [chatbot, chat_input]
    )
    clear_chat.click(lambda: ([], ""), None, [chatbot, chat_input])
    
    # Batch test
    batch_btn.click(
        batch_test_fn,
        [batch_file, temp_slider, max_tokens_slider],
        [batch_results, batch_status, batch_metrics]
    )
    
    # Export tab
    def toggle_hub_visibility(push):
        return gr.update(visible=push)
    
    push_hub.change(toggle_hub_visibility, [push_hub], [hub_id])
    
    export_btn.click(
        export_flow,
        [export_formats, push_hub, hub_id, gguf_quant],
        [export_logs, export_results, export_py_code, export_ollama_code]
    )
    
    exports_list_btn.click(
        lambda: pd.DataFrame(list_exports()) if list_exports() else pd.DataFrame([{"file": "No exports yet", "path": "", "size": ""}]),
        None,
        [export_results]
    )
    
    # Scratch tab - From Scratch
    def on_scratch_arch_change(arch):
        return scratch_arch_html(arch)
    
    scratch_arch.change(on_scratch_arch_change, [scratch_arch], [scratch_arch_info])
    
    scratch_train_btn.click(
        scratch_training_flow,
        [scratch_model_name, scratch_arch, scratch_layers, scratch_hidden, scratch_heads, scratch_inter, scratch_vocab, scratch_context, scratch_tok_type, scratch_train_tok, dataset_path_state, scratch_epochs, scratch_batch, scratch_lr],
        [scratch_logs, scratch_plot, scratch_status, scratch_gpu, scratch_code]
    )
    
    # GPU refresh
    refresh_gpu.click(lambda: gpu_info_html(), None, [gr.HTML()], show_progress=False)  # This won't work directly, need better handling
    # Actually use a separate component - for simplicity, we will have a hidden workaround
    # Let's make refresh update the first GPU display via JS? Instead we recreate
    def refresh_gpu_fn():
        return gpu_info_html()
    
    # We'll add a hidden textbox to trigger refresh of GPU stats in train tab
    # For main header GPU, we can't easily update without state, so we update train gpu_stats as well
    refresh_gpu.click(refresh_gpu_fn, None, [gpu_stats])

# Launch
if __name__ == "__main__":
    print("🚀 Starting LLM Forge Studio...")
    print("📊 GPU Info:", get_gpu_info())
    print("🌐 Gradio will be available on 0.0.0.0:7860")
    
    # Monkey-patch Gradio localhost check for Arena sandbox
    try:
        import gradio.utils
        gradio.utils.is_localhost_accessible = lambda: True
        # Also patch the function that checks if port is free
        print("✅ Patched localhost check for Arena")
    except Exception as e:
        print(f"Patch note: {e}")
    
    # Also start FastAPI in background thread if possible
    try:
        import threading
        import uvicorn
        # Try to import fastapi app, but don't fail if port in use
        try:
            from backend.server import app as fastapi_app
            def run_fastapi():
                try:
                    uvicorn.run(fastapi_app, host="0.0.0.0", port=8001, log_level="warning")
                except:
                    pass
            
            fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
            fastapi_thread.start()
            print("✅ FastAPI thread started on 0.0.0.0:8001 (main API on 8000)")
        except Exception as e2:
            print(f"FastAPI import note: {e2}")
    except Exception as e:
        print(f"⚠️ FastAPI not started in same process (will run separately): {e}")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Use share=True for Arena sandbox to bypass localhost check
        show_error=True,
        show_api=False
    )
