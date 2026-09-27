"""
LLM Forge Studio - Fixed Version for Gradio 4.44.1
Bug-free, No-Code LLM Factory with From-Scratch support
"""
import os
import json
import time
import random
import pandas as pd
import gradio as gr

# Import backend
try:
    from backend.utils import get_gpu_info, get_model_info, load_dataset_auto, generate_maker_code, MODEL_DB
    from backend.trainer import train_model
    from backend.tuner import run_tuning, generate_tuning_code
    from backend.tester import generate_response, batch_inference, calculate_metrics, generate_api_code, mock_generate
    from backend.exporter import export_model, generate_export_code, list_exports
    from backend.scratch import train_from_scratch, generate_scratch_code, SCRATCH_ARCHS, get_arch_info
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
    from utils import get_gpu_info, get_model_info, load_dataset_auto, generate_maker_code, MODEL_DB
    from trainer import train_model
    from tuner import run_tuning, generate_tuning_code
    from tester import generate_response, batch_inference, calculate_metrics, generate_api_code, mock_generate
    from exporter import export_model, generate_export_code, list_exports
    from scratch import train_from_scratch, generate_scratch_code, SCRATCH_ARCHS, get_arch_info

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.gradio-container { background: #0a0a0b !important; color: #e4e4e7 !important; }
.header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 24px 32px; border-radius: 16px; margin-bottom: 24px; }
.header h1 { font-size: 32px !important; font-weight: 800 !important; color: white !important; margin: 0 !important; }
.header p { color: rgba(255,255,255,0.85) !important; font-size: 15px !important; margin: 8px 0 0 0 !important; }
.metric-card { background: #18181b !important; border: 1px solid #27272a !important; border-radius: 12px !important; padding: 16px !important; }
.primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; border: none !important; }
"""

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
    if info['params'] == "Unknown":
        info = {"params": "Custom", "size": "Varies", "vram": "Check HF", "type": task, "context": "Varies", "fast": False}
    recommended = "🔥 Recommended" if info.get('recommended') or info.get('fast') else ""
    return f"""
    <div class="metric-card" style="margin: 12px 0;">
        <div style="font-weight: 700; font-size: 16px; color: #e4e4e7; margin-bottom: 8px;">📦 {model_id}</div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 13px; color: #a1a1aa;">
            <div>Params: <b style="color: #e4e4e7;">{info['params']}</b></div>
            <div>Size: <b style="color: #e4e4e7;">{info['size']}</b></div>
            <div>VRAM: <b style="color: #10b981;">{info['vram']}</b></div>
            <div>Type: <b style="color: #e4e4e7;">{info['type']}</b></div>
        </div>
        <div style="margin-top: 10px; font-size: 12px; color: #a78bfa;">{recommended}</div>
    </div>
    """

def gpu_info_html():
    info = get_gpu_info()
    color = "#10b981" if info['has_gpu'] else "#f59e0b"
    return f"""
    <div style="background: #18181b; border: 1px solid #27272a; border-radius: 8px; padding: 12px; font-size: 13px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 8px; height: 8px; background: {color}; border-radius: 50%;"></div>
            <b style="color: #e4e4e7;">{info['available']}</b>
        </div>
        <div style="color: #a1a1aa; font-size: 12px; margin-top: 4px;">Torch: {info.get('torch_version', 'N/A')} | {info.get('gpu_name', 'CPU')}</div>
    </div>
    """

def handle_dataset_upload(file, hf_name, split_ratio):
    if file is None and not hf_name:
        return None, "⚠️ Upload file or enter HF dataset name", "", ""
    file_path = file.name if file else None
    df, msg = load_dataset_auto(file_path, hf_name, split_ratio)
    if df is None:
        return None, msg, "", ""
    if file_path:
        code = f'import pandas as pd\ndf = pd.read_csv("{os.path.basename(file_path)}")\nprint(df.head())'
    else:
        code = f'from datasets import load_dataset\nds = load_dataset("{hf_name}", split="train")\nprint(ds)'
    preview = df.head(20) if isinstance(df, pd.DataFrame) else df
    return preview, msg, code, file_path or hf_name

def training_flow(model_choice, custom_id, task, dataset_file_state, epochs, batch_size, grad_accum, lr, lora_r, lora_alpha, lora_dropout, use_lora, use_qlora, mixed_prec, optimizer, use_flash, use_compile):
    model_id = custom_id.strip() if get_clean_model_id(model_choice) == "custom" else get_clean_model_id(model_choice)
    if not model_id or model_id == "custom":
        yield "❌ Select valid model", None, "❌ Error", gpu_info_html(), "Select model first"
        return
    config = {
        "model_id": model_id, "task": task, "epochs": int(epochs), "batch_size": int(batch_size),
        "grad_accum": int(grad_accum), "lr": float(lr), "lora_r": int(lora_r), "lora_alpha": int(lora_alpha),
        "lora_dropout": float(lora_dropout), "use_lora": use_lora, "use_qlora": use_qlora,
        "mixed_precision": mixed_prec, "optimizer": optimizer, "use_flash": use_flash, "use_compile": use_compile,
        "dataset_path": dataset_file_state
    }
    from backend.utils import generate_trainer_code
    code = generate_trainer_code(config)
    logs = [f"🚀 Starting training for {model_id}", f"📊 Config: Epochs={epochs}, Batch={batch_size}, LR={lr}, LoRA r={lora_r}"]
    train_losses, val_losses, steps = [], [], []
    for update in train_model(config, dataset_file_state):
        metrics = update['metrics']
        logs.append(update['log'])
        train_losses.append(metrics['train_loss'])
        val_losses.append(metrics['val_loss'])
        steps.append(metrics['step'])
        plot_data = []
        for s, tl, vl in zip(steps, train_losses, val_losses):
            plot_data.append({"step": s, "loss": tl, "type": "train"})
            plot_data.append({"step": s, "loss": vl, "type": "val"})
        plot_df = pd.DataFrame(plot_data)
        status = f"🔥 Training... Step {metrics['step']}/{metrics['total_steps']} | Loss: {metrics['train_loss']} | ETA: {metrics['eta']}"
        gpu_html = f"<div style='background: #18181b; border: 1px solid #27272a; border-radius: 8px; padding: 12px; font-size: 12px;'><div>VRAM: <b style='color: #10b981;'>{metrics['gpu_mem']}</b> | Speed: {metrics['steps_per_sec']} steps/s | LR: {metrics['learning_rate']:.2e}</div></div>"
        yield "\n".join(logs[-50:]), plot_df, status, gpu_html, code
        if update['finished']:
            break
    final_status = f"✅ Training Complete! Final Loss: {train_losses[-1] if train_losses else 'N/A'}"
    logs.append(final_status)
    final_plot = pd.DataFrame([{"step": s, "loss": l, "type": "train"} for s, l in zip(steps, train_losses)]) if steps else pd.DataFrame([{"step":0,"loss":0,"type":"train"}])
    yield "\n".join(logs), final_plot, final_status, gpu_info_html(), code

def tuning_flow(tune_lr, tune_r, tune_bs, tune_epochs, n_trials):
    search_space = {"tune_lr": tune_lr, "tune_lora_r": tune_r, "tune_batch_size": tune_bs, "tune_epochs": tune_epochs}
    logs, trials_data = [], []
    best_loss, best_params = float('inf'), {}
    logs.append(f"🔍 Starting tuning with {n_trials} trials")
    for update in run_tuning(search_space, int(n_trials)):
        trial = update['trial']
        logs.append(update['log'])
        trials_data.append({"Trial": trial['trial'], "Loss": trial['loss'], "Val Loss": trial['val_loss'], "LR": trial['params'].get('lr', 'N/A'), "LoRA R": trial['params'].get('lora_r', 'N/A'), "Best?": "🔥" if trial['is_best'] else ""})
        if trial['is_best']:
            best_loss = trial['loss']
            best_params = update['best_params']
        df = pd.DataFrame(trials_data)
        status = f"🔍 Tuning... Trial {trial['trial']}/{n_trials} | Best Loss: {best_loss:.4f}"
        code = generate_tuning_code(search_space, int(n_trials))
        yield "\n".join(logs), df, json.dumps(best_params, indent=2), status, code
        if update['finished']:
            break
    logs.append(f"✅ Tuning Complete! Best Loss: {best_loss:.4f}")
    yield "\n".join(logs), pd.DataFrame(trials_data), json.dumps(best_params, indent=2), f"✅ Best Loss: {best_loss:.4f}", code

def chat_interface(message, history, temp, top_p, top_k, max_tokens, rep_penalty, model_type):
    if not message or not message.strip():
        return history, ""
    config = {"temperature": temp, "top_p": top_p, "top_k": top_k, "max_tokens": max_tokens, "repetition_penalty": rep_penalty, "model_type": model_type}
    full_response = ""
    for chunk in generate_response(message, config):
        full_response = chunk
        temp_history = history + [[message, full_response]]
        yield temp_history, ""
        time.sleep(0.02)
    history = history + [[message, full_response]]
    yield history, ""

def batch_test_fn(file, temp, max_tokens):
    if file is None:
        return None, "⚠️ Upload CSV", ""
    try:
        df = pd.read_csv(file.name)
        prompt_col = None
        for col in ['prompt', 'text', 'input', 'question', 'instruction']:
            if col in df.columns:
                prompt_col = col
                break
        if not prompt_col:
            prompt_col = df.columns[0]
        prompts = df[prompt_col].astype(str).tolist()[:20]
        config = {"temperature": temp, "max_tokens": max_tokens}
        results = batch_inference(prompts, config)
        results_df = pd.DataFrame(results)
        metrics = calculate_metrics([r['full_response'] for r in results])
        metrics_text = f"Perplexity: {metrics['perplexity']} | ROUGE-1: {metrics['rouge1']} | BLEU: {metrics['bleu']} | Latency: {metrics['avg_latency']}"
        return results_df, f"✅ Tested {len(results)} prompts", metrics_text
    except Exception as e:
        return None, f"❌ Error: {str(e)[:200]}", ""

def export_flow(formats, push_hub, hub_id, gguf_quant):
    if not formats:
        return "⚠️ Select format", None, "", ""
    config = {"formats": formats, "push_to_hub": push_hub, "hub_model_id": hub_id, "ollama_gguf": f"model-{gguf_quant}.gguf"}
    result = export_model(config)
    logs = "\n".join(result['logs'])
    df = pd.DataFrame(result['results'])
    codes = generate_export_code(config)
    return logs, df, codes['python'], codes['ollama'] + "\n\n" + codes['hf']

def scratch_arch_html(arch_choice):
    info = get_arch_info(arch_choice)
    return f"<div class='metric-card'><b>🧬 {arch_choice} - {info['params']}</b><br>Layers: {info['layers']} | Hidden: {info['hidden']} | Heads: {info['heads']} | Vocab: {info['vocab']}<br>{info['desc']}</div>"

def scratch_training_flow(model_name, arch_choice, num_layers, hidden_size, num_heads, inter_size, vocab_size, context_len, tokenizer_type, train_tokenizer, dataset_path_state, epochs, batch_size, lr):
    arch_info = get_arch_info(arch_choice)
    config = {
        "model_name": model_name or "my-scratch-llm", "architecture": arch_choice,
        "num_layers": int(num_layers) if arch_choice == "custom" else arch_info['layers'],
        "hidden_size": int(hidden_size) if arch_choice == "custom" else arch_info['hidden'],
        "num_heads": int(num_heads) if arch_choice == "custom" else arch_info['heads'],
        "intermediate_size": int(inter_size) if arch_choice == "custom" else arch_info['intermediate'],
        "vocab_size": int(vocab_size), "context_length": int(context_len),
        "tokenizer_type": tokenizer_type, "train_tokenizer": train_tokenizer,
        "epochs": int(epochs), "batch_size": int(batch_size), "lr": float(lr),
        "dataset_path": dataset_path_state
    }
    code = generate_scratch_code(config)
    logs = [f"🧬 Starting FROM SCRATCH: {config['model_name']}", f"📐 Arch: {config['num_layers']} layers, {config['hidden_size']} hidden"]
    train_losses, steps = [], []
    for update in train_from_scratch(config, dataset_path_state):
        if update.get('phase') == 'tokenizer':
            logs.append(update['log'])
            status = f"🔤 Tokenizer: {update['step']}/{update['total']} | {int(update['progress']*100)}%"
            plot_df = pd.DataFrame([{"step": update['step'], "loss": 0, "type": "tokenizer"}])
            yield "\n".join(logs[-50:]), plot_df, status, gpu_info_html(), code
        else:
            metrics = update['metrics']
            logs.append(update['log'])
            train_losses.append(metrics['train_loss'])
            steps.append(metrics['step'])
            plot_df = pd.DataFrame([{"step": s, "loss": l, "type": "scratch"} for s, l in zip(steps, train_losses)])
            status = f"🧬 Step {metrics['step']}/{metrics['total_steps']} | Loss: {metrics['train_loss']} | PPL: {metrics['perplexity']}"
            gpu_html = f"<div style='background: #18181b; border: 1px solid #27272a; border-radius: 8px; padding: 12px;'>VRAM: {metrics['gpu_mem']} | Tokens/s: {metrics['tokens_per_sec']}</div>"
            yield "\n".join(logs[-50:]), plot_df, status, gpu_html, code
            if update['finished']:
                break
    final_status = f"✅ From-Scratch Complete! {config['model_name']} | Loss: {train_losses[-1] if train_losses else 'N/A'}"
    logs.append(final_status)
    final_plot = pd.DataFrame([{"step": s, "loss": l, "type": "scratch"} for s, l in zip(steps, train_losses)]) if steps else pd.DataFrame([{"step":0,"loss":0,"type":"scratch"}])
    yield "\n".join(logs), final_plot, final_status, gpu_info_html(), code

# Build App
with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Monochrome(), title="LLM Forge Studio") as app:
    dataset_path_state = gr.State(None)
    
    gr.HTML("""
    <div class="header">
        <h1>⚡ LLM Forge Studio</h1>
        <p>No-Code • Python-Friendly • From Scratch + Fine-Tuning • Highly Efficient</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML(gpu_info_html())
        with gr.Column(scale=1):
            refresh_gpu = gr.Button("🔄 Refresh GPU", size="sm")
    
    with gr.Tabs():
        with gr.Tab("🏗️ Maker"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 🎯 Choose Base Model (Fine-Tuning)")
                    model_dropdown = gr.Dropdown(choices=MODEL_CHOICES, value="Qwen/Qwen2-0.5B-Instruct (Recommended - Fastest)", label="Base Model")
                    custom_model_id = gr.Textbox(label="Custom HF Model ID", placeholder="e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                    task_dropdown = gr.Dropdown(choices=["Text Generation", "Chatbot", "Instruction Tuning", "Summarization", "Classification", "Code Generation"], value="Chatbot", label="Task")
                    model_info = gr.HTML("<div class='metric-card'>Select model</div>")
                    with gr.Accordion("🐍 Python Code", open=False):
                        maker_code = gr.Code(language="python", label="Code")
                        generate_maker_code_btn = gr.Button("Generate Code", size="sm")
                with gr.Column(scale=1):
                    gr.Markdown("### ⚡ Quick Start")
                    gr.HTML("<div class='metric-card'>1. Maker: Pick model<br>2. Data: Upload CSV<br>3. Train: QLoRA<br>4. Test: Chat<br>5. Export: GGUF/Ollama</div>")
        
        with gr.Tab("🧬 From Scratch"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🧬 Create NEW Model From Scratch")
                    gr.HTML("<div style='background: #667eea20; border: 1px solid #667eea50; border-radius: 8px; padding: 12px; font-size: 12px;'><b style='color: #a78bfa;'>FROM SCRATCH = Brand new LLM, no pretrained weights!</b></div>")
                    scratch_model_name = gr.Textbox(label="Model Name", value="my-scratch-llm")
                    scratch_arch = gr.Dropdown(choices=["tiny-10M", "small-50M", "base-124M", "medium-350M", "custom"], value="tiny-10M", label="Architecture")
                    scratch_arch_info = gr.HTML("<div class='metric-card'>Select arch</div>")
                    with gr.Group():
                        gr.Markdown("**Custom Architecture**")
                        scratch_layers = gr.Slider(2, 32, value=8, step=1, label="Num Layers")
                        scratch_hidden = gr.Slider(128, 2048, value=512, step=64, label="Hidden Size")
                        scratch_heads = gr.Slider(2, 32, value=8, step=1, label="Heads")
                        scratch_inter = gr.Slider(512, 8192, value=2048, step=128, label="Intermediate Size")
                    with gr.Group():
                        gr.Markdown("**Tokenizer**")
                        scratch_vocab = gr.Slider(1000, 50000, value=16000, step=1000, label="Vocab Size")
                        scratch_context = gr.Slider(128, 4096, value=1024, step=128, label="Context Length")
                        scratch_tok_type = gr.Dropdown(choices=["BPE", "WordLevel", "Char", "ByteLevel"], value="BPE", label="Tokenizer Type")
                        scratch_train_tok = gr.Checkbox(value=True, label="Train Tokenizer on Dataset")
                    with gr.Group():
                        gr.Markdown("**Training**")
                        scratch_epochs = gr.Slider(1, 10, value=1, step=1, label="Epochs")
                        scratch_batch = gr.Slider(1, 32, value=4, step=1, label="Batch Size")
                        scratch_lr = gr.Dropdown(choices=["1e-4", "3e-4", "5e-4", "1e-3"], value="3e-4", label="LR")
                    scratch_train_btn = gr.Button("🧬 START FROM SCRATCH", variant="primary", size="lg")
                with gr.Column(scale=2):
                    gr.Markdown("### 📈 Training Dashboard")
                    scratch_status = gr.Textbox(label="Status", value="Ready...")
                    scratch_plot = gr.LinePlot(x="step", y="loss", color="type", title="Loss", height=300)
                    scratch_gpu = gr.HTML(gpu_info_html())
                    scratch_logs = gr.Textbox(label="Logs", lines=15, autoscroll=True)
            with gr.Accordion("🐍 Code - From Scratch", open=False):
                scratch_code = gr.Code(language="python", label="Code")
        
        with gr.Tab("📊 Data"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📁 Upload Dataset")
                    file_upload = gr.File(label="CSV, JSONL, JSON, TXT", file_types=[".csv", ".jsonl", ".json", ".txt"])
                    hf_dataset_input = gr.Textbox(label="Or HF Dataset", placeholder="yahma/alpaca-cleaned")
                    split_slider = gr.Slider(0.05, 0.5, value=0.2, step=0.05, label="Val Split")
                    load_data_btn = gr.Button("📥 Load & Preview", variant="primary")
                with gr.Column():
                    gr.Markdown("### 👀 Preview")
                    data_status = gr.Textbox(label="Status", lines=2)
                    data_preview = gr.Dataframe(label="Preview", wrap=True)
            with gr.Accordion("🐍 Code - Data", open=False):
                data_code = gr.Code(language="python", label="Code")
        
        with gr.Tab("🚀 Train"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Training Config")
                    with gr.Group():
                        gr.Markdown("**LoRA / QLoRA**")
                        use_lora = gr.Checkbox(value=True, label="Enable LoRA")
                        use_qlora = gr.Checkbox(value=True, label="Enable QLoRA 4-bit")
                        lora_r = gr.Slider(4, 128, value=16, step=4, label="LoRA Rank")
                        lora_alpha = gr.Slider(8, 128, value=32, step=8, label="LoRA Alpha")
                        lora_dropout = gr.Slider(0.0, 0.5, value=0.05, step=0.05, label="LoRA Dropout")
                    with gr.Group():
                        gr.Markdown("**Hyperparams**")
                        epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                        batch_size = gr.Slider(1, 32, value=4, step=1, label="Batch Size")
                        grad_accum = gr.Slider(1, 16, value=4, step=1, label="Grad Accum")
                        lr = gr.Dropdown(choices=["1e-5", "2e-5", "5e-5", "1e-4", "2e-4", "3e-4", "5e-4"], value="2e-4", label="LR")
                        mixed_prec = gr.Dropdown(choices=["fp16", "bf16", "fp32"], value="fp16", label="Mixed Precision")
                        optimizer = gr.Dropdown(choices=["paged_adamw_8bit", "adamw_torch", "adamw_8bit"], value="paged_adamw_8bit", label="Optimizer")
                    with gr.Group():
                        gr.Markdown("**Speed**")
                        use_flash = gr.Checkbox(value=True, label="Flash Attention 2")
                        use_compile = gr.Checkbox(value=True, label="torch.compile")
                    train_btn = gr.Button("🚀 START TRAINING", variant="primary", size="lg")
                with gr.Column(scale=2):
                    gr.Markdown("### 📈 Live Dashboard")
                    train_status = gr.Textbox(label="Status", value="Ready...")
                    train_plot = gr.LinePlot(x="step", y="loss", color="type", title="Loss", height=300)
                    gpu_stats = gr.HTML(gpu_info_html())
                    train_logs = gr.Textbox(label="Logs", lines=15, autoscroll=True)
            with gr.Accordion("🐍 Code - Training", open=False):
                train_code = gr.Code(language="python", label="Code")
        
        with gr.Tab("🎛️ Tune"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🔍 Auto Tuning")
                    tune_lr = gr.Checkbox(value=True, label="Tune LR")
                    tune_r = gr.Checkbox(value=True, label="Tune LoRA Rank")
                    tune_bs = gr.Checkbox(value=False, label="Tune Batch Size")
                    tune_epochs = gr.Checkbox(value=False, label="Tune Epochs")
                    n_trials = gr.Slider(3, 50, value=10, step=1, label="Trials")
                    tune_btn = gr.Button("🔍 Start Tuning", variant="primary")
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Results")
                    tune_status = gr.Textbox(label="Status")
                    tune_leaderboard = gr.Dataframe(label="Leaderboard")
                    best_params = gr.Code(language="json", label="Best Params")
                    tune_logs = gr.Textbox(label="Logs", lines=10, autoscroll=True)
            with gr.Accordion("🐍 Code - Tuning", open=False):
                tune_code = gr.Code(language="python", label="Code")
        
        with gr.Tab("🧪 Test"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 💬 Chat Playground")
                    chatbot = gr.Chatbot(label="Chat", height=400, show_copy_button=True)
                    chat_input = gr.Textbox(label="Message", placeholder="Ask anything...")
                    with gr.Row():
                        chat_btn = gr.Button("Send", variant="primary")
                        clear_chat = gr.Button("Clear", size="sm")
                with gr.Column(scale=1):
                    gr.Markdown("### 🎛️ Config")
                    model_type_test = gr.Dropdown(choices=["Fine-Tuned Model", "Base Model", "From Scratch Model"], value="Fine-Tuned Model", label="Model")
                    temp_slider = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                    top_p_slider = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P")
                    top_k_slider = gr.Slider(1, 100, value=40, step=1, label="Top-K")
                    max_tokens_slider = gr.Slider(10, 512, value=150, step=10, label="Max Tokens")
                    rep_penalty = gr.Slider(1.0, 2.0, value=1.1, step=0.1, label="Rep Penalty")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📦 Batch Testing")
                    batch_file = gr.File(label="CSV with prompts", file_types=[".csv"])
                    batch_btn = gr.Button("🧪 Run Batch")
                    batch_status = gr.Textbox(label="Status")
                    batch_results = gr.Dataframe(label="Results", wrap=True)
                    batch_metrics = gr.Markdown("Metrics")
            with gr.Accordion("🔌 API", open=True):
                with gr.Row():
                    with gr.Column():
                        api_code = gr.Code(language="python", value=generate_api_code()["python"], label="Python API")
                    with gr.Column():
                        api_curl = gr.Code(language="python", value=generate_api_code()["curl"], label="cURL")
        
        with gr.Tab("📤 Export"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Export")
                    export_formats = gr.CheckboxGroup(choices=[("SafeTensors", "safetensors"), ("PyTorch", "pytorch"), ("GGUF Q4_K_M", "gguf_q4"), ("GGUF Q8_0", "gguf_q8"), ("GGUF F16", "gguf_f16"), ("ONNX", "onnx"), ("Ollama", "ollama")], value=["safetensors", "gguf_q4", "ollama"], label="Formats")
                    gguf_quant = gr.Dropdown(choices=["gguf_q4", "gguf_q8", "gguf_f16"], value="gguf_q4", label="GGUF Quant")
                    push_hub = gr.Checkbox(value=False, label="Push to HF Hub?")
                    hub_id = gr.Textbox(label="HF Model ID", placeholder="username/my-model", visible=False)
                    export_btn = gr.Button("📤 EXPORT NOW", variant="primary", size="lg")
                with gr.Column(scale=2):
                    gr.Markdown("### ✅ Results")
                    export_logs = gr.Textbox(label="Logs", lines=12, autoscroll=True)
                    export_results = gr.Dataframe(label="Files")
                    exports_list_btn = gr.Button("🔄 Refresh", size="sm")
            with gr.Accordion("🐍 Export Code", open=False):
                with gr.Row():
                    with gr.Column():
                        export_py_code = gr.Code(language="python", label="Python")
                    with gr.Column():
                        export_ollama_code = gr.Code(language="python", label="Ollama & HF")

    # Events
    def on_model_change(choice, custom_id, task):
        html = model_info_html(choice, custom_id, task)
        code = generate_maker_code(get_clean_model_id(choice) if get_clean_model_id(choice) != "custom" else custom_id, task)
        return html, code
    model_dropdown.change(on_model_change, [model_dropdown, custom_model_id, task_dropdown], [model_info, maker_code])
    custom_model_id.change(on_model_change, [model_dropdown, custom_model_id, task_dropdown], [model_info, maker_code])
    task_dropdown.change(on_model_change, [model_dropdown, custom_model_id, task_dropdown], [model_info, maker_code])
    generate_maker_code_btn.click(on_model_change, [model_dropdown, custom_model_id, task_dropdown], [model_info, maker_code])
    
    def on_scratch_arch_change(arch):
        return scratch_arch_html(arch)
    scratch_arch.change(on_scratch_arch_change, [scratch_arch], [scratch_arch_info])
    scratch_train_btn.click(scratch_training_flow, [scratch_model_name, scratch_arch, scratch_layers, scratch_hidden, scratch_heads, scratch_inter, scratch_vocab, scratch_context, scratch_tok_type, scratch_train_tok, dataset_path_state, scratch_epochs, scratch_batch, scratch_lr], [scratch_logs, scratch_plot, scratch_status, scratch_gpu, scratch_code])
    
    load_data_btn.click(handle_dataset_upload, [file_upload, hf_dataset_input, split_slider], [data_preview, data_status, data_code, dataset_path_state])
    train_btn.click(training_flow, [model_dropdown, custom_model_id, task_dropdown, dataset_path_state, epochs, batch_size, grad_accum, lr, lora_r, lora_alpha, lora_dropout, use_lora, use_qlora, mixed_prec, optimizer, use_flash, use_compile], [train_logs, train_plot, train_status, gpu_stats, train_code])
    tune_btn.click(tuning_flow, [tune_lr, tune_r, tune_bs, tune_epochs, n_trials], [tune_logs, tune_leaderboard, best_params, tune_status, tune_code])
    chat_btn.click(chat_interface, [chat_input, chatbot, temp_slider, top_p_slider, top_k_slider, max_tokens_slider, rep_penalty, model_type_test], [chatbot, chat_input])
    chat_input.submit(chat_interface, [chat_input, chatbot, temp_slider, top_p_slider, top_k_slider, max_tokens_slider, rep_penalty, model_type_test], [chatbot, chat_input])
    clear_chat.click(lambda: ([], ""), None, [chatbot, chat_input])
    batch_btn.click(batch_test_fn, [batch_file, temp_slider, max_tokens_slider], [batch_results, batch_status, batch_metrics])
    def toggle_hub_visibility(push):
        return gr.update(visible=push)
    push_hub.change(toggle_hub_visibility, [push_hub], [hub_id])
    export_btn.click(export_flow, [export_formats, push_hub, hub_id, gguf_quant], [export_logs, export_results, export_py_code, export_ollama_code])
    exports_list_btn.click(lambda: pd.DataFrame(list_exports()) if list_exports() else pd.DataFrame([{"file": "No exports yet"}]), None, [export_results])
    def refresh_gpu_fn():
        return gpu_info_html()
    refresh_gpu.click(refresh_gpu_fn, None, [gpu_stats])

if __name__ == "__main__":
    print("🚀 Starting LLM Forge Studio - Fixed Version...")
    print("📊 GPU:", get_gpu_info())
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
