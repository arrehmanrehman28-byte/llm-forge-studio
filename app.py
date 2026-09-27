# LLM Forge Studio - Ultra Robust for Gradio 6.28.0 + Python 3.13 Colab
# Guaranteed to launch even if backend fails

import os
import sys

# Compatibility patches - safe for Gradio 4,5,6
try:
    import gradio_client.utils as client_utils
    orig_func = client_utils._json_schema_to_python_type
    orig_get = client_utils.get_type
    def patched_get(schema):
        if isinstance(schema, bool): return "Any"
        try: return orig_get(schema)
        except: return "Any"
    def patched_json(schema, defs=None):
        if isinstance(schema, bool): return "Any"
        if isinstance(schema, dict) and "additionalProperties" in schema and isinstance(schema["additionalProperties"], bool):
            schema = dict(schema)
            schema["additionalProperties"] = {"type": "object"} if schema["additionalProperties"] else {}
            if not schema["additionalProperties"]:
                del schema["additionalProperties"]
        try: return orig_func(schema, defs)
        except: return "Any"
    client_utils.get_type = patched_get
    client_utils._json_schema_to_python_type = patched_json
except: pass

try:
    import gradio.networking
    gradio.networking.is_local_port_accessible = lambda *a, **k: True
except: pass

try:
    import gradio.utils
    if hasattr(gradio.utils, 'is_localhost_accessible'): gradio.utils.is_localhost_accessible = lambda: True
    if hasattr(gradio.utils, 'is_port_accessible'): gradio.utils.is_port_accessible = lambda *a, **k: True
except: pass

import gradio as gr
import pandas as pd

print(f"Gradio version: {gr.__version__}")

# Try to import backend, but don't fail if it doesn't work
BACKEND_OK = False
BACKEND_ERROR = None
try:
    from backend.utils import get_gpu_info, get_model_info, load_dataset_auto, generate_maker_code, MODEL_DB
    from backend.trainer import train_model
    from backend.efficiency import calculate_efficiency_gains
    BACKEND_OK = True
    print(f"✅ Backend OK: {len(MODEL_DB)} models")
except Exception as e:
    BACKEND_ERROR = str(e)
    print(f"⚠️ Backend import failed: {e}")
    # Create dummy data so UI still works
    MODEL_DB = {
        "Qwen2-0.5B": {"id": "Qwen/Qwen2-0.5B-Instruct", "vram": "1.7GB", "desc": "Fastest"},
        "TinyLlama": {"id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "vram": "2.5GB", "desc": "Good chatbot"}
    }
    def get_gpu_info(): return "GPU info not available in fallback mode"
    def get_model_info(x): return "Model info not available"
    def load_dataset_auto(x,y,z): return None, "Fallback mode", None
    def generate_maker_code(x): return "# Fallback code"
    def train_model(*args, **kwargs):
        yield {"step": 0, "metrics": {"train_loss": 0, "val_loss": 0}, "finished": True}
    def calculate_efficiency_gains(x): return {"vram_usage": "1.7GB", "vram_saved": "93%", "speed_multiplier": "1.6x", "quality_boost": "+6%"}

# CSS
CSS = """
.header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; color: white; margin-bottom: 20px; }
.card { background: #1a1a1a; padding: 20px; border-radius: 15px; border: 1px solid #333; margin-bottom: 15px; }
.card-green { background: #10b98120; border: 2px solid #10b981; padding: 20px; border-radius: 15px; margin-bottom: 15px; }
"""

def is_colab():
    try:
        import google.colab
        return True
    except:
        return False

# Models
MODELS = [
    "⚡ Qwen 0.5B - Fastest, Works on Any Computer (Recommended)",
    "💬 TinyLlama 1.1B - Good Chatbot",
    "🧠 Phi-2 2.7B - Powerful, Needs 8GB VRAM",
]

# UI - Gradio 6 compatible (theme/css in launch, no show_copy_button)
with gr.Blocks(title="LLM Forge - Easy LLM Builder") as app:
    gr.HTML(f"""
    <div class="header">
        <h1>⚡ LLM Forge Studio</h1>
        <p>Build Your Own AI in 5 Minutes - No Coding! Gradio {gr.__version__} - {'✅ Backend OK' if BACKEND_OK else f'⚠️ Fallback Mode: {BACKEND_ERROR[:100]}'}</p>
    </div>
    """)
    
    if not BACKEND_OK:
        gr.HTML(f"""
        <div class="card" style="border: 2px solid #f59e0b;">
            <h3>⚠️ Fallback Mode</h3>
            <p>Backend failed: {BACKEND_ERROR}</p>
            <p>Full features need: pip install transformers peft trl accelerate datasets torch</p>
            <p>But UI still works! Try Test tab.</p>
        </div>
        """)
    
    with gr.Tabs():
        with gr.Tab("👋 Welcome"):
            gr.Markdown("""
            ### 🎯 3 Easy Steps!
            1. **Choose Model** - Qwen 0.5B fastest
            2. **Add Data** - Use sample data button
            3. **Train** - Click TRAIN MY AI NOW!
            
            Works in Colab, local, any computer!
            """)
            gr.HTML(f"<div class='card-green'>✅ Gradio {gr.__version__} - Python 3.13 Compatible - Ready!</div>")
        
        with gr.Tab("🧪 Test - Chat"):
            chatbot = gr.Chatbot(label="Your AI Chat", height=400)
            msg = gr.Textbox(label="Type message", placeholder="Hello!")
            clear = gr.Button("Clear")
            
            def respond(message, history):
                history = history or []
                if not BACKEND_OK:
                    history.append((message, f"Fallback mode: Echo: {message}. Backend error: {BACKEND_ERROR[:100]}. Install dependencies for full AI!"))
                else:
                    history.append((message, f"AI response to: {message} (Install transformers for real AI)"))
                return history, ""
            
            msg.submit(respond, [msg, chatbot], [chatbot, msg])
            clear.click(lambda: (None, ""), None, [chatbot, msg])
        
        with gr.Tab("📊 Status"):
            gr.Markdown(f"""
            - **Gradio:** {gr.__version__} ✅
            - **Backend:** {'✅ OK' if BACKEND_OK else f'❌ {BACKEND_ERROR[:200]}'}
            - **Python:** {sys.version.split()[0]}
            - **Colab:** {is_colab()}
            """)
            if not BACKEND_OK:
                gr.Markdown(f"**Fix:** `pip install transformers peft trl accelerate datasets torch`")

# Launch with theme/css in launch() for Gradio 6 compatibility
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=is_colab(), show_error=True, debug=True, theme=gr.themes.Monochrome(), css=CSS)
else:
    # For import in Colab
    pass
