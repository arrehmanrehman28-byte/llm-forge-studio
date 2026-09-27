"""
LLM Forge - FastAPI Backend Server
Provides API for inference and status
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os

# Import local modules
try:
    from .tester import generate_response, batch_inference, calculate_metrics
    from .utils import get_gpu_info
    from .exporter import export_model, list_exports
except ImportError:
    from tester import generate_response, batch_inference, calculate_metrics
    from utils import get_gpu_info
    from exporter import export_model, list_exports

app = FastAPI(
    title="LLM Forge Studio API",
    description="Highly Efficient LLM Maker, Trainer, Tester, Exporter",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 150
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    model_path: Optional[str] = "./checkpoints"

class BatchRequest(BaseModel):
    prompts: List[str]
    temperature: float = 0.7
    max_tokens: int = 100

class ExportRequest(BaseModel):
    formats: List[str] = ["safetensors"]
    push_to_hub: bool = False
    hub_model_id: str = "username/my-model"
    ollama_gguf: str = "model-gguf_q4.gguf"

@app.get("/")
def root():
    return {
        "name": "LLM Forge Studio API",
        "status": "🚀 Running",
        "version": "1.0.0",
        "endpoints": ["/api/generate", "/api/status", "/api/export", "/docs"]
    }

@app.get("/api/status")
def get_status():
    gpu_info = get_gpu_info()
    exports = list_exports()
    return {
        "gpu": gpu_info,
        "exports": exports,
        "checkpoints_exist": os.path.exists("./checkpoints/final_info.json"),
        "server": "FastAPI - LLM Forge"
    }

@app.post("/api/generate")
def api_generate(req: GenerateRequest):
    try:
        # For API, return non-streaming quick response
        from .tester import mock_generate
    except ImportError:
        from tester import mock_generate
    
    config = {
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "top_p": req.top_p,
        "top_k": req.top_k
    }
    
    # Generate
    response_text = mock_generate(req.prompt, config)
    
    return {
        "prompt": req.prompt,
        "response": response_text,
        "config": config,
        "model": req.model_path,
        "usage": {
            "prompt_tokens": len(req.prompt.split()),
            "completion_tokens": len(response_text.split()),
            "total_tokens": len(req.prompt.split()) + len(response_text.split())
        }
    }

@app.post("/api/batch")
def api_batch(req: BatchRequest):
    results = batch_inference(req.prompts, {"temperature": req.temperature, "max_tokens": req.max_tokens})
    return {"results": results, "count": len(results)}

@app.post("/api/export")
def api_export(req: ExportRequest):
    result = export_model(req.dict())
    return result

@app.get("/api/models")
def list_models():
    from .utils import MODEL_DB
    return {"models": MODEL_DB}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting LLM Forge FastAPI on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
