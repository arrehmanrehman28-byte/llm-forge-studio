# Contributing to LLM Forge Studio

Thank you for your interest in contributing! 🎉

## 🚀 Quick Start for Contributors

1. **Fork** the repo
2. **Clone** your fork:
   ```bash
   git clone https://github.com/yourusername/llm-forge-studio.git
   cd llm-forge-studio
   ```
3. **Create branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```
4. **Install dev dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install black ruff pytest
   ```
5. **Make changes & test**:
   ```bash
   python app.py  # Test UI on http://localhost:7860
   python backend/server.py  # Test API on http://localhost:8000/docs
   ```
6. **Commit & Push**:
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   git push origin feature/amazing-feature
   ```
7. **Open PR** on GitHub!

## 📋 Development Guidelines

### Code Style
- Use `black` for formatting: `black app.py backend/`
- Follow PEP8
- Add docstrings for functions
- Keep functions small and focused

### Commit Messages
We use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `style:` Formatting
- `refactor:` Code refactor
- `test:` Tests
- `chore:` Maintenance

Example: `feat: add Llama3 from-scratch architecture`

### Project Structure
```
llm-forge/
├── app.py              # Main Gradio UI
├── backend/
│   ├── server.py       # FastAPI backend
│   ├── trainer.py      # QLoRA trainer
│   ├── scratch.py      # From-scratch trainer
│   ├── tuner.py        # Optuna tuning
│   ├── tester.py       # Inference & chat
│   ├── exporter.py     # GGUF, ONNX, Ollama
│   └── utils.py        # GPU, dataset loader
├── examples/           # Example scripts
├── docs/               # Documentation
├── .github/            # GitHub templates & CI
├── requirements.txt
└── README.md
```

## 🎯 Good First Issues

Look for issues labeled `good first issue`:
- Add new model to dropdown
- Improve UI tooltips
- Add new export format
- Write example notebooks
- Improve docs

## 🧪 Testing

```bash
# Test imports
python -m py_compile app.py backend/*.py

# Test FastAPI
curl http://localhost:8000/

# Test Gradio
python app.py
# Open http://localhost:7860
```

## 💡 Feature Ideas

- [ ] Add Llama3, Mistral, Phi-3 from-scratch templates
- [ ] Support for vision models (LLaVA)
- [ ] Multi-GPU training
- [ ] Integration with Weights & Biases
- [ ] One-click deploy to HuggingFace Spaces
- [ ] Mobile app for testing models

## 📞 Questions?

- Open an issue
- Start a discussion
- Tag @maintainer

Thank you for contributing! 🙏
