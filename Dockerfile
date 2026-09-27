FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt pyproject.toml ./
COPY backend/ ./backend/
COPY app.py ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gradio==4.44.1 huggingface_hub==0.25.2 fastapi uvicorn pandas plotly

# Copy rest of app
COPY . .

# Create necessary directories
RUN mkdir -p checkpoints exports scratch_models

# Expose ports
EXPOSE 7860 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Start both servers
CMD ["sh", "-c", "python backend/server.py & python app.py"]

# Labels for GitHub
LABEL org.opencontainers.image.title="LLM Forge Studio"
LABEL org.opencontainers.image.description="No-Code LLM Factory - From Scratch + Fine-Tuning"
LABEL org.opencontainers.image.source="https://github.com/yourusername/llm-forge-studio"
