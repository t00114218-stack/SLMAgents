FROM python:3.10-slim

# Install system dependencies (git, compilers, and OpenMP runtime for ONNX Runtime CPU)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements file first to leverage Docker build cache
COPY requirements.txt .

# Install dependencies, prioritizing CPU-only torch to keep image size small
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu


# Copy the rest of the monorepo codebase
COPY . .

# Pre-download verified ONNX models into the Docker image filesystem at build time
RUN python3 download_models.py

# Expose Hugging Face Space port
EXPOSE 7860

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
