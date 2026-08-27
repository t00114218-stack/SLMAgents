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

# Set environment variables for 2 vCPU SIMD / AVX Dynamic Batching on Hugging Face Spaces
ENV OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    SLM_N_THREADS=2 \
    OMP_WAIT_POLICY=PASSIVE \
    KMP_BLOCKTIME=0 \
    ORT_ENABLE_AVX2=1 \
    SLM_MAX_BATCH_SIZE=16 \
    SLM_BATCH_TIMEOUT_MS=3.0

# Expose Hugging Face Space port
EXPOSE 7860

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
