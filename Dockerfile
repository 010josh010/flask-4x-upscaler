FROM python:3.12-slim

# Build argument to switch between GPU and CPU PyTorch
ARG COMPUTE_MODE=gpu

WORKDIR /app

# Install system dependencies for OpenCV, image processing, git (for basicsr), and curl (for models)
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch first based on COMPUTE_MODE
RUN if [ "$COMPUTE_MODE" = "gpu" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
        torch torchvision torchaudio; \
    else \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision torchaudio; \
    fi

# Copy requirements and install remaining dependencies
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Patch RealESRGAN GPU bug (line 48 of utils.py)
# Bug: "if gpu_id:" evaluates to False when gpu_id=0 (single GPU), defaulting to CPU
# Fix: Change to "if gpu_id is not None:" so gpu_id=0 is truthy
RUN UTILS_PATH=$(python -c "import realesrgan; import os; print(os.path.join(os.path.dirname(realesrgan.__file__), 'utils.py'))") && \
    sed -i 's/if gpu_id:/if gpu_id is not None:/' "$UTILS_PATH"

# Copy application code
COPY app.py upscaler.py ./
COPY templates/ templates/
COPY static/ static/

# Create directories for runtime data
RUN mkdir -p uploads outputs weights

# Download Real-ESRGAN model weights
RUN curl -L -o weights/RealESRGAN_x4plus.pth \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" && \
    curl -L -o weights/RealESRGAN_x4plus_anime_6B.pth \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"

# Expose Flask port
EXPOSE 5000

# Run Flask (bind to 0.0.0.0 for Docker networking)
ENV FLASK_APP=app.py
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
