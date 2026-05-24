FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for image processing libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch first to save gigabytes of space
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Hugging Face model so Cloud Run cold starts are instant
RUN python -c "\
from transformers import AutoImageProcessor, AutoModelForObjectDetection; \
AutoImageProcessor.from_pretrained('yainage90/fashion-object-detection'); \
AutoModelForObjectDetection.from_pretrained('yainage90/fashion-object-detection')"

COPY . .

# Cloud Run injects the PORT environment variable
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
