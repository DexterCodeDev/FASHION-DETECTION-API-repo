FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Accept the token as a build argument and set it as an environment variable
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

# Pre-download the Hugging Face model directly into the Docker image
RUN python -c "from transformers import AutoImageProcessor, AutoModelForObjectDetection; \
               AutoImageProcessor.from_pretrained('yainage90/fashion-object-detection'); \
               AutoModelForObjectDetection.from_pretrained('yainage90/fashion-object-detection')"

# Copy application files
COPY app.py .

ENV PORT=8080
EXPOSE 8080

# Run Uvicorn production server
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
