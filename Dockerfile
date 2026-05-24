FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download the Hugging Face model directly into the Docker image
# This prevents Cloud Run from timing out on cold starts
RUN python -c "from transformers import AutoImageProcessor, AutoModelForObjectDetection; \
               AutoImageProcessor.from_pretrained('yainage90/fashion-object-detection'); \
               AutoModelForObjectDetection.from_pretrained('yainage90/fashion-object-detection')"

COPY app.py .

ENV PORT=8080
EXPOSE 8080

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
