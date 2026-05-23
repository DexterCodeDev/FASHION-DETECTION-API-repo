FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Tell Hugging Face to save the model data inside the /app folder
ENV HF_HOME=/app/models

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The Bake-In Trick: Download the model from Hugging Face during the build
RUN python -c "from transformers import AutoImageProcessor, AutoModelForObjectDetection; \
AutoImageProcessor.from_pretrained('yainage90/fashion-object-detection'); \
AutoModelForObjectDetection.from_pretrained('yainage90/fashion-object-detection')"

COPY . .

CMD ["python", "app.py"]
