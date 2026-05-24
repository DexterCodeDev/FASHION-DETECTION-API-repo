FROM python:3.10-slim

# Install required system packages
RUN apt-get update && apt-get install -y git && apt-get clean

WORKDIR /app

# Install Python dependencies first (cached in build)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Cloud Run uses port 8080
EXPOSE 8080

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
