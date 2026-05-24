FROM pytorch/pytorch:2.1.0-cpu

WORKDIR /app

# Install only the libraries NOT included in the base image
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
