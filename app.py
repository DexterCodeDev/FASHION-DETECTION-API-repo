from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import io
import os

app = FastAPI(title="Fashion Visual Search API")

MODEL_NAME = "yainage90/fashion-object-detection"

# Grab the token from Cloud Run's environment variables
HF_TOKEN = os.environ.get("HF_TOKEN")

# Fail instantly if the token is missing so we don't hit the 429 rate limit
if not HF_TOKEN:
    raise ValueError("🚨 MISSING TOKEN: Cloud Run could not find the HF_TOKEN environment variable!")

# Pass the token so Hugging Face allows the download
processor = AutoImageProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForObjectDetection.from_pretrained(MODEL_NAME, token=HF_TOKEN)

@app.get("/")
def health_check():
    return {"status": "healthy", "model": MODEL_NAME}

@app.post("/detect")
async def detect_fashion(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    inputs = processor(images=[image], return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    target_sizes = torch.tensor([[image.size[1], image.size[0]]])
    results = processor.post_process_object_detection(
        outputs, threshold=0.4, target_sizes=target_sizes
    )[0]
    
    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections.append({
            "label": model.config.id2label[label.item()],
            "confidence": round(score.item(), 3),
            "box": [round(coord, 2) for coord in box.tolist()]
        })
        
    return {"detections": detections}
