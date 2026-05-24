from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import io
import os

app = FastAPI(title="Fashion Visual Search API")

MODEL_NAME = "yainage90/fashion-object-detection"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Define these as global variables so we can load them after the server starts
processor = None
model = None

@app.on_event("startup")
async def load_model():
    """This runs AFTER port 8080 is open, preventing Cloud Run timeouts!"""
    global processor, model
    if not HF_TOKEN:
        print("🚨 WARNING: HF_TOKEN environment variable is missing!")
    
    print("Downloading and loading model from Hugging Face...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForObjectDetection.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    print("Model loaded successfully!")

@app.get("/")
def health_check():
    # If the model isn't loaded yet, tell the health checker it's starting
    status = "healthy" if model is not None else "starting up"
    return {"status": status, "model": MODEL_NAME}

@app.post("/detect")
async def detect_fashion(file: UploadFile = File(...)):
    global processor, model
    
    # Catch any requests that come in before the download finishes
    if processor is None or model is None:
        return {"error": "Model is still downloading into the container. Try again in a few seconds!"}
        
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
