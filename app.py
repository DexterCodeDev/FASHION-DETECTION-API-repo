from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import io
import os
import urllib.request

app = FastAPI(title="Fashion Visual Search API")

# Local folder to store the downloaded files inside the container
MODEL_DIR = "./local_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Direct unauthenticated Hugging Face resolve URLs
MODEL_URLS = {
    "config.json": "https://huggingface.co/yainage90/fashion-object-detection/resolve/main/config.json",
    "preprocessor_config.json": "https://huggingface.co/yainage90/fashion-object-detection/resolve/main/preprocessor_config.json",
    "model.safetensors": "https://huggingface.co/yainage90/fashion-object-detection/resolve/main/model.safetensors"
}

processor = None
model = None

@app.on_event("startup")
async def load_model():
    global processor, model
    
    try:
        for filename, url in MODEL_URLS.items():
            destination = os.path.join(MODEL_DIR, filename)
            
            if not os.path.exists(destination):
                print(f"Trying direct unauthenticated download for {filename}...")
                urllib.request.urlretrieve(url, destination)
                print(f"Successfully downloaded {filename}!")
                
        print("Loading model into memory from local storage...")
        processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
        model = AutoModelForObjectDetection.from_pretrained(MODEL_DIR)
        print("Model is fully loaded and ready!")
        
    except Exception as e:
        print(f"🚨 Download failed or was rate-limited! Error: {e}")
        # We don't raise the error here so the container stays alive on port 8080 
        # allowing you to check the logs cleanly.

@app.get("/")
def health_check():
    if model is not None:
        return {"status": "healthy", "method": "Direct HF URL (Success)"}
    return {"status": "starting up or download failed", "check_logs": "Look at Logs Explorer for errors"}

@app.post("/detect")
async def detect_fashion(file: UploadFile = File(...)):
    global processor, model
    if processor is None or model is None:
        return {"error": "Model is not loaded. The download might have been blocked or is still running."}
        
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
