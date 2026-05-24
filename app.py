from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import io

app = FastAPI(title="Fashion Visual Search API")

# Load the fine-tuned model directly from Hugging Face
MODEL_NAME = "yainage90/fashion-object-detection"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForObjectDetection.from_pretrained(MODEL_NAME)

@app.get("/")
def health_check():
    return {"status": "healthy", "model": MODEL_NAME}

@app.post("/detect")
async def detect_fashion(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Process image for the model
    inputs = processor(images=[image], return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Format and filter raw object detection results
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
