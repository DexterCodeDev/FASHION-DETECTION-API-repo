import os
import io
import torch
from fastapi import FastAPI, UploadFile, File
import uvicorn
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

app = FastAPI()

MODEL_NAME = "yainage90/fashion-object-detection"

# 1. Load the Hugging Face Processor and Model
print("Loading DETR Fashion Model...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForObjectDetection.from_pretrained(MODEL_NAME)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image using Pillow (instead of OpenCV)
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 2. Run AI Inference
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    
    # 3. Post-process the bounding boxes to match the image size
    # We are setting a confidence threshold of 50% (0.5)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
    
    boxes = results["boxes"].tolist()
    scores = results["scores"].tolist()
    labels = results["labels"].tolist()
    
    # 4. Map the AI's internal ID numbers back to human-readable fashion words
    class_names = [model.config.id2label[label] for label in labels]
    
    return {
        "boxes": boxes, 
        "classes": class_names, 
        "confidences": scores
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
