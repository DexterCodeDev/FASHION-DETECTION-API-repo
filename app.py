import os
import io
import torch
from fastapi import FastAPI, UploadFile, File
import uvicorn
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

app = FastAPI()

MODEL_NAME = "yainage90/fashion-object-detection"

print("Loading DETR Fashion Model...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForObjectDetection.from_pretrained(MODEL_NAME)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # THE FIX: Tell PyTorch to turn off gradient tracking to save RAM!
    with torch.no_grad():
        inputs = processor(images=[image], return_tensors="pt")
        outputs = model(**inputs)
        
        # The exact target size format from the documentation
        target_sizes = torch.tensor([[image.size[1], image.size[0]]])
        
        # Using the author's recommended 0.4 threshold
        results = processor.post_process_object_detection(outputs, threshold=0.4, target_sizes=target_sizes)[0]
    
    boxes_list = []
    classes_list = []
    confidences_list = []
    
    # The exact data extraction loop from the documentation
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        confidences_list.append(score.item())
        classes_list.append(model.config.id2label[label.item()])
        boxes_list.append([i.item() for i in box])
        
    return {
        "boxes": boxes_list, 
        "classes": classes_list, 
        "confidences": confidences_list
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
