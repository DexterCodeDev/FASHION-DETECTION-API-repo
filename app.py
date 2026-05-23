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
    
    # Tell PyTorch to turn off gradient tracking to save RAM!
    with torch.no_grad():
        inputs = processor(images=[image], return_tensors="pt")
        outputs = model(**inputs)
        
        target_sizes = torch.tensor([[image.size[1], image.size[0]]])
        results = processor.post_process_object_detection(outputs, threshold=0.4, target_sizes=target_sizes)[0]
    
    boxes_list = []
    classes_list = []
    confidences_list = []
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        confidences_list.append(score.item())
        
        # THE FAIL-SAFE DICTIONARY LOOKUP
        label_id = label.item()
        # Try integer first, then try string, fallback to "Unknown" if it's completely missing
        class_name = model.config.id2label.get(label_id, model.config.id2label.get(str(label_id), "Unknown"))
        classes_list.append(class_name)
        
        boxes_list.append([i.item() for i in box])
        
    return {
        "boxes": boxes_list, 
        "classes": classes_list, 
        "confidences": confidences_list
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
