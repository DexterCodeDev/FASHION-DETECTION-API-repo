import os
import io
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

app = FastAPI()
MODEL_NAME = "yainage90/fashion-object-detection"

print("Loading DETR Fashion Model...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForObjectDetection.from_pretrained(MODEL_NAME)
model.eval() 

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 1. The safest resizing method for Hugging Face Transformers
        max_size = 800
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        with torch.no_grad():
            # 2. No brackets, explicitly passed as a single image
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            
            target_sizes = torch.tensor([[image.size[1], image.size[0]]])
            results = processor.post_process_object_detection(outputs, threshold=0.4, target_sizes=target_sizes)[0]
        
        boxes_list = []
        classes_list = []
        confidences_list = []
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            confidences_list.append(score.item())
            
            label_id = label.item()
            class_name = model.config.id2label.get(label_id, model.config.id2label.get(str(label_id), "Unknown"))
            
            classes_list.append(class_name)
            boxes_list.append([i.item() for i in box])
            
        return {
            "boxes": boxes_list, 
            "classes": classes_list, 
            "confidences": confidences_list
        }
        
    except Exception as e:
        # 3. THE TRACER BULLET
        raise HTTPException(status_code=500, detail=f"V3_CRASH: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
