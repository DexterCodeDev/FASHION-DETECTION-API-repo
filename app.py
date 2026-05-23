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
model.eval() # Put the model in pure inference mode

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # MEMORY FIX: Automatically resize huge images
        max_size = 800
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        with torch.no_grad():
            # THE FIX: Added padding=True and formatted the image exactly as the docs requested!
            inputs = processor(images=[image], return_tensors="pt", padding=True)
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
        # Our beautiful error catcher!
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"AI Crash: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
