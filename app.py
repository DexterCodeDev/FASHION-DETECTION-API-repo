import os
import io
import torch
import numpy as np
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
        
        max_size = 800
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        with torch.no_grad():
            # 1. REMOVED return_tensors="pt" to bypass the Hugging Face bug!
            inputs = processor(images=image)
            
            # 2. THE ULTIMATE BYPASS: Manually convert the raw math to PyTorch tensors
            pixel_values = torch.from_numpy(np.array(inputs["pixel_values"]))
            model_inputs = {"pixel_values": pixel_values}
            
            if "pixel_mask" in inputs:
                model_inputs["pixel_mask"] = torch.from_numpy(np.array(inputs["pixel_mask"]))
            
            # 3. Run the model using our manually created, bug-free tensors
            outputs = model(**model_inputs)
            
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
        # V4 Tracer Bullet just in case!
        raise HTTPException(status_code=500, detail=f"V4_CRASH: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
