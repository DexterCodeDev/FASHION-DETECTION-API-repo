from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import io

app = FastAPI(title="Fashion Object Detection API")

# Define the model checkpoint
ckpt = 'yainage90/fashion-object-detection'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model globally so it only loads once per instance boot
print("Loading model and processor...")
processor = AutoImageProcessor.from_pretrained(ckpt)
model = AutoModelForObjectDetection.from_pretrained(ckpt).to(device)
print("Model loaded successfully.")

@app.post("/predict/")
async def predict(file: UploadFile = File(...), threshold: float = 0.4):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    # Run Inference
    with torch.no_grad():
        inputs = processor(images=[image], return_tensors="pt")
        outputs = model(**inputs.to(device))
        
        target_sizes = torch.tensor([[image.size[1], image.size[0]]])
        results = processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

        # Format results
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections.append({
                "label": model.config.id2label[label.item()],
                "score": round(score.item(), 3),
                "box": [round(i, 2) for i in box.tolist()]
            })

    return JSONResponse(content={"detections": detections})

@app.get("/health")
def health_check():
    return {"status": "healthy"}
