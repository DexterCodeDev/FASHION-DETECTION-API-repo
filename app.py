from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

app = FastAPI()

# Load once at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "yainage90/fashion-object-detection"

processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForObjectDetection.from_pretrained(model_id).to(device)


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # validate image
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return {"error": "Only JPG, JPEG, PNG images are allowed"}

    image = Image.open(file.file).convert("RGB")

    inputs = processor(images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        target_sizes = torch.tensor([[image.size[1], image.size[0]]])
        results = processor.post_process_object_detection(
            outputs, threshold=0.4, target_sizes=target_sizes
        )[0]

    # Format response
    detections = []
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        detections.append({
            "score": float(score.item()),
            "label": model.config.id2label[int(label.item())],
            "box": [float(x.item()) for x in box],
        })

    return {"detections": detections}
