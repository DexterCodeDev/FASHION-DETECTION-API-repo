import io
import os
import threading
from typing import Optional

import requests
import torch
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoImageProcessor, AutoModelForObjectDetection

MODEL_ID = "yainage90/fashion-object-detection"
DEVICE = "cpu"

HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI(
    title="Fashion Object Detection API",
    version="1.0.0"
)

model = None
processor = None
model_lock = threading.Lock()


def load_model():
    global model, processor

    print("Loading model...")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN
    )

    model = AutoModelForObjectDetection.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN
    )

    model.eval()
    model.to(DEVICE)

    print("Model loaded successfully!")


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/")
def home():
    return {
        "status": "running",
        "model": MODEL_ID
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


def crop_image(image, box):
    x1, y1, x2, y2 = map(int, box)

    width, height = image.size

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    cropped = image.crop((x1, y1, x2, y2))

    buffer = io.BytesIO()
    cropped.save(buffer, format="JPEG")

    return {
        "width": cropped.width,
        "height": cropped.height
    }


def predict_image(image, threshold=0.4, return_crops=False):

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with model_lock:
        with torch.no_grad():
            outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])

    results = processor.post_process_object_detection(
        outputs=outputs,
        threshold=threshold,
        target_sizes=target_sizes
    )[0]

    detections = []

    for score, label, box in zip(
        results["scores"],
        results["labels"],
        results["boxes"]
    ):

        label_name = model.config.id2label[label.item()]

        box = [round(x, 2) for x in box.tolist()]

        item = {
            "label": label_name,
            "score": round(score.item(), 4),
            "box": box
        }

        if return_crops:
            item["crop_info"] = crop_image(image, box)

        detections.append(item)

    return detections


@app.post("/predict")
async def predict(
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    threshold: float = Form(0.4),
    return_crops: bool = Form(False)
):

    if image is None and image_url is None:
        raise HTTPException(
            status_code=400,
            detail="Provide image file or image_url"
        )

    try:

        if image:
            contents = await image.read()
            pil_image = Image.open(
                io.BytesIO(contents)
            ).convert("RGB")

        else:
            response = requests.get(
                image_url,
                timeout=30
            )
            response.raise_for_status()

            pil_image = Image.open(
                io.BytesIO(response.content)
            ).convert("RGB")

        detections = predict_image(
            pil_image,
            threshold=threshold,
            return_crops=return_crops
        )

        return JSONResponse({
            "success": True,
            "detections": detections,
            "count": len(detections)
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
