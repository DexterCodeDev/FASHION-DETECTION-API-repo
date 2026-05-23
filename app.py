import os
import io
import traceback
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection
)

app = FastAPI()

MODEL_NAME = "yainage90/fashion-object-detection"

print("Loading DETR Fashion Model...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForObjectDetection.from_pretrained(MODEL_NAME)

model.eval()


@app.get("/")
def health():
    return {"status": "running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")

        max_size = 800

        if max(image.size) > max_size:
            ratio = max_size / max(image.size)

            new_size = (
                int(image.size[0] * ratio),
                int(image.size[1] * ratio)
            )

            image = image.resize(new_size, Image.LANCZOS)

        with torch.no_grad():
            inputs = processor(
                images=image,
                return_tensors="pt"
            )

            outputs = model(**inputs)

            target_sizes = torch.tensor(
                [[image.height, image.width]]
            )

            results = processor.post_process_object_detection(
                outputs,
                threshold=0.4,
                target_sizes=target_sizes
            )[0]

        detections = []

        for score, label, box in zip(
            results["scores"],
            results["labels"],
            results["boxes"]
        ):
            label_id = label.item()

            class_name = model.config.id2label.get(
                label_id,
                "Unknown"
            )

            detections.append({
                "class": class_name,
                "confidence": float(score.item()),
                "box": [float(v) for v in box.tolist()]
            })

        return {"detections": detections}

    except Exception as e:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"AI Crash: {str(e)}"
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
