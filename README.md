# Fashion Object Detection API

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open:

http://localhost:8000/docs

---

## Health

GET

```bash
curl http://localhost:8000/health
```

---

## Upload image

```bash
curl -X POST \
  -F "image=@dress.jpg" \
  -F "threshold=0.4" \
  http://localhost:8000/predict
```

---

## URL image

```bash
curl -X POST \
  -F "image_url=https://example.com/test.jpg" \
  http://localhost:8000/predict
```

---

## Return crop info

```bash
curl -X POST \
  -F "image=@dress.jpg" \
  -F "return_crops=true" \
  http://localhost:8000/predict
```
