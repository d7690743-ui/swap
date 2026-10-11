import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from insightface.app import FaceAnalysis

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MODEL_PACK = os.getenv("FACE_MODEL_PACK", "buffalo_l")
MODEL_ROOT = Path(os.getenv("INSIGHTFACE_ROOT", "~/.insightface")).expanduser()

app = FastAPI(title="Rostro a rostro", docs_url=None, redoc_url=None)
face_analysis = None


@app.on_event("startup")
def load_face_model():
    global face_analysis
    # Some apps store the same Buffalo L files in a directory named `buffalo_1`.
    # Prefer that local directory when it contains both models; otherwise let
    # InsightFace download the official `buffalo_l` pack into its model cache.
    local_alias = MODEL_ROOT / "models" / "buffalo_1"
    pack_name = MODEL_PACK
    if (local_alias / "det_10g.onnx").is_file() and (local_alias / "2d106det.onnx").is_file():
        pack_name = "buffalo_1"

    face_analysis = FaceAnalysis(
        name=pack_name,
        root=str(MODEL_ROOT),
        allowed_modules=["detection", "landmark_2d_106"],
        providers=["CPUExecutionProvider"],
    )
    face_analysis.prepare(ctx_id=-1, det_size=(640, 640))


@app.get("/")
def home():
    return FileResponse(ROOT / "index.html")


@app.get("/healthz")
def health():
    return {"ok": True, "model_loaded": face_analysis is not None}


@app.post("/api/detect")
async def detect_face(request: Request):
    if request.headers.get("content-type", "").split(";", 1)[0] not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Usa una imagen JPG, PNG o WebP.")

    chunks = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="La imagen debe pesar menos de 12 MB.")
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise HTTPException(status_code=400, detail="No se recibió ninguna imagen.")

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen.")

    height, width = image.shape[:2]
    if width * height > 24_000_000:
        scale = (24_000_000 / (width * height)) ** 0.5
        image = cv2.resize(image, (round(width * scale), round(height * scale)))
        height, width = image.shape[:2]

    faces = face_analysis.get(image)
    if not faces:
        raise HTTPException(status_code=422, detail="No encontré una cara. Prueba con una foto frontal y nítida.")

    # Choose the largest face to avoid an ambiguous automatic selection.
    face = max(faces, key=lambda item: max(0, item.bbox[2] - item.bbox[0]) * max(0, item.bbox[3] - item.bbox[1]))
    landmarks = getattr(face, "landmark_2d_106", None)
    if landmarks is None or len(landmarks) < 20:
        raise HTTPException(status_code=503, detail="El paquete buffalo_l no devolvió los puntos faciales esperados.")

    return {
        "width": width,
        "height": height,
        "bbox": [round(float(value), 2) for value in face.bbox[:4]],
        "landmarks": [[round(float(x), 2), round(float(y), 2)] for x, y in landmarks],
    }
