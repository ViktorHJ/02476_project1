from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision import transforms

from cifakeclassification.model import Cifake_CNN
from dotenv import load_dotenv

load_dotenv()

model: Cifake_CNN | None = None
device: torch.device | None = None
preprocess = None
class_labels = ["FAKE", "REAL"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # -> 02476_project1


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, device, preprocess

    ckpt_path = os.getenv("MODEL_CKPT", "models/model.ckpt")
    image_size = int(os.getenv("IMAGE_SIZE", "32"))

    # ✅ Make relative paths work (so you can set MODEL_CKPT=artifacts/... in .env)
    p = Path(ckpt_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p

    print("Loading checkpoint:", p)

    if not p.exists():
        raise RuntimeError(f"Checkpoint not found: {p.resolve()}")

    device = pick_device()

    model = Cifake_CNN.load_from_checkpoint(str(p), map_location=device)
    model.eval()
    model.to(device)

    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    yield

    print("Cleaning up")
    del model
    model = None
    preprocess = None
    if device is not None and device.type == "cuda":
        torch.cuda.empty_cache()
    device = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device) if device else None,
        "checkpoint": os.getenv("MODEL_CKPT", "models/model.ckpt"),
    }


@app.post("/predict/")
async def predict(data: UploadFile = File(...)):
    if model is None or device is None or preprocess is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        i_image = Image.open(data.file).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    x = preprocess(i_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = int(torch.argmax(probs).item())

    return {
        "prediction": class_labels[pred_idx],
        "probs": {
            class_labels[0]: float(probs[0].item()),
            class_labels[1]: float(probs[1].item()),
        },
    }