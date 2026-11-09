from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
from tensorflow.keras.applications.efficientnet import preprocess_input

# ─── App Setup ─────────────────────────────────────────────────
app = FastAPI(title="Brain Tumor Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Constants ─────────────────────────────────────────────────
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
IMG_SIZE    = (224, 224)

# ─── Model Load ────────────────────────────────────────────────
model = tf.keras.models.load_model('brain_tumor_b3.keras')
print("✅ Model loaded!")

# ─── Routes ────────────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "message": "Brain Tumor Detection API",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "health":  "/health"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": "EfficientNetB3"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Image read karo
        contents = await file.read()
        img_pil  = Image.open(io.BytesIO(contents)).convert("RGB")
        img_arr  = np.array(img_pil)

        # Preprocess
        img_resized    = cv2.resize(img_arr, IMG_SIZE)
        img_processed  = preprocess_input(img_resized.astype(np.float32))
        img_batch      = np.expand_dims(img_processed, axis=0)

        # Predict
        preds     = model(img_batch, training=False).numpy()[0]
        pred_idx  = int(np.argmax(preds))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(preds[pred_idx]) * 100

        # Sab classes ki probabilities
        all_probs = {
            CLASS_NAMES[i]: round(float(preds[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        return {
            "success":     True,
            "prediction":  pred_class,
            "confidence":  round(confidence, 2),
            "all_classes": all_probs,
            "has_tumor":   pred_class != "No Tumor"
        }

    except Exception as e:
        return {
            "success": False,
            "error":   str(e)
        }