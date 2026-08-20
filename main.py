"""
Plant Disease Detector — FastAPI App
--------------------------------------
Upload a photo of a plant leaf to /predict and get back the predicted
disease class using the MobileNetV2 model trained in the notebook.

Run locally:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Then open http://localhost:8000/docs to test it interactively.

Expects the trained model file `mobilenetv2_plant_disease.keras` to sit in
the same folder as this script.
"""

import io
from contextlib import asynccontextmanager

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from PIL import Image

# Config

MODEL_PATH = "mobilenetv2_plant_disease.keras"
IMG_SIZE = (128, 128)  # must match training input size

# Class names, in the exact index order produced by
# tf.keras.utils.image_dataset_from_directory during training.
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def format_class_name(raw_name: str) -> tuple[str, str]:
    """Split a raw folder-style label like 'Apple___Apple_scab' into
    ('Apple', 'Apple scab') for friendlier display."""
    if "___" in raw_name:
        plant, condition = raw_name.split("___", 1)
    else:
        plant, condition = raw_name, ""
    plant = plant.replace("_", " ")
    condition = condition.replace("_", " ").strip()
    return plant, condition

# Response schemas

class PredictionItem(BaseModel):
    plant: str
    condition: str
    label: str
    confidence: float


class PredictionResponse(BaseModel):
    top_prediction: PredictionItem
    top_5: list[PredictionItem]


# Model loading (once, at startup)

model_holder = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model once
    model_holder["model"] = tf.keras.models.load_model(MODEL_PATH)
    yield
    # Shutdown: clear it
    model_holder.clear()


app = FastAPI(
    title="Plant Disease Detector API",
    description="Upload a leaf image and get a predicted disease class.",
    version="1.0.0",
    lifespan=lifespan,
)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize/convert a PIL image to the array shape the model expects.
    Note: MobileNetV2 preprocessing (scaling to [-1, 1]) is baked into the
    model itself (see the `preprocess_input` layer added during training),
    so we only resize and batch here — do NOT rescale by /255 as well.
    """
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = tf.keras.utils.img_to_array(image)
    arr = np.expand_dims(arr, axis=0)
    return arr

# Routes

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": "model" in model_holder}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a JPG or PNG image.",
        )

    model = model_holder.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    arr = preprocess_image(image)
    preds = model.predict(arr, verbose=0)[0]

    top5_idx = np.argsort(preds)[::-1][:5]
    top5 = []
    for idx in top5_idx:
        plant, condition = format_class_name(CLASS_NAMES[idx])
        label = f"{plant} — {condition}" if condition else plant
        top5.append(
            PredictionItem(
                plant=plant,
                condition=condition,
                label=label,
                confidence=float(preds[idx]),
            )
        )

    return PredictionResponse(top_prediction=top5[0], top_5=top5)