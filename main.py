from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
import traceback
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# -----------------------------
# App Initialization
# -----------------------------
app = FastAPI(title="Diatom Forensic API")

# Allow Flutter / Web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load Model & Labels (ONCE)
# -----------------------------
try:
    model = load_model("best_model.h5")
    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
    print("✅ Model and class names loaded successfully")
except Exception as e:
    print("❌ Error loading model or class names")
    print(e)
    raise e

# -----------------------------
# Species → Location Mapping
# -----------------------------
species_location_map = {
    "Sellaphora nigri": {
        "water_body": "Freshwater (River / Lake)",
        "region": "Temperate freshwater regions"
    },
    "Achnanthidium atomoides": {
        "water_body": "Freshwater streams",
        "region": "Europe / Similar regions"
    },
    "Navicula rostellata": {
        "water_body": "Rivers and ponds",
        "region": "Widespread freshwater regions"
    }
}

# -----------------------------
# Root Route (Health Check)
# -----------------------------
@app.get("/")
def root():
    return {"status": "Diatom Forensic API is running"}

# -----------------------------
# Prediction Route
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await file.read()

        # 🔴 IMPORTANT FIX: FORCE RGB (3 channels)
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))  # match model input size

        # Preprocess image
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Run prediction
        prediction = model.predict(img_array)
        predicted_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

        predicted_species = class_names[predicted_index]

        # Location info
        info = species_location_map.get(predicted_species, {})

        return {
            "species": predicted_species,
            "confidence": confidence,
            "water_body": info.get("water_body", "Unknown"),
            "region": info.get("region", "Unknown")
        }

    except Exception as e:
        # Print full error to Render logs
        print("❌ Prediction Error")
        print(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to server error"
        )
