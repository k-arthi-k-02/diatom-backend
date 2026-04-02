from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
import traceback
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# App Initialization
# -----------------------------
app = FastAPI(title="Diatom Forensic API - Telangana")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU-bound inference tasks
_executor = ThreadPoolExecutor(max_workers=2)

# -----------------------------
# Load Model & Class Names
# -----------------------------
try:
    model = load_model("best_model.h5", compile=False)
    # Warm up the model with a dummy prediction to pre-allocate GPU/CPU memory
    _dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    model.predict(_dummy, verbose=0)
    del _dummy

    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)

    # Pre-convert class_names to a tuple for faster indexing
    class_names = tuple(class_names)

    print("✅ Model loaded and warmed up successfully")
    print("✅ Number of classes:", len(class_names))
    print("✅ Model input shape:", model.input_shape)
except Exception as e:
    print("❌ Error loading model or class names")
    print(f"Error details: {str(e)}")
    raise e

# -----------------------------
# Genus-based Ecology with Telangana Reservoirs
# -----------------------------
genus_knowledge_map = {
    "Achnanthidium": {
        "type": "Small benthic freshwater diatom",
        "water_body": "Clean freshwater streams and tanks",
        "locations": ["Osmansagar Lake", "Himayatsagar Lake", "Durgam Cheruvu"],
        "indicator": "Good water quality",
        "pollution_level": "Low"
    },
    "Navicula": {
        "type": "Free-living freshwater diatom",
        "water_body": "Rivers, ponds, lake sediments",
        "locations": ["Hussain Sagar Lake", "Lower Manair Dam", "Singur Dam"],
        "indicator": "Normal freshwater environment",
        "pollution_level": "Moderate"
    },
    "Nitzschia": {
        "type": "Pollution-tolerant freshwater diatom",
        "water_body": "Polluted rivers and urban lakes",
        "locations": ["Musi River", "Hussain Sagar Lake", "Safilguda Lake"],
        "indicator": "Organic pollution",
        "pollution_level": "High"
    },
    "Gomphonema": {
        "type": "Attached freshwater diatom",
        "water_body": "Rivers and flowing streams",
        "locations": ["Krishna River", "Godavari River", "Manjeera River"],
        "indicator": "Moderate water quality",
        "pollution_level": "Moderate"
    },
    "Fragilaria": {
        "type": "Chain-forming planktonic diatom",
        "water_body": "Lakes and reservoirs",
        "locations": ["Nizam Sagar", "Sriram Sagar Project (SRSP)", "Pakhal Lake"],
        "indicator": "Standing water",
        "pollution_level": "Low to Moderate"
    },
    "Cyclotella": {
        "type": "Planktonic centric diatom",
        "water_body": "Lakes and reservoirs",
        "locations": ["Nagarjuna Sagar", "Lower Manair Dam", "Mid Manair Dam"],
        "indicator": "Standing water",
        "pollution_level": "Low to Moderate"
    },
    "Stephanodiscus": {
        "type": "Centric freshwater diatom",
        "water_body": "Nutrient-rich lakes",
        "locations": ["Hussain Sagar Lake", "Fox Sagar Lake", "Mir Alam Tank"],
        "indicator": "Eutrophic conditions",
        "pollution_level": "High"
    },
    "Sellaphora": {
        "type": "Benthic sediment-dwelling diatom",
        "water_body": "River beds and lake sediments",
        "locations": ["Godavari River", "Krishna River", "Laknavaram Lake"],
        "indicator": "Freshwater sediment",
        "pollution_level": "Low to Moderate"
    },
    "Pinnularia": {
        "type": "Benthic freshwater diatom",
        "water_body": "Ponds and wetlands",
        "locations": ["Shamirpet Lake", "Ameenpur Lake", "Ramappa Lake"],
        "indicator": "Low-flow freshwater",
        "pollution_level": "Low"
    }
}

# Pre-build the default fallback info to avoid repeated dict creation
_DEFAULT_INFO = {
    "type": "Unknown diatom type",
    "water_body": "Unknown freshwater body",
    "locations": ["Various water bodies in Telangana"],
    "indicator": "Unknown",
    "pollution_level": "Unknown"
}

# -----------------------------
# Root Route
# -----------------------------
@app.get("/")
def root():
    return {
        "status": "Diatom Forensic API (Telangana) is running",
        "version": "1.0.0",
        "model_input_shape": str(model.input_shape),
        "total_classes": len(class_names)
    }

# -----------------------------
# Health Check Route
# -----------------------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes_loaded": len(class_names) > 0
    }

# -----------------------------
# CPU-bound preprocessing + inference (runs in thread pool)
# -----------------------------
def _preprocess_and_predict(contents: bytes):
    """Decode, resize, normalize image and run model inference."""
    # Open and convert image
    img = Image.open(io.BytesIO(contents))
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize using BILINEAR (faster than default BICUBIC, sufficient for 224x224)
    img = img.resize((224, 224), Image.BILINEAR)

    # Convert to float32 array and normalize in one step (avoids double copy)
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)

    # Run inference
    prediction = model.predict(img_array, verbose=0)
    return prediction

# -----------------------------
# Prediction Route
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read file contents
        contents = await file.read()

        # Validate that the file is an image before passing to the thread pool
        try:
            Image.open(io.BytesIO(contents)).verify()
        except Exception as img_error:
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded file is not a valid image. Error: {str(img_error)}"
            )

        # Offload CPU-bound work to thread pool so the event loop stays free
        loop = asyncio.get_event_loop()
        try:
            prediction = await loop.run_in_executor(
                _executor, _preprocess_and_predict, contents
            )
        except Exception as pred_error:
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {str(pred_error)}"
            )

        # Validate prediction shape
        if prediction.shape[1] != len(class_names):
            raise HTTPException(
                status_code=500,
                detail=f"Model output size ({prediction.shape[1]}) does not match class names ({len(class_names)})"
            )

        # Get predicted class
        predicted_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)
        predicted_species = class_names[predicted_index]

        # Extract genus (first word of species name)
        space_idx = predicted_species.find(" ")
        genus = predicted_species[:space_idx] if space_idx != -1 else predicted_species

        # Get ecological information
        info = genus_knowledge_map.get(genus, _DEFAULT_INFO)

        # Prepare location information
        primary_location = info["locations"][0]
        all_locations = ", ".join(info["locations"])

        # Return comprehensive response
        return {
            "success": True,
            "species": predicted_species,
            "genus": genus,
            "confidence": round(confidence, 2),
            "diatom_type": info["type"],
            "water_body": info["water_body"],
            "region": primary_location,
            "all_locations": all_locations,
            "state": "Telangana, India",
            "ecological_indicator": info["indicator"],
            "pollution_level": info["pollution_level"],
            "inference_note": "Ecology inferred using genus-level knowledge from Telangana water bodies"
        }

    except HTTPException as http_error:
        raise http_error

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"❌ Prediction error: {str(e)}")
        print(error_trace)

        return {
            "success": False,
            "error": str(e),
            "trace": error_trace,
            "message": "An error occurred during prediction. Please try again with a valid diatom microscopic image."
        }
