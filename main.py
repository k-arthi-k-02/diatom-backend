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
app = FastAPI(title="Diatom Forensic API - Telangana")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load Model & Class Names
# -----------------------------
try:
    model = load_model("best_model.h5", compile=False)
    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
    print("✅ Model loaded")
    print("✅ Number of classes:", len(class_names))
except Exception as e:
    print("❌ Error loading model or class names")
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

# -----------------------------
# Root Route
# -----------------------------
@app.get("/")
def root():
    return {"status": "Diatom Forensic API (Telangana) is running"}

# -----------------------------
# Prediction Route
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        try:
            img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a valid image"
            )

        img = img.resize((256, 256))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        if prediction.shape[1] != len(class_names):
            raise HTTPException(
                status_code=500,
                detail="Model output size does not match class names"
            )

        predicted_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)
        predicted_species = class_names[predicted_index]

        # Extract genus and get ecological info
        genus = predicted_species.split(" ")[0]
        info = genus_knowledge_map.get(
            genus,
            {
                "type": "Unknown diatom type",
                "water_body": "Unknown freshwater body",
                "locations": ["Various water bodies in Telangana"],
                "indicator": "Unknown",
                "pollution_level": "Unknown"
            }
        )

        # Select primary location (first in list)
        primary_location = info["locations"][0]
        all_locations = ", ".join(info["locations"])

        return {
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

    except HTTPException as e:
        raise e
    except Exception as e:
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }