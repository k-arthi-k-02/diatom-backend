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
    model = load_model("best_model.h5")
    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
    print("✅ Model and class names loaded")
except Exception as e:
    print("❌ Error loading model")
    raise e

# -----------------------------
# Genus-based Ecology (Telangana)
# -----------------------------
genus_knowledge_map = {
    "Achnanthidium": {
        "type": "Small benthic freshwater diatom",
        "water_body": "Clean freshwater streams and tanks",
        "region": "Hill streams, irrigation channels",
        "indicator": "Good water quality"
    },
    "Navicula": {
        "type": "Free-living freshwater diatom",
        "water_body": "Rivers, ponds, lake sediments",
        "region": "Godavari basin, Musi river stretches, village tanks",
        "indicator": "Normal freshwater environment"
    },
    "Nitzschia": {
        "type": "Pollution-tolerant freshwater diatom",
        "water_body": "Polluted rivers and urban lakes",
        "region": "Urban drains, Musi downstream, industrial canals",
        "indicator": "Organic pollution"
    },
    "Gomphonema": {
        "type": "Attached freshwater diatom",
        "water_body": "Rivers and flowing streams",
        "region": "Godavari tributaries, Manjeera river",
        "indicator": "Moderate water quality"
    },
    "Fragilaria": {
        "type": "Chain-forming planktonic diatom",
        "water_body": "Lakes and reservoirs",
        "region": "Hussain Sagar, Durgam Cheruvu, irrigation reservoirs",
        "indicator": "Standing water"
    },
    "Cyclotella": {
        "type": "Planktonic centric diatom",
        "water_body": "Lakes and reservoirs",
        "region": "Urban lakes, drinking water reservoirs",
        "indicator": "Standing water"
    },
    "Stephanodiscus": {
        "type": "Centric freshwater diatom",
        "water_body": "Nutrient-rich lakes",
        "region": "Eutrophic urban lakes",
        "indicator": "Eutrophic conditions"
    },
    "Sellaphora": {
        "type": "Benthic sediment-dwelling diatom",
        "water_body": "River beds and lake sediments",
        "region": "River bottoms, tank sediments",
        "indicator": "Freshwater sediment"
    },
    "Pinnularia": {
        "type": "Benthic freshwater diatom",
        "water_body": "Ponds, wetlands, low-flow waters",
        "region": "Village ponds, marshy tanks",
        "indicator": "Low-flow freshwater"
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

        # Force RGB
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))

        # Preprocess
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        predicted_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

        predicted_species = class_names[predicted_index]

        # -----------------------------
        # Species → Genus → Telangana Ecology
        # -----------------------------
        genus = predicted_species.split(" ")[0]

        info = genus_knowledge_map.get(
            genus,
            {
                "type": "Unknown diatom type",
                "water_body": "Unknown freshwater body",
                "region": "Unknown",
                "indicator": "Unknown"
            }
        )

        return {
            "species": predicted_species,
            "genus": genus,
            "confidence": round(confidence, 2),
            "diatom_type": info["type"],
            "water_body": info["water_body"],
            "region": info["region"],
            "example_locations": info["examples"],
            "ecological_indicator": info["indicator"],
            "inference_note": "Location inferred based on diatom genus ecology within Telangana state"
        }

    except Exception as e:
        print("❌ Prediction Error")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to server error"
        )
