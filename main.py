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
    print("✅ Model loaded successfully")
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
# Prediction Route
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read file contents
        contents = await file.read()
        
        # Validate file is an image
        try:
            img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as img_error:
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded file is not a valid image. Error: {str(img_error)}"
            )
        
        # CRITICAL: Resize to 224x224 to match model training size
        print(f"Original image size: {img.size}")
        img = img.resize((224, 224))
        print(f"Resized image size: {img.size}")
        
        # Preprocess image
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"Input array shape: {img_array.shape}")
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        
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
        
        print(f"Predicted: {predicted_species} with confidence: {confidence:.2f}%")
        
        # Extract genus (first word of species name)
        genus = predicted_species.split(" ")[0] if " " in predicted_species else predicted_species
        
        # Get ecological information
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
        # Re-raise HTTP exceptions
        raise http_error
    
    except Exception as e:
        # Log error for debugging
        error_trace = traceback.format_exc()
        print(f"❌ Prediction error: {str(e)}")
        print(error_trace)
        
        # Return error response (remove trace in production)
        return {
            "success": False,
            "error": str(e),
            "trace": error_trace,
            "message": "An error occurred during prediction. Please try again with a valid diatom microscopic image."
        }