from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = FastAPI()

# Allow Flutter app to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and class names
model = load_model("best_model.h5")
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    img = img.resize((224, 224))
    
    # Preprocess
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_species = class_names[predicted_index]
    confidence = float(np.max(prediction) * 100)
    
    # Get location info
    info = species_location_map.get(predicted_species, {})
    
    return {
        "species": predicted_species,
        "confidence": confidence,
        "water_body": info.get("water_body", "Unknown"),
        "region": info.get("region", "Unknown")
    }

@app.get("/")
def root():
    return {"message": "Diatom Forensic API is running"}