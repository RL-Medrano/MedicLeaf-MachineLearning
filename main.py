import os
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from utils.detection import detect_plant

script_dir = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR = os.path.join(
    script_dir, "image_testing"
)

app = FastAPI(
    title="MedicLeaf API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class PredictRequest(BaseModel):
    image: str

@app.get("/")
def read_root():
    return {
        "title": "MediCLeaf API",
        "version": "1.0.0"
    }

@app.post("/predict")
def predict(img: PredictRequest):
    image_path = os.path.join(IMAGE_DIR, img.image)

    if not os.path.exists(image_path):
        return {
            "error": f"Image not found: {image_path}"
        }

    result = detect_plant(image_path)

    return {
        "plant_name": result["plant_name"],
        "confidence": round(result["confidence"], 4),
        "is_confident": result["is_confident"],
        "plant_info": result["plant_info"]
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="localhost",
        port=5000
    )
