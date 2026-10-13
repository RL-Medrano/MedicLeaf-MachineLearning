
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

IMG_SIZE = (224, 224)

script_dir  = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(script_dir)

MODEL_PATH = os.path.join(
    BACKEND_DIR, "machine_learning", "model", "best_model.keras"
)

PLANT_INFO_PATH = os.path.join(
    BACKEND_DIR, "machine_learning", "plant_info.json"
)

TRAIN_DIR = os.path.join(
    BACKEND_DIR, "machine_learning", "dataset", "train"
)

if os.path.isdir(TRAIN_DIR):
    Class_names = sorted(os.listdir(TRAIN_DIR))
else:
    Class_names = []

_model = None

def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

def preprocess_image(image_input):
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    image = image.resize(IMG_SIZE)
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # CRITICAL: must match preprocessing used during training
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

    return image_array

def load_plant_info():
    if not os.path.exists(PLANT_INFO_PATH):
        return {}
    with open(PLANT_INFO_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

def detect_plant(image_input, threshold=0.80):
    try:
        model = load_model()
        plant_info = load_plant_info()

        processed_image = preprocess_image(image_input)
        predictions = model.predict(processed_image, verbose=0)
        prediction_scores = predictions[0]
        predicted_idx = np.argmax(prediction_scores)
        confidence = float(prediction_scores[predicted_idx])
        predicted_class = Class_names[predicted_idx]

        is_confident = confidence >= threshold
        plant_name = predicted_class if is_confident else "unknown"

        plant_details = plant_info.get(predicted_class, {}) if is_confident else {}

        return {
            "plant_name": plant_name,
            "confidence": confidence,
            "is_confident": is_confident,
            "plant_info": plant_details
        }

    except Exception as e:
        raise RuntimeError(f"Detection failed: {str(e)}")

def batch_detect(image_paths, threshold=0.85):
    results = []
    for image_path in image_paths:
        try:
            result = detect_plant(image_path, threshold)
            results.append(result)
        except Exception as e:
            results.append({
                "image": str(image_path),
                "error": str(e)
            })
    return results
