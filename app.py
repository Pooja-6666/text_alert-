from fastapi import FastAPI
import numpy as np
import re
import joblib
from scipy.sparse import hstack

app = FastAPI()

# LOAD MODELS
vectorizer = joblib.load("vectorizer.pkl")
emotion_model = joblib.load("emotion_model.pkl")
intensity_model = joblib.load("intensity_model.pkl")

def clean_text(text):
    return re.sub(r'[^a-z\s]', '', text.lower())

def decide_action(state, intensity, stress, energy, time_of_day):

    if stress >= 4 and intensity >= 4:
        return "box_breathing", "now"

    if stress >= 4:
        return "grounding", "within_15_min"

    if energy <= 2:
        return "rest", "later_today"

    if state in ["calm", "focused"] and energy >= 3:
        return "deep_work", "now"

    return "pause", "within_15_min"

@app.get("/")
def home():
    return {"message": "ArvyaX ML API 🚀"}

@app.post("/predict")
def predict(data: dict):

    text = clean_text(data.get("text", ""))
    stress = data.get("stress", 3)
    energy = data.get("energy", 3)
    sleep = data.get("sleep", 6)

    X_text = vectorizer.transform([text])
    X_meta = np.array([[sleep, energy, stress]])
    X = hstack([X_text, X_meta])

    state = emotion_model.predict(X)[0]
    intensity = int(intensity_model.predict(X)[0])

    probs = emotion_model.predict_proba(X)
    confidence = float(np.max(probs))

    uncertain = 1 if confidence < 0.5 or len(text.split()) < 3 else 0

    action, timing = decide_action(state, intensity, stress, energy, "morning")

    return {
        "state": state,
        "intensity": intensity,
        "confidence": confidence,
        "uncertain": uncertain,
        "action": action,
        "timing": timing,
        "message": f"You seem {state}. Try {action}."
    }