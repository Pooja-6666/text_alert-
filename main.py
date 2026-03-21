import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# -------- FUNCTIONS --------

def clean_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r'[^a-z\s]', '', text.lower())

def get_uncertain_flag(conf, text, stress, energy):
    # low confidence
    if conf < 0.55:
        return 1

    # very short text
    if len(text.split()) < 3:
        return 1

    # conflicting signals
    if stress >= 4 and "calm" in text:
        return 1

    return 0

def decide_action(state, intensity, stress, energy, time_of_day):

    # extreme stress + intensity
    if stress >= 4 and intensity >= 4:
        return "box_breathing", "now"

    # high stress
    if stress >= 4:
        return "grounding", "within_15_min"

    # low energy + high intensity
    if energy <= 2 and intensity >= 4:
        return "rest", "now"

    # low energy
    if energy <= 2:
        if time_of_day in ["night", "late_evening"]:
            return "rest", "tonight"
        return "movement", "within_15_min"

    # productive state
    if state in ["calm", "focused"] and energy >= 3:
        return "deep_work", "now"

    # moderate stress
    if stress == 3:
        return "light_planning", "later_today"

    # positive mood
    if state == "happy":
        return "movement", "later_today"

    return "pause", "within_15_min"

def generate_message(state, action):
    return f"You seem {state}. Let's try {action} to help you rebalance."

# -------- LOAD DATA --------

print("Loading data...")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# -------- HANDLE MISSING VALUES --------

print("Handling missing values...")

# numeric
train_df["sleep_hours"].fillna(train_df["sleep_hours"].mean(), inplace=True)
test_df["sleep_hours"].fillna(train_df["sleep_hours"].mean(), inplace=True)

train_df["energy_level"].fillna(3, inplace=True)
test_df["energy_level"].fillna(3, inplace=True)

train_df["stress_level"].fillna(3, inplace=True)
test_df["stress_level"].fillna(3, inplace=True)

train_df["duration_min"].fillna(train_df["duration_min"].mean(), inplace=True)
test_df["duration_min"].fillna(train_df["duration_min"].mean(), inplace=True)

# categorical
train_df["time_of_day"].fillna("unknown", inplace=True)
test_df["time_of_day"].fillna("unknown", inplace=True)

train_df["previous_day_mood"].fillna("neutral", inplace=True)
test_df["previous_day_mood"].fillna("neutral", inplace=True)


# -------- PREPROCESS --------

print("Cleaning text...")
train_df["clean_text"] = train_df["journal_text"].apply(clean_text)
test_df["clean_text"] = test_df["journal_text"].apply(clean_text)

# -------- FEATURES --------

print("Extracting features...")

# Text features
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
X_text_train = vectorizer.fit_transform(train_df["clean_text"])
X_text_test = vectorizer.transform(test_df["clean_text"])

# Metadata
meta_cols = ["sleep_hours", "energy_level", "stress_level"]
X_meta_train = train_df[meta_cols]
X_meta_test = test_df[meta_cols]

# Scale metadata
scaler = StandardScaler()
X_meta_train = scaler.fit_transform(X_meta_train)
X_meta_test = scaler.transform(X_meta_test)

# Combine
X_train = hstack([X_text_train, X_meta_train])
X_test = hstack([X_text_test, X_meta_test])

# -------- MODELS --------

print("Training models...")

# Emotion model
emotion_model = LogisticRegression(max_iter=1000)
emotion_model.fit(X_train, train_df["emotional_state"])

# Intensity model
intensity_model = RandomForestClassifier()
intensity_model.fit(X_train, train_df["intensity"])

# -------- PREDICT --------

print("Making predictions...")

emotion_preds = emotion_model.predict(X_test)
emotion_probs = emotion_model.predict_proba(X_test)
intensity_preds = intensity_model.predict(X_test)

confidence = np.max(emotion_probs, axis=1)

# Uncertainty
uncertain_flags = [
    get_uncertain_flag(conf, txt, stress, energy)
    for conf, txt, stress, energy in zip(
        confidence,
        test_df["clean_text"],
        test_df["stress_level"],
        test_df["energy_level"]
    )
]

# -------- DECISION --------

print("Applying decision logic...")

actions, timings, messages = [], [], []

for i in range(len(test_df)):
    action, timing = decide_action(
        emotion_preds[i],
        intensity_preds[i],
        test_df.loc[i, "stress_level"],
        test_df.loc[i, "energy_level"],
        test_df.loc[i, "time_of_day"]
    )

    message = generate_message(emotion_preds[i], action)

    actions.append(action)
    timings.append(timing)
    messages.append(message)

# -------- SAVE --------

print("Saving predictions...")

output = pd.DataFrame({
    "id": test_df["id"],
    "predicted_state": emotion_preds,
    "predicted_intensity": intensity_preds,
    "confidence": confidence,
    "uncertain_flag": uncertain_flags,
    "what_to_do": actions,
    "when_to_do": timings,
    "message": messages   # BONUS
})

output.to_csv("predictions.csv", index=False)

print("✅ Done! predictions.csv generated successfully")

import joblib

joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(emotion_model, "emotion_model.pkl")
joblib.dump(intensity_model, "intensity_model.pkl")

print("✅ Models saved!")