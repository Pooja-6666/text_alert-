# text_alert-
AI-powered emotional intelligence system that analyzes user text, predicts emotional state &amp; intensity, and suggests smart actions in real-time using FastAPI + Streamlit.
# 🌿 ArvyaX Emotional AI

An intelligent emotional analysis system that understands user feelings from text input and provides actionable suggestions to improve mental well-being.

---

## 🚀 Features

* 🧠 Emotion Detection using Machine Learning
* 🔥 Intensity Prediction of emotions
* 🎯 Smart Action Recommendation System
* ⏰ Context-aware timing suggestions
* 💬 Human-like response generation
* 🎨 Dynamic UI (changes based on emotion)
* ⚡ FastAPI backend + Streamlit frontend

---

## 🛠️ Tech Stack

* **Backend:** FastAPI
* **Frontend:** Streamlit
* **ML Models:**

  * Logistic Regression (Emotion Classification)
  * Random Forest (Intensity Prediction)
* **NLP:** TF-IDF Vectorization
* **Libraries:** scikit-learn, numpy, pandas, scipy

---

## 📂 Project Structure

```
arvyax-assignment/
│── app.py          # FastAPI backend (API + model inference)
│── main.py         # Model training script
│── ui.py           # Streamlit frontend
│── vectorizer.pkl
│── emotion_model.pkl
│── intensity_model.pkl
│── train.csv
│── test.csv
│── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```
git clone <your-repo-link>
cd arvyax-assignment
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run Backend (FastAPI)

```
python -m uvicorn app:app --reload
```

👉 API runs at: http://127.0.0.1:8000

---

### 4. Run Frontend (Streamlit)

```
streamlit run ui.py
```

👉 UI runs at: http://localhost:8501

---

## 🧪 API Endpoint

### POST `/predict`

**Request:**

```json
{
  "text": "I feel very stressed today",
  "stress": 4,
  "energy": 2,
  "sleep": 5
}
```

**Response:**

```json
{
  "state": "stressed",
  "intensity": 4,
  "confidence": 0.82,
  "uncertain": 0,
  "action": "grounding",
  "timing": "within_15_min",
  "message": "You seem stressed. Try grounding."
}
```

---

## 🧠 How It Works

1. User inputs journal text + metadata
2. Text is cleaned and vectorized using TF-IDF
3. Features are combined with user metadata
4. ML models predict:

   * Emotional state
   * Intensity level
5. Decision logic suggests:

   * What action to take
   * When to take it
6. UI dynamically adapts to emotion

---

## 🎯 Future Improvements

* 🤖 Use Deep Learning (BERT / LSTM)
* 📊 Emotion tracking dashboard
* 🗣 Voice input support
* ☁️ Cloud deployment (AWS/GCP)
* 📱 Mobile app integration

---

## 👩‍💻 Author

**Pooja B L**

---

## ⭐ Acknowledgment

Built as part of an AI/ML project to combine machine learning with real-world emotional intelligence applications.

---
