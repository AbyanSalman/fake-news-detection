from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
from nltk.corpus import stopwords

# Load model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
stop_words = set(stopwords.words('english'))

# Initialize FastAPI
app = FastAPI(title="Fake News Detection API", version="1.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model
class NewsArticle(BaseModel):
    text: str

# Preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(words)

# Root endpoint
@app.get("/")
def root():
    return {
        "message": "Fake News Detection API",
        "author": "Muhamad Abyan Avriliansyah Salman (2441049)",
        "endpoints": {
            "predict": "/predict [POST]",
            "docs": "/docs"
        }
    }

# Prediction endpoint
@app.post("/predict")
def predict(data: NewsArticle):
    if len(data.text.strip()) < 10:
        return {"error": "Text too short (minimum 10 characters)"}
    
    # Preprocess and vectorize
    clean_text = preprocess_text(data.text)
    text_vector = vectorizer.transform([clean_text])
    
    # Predict
    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    
    # Return result
    return {
        "prediction": "REAL" if prediction == 1 else "FAKE",
        "confidence": round(float(max(probabilities)) * 100, 2),
        "probabilities": {
            "fake": round(float(probabilities[0]) * 100, 2),
            "real": round(float(probabilities[1]) * 100, 2)
        }
    }
