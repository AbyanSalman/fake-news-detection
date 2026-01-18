# Fake News Detection
This project implements a machine learning model for detecting fake news articles using Natural Language Processing (NLP) and Logistic Regression. The system includes data preprocessing, model training with TF-IDF vectorization, and deployment via FastAPI with a web interface.


## Dataset
Source: Fake and Real News Dataset - Kaggle
Total Articles: 44,898
Real News: 21,417 articles (True.csv)
Fake News: 23,481 articles (Fake.csv)
Features: TF-IDF vectorization with 5,000 features (unigrams + bigrams)


## Tech Stack
Language: Python 3.11+
ML Framework: scikit-learn 1.3.2
NLP: NLTK 3.8.1
API Framework: FastAPI 0.104.1
Server: Uvicorn
Frontend: HTML5 + Tailwind CSS + JavaScript


## Setup
```bash
  git clone https://github.com/AbyanSalman/fake-news-detection.git
  cd fake-news-detection
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## Running Training
```bash
  python train.py
```

## Running FastAPI
```bash
  uvicorn main:app --reload
```


## API Endpoints:
- GET / = API information
- POST /predict = Predict if news is fake or real
- GET /docs = Interactive API documentation (Swagger UI)


## Using the Application
- Option 1: Web Interface
  - 1. Open index.html in your web browser
  - 2. Paste a news article in the text area
  - 3. Click "Analyze News"
  - 4. View the prediction result with confidence scores
- Option 2: API (Swagger UI)
  - 1. Navigate to http://127.0.0.1:8000/docs
  - 2. Click on POST /predict endpoint
  - 3. Click "Try it out"
  - 4. Enter your news text in the request body:
  - 5. Click "Execute"
  - 6. View the JSON response
- Option 3: cURL
```bash
  curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news article here..."}'
```
- Option 4: Python Script
```python
  import requests

  response = requests.post('http://127.0.0.1:8000/predict', 
      json={'text': 'Your news article here...'})

  result = response.json()
  print(f"Prediction: {result['prediction']}")
  print(f"Confidence: {result['confidence']}%")
```


## API Response Format
```json
  {
    "prediction": "REAL",
    "confidence": 97.21,
    "probabilities": {
      "fake": 2.79,
      "real": 97.21
    }
  }
```


## Resources
- Fast API : https://fastapi.tiangolo.com/
- scikit-learn: https://scikit-learn.org/stable/
- NLTK: https://www.nltk.org/
- Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- Reference Paper: Shu, K., et al. (2017). Fake News Detection on Social Media: A Data Mining Perspective. ACM SIGKDD Explorations Newsletter, 19(1), 22-36.
