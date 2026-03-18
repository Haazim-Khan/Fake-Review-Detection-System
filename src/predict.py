import os
import pickle
from src.preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "model", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "model", "vectorizer.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)


def predict_review(review):
    if model is None or vectorizer is None:
        return "Model not loaded properly ❌"
    try:
        cleaned = clean_text(review)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        return "Genuine Review ✅" if prediction == 1 else "Fake Review ❌"
    except Exception as e:
        return f"Error: {str(e)}"