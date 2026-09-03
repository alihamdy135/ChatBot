"""
Language detection inference.
Loads the TF-IDF vectorizer + LogisticRegression model trained in
kaggle_notebooks/01_language_detection.ipynb.
"""
import json
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models" / "language"


class LanguageDetector:
    def __init__(self):
        self.vectorizer = joblib.load(MODELS_DIR / "lang_vectorizer.pkl")
        self.model = joblib.load(MODELS_DIR / "lang_model.pkl")
        with open(MODELS_DIR / "lang_labels.json") as f:
            self.labels = json.load(f)  # {"0": "en", "1": "fr", ...}

    def predict(self, text: str) -> str:
        """Returns the ISO-ish language code predicted (matches papluca dataset labels)."""
        clean = text.strip().lower()
        vec = self.vectorizer.transform([clean])
        pred_idx = self.model.predict(vec)[0]
        return self.labels[str(pred_idx)]
