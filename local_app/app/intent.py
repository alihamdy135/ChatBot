"""
Intent classifier inference.
Loads the TF-IDF + calibrated LinearSVC model trained in
kaggle_notebooks/03_intent_classifier.ipynb, then maps the fine-grained (27-class)
prediction down to the 7-category routing bucket.
"""
import json
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models" / "intent"


class IntentClassifier:
    def __init__(self):
        self.vectorizer = joblib.load(MODELS_DIR / "intent_vectorizer.pkl")
        self.model = joblib.load(MODELS_DIR / "intent_model.pkl")

        with open(MODELS_DIR / "intent_fine_labels.json") as f:
            self.fine_labels = json.load(f)  # {"0": "cancel_order", ...}

        with open(MODELS_DIR / "intent_to_route.json") as f:
            self.intent_to_route = json.load(f)  # {"cancel_order": "order_management", ...}

    def predict(self, text: str) -> dict:
        """Returns {"fine_intent": str, "route": str, "confidence": float}"""
        clean = text.strip().lower()
        vec = self.vectorizer.transform([clean])

        pred_idx = self.model.predict(vec)[0]
        fine_intent = self.fine_labels[str(pred_idx)]

        # confidence via predict_proba (model is a CalibratedClassifierCV)
        probs = self.model.predict_proba(vec)[0]
        confidence = float(max(probs))

        route = self.intent_to_route.get(fine_intent, "out_of_scope")

        return {
            "fine_intent": fine_intent,
            "route": route,
            "confidence": confidence,
        }
