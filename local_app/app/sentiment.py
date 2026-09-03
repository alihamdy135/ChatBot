"""
Sentiment/emotion inference.
Loads the fine-tuned DistilBERT model trained in
kaggle_notebooks/02_sentiment_distilbert.ipynb.
Runs on CPU locally -- DistilBERT is small enough that this is fast (~10-50ms/msg).
"""
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODELS_DIR = Path(__file__).parent.parent / "models" / "sentiment"


class SentimentClassifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(MODELS_DIR))
        self.model.eval()

        with open(MODELS_DIR / "bucket_labels.json") as f:
            self.labels = json.load(f)  # {"0": "negative", "1": "neutral", "2": "positive"}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def predict(self, text: str) -> dict:
        """Returns {"label": "negative"|"neutral"|"positive", "confidence": float}"""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=64
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        pred_idx = int(torch.argmax(probs).item())
        return {
            "label": self.labels[str(pred_idx)],
            "confidence": float(probs[pred_idx].item()),
        }
