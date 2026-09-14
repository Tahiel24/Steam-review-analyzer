import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ruta local por defecto
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "distilbert_sentiment"
# Identificador oficial en Hugging Face Hub
HF_MODEL_ID = "Tahiel24/steam-distilbert-sentiment"


class SentimentClassifier:
    """
    Clasificador en producción que consume el modelo fine-tuneado en Fase 3.
    """
    def __init__(self, model_path: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Prioridad: parámetro pasado o variable de entorno
        env_path = os.getenv("MODEL_PATH")
        selected_path = model_path or env_path

        if selected_path:
            load_target = selected_path
            print(f"[INFO] Cargando modelo especificado: {load_target}")
        elif DEFAULT_MODEL_DIR.exists():
            load_target = str(DEFAULT_MODEL_DIR)
            print(f"[INFO] Cargando modelo local desde: {load_target}")
        else:
            # Si no hay ruta local (como en Streamlit Cloud), descarga desde Hugging Face
            load_target = HF_MODEL_ID
            print(f"[INFO] Directorio local no detectado. Descargando desde Hugging Face Hub: {load_target}")

        self.model = AutoModelForSequenceClassification.from_pretrained(load_target)
        self.tokenizer = AutoTokenizer.from_pretrained(load_target)
        self.model.to(self.device)
        self.model.eval()

        # Mapeo según Fase 3: 0 -> Negativo (No Recomendado), 1 -> Positivo (Recomendado)
        self.label_map = {0: "Negativo", 1: "Positivo"}

    def predict(self, text: str) -> dict:
        """Predice la polaridad y confianza para un texto individual."""
        if not isinstance(text, str) or not text.strip():
            return {"label": "Desconocido", "sentiment_id": -1, "confidence": 0.0}

        inputs = self.tokenizer(
            text,
            max_length=128,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]
            sentiment_id = int(torch.argmax(probabilities).item())

        return {
            "label": self.label_map.get(sentiment_id, "Desconocido"),
            "sentiment_id": sentiment_id,
            "confidence": round(float(probabilities[sentiment_id].item()), 4)
        }