import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class AnxietyPredictor:
    def __init__(self, model_name="bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Note: In a real scenario, you'd load your fine-tuned model here.
        # For now, we'll load the base model and use it for prediction logic.
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(self.device)
        self.labels = ["Low Anxiety", "Moderate Anxiety", "High Anxiety"]

    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][prediction].item()
        
        return {
            "prediction": self.labels[prediction],
            "confidence": confidence
        }

import os

# Singleton instance
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        fine_tuned_path = "./fine-tuned-bert"
        if os.path.exists(fine_tuned_path):
            print(f"Loading fine-tuned model from {fine_tuned_path}")
            predictor = AnxietyPredictor(fine_tuned_path)
        else:
            print("Fine-tuned model not found, loading base model 'bert-base-uncased'")
            predictor = AnxietyPredictor("bert-base-uncased")
    return predictor
