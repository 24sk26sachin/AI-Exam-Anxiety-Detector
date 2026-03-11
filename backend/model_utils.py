import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os

class AnxietyPredictor:
    def __init__(self, model_name="bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.labels = ["1. Low Anxiety", "2. Moderate Anxiety", "3. High Anxiety", "4. Very High Anxiety"]
        
        if model_name == "bert-base-uncased":
            # Using a pre-trained sentiment pipeline for more accurate zero-shot evaluation 
            # according to the current statement, instead of random sequence classifier weights.
            self.analyzer = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4, ignore_mismatched_sizes=True).to(self.device)

    def predict(self, text: str):
        if self.model_name == "bert-base-uncased":
            # Truncate text to 512 chars roughly to avoid token length issues with default pipeline
            res = self.analyzer(text[:1500])[0]
            label = res['label']
            score = res['score']
            
            # Map sentiment dynamically to 4 anxiety classes
            if label == 'POSITIVE':
                if score > 0.85:
                    pred_idx = 0  # 1. Low Anxiety
                else:
                    pred_idx = 1  # 2. Moderate Anxiety
            else: # NEGATIVE
                if score > 0.85:
                    pred_idx = 3  # 4. Very High Anxiety
                else:
                    pred_idx = 2  # 3. High Anxiety
                    
            confidence = score
            return {
                "prediction": self.labels[pred_idx],
                "confidence": confidence
            }
        else:
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

# Singleton instance
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fine_tuned_path = os.path.join(base_dir, "fine-tuned-bert")
        if os.path.exists(fine_tuned_path):
            print(f"Loading fine-tuned model from {fine_tuned_path}")
            predictor = AnxietyPredictor(fine_tuned_path)
        else:
            print("Fine-tuned model not found, loading heuristic analyzer")
            predictor = AnxietyPredictor("bert-base-uncased")
    return predictor
