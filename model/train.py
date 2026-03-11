import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd

def train_model(data_path, model_name="bert-base-uncased", output_dir="./results"):
    # Load dataset
    df = pd.read_csv(data_path)
    # Subset for faster training in demo/lite mode
    if len(df) > 500:
        df = df.sample(500, random_state=42)
    dataset = Dataset.from_pandas(df)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Split dataset
    train_test = tokenized_datasets.train_test_split(test_size=0.1)
    
    # Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,  # Reduced for speed
        max_steps=10,        # Fast-track for verification
        weight_decay=0.1,
        logging_steps=2,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_test["train"],
        eval_dataset=train_test["test"],
    )
    
    # Train
    trainer.train()
    
    # Save the model
    model.save_pretrained("./fine-tuned-bert")
    tokenizer.save_pretrained("./fine-tuned-bert")
    print("Model fine-tuning complete and saved to ./fine-tuned-bert")

if __name__ == "__main__":
    data_csv = "data/processed/processed_data.csv"
    train_model(data_csv)
