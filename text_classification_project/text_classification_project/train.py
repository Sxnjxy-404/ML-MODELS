import argparse
import os
import json
import numpy as np
import pandas as pd
from datasets import Dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score

DEFAULT_MODEL = "distilbert-base-uncased"

def encode_labels(dataset, class_labels=None):
    """Map string labels to integers, create ClassLabel if needed."""
    if class_labels is None:
        unique = sorted(list(set(dataset["label"])))
        class_labels = ClassLabel(names=unique)

    def _map(example):
        example["label"] = class_labels.str2int(example["label"])
        return example

    mapped = dataset.map(_map)
    return mapped, class_labels

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted")
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with columns: text,label")
    parser.add_argument("--out_dir", type=str, default="./models/textclf", help="Where to save the fine-tuned model")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL, help="Base model name")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load CSV
    df = pd.read_csv(args.csv)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have 'text' and 'label' columns")

    # Create Dataset and split
    dataset = Dataset.from_pandas(df[["text", "label"]])
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    # Encode labels for train and test separately
    train_dataset, class_labels = encode_labels(dataset["train"])
    test_dataset, _ = encode_labels(dataset["test"], class_labels=class_labels)

    # Tokenizer & tokenization
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_test = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    tokenized_train = tokenized_train.remove_columns(["text"])
    tokenized_test = tokenized_test.remove_columns(["text"])
    tokenized_train.set_format("torch")
    tokenized_test.set_format("torch")

    # Model
    num_labels = len(class_labels.names)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    # Training
    training_args = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "results"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save model + label mapping
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    with open(os.path.join(args.out_dir, "label_names.json"), "w", encoding="utf-8") as f:
        json.dump(class_labels.names, f, ensure_ascii=False, indent=2)

    print(f"✅ Model saved to {args.out_dir}")

if __name__ == "__main__":
    main()
