from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np

BEST_MODEL_PATH = "./models/epoch_num_4_lr_0.0001_batch_size_64"
WORST_MODEL_PATH = "./models/epoch_num_2_lr_0.0002_batch_size_64"
BATCH_SIZE = 64
OUTPUT_FILE = "comparison.txt"


def preprocess_function(tokenizer, examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True
    )


def get_predictions(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="./tmp", per_device_eval_batch_size=BATCH_SIZE),
        processing_class=tokenizer
    )
    predictions = trainer.predict(tokenized_comparison_data).predictions
    return np.argmax(predictions, axis=1)


ds = load_dataset("nyu-mll/glue", "mrpc")

comparison_data = ds["test"]

tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_PATH)

tokenized_comparison_data = comparison_data.map(lambda examples: preprocess_function(tokenizer, examples),
                                                batched=True).remove_columns(["sentence1", "sentence2"])

best_predictions = get_predictions(BEST_MODEL_PATH)
worst_predictions = get_predictions(WORST_MODEL_PATH)
labels = np.array(comparison_data["label"])

mask = (best_predictions == labels) & (worst_predictions != labels)
matched_indices = np.where(mask)[0]

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for idx in matched_indices:
        idx = int(idx)
        sentence1 = comparison_data[idx]["sentence1"]
        sentence2 = comparison_data[idx]["sentence2"]
        label = labels[idx]
        f.write("------------------\n")
        f.write(f"Validation Sample {idx}\n")
        f.write(f"Sentence 1: {sentence1}\n")
        f.write(f"Sentence 2: {sentence2}\n")
        f.write(f"True Label: {label}\n")
        f.write("------------------\n\n")

print(f"Exported {len(matched_indices)} comparison samples to {OUTPUT_FILE}")
