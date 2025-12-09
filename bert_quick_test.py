import pandas as pd
import numpy as np
import pathlib
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import f1_score

DATA_DIR   = pathlib.Path("data/final")
MODEL_DIR  = pathlib.Path("./models/bert-base-chinese")
label_list = sorted(pd.read_csv(DATA_DIR / "train.csv")["label"].unique())
label2id   = {label: idx for idx, label in enumerate(label_list)}


tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)

dataset = load_dataset("csv", data_files={
    "train": str(DATA_DIR / "train.csv"),
    "test":  str(DATA_DIR / "test.csv")
})

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.map(
    lambda batch: {"labels": [label2id[l] for l in batch["label"]]},
    batched=True,
    desc="label2id"
)
dataset = dataset.remove_columns(["label", "text"])
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

dataset["train"] = dataset["train"].shuffle(seed=42).select(range(5000))  # 只用 5k
dataset["test"]  = dataset["test"].shuffle(seed=42).select(range(1000))

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, local_files_only=True, num_labels=len(label_list)
)

args = TrainingArguments(
    output_dir="ckpt",
    per_device_train_batch_size=16,   # ← 调大
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    num_train_epochs=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    seed=42,
    fp16=True,
    dataloader_pin_memory=True,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"eval_f1": float(f1_score(labels, preds, average="macro"))}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
result = trainer.evaluate()
print("Quick-test F1:", result["eval_f1"])

preds = trainer.predict(dataset["test"])
pd.DataFrame({"labels": preds.label_ids, "predictions": preds.predictions.argmax(-1)}).to_json("ckpt/predictions.json", indent=2)