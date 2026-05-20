import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from experiment_outputs import add_binary_outcomes, save_predictions_excel


MODEL_NAME = "DeepPavlov/rubert-base-cased"


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class IronyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        labels = np.array(self.train_dataset.labels)
        class_sample_count = np.bincount(labels, minlength=2)
        class_weights = 1.0 / np.maximum(class_sample_count, 1)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="RuBERT binary irony classification")
    parser.add_argument("--train", default="train.csv", help="Path to train CSV")
    parser.add_argument("--val", default="val.csv", help="Path to val CSV")
    parser.add_argument("--test", default="test.csv", help="Path to test CSV")
    parser.add_argument("--text-col", default="sentences", help="Text column name")
    parser.add_argument("--label-col", default="marked irony", help="Label column name")
    parser.add_argument("--output-dir", default="results/rubert_irony", help="Output directory")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0, help="If > 0, use subset of train for quick run")
    parser.add_argument("--max-val-samples", type=int, default=0, help="If > 0, use subset of val for quick run")
    parser.add_argument("--max-test-samples", type=int, default=0, help="If > 0, use subset of test for quick run")
    return parser.parse_args()


def clean_df(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    out = df.copy()
    if text_col not in out.columns or label_col not in out.columns:
        raise KeyError(f"Expected columns '{text_col}' and '{label_col}' in dataframe")
    out = out.dropna(subset=[text_col, label_col]).copy()
    out[text_col] = out[text_col].astype(str).str.replace(r"\\s+", " ", regex=True).str.strip()
    out[label_col] = out[label_col].astype(int)
    out = out[out[text_col] != ""].copy()
    out = out[out[label_col].isin([0, 1])].copy()
    return out.reset_index(drop=True)


def load_splits(args) -> SplitData:
    train = clean_df(pd.read_csv(args.train), args.text_col, args.label_col)
    val = clean_df(pd.read_csv(args.val), args.text_col, args.label_col)
    test = clean_df(pd.read_csv(args.test), args.text_col, args.label_col)
    return SplitData(train=train, val=val, test=test)


def maybe_subsample(df: pd.DataFrame, label_col: str, max_samples: int, seed: int) -> pd.DataFrame:
    if max_samples <= 0 or len(df) <= max_samples:
        return df
    sampled, _ = train_test_split(
        df,
        train_size=max_samples,
        random_state=seed,
        stratify=df[label_col],
    )
    return sampled.reset_index(drop=True)


def tokenize_texts(tokenizer, texts, max_length):
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def softmax_probs(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


def metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_threshold(y_true: np.ndarray, proba_pos: np.ndarray):
    candidates = np.unique(np.clip(proba_pos, 0.0, 1.0))
    grid = np.unique(np.concatenate(([0.001, 0.005, 0.01, 0.02], candidates, [0.5])))
    best_thr = 0.5
    best = metrics_from_preds(y_true, (proba_pos >= best_thr).astype(int))
    for thr in grid:
        pred = (proba_pos >= thr).astype(int)
        cur = metrics_from_preds(y_true, pred)
        if (cur["f1"] > best["f1"]) or (cur["f1"] == best["f1"] and cur["recall"] > best["recall"]):
            best = cur
            best_thr = float(thr)
    return best_thr, best


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    splits = load_splits(args)
    splits = SplitData(
        train=maybe_subsample(splits.train, args.label_col, args.max_train_samples, args.seed),
        val=maybe_subsample(splits.val, args.label_col, args.max_val_samples, args.seed),
        test=maybe_subsample(splits.test, args.label_col, args.max_test_samples, args.seed),
    )

    print("Train:", splits.train.shape, "Val:", splits.val.shape, "Test:", splits.test.shape)
    print("Train label distribution:")
    print(splits.train[args.label_col].value_counts(dropna=False).sort_index())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_enc = tokenize_texts(tokenizer, splits.train[args.text_col].tolist(), args.max_length)
    val_enc = tokenize_texts(tokenizer, splits.val[args.text_col].tolist(), args.max_length)
    test_enc = tokenize_texts(tokenizer, splits.test[args.text_col].tolist(), args.max_length)

    train_ds = IronyDataset(train_enc, splits.train[args.label_col].tolist())
    val_ds = IronyDataset(val_enc, splits.val[args.label_col].tolist())
    test_ds = IronyDataset(test_enc, splits.test[args.label_col].tolist())

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=splits.train[args.label_col].values,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print("Class weights:", class_weights.tolist())

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs_pos = softmax_probs(np.array(logits))[:, 1]
        y_pred = (probs_pos >= 0.5).astype(int)
        m = metrics_from_preds(np.array(labels), y_pred)
        return {
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
        }

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    val_out = trainer.predict(val_ds)
    val_probs = softmax_probs(np.array(val_out.predictions))[:, 1]
    val_labels = np.array(val_out.label_ids)

    best_thr, best_val_metrics = find_best_threshold(val_labels, val_probs)
    print(f"Best threshold on val: {best_thr:.2f}")
    print("Best val metrics:", best_val_metrics)

    test_out = trainer.predict(test_ds)
    test_probs = softmax_probs(np.array(test_out.predictions))[:, 1]
    test_labels = np.array(test_out.label_ids)

    test_pred_default = (test_probs >= 0.5).astype(int)
    test_pred_best = (test_probs >= best_thr).astype(int)

    test_metrics_default = metrics_from_preds(test_labels, test_pred_default)
    test_metrics_best = metrics_from_preds(test_labels, test_pred_best)

    print("Test metrics @0.50:", test_metrics_default)
    print(f"Test metrics @{best_thr:.2f}:", test_metrics_best)

    pred_df = splits.test.copy()
    pred_df["rubert_proba_irony"] = test_probs
    pred_df["rubert_pred"] = test_pred_default
    pred_df["rubert_pred_optimized"] = test_pred_best
    pred_df = add_binary_outcomes(pred_df, args.label_col, "rubert_pred_optimized")
    pred_path = os.path.join(args.output_dir, "rubert_test_predictions.csv")
    excel_path = os.path.join(args.output_dir, "rubert_test_predictions.xlsx")
    pred_df.to_csv(pred_path, index=False)
    save_predictions_excel(pred_df, excel_path)

    metrics_payload = {
        "model": MODEL_NAME,
        "text_col": args.text_col,
        "label_col": args.label_col,
        "best_threshold_val": best_thr,
        "val_best_metrics": best_val_metrics,
        "test_metrics_default_0_50": test_metrics_default,
        "test_metrics_best_threshold": test_metrics_best,
    }
    metrics_path = os.path.join(args.output_dir, "rubert_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    trainer.save_model(os.path.join(args.output_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))

    print("Saved:")
    print(" -", pred_path)
    print(" -", excel_path)
    print(" -", metrics_path)
    print(" -", os.path.join(args.output_dir, "best_model"))


if __name__ == "__main__":
    main()
