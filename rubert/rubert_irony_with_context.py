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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
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


class ContextTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor | None, use_sampler: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_sampler = use_sampler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        if self.class_weights is None:
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss
        logits = outputs.get("logits")
        loss = CrossEntropyLoss(weight=self.class_weights.to(logits.device))(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        if not self.use_sampler:
            return super().get_train_dataloader()
        if self.train_dataset is None:
            raise ValueError("Trainer requires a train_dataset.")
        labels = np.array(self.train_dataset.labels)
        class_counts = np.bincount(labels, minlength=2)
        inv = 1.0 / np.maximum(class_counts, 1)
        sample_weights = inv[labels]
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
    p = argparse.ArgumentParser(description="RuBERT irony detection with explicit context")
    p.add_argument("--train", default="train.csv")
    p.add_argument("--val", default="val.csv")
    p.add_argument("--test", default="test.csv")
    p.add_argument("--text-col", default="sentences")
    p.add_argument("--label-col", default="marked irony")
    p.add_argument("--source-col", default="source")
    p.add_argument("--paragraph-col", default="paragraph")
    p.add_argument(
        "--context-corpus",
        default="full_dataset_with_context.csv",
        help="Full corpus CSV used to attach true prev/next context by source+paragraph; ignored if missing.",
    )
    p.add_argument("--output-dir", default="results/rubert_context")
    p.add_argument("--model-name", default=MODEL_NAME)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--context-mode",
        choices=["current", "prev", "prev_curr", "prev_curr_next"],
        default="prev_curr",
    )
    p.add_argument("--use-context-markers", action="store_true", help="Use [PREV]/[CURR]/[NEXT] markers")
    p.add_argument("--balancing", choices=["none", "weights", "sampler", "both"], default="weights")
    p.add_argument("--positive-weight-scale", type=float, default=1.0)
    p.add_argument("--threshold-min-precision", type=float, default=0.0)
    p.add_argument("--threshold-min-recall", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--early-stopping-patience", type=int, default=1)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--max-test-samples", type=int, default=0)
    return p.parse_args()


def clean_df(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=[text_col, label_col]).copy()
    out[text_col] = out[text_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    out[label_col] = out[label_col].astype(int)
    out = out[out[text_col] != ""]
    out = out[out[label_col].isin([0, 1])]
    return out.reset_index(drop=True)


def attach_context_from_corpus(
    df: pd.DataFrame,
    corpus: pd.DataFrame | None,
    text_col: str,
    source_col: str,
    paragraph_col: str,
) -> pd.DataFrame:
    out = df.copy()
    has_keys = source_col in out.columns and paragraph_col in out.columns

    if corpus is not None and has_keys and source_col in corpus.columns and paragraph_col in corpus.columns:
        ctx = corpus.copy()
        ctx[text_col] = ctx[text_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        ctx = ctx.sort_values([source_col, paragraph_col]).reset_index(drop=True)
        if "prev_text" not in ctx.columns:
            ctx["prev_text"] = ctx.groupby(source_col)[text_col].shift(1)
        ctx["next_text"] = ctx.groupby(source_col)[text_col].shift(-1)
        ctx = ctx[[source_col, paragraph_col, "prev_text", "next_text"]].rename(
            columns={"prev_text": "_corpus_prev_text", "next_text": "_corpus_next_text"}
        )
        out = out.merge(ctx, on=[source_col, paragraph_col], how="left", sort=False)
        if "prev_text" in out.columns:
            out["prev_text"] = out["prev_text"].combine_first(out["_corpus_prev_text"])
        else:
            out["prev_text"] = out["_corpus_prev_text"]
        out["next_text"] = out["_corpus_next_text"]
        out = out.drop(columns=["_corpus_prev_text", "_corpus_next_text"])
    elif has_keys:
        out = out.sort_values([source_col, paragraph_col]).reset_index(drop=True)
        if "prev_text" not in out.columns:
            out["prev_text"] = out.groupby(source_col)[text_col].shift(1)
        out["next_text"] = out.groupby(source_col)[text_col].shift(-1)
    else:
        out = out.reset_index(drop=True)
        if "prev_text" not in out.columns:
            out["prev_text"] = out[text_col].shift(1)
        out["next_text"] = out[text_col].shift(-1)

    out["prev_text"] = out["prev_text"].fillna("").astype(str)
    out["next_text"] = out["next_text"].fillna("").astype(str)
    return out


def build_context(df: pd.DataFrame, text_col: str, mode: str) -> pd.DataFrame:
    out = df.copy()
    if mode == "current":
        out["context_text"] = out[text_col]
    elif mode in {"prev", "prev_curr"}:
        out["context_text"] = out["prev_text"] + " [SEP] " + out[text_col]
    else:
        out["context_text"] = out["prev_text"] + " [SEP] " + out[text_col] + " [SEP] " + out["next_text"]

    return out


def add_context_markers(df: pd.DataFrame, text_col: str, mode: str) -> pd.DataFrame:
    out = df.copy()
    if mode == "current":
        out["context_text"] = "[CURR] " + out[text_col].astype(str)
    elif mode in {"prev", "prev_curr"}:
        out["context_text"] = (
            "[PREV] " + out["prev_text"].astype(str) + " [CURR] " + out[text_col].astype(str)
        )
    else:
        out["context_text"] = (
            "[PREV] " + out["prev_text"].astype(str) +
            " [CURR] " + out[text_col].astype(str) +
            " [NEXT] " + out["next_text"].astype(str)
        )
    return out


def maybe_subsample(df: pd.DataFrame, label_col: str, max_samples: int, seed: int) -> pd.DataFrame:
    if max_samples <= 0 or len(df) <= max_samples:
        return df
    sampled, _ = train_test_split(df, train_size=max_samples, random_state=seed, stratify=df[label_col])
    return sampled.reset_index(drop=True)


def softmax_probs(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def find_best_threshold(
    y_true: np.ndarray,
    proba_pos: np.ndarray,
    min_precision: float = 0.0,
    min_recall: float = 0.0,
):
    candidates = np.unique(np.clip(proba_pos, 0.0, 1.0))
    grid = np.unique(np.concatenate(([0.001, 0.005, 0.01, 0.02, 0.05], candidates, [0.5])))
    best_thr = 0.5
    best = metrics(y_true, (proba_pos >= best_thr).astype(int))
    for thr in grid:
        cur = metrics(y_true, (proba_pos >= thr).astype(int))
        if cur["precision"] < min_precision or cur["recall"] < min_recall:
            continue
        if (cur["f1"] > best["f1"]) or (cur["f1"] == best["f1"] and cur["recall"] > best["recall"]):
            best_thr, best = float(thr), cur
    return best_thr, best


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    corpus = None
    if args.context_corpus and os.path.exists(args.context_corpus):
        corpus = pd.read_csv(args.context_corpus)
        print("Context corpus:", args.context_corpus, corpus.shape)
    elif args.context_mode == "prev_curr_next":
        print("Warning: context corpus not found; next_text will be inferred inside each split.")

    train = clean_df(pd.read_csv(args.train), args.text_col, args.label_col)
    val = clean_df(pd.read_csv(args.val), args.text_col, args.label_col)
    test = clean_df(pd.read_csv(args.test), args.text_col, args.label_col)

    train = attach_context_from_corpus(train, corpus, args.text_col, args.source_col, args.paragraph_col)
    val = attach_context_from_corpus(val, corpus, args.text_col, args.source_col, args.paragraph_col)
    test = attach_context_from_corpus(test, corpus, args.text_col, args.source_col, args.paragraph_col)

    train = build_context(train, args.text_col, args.context_mode)
    val = build_context(val, args.text_col, args.context_mode)
    test = build_context(test, args.text_col, args.context_mode)
    if args.use_context_markers:
        train = add_context_markers(train, args.text_col, args.context_mode)
        val = add_context_markers(val, args.text_col, args.context_mode)
        test = add_context_markers(test, args.text_col, args.context_mode)

    train = maybe_subsample(train, args.label_col, args.max_train_samples, args.seed)
    val = maybe_subsample(val, args.label_col, args.max_val_samples, args.seed)
    test = maybe_subsample(test, args.label_col, args.max_test_samples, args.seed)

    print("Train:", train.shape, "Val:", val.shape, "Test:", test.shape)
    print("Train label distribution:")
    print(train[args.label_col].value_counts(dropna=False).sort_index())

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.use_context_markers:
        tokenizer.add_special_tokens({"additional_special_tokens": ["[PREV]", "[CURR]", "[NEXT]"]})

    train_enc = tokenizer(train["context_text"].tolist(), truncation=True, padding="max_length", max_length=args.max_length)
    val_enc = tokenizer(val["context_text"].tolist(), truncation=True, padding="max_length", max_length=args.max_length)
    test_enc = tokenizer(test["context_text"].tolist(), truncation=True, padding="max_length", max_length=args.max_length)

    y_train = train[args.label_col].tolist()
    y_val = val[args.label_col].tolist()
    y_test = test[args.label_col].tolist()

    train_ds = IronyDataset(train_enc, y_train)
    val_ds = IronyDataset(val_enc, y_val)
    test_ds = IronyDataset(test_enc, y_test)

    cw = None
    if args.balancing in {"weights", "both"}:
        raw_cw = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=np.array(y_train))
        raw_cw[1] *= args.positive_weight_scale
        cw = torch.tensor(raw_cw, dtype=torch.float)
        print("Class weights:", cw.tolist())
    else:
        print("Class weights: disabled")
    use_sampler = args.balancing in {"sampler", "both"}
    print("Weighted sampler:", "enabled" if use_sampler else "disabled")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    if args.use_context_markers:
        model.resize_token_embeddings(len(tokenizer))

    ta = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
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
        probs = softmax_probs(np.array(logits))[:, 1]
        labels = np.array(labels)
        m05 = metrics(labels, (probs >= 0.5).astype(int))
        _, mbest = find_best_threshold(
            labels,
            probs,
            min_precision=args.threshold_min_precision,
            min_recall=args.threshold_min_recall,
        )
        return {
            "accuracy": mbest["accuracy"],
            "precision": mbest["precision"],
            "recall": mbest["recall"],
            "f1": mbest["f1"],
            "f1_at_0_50": m05["f1"],
        }

    trainer = ContextTrainer(
        class_weights=cw,
        use_sampler=use_sampler,
        model=model,
        args=ta,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()

    val_out = trainer.predict(val_ds)
    val_probs = softmax_probs(np.array(val_out.predictions))[:, 1]
    val_labels = np.array(val_out.label_ids)
    best_thr, best_val = find_best_threshold(
        val_labels,
        val_probs,
        min_precision=args.threshold_min_precision,
        min_recall=args.threshold_min_recall,
    )

    test_out = trainer.predict(test_ds)
    test_probs = softmax_probs(np.array(test_out.predictions))[:, 1]
    test_labels = np.array(test_out.label_ids)

    pred05 = (test_probs >= 0.5).astype(int)
    pred_opt = (test_probs >= best_thr).astype(int)
    m05 = metrics(test_labels, pred05)
    mopt = metrics(test_labels, pred_opt)
    oracle_thr, oracle_test = find_best_threshold(test_labels, test_probs)

    out = test.copy()
    out["rubert_context_proba_irony"] = test_probs
    out["rubert_context_pred"] = pred05
    out["rubert_context_pred_optimized"] = pred_opt
    out = add_binary_outcomes(out, args.label_col, "rubert_context_pred_optimized")
    pred_path = os.path.join(args.output_dir, "rubert_context_test_predictions.csv")
    excel_path = os.path.join(args.output_dir, "rubert_context_test_predictions.xlsx")
    out.to_csv(pred_path, index=False)
    save_predictions_excel(out, excel_path)

    payload = {
        "model": args.model_name,
        "context_mode": args.context_mode,
        "context_corpus": args.context_corpus if corpus is not None else None,
        "balancing": args.balancing,
        "positive_weight_scale": args.positive_weight_scale,
        "max_length": args.max_length,
        "best_threshold_val": best_thr,
        "val_best_metrics": best_val,
        "test_metrics_default_0_50": m05,
        "test_metrics_best_threshold": mopt,
        "test_oracle_threshold_for_diagnostics": oracle_thr,
        "test_oracle_metrics_for_diagnostics": oracle_test,
    }
    with open(os.path.join(args.output_dir, "rubert_context_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    trainer.save_model(os.path.join(args.output_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))

    print("Best threshold on val:", round(best_thr, 4))
    print("Val best:", best_val)
    print("Test @0.50:", m05)
    print("Test @best:", mopt)
    print("Test oracle diagnostic:", round(oracle_thr, 4), oracle_test)
    print("Saved:")
    print(" -", pred_path)
    print(" -", excel_path)


if __name__ == "__main__":
    main()
