import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from misinfo_verifier.transformer_data import (
    LABELS,
    ID2LABEL,
    LABEL2ID,
    FeverTransformerDataset,
    TransformerCollator,
)


ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed" / "transformer"
MODEL_DIR = ROOT / "models" / "roberta_fever_baseline"

MODEL_NAME = "FacebookAI/roberta-base"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(model, loader, device, use_fp16):
    model.eval()

    all_predictions = []
    all_labels = []
    total_loss = 0.0

    with torch.inference_mode():
        for batch in loader:
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_fp16,
            ):
                outputs = model(**batch)

            total_loss += outputs.loss.item()

            predictions = outputs.logits.argmax(dim=-1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(len(LABELS))),
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(LABELS))),
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "validation_loss": total_loss / max(len(loader), 1),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Gradient accumulation steps must be positive.")

    train_path = DATA_DIR / "train.jsonl"
    validation_path = DATA_DIR / "validation.jsonl"

    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError(
            "Training datasets not found. Run "
            "scripts/prepare_transformer_dataset.py first."
        )

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_fp16 = device.type == "cuda" and not args.no_fp16

    print("=" * 65)
    print("TRANSFORMER VERIFIER TRAINING")
    print("=" * 65)
    print("Model:", MODEL_NAME)
    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    print("FP16:", use_fp16)
    print("Batch size:", args.batch_size)
    print("Gradient accumulation:", args.gradient_accumulation_steps)
    print("Effective batch size:", args.batch_size * args.gradient_accumulation_steps)
    print("Max sequence length:", args.max_length)
    print("Epochs:", args.epochs)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading tokenizer...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = FeverTransformerDataset(
        train_path,
        tokenizer,
        max_length=args.max_length,
    )

    validation_dataset = FeverTransformerDataset(
        validation_path,
        tokenizer,
        max_length=args.max_length,
    )

    collator = TransformerCollator(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )

    print("Training records:", f"{len(train_dataset):,}")
    print("Validation records:", f"{len(validation_dataset):,}")

    print("\nLoading pretrained RoBERTa...", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )

    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.10)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_fp16,
    )

    best_macro_f1 = -1.0
    history = []

    start_time = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = 0.0
        optimizer_steps = 0

        print(f"\n{'=' * 65}")
        print(f"EPOCH {epoch + 1}/{args.epochs}")
        print("=" * 65)

        for step, batch in enumerate(train_loader):
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_fp16,
            ):
                outputs = model(**batch)
                loss = outputs.loss

            epoch_loss += loss.item()

            scaled_loss = loss / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()

            should_update = (
                (step + 1) % args.gradient_accumulation_steps == 0
                or (step + 1) == len(train_loader)
            )

            if should_update:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                scaler.step(optimizer)
                scaler.update()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                optimizer_steps += 1

            if (step + 1) % 100 == 0:
                print(
                    f"Batch {step + 1}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}",
                    flush=True,
                )

        average_train_loss = epoch_loss / max(len(train_loader), 1)

        print("\nRunning validation...", flush=True)

        validation_metrics = evaluate_model(
            model,
            validation_loader,
            device,
            use_fp16,
        )

        epoch_result = {
            "epoch": epoch + 1,
            "training_loss": average_train_loss,
            "optimizer_steps": optimizer_steps,
            **validation_metrics,
        }

        history.append(epoch_result)

        print(f"\nTraining loss: {average_train_loss:.4f}")
        print(f"Validation loss: {validation_metrics['validation_loss']:.4f}")
        print(f"Accuracy: {validation_metrics['accuracy']:.4f}")
        print(f"Macro-F1: {validation_metrics['macro_f1']:.4f}")

        for label in LABELS:
            metrics = validation_metrics["classification_report"][label]

            print(
                f"{label}: "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, "
                f"F1={metrics['f1-score']:.4f}"
            )

        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]

            best_dir = MODEL_DIR / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

            print("\nBest model saved:", best_dir)

        report_path = MODEL_DIR / "training_metrics.json"

        report = {
            "model": MODEL_NAME,
            "task": "Evidence-conditioned FEVER three-class verification",
            "training_file": str(train_path.relative_to(ROOT)),
            "validation_file": str(validation_path.relative_to(ROOT)),
            "label_order": LABELS,
            "top_k_evidence": args.top_k,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "fp16": use_fp16,
            "gradient_checkpointing": not args.no_gradient_checkpointing,
            "seed": args.seed,
            "best_validation_macro_f1": best_macro_f1,
            "history": history,
        }

        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print("=" * 65)
    print("Best validation Macro-F1:", f"{best_macro_f1:.4f}")
    print("Elapsed minutes:", f"{elapsed / 60:.2f}")
    print("Best model:", MODEL_DIR / "best_model")
    print("Metrics:", MODEL_DIR / "training_metrics.json")

    train_dataset.close()
    validation_dataset.close()


if __name__ == "__main__":
    main()
