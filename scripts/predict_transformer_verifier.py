import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from misinfo_verifier.transformer_data import (
    LABELS,
    build_evidence_context,
)


DEFAULT_MODEL_DIR = ROOT / "models" / "roberta_fever_baseline" / "best_model"


def predict(record, model, tokenizer, device, max_length=512, top_k=5):
    claim = record.get("claim")

    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("Input record must contain a non-empty claim.")

    evidence = record.get("retrieved_evidence", [])

    context = build_evidence_context(
        evidence,
        top_k=top_k,
    )

    inputs = tokenizer(
        claim,
        context,
        return_tensors="pt",
        max_length=max_length,
        truncation="longest_first",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    model.eval()

    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    predicted_id = int(probabilities.argmax().item())

    return {
        "claim": claim,
        "prediction": LABELS[predicted_id],
        "confidence": float(probabilities[predicted_id].item()),
        "class_probabilities": {
            LABELS[index]: float(probabilities[index].item())
            for index in range(len(LABELS))
        },
        "evidence_sentences_used": min(
            top_k,
            len(evidence) if isinstance(evidence, list) else 0,
        ),
        "model": "RoBERTa-base",
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to a JSON object containing claim and retrieved_evidence.",
    )

    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
    )

    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    input_path = Path(args.input_json)
    model_path = Path(args.model_dir)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. "
            "Train the model first."
        )

    record = json.loads(input_path.read_text(encoding="utf-8"))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    ).to(device)

    result = predict(
        record,
        model,
        tokenizer,
        device,
        max_length=args.max_length,
        top_k=args.top_k,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
