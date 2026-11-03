
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}


def build_evidence_context(evidence, top_k=5):
    """Serialize retrieved evidence into a consistent text context."""
    if not isinstance(evidence, list):
        evidence = []

    ranked = sorted(
        evidence,
        key=lambda item: (
            item.get("final_rank")
            if isinstance(item, dict)
            and isinstance(item.get("final_rank"), int)
            else 10**9
        ),
    )

    sections = []

    for index, item in enumerate(ranked, start=1):
        if len(sections) >= top_k:
            break

        if not isinstance(item, dict):
            continue

        title = str(item.get("page_title") or "").replace("_", " ")
        sentence = str(item.get("sentence_text") or "")

        if not sentence.strip():
            continue

        sections.append(
            f"[Evidence {len(sections) + 1}]\n"
            f"Source: {title}\n"
            f"Text: {sentence}"
        )

    return "\n\n".join(sections) or "No relevant evidence retrieved."


class FeverTransformerDataset(Dataset):
    """Memory-efficient random-access JSONL dataset."""

    def __init__(
        self,
        jsonl_path,
        tokenizer,
        max_length=512,
        top_k=5,
    ):
        self.path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.top_k = top_k

        self.offsets = []
        self._file = None

        if self.top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.path}"
            )

        with self.path.open("rb") as file:
            while True:
                offset = file.tell()
                line = file.readline()

                if not line:
                    break

                if line.strip():
                    self.offsets.append(offset)

        if not self.offsets:
            raise ValueError(
                f"No records found in {self.path}"
            )

    def __len__(self):
        return len(self.offsets)

    def _get_file(self):
        if self._file is None or self._file.closed:
            self._file = self.path.open("rb")

        return self._file

    def get_record(self, index):
        """Return the original JSONL record without tokenization."""
        if index < 0 or index >= len(self.offsets):
            raise IndexError(f"Dataset index out of range: {index}")

        file = self._get_file()
        file.seek(self.offsets[index])

        line = file.readline()

        if not line:
            raise ValueError(
                f"Unable to read record at index {index}"
            )

        return json.loads(line)

    def __getitem__(self, index):
        record = self.get_record(index)

        claim = record["claim"]
        evidence = record.get("evidence", [])

        context = build_evidence_context(
            evidence,
            top_k=self.top_k,
        )

        encoded = self.tokenizer(
            claim,
            context,
            max_length=self.max_length,
            truncation="longest_first",
            padding=False,
        )

        encoded["labels"] = LABEL2ID[record["label"]]

        return encoded

    def close(self):
        file = getattr(self, "_file", None)

        if file is not None and not file.closed:
            file.close()

        self._file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class TransformerCollator:
    """Dynamically pad sequences within each batch."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        labels = torch.tensor(
            [item["labels"] for item in features],
            dtype=torch.long,
        )

        model_features = [
            {
                key: value
                for key, value in item.items()
                if key != "labels"
            }
            for item in features
        ]

        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            return_tensors="pt",
        )

        batch["labels"] = labels

        return batch
