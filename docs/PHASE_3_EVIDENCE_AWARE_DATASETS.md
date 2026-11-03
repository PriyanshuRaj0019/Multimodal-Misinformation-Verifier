
# Phase 3 — Evidence-Aware Dataset Engineering

## 1. Overview

**Project:** Multimodal Misinformation Verifier
**Phase:** 3 — Evidence-Aware Dataset Engineering
**Status:** Completed

Phase 3 focused on preparing datasets that connect claims with retrieved evidence.

The objective was to move beyond claim-only classification and establish a dataset format suitable for evidence-conditioned machine learning.

## 2. Objectives

- Combine claim records with retrieved evidence.
- Preserve original claim labels.
- Retain gold evidence annotations for evaluation.
- Store retrieved evidence separately from gold evidence.
- Preserve evidence identity and retrieval metadata.
- Prepare reproducible training and validation datasets.

## 3. Evidence-Aware Dataset Design

The V2 retrieval dataset is stored at:

```text
data/processed/
└── fever_train_retrieved_evidence_v2_full_candidate.jsonl
```

Previously verified dataset size:

| Property                    |   Value |
| --------------------------- | ------: |
| Total records               | 145,449 |
| SUPPORTS                    |  80,035 |
| REFUTES                     |  29,775 |
| NOT ENOUGH INFO             |  35,639 |
| Records with empty evidence |      93 |

## 4. Dataset Schema

Each JSONL record contains fields including:

| Field                  | Description                       |
| ---------------------- | --------------------------------- |
| `id`                 | Original FEVER example identifier |
| `claim`              | Claim being verified              |
| `label`              | Ground-truth verification label   |
| `gold_evidence`      | Annotated gold evidence           |
| `bm25_evidence`      | BM25 retrieval candidates         |
| `title_evidence`     | Title-based retrieval candidates  |
| `retrieved_evidence` | Combined retrieved evidence       |
| `retrieval_metadata` | Retrieval-related metadata        |

### Retrieved evidence structure

A retrieved evidence item contains fields such as:

```json
{
  "sentence_text": "Retrieved Wikipedia sentence.",
  "page_title": "Example_Page",
  "sentence_id": 12,
  "bm25_rank": 1,
  "title_rank": null,
  "rrf_score": 0.01639344262295082,
  "retrieval_source": "BM25",
  "final_rank": 1
}
```

The example illustrates the schema; it is not a claim about a particular dataset record.

## 5. Gold Evidence and Retrieved Evidence

The distinction between gold evidence and retrieved evidence is fundamental.

### Gold evidence

Gold evidence represents the annotated evidence associated with a FEVER claim.

The V2 dataset stores it as a flat list of dictionaries containing:

- `page_title`
- `sentence_id`

Example:

```json
[
  {
    "page_title": "Example_Page",
    "sentence_id": 0
  },
  {
    "page_title": "Another_Page",
    "sentence_id": 3
  }
]
```

### Retrieved evidence

Retrieved evidence represents candidate sentences found by the retrieval infrastructure.

It is used as model input.

**Critical data-leakage prevention rule:** Gold evidence must not be provided to the model as an input feature during training or inference. It is reserved for evaluation and auditing.

## 6. Transformer Dataset Preparation

The Transformer-specific datasets are stored in:

```text
data/processed/transformer/
├── train.jsonl
└── validation.jsonl
```

The preparation process uses a grouped claim split to reduce exact duplicate claim leakage between training and validation.

The recorded split configuration:

- GroupShuffleSplit
- Validation fraction: 10%
- Random state: 42

Previously observed validation size: 14,540 examples.

Each Transformer dataset record contains:

```json
{
  "id": 12345,
  "claim": "Example claim.",
  "label": "SUPPORTS",
  "evidence": [
    {
      "page_title": "Example_Page",
      "sentence_text": "Retrieved evidence sentence.",
      "final_rank": 1
    }
  ]
}
```

The Transformer dataset retains retrieved evidence text and page titles for model input. It does not include gold evidence as model input.

## 7. Dataset Processing Principles

- Preserve original example IDs.
- Keep claim labels consistent.
- Separate gold and retrieved evidence.
- Preserve evidence ranking.
- Use consistent evidence serialization.
- Maintain training/validation separation.
- Avoid rebuilding existing large indexes unnecessarily.

## 8. Phase Deliverables

- V2 evidence-aware FEVER dataset.
- Combined retrieval candidates.
- Retrieval metadata.
- Transformer training dataset.
- Transformer validation dataset.
- Gold-versus-retrieved evidence separation.
- Reproducible dataset preparation workflow.

## 9. Scope Limitations

The evidence-aware dataset is still text-based.

Image, audio, and video evidence are not part of the implemented dataset pipeline at this stage.

The dataset also retains a flat gold evidence representation. It should not be interpreted as preserving every alternative FEVER evidence annotation group.

## 10. Completion Criteria

Phase 3 establishes the data pipeline required for evidence-conditioned verification while maintaining separation between model inputs and evaluation annotations.

**Status: Completed.**
