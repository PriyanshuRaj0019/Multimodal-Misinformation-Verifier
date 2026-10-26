# Multimodal Misinformation Verifier

An evidence-grounded misinformation verification system designed to evaluate claims against supporting or refuting evidence, communicate uncertainty, and provide traceable explanations.

**Project status:** Dataset exploration and audit completed. Evidence retrieval and model development are upcoming.

## Overview

Misinformation verification requires more than classifying a statement as true or false. A reliable system must retrieve relevant evidence, assess whether that evidence supports or refutes a claim, and recognize when the available information is insufficient.

This project aims to develop a modular verification pipeline that prioritizes evidence traceability, measurable performance, uncertainty handling, and reproducibility.

## Objectives

- Retrieve relevant evidence for a given claim.
- Classify claims into `SUPPORTS`, `REFUTES`, or `NOT ENOUGH INFO`.
- Connect predictions to the evidence used to produce them.
- Evaluate retrieval quality and classification performance separately.
- Identify failure cases, uncertainty, and limitations.
- Build toward a multimodal architecture that can incorporate text, images, audio, and video.

These are project objectives, not claims that all features have already been implemented.

## Dataset

The initial dataset exploration uses the FEVER benchmark:

| Split       | Records | Purpose                      |
| ----------- | ------: | ---------------------------- |
| Train       | 145,449 | Training-data exploration    |
| Development |  19,998 | Development-data exploration |

The development split contains 6,666 examples per label in the selected file.

**Labels**

- `SUPPORTS`
- `REFUTES`
- `NOT ENOUGH INFO`

The dataset audit identified duplicate claim texts and some claims associated with conflicting labels. These cases require careful treatment rather than automatic deletion or relabeling.

Raw datasets are kept locally and excluded from Git.

## Current Progress

### Completed

- Initial repository structure and documentation scaffold
- FEVER training and development dataset exploration
- Label distribution and claim-length analysis
- Evidence annotation structure inspection
- Missing-value and duplicate-claim analysis
- Conflicting-label and train/development overlap reports
- Representative evidence cases inspected
- EDA and audit scripts committed to the repository

### Next

1. Inspect and validate the Wikipedia evidence corpus.
2. Map FEVER evidence annotations to source sentences.
3. Implement a reproducible evidence-loading pipeline.
4. Build a baseline retrieval system.
5. Evaluate retrieval using metrics such as Recall@k.
6. Develop and evaluate claim-verification models.

## Planned Architecture

The intended system is organized into modular stages:

1. **Input processing** — accept and validate claims and supported media inputs.
2. **Evidence retrieval** — retrieve relevant source passages or sentences.
3. **Evidence analysis** — assess claim–evidence relationships.
4. **Verification** — predict `SUPPORTS`, `REFUTES`, or `NOT ENOUGH INFO`.
5. **Reliability layer** — report uncertainty and handle insufficient evidence.
6. **Result presentation** — return predictions with traceable evidence and limitations.

The architecture will evolve as individual components are implemented and evaluated.

## Repository Structure

```text
Multimodal-Misinformation-Verifier/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
│   ├── architecture.md
│   ├── data-card.md
│   ├── model-card.md
│   └── requirements-specification.md
├── notebooks/
├── reports/
│   └── eda/
├── scripts/
│   ├── audit_fever.py
│   ├── eda_fever.py
│   └── inspect_fever_cases.py
├── src/
│   └── misinfo_verifier/
└── tests/
```

## Reproducibility

Use Python 3.12 and a virtual environment.

Create the environment:

```powershell
python -m venv .venv
```

Install the packages needed for the current EDA scripts:

```powershell
.\.venv\Scripts\python.exe -m pip install pandas matplotlib
```

Run the dataset exploration:

```powershell
.\.venv\Scripts\python.exe scripts/eda_fever.py
```

Run the dataset audit:

```powershell
.\.venv\Scripts\python.exe scripts/audit_fever.py
```

Inspect representative cases:

```powershell
.\.venv\Scripts\python.exe scripts/inspect_fever_cases.py
```

These commands assume the FEVER JSONL files are available locally at the paths expected by the scripts. Later pipeline stages may require additional dependencies.

## Evaluation Strategy

The project will report relevant metrics at each stage:

- **Retrieval:** Recall@k and evidence coverage
- **Classification:** precision, recall, F1-score, macro-F1, and confusion matrix
- **Reliability:** confidence calibration, abstention behavior, and error analysis
- **Engineering:** reproducibility, latency, input validation, and test coverage

Metrics will be reported only after the corresponding component has been implemented and evaluated.

## Engineering Principles

- Evidence-backed predictions instead of unsupported assertions
- Explicit uncertainty when evidence is insufficient
- Separation of retrieval quality from classification quality
- Reproducible data processing and evaluation
- No fabricated performance metrics
- Clear documentation of dataset limitations and model failures
- Modular design for future multimodal support

## Limitations

The current work covers dataset exploration and auditing. A functioning evidence retrieval engine, trained verification model, multimodal processing pipeline, and deployed API are not yet established by this milestone.

FEVER is a benchmark dataset, not a complete representation of real-world misinformation. Benchmark performance alone will not establish factual correctness across arbitrary claims or domains.

## License and Data

Review the applicable FEVER dataset license and the licenses of any additional corpora or pretrained models before redistribution. Raw datasets are not included in this repository.

## Author

**Priyanshu Raj**

GitHub: [PriyanshuRaj0019](https://github.com/PriyanshuRaj0019)

Repository: [Multimodal Misinformation Verifier](https://github.com/PriyanshuRaj0019/Multimodal-Misinformation-Verifier)
