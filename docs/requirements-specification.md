# Software Requirements Specification

## 1. Document Information

- Project: Multimodal Misinformation Verifier
- Version: 0.1.0
- Status: Draft — implementation not yet verified
- Initial release scope: Text-only MVP

## 2. Problem Statement

Online claims may be false, misleading, disputed, outdated, or unsupported by available evidence. A conventional binary classifier cannot reliably explain why a claim receives a label or distinguish a lack of evidence from evidence that contradicts the claim.

This project aims to build an evidence-grounded verification system that retrieves relevant sources, evaluates claim-evidence relationships, and communicates uncertainty and limitations.

The system is a decision-support tool, not an infallible truth detector.

## 3. Objectives

1. Accept a textual claim for analysis.
2. Retrieve relevant evidence from a defined and documented corpus.
3. preserve source identifiers and evidence provenance throughout processing.
4. Assess whether the retrieved evidence supports, contradicts, or fails to resolve the claim.
5. Return an interpretable result with evidence references and limitations.
6. Abstain or request review when evidence is insufficient, conflicting, or unreliable.
7. Evaluate retrieval and verification independently using reproducible experiments.
8. Provide automated tests and a documented API as the MVP matures.

## 4. Scope

### 4.1 Included in the initial MVP

- Text claim input and validation.
- A documented, versioned evidence corpus.
- Text normalization and evidence preprocessing.
- Baseline evidence retrieval.
- Claim-evidence relevance assessment.
- A transparent verdict policy.
- Source references for retrieved evidence.
- Explicit handling of insufficient and conflicting evidence.
- Evaluation scripts, test cases, and reproducible configuration.
- A Python package with modular components.

### 4.2 Planned after the initial MVP

- REST API and interactive user interface.
- Stronger semantic retrieval, reranking, and natural-language inference models.
- Image claim verification.
- Audio and video analysis.
- Production deployment, monitoring, and model/data drift tracking.

These capabilities are plans, not completed features.

### 4.3 Out of scope for the initial MVP

- Guaranteeing that every verdict is factually correct.
- Treating popularity, repetition, or model confidence as proof.
- Automatically declaring a publisher trustworthy without a documented method.
- Generating citations that do not resolve to actual evidence records.
- Making high-stakes decisions without human oversight.
- Claiming comprehensive internet coverage.

## 5. Intended Users

- Researchers and analysts investigating textual claims.
- Journalists and fact-checking teams conducting preliminary verification.
- Developers evaluating evidence retrieval and claim-verification pipelines.
- Portfolio reviewers assessing practical machine-learning and software-engineering skills.

## 6. Functional Requirements

### FR-01: Claim input

The system shall accept a non-empty textual claim and reject invalid or oversized input according to configurable limits.

### FR-02: Evidence corpus

The system shall read evidence records from a documented source with stable record identifiers and available provenance metadata.

### FR-03: Evidence retrieval

The system shall retrieve a configurable number of candidate evidence passages and preserve their source identifiers and relevance scores where available.

### FR-04: Evidence assessment

The system shall assess candidate passages in relation to the claim. Retrieval relevance alone shall not be treated as proof of support or contradiction.

### FR-05: Verdicts

The result schema shall support the following outcomes:

- Supported: sufficient admissible evidence supports the claim under the documented decision policy.
- Refuted: sufficient admissible evidence contradicts the claim under the documented decision policy.
- Insufficient evidence: the available evidence does not justify a supported or refuted conclusion.
- Review required: evidence conflicts materially, processing fails, or a defined review condition is triggered.

The implementation must define the operational criteria for these labels. A verdict must not be presented as objectively established merely because a model produced it.

### FR-06: Evidence references

Every returned evidence item shall retain its source ID and available source metadata. The system shall not fabricate URLs, quotations, publication dates, or source details.

### FR-07: Uncertainty and abstention

The system shall be able to return an unresolved outcome instead of forcing a binary prediction. Any confidence or score must have a documented interpretation and must not automatically be described as a calibrated probability.

### FR-08: Result explanation

The result shall include the claim, verdict, evidence references, a concise rationale when supported by the implementation, and relevant limitations or warnings.

### FR-09: Reproducibility

Experiments shall record configuration, dataset version, random seed where applicable, and relevant model or dependency versions.

### FR-10: Evaluation

Evaluation shall measure retrieval quality and verification quality separately. Results shall be reported only after running the corresponding evaluation.

### FR-11: Failure handling

Invalid inputs, missing evidence, retrieval failures, and model errors shall produce controlled, documented outcomes rather than fabricated evidence.

### FR-12: Testing

Core processing, verdict policy, evidence provenance, invalid inputs, and failure paths shall have automated tests.

## 7. Non-Functional Requirements

### NFR-01: Modularity

Input validation, preprocessing, retrieval, assessment, verdict policy, and output formatting shall be separable components.

### NFR-02: Traceability

A result shall be traceable to the evidence records and configuration used to produce it.

### NFR-03: Data quality

Data processing shall document duplicates, missing fields, label quality, class distribution, and potential leakage.

### NFR-04: Security

The system shall validate external input, avoid executing retrieved content, and treat source text as untrusted data. Secrets shall not be committed to version control.

### NFR-05: Maintainability

Core logic shall be readable, type-aware where appropriate, and covered by meaningful tests.

### NFR-06: Performance

Latency and resource use shall be measured against a documented test setup before performance targets are claimed.

### NFR-07: Reproducibility

Data splits and experiments shall be repeatable where the chosen tools permit it.

## 8. Acceptance Criteria

The initial MVP is acceptable for evaluation when:

1. A valid textual claim can be processed end to end.
2. Evidence retrieval returns real records from the configured corpus.
3. Evidence IDs and available provenance survive through the result.
4. Verdict policy distinguishes supporting evidence, contradicting evidence, and unresolved cases.
5. Conflicting evidence can trigger review rather than an unsupported forced conclusion.
6. Missing evidence does not result in invented citations.
7. Automated tests cover the primary success and failure paths.
8. An evaluation script produces reproducible metrics on a documented test split.
9. Data sources, limitations, and known failure modes are documented.
10. README status and reported results match the actual implementation.

## 9. Evaluation Plan

The evaluation design shall be finalized after the dataset and baseline are selected.

Candidate retrieval metrics:
- Recall@k
- Precision@k
- Mean Reciprocal Rank (MRR)

Candidate verification metrics:
- Per-class precision, recall, and F1
- Macro-F1
- Confusion matrix
- Coverage and abstention rate
- Error analysis by claim type and evidence availability

Calibration metrics shall be considered only if the system produces probability estimates for which calibration is meaningful.

Metrics shall not be reported before the relevant experiment has run. Dataset splits must avoid leakage from duplicate or near-duplicate claims and related evidence where practical.

## 10. Risks and Limitations

- Incomplete or outdated evidence can cause incorrect conclusions.
- Retrieval can miss relevant evidence.
- A source can be relevant without being authoritative.
- Language models can misinterpret evidence or generate unsupported explanations.
- Dataset labels may contain disagreement or historical bias.
- A claim may require temporal, geographic, or domain-specific context.
- The four verdict categories require careful operational definitions and evaluation.

## 11. Development Milestones

1. Finalize requirements and architecture.
2. Select and document a suitable dataset.
3. Perform data validation and exploratory data analysis.
4. Implement a simple retrieval baseline.
5. Establish a verification baseline and verdict policy.
6. Add unit and integration tests.
7. Evaluate errors, limitations, and abstention behavior.
8. Add an API after the core pipeline is reliable.
9. Consider image verification only after the text pipeline is evaluated.

## 12. Change Management

Changes to scope, verdict definitions, datasets, or evaluation methodology shall be reflected in this document and the relevant data, model, and architecture documentation.
