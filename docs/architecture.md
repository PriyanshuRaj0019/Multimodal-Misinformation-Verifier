# System Architecture

## 1. Document Information

- Project: Multimodal Misinformation Verifier
- Version: 0.1.0
- Status: Proposed architecture
- Target: Text-only MVP

This document describes the intended design. It does not assert that the components have been implemented.

## 2. Architectural Principles

- Evidence before verdict.
- Retrieval relevance is not equivalent to truth.
- Every evidence item must retain provenance.
- Uncertainty and conflicting evidence must be explicit.
- Components should be testable in isolation.
- Dataset and model versions must be traceable.
- The initial design should remain simple enough to evaluate and improve.

## 3. High-Level Data Flow

Claim input
    |
    v
Input validation
    |
    v
Text preprocessing
    |
    v
Evidence retrieval <--- Versioned evidence corpus
    |
    v
Candidate evidence filtering and ranking
    |
    v
Claim-evidence assessment
    |
    v
Verdict policy and uncertainty handling
    |
    v
Result assembly
    |
    v
Structured verification result

The first implementation may use simple lexical retrieval. Semantic embeddings, reranking, and stronger inference models are optional later improvements, not prerequisites for the first baseline.

## 4. Logical Components

### 4.1 Input and validation

Responsibilities:
- Accept a textual claim.
- Validate type, whitespace, and configured length limits.
- Return clear validation errors.
- Avoid interpreting arbitrary user input as executable instructions.

### 4.2 Preprocessing

Responsibilities:
- Apply consistent text normalization.
- Preserve the original claim for display and audit.
- Normalize evidence text without destroying meaningful names, dates, quantities, or negation.
- Record preprocessing configuration.

Normalization must be tested because aggressive transformations can alter claim meaning.

### 4.3 Evidence store

The evidence store shall expose records with a stable schema.

Proposed fields:
- evidence_id: stable unique identifier
- text: evidence passage
- source_title: title when available
- source_url: URL when available and verified
- publisher: publisher name when available
- published_at: publication timestamp when available
- retrieved_at: acquisition timestamp when available
- source_type: documented source category
- dataset_version: version or snapshot identifier

Unavailable metadata must remain missing rather than being invented.

### 4.4 Retrieval

Responsibilities:
- Search the configured evidence corpus.
- Return a bounded list of candidate passages.
- Preserve evidence IDs and source metadata.
- Record retrieval method and parameters.

Initial baseline:
- A simple lexical retrieval method such as TF-IDF or BM25, selected after reviewing the dataset and available dependencies.

Potential future improvements:
- Dense embedding retrieval.
- Hybrid lexical and dense retrieval.
- Cross-encoder reranking.
- Query reformulation.

Retrieval performance must be measured independently from final verdict accuracy.

### 4.5 Claim-evidence assessment

Responsibilities:
- Compare a claim with each candidate evidence passage.
- Determine whether the passage supports, contradicts, or does not resolve the claim.
- Preserve the link between assessment and evidence ID.
- Handle irrelevant and ambiguous passages.

A retrieval score alone shall not determine support or refutation. The first assessment baseline may use documented rules or a selected model, but its limitations must be evaluated.

### 4.6 Verdict policy

The verdict policy combines evidence assessments and applies explicit decision rules.

Possible outputs:
- Supported
- Refuted
- Insufficient evidence
- Review required

The policy must account for:
- Evidence quality and relevance.
- Agreement or disagreement among evidence items.
- Missing or incomplete evidence.
- Contradictory sources.
- Configured decision thresholds, if applicable.

Thresholds must be selected using validation data, not chosen to make reported results look better. The policy should prefer unresolved outcomes when the evidence does not justify a conclusion.

### 4.7 Result assembly

Proposed response fields:
- claim
- verdict
- rationale
- evidence: list of evidence references and assessment details
- warnings
- unresolved_questions
- pipeline_version
- dataset_version

Optional scores may be included only when their meaning is documented. They must not be described as calibrated confidence without appropriate validation.

### 4.8 Evaluation

Responsibilities:
- Evaluate retrieval separately from verification.
- Compute documented metrics on fixed data splits.
- Record experiment configuration and versions.
- Produce error analyses and reproducible outputs.
- Test abstention behavior and evidence provenance.

### 4.9 API boundary — future milestone

A REST API may later expose the pipeline through a framework such as FastAPI.

Expected concerns:
- Request and response schemas.
- Input validation.
- Controlled error responses.
- Request size limits and timeouts.
- Logging that avoids unnecessary sensitive input retention.
- Health checks and API tests.

The API is not considered implemented until its code and tests exist.

## 5. Suggested Package Boundaries

Proposed source layout:

src/misinfo_verifier/
    __init__.py
    schemas.py
    preprocessing.py
    retrieval.py
    assessment.py
    verdict.py
    pipeline.py
    evaluation.py

This is a proposed target layout. Add modules when their responsibilities are ready to implement rather than creating unnecessary empty abstractions.

## 6. Configuration and Reproducibility

Configuration should separate code from changeable settings, including:
- Dataset paths and versions.
- Retrieval method and top-k.
- Model identifier and version, if applicable.
- Decision thresholds, if applicable.
- Random seeds.
- Evaluation split and metric settings.

Use a version-controlled example configuration. Do not commit API keys, passwords, or other secrets.

## 7. Data and Model Lifecycle

1. Acquire a dataset from a documented source.
2. Record its license, provenance, intended use, and limitations.
3. Validate schema, missing values, duplicates, and labels.
4. Inspect distributions and potential leakage.
5. Create reproducible train, validation, and test splits as appropriate.
6. Fit preprocessing and retrieval artifacts only on the appropriate training data.
7. Tune parameters on validation data.
8. Evaluate once the protocol is fixed.
9. Save metrics, configuration, and artifact versions.
10. Document limitations and failure cases.

A benchmark dataset may provide labels or evidence annotations that are unavailable in ordinary inference. Evaluation code must not accidentally use test labels as model inputs.

## 8. Testing Strategy

### Unit tests
- Input validation.
- Text preprocessing.
- Evidence schema validation.
- Retrieval output structure.
- Verdict rules.
- Missing and conflicting evidence.
- Result serialization.

### Integration tests
- Claim-to-result pipeline.
- Evidence ID and source metadata preservation.
- Empty-corpus behavior.
- Retrieval and assessment failure handling.

### Evaluation tests
- Fixed small examples with known expected behavior.
- Metric computation on controlled inputs.
- Repeatability of data splits and configured experiments.

### Safety and robustness tests
- Empty, malformed, and oversized claims.
- Untrusted instructions embedded in retrieved text.
- Missing or invalid source metadata.
- Conflicting evidence and unsupported conclusions.

## 9. Observability and Operational Readiness

Before deployment, consider recording:
- Request and pipeline version identifiers.
- Retrieval duration and candidate counts.
- Component errors and timeouts.
- Verdict distribution.
- Abstention and review rates.
- Evidence source coverage.

Avoid logging sensitive content unnecessarily. Production monitoring and drift detection are future work until implemented and tested.

## 10. Future Multimodal Extension

Images, audio, and video are outside the initial text-only MVP.

A future multimodal design may introduce modality-specific ingestion and extraction components that produce normalized claims or evidence with timestamps, source references, and transformation metadata. These components must preserve provenance and uncertainty.

Adding a modality requires its own data documentation, evaluation protocol, tests, and failure analysis. Text-only performance must not be presented as evidence of multimodal capability.

## 11. Known Architectural Risks

- Retrieval misses relevant evidence.
- Duplicate or correlated sources create false agreement.
- Evidence may be stale or context-dependent.
- Claim decomposition may lose qualifiers or temporal context.
- Assessment models may confuse topical similarity with entailment.
- Thresholds may fail under distribution shift.
- Evaluation datasets may not reflect real-world claims.

These risks must be evaluated empirically rather than assumed to be solved by the architecture.
