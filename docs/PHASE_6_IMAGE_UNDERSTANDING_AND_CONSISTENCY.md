
# Phase 6 — Image Understanding and Image–Text Consistency

**Status:** Complete as an experimental image-understanding pilot
**Scope:** Image validation, Fakeddit pilot data, CLIP visual/text embeddings, OCR, image–text consistency signals, and image evidence retrieval.

## 1. Objective

Phase 6 extends the project's existing text-based verification work with an initial image-processing and image-evidence pipeline.

The objective was to establish the foundations for processing images, extracting visual and textual signals, comparing image content with associated titles or claims, and retrieving images using text queries.

This phase is an experimental pilot. It does not implement complete multimodal misinformation verification or combine image and text predictions into a final risk score.

## 2. Implementation summary

| Step | Area                                   | Result                                       |
| ---- | -------------------------------------- | -------------------------------------------- |
| 6.1  | Image processing foundation            | Implemented and tested                       |
| 6.2  | Image dataset selection and management | Fakeddit metadata audit and 500-record pilot |
| 6.3  | Visual embeddings and similarity       | CLIP image/text embeddings generated         |
| 6.4  | OCR and image text extraction          | OCR implemented; captioning deferred         |
| 6.5  | Image–text consistency signals        | CLIP and lexical signals analyzed            |
| 6.6  | Image evidence retrieval               | Text-to-image retrieval smoke-tested         |
| 6.7  | Evaluation and documentation           | Validation completed and documented          |

## 3. Image processing foundation

Implemented in:

- `src/misinfo_verifier/image_processing.py`
- `tests/test_image_processing.py`

The image-processing module provides:

- Image validation for JPEG, PNG, and WEBP.
- A 15 MB maximum file-size limit.
- A 25-million-pixel maximum dimension-area limit.
- Image metadata extraction.
- SHA-256 image hashing using chunked file reads.
- EXIF orientation correction.
- Conversion to RGB.
- White-background compositing for transparent images.
- Image resizing using the LANCZOS resampling filter.
- A dedicated `ImageValidationError` exception.
- A frozen `ImageMetadata` dataclass.

The image-processing tests are included in the project's test suite.

## 4. Dataset selection and pilot preparation

### 4.1 Dataset

The pilot uses the Fakeddit multimodal dataset.

Only the metadata TSV files and a small image sample were used. The full 102 GB image archive was not downloaded.

The downloaded metadata splits were:

- `multimodal_train.tsv`
- `multimodal_validate.tsv`
- `multimodal_test_public.tsv`

The metadata audit reported:

| Split           |           Records | Missing image URLs | Duplicate IDs |
| --------------- | ----------------: | -----------------: | ------------: |
| Train           |           564,000 |              1,534 |             0 |
| Validation      |            59,342 |                173 |             0 |
| Public test     |            59,319 |                156 |             0 |
| **Total** | **682,661** |    **1,863** |   **0** |

No overlapping IDs were found across the three splits.

### 4.2 Pilot manifest

A reproducible 500-record pilot manifest was generated from the training split.

- Sampling seed: `42`
- Sampling strategy: balanced on the Fakeddit 2-way label
- Label 0: 250 records
- Label 1: 250 records

The manifest is stored locally at:

`data/raw/fakeddit/manifests/phase6_train_pilot_manifest.csv`

### 4.3 Image download results

The pilot downloader retrieved images using the metadata image URLs.

| Metric                      |                 Result |
| --------------------------- | ---------------------: |
| Requested images            |                    500 |
| Successfully downloaded     |                    476 |
| Failed downloads            |                     24 |
| Download success rate       |                  95.2% |
| Total downloaded image size | Approximately 47.12 MB |

The 24 failures consisted of 23 HTTP 404 responses and one HTTP 500 response.

The downloader includes host allowlisting, redirect destination checks, file-size limits, timeouts, transient retries, image validation, ID-based filenames, and incremental manifest updates.

These controls reduce download risks but do not constitute a complete SSRF security boundary.

### 4.4 Image integrity audit

All 476 downloaded images passed the integrity audit.

| Measurement              | Result |
| ------------------------ | -----: |
| Valid images             |    476 |
| Invalid images           |      0 |
| Missing downloaded files |      0 |
| Orphan image files       |      0 |
| JPEG images              |    460 |
| PNG images               |     16 |
| Unique SHA-256 hashes    |    471 |

One duplicate-image group contained six records. This is a potential data-leakage concern and must be considered when designing future evaluation splits.

Download availability was not identical across the two pilot labels:

- Label 0: 244 of 250 downloaded.
- Label 1: 232 of 250 downloaded.

This differential availability is a potential source of sampling bias.

## 5. CLIP visual and text embeddings

### 5.1 Model

The pilot uses:

`openai/clip-vit-base-patch32`

The model was run using PyTorch and Hugging Face Transformers on an NVIDIA RTX 4060 Laptop GPU with 8 GB VRAM.

The embedding pipeline produces normalized 512-dimensional image and text vectors.

### 5.2 Embedding outputs

The pipeline processed all 476 valid images.

| Output           | Shape          |
| ---------------- | -------------- |
| Image embeddings | `(476, 512)` |
| Text embeddings  | `(476, 512)` |

Embeddings were saved locally as:

- `data/processed/fakeddit/clip_pilot/image_embeddings.npy`
- `data/processed/fakeddit/clip_pilot/text_embeddings.npy`

The corresponding similarity results were saved as:

`data/processed/fakeddit/clip_pilot/clip_similarity_results.csv`

### 5.3 Image–text cosine similarity

The normalized image and text embeddings were compared using cosine similarity.

| Statistic          | Similarity |
| ------------------ | ---------: |
| Mean               |     0.2713 |
| Standard deviation |     0.0541 |
| Minimum            |     0.1273 |
| 25th percentile    |     0.2298 |
| Median             |     0.2778 |
| 75th percentile    |     0.3122 |
| Maximum            |     0.3874 |

Similarity grouped by Fakeddit's 2-way label:

| Label | Samples | Mean similarity | Median similarity |
| ----- | ------: | --------------: | ----------------: |
| 0     |     244 |          0.2443 |            0.2349 |
| 1     |     232 |          0.2997 |            0.3024 |

These are descriptive observations from a small pilot. They do not establish that either label represents factual truth or image–text consistency.

A CLIP similarity score is not a probability of truth, authenticity, or misinformation.

## 6. OCR extraction

OCR was implemented using RapidOCR.

The OCR pipeline processes the 476 pilot images and records extracted text, confidence-related information, processing status, and timing information.

Results:

| OCR outcome       |        Images |
| ----------------- | ------------: |
| Text detected     |           188 |
| No text detected  |           288 |
| Processing errors |             0 |
| **Total**   | **476** |

Output:

`data/processed/fakeddit/clip_pilot/ocr_results.csv`

Image captioning was considered in the initial plan but was not implemented in this phase. It is deferred.

## 7. Image–text consistency signals

The consistency analysis joins CLIP similarity results and OCR results by sample ID.

It calculates:

- CLIP image–text cosine similarity.
- Token overlap between the associated title and OCR text.
- Shared token count.
- Title-token recall.
- OCR-token precision.
- Jaccard similarity.
- OCR evidence availability.
- A descriptive lexical-overlap signal.

Outputs:

- `reports/phase6_image_text_consistency/image_text_consistency_results.csv`
- `reports/phase6_image_text_consistency/analysis_summary.json`

### 7.1 Pilot observations

| Signal                      | Samples |
| --------------------------- | ------: |
| OCR text available          |     188 |
| No OCR text detected        |     288 |
| Lexical overlap observed    |      32 |
| No lexical overlap observed |     129 |
| Not assessable              |     315 |

The lexical signal is not a semantic contradiction detector. A lack of shared words does not imply that an image contradicts its title or claim.

Likewise, OCR text may be absent even when an image is relevant to the claim.

These outputs are signals for subsequent analysis, not final verification decisions.

## 8. Image evidence retrieval

A text-to-image retrieval script was implemented using the CLIP text encoder and stored image embeddings.

Script:

`scripts/search_fakeddit_image_evidence.py`

The command-line interface accepts a text query and a configurable top-k value. It encodes the query, normalizes the vector, compares it with the stored image embeddings, and returns the highest-scoring image records with their IDs, titles, similarity scores, and paths.

Two functional smoke tests were performed:

**Query:** `a historical British army poster`

The top result was a British recruitment poster with a similarity score of approximately `0.3213`.

**Query:** `a person holding a Pepsi can`

The top result was the matching record titled “my sister in a field holding a can of pepsi”, with a similarity score of approximately `0.3131`.

These checks demonstrate that the retrieval pipeline executes and can return relevant examples. They are not a formal retrieval benchmark and do not establish measured retrieval accuracy.

Retrieval scores are similarity scores, not truth probabilities.

## 9. Evaluation and validation

The complete project test suite passed after Phase 6 implementation.

| Validation   | Result    |
| ------------ | --------- |
| Pytest suite | 77 passed |
|              |           |
