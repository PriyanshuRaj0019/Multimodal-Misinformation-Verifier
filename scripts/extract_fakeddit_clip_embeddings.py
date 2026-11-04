
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor

from misinfo_verifier.image_processing import validate_image


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path.cwd().resolve()

MANIFEST_FILE = (
    PROJECT_ROOT
    / "data/raw/fakeddit/manifests/"
    / "phase6_train_pilot_download_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/fakeddit/clip_pilot"
)

MODEL_NAME = "openai/clip-vit-base-patch32"

BATCH_SIZE = 8
MAX_TEXT_LENGTH = 77

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_BF16 = (
    DEVICE.type == "cuda"
    and torch.cuda.is_bf16_supported()
)

IMAGE_EMBEDDINGS_FILE = (
    OUTPUT_DIR / "image_embeddings.npy"
)

TEXT_EMBEDDINGS_FILE = (
    OUTPUT_DIR / "text_embeddings.npy"
)

RESULTS_FILE = (
    OUTPUT_DIR / "clip_similarity_results.csv"
)


# ============================================================
# 2. VALIDATE INPUT AND OUTPUT PATHS
# ============================================================

if not MANIFEST_FILE.is_file():
    raise FileNotFoundError(
        f"Download manifest not found: {MANIFEST_FILE}"
    )

existing_outputs = [
    path
    for path in [
        IMAGE_EMBEDDINGS_FILE,
        TEXT_EMBEDDINGS_FILE,
        RESULTS_FILE,
    ]
    if path.exists()
]

if existing_outputs:
    raise FileExistsError(
        "Output files already exist. Refusing to overwrite:\n"
        + "\n".join(str(path) for path in existing_outputs)
    )

manifest = pd.read_csv(
    MANIFEST_FILE,
    dtype={"id": "string"},
)

required_columns = [
    "id",
    "clean_title",
    "2_way_label",
    "local_image_path",
    "download_status",
    "image_sha256",
]

missing_columns = [
    column
    for column in required_columns
    if column not in manifest.columns
]

if missing_columns:
    raise ValueError(
        f"Missing manifest columns: {missing_columns}"
    )

manifest = manifest.loc[
    manifest["download_status"] == "downloaded"
].copy()

manifest = manifest.reset_index(drop=True)

if manifest.empty:
    raise ValueError(
        "No successfully downloaded images found."
    )

if manifest["id"].isna().any():
    raise ValueError(
        "Missing sample IDs detected."
    )

if manifest["id"].duplicated().any():
    raise ValueError(
        "Duplicate sample IDs detected."
    )

if manifest["clean_title"].isna().any():
    raise ValueError(
        "Missing text titles detected."
    )

manifest["clean_title"] = (
    manifest["clean_title"].astype(str).str.strip()
)

if manifest["clean_title"].eq("").any():
    raise ValueError(
        "Empty text titles detected."
    )


# ============================================================
# 3. LOAD CLIP MODEL
# ============================================================

print("=" * 65)
print("MMV — CLIP IMAGE/TEXT EMBEDDING EXTRACTION")
print("=" * 65)

print(f"Device: {DEVICE}")
print(f"BF16 enabled: {USE_BF16}")
print(f"Samples: {len(manifest):,}")
print(f"Model: {MODEL_NAME}")
print(f"Batch size: {BATCH_SIZE}")

processor = CLIPProcessor.from_pretrained(
    MODEL_NAME
)

model = CLIPModel.from_pretrained(
    MODEL_NAME
)

model = model.to(DEVICE)
model.eval()


# ============================================================
# 4. EXTRACT CLIP FEATURES COMPATIBLY
# ============================================================

def get_projected_features(output, projection):
    """
    Handle both already-projected and unprojected CLIP outputs.

    If features already have the projection output dimension,
    return them directly.

    If features have the projection input dimension,
    apply the projection layer once.
    """

    if isinstance(output, torch.Tensor):
        features = output

    else:
        features = getattr(
            output,
            "pooler_output",
            None,
        )

        if features is None:
            raise TypeError(
                "Unexpected CLIP output: pooler_output not found."
            )

    if features.ndim != 2:
        raise ValueError(
            "Expected a 2D feature tensor, "
            f"received shape {tuple(features.shape)}."
        )

    feature_dimension = features.shape[-1]

    # Already projected: do not project again.
    if feature_dimension == projection.out_features:
        return features

    # Not projected yet: apply projection.
    if feature_dimension == projection.in_features:
        return projection(features)

    raise ValueError(
        "Unexpected CLIP feature dimension: "
        f"{feature_dimension}. Expected either "
        f"{projection.in_features} or "
        f"{projection.out_features}."
    )


# ============================================================
# 5. LOAD AND VALIDATE IMAGES
# ============================================================

def load_validated_image(relative_path):
    """
    Load an image from the project directory,
    validate it, correct EXIF orientation,
    and convert it to RGB.
    """

    image_path = (
        PROJECT_ROOT / str(relative_path)
    ).resolve()

    # Prevent paths from escaping the project directory.
    image_path.relative_to(PROJECT_ROOT)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image does not exist: {image_path}"
        )

    validate_image(image_path)

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)

        image.load()

        return image.convert("RGB")


# ============================================================
# 6. EXTRACT IMAGE AND TEXT EMBEDDINGS
# ============================================================

image_embeddings = []
text_embeddings = []
similarity_scores = []
processed_records = []

total = len(manifest)

with torch.inference_mode():

    for start in range(0, total, BATCH_SIZE):

        batch = manifest.iloc[
            start:start + BATCH_SIZE
        ]

        images = []
        texts = []

        records = batch.to_dict(
            orient="records"
        )

        for row in records:

            image = load_validated_image(
                row["local_image_path"]
            )

            images.append(image)

            texts.append(
                str(row["clean_title"])
            )

        # Prepare image and text inputs.
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LENGTH,
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        if USE_BF16:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):

                image_output = model.get_image_features(
                    pixel_values=inputs["pixel_values"]
                )

                text_output = model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )

                image_features = get_projected_features(
                    image_output,
                    model.visual_projection,
                )

                text_features = get_projected_features(
                    text_output,
                    model.text_projection,
                )

        else:

            image_output = model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )

            text_output = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            image_features = get_projected_features(
                image_output,
                model.visual_projection,
            )

            text_features = get_projected_features(
                text_output,
                model.text_projection,
            )

        # ----------------------------------------------------
        # Normalize embeddings
        # ----------------------------------------------------

        image_features = F.normalize(
            image_features.float(),
            p=2,
            dim=1,
        )

        text_features = F.normalize(
            text_features.float(),
            p=2,
            dim=1,
        )

        # ----------------------------------------------------
        # Calculate cosine similarity
        # ----------------------------------------------------

        similarities = (
            image_features * text_features
        ).sum(dim=1)

        # ----------------------------------------------------
        # Store results in CPU memory
        # ----------------------------------------------------

        image_embeddings.extend(
            image_features.cpu().numpy()
        )

        text_embeddings.extend(
            text_features.cpu().numpy()
        )

        similarity_scores.extend(
            similarities.cpu().numpy()
        )

        processed_records.extend(records)

        completed = min(
            start + BATCH_SIZE,
            total,
        )

        print(
            f"Processed {completed}/{total} images"
        )


# ============================================================
# 7. CONVERT RESULTS TO NUMPY ARRAYS
# ============================================================

image_array = np.asarray(
    image_embeddings,
    dtype=np.float32,
)

text_array = np.asarray(
    text_embeddings,
    dtype=np.float32,
)

similarity_array = np.asarray(
    similarity_scores,
    dtype=np.float32,
)


# ============================================================
# 8. VALIDATE EMBEDDINGS
# ============================================================

if image_array.ndim != 2:
    raise ValueError(
        f"Invalid image embedding shape: {image_array.shape}"
    )

if text_array.ndim != 2:
    raise ValueError(
        f"Invalid text embedding shape: {text_array.shape}"
    )

if image_array.shape != text_array.shape:
    raise ValueError(
        "Image and text embedding dimensions do not match."
    )

if len(image_array) != total:
    raise ValueError(
        "Unexpected number of image embeddings."
    )

if len(text_array) != total:
    raise ValueError(
        "Unexpected number of text embeddings."
    )

if not np.isfinite(image_array).all():
    raise ValueError(
        "Invalid values in image embeddings."
    )

if not np.isfinite(text_array).all():
    raise ValueError(
        "Invalid values in text embeddings."
    )

if not np.isfinite(similarity_array).all():
    raise ValueError(
        "Invalid cosine similarity values."
    )

image_norms = np.linalg.norm(
    image_array,
    axis=1,
)

text_norms = np.linalg.norm(
    text_array,
    axis=1,
)

if not np.allclose(
    image_norms,
    1.0,
    atol=1e-4,
):
    raise ValueError(
        "Image embeddings are not properly normalized."
    )

if not np.allclose(
    text_norms,
    1.0,
    atol=1e-4,
):
    raise ValueError(
        "Text embeddings are not properly normalized."
    )

if np.any(similarity_array < -1.0001) or np.any(
    similarity_array > 1.0001
):
    raise ValueError(
        "Cosine similarity is outside its valid range."
    )


# ============================================================
# 9. PREPARE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 10. SAVE EMBEDDINGS
# ============================================================

np.save(
    IMAGE_EMBEDDINGS_FILE,
    image_array,
)

np.save(
    TEXT_EMBEDDINGS_FILE,
    text_array,
)


# ============================================================
# 11. SAVE SIMILARITY RESULTS
# ============================================================

results = pd.DataFrame(
    processed_records
)

results["clip_cosine_similarity"] = (
    similarity_array
)

results["embedding_index"] = np.arange(
    len(results)
)

results.to_csv(
    RESULTS_FILE,
    index=False,
)


# ============================================================
# 12. FINAL REPORT
# ============================================================

print("\n" + "=" * 65)
print("CLIP EMBEDDING EXTRACTION COMPLETE")
print("=" * 65)

print(
    "Processed samples:",
    len(image_array),
)

print(
    "Image embedding shape:",
    image_array.shape,
)

print(
    "Text embedding shape:",
    text_array.shape,
)

print(
    "Image embedding dtype:",
    image_array.dtype,
)

print(
    "Text embedding dtype:",
    text_array.dtype,
)

print(
    "Image embedding norm range:",
    image_norms.min(),
    "to",
    image_norms.max(),
)

print(
    "Text embedding norm range:",
    text_norms.min(),
    "to",
    text_norms.max(),
)

print("\nCosine similarity statistics:")

print(
    pd.Series(
        similarity_array
    ).describe().round(4)
)

print("\nOutput files:")

print(IMAGE_EMBEDDINGS_FILE)
print(TEXT_EMBEDDINGS_FILE)
print(RESULTS_FILE)

print("\nAll embedding validation checks passed.")
