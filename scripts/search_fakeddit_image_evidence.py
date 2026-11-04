
import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import CLIPModel, CLIPProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "openai/clip-vit-base-patch32"

DATA_DIR = PROJECT_ROOT / "data/processed/fakeddit/clip_pilot"

EMBEDDINGS_FILE = DATA_DIR / "image_embeddings.npy"
METADATA_FILE = DATA_DIR / "clip_similarity_results.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_projected_features(output, projection):
    """Handle different Transformers CLIP feature output formats."""

    if isinstance(output, torch.Tensor):
        features = output
    else:
        features = getattr(output, "pooler_output", None)

        if features is None:
            raise ValueError("CLIP did not return usable pooled features.")

    feature_size = features.shape[-1]

    if feature_size == projection.out_features:
        return features

    if feature_size == projection.in_features:
        return projection(features)

    raise ValueError(
        f"Unexpected feature dimension: {feature_size}"
    )


def encode_query(query, model, processor):
    """Convert a text query into a normalized CLIP embedding."""

    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    context = (
        torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )
        if DEVICE == "cuda" and torch.cuda.is_bf16_supported()
        else nullcontext()
    )

    with torch.inference_mode(), context:
        output = model.get_text_features(**inputs)

        features = get_projected_features(
            output,
            model.text_projection,
        )

        features = torch.nn.functional.normalize(
            features.float(),
            p=2,
            dim=-1,
        )

    return features.cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser(
        description="Search Fakeddit images using a text query."
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Describe the image you want to retrieve.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to display.",
    )

    args = parser.parse_args()

    if not args.query.strip():
        raise ValueError("Query cannot be empty.")

    if args.top_k <= 0:
        raise ValueError("top-k must be greater than zero.")

    if not EMBEDDINGS_FILE.is_file():
        raise FileNotFoundError(EMBEDDINGS_FILE)

    if not METADATA_FILE.is_file():
        raise FileNotFoundError(METADATA_FILE)

    image_embeddings = np.load(
        EMBEDDINGS_FILE,
        allow_pickle=False,
    )

    metadata = pd.read_csv(METADATA_FILE)

    if image_embeddings.ndim != 2:
        raise ValueError("Image embeddings must be a 2D array.")

    if image_embeddings.shape[1] != 512:
        raise ValueError(
            f"Expected 512-dimensional embeddings, "
            f"got {image_embeddings.shape[1]}."
        )

    if len(metadata) != len(image_embeddings):
        raise ValueError(
            "Embedding count and metadata row count do not match."
        )

    required_columns = {"id", "clean_title", "local_image_path"}

    if not required_columns.issubset(metadata.columns):
        raise ValueError(
            f"Missing metadata columns: "
            f"{required_columns - set(metadata.columns)}"
        )

    if metadata["id"].duplicated().any():
        raise ValueError("Duplicate image IDs found.")

    # Verify that every indexed image still exists.
    image_paths = [
        PROJECT_ROOT / path
        for path in metadata["local_image_path"]
    ]

    missing_images = [
        str(path) for path in image_paths
        if not path.is_file()
    ]

    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} indexed images are missing. "
            f"First missing image: {missing_images[0]}"
        )

    print(f"Loading CLIP model on {DEVICE}...")

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    model = CLIPModel.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,
    )

    model.to(DEVICE)
    model.eval()

    print(f"\nQuery: {args.query}")

    query_embedding = encode_query(
        args.query,
        model,
        processor,
    )

    # Normalize stored vectors before computing cosine similarity.
    image_embeddings = image_embeddings.astype(
        np.float32,
        copy=False,
    )

    norms = np.linalg.norm(
        image_embeddings,
        axis=1,
        keepdims=True,
    )

    if np.any(norms == 0):
        raise ValueError("Zero-length image embedding detected.")

    normalized_embeddings = image_embeddings / norms

    similarities = normalized_embeddings @ query_embedding

    top_indices = np.argsort(similarities)[::-1][
        :min(args.top_k, len(similarities))
    ]

    print("\n" + "=" * 75)
    print("IMAGE EVIDENCE RETRIEVAL RESULTS")
    print("=" * 75)

    for rank, index in enumerate(top_indices, start=1):
        record = metadata.iloc[index]

        image_path = image_paths[index]

        print(f"\nRank: {rank}")
        print(f"ID: {record['id']}")
        print(f"Similarity: {similarities[index]:.4f}")
        print(f"Title: {record['clean_title']}")
        print(f"Image: {image_path}")

    print("\n" + "=" * 75)
    print(f"Retrieved {len(top_indices)} images from {len(metadata)} images.")
    print("Similarity is a retrieval score, not a truth probability.")
    print("=" * 75)


if __name__ == "__main__":
    main()
