
import csv
from pathlib import Path

from PIL import Image, ImageOps
from rapidocr_onnxruntime import RapidOCR

from misinfo_verifier.image_processing import validate_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fakeddit"
    / "clip_pilot"
    / "clip_similarity_results.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fakeddit"
    / "clip_pilot"
    / "ocr_results.csv"
)

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fakeddit"
    / "images"
    / "train_pilot"
)


def extract_ocr_result(ocr_output):
    """Parse RapidOCR output safely."""

    ocr_result, elapsed_time = ocr_output

    # RapidOCR returns None when it finds no text.
    if not ocr_result:
        return {
            "ocr_text": "",
            "detected_text_regions": 0,
            "mean_ocr_confidence": None,
            "ocr_elapsed_seconds": (
                sum(float(value) for value in elapsed_time if value is not None)
                if isinstance(elapsed_time, (list, tuple))
                else float(elapsed_time)
                if elapsed_time is not None
                else None
            ),
            "ocr_status": "no_text_detected",
        }

    detected_regions = []

    for detection in ocr_result:
        if not isinstance(detection, (list, tuple)) or len(detection) < 3:
            continue

        detected_text = str(detection[1]).strip()

        if not detected_text:
            continue

        confidence = float(detection[2])

        detected_regions.append((detected_text, confidence))

    ocr_text = " ".join(
        text for text, confidence in detected_regions
    )

    detected_count = len(detected_regions)

    mean_confidence = (
        sum(confidence for text, confidence in detected_regions)
        / detected_count
        if detected_count
        else None
    )

    # RapidOCR may return a list of three timing measurements.
    if isinstance(elapsed_time, (list, tuple)):
        elapsed_seconds = sum(
            float(value)
            for value in elapsed_time
            if value is not None
        )
    elif elapsed_time is not None:
        elapsed_seconds = float(elapsed_time)
    else:
        elapsed_seconds = None

    return {
        "ocr_text": ocr_text,
        "detected_text_regions": detected_count,
        "mean_ocr_confidence": mean_confidence,
        "ocr_elapsed_seconds": elapsed_seconds,
        "ocr_status": (
            "success" if detected_count > 0 else "no_text_detected"
        ),
    }


def process_image(record, ocr_engine):
    """Validate an image and extract its text."""

    image_path = PROJECT_ROOT / record["local_image_path"]

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Verify image integrity using the existing Phase 6.1 module.
    validate_image(image_path)

    # Load the image with EXIF orientation corrected.
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image.load()

        # RapidOCR accepts the image path directly.
        ocr_output = ocr_engine(str(image_path))

    return extract_ocr_result(ocr_output)


def main():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    # Never silently overwrite an existing OCR result.
    if OUTPUT_CSV.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_CSV}\n"
            "Preserve the existing output before rerunning."
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with INPUT_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as input_file:
        reader = csv.DictReader(input_file)
        records = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not records:
        raise ValueError("Input CSV contains no records.")

    output_fields = fieldnames + [
        "ocr_text",
        "detected_text_regions",
        "mean_ocr_confidence",
        "ocr_elapsed_seconds",
        "ocr_status",
        "ocr_error",
    ]

    ocr_engine = RapidOCR()

    total = len(records)
    success_count = 0
    no_text_count = 0
    error_count = 0

    print("=" * 65)
    print("FAKEDDIT PILOT OCR EXTRACTION")
    print("=" * 65)
    print(f"Total images: {total}")

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=output_fields,
        )

        writer.writeheader()

        for index, record in enumerate(records, start=1):
            image_id = record.get("id", "unknown")

            try:
                result = process_image(record, ocr_engine)
                record.update(result)
                record["ocr_error"] = ""

                if result["ocr_status"] == "success":
                    success_count += 1
                else:
                    no_text_count += 1

            except Exception as error:
                record.update({
                    "ocr_text": "",
                    "detected_text_regions": 0,
                    "mean_ocr_confidence": None,
                    "ocr_elapsed_seconds": None,
                    "ocr_status": "error",
                    "ocr_error": str(error),
                })

                error_count += 1

            writer.writerow(record)

            # Persist each record so interrupted runs retain progress.
            output_file.flush()

            if index % 25 == 0 or index == total:
                print(
                    f"Processed {index}/{total} | "
                    f"Detected: {success_count} | "
                    f"No text: {no_text_count} | "
                    f"Errors: {error_count}"
                )

    print("\n" + "=" * 65)
    print("OCR EXTRACTION COMPLETE")
    print("=" * 65)
    print(f"Total images: {total}")
    print(f"Images with detected text: {success_count}")
    print(f"Images with no detected text: {no_text_count}")
    print(f"Images with processing errors: {error_count}")
    print(f"\nOutput: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
