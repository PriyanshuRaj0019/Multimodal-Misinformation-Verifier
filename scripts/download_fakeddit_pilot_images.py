
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
)
import re
import time

import pandas as pd

from misinfo_verifier.image_processing import (
    ImageValidationError,
    validate_image,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path.cwd()

INPUT_FILE = (
    PROJECT_ROOT
    / "data/raw/fakeddit/manifests/phase6_train_pilot_manifest.csv"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "data/raw/fakeddit/images/train_pilot"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data/raw/fakeddit/manifests/"
    / "phase6_train_pilot_download_manifest.csv"
)

ALLOWED_HOSTS = {
    "external-preview.redd.it",
    "preview.redd.it",
    "i.imgur.com",
    "imgur.com",
    "img.gawkerassets.com",
}

MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024
REQUEST_TIMEOUT = 20
MAX_ATTEMPTS = 3
REQUEST_DELAY_SECONDS = 0.25

USER_AGENT = "MMV-Fakeddit-Research/1.0"

FORMAT_EXTENSIONS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
}

RESULT_COLUMNS = [
    "download_status",
    "local_image_path",
    "image_format",
    "image_width",
    "image_height",
    "image_size_bytes",
    "image_sha256",
    "download_error",
]


# --------------------------------------------------
# URL validation
# --------------------------------------------------

def validate_download_url(url):
    parsed = urlparse(str(url).strip())

    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS URLs are allowed.")

    if not parsed.hostname:
        raise ValueError("URL has no hostname.")

    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed.")

    hostname = parsed.hostname.lower()

    if hostname not in ALLOWED_HOSTS:
        raise ValueError(
            f"Image host is not allowlisted: {hostname}"
        )

    return url


class SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req,
        fp,
        code,
        msg,
        headers,
        newurl,
    ):
        validate_download_url(newurl)

        return super().redirect_request(
            req,
            fp,
            code,
            msg,
            headers,
            newurl,
        )


OPENER = build_opener(SafeRedirectHandler())


# --------------------------------------------------
# Download image bytes
# --------------------------------------------------

def fetch_image_bytes(url):
    validate_download_url(url)

    request = Request(
        url,
        headers={"User-Agent": USER_AGENT},
    )

    with OPENER.open(
        request,
        timeout=REQUEST_TIMEOUT,
    ) as response:

        final_url = response.geturl()
        validate_download_url(final_url)

        content = response.read(MAX_DOWNLOAD_BYTES + 1)

    if not content:
        raise ValueError("Downloaded response is empty.")

    if len(content) > MAX_DOWNLOAD_BYTES:
        raise ValueError(
            "Image exceeds the 5 MB download limit."
        )

    return content


# --------------------------------------------------
# Validate and save downloaded image
# --------------------------------------------------

def save_validated_image(sample_id, content):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", sample_id):
        raise ValueError("Unsafe sample ID.")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    temporary_path = None

    try:
        with NamedTemporaryFile(
            mode="wb",
            suffix=".part",
            prefix="mmv_",
            dir=IMAGE_DIR,
            delete=False,
        ) as temporary_file:

            temporary_file.write(content)
            temporary_path = Path(temporary_file.name)

        metadata = validate_image(temporary_path)

        extension = FORMAT_EXTENSIONS.get(
            metadata.image_format
        )

        if extension is None:
            raise ImageValidationError(
                f"Unsupported image format: {metadata.image_format}"
            )

        final_path = IMAGE_DIR / f"{sample_id}{extension}"

        if final_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing image: {final_path}"
            )

        temporary_path.replace(final_path)
        temporary_path = None

        return metadata, final_path

    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


# --------------------------------------------------
# Download one sample with retries
# --------------------------------------------------

def download_one(sample_id, image_url):
    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            content = fetch_image_bytes(image_url)

            metadata, image_path = save_validated_image(
                sample_id,
                content,
            )

            return {
                "download_status": "downloaded",
                "local_image_path": str(
                    image_path.relative_to(PROJECT_ROOT)
                ),
                "image_format": metadata.image_format,
                "image_width": metadata.width,
                "image_height": metadata.height,
                "image_size_bytes": metadata.file_size_bytes,
                "image_sha256": metadata.sha256,
                "download_error": "",
            }

        except HTTPError as error:
            last_error = error

            if error.code not in {429, 500, 502, 503, 504}:
                break

        except (
            URLError,
            TimeoutError,
            ConnectionError,
        ) as error:
            last_error = error

        except Exception as error:
            last_error = error
            break

        if attempt < MAX_ATTEMPTS:
            time.sleep(attempt * 2)

    return {
        "download_status": "failed",
        "local_image_path": "",
        "image_format": "",
        "image_width": None,
        "image_height": None,
        "image_size_bytes": None,
        "image_sha256": "",
        "download_error": (
            f"{type(last_error).__name__}: {last_error}"
        ),
    }


# --------------------------------------------------
# Save progress safely
# --------------------------------------------------

def save_manifest(results):
    output = pd.DataFrame(results)

    temporary_output = OUTPUT_FILE.with_suffix(".tmp")

    output.to_csv(
        temporary_output,
        index=False,
    )

    temporary_output.replace(OUTPUT_FILE)


# --------------------------------------------------
# Main workflow
# --------------------------------------------------

def main():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Pilot manifest not found: {INPUT_FILE}"
        )

    source = pd.read_csv(
        INPUT_FILE,
        dtype={"id": "string"},
    )

    if source["id"].isna().any():
        raise ValueError("Pilot manifest contains missing IDs.")

    if source["id"].duplicated().any():
        raise ValueError("Pilot manifest contains duplicate IDs.")

    if OUTPUT_FILE.exists():
        previous = pd.read_csv(
            OUTPUT_FILE,
            dtype={"id": "string"},
        )

        if set(previous["id"]) != set(source["id"]):
            raise ValueError(
                "Existing download manifest does not match "
                "the current pilot sample."
            )

        previous_results = previous.set_index("id").to_dict(
            orient="index"
        )

        print("Resuming from the existing download manifest.")

    else:
        previous_results = {}

    results = []

    print("\n" + "=" * 65)
    print("FAKEDDIT PILOT IMAGE DOWNLOAD")
    print("=" * 65)
    print(f"Samples: {len(source):,}")
    print(f"Image directory: {IMAGE_DIR}")

    for position, row in enumerate(
        source.to_dict(orient="records"),
        start=1,
    ):
        sample_id = str(row["id"])
        image_url = str(row["image_url"]).strip()

        previous = previous_results.get(sample_id, {})

        previous_path = previous.get("local_image_path", "")
        previous_status = previous.get("download_status", "")

        # Skip a previously downloaded image only if it
        # still exists and passes image validation.
        if previous_status == "downloaded" and previous_path:
            existing_path = PROJECT_ROOT / previous_path

            if existing_path.is_file():
                try:
                    metadata = validate_image(existing_path)

                    result = {
                        "download_status": "downloaded",
                        "local_image_path": previous_path,
                        "image_format": metadata.image_format,
                        "image_width": metadata.width,
                        "image_height": metadata.height,
                        "image_size_bytes": metadata.file_size_bytes,
                        "image_sha256": metadata.sha256,
                        "download_error": "",
                    }

                    results.append({
                        **row,
                        **result,
                    })

                    continue

                except Exception:
                    print(
                        f"\nExisting image failed validation: {sample_id}"
                    )

        result = download_one(
            sample_id,
            image_url,
        )

        results.append({
            **row,
            **result,
        })

        # Persist progress after every sample.
        save_manifest(results)

        print(
            f"[{position:03d}/{len(source):03d}] "
            f"{sample_id}: {result['download_status']}"
        )

        time.sleep(REQUEST_DELAY_SECONDS)

    # Final manifest
    save_manifest(results)

    report = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print("DOWNLOAD SUMMARY")
    print("=" * 65)

    print(
        report["download_status"].value_counts(
            dropna=False
        )
    )

    successful = report.loc[
        report["download_status"] == "downloaded"
    ]

    failed = report.loc[
        report["download_status"] == "failed"
    ]

    print(f"\nSuccessful downloads: {len(successful):,}")
    print(f"Failed downloads: {len(failed):,}")

    if not successful.empty:
        total_bytes = successful["image_size_bytes"].sum()

        print(
            f"Total image storage: "
            f"{total_bytes / (1024 * 1024):.2f} MB"
        )

        print("\nDownloaded image formats:")
        print(successful["image_format"].value_counts())

    if not failed.empty:
        print("\nFailure reasons:")
        print(
            failed["download_error"]
            .value_counts()
            .head(10)
        )

    print(f"\nDownload manifest: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
