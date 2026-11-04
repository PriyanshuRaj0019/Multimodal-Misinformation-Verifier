
"""Safe image validation, metadata extraction, loading, and preprocessing."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


class ImageValidationError(ValueError):
    """Raised when an image fails validation."""


@dataclass(frozen=True)
class ImageMetadata:
    """Metadata extracted from a validated image."""

    file_name: str
    image_format: str
    width: int
    height: int
    mode: str
    file_size_bytes: int
    sha256: str
    has_transparency: bool

    def to_dict(self) -> dict:
        """Return metadata as a dictionary."""
        return asdict(self)


def _get_file_path(path: str | Path) -> Path:
    """Check that the supplied path points to an existing regular file."""
    image_path = Path(path)

    if not image_path.exists():
        raise ImageValidationError(f"Image file does not exist: {image_path}")

    if not image_path.is_file():
        raise ImageValidationError(f"Path is not a regular file: {image_path}")

    return image_path


def calculate_sha256(path: str | Path) -> str:
    """Calculate the SHA-256 hash of a file without loading it all into memory."""
    image_path = _get_file_path(path)
    digest = hashlib.sha256()

    with image_path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def validate_image(path: str | Path) -> ImageMetadata:
    """Validate an image and return its metadata."""
    image_path = _get_file_path(path)

    file_size = image_path.stat().st_size

    if file_size == 0:
        raise ImageValidationError("Image file is empty.")

    if file_size > MAX_FILE_SIZE_BYTES:
        raise ImageValidationError(
            f"Image exceeds the {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB size limit."
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)

            with Image.open(image_path) as image:
                image_format = image.format
                width, height = image.size
                mode = image.mode
                has_transparency = (
                    "A" in image.getbands() or "transparency" in image.info
                )

                if image_format not in SUPPORTED_FORMATS:
                    raise ImageValidationError(
                        f"Unsupported image format: {image_format}"
                    )

                if width <= 0 or height <= 0:
                    raise ImageValidationError("Image dimensions must be positive.")

                if width * height > MAX_IMAGE_PIXELS:
                    raise ImageValidationError(
                        "Image exceeds the maximum allowed pixel count."
                    )

                image.verify()

    except ImageValidationError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise ImageValidationError(f"Invalid or unsafe image: {error}") from error

    return ImageMetadata(
        file_name=image_path.name,
        image_format=image_format,
        width=width,
        height=height,
        mode=mode,
        file_size_bytes=file_size,
        sha256=calculate_sha256(image_path),
        has_transparency=has_transparency,
    )


def load_image(
    path: str | Path,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Validate, orient, and load an image as RGB."""
    validate_image(path)
    image_path = Path(path)

    try:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source)
            image.load()

            if "A" in image.getbands() or "transparency" in image.info:
                rgba_image = image.convert("RGBA")
                background_image = Image.new("RGB", rgba_image.size, background)
                background_image.paste(rgba_image, mask=rgba_image.getchannel("A"))
                return background_image

            return image.convert("RGB")

    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise ImageValidationError(f"Could not load image safely: {error}") from error


def preprocess_image(
    path: str | Path,
    max_size: tuple[int, int] = (1024, 1024),
) -> Image.Image:
    """Load an image and resize it while preserving its aspect ratio."""
    if len(max_size) != 2 or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in max_size
    ):
        raise ValueError("max_size must contain two positive integers.")

    image = load_image(path)
    image.thumbnail(max_size, Image.Resampling.LANCZOS)

    return image
