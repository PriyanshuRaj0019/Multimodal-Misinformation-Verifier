
"""Tests for safe image processing."""

import pytest
from PIL import Image

from misinfo_verifier.image_processing import (
    MAX_FILE_SIZE_BYTES,
    ImageValidationError,
    calculate_sha256,
    load_image,
    preprocess_image,
    validate_image,
)


def create_test_image(
    path,
    image_format="PNG",
    size=(200, 100),
    color="red",
):
    """Create a small image for testing."""
    image = Image.new("RGB", size, color)
    image.save(path, format=image_format)
    return path


def test_validate_valid_png(tmp_path):
    image_path = create_test_image(tmp_path / "test.png")

    metadata = validate_image(image_path)

    assert metadata.image_format == "PNG"
    assert metadata.width == 200
    assert metadata.height == 100
    assert metadata.mode == "RGB"
    assert metadata.file_size_bytes > 0
    assert len(metadata.sha256) == 64
    assert metadata.has_transparency is False


def test_validate_supported_formats(tmp_path):
    for image_format, extension in [
        ("JPEG", ".jpg"),
        ("PNG", ".png"),
        ("WEBP", ".webp"),
    ]:
        image_path = create_test_image(
            tmp_path / f"test{extension}",
            image_format=image_format,
        )

        metadata = validate_image(image_path)

        assert metadata.image_format == image_format


def test_reject_unsupported_format(tmp_path):
    image_path = create_test_image(
        tmp_path / "test.gif",
        image_format="GIF",
    )

    with pytest.raises(ImageValidationError, match="Unsupported image format"):
        validate_image(image_path)


def test_reject_missing_file(tmp_path):
    with pytest.raises(ImageValidationError, match="does not exist"):
        validate_image(tmp_path / "missing.png")


def test_reject_empty_file(tmp_path):
    image_path = tmp_path / "empty.png"
    image_path.write_bytes(b"")

    with pytest.raises(ImageValidationError, match="empty"):
        validate_image(image_path)


def test_reject_invalid_image(tmp_path):
    image_path = tmp_path / "invalid.png"
    image_path.write_bytes(b"This is not an image.")

    with pytest.raises(ImageValidationError):
        validate_image(image_path)


def test_reject_oversized_file(tmp_path, monkeypatch):
    image_path = create_test_image(tmp_path / "large.png")

    monkeypatch.setattr(
        "misinfo_verifier.image_processing.MAX_FILE_SIZE_BYTES",
        1,
    )

    with pytest.raises(ImageValidationError, match="exceeds"):
        validate_image(image_path)

    assert MAX_FILE_SIZE_BYTES > 1


def test_sha256_is_consistent(tmp_path):
    image_path = create_test_image(tmp_path / "test.png")

    first_hash = calculate_sha256(image_path)
    second_hash = calculate_sha256(image_path)

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_load_image_returns_rgb(tmp_path):
    image_path = create_test_image(tmp_path / "test.png")

    image = load_image(image_path)

    assert image.mode == "RGB"
    assert image.size == (200, 100)


def test_transparent_image_uses_white_background(tmp_path):
    image_path = tmp_path / "transparent.png"

    image = Image.new("RGBA", (10, 10), (255, 0, 0, 0))
    image.save(image_path)

    loaded_image = load_image(image_path)

    assert loaded_image.mode == "RGB"
    assert loaded_image.getpixel((0, 0)) == (255, 255, 255)


def test_preprocess_preserves_aspect_ratio(tmp_path):
    image_path = create_test_image(
        tmp_path / "wide.png",
        size=(2000, 1000),
    )

    image = preprocess_image(image_path, max_size=(500, 500))

    assert image.size == (500, 250)
    assert image.mode == "RGB"


def test_preprocess_rejects_invalid_max_size(tmp_path):
    image_path = create_test_image(tmp_path / "test.png")

    with pytest.raises(ValueError, match="positive integers"):
        preprocess_image(image_path, max_size=(0, 500))
