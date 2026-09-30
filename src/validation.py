"""
validation.py
=============
Input-validation helpers for the DeepGuard prototype.

These functions check that an uploaded file is a supported, readable image
before the rest of the pipeline is allowed to use it.

Design note for beginners:
    Every function that can fail raises a single, easy-to-catch exception
    called ``ImageValidationError`` with a friendly message.  The Streamlit
    user interface catches that one exception and shows the message to the
    user, so raw Python tracebacks are never displayed.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Configuration (change these values if you want to accept other inputs)
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")
SUPPORTED_FORMATS = ("JPEG", "PNG")

MIN_WIDTH = 32
MIN_HEIGHT = 32
MAX_WIDTH = 12000
MAX_HEIGHT = 12000
MAX_FILE_SIZE_MB = 25


class ImageValidationError(Exception):
    """Raised when an uploaded file cannot be accepted by the prototype."""


@dataclass
class ImageInfo:
    """A small, plain container describing a validated image."""

    filename: str
    extension: str
    mime_type: str
    image_format: str
    width: int
    height: int
    channels: int
    mode: str
    size_bytes: int

    @property
    def size_kb(self) -> float:
        """File size in kilobytes, rounded for display."""
        return round(self.size_bytes / 1024.0, 2)


# ---------------------------------------------------------------------------
# Small, single-purpose checks
# ---------------------------------------------------------------------------
def get_extension(filename: str) -> str:
    """Return the lower-case file extension (including the dot), or ''."""
    if not filename or "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def is_supported_extension(filename: str) -> bool:
    """Return True if the file name ends with a supported image extension."""
    return get_extension(filename) in SUPPORTED_EXTENSIONS


def check_extension(filename: str) -> None:
    """Raise ImageValidationError if the file extension is not supported."""
    extension = get_extension(filename)
    if extension == "":
        raise ImageValidationError(
            "The selected file has no extension, so its type could not be recognised."
        )
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise ImageValidationError(
            f"Unsupported file type '{extension}'. This prototype accepts: {supported}."
        )


def check_file_size(size_bytes: int, max_mb: int = MAX_FILE_SIZE_MB) -> None:
    """Raise ImageValidationError if the file is empty or too large."""
    if size_bytes <= 0:
        raise ImageValidationError("The uploaded file is empty.")
    if size_bytes > max_mb * 1024 * 1024:
        raise ImageValidationError(
            f"The file is larger than the allowed limit of {max_mb} MB."
        )


def check_dimensions(width: int, height: int) -> None:
    """Raise ImageValidationError if the image dimensions are unreasonable."""
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        raise ImageValidationError(
            f"This image is very small ({width} x {height} px). "
            f"At least {MIN_WIDTH} x {MIN_HEIGHT} px is required."
        )
    if width > MAX_WIDTH or height > MAX_HEIGHT:
        raise ImageValidationError(
            f"This image is very large ({width} x {height} px) and was rejected."
        )


def load_image(data: bytes) -> Image.Image:
    """
    Open raw file bytes as a Pillow image.

    Pillow is used here because it validates the file safely: a corrupted or
    non-image file raises an error that we convert into a friendly message.
    """
    if not data:
        raise ImageValidationError("The uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # Force Pillow to read the pixel data (catches truncation).
    except UnidentifiedImageError:
        raise ImageValidationError(
            "This file is not a readable image. It may be corrupted, or it may "
            "not really be a JPG/PNG file."
        )
    except (OSError, ValueError) as error:
        raise ImageValidationError(f"Unable to process this image: {error}")

    return image


# ---------------------------------------------------------------------------
# Main entry point used by the app
# ---------------------------------------------------------------------------
def validate_uploaded_file(
    filename: str,
    mime_type: str | None,
    data: bytes,
) -> ImageInfo:
    """
    Run every validation check on an uploaded file.

    Returns an ImageInfo object when the file is valid.
    Raises ImageValidationError with a friendly message otherwise.
    """
    check_extension(filename)
    check_file_size(len(data))

    image = load_image(data)
    check_dimensions(image.width, image.height)

    image_format = (image.format or get_extension(filename).lstrip(".")).upper()
    if image_format not in SUPPORTED_FORMATS:
        raise ImageValidationError(
            f"Unsupported image format '{image_format}'. "
            "Supported formats are JPEG and PNG."
        )

    channels = len(image.getbands())

    return ImageInfo(
        filename=filename,
        extension=get_extension(filename),
        mime_type=mime_type or "unknown",
        image_format=image_format,
        width=image.width,
        height=image.height,
        channels=channels,
        mode=image.mode,
        size_bytes=len(data),
    )
