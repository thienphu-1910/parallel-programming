import argparse
from pathlib import Path

import cv2
import numpy as np


RESOLUTIONS = {
    "small": (1280, 720),
    "medium": (1920, 1080),
    "large": (3840, 2160),
}


def find_raw_image(raw_directory: Path, side: str) -> Path:
    expected_stem = f"indoor_{side}"

    candidates = [
        path
        for path in raw_directory.iterdir()
        if path.is_file()
        and path.stem.lower() == expected_stem
        and path.suffix.lower() == ".png"
    ]

    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one PNG named {expected_stem}.png "
            f"in {raw_directory}; found {len(candidates)}."
        )

    return candidates[0]


def load_png_bgr(path: Path) -> np.ndarray:
    # Read through pathlib to support Unicode paths on Windows.
    encoded = np.frombuffer(
        path.read_bytes(),
        dtype=np.uint8,
    )

    if encoded.size == 0:
        raise ValueError(f"PNG file is empty: {path}")

    # Decode directly to three-channel, 8-bit BGR.
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Cannot decode PNG image: {path}")

    return image


def crop_to_16_9(image: np.ndarray):
    height, width = image.shape[:2]

    # Largest exact 16:9 rectangle that fits inside the image.
    unit = min(width // 16, height // 9)

    if unit == 0:
        raise ValueError("Image is too small for a 16:9 crop.")

    crop_width = 16 * unit
    crop_height = 9 * unit

    left = (width - crop_width) // 2
    top = (height - crop_height) // 2

    cropped = image[
        top:top + crop_height,
        left:left + crop_width,
    ]

    return cropped, (left, top, crop_width, crop_height)


def resize_image(
    image: np.ndarray,
    width: int,
    height: int,
):
    source_height, source_width = image.shape[:2]

    if source_width == width and source_height == height:
        return image.copy(), "no resizing"

    if width <= source_width and height <= source_height:
        interpolation = cv2.INTER_AREA
        method = "INTER_AREA"
    else:
        interpolation = cv2.INTER_CUBIC
        method = "INTER_CUBIC (upscaled)"

    resized = cv2.resize(
        image,
        (width, height),
        interpolation=interpolation,
    )

    return resized, method


def save_jpeg(path: Path, image: np.ndarray):
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    if not success:
        raise RuntimeError(f"JPEG encoding failed: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())

    # Verify that the saved file decodes to the expected dimensions.
    saved = cv2.imdecode(
        np.frombuffer(path.read_bytes(), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if saved is None or saved.shape != image.shape:
        raise RuntimeError(
            f"Saved-image verification failed: {path}"
        )


def output_path(
    site_directory: Path,
    side: str,
    label: str,
    width: int,
    height: int,
) -> Path:
    return (
        site_directory
        / f"{label}_{width}x{height}"
        / f"indoor_{side}_{width}x{height}.jpg"
    )


def main():
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description=(
            "Crop the indoor PNG pair to 16:9 and resize it "
            "to 720p, 1080p, and 4K."
        )
    )

    parser.add_argument(
        "--site",
        type=Path,
        default=project_root / "dataset" / "site1_indoor",
        help="Site directory containing the raw subdirectory.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing resized images.",
    )

    args = parser.parse_args()

    site_directory = args.site.resolve()
    raw_directory = site_directory / "raw"

    if not raw_directory.is_dir():
        raise FileNotFoundError(
            f"Raw directory does not exist: {raw_directory}"
        )

    sources = {
        side: find_raw_image(raw_directory, side)
        for side in ("left", "right")
    }

    # Check all destinations before writing any output.
    for side in sources:
        for label, (width, height) in RESOLUTIONS.items():
            destination = output_path(
                site_directory, side, label, width, height
            )

            if destination.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Output already exists: {destination}\n"
                    "Use --overwrite to replace existing outputs."
                )

    # Decode both inputs before generating output files.
    images = {
        side: load_png_bgr(source)
        for side, source in sources.items()
    }

    print(f"OpenCV version: {cv2.__version__}")

    for side, image in images.items():
        source_height, source_width = image.shape[:2]

        cropped, crop = crop_to_16_9(image)
        left, top, crop_width, crop_height = crop

        print(f"\nSource: {sources[side].name}")
        print(f"Decoded dimensions: {source_width}x{source_height}")
        print(
            f"Crop: x={left}, y={top}, "
            f"width={crop_width}, height={crop_height}"
        )

        for label, (width, height) in RESOLUTIONS.items():
            # Every resolution comes directly from the same crop.
            resized, method = resize_image(
                cropped, width, height
            )

            destination = output_path(
                site_directory, side, label, width, height
            )

            save_jpeg(destination, resized)

            print(
                f"Saved: {destination}\n"
                f"  Dimensions: {width}x{height}; method: {method}"
            )


if __name__ == "__main__":
    main()