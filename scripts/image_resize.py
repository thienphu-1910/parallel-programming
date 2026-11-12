import argparse
from pathlib import Path

import cv2
import numpy as np


RESOLUTIONS = {
    "small": (1280, 720),
    "medium": (1920, 1080),
    "large": (3840, 2160),
}

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".heic",
    ".heif",
}


def discover_pairs(raw_directory: Path):
    pairs = {}

    for path in sorted(raw_directory.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        prefix, separator, side = path.stem.lower().rpartition("_")

        if not separator or not prefix or side not in {"left", "right"}:
            raise ValueError(
                f"Invalid raw filename: {path.name}. "
                "Expected <name>_left or <name>_right."
            )

        sources = pairs.setdefault(prefix, {})

        if side in sources:
            raise ValueError(
                f"Duplicate {side} image for pair '{prefix}' "
                f"in {raw_directory}."
            )

        sources[side] = path

    for prefix, sources in pairs.items():
        missing = {"left", "right"} - sources.keys()

        if missing:
            raise ValueError(
                f"Pair '{prefix}' in {raw_directory} is missing: "
                f"{', '.join(sorted(missing))}."
            )

    return pairs


def load_image_bgr(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".heic", ".heif"}:
        # HEIC decoding is separate from OpenCV resizing.
        from PIL import Image, ImageOps
        from pillow_heif import register_heif_opener

        register_heif_opener(thumbnails=False)

        with Image.open(path) as image:
            oriented = ImageOps.exif_transpose(image)
            rgb = np.array(oriented.convert("RGB"))

        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Reading bytes through pathlib supports Unicode filenames.
    encoded = np.frombuffer(
        path.read_bytes(),
        dtype=np.uint8,
    )

    if encoded.size == 0:
        raise ValueError(f"Image file is empty: {path}")

    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Cannot decode image: {path}")

    return image


def crop_to_16_9(image: np.ndarray):
    height, width = image.shape[:2]

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


def resize_image(image: np.ndarray, width: int, height: int):
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


def output_path(
    site_directory: Path,
    prefix: str,
    side: str,
    label: str,
    width: int,
    height: int,
) -> Path:
    return (
        site_directory
        / f"{label}_{width}x{height}"
        / f"{prefix}_{side}_{width}x{height}.jpg"
    )


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

    saved = cv2.imdecode(
        np.frombuffer(path.read_bytes(), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if saved is None or saved.shape != image.shape:
        raise RuntimeError(
            f"Saved-image verification failed: {path}"
        )


def prepare_pair(site_directory: Path, prefix: str, sources):
    # Decode both images before writing this pair's outputs.
    images = {
        side: load_image_bgr(sources[side])
        for side in ("left", "right")
    }

    print(f"\nSite: {site_directory.name}; pair: {prefix}")

    for side, image in images.items():
        source_height, source_width = image.shape[:2]

        cropped, crop = crop_to_16_9(image)
        left, top, crop_width, crop_height = crop

        print(f"Source: {sources[side].name}")
        print(f"Decoded dimensions: {source_width}x{source_height}")
        print(
            f"Crop: x={left}, y={top}, "
            f"width={crop_width}, height={crop_height}"
        )

        for label, (width, height) in RESOLUTIONS.items():
            # Generate every size directly from the original crop.
            resized, method = resize_image(cropped, width, height)

            destination = output_path(
                site_directory,
                prefix,
                side,
                label,
                width,
                height,
            )

            save_jpeg(destination, resized)

            print(f"Saved: {destination}")
            print(f"  {width}x{height}; {method}")


def main():
    project_root = Path(__file__).resolve().parent.parent
    dataset_directory = project_root / "dataset"

    parser = argparse.ArgumentParser(
        description=(
            "Prepare raw image pairs across dataset folders "
            "at 720p, 1080p, and 4K."
        )
    )

    parser.add_argument(
        "--site",
        type=Path,
        action="append",
        help=(
            "Process one site directory. Repeat this option "
            "for multiple sites. Default: all dataset sites."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing resized images.",
    )

    args = parser.parse_args()

    if args.site:
        sites = sorted({path.resolve() for path in args.site})
    else:
        sites = sorted(
            path
            for path in dataset_directory.iterdir()
            if path.is_dir() and (path / "raw").is_dir()
        )

    jobs = []

    for site_directory in sites:
        raw_directory = site_directory / "raw"

        if not raw_directory.is_dir():
            raise FileNotFoundError(
                f"Raw directory does not exist: {raw_directory}"
            )

        pairs = discover_pairs(raw_directory)

        if not pairs:
            print(f"Skipping empty raw folder: {raw_directory}")
            continue

        for prefix, sources in pairs.items():
            jobs.append((site_directory, prefix, sources))

    if not jobs:
        raise ValueError("No complete raw image pairs were found.")

    # Check all destinations before writing any output.
    for site_directory, prefix, _ in jobs:
        for side in ("left", "right"):
            for label, (width, height) in RESOLUTIONS.items():
                destination = output_path(
                    site_directory,
                    prefix,
                    side,
                    label,
                    width,
                    height,
                )

                if destination.exists() and not args.overwrite:
                    raise FileExistsError(
                        f"Output already exists: {destination}\n"
                        "Use --overwrite to replace existing outputs."
                    )

    print(f"OpenCV version: {cv2.__version__}")
    print(f"Pairs to prepare: {len(jobs)}")

    for site_directory, prefix, sources in jobs:
        prepare_pair(site_directory, prefix, sources)

    print(f"\nCompleted: {len(jobs)} pairs.")


if __name__ == "__main__":
    main()