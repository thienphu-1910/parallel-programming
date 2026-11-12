"""V0 benchmark orchestration shared by the report notebook and command line.

Only the C++ executable measures performance. This module does not time subprocesses.
No third-party Python packages are required.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import uuid


CASES = (
    ("site1_indoor", "indoor"),
    ("site2_outdoor", "outdoor"),
    ("site3_custom", "skyline"),
    ("challenging_pair", "flats"),
)
SIZES = (
    ("small_1280x720", 1280, 720),
    ("medium_1920x1080", 1920, 1080),
    ("large_3840x2160", 3840, 2160),
)
STAGES = (
    "Preprocessing", "Keypoint detection", "Descriptor extraction",
    "Matching", "Homography estimation", "Warping and blending",
)


def run_command(command, root):
    return subprocess.run(
        [str(value) for value in command], cwd=root,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )


def command_text(command, root):
    try:
        result = run_command(command, root)
        return result.stdout.strip() if result.returncode == 0 else "Unavailable"
    except OSError:
        return "Unavailable"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jpeg_dimensions(path):
    """Read JPEG frame dimensions without decoding/resizing benchmark inputs."""
    with Path(path).open("rb") as source:
        if source.read(2) != b"\xff\xd8":
            raise ValueError(f"Expected a JPEG: {path}")
        while True:
            if source.read(1) != b"\xff":
                raise ValueError(f"Invalid JPEG marker: {path}")
            marker = source.read(1)
            while marker == b"\xff":
                marker = source.read(1)
            if not marker or marker in (b"\xd9", b"\xda"):
                raise ValueError(f"JPEG frame dimensions missing: {path}")
            if marker == b"\x01" or 0xD0 <= marker[0] <= 0xD7:
                continue
            size_bytes = source.read(2)
            if len(size_bytes) != 2:
                raise ValueError(f"Truncated JPEG: {path}")
            size = struct.unpack(">H", size_bytes)[0]
            payload = source.read(size - 2) if size >= 2 else b""
            if size < 2 or len(payload) != size - 2:
                raise ValueError(f"Invalid JPEG segment: {path}")
            if marker[0] in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                              0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                if len(payload) < 6 or payload[5] != 3:
                    raise ValueError(f"Expected a three-channel input: {path}")
                height, width = struct.unpack(">HH", payload[1:5])
                return width, height


def dataset_manifest(root):
    """Explicit pair selection excludes stale resized files in site directories."""
    jobs = []
    for site, prefix in CASES:
        for folder, width, height in SIZES:
            job = dict(site=site, prefix=prefix, folder=folder,
                       resolution=f"{width}x{height}", width=width, height=height)
            for side in ("left", "right"):
                relative = Path("dataset") / site / folder / f"{prefix}_{side}_{width}x{height}.jpg"
                image = root / relative
                if jpeg_dimensions(image) != (width, height):
                    raise ValueError(f"Unexpected dimensions: {image}")
                job[side] = relative.as_posix()
                job[f"{side}_sha256"] = sha256(image)
            jobs.append(job)
    return jobs


def source_manifest(root):
    paths = [root / "Makefile"]
    for folder in ("include", "src", "third_party", "scripts"):
        paths.extend(p for p in (root / folder).rglob("*")
                     if p.is_file() and p.suffix in (".h", ".cuh", ".cpp", ".cu", ".py"))
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths)}


def gpu_identity(root):
    text = command_text([
        "nvidia-smi", "--id=0", "--query-gpu=name,uuid,driver_version",
        "--format=csv,noheader",
    ], root)
    if text == "Unavailable":
        return None
    values = next(csv.reader([text]))
    if len(values) != 3:
        raise ValueError(f"Unexpected nvidia-smi output: {text}")
    return dict(zip(("name", "uuid", "driver"), (v.strip() for v in values)))


def parse_metrics(text, runs, width, height):
    """Fail closed if the CLI output changes or omits required measurements."""
    def number(pattern, cast=float):
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            raise ValueError(f"Missing output field: {pattern}")
        value = cast(match.group(1))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid measurement: {value}")
        return value

    if not re.search(r"^Version: V0,", text, re.MULTILINE):
        raise ValueError("Expected V0 output.")
    if number(r"^Timed runs: (\d+)\s*$", int) != runs:
        raise ValueError("Timed run count differs from the requested count.")
    for image in ("A", "B"):
        if not re.search(rf"^Image {image}: {width} x {height}\s*$", text, re.MULTILINE):
            raise ValueError(f"Unexpected image {image} dimensions.")
    metrics = {
        f"stage_{i}_mean_ms": number(
            rf"^Stage {i} \({re.escape(name)}\): ([0-9.]+) ms average\s*$"
        ) for i, name in enumerate(STAGES, 1)
    }
    metrics["end_to_end_mean_ms"] = number(r"^Pipeline total: ([0-9.]+) ms average\s*$")
    metrics["end_to_end_min_ms"] = number(r"^Pipeline minimum: ([0-9.]+) ms\s*$")
    if not 0 < metrics["end_to_end_min_ms"] <= metrics["end_to_end_mean_ms"]:
        raise ValueError("Invalid minimum/mean ordering.")
    for field, label in (("keypoints_a", "Keypoints A"), ("keypoints_b", "Keypoints B"),
                         ("matches", "Matches"), ("inliers", "Inliers")):
        metrics[field] = number(rf"^{label}: (\d+)\s*$", int)
    if not 0 < metrics["inliers"] <= metrics["matches"]:
        raise ValueError("Invalid successful-model inlier count.")
    return metrics


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(root, destination):
    make = shutil.which("mingw32-make" if os.name == "nt" else "make")
    if not make:
        raise RuntimeError("Install g++ and make (mingw32-make on Windows).")
    result = run_command([make, "-B"], root)
    (destination / "build.log").write_text(result.stdout, encoding="utf-8")
    print(result.stdout, end="", flush=True)
    if result.returncode:
        raise RuntimeError("Build failed; see build.log.")
    executable = root / ("stitch.exe" if os.name == "nt" else "stitch")
    # Preserve the exact executable used by this batch, even after another build.
    archived = destination / executable.name
    shutil.copy2(executable, archived)

    quality_executable = destination / ("quality.exe" if os.name == "nt" else "quality")
    sources = [root / "scripts/benchmark_quality.cpp", root / "src/image_io.cpp"]
    sources += sorted((root / "src/v0_sequential").glob("v0_stage[1-5]_*.cpp"))
    if len(sources) != 7:
        raise RuntimeError("Expected exactly five V0 stage sources for quality checks.")
    command = ["g++", "-std=c++17", "-O3", "-Wall", "-Wextra", "-pedantic",
               "-Iinclude", "-isystem", "third_party", *sources, "-o", quality_executable]
    result = run_command(command, root)
    (destination / "quality_build.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Quality diagnostic build failed:\n{result.stdout}")
    return archived, quality_executable


def run_batch(root, runs=5, require_colab_gpu=False):
    root = Path(root).resolve()
    if runs < 5:
        raise ValueError("Use at least five timed runs.")
    gpu = gpu_identity(root)
    in_colab = bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION"))
    if require_colab_gpu and (gpu is None or not in_colab):
        raise RuntimeError("Final-report mode requires a Colab GPU runtime.")
    jobs = dataset_manifest(root)  # Preflight every input before running anything.
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    destination = root / "outputs/benchmarks" / batch_id
    destination.mkdir(parents=True, exist_ok=False)
    gpu_name = gpu["name"] if gpu else "Unavailable (CPU development run)"
    metadata = {
        "batch_id": batch_id, "started_utc": datetime.now(timezone.utc).isoformat(),
        "environment": "Colab GPU baseline" if in_colab and gpu else "LOCAL DEVELOPMENT ONLY",
        "gpu": gpu, "platform": platform.platform(), "cpu": platform.processor(),
        "python": platform.python_version(), "runs": runs, "warmups_per_case": 1,
        "versions": [0], "git_commit": command_text(["git", "rev-parse", "HEAD"], root),
        "git_status": command_text(["git", "status", "--short"], root),
        "compiler": command_text(["g++", "--version"], root),
        "nvidia_smi": command_text(["nvidia-smi"], root),
        "cuda_compiler": command_text(["nvcc", "--version"], root),
        "sources": source_manifest(root), "inputs": jobs,
        "timing_precision_ms": 0.001,
        "timing_scope": "C++ pipeline only; no disk I/O, Python, or quality diagnostics",
        "dispersion": "Minimum always reported; per-run samples/standard deviation not emitted by current CLI",
    }
    write_json(destination / "metadata.json", metadata)
    executable, diagnostic = build(root, destination)
    metadata["executable_sha256"] = sha256(executable)
    metadata["diagnostic_sha256"] = sha256(diagnostic)
    write_json(destination / "metadata.json", metadata)
    text = (root / "src/v0_sequential/v0_stage5_ransac.cpp").read_text(encoding="utf-8")
    threshold_match = re.search(r"kInlierThreshold\s*=\s*([0-9.]+)\s*;", text)
    if not threshold_match:
        raise RuntimeError("Cannot read the production inlier threshold; update the diagnostic adapter.")
    threshold = float(threshold_match.group(1))
    rows = []
    for job in jobs:
        if gpu_identity(root) != gpu:
            raise RuntimeError("GPU identity changed; start a new batch.")
        for side in ("left", "right"):
            if sha256(root / job[side]) != job[f"{side}_sha256"]:
                raise RuntimeError("Input changed during the batch.")
        panorama = root / "outputs/panoramas" / job["site"] / job["folder"] / batch_id / f'{job["prefix"]}_v0.png'
        log = destination / f'{job["site"]}_{job["resolution"]}_v0.log'
        command = [executable, root / job["left"], root / job["right"], panorama,
                   "--version", "0", "--runs", str(runs)]
        heading = f'GPU: {gpu_name}\nBatch: {batch_id}\n{job["site"]} / {job["resolution"]} / V0\n'
        print("\n" + heading, end="", flush=True)
        result = run_command(command, root)
        log.write_text(heading + json.dumps([str(v) for v in command]) + "\n" + result.stdout, encoding="utf-8")
        print(result.stdout, end="", flush=True)
        row = dict(batch_id=batch_id, environment=metadata["environment"], gpu=gpu_name,
                   site=job["site"], resolution=job["resolution"], version=0, runs=runs,
                   status="failed", error="", kernel_mean_ms=None,
                   h2d_mean_ms=0.0, d2h_mean_ms=0.0,
                   panorama=panorama.relative_to(root).as_posix(), log=log.relative_to(root).as_posix())
        try:
            if result.returncode:
                raise ValueError(f"CLI exited {result.returncode}; see log.")
            metrics = parse_metrics(result.stdout, runs, job["width"], job["height"])
            if not panorama.is_file():
                raise ValueError("CLI succeeded but panorama was not saved.")
            row.update(metrics)
            row["status"] = "ok"
        except ValueError as error:
            row["error"] = str(error)
        rows.append(row)
        write_csv(destination / "timings.csv", rows)

    # All diagnostic runs occur after the complete timing batch.
    for job, row in zip(jobs, rows):
        print(f'Quality check: {job["site"]} / {job["resolution"]}', flush=True)
        result = run_command([diagnostic, root / job["left"], root / job["right"], str(threshold)], root)
        quality_log = destination / f'{job["site"]}_{job["resolution"]}_quality.log'
        quality_log.write_text(result.stdout, encoding="utf-8")
        row["quality_status"] = "diagnostic_failed"
        row["inlier_rmse_px"] = None
        try:
            if result.returncode:
                raise ValueError("Quality executable failed; see diagnostic log.")
            quality = json.loads(result.stdout)
            write_json(quality_log.with_suffix(".json"), quality)
            if row["status"] == "ok":
                if not quality["homography_valid"]:
                    raise ValueError("Timed pipeline succeeded but diagnostic homography failed.")
                for key in ("matches", "inliers", "keypoints_a", "keypoints_b"):
                    if quality[key] != row[key]:
                        raise ValueError(f"Non-repeatable V0 output: {key}")
            row["quality_status"] = "measured" if quality["homography_valid"] else "no_valid_homography"
            row["inlier_rmse_px"] = quality["inlier_rmse_px"]
            row["inlier_ratio"] = quality["inlier_ratio"]
            row["inlier_max_error_px"] = quality["inlier_max_error_px"]
        except (ValueError, KeyError) as error:
            row["quality_error"] = str(error)
        write_csv(destination / "results.csv", rows)
    metadata["sources_unchanged"] = source_manifest(root) == metadata["sources"]
    metadata["inputs_unchanged"] = dataset_manifest(root) == jobs
    metadata["gpu_unchanged"] = gpu_identity(root) == gpu
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["successful_cases"] = sum(row["status"] == "ok" for row in rows)
    write_json(destination / "metadata.json", metadata)
    if not all(metadata[key] for key in ("sources_unchanged", "inputs_unchanged", "gpu_unchanged")):
        raise RuntimeError("Environment/input/source changed; this batch cannot be used for comparison.")
    print(f"\nSaved {len(rows)} results: {destination}", flush=True)
    return destination, rows, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--require-colab-gpu", action="store_true")
    args = parser.parse_args()
    _, results, _ = run_batch(args.root, args.runs, args.require_colab_gpu)
    raise SystemExit(0 if all(r["status"] == "ok" and r["quality_status"] == "measured" for r in results) else 1)
