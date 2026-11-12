# Panorama Stitching

## 1. Team Participants

- Hà Thư Hoàng - 23127367
- Trương Công Thiên Phú - 23127455

## Implementation status

V0 implements the complete single-threaded C++ pipeline. The current executable supports `--version 0`. CUDA V1, V2, and V3, the cross-version verifier, and the final-report notebook/PDF are placeholders.

The implemented methods are grayscale conversion and separable Gaussian smoothing, Harris keypoints, 256-bit BRIEF descriptors, exhaustive Hamming matching with a strict 0.8 ratio test and mutual checking, deterministic RANSAC, and inverse bilinear warping with feather blending. RANSAC uses 2,000 iterations, a 3-pixel threshold, and at least 8 inliers. Its normalized eight-parameter fit uses normal equations with `h33 = 1`.

The homography maps image B (right) into image A's (left) coordinate system.

## Get the development source

```bash
git clone --branch dev --single-branch https://github.com/thienphu-1910/parallel-programming.git
cd parallel-programming
```

Update an existing `dev` checkout with:

```bash
git pull --ff-only origin dev
```

Git downloads files committed to the branch. The notebook also supports building from the submitted source bundle.

## Build and run

V0 requires a C++17 compiler and Make. Image I/O uses the bundled stb headers. The Makefile uses `-std=c++17 -O3 -Wall -Wextra -pedantic`.

### Windows PowerShell

Use the MSYS2 UCRT64 toolchain with `g++` and `mingw32-make` on `PATH`. The local toolchain directory is `C:\msys64\ucrt64\bin`. Run from the project root:

```powershell
mingw32-make -B
if ($LASTEXITCODE -ne 0) { throw "Build failed." }

.\stitch.exe `
    dataset/site1_indoor/medium_1920x1080/indoor_left_1920x1080.jpg `
    dataset/site1_indoor/medium_1920x1080/indoor_right_1920x1080.jpg `
    outputs/panoramas/site1_indoor/medium_1920x1080/indoor_v0.png `
    --version 0 --runs 5
```

### Linux or Colab

Use `g++` and `make`, supplied by `build-essential` on Ubuntu.

```bash
make -B && ./stitch \
    dataset/site1_indoor/medium_1920x1080/indoor_left_1920x1080.jpg \
    dataset/site1_indoor/medium_1920x1080/indoor_right_1920x1080.jpg \
    outputs/panoramas/site1_indoor/medium_1920x1080/indoor_v0.png \
    --version 0 --runs 5
```

### Command-line interface

```text
./stitch <left_image> <right_image> <output.png> --version 0 [--runs N]
```

- `--runs` defaults to 5. Each invocation also performs one untimed warm-up.
- Output must end in `.png`. Missing parent directories are created.
- `--help` prints usage. Invalid inputs and unsuccessful homography/panorama estimation produce a nonzero exit code.
- Smaller repetition counts are supported for debugging. Reported benchmarks use at least five timed runs.

## Dataset and preparation

Download the raw photographs and prepared images from the [team Drive folder](https://drive.google.com/drive/folders/1Saplb-gGh9_TM0MExYXlqH89Je0stNu9?usp=sharing). Place the prepared `dataset/` directory at the project root. Shared-folder access and capture records should be confirmed for submission.

| Site folder | Pair prefix | Scene |
|---|---|---|
| `site1_indoor` | `indoor` | Classroom |
| `site2_outdoor` | `outdoor` | Street-facing buildings |
| `site3_custom` | `skyline` | Skyline and a nearby building |
| `challenging_pair` | `flats` | Apartment building behind foliage |

| Size folder | Input dimensions |
|---|---|
| `small_1280x720` | 1280 × 720 |
| `medium_1920x1080` | 1920 × 1080 |
| `large_3840x2160` | 3840 × 2160 |

Prepared filenames follow `<prefix>_<left|right>_<width>x<height>.jpg`. For example, the custom-site 1080p pair is `skyline_left_1920x1080.jpg` and `skyline_right_1920x1080.jpg`.

Put originals in each site's `raw/` folder using `<prefix>_left` and `<prefix>_right` stems. The resize script accepts PNG, JPG/JPEG, and HEIC/HEIF. Each pair needs exactly one file for each side.

Create a Python environment and prepare the images on Windows:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install opencv-python-headless numpy pillow pillow-heif
.\.venv\Scripts\python.exe scripts/image_resize.py
```

On Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install opencv-python-headless numpy pillow pillow-heif
.venv/bin/python scripts/image_resize.py
```

The script makes a centered 16:9 crop and generates every resolution directly from that crop. It uses area interpolation for downsampling, cubic interpolation for upsampling, and JPEG quality 95. HEIC/HEIF decoding includes EXIF orientation correction.

To replace prepared images or process a single site:

```powershell
.\.venv\Scripts\python.exe scripts/image_resize.py --overwrite
.\.venv\Scripts\python.exe scripts/image_resize.py --site dataset/site2_outdoor --overwrite
```

The build uses the committed BRIEF pattern. To regenerate the header with the generator's fixed seed:

```bash
python scripts/generate_brief_pattern.py
```

The script automatically writes `include/brief_pattern.h`. Use the same pattern across compared versions.

## Batch benchmarking

The runner uses Python's standard library and the C++ toolchain. From the project root:

```powershell
python scripts/benchmark_runner.py --runs 5
```

On Linux, use `python3`. The runner builds the pipeline and the separate quality diagnostic, then processes four pairs at three resolutions. It checks input dimensions and hashes, preserves logs and executables, and runs additional quality diagnostics after the timed batch.

```text
outputs/benchmarks/<batch_id>/
├── metadata.json
├── build.log
├── quality_build.log
├── timings.csv
├── results.csv
├── stitch.exe or stitch
├── quality.exe or quality
└── per-case execution and diagnostic logs

outputs/panoramas/<site>/<size_folder>/<batch_id>/<prefix>_v0.png
```

Results include stage means, end-to-end mean and minimum, feature counts, inlier ratio, and inlier reprojection RMSE. Failed cases retain a failure status. The runner exits nonzero if a pipeline case fails or its quality diagnostic is unavailable.

CPU timings use `std::chrono::high_resolution_clock`. Disk loading, saving, Python orchestration, and additional quality diagnostics are outside the measured pipeline. End-to-end time includes intermediate-buffer cleanup. V0 reports zero host–device transfers and an unavailable CUDA kernel time.

Comparison experiments run V0 and GPU versions in the same Colab GPU session. The runner records GPU identity with each result. Enable its Colab/GPU requirement with:

```bash
python3 scripts/benchmark_runner.py --runs 5 --require-colab-gpu
```

## Midterm notebook

Open [Midterm_Project_Report.ipynb](notebooks/Midterm_Project_Report.ipynb) in Colab or Jupyter. It contains saved V0 results, synthetic ground-truth validation, panoramas, timing charts, an Amdahl estimate, dataset documentation, and a GPU plan.

1. Make source available through the notebook's Git mode or submitted-bundle mode. Submitted mode expects the build root at `/content/midterm_submission/src`, or the path specified by `SUBMITTED_ROOT`.
2. Make the prepared dataset available. The setup accepts a dataset ZIP when required images are missing.
3. Execute all cells from top to bottom. The notebook builds and runs the C++ implementation.
4. Save the executed notebook with its outputs. Synchronize numerical discussion with the saved execution.

The notebook uses OpenCV, NumPy, Matplotlib, and IPython. For local notebook execution:

```powershell
.\.venv\Scripts\python.exe -m pip install opencv-python-headless numpy matplotlib ipython jupyter
.\.venv\Scripts\python.exe -m jupyter notebook notebooks/Midterm_Project_Report.ipynb
```

Synthetic validation compares the estimated B-to-A homography against a known transformation on a fixed point grid. Inlier RMSE measures the fit to selected feature correspondences. The report includes both measures and panorama previews for visual assessment.

The current report has administrative fields awaiting confirmation and timing discussion requiring synchronization with its latest saved output. Source-file author/adaptation headers and fresh execution with the final submitted source bundle remain submission checks.

## Project organization

```
CSC14120_Panorama_Stitching/
├── README.md                           # Setup, build, CLI usage, hardware specs, and YouTube link
├── Makefile                            # Compiles current V0 .cpp sources
│
├── notebooks/
│   ├── Midterm_Project_Report.ipynb    # Executed midterm report
│   ├── Final_Report.ipynb              # Final-report placeholder
│   └── Final_Report.pdf                # Final PDF placeholder
│
├── dataset/                            # Self-captured 16:9 image pairs across 3 resolutions
│   ├── site1_indoor/
│   │   ├── raw/                       # Original image pair
│   │   ├── small_1280x720/             # indoor_left/right_1280x720.jpg
│   │   ├── medium_1920x1080/           # indoor_left/right_1920x1080.jpg
│   │   └── large_3840x2160/            # left.jpg, right.jpg
│   ├── site2_outdoor/
│   │   ├── small_1280x720/
│   │   ├── medium_1920x1080/
│   │   └── large_3840x2160/
│   ├── site3_custom/
│   │   ├── small_1280x720/
│   │   ├── medium_1920x1080/
│   │   └── large_3840x2160/
│   └── challenging_pair/               # Apartment building and foreground foliage
│       ├── small_1280x720/
│       ├── medium_1920x1080/
│       └── large_3840x2160/
│
├── include/                            # Shared headers and GPU placeholders
│   ├── common.h                        # Image, descriptor, match, and timing types
│   ├── deterministic_tables.h          # Gaussian coefficients
│   ├── brief_pattern.h                 # Fixed BRIEF sampling pairs
│   ├── image_io.h                      # Wrapper for image loading/saving (excluded from timers)
│   ├── timer.h                         # std::chrono CPU timer
│   ├── verifier.h                      # Cross-version verifier placeholder
│   ├── v0_sequential.h                 # Declarations for V0 single-thread C++ pipeline
│   └── gpu_versions.cuh                # GPU interface placeholder
│
├── src/                                # Implementation files (.cpp, .cu)
│   ├── main.cpp                        # CLI parser (--version, --runs), warm-up, and timing reporter
│   ├── image_io.cpp                    # Image file read/write implementation
│   ├── verifier.cpp                    # Cross-version verifier placeholder
│   │
│   ├── v0_sequential/                  # Version 0: Single-threaded C/C++ reference implementation
│   │   ├── v0_pipeline.cpp             # Orchestrates Stages 1-6 with std::chrono per-stage timers
│   │   ├── v0_stage1_preprocess.cpp    # Greyscale conversion & separable Gaussian blur
│   │   ├── v0_stage2_keypoints.cpp     # Harris response, NMS, and Top-K selection
│   │   ├── v0_stage3_descriptor.cpp    # 256-bit binary descriptor extraction
│   │   ├── v0_stage4_matching.cpp      # Brute-force Hamming distance, ratio test, cross-check
│   │   ├── v0_stage5_ransac.cpp        # Normalized fitting, RANSAC, and refinement
│   │   └── v0_stage6_warp_blend.cpp    # Canvas bounding box, inverse bilinear warp, feathering
│   │
│   ├── v1_cuda_basic/                  # Planned V1; current files are placeholders
│   │   ├── v1_pipeline.cu
│   │   ├── v1_stages_1_to_3.cu
│   │   └── v1_stages_4_to_6.cu
│   │
│   ├── v2_cuda_opt1/                   # Planned first optimization
│   │   └── v2_pipeline.cu              # Placeholder
│   │
│   └── v3_cuda_opt2/                   # Planned second optimization
│       └── v3_pipeline.cu              # Placeholder
│
├── third_party/                        # External single-header libraries (must be cited in report)
│   ├── stb_image.h                     # Lightweight image loader
│   └── stb_image_write.h               # Lightweight image writer
│
├── scripts/
│   ├── image_resize.py                # Raw image decoding, cropping, and resizing
│   ├── generate_brief_pattern.py      # Writes include/brief_pattern.h
│   ├── benchmark_runner.py            # V0 batch measurements and diagnostics
│   ├── benchmark_quality.cpp          # Untimed V0 homography/inlier diagnostics
│   ├── prepare_dataset.py             # Placeholder
│   └── run_benchmarks.sh              # Placeholder
│
└── outputs/                            # Generated panoramas and profiling logs
    ├── panoramas/                      # Grouped by site, size, and benchmark batch
    ├── benchmarks/                     # Logs, CSV tables, metadata, and archived executables
    ├── validation/                     # Synthetic correctness evidence
    └── profiles/                       # Reserved for GPU profiling evidence
```

## References and assistance

- Harris and Stephens, *A Combined Corner and Edge Detector*, 1988. [Publication](https://www.bmva.org/bmvc/1988/avc-88-023.html)
- Calonder et al., *BRIEF: Binary Robust Independent Elementary Features*, 2010. [Authors' project](https://www.epfl.ch/labs/cvlab/research/descriptors-and-keypoints/research-detect-brief/)
- Fischler and Bolles, *Random Sample Consensus*, 1981. [Publication](https://doi.org/10.1145/358669.358692)
- Image I/O uses `stb_image` and `stb_image_write`, with their bundled notices retained. [stb repository](https://github.com/nothings/stb)

OpenAI ChatGPT/Codex assisted with implementation proposals, explanations, debugging, benchmark and validation code, and report drafting. The team reviews the code and remains responsible for its behavior, results, and source attribution.
