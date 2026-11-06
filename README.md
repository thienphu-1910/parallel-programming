# PANORAMA

## 1. Team Participants

- Hà Thư Hoàng - 23127467
- Trương Công Thiên Phú - 23127455

## 2. Project Organization

```
CSC14120_Panorama_Stitching/
├── README.md                           # Setup, build, CLI usage, hardware specs, and YouTube link
├── Makefile                            # Compiles all .cpp and .cu files into ./stitch
│
├── notebooks/
│   ├── Final_Report.ipynb              # Main Colab-ready report (runs top-to-bottom via Run all)
│   └── Final_Report.pdf                # Exported PDF archive of the executed notebook
│
├── dataset/                            # Self-captured 16:9 image pairs across 3 resolutions
│   ├── site1_indoor/
│   │   ├── small_1280x720/             # left.jpg, right.jpg
│   │   ├── medium_1920x1080/           # left.jpg, right.jpg
│   │   └── large_3840x2160/            # left.jpg, right.jpg
│   ├── site2_outdoor/
│   │   ├── small_1280x720/
│   │   ├── medium_1920x1080/
│   │   └── large_3840x2160/
│   ├── site3_custom/
│   │   ├── small_1280x720/
│   │   ├── medium_1920x1080/
│   │   └── large_3840x2160/
│   └── challenging_pair/               # For failure analysis (e.g., low texture, moving people)
│       ├── small_1280x720/
│       ├── medium_1920x1080/
│       └── large_3840x2160/
│
├── include/                            # Header files (.h, .cuh)
│   ├── common.h                        # Shared structs (Image, Keypoint, Match, Homography, Config)
│   ├── deterministic_tables.h          # Fixed BRIEF bit-patterns & RANSAC seeds for V0-V3 consistency
│   ├── image_io.h                      # Wrapper for image loading/saving (excluded from timers)
│   ├── timer.h                         # std::chrono (CPU) and cudaEvent (GPU) timing utilities
│   ├── verifier.h                      # Compares V1/V2/V3 outputs against V0 ground truth
│   ├── v0_sequential.h                 # Declarations for V0 single-thread C++ pipeline
│   └── gpu_versions.cuh                # Declarations for V1 (Basic) and V2, V3+ (Optimised) pipelines
│
├── src/                                # Implementation files (.cpp, .cu)
│   ├── main.cpp                        # CLI parser (--version, --runs), warm-up, and timing reporter
│   ├── image_io.cpp                    # Image file read/write implementation
│   ├── verifier.cpp                    # Quantitative verification against V0 (inliers, RMSE, pixel diff)
│   │
│   ├── v0_sequential/                  # Version 0: Single-threaded C/C++ reference implementation
│   │   ├── v0_pipeline.cpp             # Orchestrates Stages 1-6 with std::chrono per-stage timers
│   │   ├── v0_stage1_preprocess.cpp    # Greyscale conversion & separable Gaussian blur
│   │   ├── v0_stage2_keypoints.cpp     # Harris/Shi-Tomasi response, NMS, and Top-K selection
│   │   ├── v0_stage3_descriptor.cpp    # 256-bit binary descriptor extraction
│   │   ├── v0_stage4_matching.cpp      # Brute-force Hamming distance, ratio test, cross-check
│   │   ├── v0_stage5_ransac.cpp        # 4-point DLT solver, RANSAC loop, and inlier refinement
│   │   └── v0_stage6_warp_blend.cpp    # Canvas bounding box, inverse bilinear warp, feathering
│   │
│   ├── v1_cuda_basic/                  # Version 1: Straightforward global-memory GPU kernels
│   │   ├── v1_pipeline.cu              # Host wrapper with cudaEvent per-stage and H2D/D2H timers
│   │   ├── v1_stages_1_to_3.cu         # Naive per-pixel and per-keypoint kernels
│   │   └── v1_stages_4_to_6.cu         # Naive matching, RANSAC, and inverse warping kernels
│   │
│   ├── v2_cuda_opt1/                   # Version 2: First targeted optimisation (builds on V1)
│   │   └── v2_pipeline.cu              # e.g., Shared-memory tiling & constant memory for stages 1 & 4
│   │
│   └── v3_cuda_opt2/                   # Version 3: Second targeted optimisation (builds on V2)
│       └── v3_pipeline.cu              # e.g., CUDA streams for Image A/B + warp-level RANSAC/popc
│
├── third_party/                        # External single-header libraries (must be cited in report)
│   ├── stb_image.h                     # Lightweight image loader
│   └── stb_image_write.h               # Lightweight image writer
│
├── scripts/                            # Helper scripts for dataset prep and profiling
│   ├── prepare_dataset.py              # Crops raw photos to 16:9 and resizes to 720p, 1080p, 4K
│   └── run_benchmarks.sh               # Automates --version 0..3 across all sizes for Colab tables
│
└── outputs/                            # Generated panoramas and profiling logs
    ├── panoramas/                      # Output images from V0, V1, V2, V3
    └── profiles/                       # Nsight Systems / Nsight Compute summaries and screenshots
```