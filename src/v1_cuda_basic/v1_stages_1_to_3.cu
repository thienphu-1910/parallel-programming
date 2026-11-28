#include "v1_stages_1_to_3.h"
#include "deterministic_tables.h"

#include <algorithm>
#include <cmath>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace panorama {
    namespace {
        constexpr unsigned kBlockSide = 16;        // A 16 × 16 block covers 256 pixels.
        constexpr unsigned kMaxGridRows = 65535;   // CUDA's grid limit in the y dimension.

        // Offsets from -radius through +radius must have one weight each.
        static_assert(kGaussianWeights.size() == 2 * kGaussianRadius + 1);

        // Passing the weights by value lets the same kernel use Stage 1's Gaussian
        // weights here and Stage 2's Harris weights when smoothing tensor products.
        struct GaussianTaps {
            float values[kGaussianWeights.size()];
        };

        __device__ std::size_t pixel_index(unsigned x, unsigned y, unsigned width) {
            // Widen before multiplying so the row offset is calculated in size_t.
            return static_cast<std::size_t>(y) * width + x;
        }

        __device__ unsigned clamped_coordinate(unsigned coordinate, int offset, unsigned extent) {
            // An offset of -2 at x = 0 must produce a negative intermediate value.
            // Signed arithmetic lets us clamp that sample back to the first pixel.
            const long long shifted = static_cast<long long>(coordinate) + offset;
            const long long last = static_cast<long long>(extent) - 1;

            if (shifted < 0) {
                return 0;
            }
            return static_cast<unsigned>(shifted > last ? last : shifted);
        }

        __global__ void rgb_to_gray(const std::uint8_t* rgb, float* gray, unsigned width, unsigned height) {
            const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
            if (x >= width) {
                return;  // The last block can extend beyond the right edge.
            }

            // For the project images, each active thread handles one pixel.
            // If the grid height is capped, it also handles later rows in this column.
            const unsigned first_y = blockIdx.y * blockDim.y + threadIdx.y;
            const unsigned row_stride = gridDim.y * blockDim.y;

            for (unsigned y = first_y; y < height; y += row_stride) {
                const std::size_t i = pixel_index(x, y, width);
                const std::size_t base = 3 * i;

                // RGB is packed as three consecutive bytes per pixel. Keep grayscale
                // in float on V0's 0–255 scale so the blur retains fractional intensity.
                const float r = static_cast<float>(rgb[base]);
                const float g = static_cast<float>(rgb[base + 1]);
                const float b = static_cast<float>(rgb[base + 2]);

                gray[i] = 0.299f * r + 0.587f * g + 0.114f * b;
            }
        }

        template <bool Horizontal>
        __global__ void gaussian_pass(const float* input, float* output, unsigned width, unsigned height, GaussianTaps taps) {
            const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
            if (x >= width) {
                return;
            }

            const unsigned first_y = blockIdx.y * blockDim.y + threadIdx.y;
            const unsigned row_stride = gridDim.y * blockDim.y;

            for (unsigned y = first_y; y < height; y += row_stride) {
                float sum = 0.0f;

                // Horizontal selects the sampling axis at compile time. Both passes
                // visit offsets -2, -1, 0, 1, 2 in the same order as V0.
                for (int offset = -kGaussianRadius; offset <= kGaussianRadius; ++offset) {
                    const unsigned sx = Horizontal ? clamped_coordinate(x, offset, width) : x;
                    const unsigned sy = Horizontal ? y : clamped_coordinate(y, offset, height);

                    // At an edge, several taps can read the same boundary pixel.
                    // All five weights still contribute, including on a 1 × 1 image.
                    const float weight = taps.values[offset + kGaussianRadius];
                    sum += weight * input[pixel_index(sx, sy, width)];
                }

                output[pixel_index(x, y, width)] = sum;
            }
        }

        using PixelOffset = std::ptrdiff_t;

        constexpr int kHarrisMargin = 32;
        constexpr float kHarrisK = 0.04f;
        constexpr float kHarrisRelativeThreshold = 0.01f;
        constexpr PixelOffset kHarrisMaxKeypoints = 2000;

        GaussianTaps make_harris_taps() {
            constexpr double sigma = 1.5;
            static_assert(kGaussianRadius == 2, "Harris uses a five-tap window");
            GaussianTaps taps{};
            double total = 0.0;
            // Match V0: accumulate the original double weights, then normalize the stored floats.
            for (int offset = -kGaussianRadius; offset <= kGaussianRadius; ++offset) {
                const double d = static_cast<double>(offset);
                const double weight = std::exp(-(d * d) / (2.0 * sigma * sigma));
                taps.values[offset + kGaussianRadius] = static_cast<float>(weight);
                total += weight;
            }
            for (float& tap : taps.values) {
                tap = static_cast<float>(tap / total);
            }
            return taps;
        }

        struct ComputeTensor {
            const float* gray;
            float* xx;
            float* yy;
            float* xy;
            unsigned width;
            unsigned height;

            __device__ void operator()(PixelOffset i) const {
                const unsigned x = static_cast<unsigned>(i % width);
                const unsigned y = static_cast<unsigned>(i / width);
                const unsigned left = clamped_coordinate(x, -1, width);
                const unsigned right = clamped_coordinate(x, 1, width);
                const unsigned top = clamped_coordinate(y, -1, height);
                const unsigned bottom = clamped_coordinate(y, 1, height);

                const float tl = gray[pixel_index(left, top, width)];
                const float tc = gray[pixel_index(x, top, width)];
                const float tr = gray[pixel_index(right, top, width)];
                const float ml = gray[pixel_index(left, y, width)];
                const float mr = gray[pixel_index(right, y, width)];
                const float bl = gray[pixel_index(left, bottom, width)];
                const float bc = gray[pixel_index(x, bottom, width)];
                const float br = gray[pixel_index(right, bottom, width)];

                // Sobel gradients feed the three independent entries of the symmetric tensor.
                const float gx = (tr + 2.0f * mr + br) - (tl + 2.0f * ml + bl);
                const float gy = (bl + 2.0f * bc + br) - (tl + 2.0f * tc + tr);
                xx[i] = gx * gx;
                yy[i] = gy * gy;
                xy[i] = gx * gy;
            }
        };

        struct ComputeResponse {
            const float* xx;
            const float* yy;
            const float* xy;
            float* response;

            __device__ void operator()(PixelOffset i) const {
                const float det = xx[i] * yy[i] - xy[i] * xy[i];
                const float trace = xx[i] + yy[i];
                response[i] = det - kHarrisK * trace * trace;
            }
        };

        struct InteriorPixel {
            int width;
            int interior_width;

            __host__ __device__ PixelOffset operator()(PixelOffset j) const {
                const PixelOffset y = j / interior_width + kHarrisMargin;
                const PixelOffset x = j % interior_width + kHarrisMargin;
                return y * width + x;
            }
        };

        struct ReadResponse {
            const float* response;

            __host__ __device__ float operator()(PixelOffset i) const {
                return response[i];
            }
        };

        struct IsHarrisPeak {
            const float* response;
            int width;
            float threshold;

            __host__ __device__ bool operator()(PixelOffset i) const {
                const float center = response[i];
                if (center <= 0.0f || center <= threshold) {
                    return false;
                }
                // Candidates are interior pixels, but their neighbors use the full image stride.
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const PixelOffset j = i + static_cast<PixelOffset>(dy) * width + dx;
                        const float neighbor = response[j];
                        // The center itself cannot reject the candidate. Equal earlier neighbors can.
                        if (neighbor > center || (neighbor == center && j < i)) {
                            return false;
                        }
                    }
                }
                return true;
            }
        };

        struct StrongerFirst {
            const float* response;

            __host__ __device__ bool operator()(PixelOffset a, PixelOffset b) const {
                if (response[a] != response[b]) {
                    return response[a] > response[b];
                }
                return a < b;  // Row-major offsets break ties by y, then x.
            }
        };

        struct MakeKeypoint {
            const float* response;
            int width;

            __host__ __device__ Keypoint operator()(PixelOffset i) const {
                return Keypoint{static_cast<float>(i % width),
                                static_cast<float>(i / width), response[i]};
            }
        };

        void check_launch(const char* operation) {
            // Check immediately after enqueueing. The pipeline also checks its later
            // synchronization because execution errors can surface after launch.
            const cudaError_t status = cudaGetLastError();
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    std::string(operation) + ": " + cudaGetErrorString(status)
                );
            }
        }
    }

    std::size_t v1_stage1_pixel_count(const ImageRGB& input){
        if (input.width <= 0 || input.height <= 0) {
        throw std::invalid_argument("V1 Stage 1: image dimensions must be positive");
    }

        const auto width = static_cast<std::size_t>(input.width);
        const auto height = static_cast<std::size_t>(input.height);
        const auto limit = std::numeric_limits<std::size_t>::max();

        // Check before multiplying. Once a size has overflowed, it can look small
        // enough to allocate even though the kernels will address the full image.
        if (width > limit / height) {
            throw std::overflow_error("V1 Stage 1: pixel count overflows");
        }

        const std::size_t count = width * height;
        if (count > limit / 3 || count > limit / sizeof(float)) {
            throw std::overflow_error("V1 Stage 1: image buffer size overflows");
        }

        if (input.pixels.size() != 3 * count) {
            throw std::invalid_argument("V1 Stage 1: expected three RGB bytes per pixel");
        }

        return count;
    }

    // The caller validates the host image before uploading it and owns all three
    // device buffers. smoothed and horizontal must be distinct float planes.
    void v1_launch_preprocess(const std::uint8_t* rgb, float* smoothed, float* horizontal, int width, int height, cudaStream_t stream){
        if (width <= 0 || height <= 0) {
        throw std::invalid_argument("V1 Stage 1: launch dimensions must be positive");
    }

        const auto w = static_cast<unsigned>(width);
        const auto h = static_cast<unsigned>(height);

        // Round up so partial blocks cover the right and bottom edges.
        const unsigned grid_x = 1 + (w - 1) / kBlockSide;
        const unsigned grid_y = 1 + (h - 1) / kBlockSide;
        const dim3 block(kBlockSide, kBlockSide);
        const dim3 grid(grid_x, grid_y < kMaxGridRows ? grid_y : kMaxGridRows);

        GaussianTaps taps{};
        for (std::size_t i = 0; i < kGaussianWeights.size(); ++i) {
            taps.values[i] = kGaussianWeights[i];
        }


        // The first and last launches share the smoothed buffer. The original RGB
        // stays available for Stage 6, which renders the panorama in color.
        rgb_to_gray<<<grid, block, 0, stream>>>(rgb, smoothed, w, h);
        check_launch("V1 Stage 1 RGB to grayscale");

        // Each output reads neighboring pixels. A separate destination prevents
        // one thread from overwriting a value another thread still needs.
        gaussian_pass<true><<<grid, block, 0, stream>>>(smoothed, horizontal, w, h, taps);
        check_launch("V1 Stage 1 horizontal blur");

        // Both passes use the same stream. The horizontal image is complete
        // before the vertical pass reads it, so smoothed is safe to reuse.
        gaussian_pass<false><<<grid, block, 0, stream>>>(horizontal, smoothed, w, h, taps);
        check_launch("V1 Stage 1 vertical blur");
    }

    // The caller owns the device buffers. xx, yy, xy, and scratch each hold one
    // full float image. candidates must fit all interior pixels before ranking.
    // output needs room for at most 2,000 keypoints.

    std::size_t v1_detect_keypoints(const float* smoothed, int width, int height, float* xx, float* yy, float* xy, float* scratch, std::ptrdiff_t* candidates, Keypoint* output, cudaStream_t stream){
        if (width <= 0 || height <= 0){
            throw std::invalid_argument("V1 Stage 2: dimensions must be positive");
        }

        if (width <= 2 * kHarrisMargin || height <= 2 * kHarrisMargin){
            return 0;
        }

        const auto w = static_cast<unsigned>(width);
        const auto h = static_cast<unsigned>(height);
        const PixelOffset pixel_count = PixelOffset(width) * height;
        const int interior_width = width - 2 * kHarrisMargin;
        const int interior_height = height - 2 * kHarrisMargin;
        const PixelOffset interior_count = PixelOffset(interior_width) * interior_height;

        // Use the same stream for Thrust and our kernels so each pass sees the completed output of the preceding pass.
        const auto policy = thrust::cuda::par.on(stream);
        const auto first = thrust::make_counting_iterator<PixelOffset>(0);

        thrust::for_each_n(policy, first, pixel_count, ComputeTensor{smoothed, xx, yy, xy, w, h});

        const dim3 block(kBlockSide, kBlockSide);
        const unsigned grid_x = 1 + (w - 1) / kBlockSide;
        const unsigned grid_y = 1 + (h - 1) / kBlockSide;
        const dim3 grid(grid_x, std::min(grid_y, kMaxGridRows));
        static const GaussianTaps taps = make_harris_taps();


        // A blur reads neighboring pixels. Writing the horizontal result to a
        // separate plane prevents one thread from overwriting another's input.
        // The vertical pass can then write the finished result back to plane.
        const auto smooth_product = [&](float* plane){
            gaussian_pass<true><<<grid, block, 0, stream>>>(plane, scratch, w, h, taps);
            check_launch("V1 Stage 2 horizontal tensor blur");

            gaussian_pass<false><<<grid, block, 0, stream>>>(scratch, plane, w, h, taps);
            check_launch("V1 Stage 2 vertical tensor blur");
        };

        smooth_product(xx);
        smooth_product(yy);
        smooth_product(xy);

        // After the last vertical pass, no blur needs scratch. Reuse that
        // allocation for the response map and keep it alive through sorting.
        float* response = scratch;
        thrust::for_each_n(policy, first, pixel_count, ComputeResponse{xx, yy, xy, response});

        const auto interior = thrust::make_transform_iterator(first, InteriorPixel{width, interior_width});
        const auto interior_end = interior + interior_count;

        // V0 sets the threshold from the strongest eligible interior response.
        // Starting from zero also handles an image with only nonpositive responses.
        const float max_response = thrust::transform_reduce(policy, interior, interior_end, ReadResponse{response}, 0.0f, thrust::maximum<float>{});

        if (max_response <= 0.0f){
            return 0;
        }

        const float threshold = kHarrisRelativeThreshold * max_response;
        const auto candidates_end = thrust::copy_if(policy, interior, interior_end, candidates, IsHarrisPeak{response, width, threshold});

        const PixelOffset survivor_count = candidates_end - candidates;
        if (survivor_count == 0) {
            return 0;
        }

        // Rank every surviving peak before taking the first 2,000. A strong
        // corner near the bottom of the image may appear late in raster order.
        thrust::sort(policy, candidates, candidates_end, StrongerFirst{response});

        // Stage 3 computes descriptor i from keypoint i, so emit the points in
        // their final ranked order and leave the array on the device.
        const PixelOffset kept = std::min(survivor_count, kHarrisMaxKeypoints);
        thrust::transform(policy, candidates, candidates + kept, output, MakeKeypoint{response, width});
        
        return static_cast<std::size_t>(kept);
    }

}