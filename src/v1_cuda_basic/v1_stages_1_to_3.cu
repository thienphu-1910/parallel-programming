#include "common.h"
#include "deterministic_tables.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace panorama {
    namespace {
        // 16 * 16 = 256 threads per block
        // 1 + (n - 1) / 16 for positive dimensions
        constexpr unsigned kBlockSide = 16; 
        constexpr unsigned kMaxGridRows = 65535 // CUDA max grid's Y dimension

        static_assert(kGaussianWeights.size() == 2 * kGaussianRadius + 1); //  Offset from -R to +R


        // GaussianTaps
        // Copies the V0 values into a plain struct and passes it by value to each blur launch
        struct GaussianTaps{
            float values[kGaussianWeights.size()];
        };

        __device__ std::size_t pixel_index(unsigned x, unsigned y, unsigned width){
            return static_cast<std::size_t>(y) * width + x; // Index of a pixel
        }
        
        // Clamp coordinate - Avoids out-of-bounds access by clamping to the image dimensions
        __device__ unsigned clamped_coordinate(unsigned coordinate, int offset, unsigned extent){   
            const long long shifted = static_cast<long long>(coordinate) + offset;
            const long long last = static_cast<long long>(extent) - 1;

            if (shifted < 0){
                return 0;
            }

            return static_cast<unsigned>(shifted > last ? last : shifted);
        }

        // Greyscale image
        __global__ void rgb_to_grey(const std::uint8_t* rgb, float* grey, unsigned width, unsigned height){
            const unsigned x = blockIdx.x * blockDim.x + threadIdx.x; // Thread Index
            if (x >= width) {
                return;
            }

            const unsigned first_y = blockIdx.y * blockDim.y + threadIdx.y;
            const unsighed row_stride = gridDim.y * blockDim.y;

            for (unsigned y = first_y; y < height; y += row_stride){
                float sum = 0.0f;
                for (int offset = -kGaussianRadius; offset <= kGaussianRadius; offset++){
                    const unsigned sx = Horizontal ? clamped_coordinate(x, offset, width) : x;
                    const unsigned sy = Horizontal ? y : clamped_coordinate(y, offset, height);
                    sum += taps.values[offset + kGaussianRadius] * input[pixel_index(sx, sy, width)];
                }
                output[pixel_index(x, y, width)] = sum;

            }
        }

        void check_launch(const char* operation){
            const cudaError_t status = cudaGetLastError();
            if (status != cudaSuccess){
                throw std:: runtime_error(std::string(operation) + ": " + cudaGetErrorString(status)); 
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

    void v1_launch_preprocess(const std::uint8_t* rgb, float* smoothed, float* horizontal, int width, int height, cudaStream_t stream){
        if (width <= 0 || height <= 0) {
        throw std::invalid_argument("V1 Stage 1: launch dimensions must be positive");
    }

        const auto w = static_cast<unsigned>(width);
        const auto h = static_cast<unsigned>(height);
        const unsigned grid_x = 1 + (w - 1) / kBlockSide;
        const unsigned grid_y = 1 + (h - 1) / kBlockSide;
        const dim3 block(kBlockSide, kBlockSide);
        const dim3 grid(grid_x, grid_y < kMaxGridRows ? grid_y : kMaxGridRows);

        GaussianTaps taps{};
        for (std::size_t i = 0; i < kGaussianWeights.size(); ++i) {
            taps.values[i] = kGaussianWeights[i];
        }

        rgb_to_gray<<<grid, block, 0, stream>>>(rgb, smoothed, w, h);
        check_launch("V1 Stage 1 RGB to grayscale");

        gaussian_pass<true><<<grid, block, 0, stream>>>(smoothed, horizontal, w, h, taps);
        check_launch("V1 Stage 1 horizontal blur");

        gaussian_pass<false><<<grid, block, 0, stream>>>(horizontal, smoothed, w, h, taps);
        check_launch("V1 Stage 1 vertical blur");
    }
}