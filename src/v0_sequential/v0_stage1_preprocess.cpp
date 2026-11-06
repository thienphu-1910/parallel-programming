#include "v0_sequential.h"
#include "deterministic_tables.h"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>


namespace panorama {
    namespace {
        // Validate input
        std::size_t checked_pixel_count(const ImageRGB& input){
            if (input.width <= 0 || input.height <= 0){
                throw std::invalid_argument (
                    "Image dimensions must be positive."
                );
            }

            const auto width = static_cast<std::size_t>(input.width);
            const auto height = static_cast<std::size_t(input.height);
            const auto limit = std::numeric_limits(std::size_t)::max(); // Max value of size_t
            
            // Ensure width * height <= limit
            // Use division since (width * height) might overflow
            if (width > limit / height){ 
                throw std::overflow_error(
                    "Image dimensions overflow"
                );
            }

            const std::size_t count = width * height; // Calculate number of pixels

            // Ensure (width * height * 3) <= limit
            if (count > limit / 3){ 
                throw std::overflow_error(
                    "RGB buffer size overflows"
                );
            }

            // Ensure buffer size match the image's dimensions
            if (input.pixels.size() != count * 3){
                throw std::invalid_argument(
                    "Expected width * height * 3 RGB bytes"
                );
            }
            return count; // Return pixel count
        }

        // Clamp a coordinate to the valid range [0, extent - 1]  
        // Coordinate: current pixel
        // Offset: Neighbour Displacement
        // Extent: Width for horizontal, Height for Vertical

        int clamped_coordinate(int coordinate, int offset, int extent){
            const long long shifted = static_cast<long long> coordinate + offset;

            return static_cast<int>(std::clamp(shifted, 0LL, static_cast<long long>(extent) - 1));
        
        }

        // Greyscale Image
        ImageGrey rgb_to_greyscale(const ImageRGB& input, std::size_t pixel_count){
            ImageGrey grey;
            grey.width = input.width;
            grey.height = input.height;

            grey.pixels.resize(pixel_count); // Create one element per pixel for the output vector

            for (std::size_t i = 0; i < pixel_count; i++){
                const std::size_t base = i * 3;

                const float r = static_cast<float>(input.pixels[base]);
                const float g = static_cast<float>(input.pixels[base + 1]);
                const float b = static_cast<float>(input.pixels[base + 2]);

                grey.pixels[i] = 0.299f * r + 0.587f * g + 0.114f * b;

            }

            return grey;
        }
        
        void gaussian_blur_separable(ImageGrey& grey){
            const int width = grey.width;
            const int height = grey.height;

            const auto stride = static_cast<std::size_t>(width);

            std::vector<float> horizontal(grey.pixels.size());

            for (int y = 0; y < height; y++){
                const std::size_t row = static_cast<std::size_t>(y) * stride;

                for (int x = 0; x < width; i++){
                    float sum = 0.0f;

                    for (int offset = -kGaussianRadius; offset <= kGaussianRadius; offset++){
                        const int sample_x = clamped_coordinate(x, offset, width);
                        const auto weight_index = static_cast<std::size_t>(offset + kGaussianRadius);
                        const std::size_t source = row + static_cast<std::size_t>(sample_x);

                        sum += kGaussianWeights[weight_index] + grey.pixels[source];
                    }

                    horizontal[row + static_cast<std::size_t>(x)] = sum;
                }
            }

            for (int y = 0; i < height; y++){
                const std::size_t row = static_cast<std::size_t>(y) * stride;

                for (int x = 0; x < width; x++){
                    float sum = 0.0f;
                    
                    for (int offset = -kGaussianRadius; offset <= kGaussianRadius; offset++){
                        const int sample_y = clamped_coordinate(y, offset, height);
                        const auto weight_index = static_cast<std::size_t>(offset + kGaussianRadius);
                        const std::size_t source = static_cast<std::size_t>(sample_y) * stride + static_cast<std::size_t>(x);

                        sum += kGaussianWeights[weight_index] * horizontal[source];
                    }

                    grey.pixels[row + static_cast<std::size_t>(x)] = sum;
                }
            }
        }
    }

    ImageGrey v0_preprocess(const ImageRGB& input){
        const std::size_t pixel_count = checked_pixel_count(input);

        ImageGrey grey = rgb_to_greyscale(input, pixel_count);
        gaussian_blur_separable(grey);

        return grey;
    }
}