#include "v0_sequential.h"
#include "brief_pattern.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace panorama {
    namespace {
        // Validate fixed sampling offsets 
        constexpr bool valid_brief_pattern()
        {
            // Check if offsets are within bounds and distinct 
            for (const BriefPair& pair : kBriefPattern) {
                if (pair.dx1 < -kBriefRadius ||
                    pair.dx1 >  kBriefRadius ||
                    pair.dy1 < -kBriefRadius ||
                    pair.dy1 >  kBriefRadius ||
                    pair.dx2 < -kBriefRadius ||
                    pair.dx2 >  kBriefRadius ||
                    pair.dy2 < -kBriefRadius ||
                    pair.dy2 >  kBriefRadius) {
                    return false;
                }
                
                // Check if a pair is identical
                if (pair.dx1 == pair.dx2 &&
                    pair.dy1 == pair.dy2) {
                    return false;
                }
            }

            return true;
        }

        // Compile time validation
        // Ensure a 256-bit descriptor
        static_assert(kBriefBits == 256,
              "Expects a 256-bit descriptor.");

        // Ensure descriptor is aligned to 32-bit words
        static_assert(kBriefBits % 32 == 0,
              "Descriptor must contain whole 32-bit words.");

        // Ensure sampling pattern is valid
        static_assert(valid_brief_pattern(),
              "Invalid BRIEF sampling pattern.");

        // Validate greyscale image dimension 
        void validate_grey_image(const ImageGrey& grey){
            if (grey.width <= 0 || grey.height <= 0){
                throw std::invalid_argument(
                    "Image dimensions must be positive"
                );
            }

            const auto width = static_cast<std::size_t>(grey.width);
            const auto height = static_cast<std::size_t>(grey.height);
            const auto limit = std::numeric_limits<std::size_t>::max();

            if (width > limit / height) {
                throw std::overflow_error(
                    "Stage 3: image dimensions overflow."
                );
            }

            if (grey.pixels.size() != width * height) {
                throw std::invalid_argument(
                    "Stage 3: grayscale buffer size is invalid."
                );
            }
        }
        
        // Validate keypoint is within image bounds
        void validate_keypoint(const Keypoint& keypoint, int width, int height){
            const double x = static_cast<double>(keypoint.x);
            const double y = static_cast<double>(keypoint.y);

            // Validate keypoint coordinates
            if (!std::isfinite(x) || !std::isfinite(y)){
                throw std::invalid_argument(
                    "Keypoint coordinates must be finite"
                );
            }

            // Check if integer coordinates stored as floats
            if (std::trunc(x) != x || std::trunc(y) != y) {
                throw std::invalid_argument(
                    "Stage 3: keypoint coordinates must be integers."
                );
            }

            if (x < kBriefRadius ||
                y < kBriefRadius ||
                x >= static_cast<double>(width) - kBriefRadius ||
                y >= static_cast<double>(height) - kBriefRadius) {
                throw std::invalid_argument(
                    "Stage 3: keypoint patch crosses the image border."
                );
            }
        }


        // Get pixel indexes for a 2D coordinate
        std::size_t pixel_index(int x, int y, std::size_t stride){
            return static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x);
        }

        // Compute Keypoints
        Descriptor describe_keypoint(const ImageGrey& grey, const Keypoint& keypoint) {
            // Checked before this function is called
            const int x = static_cast<int>(keypoint.x);
            const int y = static_cast<int>(keypoint.y);

            const auto stride = static_cast<std::size_t>(grey.width);

            // Allocate descriptor
            Descriptor descriptor{};   

            // Compute each bit of the descriptors
            for (std::size_t bit = 0; bit < kBriefBits; bit++){
                const BriefPair& pair = kBriefPattern[bit];

                const int x1 = x + static_cast<int>(pair.dx1); // First pixel
                const int y1 = y + static_cast<int>(pair.dy1);
                const int x2 = x + static_cast<int>(pair.dx2); // Second pixel
                const int y2 = y + static_cast<int>(pair.dy2);

                const float first = grey.pixels[pixel_index(x1, y1, stride)]; 
                const float second = grey.pixels[pixel_index(x2, y2, stride)];


                // Compute and store in the descriptor
                if (first < second){
                    const std::size_t word = bit / 32;

                    const unsigned position = static_cast<unsigned>(bit % 32);

                    descriptor.words[word] |= (std::uint32_t{1} << position);
                }
            }

            return descriptor;
        }
    }
    
    std::vector<Descriptor> v0_compute_descriptors(const ImageGrey& grey, const std::vector<Keypoint>& keypoints){
        
        validate_grey_image(grey);

        for(const Keypoint& keypoint : keypoints){
            validate_keypoint(keypoint, grey.width, grey.height);
        }

        std::vector<Descriptor> descriptors(keypoints.size());

        for (std::size_t i = 0; i < keypoints.size(); i++){
            descriptors[i] = describe_keypoint(grey, keypoints[i]);
        }

        return descriptors;
    }
}