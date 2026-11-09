#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
#include <vector>
#include <array>
#include <cstddef>

namespace panorama {

    // RGB Image Struct
    struct ImageRGB {
        int width = 0;
        int height = 0;
        std::vector<std::uint8_t> pixels;
    };

    // Greyscaled Image
    struct ImageGrey {
        int width = 0;
        int height = 0;
        std::vector<float> pixels;
    };

    // 256-bit BRIEF Descriptor
    inline constexpr std::size_t kBriefBits = 256; // Number of bits
    inline constexpr std::size_t kBriefWords = kBriefBits / 32; // Number of 32-bit words
    struct Descriptor {
        std::array<std::uint32_t, kBriefWords> words{};
    };

    // Match between two keypoints
    struct Match {
        std::size_t index_a;
        std::size_t index_b;
        int distance; // Hamming Distance      
    };

    // Threshold = 0.8
    inline constexpr int kMatchRatioNumerator = 4;
    inline constexpr int kMatchRatioDenominator = 5;
}


#endif // COMMON_H

