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

}


#endif // COMMON_H

