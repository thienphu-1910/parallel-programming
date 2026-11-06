#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
#include <vector>

namespace panorama {

    // RGB Image Struct
    struct ImageRGB {
        int width = 0;
        int height = 0;
        std::vector<std::uit8_t> pixels;
    };

    // Greyscaled Image
    struct ImageGrey {
        int width = 0;
        int height = 0;
        std::vector<float> pixels;
    };

}


#endif COMMON_H

