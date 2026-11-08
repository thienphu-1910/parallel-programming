#ifndef V0_SEQUENTIAL_H
#define V0_SEQUENTIAL_H

#include "common.h"

namespace panorama {

    // Returns Gaussian-smoothed greyscale
    ImageGray v0_preprocess(const ImageRGB& input);
    struct Keypoint { float x; float y; float response; };
    std::vector<Keypoint> v0_detect_keypoints(const ImageGrey& grey);
}


#endif // V0_SEQUENTIAL_H