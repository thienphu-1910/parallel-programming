#ifndef V0_SEQUENTIAL_H
#define V0_SEQUENTIAL_H

#include "common.h"

namespace panorama {

    // Returns Gaussian-smoothed greyscale
    ImageGrey v0_preprocess(const ImageRGB& input);
    struct Keypoint { float x; float y; float response; };
    std::vector<Keypoint> v0_detect_keypoints(const ImageGrey& grey);

    // Calculate BRIEF Descriptors
    std::vector<Descriptor> v0_compute_descriptors(
        const ImageGrey& grey,
        const std::vector<Keypoint>& keypoints
    );
}


#endif // V0_SEQUENTIAL_H