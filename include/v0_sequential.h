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

    // Match descriptors
    std::vector<Match> v0_match_descriptors(const std::vector<Descriptor>& descriptor_a,
                                            const std::vector<Descriptor>& descriptor_b);
                                            
}


#endif // V0_SEQUENTIAL_H