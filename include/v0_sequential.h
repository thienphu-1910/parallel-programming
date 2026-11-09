#ifndef V0_SEQUENTIAL_H
#define V0_SEQUENTIAL_H

#include "common.h"

#include <array>
#include <cstddef>
#include <vector>

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

    // ---------------- Stage 5: homography estimation ----------------
    // H maps B coordinates into A coordinates (row-major 3x3).
    struct Homography {
        std::array<double, 9> h{};
        std::size_t inlier_count = 0;
        bool valid = false;
    };

    // Estimate H from matches (A[i] <-> B[j]) with deterministic RANSAC.
    // Returns valid = false when there are too few matches/inliers.
    Homography v0_estimate_homography(const std::vector<Keypoint>& keypoints_a,
                                      const std::vector<Keypoint>& keypoints_b,
                                      const std::vector<Match>& matches);

    // ---------------- Stage 6: warp + blend ----------------
    // Warps B into A's frame and feather-blends the overlap.
    ImageRGB v0_stitch(const ImageRGB& image_a,
                       const ImageRGB& image_b,
                       const Homography& homography);

}

#endif // V0_SEQUENTIAL_H

