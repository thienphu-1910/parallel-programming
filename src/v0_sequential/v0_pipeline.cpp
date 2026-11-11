#include "v0_sequential.h"
#include "timer.h"

#include <stdexcept>
#include <string>

namespace panorama {

    V0Result v0_run_pipeline(const ImageRGB& image_a, const ImageRGB& image_b){
        V0Result result;
        CpuTimer total_timer;

        // Intermediate images and feature buffers are destroyed
        // before recording total pipeline time.
        {
            CpuTimer stage_timer;

            // --------------------------------------------------
            // Stage 1: RGB -> smoothed grayscale
            // --------------------------------------------------
            const ImageGrey gray_a = v0_preprocess(image_a);
            const ImageGrey gray_b = v0_preprocess(image_b);

            result.timings.stage_ms[0] = stage_timer.elapsed_ms();

            // --------------------------------------------------
            // Stage 2: Harris keypoint detection
            // --------------------------------------------------
            stage_timer.reset();

            const auto keypoints_a = v0_detect_keypoints(gray_a);
            const auto keypoints_b = v0_detect_keypoints(gray_b);

            result.timings.stage_ms[1] = stage_timer.elapsed_ms();

            result.keypoints_a = keypoints_a.size();
            result.keypoints_b = keypoints_b.size();

            // --------------------------------------------------
            // Stage 3: BRIEF descriptor extraction
            // --------------------------------------------------
            stage_timer.reset();

            const auto descriptors_a =
                v0_compute_descriptors(gray_a, keypoints_a);

            const auto descriptors_b =
                v0_compute_descriptors(gray_b, keypoints_b);

            result.timings.stage_ms[2] = stage_timer.elapsed_ms();

            // --------------------------------------------------
            // Stage 4: Hamming matching, ratio test, cross-check
            // --------------------------------------------------
            stage_timer.reset();

            const auto matches = v0_match_descriptors(
                descriptors_a,
                descriptors_b
            );

            result.timings.stage_ms[3] = stage_timer.elapsed_ms();
            result.match_count = matches.size();

            if (matches.size() < 4) {
                throw std::runtime_error(
                    "V0: homography estimation requires at least "
                    "four matches; found " +
                    std::to_string(matches.size()) + "."
                );
            }

            // --------------------------------------------------
            // Stage 5: Estimate homography mapping B -> A
            // --------------------------------------------------
            stage_timer.reset();

            result.homography = v0_estimate_homography(
                keypoints_a,
                keypoints_b,
                matches
            );

            result.timings.stage_ms[4] = stage_timer.elapsed_ms();

            if (!result.homography.valid) {
                throw std::runtime_error(
                    "V0: RANSAC failed. Keypoints A=" +
                    std::to_string(keypoints_a.size()) +
                    ", keypoints B=" +
                    std::to_string(keypoints_b.size()) +
                    ", accepted matches=" +
                    std::to_string(matches.size()) + "."
                );
            }

            // --------------------------------------------------
            // Stage 6: Warp and blend the original RGB images
            // --------------------------------------------------
            stage_timer.reset();

            result.panorama = v0_stitch(
                image_a,
                image_b,
                result.homography
            );

            result.timings.stage_ms[5] = stage_timer.elapsed_ms();
        }

        result.timings.total_ms = total_timer.elapsed_ms();

        return result;
    }
}