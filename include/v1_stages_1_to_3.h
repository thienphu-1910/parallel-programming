#ifndef PANORAMA_V1_STAGES_1_TO_3_H
#define PANORAMA_V1_STAGES_1_TO_3_H

#include "v0_sequential.h"  // ImageRGB and Keypoint

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace panorama {

    // Validate the host RGB image before sizing the device buffers.
    std::size_t v1_stage1_pixel_count(const ImageRGB& input);

    // Enqueue Stage 1. smoothed and horizontal are distinct width * height
    // device float buffers. The final grayscale image is written to smoothed.
    void v1_launch_preprocess(const std::uint8_t* rgb, float* smoothed,
                              float* horizontal, int width, int height,
                              cudaStream_t stream);

    // xx, yy, xy and scratch are separate full-image device float buffers.
    // candidates holds one offset per interior pixel. output has capacity
    // for 2,000 ordered keypoints. The return value is the number written.
    std::size_t v1_detect_keypoints(const float* smoothed, int width, int height,
                                    float* xx, float* yy, float* xy, float* scratch,
                                    std::ptrdiff_t* candidates, Keypoint* output,
                                    cudaStream_t stream);

}  // namespace panorama

#endif  // PANORAMA_V1_STAGES_1_TO_3_H
