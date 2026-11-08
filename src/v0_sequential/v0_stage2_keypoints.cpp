#include "v0_sequential.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace panorama {
    namespace {

        // Harris parameters. Keep these identical in every version (V0..Vn).
        // Consider moving them to deterministic_tables.h later.
        constexpr float kHarrisK             = 0.04f; // Harris constant k in R = det - k * trace^2
        constexpr float kWindowSigma         = 1.5f;  // Gaussian window sigma for structure tensor
        constexpr int   kWindowRadius        = 2;     // 5x5 window (radius 2)
        constexpr float kRelativeThreshold   = 0.01f; // Discard R below 1% of max R
        constexpr int   kBorderMargin        = 32;    // Ignore pixels near the border
        constexpr std::size_t kMaxKeypoints  = 2000;  // Top-K

        inline std::size_t pixel_index(int x, int y, int width) {
            return static_cast<std::size_t>(y) * static_cast<std::size_t>(width)
                 + static_cast<std::size_t>(x);
        }

        inline int clamp_int(int value, int lo, int hi) {
            return value < lo ? lo : (value > hi ? hi : value);
        }

        // Sobel gradients Ix and Iy. Borders are handled by clamping coordinates.
        void compute_sobel_gradients(const ImageGrey& grey,
                                     std::vector<float>& ix,
                                     std::vector<float>& iy) {
            const int width  = grey.width;
            const int height = grey.height;

            ix.assign(grey.pixels.size(), 0.0f);
            iy.assign(grey.pixels.size(), 0.0f);

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    auto at = [&](int dx, int dy) -> float {
                        const int sx = clamp_int(x + dx, 0, width - 1);
                        const int sy = clamp_int(y + dy, 0, height - 1);
                        return grey.pixels[pixel_index(sx, sy, width)];
                    };

                    // Sobel X: [-1 0 1; -2 0 2; -1 0 1]
                    const float gx =
                        (at(1, -1) + 2.0f * at(1, 0) + at(1, 1)) -
                        (at(-1, -1) + 2.0f * at(-1, 0) + at(-1, 1));

                    // Sobel Y: [-1 -2 -1; 0 0 0; 1 2 1]
                    const float gy =
                        (at(-1, 1) + 2.0f * at(0, 1) + at(1, 1)) -
                        (at(-1, -1) + 2.0f * at(0, -1) + at(1, -1));

                    const std::size_t i = pixel_index(x, y, width);
                    ix[i] = gx;
                    iy[i] = gy;
                }
            }
        }

        // 1D normalized Gaussian weights of length 2*radius + 1.
        // Computed in double, then stored as float.
        std::vector<float> make_gaussian_weights(float sigma, int radius) {
            std::vector<float> weights(static_cast<std::size_t>(2 * radius + 1));
            double sum = 0.0;

            for (int i = -radius; i <= radius; i++) {
                const double d = static_cast<double>(i);
                const double w = std::exp(-(d * d) / (2.0 * sigma * sigma));
                weights[static_cast<std::size_t>(i + radius)] = static_cast<float>(w);
                sum += w;
            }

            for (float& w : weights) {
                w = static_cast<float>(static_cast<double>(w) / sum);
            }
            return weights;
        }

        // Separable Gaussian window sum: sums src over the window around each pixel.
        // This implements the sum over w(x, y) in M = sum w(x,y) * [Ix^2 IxIy; IxIy Iy^2].
        void gaussian_window_sum(const std::vector<float>& src,
                                 std::vector<float>& dst,
                                 int width,
                                 int height,
                                 const std::vector<float>& weights,
                                 int radius) {
            std::vector<float> tmp(src.size(), 0.0f);
            dst.assign(src.size(), 0.0f);

            // Horizontal pass
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    float sum = 0.0f;
                    for (int off = -radius; off <= radius; off++) {
                        const int sx = clamp_int(x + off, 0, width - 1);
                        sum += weights[static_cast<std::size_t>(off + radius)]
                             * src[pixel_index(sx, y, width)];
                    }
                    tmp[pixel_index(x, y, width)] = sum;
                }
            }

            // Vertical pass
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    float sum = 0.0f;
                    for (int off = -radius; off <= radius; off++) {
                        const int sy = clamp_int(y + off, 0, height - 1);
                        sum += weights[static_cast<std::size_t>(off + radius)]
                             * tmp[pixel_index(x, sy, width)];
                    }
                    dst[pixel_index(x, y, width)] = sum;
                }
            }
        }

        // Harris response R = det(M) - k * trace(M)^2 for every pixel.
        std::vector<float> harris_response(const ImageGrey& grey) {
            const int width  = grey.width;
            const int height = grey.height;
            const std::size_t count = grey.pixels.size();

            std::vector<float> ix, iy;
            compute_sobel_gradients(grey, ix, iy);

            // Products of gradients
            std::vector<float> ixx(count), iyy(count), ixy(count);
            for (std::size_t i = 0; i < count; i++) {
                ixx[i] = ix[i] * ix[i];
                iyy[i] = iy[i] * iy[i];
                ixy[i] = ix[i] * iy[i];
            }

            // Window sums give the entries of M at every pixel
            const std::vector<float> weights =
                make_gaussian_weights(kWindowSigma, kWindowRadius);

            std::vector<float> sxx, syy, sxy;
            gaussian_window_sum(ixx, sxx, width, height, weights, kWindowRadius);
            gaussian_window_sum(iyy, syy, width, height, weights, kWindowRadius);
            gaussian_window_sum(ixy, sxy, width, height, weights, kWindowRadius);

            // R = det(M) - k * trace(M)^2
            std::vector<float> response(count);
            for (std::size_t i = 0; i < count; i++) {
                const float det   = sxx[i] * syy[i] - sxy[i] * sxy[i];
                const float trace = sxx[i] + syy[i];
                response[i] = det - kHarrisK * trace * trace;
            }
            return response;
        }

    } // namespace

    std::vector<Keypoint> v0_detect_keypoints(const ImageGrey& grey) {
        const int width  = grey.width;
        const int height = grey.height;

        if (width <= 0 || height <= 0) {
            throw std::invalid_argument("Image dimensions must be positive.");
        }
        if (grey.pixels.size() != static_cast<std::size_t>(width) * static_cast<std::size_t>(height)) {
            throw std::invalid_argument("Grey buffer size does not match width * height.");
        }

        const int margin = kBorderMargin;
        if (width <= 2 * margin || height <= 2 * margin) {
            return {}; // Image too small to have any valid interior pixels
        }

        const std::vector<float> response = harris_response(grey);

        // Maximum response inside the valid interior (ignores the border margin)
        float max_response = 0.0f;
        for (int y = margin; y < height - margin; y++) {
            for (int x = margin; x < width - margin; x++) {
                max_response = std::max(max_response, response[pixel_index(x, y, width)]);
            }
        }
        const float threshold = kRelativeThreshold * max_response;

        // Threshold + non-maximum suppression (3x3)
        // Tie-break: a neighbour earlier in raster order must be strictly smaller,
        // a neighbour later in raster order may be equal. This keeps exactly one
        // pixel per plateau and makes the result deterministic.
        std::vector<Keypoint> candidates;
        for (int y = margin; y < height - margin; y++) {
            for (int x = margin; x < width - margin; x++) {
                const float r = response[pixel_index(x, y, width)];
                if (r <= 0.0f || r <= threshold) {
                    continue;
                }

                bool is_local_max = true;
                for (int dy = -1; dy <= 1 && is_local_max; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) {
                            continue;
                        }
                        const int nx = x + dx;
                        const int ny = y + dy;
                        const float nr = response[pixel_index(nx, ny, width)];
                        const bool earlier = (ny < y) || (ny == y && nx < x);

                        if (earlier ? (nr >= r) : (nr > r)) {
                            is_local_max = false;
                            break;
                        }
                    }
                }

                if (is_local_max) {
                    candidates.push_back(Keypoint{
                        static_cast<float>(x),
                        static_cast<float>(y),
                        r
                    });
                }
            }
        }

        // Top-K by response. Ties are broken by (y, x) so the order is deterministic.
        std::sort(candidates.begin(), candidates.end(),
                  [](const Keypoint& a, const Keypoint& b) {
                      if (a.response != b.response) {
                          return a.response > b.response;
                      }
                      if (a.y != b.y) {
                          return a.y < b.y;
                      }
                      return a.x < b.x;
                  });

        if (candidates.size() > kMaxKeypoints) {
            candidates.resize(kMaxKeypoints);
        }

        return candidates;
    }

}