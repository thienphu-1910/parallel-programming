#include "v0_sequential.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace panorama {
    namespace {
        using Mat3 = std::array<double, 9>;

        // Safety cap on canvas size (a bad homography can explode the bounds)
        constexpr long long kMaxCanvasSide   = 16384;
        constexpr long long kMaxCanvasPixels = 100000000LL;

        // Pixel-coordinate tolerance for floating-point roundoff.
        // This is not the RANSAC inlier threshold.
        constexpr double kPixelTolerance = 1e-7;

        double snap_near_integer(double value)
        {
            const double rounded = std::round(value);

            return std::fabs(value - rounded) <= kPixelTolerance
                ? rounded
                : value;
        }

        bool inside_sample_domain(
            double& x,
            double& y,
            int width,
            int height)
        {
            const double max_x = static_cast<double>(width - 1);
            const double max_y = static_cast<double>(height - 1);

            if (!std::isfinite(x) || !std::isfinite(y) ||
                x < -kPixelTolerance ||
                y < -kPixelTolerance ||
                x > max_x + kPixelTolerance ||
                y > max_y + kPixelTolerance) {
                return false;
            }

            x = std::clamp(x, 0.0, max_x);
            y = std::clamp(y, 0.0, max_y);

            return true;
        }

        void validate_rgb(const ImageRGB& image, const char* name) {
            if (image.width <= 0 || image.height <= 0) {
                throw std::invalid_argument(std::string("Stage 6: ") + name + " dimensions must be positive.");
            }
            const std::size_t expected =
                static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.height) * 3;
            if (image.pixels.size() != expected) {
                throw std::invalid_argument(std::string("Stage 6: ") + name + " buffer size is invalid.");
            }
        }

        bool invert(const Mat3& m, Mat3& out) {
            const double a = m[0], b = m[1], c = m[2];
            const double d = m[3], e = m[4], f = m[5];
            const double g = m[6], h = m[7], i = m[8];

            const double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
            if (std::fabs(det) < 1e-12) {
                return false;
            }
            const double inv = 1.0 / det;

            out = { (e * i - f * h) * inv, (c * h - b * i) * inv, (b * f - c * e) * inv,
                    (f * g - d * i) * inv, (a * i - c * g) * inv, (c * d - a * f) * inv,
                    (d * h - e * g) * inv, (b * g - a * h) * inv, (a * e - b * d) * inv };
            return true;
        }

        bool project(const Mat3& h, double x, double y, double& ox, double& oy){
            const double w = h[6] * x + h[7] * y + h[8];

            if (!std::isfinite(w) || std::fabs(w) < 1e-12) {
                return false;
            }

            ox = (h[0] * x + h[1] * y + h[2]) / w;
            oy = (h[3] * x + h[4] * y + h[5]) / w;

            return std::isfinite(ox) && std::isfinite(oy);
        }

        // Feather weight: distance to the nearest image edge (>= 1 inside the image).
        float edge_weight(double x, double y, int width, int height) {
            const double dx = std::min(x + 1.0, static_cast<double>(width) - x);
            const double dy = std::min(y + 1.0, static_cast<double>(height) - y);
            return static_cast<float>(std::max(0.0, std::min(dx, dy)));
        }

        // Bilinear sample of one RGB image at (x, y). Caller guarantees the point is inside.
        std::array<float, 3> sample_bilinear(const ImageRGB& image, double x, double y) {
            const int x0 = static_cast<int>(std::floor(x));
            const int y0 = static_cast<int>(std::floor(y));
            const int x1 = std::min(x0 + 1, image.width - 1);
            const int y1 = std::min(y0 + 1, image.height - 1);

            const float fx = static_cast<float>(x - x0);
            const float fy = static_cast<float>(y - y0);

            const std::size_t stride = static_cast<std::size_t>(image.width);
            auto px = [&](int xx, int yy, std::size_t ch) -> float {
                return static_cast<float>(
                    image.pixels[(static_cast<std::size_t>(yy) * stride + static_cast<std::size_t>(xx)) * 3 + ch]);
            };

            std::array<float, 3> rgb{};
            for (std::size_t ch = 0; ch < 3; ch++) {
                const float top    = px(x0, y0, ch) * (1.0f - fx) + px(x1, y0, ch) * fx;
                const float bottom = px(x0, y1, ch) * (1.0f - fx) + px(x1, y1, ch) * fx;
                rgb[ch] = top * (1.0f - fy) + bottom * fy;
            }
            return rgb;
        }

        std::uint8_t to_byte(float v) {
            const float r = std::floor(v + 0.5f);
            return static_cast<std::uint8_t>(std::clamp(r, 0.0f, 255.0f));
        }
    } // namespace

    ImageRGB v0_stitch(const ImageRGB& image_a,
                       const ImageRGB& image_b,
                       const Homography& homography) {
        validate_rgb(image_a, "image A");
        validate_rgb(image_b, "image B");

        if (!homography.valid) {
            throw std::invalid_argument("Stage 6: homography is not valid.");
        }

        const Mat3 h_b_to_a = homography.h;

        for (double value : h_b_to_a) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "Stage 6: nonfinite homography."
                );
            }
        }

        Mat3 h_a_to_b{};

        if (!invert(h_b_to_a, h_a_to_b)) {
            throw std::invalid_argument(
                "Stage 6: homography is not invertible."
            );
        }

        for (double value : h_a_to_b) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "Stage 6: nonfinite inverse homography."
                );
            }
        }

        // Use integer pixel centers consistently.
        // A occupies x = 0 ... width-1, y = 0 ... height-1.
        double min_x = 0.0;
        double min_y = 0.0;
        double max_x = static_cast<double>(image_a.width - 1);
        double max_y = static_cast<double>(image_a.height - 1);

        const double last_x = static_cast<double>(image_b.width - 1);
        const double last_y = static_cast<double>(image_b.height - 1);

        const double corners[4][2] = {
            {0.0,    0.0},
            {last_x, 0.0},
            {0.0,    last_y},
            {last_x, last_y}
        };

        // Reject a projective denominator that reaches or crosses zero
        // anywhere within B's rectangular sampling domain.
        bool positive = false;
        bool negative = false;

        for (const auto& corner : corners) {
            const double w =
                h_b_to_a[6] * corner[0] +
                h_b_to_a[7] * corner[1] +
                h_b_to_a[8];

            if (!std::isfinite(w) || std::fabs(w) < 1e-12) {
                throw std::invalid_argument(
                    "Stage 6: projection is singular at an image corner."
                );
            }

            positive = positive || w > 0.0;
            negative = negative || w < 0.0;
        }

        if (positive && negative) {
            throw std::invalid_argument(
                "Stage 6: projection crosses infinity inside image B."
            );
        }

        // The projected rectangle is bounded after the denominator check.
        for (const auto& corner : corners) {
            double ox = 0.0;
            double oy = 0.0;

            if (!project(h_b_to_a, corner[0], corner[1], ox, oy)) {
                throw std::invalid_argument(
                    "Stage 6: invalid projected corner."
                );
            }

            ox = snap_near_integer(ox);
            oy = snap_near_integer(oy);

            min_x = std::min(min_x, ox);
            min_y = std::min(min_y, oy);
            max_x = std::max(max_x, ox);
            max_y = std::max(max_y, oy);
        }

        // Compute bounds in floating point before converting to integers.
        const double left = std::floor(min_x);
        const double top = std::floor(min_y);

        const double width = std::ceil(max_x) - left + 1.0;
        const double height = std::ceil(max_y) - top + 1.0;

        if (!std::isfinite(left) ||
            !std::isfinite(top) ||
            !std::isfinite(width) ||
            !std::isfinite(height) ||
            width < 1.0 ||
            height < 1.0 ||
            width > static_cast<double>(kMaxCanvasSide) ||
            height > static_cast<double>(kMaxCanvasSide) ||
            width * height > static_cast<double>(kMaxCanvasPixels)) {
            throw std::overflow_error(
                "Stage 6: panorama canvas is too large."
            );
        }

        // A's origin is included in the union, so the dimension limits
        // also bound the magnitude of the negative canvas origin.
        const long long origin_x = static_cast<long long>(left);
        const long long origin_y = static_cast<long long>(top);
        const long long out_w = static_cast<long long>(width);
        const long long out_h = static_cast<long long>(height);

        ImageRGB canvas;
        canvas.width  = static_cast<int>(out_w);
        canvas.height = static_cast<int>(out_h);
        canvas.pixels.assign(static_cast<std::size_t>(out_w) * static_cast<std::size_t>(out_h) * 3, 0);

        for (long long cy = 0; cy < out_h; cy++) {
            for (long long cx = 0; cx < out_w; cx++) {
                // Position in A's coordinate frame (pixel centres at integer coords)
                const double ax = static_cast<double>(cx + origin_x);
                const double ay = static_cast<double>(cy + origin_y);

                float weight_sum = 0.0f;
                std::array<float, 3> color_sum{ 0.0f, 0.0f, 0.0f };

                // Contribution from A
                if (ax >= 0.0 && ay >= 0.0 &&
                    ax <= static_cast<double>(image_a.width - 1) &&
                    ay <= static_cast<double>(image_a.height - 1)) {
                    const float w = edge_weight(ax, ay, image_a.width, image_a.height);
                    const auto rgb = sample_bilinear(image_a, ax, ay);
                    for (std::size_t ch = 0; ch < 3; ch++) {
                        color_sum[ch] += w * rgb[ch];
                    }
                    weight_sum += w;
                }

                // Contribution from B (inverse mapping: canvas -> B)
                double bx = 0.0, by = 0.0;
                if (project(h_a_to_b, ax, ay, bx, by) && 
                    inside_sample_domain(bx, by, image_b.width, image_b.height)) {
                    const float w = edge_weight(bx, by, image_b.width, image_b.height);
                    const auto rgb = sample_bilinear(image_b, bx, by);
                    for (std::size_t ch = 0; ch < 3; ch++) {
                        color_sum[ch] += w * rgb[ch];
                    }
                    weight_sum += w;
                }

                if (weight_sum > 0.0f) {
                    const std::size_t base =
                        (static_cast<std::size_t>(cy) * static_cast<std::size_t>(out_w) +
                         static_cast<std::size_t>(cx)) * 3;
                    for (std::size_t ch = 0; ch < 3; ch++) {
                        canvas.pixels[base + ch] = to_byte(color_sum[ch] / weight_sum);
                    }
                }
            }
        }

        return canvas;
    }
}
