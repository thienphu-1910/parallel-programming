// Untimed V0 diagnostics. Reuse the production stages; do not replace the CLI.
#include "image_io.h"
#include "v0_sequential.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[]) {
    try {
        if (argc != 4) {
            throw std::invalid_argument("Expected left, right, inlier threshold.");
        }
        const double threshold = std::stod(argv[3]);
        if (!std::isfinite(threshold) || threshold <= 0.0) {
            throw std::invalid_argument("Invalid inlier threshold.");
        }
        const auto a = panorama::load_rgb_image(std::filesystem::u8path(argv[1]));
        const auto b = panorama::load_rgb_image(std::filesystem::u8path(argv[2]));
        const auto ga = panorama::v0_preprocess(a);
        const auto gb = panorama::v0_preprocess(b);
        const auto ka = panorama::v0_detect_keypoints(ga);
        const auto kb = panorama::v0_detect_keypoints(gb);
        const auto da = panorama::v0_compute_descriptors(ga, ka);
        const auto db = panorama::v0_compute_descriptors(gb, kb);
        const auto matches = panorama::v0_match_descriptors(da, db);
        const auto model = panorama::v0_estimate_homography(ka, kb, matches);

        std::size_t count = 0;
        double squared_sum = 0.0;
        double maximum = 0.0;
        const auto& h = model.h;
        if (model.valid) {
            for (double value : h) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error("Non-finite homography.");
                }
            }
            for (const auto& match : matches) {
                const auto& src = kb.at(match.index_b);
                const auto& dst = ka.at(match.index_a);
                const double w = h[6] * src.x + h[7] * src.y + h[8];
                if (!std::isfinite(w) || std::fabs(w) < 1e-12) continue;
                const double x = (h[0] * src.x + h[1] * src.y + h[2]) / w;
                const double y = (h[3] * src.x + h[4] * src.y + h[5]) / w;
                const double dx = x - dst.x;
                const double dy = y - dst.y;
                const double e2 = dx * dx + dy * dy;
                if (std::isfinite(e2) && e2 <= threshold * threshold) {
                    ++count;
                    squared_sum += e2;
                    if (e2 > maximum) maximum = e2;
                }
            }
            if (count == 0 || count != model.inlier_count) {
                throw std::runtime_error("Diagnostic and RANSAC inlier counts differ.");
            }
        }

        std::cout << std::setprecision(17)
                  << "{\"homography_valid\":" << (model.valid ? "true" : "false")
                  << ",\"keypoints_a\":" << ka.size()
                  << ",\"keypoints_b\":" << kb.size()
                  << ",\"matches\":" << matches.size()
                  << ",\"inliers\":" << count
                  << ",\"inlier_threshold_px\":" << threshold
                  << ",\"inlier_ratio\":";
        if (matches.empty()) std::cout << "null";
        else std::cout << static_cast<double>(count) / matches.size();
        std::cout << ",\"inlier_rmse_px\":";
        if (count == 0) std::cout << "null";
        else std::cout << std::sqrt(squared_sum / count);
        std::cout << ",\"inlier_max_error_px\":";
        if (count == 0) std::cout << "null";
        else std::cout << std::sqrt(maximum);
        std::cout << ",\"homography_b_to_a\":";
        if (!model.valid) std::cout << "null";
        else {
            std::cout << '[';
            for (std::size_t i = 0; i < h.size(); ++i) {
                if (i != 0) std::cout << ',';
                std::cout << h[i];
            }
            std::cout << ']';
        }
        std::cout << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
