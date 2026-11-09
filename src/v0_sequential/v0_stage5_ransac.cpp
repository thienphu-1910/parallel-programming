#include "v0_sequential.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace panorama {
    namespace {
        // ---- RANSAC parameters. Keep identical in every version (V0..Vn). ----
        constexpr int           kRansacIterations   = 2000;
        constexpr double        kInlierThreshold    = 3.0;   // pixels (reprojection error in A)
        constexpr std::size_t   kMinInliers         = 8;
        constexpr std::uint32_t kRansacSeed         = 0x9E3779B9u;

        using Mat3 = std::array<double, 9>;

        struct Point {
            double x = 0.0;
            double y = 0.0;
        };

        struct Correspondence {
            Point src; // point in B
            Point dst; // point in A
        };

        // ---- Adapter: change here if Match field names differ ----
        inline std::size_t match_a(const Match& m) { return m.index_a; }
        inline std::size_t match_b(const Match& m) { return m.index_b; }

        // Deterministic xorshift32 PRNG (same sequence on every platform)
        struct XorShift32 {
            std::uint32_t state;

            std::uint32_t next() {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                return state;
            }

            // Value in [0, bound). Modulo bias is irrelevant here; determinism matters.
            std::size_t below(std::size_t bound) {
                return static_cast<std::size_t>(next()) % bound;
            }
        };

        // Apply H to a point. Returns false if the point maps to infinity.
        bool project(const Mat3& h, const Point& p, Point& out) {
            const double w = h[6] * p.x + h[7] * p.y + h[8];
            if (std::fabs(w) < 1e-12) {
                return false;
            }
            out.x = (h[0] * p.x + h[1] * p.y + h[2]) / w;
            out.y = (h[3] * p.x + h[4] * p.y + h[5]) / w;
            return true;
        }

        Mat3 multiply(const Mat3& a, const Mat3& b) {
            Mat3 r{};
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    double s = 0.0;
                    for (int k = 0; k < 3; k++) {
                        s += a[static_cast<std::size_t>(i * 3 + k)]
                           * b[static_cast<std::size_t>(k * 3 + j)];
                    }
                    r[static_cast<std::size_t>(i * 3 + j)] = s;
                }
            }
            return r;
        }

        // Hartley normalization: centroid at origin, mean distance sqrt(2).
        // Returns T (point -> normalized) and Tinv.
        bool make_normalization(const std::vector<Point>& pts, Mat3& t, Mat3& t_inv) {
            double cx = 0.0, cy = 0.0;
            for (const Point& p : pts) {
                cx += p.x;
                cy += p.y;
            }
            const double n = static_cast<double>(pts.size());
            cx /= n;
            cy /= n;

            double mean_dist = 0.0;
            for (const Point& p : pts) {
                mean_dist += std::sqrt((p.x - cx) * (p.x - cx) + (p.y - cy) * (p.y - cy));
            }
            mean_dist /= n;
            if (mean_dist < 1e-9) {
                return false; // all points coincide
            }

            const double s = std::sqrt(2.0) / mean_dist;
            t     = { s, 0.0, -s * cx,
                      0.0, s, -s * cy,
                      0.0, 0.0, 1.0 };
            t_inv = { 1.0 / s, 0.0, cx,
                      0.0, 1.0 / s, cy,
                      0.0, 0.0, 1.0 };
            return true;
        }

        // Solve an 8x8 system (augmented 8x9) with partial pivoting.
        bool solve_8x8(std::array<std::array<double, 9>, 8>& m, std::array<double, 8>& x) {
            for (int col = 0; col < 8; col++) {
                int pivot = col;
                double best = std::fabs(m[static_cast<std::size_t>(col)][static_cast<std::size_t>(col)]);
                for (int r = col + 1; r < 8; r++) {
                    const double v = std::fabs(m[static_cast<std::size_t>(r)][static_cast<std::size_t>(col)]);
                    if (v > best) {
                        best = v;
                        pivot = r;
                    }
                }
                if (best < 1e-12) {
                    return false; // singular / degenerate configuration
                }
                std::swap(m[static_cast<std::size_t>(col)], m[static_cast<std::size_t>(pivot)]);

                for (int r = col + 1; r < 8; r++) {
                    auto& row = m[static_cast<std::size_t>(r)];
                    const auto& prow = m[static_cast<std::size_t>(col)];
                    const double f = row[static_cast<std::size_t>(col)] / prow[static_cast<std::size_t>(col)];
                    for (int c = col; c < 9; c++) {
                        row[static_cast<std::size_t>(c)] -= f * prow[static_cast<std::size_t>(c)];
                    }
                }
            }

            for (int r = 7; r >= 0; r--) {
                double s = m[static_cast<std::size_t>(r)][8];
                for (int c = r + 1; c < 8; c++) {
                    s -= m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] * x[static_cast<std::size_t>(c)];
                }
                x[static_cast<std::size_t>(r)] = s / m[static_cast<std::size_t>(r)][static_cast<std::size_t>(r)];
            }
            return true;
        }

        // Least-squares homography (h33 = 1) from >= 4 correspondences.
        // With exactly 4 points this is the exact solution.
        bool fit_homography(const std::vector<Correspondence>& data,
                            const std::vector<std::size_t>& indices,
                            Mat3& result) {
            if (indices.size() < 4) {
                return false;
            }

            std::vector<Point> src, dst;
            src.reserve(indices.size());
            dst.reserve(indices.size());
            for (std::size_t idx : indices) {
                src.push_back(data[idx].src);
                dst.push_back(data[idx].dst);
            }

            Mat3 ts{}, ts_inv{}, td{}, td_inv{};
            if (!make_normalization(src, ts, ts_inv) || !make_normalization(dst, td, td_inv)) {
                return false;
            }

            // Normal equations: (A^T A) h = A^T b
            std::array<std::array<double, 9>, 8> ata{};
            for (auto& row : ata) {
                row.fill(0.0);
            }

            for (std::size_t k = 0; k < src.size(); k++) {
                const double x = ts[0] * src[k].x + ts[1] * src[k].y + ts[2];
                const double y = ts[3] * src[k].x + ts[4] * src[k].y + ts[5];
                const double u = td[0] * dst[k].x + td[1] * dst[k].y + td[2];
                const double v = td[3] * dst[k].x + td[4] * dst[k].y + td[5];

                const double rows[2][8] = {
                    { x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y },
                    { 0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y }
                };
                const double rhs[2] = { u, v };

                for (int r = 0; r < 2; r++) {
                    for (int i = 0; i < 8; i++) {
                        for (int j = 0; j < 8; j++) {
                            ata[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] += rows[r][i] * rows[r][j];
                        }
                        ata[static_cast<std::size_t>(i)][8] += rows[r][i] * rhs[r];
                    }
                }
            }

            std::array<double, 8> sol{};
            if (!solve_8x8(ata, sol)) {
                return false;
            }

            const Mat3 hn = { sol[0], sol[1], sol[2],
                              sol[3], sol[4], sol[5],
                              sol[6], sol[7], 1.0 };

            // Denormalize: H = Td^-1 * Hn * Ts
            Mat3 h = multiply(td_inv, multiply(hn, ts));

            const double scale = h[8];
            if (std::fabs(scale) < 1e-12) {
                return false;
            }
            for (double& v : h) {
                v /= scale;
            }
            result = h;
            return true;
        }

        // Squared reprojection error of a correspondence under H (in A's frame).
        // Returns a huge value if the projection is invalid.
        double reprojection_error_sq(const Mat3& h, const Correspondence& c) {
            Point p;
            if (!project(h, c.src, p)) {
                return 1e300;
            }
            const double dx = p.x - c.dst.x;
            const double dy = p.y - c.dst.y;
            return dx * dx + dy * dy;
        }

        std::vector<std::size_t> collect_inliers(const Mat3& h, const std::vector<Correspondence>& data) {
            std::vector<std::size_t> inliers;
            const double limit = kInlierThreshold * kInlierThreshold;
            for (std::size_t i = 0; i < data.size(); i++) {
                if (reprojection_error_sq(h, data[i]) <= limit) {
                    inliers.push_back(i);
                }
            }
            return inliers;
        }
    } // namespace

    Homography v0_estimate_homography(const std::vector<Keypoint>& keypoints_a,
                                      const std::vector<Keypoint>& keypoints_b,
                                      const std::vector<Match>& matches) {
        Homography output;

        // Build correspondences, validating indices
        std::vector<Correspondence> data;
        data.reserve(matches.size());
        for (const Match& m : matches) {
            const std::size_t a = match_a(m);
            const std::size_t b = match_b(m);
            if (a >= keypoints_a.size() || b >= keypoints_b.size()) {
                throw std::invalid_argument("Stage 5: match index out of range.");
            }
            Correspondence c;
            c.src = { static_cast<double>(keypoints_b[b].x), static_cast<double>(keypoints_b[b].y) };
            c.dst = { static_cast<double>(keypoints_a[a].x), static_cast<double>(keypoints_a[a].y) };
            data.push_back(c);
        }

        if (data.size() < 4) {
            return output; // not enough matches; valid = false
        }

        XorShift32 rng{ kRansacSeed };

        std::vector<std::size_t> best_inliers;

        for (int iter = 0; iter < kRansacIterations; iter++) {
            // Sample 4 distinct indices (deterministic)
            std::vector<std::size_t> sample;
            sample.reserve(4);
            while (sample.size() < 4) {
                const std::size_t candidate = rng.below(data.size());
                bool duplicate = false;
                for (std::size_t s : sample) {
                    if (s == candidate) {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate) {
                    sample.push_back(candidate);
                }
            }

            Mat3 h{};
            if (!fit_homography(data, sample, h)) {
                continue;
            }

            std::vector<std::size_t> inliers = collect_inliers(h, data);

            // Strictly greater: the earliest best model wins ties -> deterministic
            if (inliers.size() > best_inliers.size()) {
                best_inliers = std::move(inliers);
            }
        }

        if (best_inliers.size() < kMinInliers) {
            return output;
        }

        // Refit on all inliers, then recompute the final inlier set
        Mat3 refined{};
        if (!fit_homography(data, best_inliers, refined)) {
            return output;
        }

        const std::vector<std::size_t> final_inliers = collect_inliers(refined, data);
        if (final_inliers.size() < kMinInliers) {
            return output;
        }

        output.h = refined;
        output.inlier_count = final_inliers.size();
        output.valid = true;
        return output;
    }
}

