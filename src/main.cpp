#include "image_io.h"
#include "v0_sequential.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace {
    struct Options {
        std::filesystem::path left;
        std::filesystem::path right;
        std::filesystem::path output;

        int version = 0;
        int runs = 5;
    };

    void print_usage(const char* program){
        std::cout
            << "Usage:\n  " << program
            << " <left_image> <right_image> <output.png>"
            << " --version 0 [--runs N]\n";
    }

    int parse_integer(std::string_view text, const char* option){
        int value = 0;

        const auto parsed = std::from_chars(
            text.data(),
            text.data() + text.size(),
            value
        );

        if (parsed.ec != std::errc{} ||
            parsed.ptr != text.data() + text.size()) {
            throw std::invalid_argument(
                std::string("Invalid integer for ") + option +
                ": " + std::string(text)
            );
        }

        return value;
    }

    Options parse_options(int argc, char* argv[]){
        if (argc < 6) {
            throw std::invalid_argument(
                "Expected three image paths and --version 0."
            );
        }

        Options options;
        options.left = std::filesystem::u8path(argv[1]);
        options.right = std::filesystem::u8path(argv[2]);
        options.output = std::filesystem::u8path(argv[3]);

        bool version_seen = false;
        bool runs_seen = false;

        for (int i = 4; i < argc; ++i) {
            const std::string_view option = argv[i];

            if (option != "--version" && option != "--runs") {
                throw std::invalid_argument(
                    "Unknown option: " + std::string(option)
                );
            }

            if (i + 1 >= argc) {
                throw std::invalid_argument(
                    "Missing value for " + std::string(option)
                );
            }

            const char* option_name = argv[i];
            ++i;
            const int value = parse_integer(argv[i], option_name);

            if (option == "--version") {
                if (version_seen) {
                    throw std::invalid_argument(
                        "--version was provided more than once."
                    );
                }

                options.version = value;
                version_seen = true;
            } else {
                if (runs_seen) {
                    throw std::invalid_argument(
                        "--runs was provided more than once."
                    );
                }

                options.runs = value;
                runs_seen = true;
            }
        }

        if (!version_seen || options.version != 0) {
            throw std::invalid_argument(
                "This executable currently supports --version 0 only."
            );
        }

        if (options.runs <= 0) {
            throw std::invalid_argument(
                "--runs must be a positive integer."
            );
        }

        if (options.output.extension() != ".png") {
            throw std::invalid_argument(
                "Output filename must end in .png."
            );
        }

        return options;
    }

}

int main(int argc, char* argv[]){

    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        print_usage(argv[0]);
        return 0;
    }

    try {
        const Options options = parse_options(argc, argv);

        // Disk I/O is outside all pipeline timings.
        const auto image_a =
            panorama::load_rgb_image(options.left);

        const auto image_b =
            panorama::load_rgb_image(options.right);

        std::cout
            << "Version: V0, single-threaded CPU\n"
            << "Image A: " << image_a.width
            << " x " << image_a.height << '\n'
            << "Image B: " << image_b.width
            << " x " << image_b.height << '\n'
            << "Timed runs: " << options.runs << '\n';

        if (options.runs < 5) {
            std::cout
                << "Debug run: use at least 5 timed runs "
                << "for reported benchmarks.\n";
        }

        // Warm-up result is discarded before measured runs.
        {
            const auto warmup =
                panorama::v0_run_pipeline(image_a, image_b);

            (void)warmup;
        }

        panorama::StageTimings accumulated{};
        double minimum_total_ms =
            std::numeric_limits<double>::infinity();

        panorama::V0Result final_result;

        for (int run = 0; run < options.runs; ++run) {
            auto result =
                panorama::v0_run_pipeline(image_a, image_b);

            for (std::size_t stage = 0; stage < 6; ++stage) {
                accumulated.stage_ms[stage] +=
                    result.timings.stage_ms[stage];
            }

            accumulated.total_ms += result.timings.total_ms;

            minimum_total_ms = std::min(
                minimum_total_ms,
                result.timings.total_ms
            );

            // Retain only the final panorama for saving.
            if (run == options.runs - 1) {
                final_result = std::move(result);
            }
        }

        const double count = static_cast<double>(options.runs);

        for (double& stage_ms : accumulated.stage_ms) {
            stage_ms /= count;
        }

        accumulated.total_ms /= count;

        // Save once, after all measured runs finish.
        panorama::save_rgb_png(
            options.output,
            final_result.panorama
        );

        constexpr std::array<const char*, 6> stage_names{
            "Preprocessing",
            "Keypoint detection",
            "Descriptor extraction",
            "Matching",
            "Homography estimation",
            "Warping and blending"
        };

        std::cout << std::fixed << std::setprecision(3);

        for (std::size_t stage = 0; stage < 6; ++stage) {
            std::cout
                << "Stage " << stage + 1
                << " (" << stage_names[stage] << "): "
                << accumulated.stage_ms[stage]
                << " ms average\n";
        }

        std::cout
            << "Pipeline total: "
            << accumulated.total_ms << " ms average\n"
            << "Pipeline minimum: "
            << minimum_total_ms << " ms\n"
            << "Keypoints A: " << final_result.keypoints_a << '\n'
            << "Keypoints B: " << final_result.keypoints_b << '\n'
            << "Matches: " << final_result.match_count << '\n'
            << "Inliers: "
            << final_result.homography.inlier_count << '\n'
            << "Panorama: "
            << final_result.panorama.width << " x "
            << final_result.panorama.height << '\n'
            << "Saved: " << options.output.u8string() << '\n';

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}