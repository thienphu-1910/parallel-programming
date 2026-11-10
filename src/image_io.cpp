#include "image_io.h"

#include <cstddef>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

// Interpret filenames as UTF-8 on Windows.
#ifdef _WIN32
#define STBI_WINDOWS_UTF8
#define STBIW_WINDOWS_UTF8
#endif

// Required Definition
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace panorama {
    namespace {
        constexpr std::size_t kMaxIoPixels = 100'000'000; 

        std::size_t checked_rgb_size(int width, int height){
            if (width <= 0 || height <= 0){
                throw std::invalid_argument(
                    "Image I/O: Dimensions must be positive"
                );
            }

            const auto w = static_cast<std::size_t>(width);
            const auto h = static_cast<std::size_t>(height);
            const auto limit = std::numeric_limits<std::size_t>::max();

            if (w > limit / h){
                throw std::overflow_error(
                    "Image I/O: Pixel count overflows"
                );
            }

            const std::size_t count = w * h;

            if (count > kMaxIoPixels || count > limit / 3){
                throw std::invalid_argument(
                    "Image I/O: Image is too large"
                );
            }

            return count * 3;
        }
            
        using StbiPixels = std::unique_ptr<stbi_uc, decltype(&stbi_image_free)>;
    }

    ImageRGB load_rgb_image(const std::filesystem::path& path){
        if (path.empty()){
            throw std::invalid_argument(
                "Image I/O: input path is empty"
            );
        }

        const std::string filename = path.u8string();

        int width = 0;
        int height = 0;
        int original_channels = 0;

        StbiPixels decoded(
            stbi_load(
                filename.c_str(),
                &width,
                &height,
                &original_channels,
                STBI_rgb
            ),
            &stbi_image_free
        );

        if (!decoded){
            const char* reason = stbi_failure_reason();

            throw std::runtime_error (
                "Cannot load image '" + filename + "': " + 
                (reason != nullptr ? reason : "unknown decoding error")
            );
        }

        const std::size_t byte_count = checked_rgb_size(width, height);

        ImageRGB image;
        image.width = width;
        image.height = height;

        image.pixels.assign(
            decoded.get(),
            decoded.get() + byte_count
        );

        return image;
    }

    void save_rgb_png(const std::filesystem::path& path, const ImageRGB& image){
        if (path.empty()){
            throw std::invalid_argument(
                "Image I/O: output path is empty"
            );
        }

        const std::size_t expected = checked_rgb_size(image.width, image.height);

        if (image.pixels.size() != expected){
            throw std::invalid_argument(
                "Image I/O: RGB buffer size does not match dimension"
            );
        }

        // stb's PNG interface takes the row stride as an int.
        if (image.width > std::numeric_limits<int>::max() / 3) {
            throw std::overflow_error(
                "Image I/O: row stride overflows."
            );
        }

        const std::filesystem::path parent = path.parent_path();

        if (!parent.empty()){
            std::filesystem::create_directories(parent);
        }

        const std::string filename = path.u8string();
        const int stride_bytes = image.width * 3;

        const int success = stbi_write_png(
            filename.c_str(),
            image.width,
            image.height,
            3,
            image.pixels.data(),
            stride_bytes
        );

        if (success == 0){
            throw std::runtime_error(
                "Cannot save PNG image '" + filename + "'"
            );
        }
    }
}
