#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include "common.h"

#include <filesystem>

namespace panorama {

    ImageRGB load_rgb_image(const std::filesystem::path& path);

    void save_rgb_png(const std::filesystem::path& path, const ImageRGB& image);
}


#endif // IMAGE_IO_H