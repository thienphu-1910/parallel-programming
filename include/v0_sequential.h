#ifndef V0_SEQUENTIAL_H
#define V0_SEQUENTIAL_H

#include "common.h"

namespace panorama {

    // Returns Gaussian-smoothed greyscale
    ImageGray v0_preprocess(const ImageRGB& input);
    
}


#endif // V0_SEQUENTIAL_H