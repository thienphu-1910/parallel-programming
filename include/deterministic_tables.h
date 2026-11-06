#ifndef DETERMINISTIC_TABLES_H
#define DETERMINISTIC_TABLES_H

namespace panorama {
    inline constexpr int kGaussianRadius = 2;

    // Normalise five-tap Gaussian, sigma = 1.0
    inline constexpr std::array<float, 5> kGaussianWeights{
        0.054488685f,
        0.244201342f,
        0.402619947f,
        0.244201342f,
        0.054488685f
    };
    
}


#endif // DETERMINISTIC_TABLES_H