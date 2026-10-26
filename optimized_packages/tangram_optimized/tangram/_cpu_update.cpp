#include <ATen/cpu/vec/vec.h>
#include <cmath>
#include <cstdint>

extern "C" void tangram_cpu_moments(
    float* mean, float* square, const float* gradient, int64_t count, int threads) {
    using Vec = at::vec::Vectorized<float>;
    constexpr int64_t width = Vec::size();
    const int64_t blocks = count / width;
    const Vec beta2(.999f), weight(.1f), complement(.001f);
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t block = 0; block < blocks; ++block) {
        const int64_t offset = block * width;
        const auto m = Vec::loadu(mean + offset);
        const auto v = Vec::loadu(square + offset);
        const auto g = Vec::loadu(gradient + offset);
        at::vec::fmadd(weight, g - m, m).store(mean + offset);
        const auto decayed = v * beta2;
        at::vec::fmadd(complement * g, g, decayed).store(square + offset);
    }
    for (int64_t offset = blocks * width; offset < count; ++offset) {
        const float g = gradient[offset], m = mean[offset];
        mean[offset] = std::fma(.1f, g - m, m);
        const float decayed = square[offset] * .999f;
        square[offset] = std::fma(.001f * g, g, decayed);
    }
}

// Deliberately separate arithmetic operations: no fastmath or FMA contraction.
// The only explicit contractions above are those of the native moment kernels.
extern "C" void tangram_cpu_update(
    float* parameter, const float* mean, const float* square_root,
    int64_t count, float correction, float value, int threads) {
    using Vec = at::vec::Vectorized<float>;
    constexpr int64_t width = Vec::size();
    const int64_t blocks = count / width;
    const Vec c(correction), epsilon(1e-8f), scale(value);
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t block = 0; block < blocks; ++block) {
        const int64_t offset = block * width;
        const auto p = Vec::loadu(parameter + offset);
        const auto m = Vec::loadu(mean + offset);
        const auto root = Vec::loadu(square_root + offset);
        const auto denominator = root / c + epsilon;
        const auto updated = p + (scale * m) / denominator;
        updated.store(parameter + offset);
    }
    for (int64_t offset = blocks * width; offset < count; ++offset) {
        const float denominator = square_root[offset] / correction + 1e-8f;
        parameter[offset] = parameter[offset] + (value * mean[offset]) / denominator;
    }
}
