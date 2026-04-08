#pragma once

#include <cstddef>
#include <vector>

namespace ternary
{

    struct Tensor
    {
        int n = 0;
        int c = 0;
        int h = 0;
        int w = 0;
        std::vector<float> data;

        Tensor() = default;
        Tensor(int batch, int channels, int height, int width);

        void resize(int batch, int channels, int height, int width);
        void reserve(std::size_t elements);
        std::size_t size() const;
        float *ptr();
        const float *ptr() const;
    };

} // namespace ternary