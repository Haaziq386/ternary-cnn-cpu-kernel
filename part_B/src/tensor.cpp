#include "tensor.h"

namespace ternary
{

    Tensor::Tensor(int batch, int channels, int height, int width)
    {
        resize(batch, channels, height, width);
    }

    void Tensor::resize(int batch, int channels, int height, int width)
    {
        n = batch;
        c = channels;
        h = height;
        w = width;
        data.resize(static_cast<std::size_t>(batch) * channels * height * width);
    }

    void Tensor::reserve(std::size_t elements)
    {
        data.reserve(elements);
    }

    std::size_t Tensor::size() const
    {
        return data.size();
    }

    float *Tensor::ptr()
    {
        return data.data();
    }

    const float *Tensor::ptr() const
    {
        return data.data();
    }

} // namespace ternary