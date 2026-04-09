#include "layers.h"

#include <algorithm>
#include <cassert>
#include <immintrin.h>

#include "ternary_kernel.h"

namespace ternary
{
    namespace
    {

        int round_up(int value, int multiple)
        {
            return ((value + multiple - 1) / multiple) * multiple;
        }

        void im2col(const Tensor &input, int kernel_h, int kernel_w, int stride_h, int stride_w,
                    int padding_h, int padding_w, int out_h, int out_w, int k_pad,
                    std::vector<float> &im2col_buffer)
        {
            const int batch = input.n;
            const int channels = input.c;
            const int input_h = input.h;
            const int input_w = input.w;
            const int kernel_elements = channels * kernel_h * kernel_w;
            const std::size_t required = static_cast<std::size_t>(batch) * out_h * out_w * k_pad;
            if (im2col_buffer.size() < required)
            {
                im2col_buffer.resize(required);
            }
            std::fill(im2col_buffer.begin(), im2col_buffer.begin() + required, 0.0f);

            for (int n = 0; n < batch; ++n)
            {
                const float *input_base = input.ptr() + static_cast<std::size_t>(n) * channels * input_h * input_w;
                for (int oy = 0; oy < out_h; ++oy)
                {
                    for (int ox = 0; ox < out_w; ++ox)
                    {
                        float *dst = im2col_buffer.data() + ((static_cast<std::size_t>(n) * out_h + oy) * out_w + ox) * k_pad;
                        int col_index = 0;
                        for (int channel = 0; channel < channels; ++channel)
                        {
                            for (int ky = 0; ky < kernel_h; ++ky)
                            {
                                const int in_y = oy * stride_h + ky - padding_h;
                                for (int kx = 0; kx < kernel_w; ++kx)
                                {
                                    const int in_x = ox * stride_w + kx - padding_w;
                                    if (in_y >= 0 && in_y < input_h && in_x >= 0 && in_x < input_w)
                                    {
                                        const std::size_t input_index = static_cast<std::size_t>(channel) * input_h * input_w + in_y * input_w + in_x;
                                        dst[col_index] = input_base[input_index];
                                    }
                                    ++col_index;
                                }
                            }
                        }
                        for (int i = kernel_elements; i < k_pad; ++i)
                        {
                            dst[i] = 0.0f;
                        }
                    }
                }
            }
        }

    } // namespace

    void conv_fp32(const Tensor &input, const Conv2DWeightsFP32 &weights, Tensor &output,
                   std::vector<float> &im2col_buffer)
    {
        output.resize(input.n, weights.out_channels, weights.output_h, weights.output_w);
        const int kernel_elements = weights.in_channels * weights.kernel_h * weights.kernel_w;
        const int k_pad = round_up(kernel_elements, 32);
        im2col(input, weights.kernel_h, weights.kernel_w, weights.stride_h, weights.stride_w,
               weights.padding_h, weights.padding_w, weights.output_h, weights.output_w, k_pad,
               im2col_buffer);

        const int output_spatial = weights.output_h * weights.output_w;
        const float *col_base = im2col_buffer.data();
        float *out_base = output.ptr();
        for (int n = 0; n < input.n; ++n)
        {
            const float *sample_cols = col_base + static_cast<std::size_t>(n) * output_spatial * k_pad;
            float *sample_out = out_base + static_cast<std::size_t>(n) * weights.out_channels * output_spatial;
            for (int oc = 0; oc < weights.out_channels; ++oc)
            {
                const float *weight_row = weights.weight.data() + static_cast<std::size_t>(oc) * kernel_elements;
                for (int spatial = 0; spatial < output_spatial; ++spatial)
                {
                    const float *activation_row = sample_cols + static_cast<std::size_t>(spatial) * k_pad;
                    float value = dot_product_fp32_avx2(weight_row, activation_row, kernel_elements);
                    if (weights.has_bias)
                    {
                        value += weights.bias[oc];
                    }
                    sample_out[static_cast<std::size_t>(oc) * output_spatial + spatial] = value;
                }
            }
        }
    }

    void conv_ternary(const Tensor &input, const TernaryConv2DWeights &weights, Tensor &output,
                      std::vector<float> &im2col_buffer, bool fuse_relu)
    {
        output.resize(input.n, weights.out_channels, weights.output_h, weights.output_w);
        im2col(input, weights.kernel_h, weights.kernel_w, weights.stride_h, weights.stride_w,
               weights.padding_h, weights.padding_w, weights.output_h, weights.output_w, weights.k_pad,
               im2col_buffer);

        const int output_spatial = weights.output_h * weights.output_w;
        const int packed_bytes = weights.k_pad / 8;
        const float *col_base = im2col_buffer.data();
        float *out_base = output.ptr();
        for (int n = 0; n < input.n; ++n)
        {
            const float *sample_cols = col_base + static_cast<std::size_t>(n) * output_spatial * weights.k_pad;
            float *sample_out = out_base + static_cast<std::size_t>(n) * weights.out_channels * output_spatial;
            for (int oc = 0; oc < weights.out_channels; ++oc)
            {
                const std::uint8_t *pos_row = weights.pos_bits.data() + static_cast<std::size_t>(oc) * packed_bytes;
                const std::uint8_t *neg_row = weights.neg_bits.data() + static_cast<std::size_t>(oc) * packed_bytes;
                for (int spatial = 0; spatial < output_spatial; ++spatial)
                {
                    const float *activation_row = sample_cols + static_cast<std::size_t>(spatial) * weights.k_pad;
                    float value = dot_product_ternary_avx2(activation_row, pos_row, neg_row, packed_bytes);
                    sample_out[static_cast<std::size_t>(oc) * output_spatial + spatial] = fuse_relu
                                                                                           ? std::max(0.0f, value * weights.scale[oc] + weights.bias[oc])
                                                                                           : value * weights.scale[oc] + weights.bias[oc];
                }
            }
        }
    }

    void relu_inplace(Tensor &tensor)
    {
        float *data = tensor.data.data();
        const std::size_t total = tensor.data.size();
        const __m256 zero = _mm256_setzero_ps();
        std::size_t i = 0;
        for (; i + 7 < total; i += 8)
        {
            _mm256_storeu_ps(data + i, _mm256_max_ps(zero, _mm256_loadu_ps(data + i)));
        }
        for (; i < total; ++i)
        {
            data[i] = data[i] > 0.0f ? data[i] : 0.0f;
        }
    }

    void add_inplace(Tensor &target, const Tensor &source)
    {
        assert(target.n == source.n && target.c == source.c && target.h == source.h && target.w == source.w);
        float *dst = target.data.data();
        const float *src = source.data.data();
        const std::size_t total = target.data.size();
        std::size_t i = 0;
        for (; i + 7 < total; i += 8)
        {
            _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(src + i)));
        }
        for (; i < total; ++i)
        {
            dst[i] += src[i];
        }
    }

    void global_avg_pool(const Tensor &input, Tensor &output)
    {
        output.resize(input.n, input.c, 1, 1);
        const int spatial = input.h * input.w;
        for (int n = 0; n < input.n; ++n)
        {
            const float *sample_in = input.ptr() + static_cast<std::size_t>(n) * input.c * spatial;
            float *sample_out = output.ptr() + static_cast<std::size_t>(n) * input.c;
            for (int channel = 0; channel < input.c; ++channel)
            {
                const float *channel_in = sample_in + static_cast<std::size_t>(channel) * spatial;
                float sum = 0.0f;
                for (int i = 0; i < spatial; ++i)
                {
                    sum += channel_in[i];
                }
                sample_out[channel] = sum / static_cast<float>(spatial);
            }
        }
    }

    void linear(const Tensor &input, const LinearWeights &weights, Tensor &output)
    {
        output.resize(input.n, weights.out_features, 1, 1);
        const int features = weights.in_features;
        for (int n = 0; n < input.n; ++n)
        {
            const float *sample_in = input.ptr() + static_cast<std::size_t>(n) * features;
            float *sample_out = output.ptr() + static_cast<std::size_t>(n) * weights.out_features;
            for (int oc = 0; oc < weights.out_features; ++oc)
            {
                const float *weight_row = weights.weight.data() + static_cast<std::size_t>(oc) * features;
                sample_out[oc] = dot_product_fp32_avx2(sample_in, weight_row, features) + weights.bias[oc];
            }
        }
    }

} // namespace ternary