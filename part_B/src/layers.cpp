#include "layers.h"

#include <algorithm>
#include <cassert>
#include <immintrin.h>
#include <omp.h>

#include "ternary_kernel.h"
#include "ternary_kernel_tl.h"

#ifdef PROFILE_LAYERS
#include <chrono>
namespace ternary
{
    TernaryConvBreakdown g_ternary_breakdown;
}
namespace
{
    using Clock = std::chrono::steady_clock;
    using us_t = std::chrono::microseconds;
    inline long long layer_elapsed_us(Clock::time_point t0, Clock::time_point t1)
    {
        return std::chrono::duration_cast<us_t>(t1 - t0).count();
    }
} // anonymous namespace
#endif

namespace ternary
{
    namespace
    {

        constexpr int kSpatialTile = 32; // autotuned: beats 64 by ~20% (better OMP balance for all 3 stages)
        constexpr int kChannelTile = 32; // Tile output channels for L2 cache blocking

        inline int round_up(int value, int multiple)
        {
            return ((value + multiple - 1) / multiple) * multiple;
        }

        void im2col_fp32(const Tensor &input, int kernel_h, int kernel_w, int stride_h, int stride_w,
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
            for (int n = 0; n < batch; ++n)
            {
                const float *input_base = input.ptr() + static_cast<std::size_t>(n) * channels * input_h * input_w;
#pragma omp parallel for collapse(2) schedule(static)
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
                                    else
                                    {
                                        dst[col_index] = 0.0f;
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

        void im2col_u8(const Tensor &input, int kernel_h, int kernel_w, int stride_h, int stride_w,
                       int padding_h, int padding_w, int out_h, int out_w, int k_pad, float activation_scale,
                       std::vector<std::uint8_t> &im2col_buffer)
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
            const float inv_scale = activation_scale > 0.0f ? (1.0f / activation_scale) : 0.0f;
            for (int n = 0; n < batch; ++n)
            {
                const float *input_base = input.ptr() + static_cast<std::size_t>(n) * channels * input_h * input_w;
#pragma omp parallel for collapse(2) schedule(static)
                for (int oy = 0; oy < out_h; ++oy)
                {
                    for (int ox = 0; ox < out_w; ++ox)
                    {
                        std::uint8_t *dst = im2col_buffer.data() + ((static_cast<std::size_t>(n) * out_h + oy) * out_w + ox) * k_pad;
                        int col_index = 0;
                        for (int channel = 0; channel < channels; ++channel)
                        {
                            for (int ky = 0; ky < kernel_h; ++ky)
                            {
                                const int in_y = oy * stride_h + ky - padding_h;
                                for (int kx = 0; kx < kernel_w; ++kx)
                                {
                                    const int in_x = ox * stride_w + kx - padding_w;
                                    std::uint8_t q = 0;
                                    if (in_y >= 0 && in_y < input_h && in_x >= 0 && in_x < input_w)
                                    {
                                        const std::size_t input_index = static_cast<std::size_t>(channel) * input_h * input_w + in_y * input_w + in_x;
                                        const float value = input_base[input_index];
                                        int quantized = static_cast<int>(value * inv_scale + 0.5f);
                                        if (quantized < 0)
                                        {
                                            quantized = 0;
                                        }
                                        else if (quantized > 255)
                                        {
                                            quantized = 255;
                                        }
                                        q = static_cast<std::uint8_t>(quantized);
                                    }
                                    dst[col_index] = q;
                                    ++col_index;
                                }
                            }
                        }
                        for (int i = kernel_elements; i < k_pad; ++i)
                        {
                            dst[i] = 0;
                        }
                    }
                }
            }
        }

        void conv_ternary_tl(const TernaryConv2DWeights &weights,
                             const std::uint8_t *col_base,
                             int batch,
                             int output_spatial,
                             float *out_base,
                             bool fuse_relu)
        {
            const float *const scale_base = weights.scale.data();
            const float *const bias_base = weights.bias.data();
            const std::uint8_t *const tl_index = weights.tl_index.data();
            const std::uint8_t *const tl_sign = weights.tl_sign.data();

            const int groups = weights.tl_groups;
            const int oc_stride = weights.tl_oc_stride;

#pragma omp parallel for collapse(2) schedule(guided)
            for (int n = 0; n < batch; ++n)
            {
                for (int spatial = 0; spatial < output_spatial; ++spatial)
                {
                    const std::uint8_t *act = col_base + ((static_cast<std::size_t>(n) * output_spatial + spatial) * weights.k_pad);
                    float *sample_out = out_base + static_cast<std::size_t>(n) * weights.out_channels * output_spatial;

                    for (int oc_base = 0; oc_base < weights.out_channels; oc_base += 16)
                    {
                        const int valid = std::min(16, weights.out_channels - oc_base);
                        alignas(64) int acc[16] = {0};

                        if (weights.storage_kind == TernaryStorageKind::kTL1)
                        {
                            dot_product_u8_tl1_16(act, tl_index, tl_sign, groups, oc_stride, oc_base, acc);
                        }
                        else
                        {
                            for (int lane = 0; lane < valid; ++lane)
                            {
                                const int oc = oc_base + lane;
                                acc[lane] = dot_product_u8_tl2_scalar(act,
                                                                      tl_index,
                                                                      tl_sign,
                                                                      groups,
                                                                      oc_stride,
                                                                      oc,
                                                                      weights.tl_tail_start,
                                                                      weights.tl1_tail_index.data(),
                                                                      weights.tl1_tail_sign.data(),
                                                                      weights.tl1_tail_groups,
                                                                      weights.tl1_tail_oc_stride);
                            }
                        }

                        for (int lane = 0; lane < valid; ++lane)
                        {
                            const int oc = oc_base + lane;
                            const float output_scale = weights.activation_scale * scale_base[oc];
                            const float value = static_cast<float>(acc[lane]) * output_scale + bias_base[oc];
                            sample_out[static_cast<std::size_t>(oc) * output_spatial + spatial] =
                                fuse_relu ? std::max(0.0f, value) : value;
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
        im2col_fp32(input, weights.kernel_h, weights.kernel_w, weights.stride_h, weights.stride_w,
                    weights.padding_h, weights.padding_w, weights.output_h, weights.output_w, k_pad,
                    im2col_buffer);

        const int output_spatial = weights.output_h * weights.output_w;
        const float *col_base = im2col_buffer.data();
        float *out_base = output.ptr();
        const float *const weight_base = weights.weight.data();
        const float *const bias_base = weights.has_bias ? weights.bias.data() : nullptr;
        for (int n = 0; n < input.n; ++n)
        {
            const float *sample_cols = col_base + static_cast<std::size_t>(n) * output_spatial * k_pad;
            float *sample_out = out_base + static_cast<std::size_t>(n) * weights.out_channels * output_spatial;
            // (M, N) cache blocking for FP32 convolution
#pragma omp parallel
            {
                for (int spatial_base = 0; spatial_base < output_spatial; spatial_base += kSpatialTile)
                {
                    const int spatial_end = std::min(spatial_base + kSpatialTile, output_spatial);
#pragma omp for schedule(static) nowait
                    for (int oc_base = 0; oc_base < weights.out_channels; oc_base += kChannelTile)
                    {
                        const int oc_end = std::min(oc_base + kChannelTile, weights.out_channels);
                        for (int spatial = spatial_base; spatial < spatial_end; ++spatial)
                        {
                            const float *activation_row = sample_cols + static_cast<std::size_t>(spatial) * k_pad;
                            for (int oc = oc_base; oc < oc_end; ++oc)
                            {
                                const float *weight_row = weight_base + static_cast<std::size_t>(oc) * kernel_elements;
                                float value = dot_product_fp32_avx2(weight_row, activation_row, kernel_elements);
                                if (weights.has_bias)
                                {
                                    value += bias_base[oc];
                                }
                                sample_out[static_cast<std::size_t>(oc) * output_spatial + spatial] = value;
                            }
                        }
                    }
                }
            }
        }
    }

    void conv_ternary(const Tensor &input, const TernaryConv2DWeights &weights, Tensor &output,
                      std::vector<std::uint8_t> &im2col_buffer, bool fuse_relu)
    {
        output.resize(input.n, weights.out_channels, weights.output_h, weights.output_w);
#ifdef PROFILE_LAYERS
        {
            auto _t0 = std::chrono::steady_clock::now();
            im2col_u8(input, weights.kernel_h, weights.kernel_w, weights.stride_h, weights.stride_w,
                      weights.padding_h, weights.padding_w, weights.output_h, weights.output_w, weights.k_pad,
                      weights.activation_scale, im2col_buffer);
            auto _t1 = std::chrono::steady_clock::now();
            g_ternary_breakdown.im2col_us += layer_elapsed_us(_t0, _t1);
        }
        auto _dot_t0 = std::chrono::steady_clock::now();
#else
        im2col_u8(input, weights.kernel_h, weights.kernel_w, weights.stride_h, weights.stride_w,
                  weights.padding_h, weights.padding_w, weights.output_h, weights.output_w, weights.k_pad,
                  weights.activation_scale, im2col_buffer);
#endif

        const int output_spatial = weights.output_h * weights.output_w;
        const int packed_bytes = weights.k_pad;
        const std::uint8_t *col_base = im2col_buffer.data();
        float *out_base = output.ptr();

        if (weights.storage_kind != TernaryStorageKind::kInt8)
        {
            conv_ternary_tl(weights, col_base, input.n, output_spatial, out_base, fuse_relu);
#ifdef PROFILE_LAYERS
            g_ternary_breakdown.dot_us += layer_elapsed_us(_dot_t0, std::chrono::steady_clock::now());
#endif
            return;
        }

        const std::int8_t *const weight_base = weights.weights.data();
        const float *const scale_base = weights.scale.data();
        const float *const bias_base = weights.bias.data();

        const int oc4_count = weights.out_channels / 4;
        const int oc4_rem = weights.out_channels % 4;
        const int spatial_tiles = (output_spatial + kSpatialTile - 1) / kSpatialTile;

#pragma omp parallel
        {
            for (int n = 0; n < input.n; ++n)
            {
                const std::uint8_t *sample_cols = col_base + static_cast<std::size_t>(n) * output_spatial * weights.k_pad;
                float *sample_out = out_base + static_cast<std::size_t>(n) * weights.out_channels * output_spatial;

#pragma omp for schedule(guided) collapse(2) nowait
                for (int oc_grp = 0; oc_grp < oc4_count; ++oc_grp)
                {
                    for (int stile = 0; stile < spatial_tiles; ++stile)
                    {
                        const int oc_base = oc_grp * 4;
                        const int sp_base = stile * kSpatialTile;
                        const int sp_end = std::min(sp_base + kSpatialTile, output_spatial);

                        const std::int8_t *w0 = weight_base + static_cast<std::size_t>(oc_base + 0) * packed_bytes;
                        const std::int8_t *w1 = weight_base + static_cast<std::size_t>(oc_base + 1) * packed_bytes;
                        const std::int8_t *w2 = weight_base + static_cast<std::size_t>(oc_base + 2) * packed_bytes;
                        const std::int8_t *w3 = weight_base + static_cast<std::size_t>(oc_base + 3) * packed_bytes;

                        const float s0 = weights.activation_scale * scale_base[oc_base + 0], b0 = bias_base[oc_base + 0];
                        const float s1 = weights.activation_scale * scale_base[oc_base + 1], b1 = bias_base[oc_base + 1];
                        const float s2 = weights.activation_scale * scale_base[oc_base + 2], b2 = bias_base[oc_base + 2];
                        const float s3 = weights.activation_scale * scale_base[oc_base + 3], b3 = bias_base[oc_base + 3];

                        float *out0 = sample_out + static_cast<std::size_t>(oc_base + 0) * output_spatial;
                        float *out1 = sample_out + static_cast<std::size_t>(oc_base + 1) * output_spatial;
                        float *out2 = sample_out + static_cast<std::size_t>(oc_base + 2) * output_spatial;
                        float *out3 = sample_out + static_cast<std::size_t>(oc_base + 3) * output_spatial;

                        for (int spatial = sp_base; spatial + 1 < sp_end; spatial += 2)
                        {
                            const std::uint8_t *act0 = sample_cols + static_cast<std::size_t>(spatial) * weights.k_pad;
                            const std::uint8_t *act1 = sample_cols + static_cast<std::size_t>(spatial + 1) * weights.k_pad;
                            alignas(16) int res0[4] = {0, 0, 0, 0};
                            alignas(16) int res1[4] = {0, 0, 0, 0};
                            dot_product_u8s8_2x4_vnni(act0, act1, w0, w1, w2, w3, packed_bytes, res0, res1);
                            if (fuse_relu)
                            {
                                out0[spatial] = std::max(0.0f, static_cast<float>(res0[0]) * s0 + b0);
                                out1[spatial] = std::max(0.0f, static_cast<float>(res0[1]) * s1 + b1);
                                out2[spatial] = std::max(0.0f, static_cast<float>(res0[2]) * s2 + b2);
                                out3[spatial] = std::max(0.0f, static_cast<float>(res0[3]) * s3 + b3);

                                out0[spatial + 1] = std::max(0.0f, static_cast<float>(res1[0]) * s0 + b0);
                                out1[spatial + 1] = std::max(0.0f, static_cast<float>(res1[1]) * s1 + b1);
                                out2[spatial + 1] = std::max(0.0f, static_cast<float>(res1[2]) * s2 + b2);
                                out3[spatial + 1] = std::max(0.0f, static_cast<float>(res1[3]) * s3 + b3);
                            }
                            else
                            {
                                out0[spatial] = static_cast<float>(res0[0]) * s0 + b0;
                                out1[spatial] = static_cast<float>(res0[1]) * s1 + b1;
                                out2[spatial] = static_cast<float>(res0[2]) * s2 + b2;
                                out3[spatial] = static_cast<float>(res0[3]) * s3 + b3;

                                out0[spatial + 1] = static_cast<float>(res1[0]) * s0 + b0;
                                out1[spatial + 1] = static_cast<float>(res1[1]) * s1 + b1;
                                out2[spatial + 1] = static_cast<float>(res1[2]) * s2 + b2;
                                out3[spatial + 1] = static_cast<float>(res1[3]) * s3 + b3;
                            }
                        }

                        if ((sp_end - sp_base) % 2 != 0)
                        {
                            const int spatial = sp_end - 1;
                            const std::uint8_t *act = sample_cols + static_cast<std::size_t>(spatial) * weights.k_pad;
                            alignas(16) int res[4] = {0, 0, 0, 0};
                            dot_product_u8s8_4x_vnni(act, w0, w1, w2, w3, packed_bytes, res);
                            if (fuse_relu)
                            {
                                out0[spatial] = std::max(0.0f, static_cast<float>(res[0]) * s0 + b0);
                                out1[spatial] = std::max(0.0f, static_cast<float>(res[1]) * s1 + b1);
                                out2[spatial] = std::max(0.0f, static_cast<float>(res[2]) * s2 + b2);
                                out3[spatial] = std::max(0.0f, static_cast<float>(res[3]) * s3 + b3);
                            }
                            else
                            {
                                out0[spatial] = static_cast<float>(res[0]) * s0 + b0;
                                out1[spatial] = static_cast<float>(res[1]) * s1 + b1;
                                out2[spatial] = static_cast<float>(res[2]) * s2 + b2;
                                out3[spatial] = static_cast<float>(res[3]) * s3 + b3;
                            }
                        }
                    }
                }

                if (oc4_rem > 0)
                {
                    const int oc_rem_base = oc4_count * 4;
#pragma omp for schedule(static)
                    for (int oc = oc_rem_base; oc < weights.out_channels; ++oc)
                    {
                        const std::int8_t *weight_row = weight_base + static_cast<std::size_t>(oc) * packed_bytes;
                        const float output_scale = weights.activation_scale * scale_base[oc];
                        for (int spatial = 0; spatial < output_spatial; ++spatial)
                        {
                            const std::uint8_t *act = sample_cols + static_cast<std::size_t>(spatial) * weights.k_pad;
                            int dot = 0;
                            dot_product_u8s8_vnni(act, weight_row, packed_bytes, &dot);
                            float value = static_cast<float>(dot) * output_scale + bias_base[oc];
                            sample_out[static_cast<std::size_t>(oc) * output_spatial + spatial] = fuse_relu ? std::max(0.0f, value) : value;
                        }
                    }
                }
            }
        }
#ifdef PROFILE_LAYERS
        g_ternary_breakdown.dot_us += layer_elapsed_us(_dot_t0, std::chrono::steady_clock::now());
#endif
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
            // Tile output features for better L2 cache reuse
            for (int oc_base = 0; oc_base < weights.out_features; oc_base += kChannelTile)
            {
                const int oc_end = std::min(oc_base + kChannelTile, weights.out_features);
                for (int oc = oc_base; oc < oc_end; ++oc)
                {
                    const float *weight_row = weights.weight.data() + static_cast<std::size_t>(oc) * features;
                    sample_out[oc] = dot_product_fp32_avx2(sample_in, weight_row, features) + weights.bias[oc];
                }
            }
        }
    }

} // namespace ternary