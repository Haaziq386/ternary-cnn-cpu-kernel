#include "layers.h"

#include <algorithm>
#include <cassert>
#include <immintrin.h>
#include <omp.h>

#include "ternary_kernel.h"

namespace ternary
{
    namespace
    {

        constexpr int kSpatialTile = 64;
        constexpr int kChannelTile = 32; // Tile output channels for L2 cache blocking

        inline int round_up(int value, int multiple)
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
#pragma omp for schedule(static)
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
        const std::uint8_t *const pos_base = weights.pos_bits.data();
        const std::uint8_t *const neg_base = weights.neg_bits.data();
        const float *const scale_base = weights.scale.data();
        const float *const bias_base = weights.bias.data();

        // Number of 4-wide output channel groups and remainder
        const int oc4_count = weights.out_channels / 4;
        const int oc4_rem = weights.out_channels % 4;
        const int spatial_tiles = (output_spatial + kSpatialTile - 1) / kSpatialTile;

#pragma omp parallel
        {
            for (int n = 0; n < input.n; ++n)
            {
                const float *sample_cols = col_base + static_cast<std::size_t>(n) * output_spatial * weights.k_pad;
                float *sample_out = out_base + static_cast<std::size_t>(n) * weights.out_channels * output_spatial;

                // collapse(2): distribute (oc_group × spatial_tile) work items across threads.
                // Each item processes 4 output channels for one spatial tile — 1 activation load
                // feeds all 4 channels (4× fewer activation reads vs single-channel loop).
                // One barrier at the end instead of one per spatial tile.
#pragma omp for schedule(static) collapse(2)
                for (int oc_grp = 0; oc_grp < oc4_count; ++oc_grp)
                {
                    for (int stile = 0; stile < spatial_tiles; ++stile)
                    {
                        const int oc_base = oc_grp * 4;
                        const int sp_base = stile * kSpatialTile;
                        const int sp_end = std::min(sp_base + kSpatialTile, output_spatial);

                        const std::uint8_t *pos0 = pos_base + static_cast<std::size_t>(oc_base + 0) * packed_bytes;
                        const std::uint8_t *neg0 = neg_base + static_cast<std::size_t>(oc_base + 0) * packed_bytes;
                        const std::uint8_t *pos1 = pos_base + static_cast<std::size_t>(oc_base + 1) * packed_bytes;
                        const std::uint8_t *neg1 = neg_base + static_cast<std::size_t>(oc_base + 1) * packed_bytes;
                        const std::uint8_t *pos2 = pos_base + static_cast<std::size_t>(oc_base + 2) * packed_bytes;
                        const std::uint8_t *neg2 = neg_base + static_cast<std::size_t>(oc_base + 2) * packed_bytes;
                        const std::uint8_t *pos3 = pos_base + static_cast<std::size_t>(oc_base + 3) * packed_bytes;
                        const std::uint8_t *neg3 = neg_base + static_cast<std::size_t>(oc_base + 3) * packed_bytes;

                        const float s0 = scale_base[oc_base + 0], b0 = bias_base[oc_base + 0];
                        const float s1 = scale_base[oc_base + 1], b1 = bias_base[oc_base + 1];
                        const float s2 = scale_base[oc_base + 2], b2 = bias_base[oc_base + 2];
                        const float s3 = scale_base[oc_base + 3], b3 = bias_base[oc_base + 3];

                        float *out0 = sample_out + static_cast<std::size_t>(oc_base + 0) * output_spatial;
                        float *out1 = sample_out + static_cast<std::size_t>(oc_base + 1) * output_spatial;
                        float *out2 = sample_out + static_cast<std::size_t>(oc_base + 2) * output_spatial;
                        float *out3 = sample_out + static_cast<std::size_t>(oc_base + 3) * output_spatial;

                        for (int spatial = sp_base; spatial + 1 < sp_end; spatial += 2)
                        {
                            const float *act0 = sample_cols + static_cast<std::size_t>(spatial) * weights.k_pad;
                            const float *act1 = sample_cols + static_cast<std::size_t>(spatial + 1) * weights.k_pad;
                            alignas(16) float res0[4];
                            alignas(16) float res1[4];
                            dot_product_ternary_2x4_avx2(act0, act1, pos0, neg0, pos1, neg1, pos2, neg2, pos3, neg3, packed_bytes, res0, res1);
                            if (fuse_relu)
                            {
                                out0[spatial] = std::max(0.0f, res0[0] * s0 + b0);
                                out1[spatial] = std::max(0.0f, res0[1] * s1 + b1);
                                out2[spatial] = std::max(0.0f, res0[2] * s2 + b2);
                                out3[spatial] = std::max(0.0f, res0[3] * s3 + b3);

                                out0[spatial + 1] = std::max(0.0f, res1[0] * s0 + b0);
                                out1[spatial + 1] = std::max(0.0f, res1[1] * s1 + b1);
                                out2[spatial + 1] = std::max(0.0f, res1[2] * s2 + b2);
                                out3[spatial + 1] = std::max(0.0f, res1[3] * s3 + b3);
                            }
                            else
                            {
                                out0[spatial] = res0[0] * s0 + b0;
                                out1[spatial] = res0[1] * s1 + b1;
                                out2[spatial] = res0[2] * s2 + b2;
                                out3[spatial] = res0[3] * s3 + b3;

                                out0[spatial + 1] = res1[0] * s0 + b0;
                                out1[spatial + 1] = res1[1] * s1 + b1;
                                out2[spatial + 1] = res1[2] * s2 + b2;
                                out3[spatial + 1] = res1[3] * s3 + b3;
                            }
                        }
                        // Handle odd spatial element if it exists
                        if ((sp_end - sp_base) % 2 != 0)
                        {
                            int spatial = sp_end - 1;
                            const float *act = sample_cols + static_cast<std::size_t>(spatial) * weights.k_pad;
                            alignas(16) float res[4];
                            dot_product_ternary_4x_avx2(act, pos0, neg0, pos1, neg1, pos2, neg2, pos3, neg3, packed_bytes, res);
                            if (fuse_relu)
                            {
                                out0[spatial] = std::max(0.0f, res[0] * s0 + b0);
                                out1[spatial] = std::max(0.0f, res[1] * s1 + b1);
                                out2[spatial] = std::max(0.0f, res[2] * s2 + b2);
                                out3[spatial] = std::max(0.0f, res[3] * s3 + b3);
                            }
                            else
                            {
                                out0[spatial] = res[0] * s0 + b0;
                                out1[spatial] = res[1] * s1 + b1;
                                out2[spatial] = res[2] * s2 + b2;
                                out3[spatial] = res[3] * s3 + b3;
                            }
                        }
                    }
                }

                // Scalar fallback for remainder channels when out_channels % 4 != 0
                if (oc4_rem > 0)
                {
                    const int oc_rem_base = oc4_count * 4;
#pragma omp for schedule(static)
                    for (int oc = oc_rem_base; oc < weights.out_channels; ++oc)
                    {
                        const std::uint8_t *pos_row = pos_base + static_cast<std::size_t>(oc) * packed_bytes;
                        const std::uint8_t *neg_row = neg_base + static_cast<std::size_t>(oc) * packed_bytes;
                        for (int spatial = 0; spatial < output_spatial; ++spatial)
                        {
                            const float *act = sample_cols + static_cast<std::size_t>(spatial) * weights.k_pad;
                            float value = dot_product_ternary_avx2(act, pos_row, neg_row, packed_bytes);
                            value = value * scale_base[oc] + bias_base[oc];
                            sample_out[static_cast<std::size_t>(oc) * output_spatial + spatial] = fuse_relu ? std::max(0.0f, value) : value;
                        }
                    }
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