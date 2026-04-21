#include "resnet20.h"

#include <array>
#include "layers.h"

#ifdef PROFILE_LAYERS
#include <chrono>
#include <cstdio>

namespace
{
    using Clock = std::chrono::steady_clock;
    using us = std::chrono::microseconds;

    inline long long elapsed_us(Clock::time_point t0, Clock::time_point t1)
    {
        return std::chrono::duration_cast<us>(t1 - t0).count();
    }
} // anonymous namespace

#define TIME_CALL(acc, call)         \
    do                               \
    {                                \
        auto _t0 = Clock::now();     \
        call;                        \
        auto _t1 = Clock::now();     \
        acc += elapsed_us(_t0, _t1); \
    } while (0)
#else
#define TIME_CALL(acc, call) call
#endif

namespace ternary
{

    Tensor run_resnet20(const ResNet20Weights &model, const Tensor &input, InferenceScratch &scratch)
    {
#ifdef PROFILE_LAYERS
        long long t_conv_fp32 = 0;
        long long t_conv_ternary = 0;
        long long t_relu = 0;
        long long t_add = 0;
        long long t_pool = 0;
        long long t_linear = 0;

        // Per-stage ternary conv totals (ResNet20: 3 blocks × 3 stages = 9 blocks total)
        // Stage 1: blocks 0-2 (16 ch, 32×32), Stage 2: blocks 3-5 (32 ch, 16×16),
        // Stage 3: blocks 6-8 (64 ch, 8×8)
        std::array<long long, 3> t_stage_ternary = {0, 0, 0};
        // im2col vs. dot-product breakdown accumulated across all 18 ternary convs
        long long t_all_im2col = 0;
        long long t_all_dot = 0;

        // Reset global breakdown counter before this forward pass
        g_ternary_breakdown = {};
#endif

        const Tensor *current = &input;

        TIME_CALL(t_conv_fp32, conv_fp32(*current, model.stem, scratch.a, scratch.im2col_fp32));
        TIME_CALL(t_relu, relu_inplace(scratch.a));
        current = &scratch.a;

        for (int block_idx = 0; block_idx < static_cast<int>(model.blocks.size()); ++block_idx)
        {
            const auto &block = model.blocks[block_idx];
#ifdef PROFILE_LAYERS
            // Snapshot breakdown before this block's two ternary convs
            long long snap_im2col = g_ternary_breakdown.im2col_us;
            long long snap_dot = g_ternary_breakdown.dot_us;
            long long snap_total = t_conv_ternary;
#endif
            (void)block_idx; // suppress unused-variable warning in non-profile builds
            Tensor *conv1_out = nullptr;
            Tensor *conv2_out = nullptr;
            Tensor *proj_out = nullptr;
            if (current == &scratch.a)
            {
                conv1_out = &scratch.b;
                conv2_out = &scratch.c;
                proj_out = &scratch.b;
            }
            else if (current == &scratch.b)
            {
                conv1_out = &scratch.c;
                conv2_out = &scratch.a;
                proj_out = &scratch.c;
            }
            else
            {
                conv1_out = &scratch.a;
                conv2_out = &scratch.b;
                proj_out = &scratch.a;
            }

            TIME_CALL(t_conv_ternary, conv_ternary(*current, block.conv1, *conv1_out, scratch.im2col_u8, true));
            // TIME_CALL(t_relu,         relu_inplace(*conv1_out));
            TIME_CALL(t_conv_ternary, conv_ternary(*conv1_out, block.conv2, *conv2_out, scratch.im2col_u8));
            if (block.has_projection)
            {
                TIME_CALL(t_conv_fp32, conv_fp32(*current, block.projection, *proj_out, scratch.im2col_fp32));
                TIME_CALL(t_add, add_inplace(*conv2_out, *proj_out));
            }
            else
            {
                TIME_CALL(t_add, add_inplace(*conv2_out, *current));
            }
            TIME_CALL(t_relu, relu_inplace(*conv2_out));
            current = conv2_out;

#ifdef PROFILE_LAYERS
            {
                int stage = block_idx / 3; // 0, 1, or 2
                t_stage_ternary[stage] += t_conv_ternary - snap_total;
                (void)snap_im2col;
                (void)snap_dot;
            }
#endif
        }

        Tensor pooled;
        pooled.reserve(static_cast<std::size_t>(input.n) * 64);
        TIME_CALL(t_pool, global_avg_pool(*current, pooled));

        Tensor logits;
        TIME_CALL(t_linear, linear(pooled, model.fc, logits));

#ifdef PROFILE_LAYERS
        t_all_im2col = g_ternary_breakdown.im2col_us;
        t_all_dot = g_ternary_breakdown.dot_us;

        long long total = t_conv_fp32 + t_conv_ternary + t_relu + t_add + t_pool + t_linear;
        std::printf("\n[PROFILE] layer breakdown (accumulated over 1 forward pass):\n");
        std::printf("  conv_fp32 total:     %6lld us  (stem + 3 projections)\n", t_conv_fp32);
        std::printf("  conv_ternary total:  %6lld us  (18 residual convs)\n", t_conv_ternary);
        std::printf("    |- im2col:         %6lld us  (%4.1f%% of ternary)\n",
                    t_all_im2col,
                    t_conv_ternary > 0 ? 100.0 * t_all_im2col / t_conv_ternary : 0.0);
        std::printf("    |- dot product:    %6lld us  (%4.1f%% of ternary)\n",
                    t_all_dot,
                    t_conv_ternary > 0 ? 100.0 * t_all_dot / t_conv_ternary : 0.0);
        std::printf("    |- stage 1 (16ch 32x32): %6lld us\n", t_stage_ternary[0]);
        std::printf("    |- stage 2 (32ch 16x16): %6lld us\n", t_stage_ternary[1]);
        std::printf("    |- stage 3 (64ch  8x8):  %6lld us\n", t_stage_ternary[2]);
        std::printf("  relu total:          %6lld us\n", t_relu);
        std::printf("  add total:           %6lld us\n", t_add);
        std::printf("  global_avg_pool:     %6lld us\n", t_pool);
        std::printf("  linear:              %6lld us\n", t_linear);
        std::printf("  TOTAL:               %6lld us\n\n", total);
#endif

        return logits;
    }

} // namespace ternary
