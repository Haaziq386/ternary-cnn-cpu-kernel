#pragma once

#include <array>
#include <cstdlib>
#include <cstdint>
#include <new>
#include <string>
#include <vector>

#include "tensor.h"

namespace ternary
{

    template <typename T, std::size_t Alignment>
    struct AlignedAllocator
    {
        using value_type = T;

        AlignedAllocator() noexcept = default;

        template <typename U>
        AlignedAllocator(const AlignedAllocator<U, Alignment> &) noexcept {}

        [[nodiscard]] T *allocate(std::size_t count)
        {
            if (count > static_cast<std::size_t>(-1) / sizeof(T))
            {
                throw std::bad_alloc();
            }
            void *ptr = nullptr;
            if (posix_memalign(&ptr, Alignment, count * sizeof(T)) != 0)
            {
                throw std::bad_alloc();
            }
            return static_cast<T *>(ptr);
        }

        void deallocate(T *ptr, std::size_t) noexcept
        {
            std::free(ptr);
        }

        template <typename U>
        struct rebind
        {
            using other = AlignedAllocator<U, Alignment>;
        };
    };

    enum class LayerKind : std::uint32_t
    {
        kFp32Conv = 0,
        kTernaryConv = 1,
        kTernaryConvInt8 = 3,
        kLinear = 2,
    };

    struct Conv2DWeightsFP32
    {
        int in_channels = 0;
        int out_channels = 0;
        int kernel_h = 0;
        int kernel_w = 0;
        int stride_h = 1;
        int stride_w = 1;
        int padding_h = 0;
        int padding_w = 0;
        int output_h = 0;
        int output_w = 0;
        bool has_bias = false;
        std::vector<float> weight;
        std::vector<float> bias;
    };

    struct TernaryConv2DWeights
    {
        int in_channels = 0;
        int out_channels = 0;
        int kernel_h = 0;
        int kernel_w = 0;
        int stride_h = 1;
        int stride_w = 1;
        int padding_h = 0;
        int padding_w = 0;
        int output_h = 0;
        int output_w = 0;
        int k_pad = 0;
        float activation_scale = 1.0f;
        std::vector<std::int8_t, AlignedAllocator<std::int8_t, 32>> weights;
        std::vector<float> scale;
        std::vector<float> bias;
    };

    struct LinearWeights
    {
        int in_features = 0;
        int out_features = 0;
        std::vector<float> weight;
        std::vector<float> bias;
    };

    struct BasicBlockWeights
    {
        TernaryConv2DWeights conv1;
        TernaryConv2DWeights conv2;
        bool has_projection = false;
        Conv2DWeightsFP32 projection;
    };

    struct ResNet20Weights
    {
        Conv2DWeightsFP32 stem;
        std::array<BasicBlockWeights, 9> blocks;
        LinearWeights fc;
        int input_channels = 0;
        int input_h = 0;
        int input_w = 0;
        int num_classes = 0;
        int sample_count = 0;
        Tensor sample_input;
        Tensor sample_outputs;
        std::vector<std::int64_t> sample_labels;
    };

    struct InferenceScratch
    {
        Tensor a;
        Tensor b;
        Tensor c;
        std::vector<float> im2col_fp32;
        std::vector<std::uint8_t> im2col_u8;
    };

    ResNet20Weights load_model(const std::string &path);
    void prepare_scratch(const ResNet20Weights &model, int batch_size, InferenceScratch &scratch);
    std::size_t estimate_im2col_capacity(const ResNet20Weights &model, int batch_size);

} // namespace ternary