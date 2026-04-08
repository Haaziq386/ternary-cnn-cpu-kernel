#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "tensor.h"

namespace ternary
{

    enum class LayerKind : std::uint32_t
    {
        kFp32Conv = 0,
        kTernaryConv = 1,
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
        std::vector<std::uint8_t> pos_bits;
        std::vector<std::uint8_t> neg_bits;
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
        Tensor sample_input;
        Tensor sample_outputs;
        std::vector<std::int64_t> sample_labels;
    };

    struct InferenceScratch
    {
        Tensor a;
        Tensor b;
        Tensor c;
        std::vector<float> im2col;
    };

    ResNet20Weights load_model(const std::string &path);
    void prepare_scratch(const ResNet20Weights &model, int batch_size, InferenceScratch &scratch);
    std::size_t estimate_im2col_capacity(const ResNet20Weights &model, int batch_size);

} // namespace ternary