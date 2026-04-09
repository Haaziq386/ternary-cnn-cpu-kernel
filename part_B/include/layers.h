#pragma once

#include <vector>

#include "model.h"

namespace ternary
{

    void conv_fp32(const Tensor &input, const Conv2DWeightsFP32 &weights, Tensor &output,
                   std::vector<float> &im2col);
    void conv_ternary(const Tensor &input, const TernaryConv2DWeights &weights, Tensor &output,
                      std::vector<float> &im2col, bool fuse_relu = false);
    void relu_inplace(Tensor &tensor);
    void add_inplace(Tensor &target, const Tensor &source);
    void global_avg_pool(const Tensor &input, Tensor &output);
    void linear(const Tensor &input, const LinearWeights &weights, Tensor &output);

} // namespace ternary