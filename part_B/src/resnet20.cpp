#include "resnet20.h"

#include "layers.h"

namespace ternary
{

    Tensor run_resnet20(const ResNet20Weights &model, const Tensor &input, InferenceScratch &scratch)
    {
        const Tensor *current = &input;

        conv_fp32(*current, model.stem, scratch.a, scratch.im2col);
        relu_inplace(scratch.a);
        current = &scratch.a;

        for (const auto &block : model.blocks)
        {
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

            conv_ternary(*current, block.conv1, *conv1_out, scratch.im2col, scratch.im2col_int8);
            relu_inplace(*conv1_out);
            conv_ternary(*conv1_out, block.conv2, *conv2_out, scratch.im2col, scratch.im2col_int8);
            if (block.has_projection)
            {
                conv_fp32(*current, block.projection, *proj_out, scratch.im2col);
                add_inplace(*conv2_out, *proj_out);
            }
            else
            {
                add_inplace(*conv2_out, *current);
            }
            relu_inplace(*conv2_out);
            current = conv2_out;
        }

        Tensor pooled;
        pooled.reserve(static_cast<std::size_t>(input.n) * 64);
        global_avg_pool(*current, pooled);

        Tensor logits;
        linear(pooled, model.fc, logits);
        return logits;
    }

} // namespace ternary