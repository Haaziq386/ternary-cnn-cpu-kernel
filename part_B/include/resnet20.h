#pragma once

#include "model.h"

namespace ternary
{

    Tensor run_resnet20(const ResNet20Weights &model, const Tensor &input, InferenceScratch &scratch);

} // namespace ternary