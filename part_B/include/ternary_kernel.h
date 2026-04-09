#pragma once

#include <cstdint>

namespace ternary
{

    float dot_product_fp32_avx2(const float *lhs, const float *rhs, int length);
    float dot_product_ternary_vnni(const std::int8_t *activation_int8, const std::uint8_t *pos_bits,
                                   const std::uint8_t *neg_bits, int packed_bytes);

} // namespace ternary