#pragma once

#include <cstdint>

namespace ternary
{

    float dot_product_fp32_avx2(const float *lhs, const float *rhs, int length);
    float dot_product_ternary_avx2(const float *activation, const std::uint8_t *pos_bits,
                                   const std::uint8_t *neg_bits, int packed_bytes);
    // Compute 4 ternary dot products simultaneously, loading activation once.
    // results[0..3] receive the 4 dot products.
    void dot_product_ternary_4x_avx2(const float *activation,
                                     const std::uint8_t *pos0, const std::uint8_t *neg0,
                                     const std::uint8_t *pos1, const std::uint8_t *neg1,
                                     const std::uint8_t *pos2, const std::uint8_t *neg2,
                                     const std::uint8_t *pos3, const std::uint8_t *neg3,
                                     int packed_bytes, float *results);

    void dot_product_ternary_2x4_avx2(const float *act0, const float *act1,
                                      const std::uint8_t *pos0, const std::uint8_t *neg0,
                                      const std::uint8_t *pos1, const std::uint8_t *neg1,
                                      const std::uint8_t *pos2, const std::uint8_t *neg2,
                                      const std::uint8_t *pos3, const std::uint8_t *neg3,
                                      int packed_bytes, float *res0, float *res1);

} // namespace ternary