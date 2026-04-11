#pragma once

#include <cstdint>

namespace ternary
{

#if defined(__GNUC__) || defined(__clang__)
#define TERNARY_RESTRICT __restrict__
#else
#define TERNARY_RESTRICT
#endif

    float dot_product_fp32_avx2(const float *TERNARY_RESTRICT lhs,
                                const float *TERNARY_RESTRICT rhs, int length);
    float dot_product_ternary_avx2(const float *TERNARY_RESTRICT activation,
                                   const std::uint8_t *TERNARY_RESTRICT pos_bits,
                                   const std::uint8_t *TERNARY_RESTRICT neg_bits,
                                   int packed_bytes);
    // Compute 4 ternary dot products simultaneously, loading activation once.
    // results[0..3] receive the 4 dot products.
    void dot_product_ternary_4x_avx2(const float *TERNARY_RESTRICT activation,
                                     const std::uint8_t *TERNARY_RESTRICT pos0,
                                     const std::uint8_t *TERNARY_RESTRICT neg0,
                                     const std::uint8_t *TERNARY_RESTRICT pos1,
                                     const std::uint8_t *TERNARY_RESTRICT neg1,
                                     const std::uint8_t *TERNARY_RESTRICT pos2,
                                     const std::uint8_t *TERNARY_RESTRICT neg2,
                                     const std::uint8_t *TERNARY_RESTRICT pos3,
                                     const std::uint8_t *TERNARY_RESTRICT neg3,
                                     int packed_bytes,
                                     float *TERNARY_RESTRICT results);

    void dot_product_ternary_2x4_avx2(const float *TERNARY_RESTRICT act0,
                                      const float *TERNARY_RESTRICT act1,
                                      const std::uint8_t *TERNARY_RESTRICT pos0,
                                      const std::uint8_t *TERNARY_RESTRICT neg0,
                                      const std::uint8_t *TERNARY_RESTRICT pos1,
                                      const std::uint8_t *TERNARY_RESTRICT neg1,
                                      const std::uint8_t *TERNARY_RESTRICT pos2,
                                      const std::uint8_t *TERNARY_RESTRICT neg2,
                                      const std::uint8_t *TERNARY_RESTRICT pos3,
                                      const std::uint8_t *TERNARY_RESTRICT neg3,
                                      int packed_bytes,
                                      float *TERNARY_RESTRICT res0,
                                      float *TERNARY_RESTRICT res1);

#undef TERNARY_RESTRICT

} // namespace ternary