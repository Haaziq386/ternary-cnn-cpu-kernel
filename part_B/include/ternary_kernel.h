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
    bool cpu_supports_avx_vnni();
    void dot_product_u8s8_vnni(const std::uint8_t *TERNARY_RESTRICT activation,
                               const std::int8_t *TERNARY_RESTRICT weights,
                               int packed_bytes, int *TERNARY_RESTRICT result);
    void dot_product_u8s8_4x_vnni(const std::uint8_t *TERNARY_RESTRICT activation,
                                  const std::int8_t *TERNARY_RESTRICT weights0,
                                  const std::int8_t *TERNARY_RESTRICT weights1,
                                  const std::int8_t *TERNARY_RESTRICT weights2,
                                  const std::int8_t *TERNARY_RESTRICT weights3,
                                  int packed_bytes,
                                  int *TERNARY_RESTRICT results);
    void dot_product_u8s8_2x4_vnni(const std::uint8_t *TERNARY_RESTRICT act0,
                                   const std::uint8_t *TERNARY_RESTRICT act1,
                                   const std::int8_t *TERNARY_RESTRICT weights0,
                                   const std::int8_t *TERNARY_RESTRICT weights1,
                                   const std::int8_t *TERNARY_RESTRICT weights2,
                                   const std::int8_t *TERNARY_RESTRICT weights3,
                                   int packed_bytes,
                                   int *TERNARY_RESTRICT res0,
                                   int *TERNARY_RESTRICT res1);

#undef TERNARY_RESTRICT

} // namespace ternary