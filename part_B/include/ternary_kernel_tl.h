#pragma once

#include <cstdint>

namespace ternary
{

#if defined(__GNUC__) || defined(__clang__)
#define TERNARY_RESTRICT __restrict__
#else
#define TERNARY_RESTRICT
#endif

    // TL1 kernel over 2-weight groups. Weight indices/signs are stored group-major:
    // [group][output-channel].
    void dot_product_u8_tl1_16(const std::uint8_t *TERNARY_RESTRICT activation,
                               const std::uint8_t *TERNARY_RESTRICT index_group_major,
                               const std::uint8_t *TERNARY_RESTRICT sign_group_major,
                               int groups,
                               int oc_stride,
                               int oc_base,
                               int *TERNARY_RESTRICT out16);

    int dot_product_u8_tl1_scalar(const std::uint8_t *TERNARY_RESTRICT activation,
                                  const std::uint8_t *TERNARY_RESTRICT index_group_major,
                                  const std::uint8_t *TERNARY_RESTRICT sign_group_major,
                                  int groups,
                                  int oc_stride,
                                  int oc);

    // TL2 reference scalar kernel (group size 3) used as fallback/validation path.
    int dot_product_u8_tl2_scalar(const std::uint8_t *TERNARY_RESTRICT activation,
                                  const std::uint8_t *TERNARY_RESTRICT index_group_major,
                                  const std::uint8_t *TERNARY_RESTRICT sign_group_major,
                                  int groups,
                                  int oc_stride,
                                  int oc,
                                  int remainder_start,
                                  const std::uint8_t *TERNARY_RESTRICT tl1_index_group_major,
                                  const std::uint8_t *TERNARY_RESTRICT tl1_sign_group_major,
                                  int tl1_groups,
                                  int tl1_oc_stride);

#undef TERNARY_RESTRICT

} // namespace ternary
