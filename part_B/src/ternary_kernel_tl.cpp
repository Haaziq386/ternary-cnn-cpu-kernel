#include "ternary_kernel_tl.h"

#include <algorithm>
#include <array>
#include <immintrin.h>

namespace ternary
{
    namespace
    {
        inline std::int16_t apply_sign(std::int16_t value, std::uint8_t sign_bit)
        {
            const std::int16_t mask = sign_bit ? static_cast<std::int16_t>(-1) : static_cast<std::int16_t>(0);
            return static_cast<std::int16_t>((value + mask) ^ mask);
        }

        inline std::int16_t tl1_lookup(std::uint8_t idx, std::uint8_t a0, std::uint8_t a1)
        {
            switch (idx)
            {
            case 0:
                return 0;
            case 1:
                return static_cast<std::int16_t>(a1);
            case 2:
                return static_cast<std::int16_t>(a0);
            case 3:
                return static_cast<std::int16_t>(static_cast<int>(a0) + static_cast<int>(a1));
            case 4:
                return static_cast<std::int16_t>(static_cast<int>(a0) - static_cast<int>(a1));
            default:
                return 0;
            }
        }

        inline std::int16_t tl2_lookup(std::uint8_t idx, std::uint8_t a0, std::uint8_t a1, std::uint8_t a2)
        {
            const int s0 = static_cast<int>(a0);
            const int s1 = static_cast<int>(a1);
            const int s2 = static_cast<int>(a2);
            switch (idx)
            {
            case 0:
                return 0;
            case 1:
                return static_cast<std::int16_t>(s2);
            case 2:
                return static_cast<std::int16_t>(s1);
            case 3:
                return static_cast<std::int16_t>(s0);
            case 4:
                return static_cast<std::int16_t>(s1 + s2);
            case 5:
                return static_cast<std::int16_t>(s0 + s2);
            case 6:
                return static_cast<std::int16_t>(s0 + s1);
            case 7:
                return static_cast<std::int16_t>(s0 + s1 + s2);
            case 8:
                return static_cast<std::int16_t>(s1 - s2);
            case 9:
                return static_cast<std::int16_t>(s0 - s2);
            case 10:
                return static_cast<std::int16_t>(s0 - s1);
            case 11:
                return static_cast<std::int16_t>(s0 + s1 - s2);
            case 12:
                return static_cast<std::int16_t>(s0 - s1 + s2);
            case 13:
                return static_cast<std::int16_t>(s0 - s1 - s2);
            default:
                return 0;
            }
        }

        inline void flush_acc16(__m128i &acc16_lo, __m128i &acc16_hi,
                                __m128i &acc32_0, __m128i &acc32_1, __m128i &acc32_2, __m128i &acc32_3)
        {
            const __m128i lo32_a = _mm_cvtepi16_epi32(acc16_lo);
            const __m128i lo32_b = _mm_cvtepi16_epi32(_mm_srli_si128(acc16_lo, 8));
            const __m128i hi32_a = _mm_cvtepi16_epi32(acc16_hi);
            const __m128i hi32_b = _mm_cvtepi16_epi32(_mm_srli_si128(acc16_hi, 8));
            acc32_0 = _mm_add_epi32(acc32_0, lo32_a);
            acc32_1 = _mm_add_epi32(acc32_1, lo32_b);
            acc32_2 = _mm_add_epi32(acc32_2, hi32_a);
            acc32_3 = _mm_add_epi32(acc32_3, hi32_b);
            acc16_lo = _mm_setzero_si128();
            acc16_hi = _mm_setzero_si128();
        }

    } // namespace

    void dot_product_u8_tl1_16(const std::uint8_t *activation,
                               const std::uint8_t *index_group_major,
                               const std::uint8_t *sign_group_major,
                               int groups,
                               int oc_stride,
                               int oc_base,
                               int *out16)
    {
        __m128i acc16_lo = _mm_setzero_si128();
        __m128i acc16_hi = _mm_setzero_si128();
        __m128i acc32_0 = _mm_setzero_si128();
        __m128i acc32_1 = _mm_setzero_si128();
        __m128i acc32_2 = _mm_setzero_si128();
        __m128i acc32_3 = _mm_setzero_si128();

        const __m128i one = _mm_set1_epi8(1);

        for (int g = 0; g < groups; ++g)
        {
            const std::uint8_t a0 = activation[g * 2 + 0];
            const std::uint8_t a1 = activation[g * 2 + 1];
            const std::int16_t e0 = 0;
            const std::int16_t e1 = static_cast<std::int16_t>(a1);
            const std::int16_t e2 = static_cast<std::int16_t>(a0);
            const std::int16_t e3 = static_cast<std::int16_t>(static_cast<int>(a0) + static_cast<int>(a1));
            const std::int16_t e4 = static_cast<std::int16_t>(static_cast<int>(a0) - static_cast<int>(a1));

            alignas(16) std::uint8_t lut_lo_bytes[16] = {
                static_cast<std::uint8_t>(e0 & 0xff),
                static_cast<std::uint8_t>(e1 & 0xff),
                static_cast<std::uint8_t>(e2 & 0xff),
                static_cast<std::uint8_t>(e3 & 0xff),
                static_cast<std::uint8_t>(e4 & 0xff),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            };
            alignas(16) std::uint8_t lut_hi_bytes[16] = {
                static_cast<std::uint8_t>((e0 >> 8) & 0xff),
                static_cast<std::uint8_t>((e1 >> 8) & 0xff),
                static_cast<std::uint8_t>((e2 >> 8) & 0xff),
                static_cast<std::uint8_t>((e3 >> 8) & 0xff),
                static_cast<std::uint8_t>((e4 >> 8) & 0xff),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            };

            const __m128i lut_lo = _mm_load_si128(reinterpret_cast<const __m128i *>(lut_lo_bytes));
            const __m128i lut_hi = _mm_load_si128(reinterpret_cast<const __m128i *>(lut_hi_bytes));

            const std::uint8_t *idx_ptr = index_group_major + static_cast<std::size_t>(g) * oc_stride + oc_base;
            const std::uint8_t *sign_ptr = sign_group_major + static_cast<std::size_t>(g) * oc_stride + oc_base;
            const __m128i idx = _mm_loadu_si128(reinterpret_cast<const __m128i *>(idx_ptr));
            const __m128i sign = _mm_loadu_si128(reinterpret_cast<const __m128i *>(sign_ptr));

            const __m128i sel_lo = _mm_shuffle_epi8(lut_lo, idx);
            const __m128i sel_hi = _mm_shuffle_epi8(lut_hi, idx);

            const __m128i lo16 = _mm_cvtepu8_epi16(sel_lo);
            const __m128i hi16 = _mm_cvtepi8_epi16(sel_hi);
            __m128i val_lo = _mm_add_epi16(lo16, _mm_slli_epi16(hi16, 8));

            const __m128i sel_lo_h = _mm_srli_si128(sel_lo, 8);
            const __m128i sel_hi_h = _mm_srli_si128(sel_hi, 8);
            const __m128i lo16_h = _mm_cvtepu8_epi16(sel_lo_h);
            const __m128i hi16_h = _mm_cvtepi8_epi16(sel_hi_h);
            __m128i val_hi = _mm_add_epi16(lo16_h, _mm_slli_epi16(hi16_h, 8));

            const __m128i sign_mask8 = _mm_cmpeq_epi8(sign, one);
            const __m128i sign_lo = _mm_unpacklo_epi8(sign_mask8, sign_mask8);
            const __m128i sign_hi = _mm_unpackhi_epi8(sign_mask8, sign_mask8);

            val_lo = _mm_xor_si128(_mm_add_epi16(val_lo, sign_lo), sign_lo);
            val_hi = _mm_xor_si128(_mm_add_epi16(val_hi, sign_hi), sign_hi);

            acc16_lo = _mm_add_epi16(acc16_lo, val_lo);
            acc16_hi = _mm_add_epi16(acc16_hi, val_hi);

            if (((g + 1) % 16) == 0)
            {
                flush_acc16(acc16_lo, acc16_hi, acc32_0, acc32_1, acc32_2, acc32_3);
            }
        }

        flush_acc16(acc16_lo, acc16_hi, acc32_0, acc32_1, acc32_2, acc32_3);

        _mm_storeu_si128(reinterpret_cast<__m128i *>(out16 + 0), acc32_0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(out16 + 4), acc32_1);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(out16 + 8), acc32_2);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(out16 + 12), acc32_3);
    }

    int dot_product_u8_tl1_scalar(const std::uint8_t *activation,
                                  const std::uint8_t *index_group_major,
                                  const std::uint8_t *sign_group_major,
                                  int groups,
                                  int oc_stride,
                                  int oc)
    {
        int total = 0;
        for (int g = 0; g < groups; ++g)
        {
            const std::uint8_t a0 = activation[g * 2 + 0];
            const std::uint8_t a1 = activation[g * 2 + 1];
            const std::uint8_t idx = index_group_major[static_cast<std::size_t>(g) * oc_stride + oc];
            const std::uint8_t s = sign_group_major[static_cast<std::size_t>(g) * oc_stride + oc] & 1u;
            std::int16_t v = tl1_lookup(idx, a0, a1);
            v = apply_sign(v, s);
            total += static_cast<int>(v);
        }
        return total;
    }

    int dot_product_u8_tl2_scalar(const std::uint8_t *activation,
                                  const std::uint8_t *index_group_major,
                                  const std::uint8_t *sign_group_major,
                                  int groups,
                                  int oc_stride,
                                  int oc,
                                  int remainder_start,
                                  const std::uint8_t *tl1_index_group_major,
                                  const std::uint8_t *tl1_sign_group_major,
                                  int tl1_groups,
                                  int tl1_oc_stride)
    {
        int total = 0;
        for (int g = 0; g < groups; ++g)
        {
            const std::uint8_t a0 = activation[g * 3 + 0];
            const std::uint8_t a1 = activation[g * 3 + 1];
            const std::uint8_t a2 = activation[g * 3 + 2];
            const std::uint8_t idx = index_group_major[static_cast<std::size_t>(g) * oc_stride + oc];
            const std::uint8_t s = sign_group_major[static_cast<std::size_t>(g) * oc_stride + oc] & 1u;
            std::int16_t v = tl2_lookup(idx, a0, a1, a2);
            v = apply_sign(v, s);
            total += static_cast<int>(v);
        }

        if (tl1_groups > 0)
        {
            const std::uint8_t *tail_act = activation + remainder_start;
            total += dot_product_u8_tl1_scalar(tail_act,
                                               tl1_index_group_major,
                                               tl1_sign_group_major,
                                               tl1_groups,
                                               tl1_oc_stride,
                                               oc);
        }
        return total;
    }

} // namespace ternary
