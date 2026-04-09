#include "ternary_kernel.h"

#include <immintrin.h>

namespace ternary
{
    namespace
    {

        alignas(32) const std::uint32_t kMaskTable[256][8] = {
#define MASK_ROW(byte)                                                             \
    {(((byte) & 0x01u) ? 0xffffffffu : 0u), (((byte) & 0x02u) ? 0xffffffffu : 0u), \
     (((byte) & 0x04u) ? 0xffffffffu : 0u), (((byte) & 0x08u) ? 0xffffffffu : 0u), \
     (((byte) & 0x10u) ? 0xffffffffu : 0u), (((byte) & 0x20u) ? 0xffffffffu : 0u), \
     (((byte) & 0x40u) ? 0xffffffffu : 0u), (((byte) & 0x80u) ? 0xffffffffu : 0u)}
            MASK_ROW(0), MASK_ROW(1), MASK_ROW(2), MASK_ROW(3), MASK_ROW(4), MASK_ROW(5), MASK_ROW(6), MASK_ROW(7),
            MASK_ROW(8), MASK_ROW(9), MASK_ROW(10), MASK_ROW(11), MASK_ROW(12), MASK_ROW(13), MASK_ROW(14), MASK_ROW(15),
            MASK_ROW(16), MASK_ROW(17), MASK_ROW(18), MASK_ROW(19), MASK_ROW(20), MASK_ROW(21), MASK_ROW(22), MASK_ROW(23),
            MASK_ROW(24), MASK_ROW(25), MASK_ROW(26), MASK_ROW(27), MASK_ROW(28), MASK_ROW(29), MASK_ROW(30), MASK_ROW(31),
            MASK_ROW(32), MASK_ROW(33), MASK_ROW(34), MASK_ROW(35), MASK_ROW(36), MASK_ROW(37), MASK_ROW(38), MASK_ROW(39),
            MASK_ROW(40), MASK_ROW(41), MASK_ROW(42), MASK_ROW(43), MASK_ROW(44), MASK_ROW(45), MASK_ROW(46), MASK_ROW(47),
            MASK_ROW(48), MASK_ROW(49), MASK_ROW(50), MASK_ROW(51), MASK_ROW(52), MASK_ROW(53), MASK_ROW(54), MASK_ROW(55),
            MASK_ROW(56), MASK_ROW(57), MASK_ROW(58), MASK_ROW(59), MASK_ROW(60), MASK_ROW(61), MASK_ROW(62), MASK_ROW(63),
            MASK_ROW(64), MASK_ROW(65), MASK_ROW(66), MASK_ROW(67), MASK_ROW(68), MASK_ROW(69), MASK_ROW(70), MASK_ROW(71),
            MASK_ROW(72), MASK_ROW(73), MASK_ROW(74), MASK_ROW(75), MASK_ROW(76), MASK_ROW(77), MASK_ROW(78), MASK_ROW(79),
            MASK_ROW(80), MASK_ROW(81), MASK_ROW(82), MASK_ROW(83), MASK_ROW(84), MASK_ROW(85), MASK_ROW(86), MASK_ROW(87),
            MASK_ROW(88), MASK_ROW(89), MASK_ROW(90), MASK_ROW(91), MASK_ROW(92), MASK_ROW(93), MASK_ROW(94), MASK_ROW(95),
            MASK_ROW(96), MASK_ROW(97), MASK_ROW(98), MASK_ROW(99), MASK_ROW(100), MASK_ROW(101), MASK_ROW(102), MASK_ROW(103),
            MASK_ROW(104), MASK_ROW(105), MASK_ROW(106), MASK_ROW(107), MASK_ROW(108), MASK_ROW(109), MASK_ROW(110), MASK_ROW(111),
            MASK_ROW(112), MASK_ROW(113), MASK_ROW(114), MASK_ROW(115), MASK_ROW(116), MASK_ROW(117), MASK_ROW(118), MASK_ROW(119),
            MASK_ROW(120), MASK_ROW(121), MASK_ROW(122), MASK_ROW(123), MASK_ROW(124), MASK_ROW(125), MASK_ROW(126), MASK_ROW(127),
            MASK_ROW(128), MASK_ROW(129), MASK_ROW(130), MASK_ROW(131), MASK_ROW(132), MASK_ROW(133), MASK_ROW(134), MASK_ROW(135),
            MASK_ROW(136), MASK_ROW(137), MASK_ROW(138), MASK_ROW(139), MASK_ROW(140), MASK_ROW(141), MASK_ROW(142), MASK_ROW(143),
            MASK_ROW(144), MASK_ROW(145), MASK_ROW(146), MASK_ROW(147), MASK_ROW(148), MASK_ROW(149), MASK_ROW(150), MASK_ROW(151),
            MASK_ROW(152), MASK_ROW(153), MASK_ROW(154), MASK_ROW(155), MASK_ROW(156), MASK_ROW(157), MASK_ROW(158), MASK_ROW(159),
            MASK_ROW(160), MASK_ROW(161), MASK_ROW(162), MASK_ROW(163), MASK_ROW(164), MASK_ROW(165), MASK_ROW(166), MASK_ROW(167),
            MASK_ROW(168), MASK_ROW(169), MASK_ROW(170), MASK_ROW(171), MASK_ROW(172), MASK_ROW(173), MASK_ROW(174), MASK_ROW(175),
            MASK_ROW(176), MASK_ROW(177), MASK_ROW(178), MASK_ROW(179), MASK_ROW(180), MASK_ROW(181), MASK_ROW(182), MASK_ROW(183),
            MASK_ROW(184), MASK_ROW(185), MASK_ROW(186), MASK_ROW(187), MASK_ROW(188), MASK_ROW(189), MASK_ROW(190), MASK_ROW(191),
            MASK_ROW(192), MASK_ROW(193), MASK_ROW(194), MASK_ROW(195), MASK_ROW(196), MASK_ROW(197), MASK_ROW(198), MASK_ROW(199),
            MASK_ROW(200), MASK_ROW(201), MASK_ROW(202), MASK_ROW(203), MASK_ROW(204), MASK_ROW(205), MASK_ROW(206), MASK_ROW(207),
            MASK_ROW(208), MASK_ROW(209), MASK_ROW(210), MASK_ROW(211), MASK_ROW(212), MASK_ROW(213), MASK_ROW(214), MASK_ROW(215),
            MASK_ROW(216), MASK_ROW(217), MASK_ROW(218), MASK_ROW(219), MASK_ROW(220), MASK_ROW(221), MASK_ROW(222), MASK_ROW(223),
            MASK_ROW(224), MASK_ROW(225), MASK_ROW(226), MASK_ROW(227), MASK_ROW(228), MASK_ROW(229), MASK_ROW(230), MASK_ROW(231),
            MASK_ROW(232), MASK_ROW(233), MASK_ROW(234), MASK_ROW(235), MASK_ROW(236), MASK_ROW(237), MASK_ROW(238), MASK_ROW(239),
            MASK_ROW(240), MASK_ROW(241), MASK_ROW(242), MASK_ROW(243), MASK_ROW(244), MASK_ROW(245), MASK_ROW(246), MASK_ROW(247),
            MASK_ROW(248), MASK_ROW(249), MASK_ROW(250), MASK_ROW(251), MASK_ROW(252), MASK_ROW(253), MASK_ROW(254), MASK_ROW(255)};

        inline __m256 mask_to_ps(std::uint8_t bits)
        {
            const auto *row = reinterpret_cast<const __m256i *>(kMaskTable[bits]);
            return _mm256_castsi256_ps(_mm256_load_si256(row));
        }

        inline float horizontal_sum(__m256 value)
        {
            __m128 low = _mm256_castps256_ps128(value);
            __m128 high = _mm256_extractf128_ps(value, 1);
            __m128 sum = _mm_add_ps(low, high);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            return _mm_cvtss_f32(sum);
        }

    } // namespace

    float dot_product_fp32_avx2(const float *lhs, const float *rhs, int length)
    {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        int index = 0;
#if defined(__GNUC__)
#pragma GCC unroll 4
#endif
        for (; index + 31 < length; index += 32)
        {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + index), _mm256_loadu_ps(rhs + index), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + index + 8), _mm256_loadu_ps(rhs + index + 8), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + index + 16), _mm256_loadu_ps(rhs + index + 16), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + index + 24), _mm256_loadu_ps(rhs + index + 24), acc3);
        }
        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);
        for (; index + 7 < length; index += 8)
        {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + index), _mm256_loadu_ps(rhs + index), acc0);
        }
        float sum = horizontal_sum(acc0);
        for (; index < length; ++index)
        {
            sum += lhs[index] * rhs[index];
        }
        return sum;
    }

    float dot_product_ternary_avx2(const float *activation, const std::uint8_t *pos_bits,
                                   const std::uint8_t *neg_bits, int packed_bytes)
    {
        __m256 pos_acc0 = _mm256_setzero_ps();
        __m256 pos_acc1 = _mm256_setzero_ps();
        __m256 neg_acc0 = _mm256_setzero_ps();
        __m256 neg_acc1 = _mm256_setzero_ps();

        int index = 0;
#if defined(__GNUC__)
#pragma GCC unroll 4
#endif
        for (; index + 3 < packed_bytes; index += 4)
        {
            const __m256 x0 = _mm256_loadu_ps(activation + index * 8);
            const __m256 x1 = _mm256_loadu_ps(activation + index * 8 + 8);
            const __m256 x2 = _mm256_loadu_ps(activation + index * 8 + 16);
            const __m256 x3 = _mm256_loadu_ps(activation + index * 8 + 24);

            pos_acc0 = _mm256_add_ps(pos_acc0, _mm256_and_ps(mask_to_ps(pos_bits[index]), x0));
            pos_acc0 = _mm256_add_ps(pos_acc0, _mm256_and_ps(mask_to_ps(pos_bits[index + 1]), x1));
            pos_acc1 = _mm256_add_ps(pos_acc1, _mm256_and_ps(mask_to_ps(pos_bits[index + 2]), x2));
            pos_acc1 = _mm256_add_ps(pos_acc1, _mm256_and_ps(mask_to_ps(pos_bits[index + 3]), x3));

            neg_acc0 = _mm256_add_ps(neg_acc0, _mm256_and_ps(mask_to_ps(neg_bits[index]), x0));
            neg_acc0 = _mm256_add_ps(neg_acc0, _mm256_and_ps(mask_to_ps(neg_bits[index + 1]), x1));
            neg_acc1 = _mm256_add_ps(neg_acc1, _mm256_and_ps(mask_to_ps(neg_bits[index + 2]), x2));
            neg_acc1 = _mm256_add_ps(neg_acc1, _mm256_and_ps(mask_to_ps(neg_bits[index + 3]), x3));
        }

        float sum = horizontal_sum(pos_acc0) + horizontal_sum(pos_acc1) - horizontal_sum(neg_acc0) - horizontal_sum(neg_acc1);
        for (; index < packed_bytes; ++index)
        {
            const __m256 x = _mm256_loadu_ps(activation + index * 8);
            sum += horizontal_sum(_mm256_and_ps(mask_to_ps(pos_bits[index]), x));
            sum -= horizontal_sum(_mm256_and_ps(mask_to_ps(neg_bits[index]), x));
        }
        return sum;
    }

} // namespace ternary

/*
Extend to 4+4 accumulators in dot_product_ternary_avx2 -> bad results.

    float dot_product_ternary_avx2(const float *activation, const std::uint8_t *pos_bits,
                                   const std::uint8_t *neg_bits, int packed_bytes)
    {
        __m256 pos_acc0 = _mm256_setzero_ps();
        __m256 pos_acc1 = _mm256_setzero_ps();
        __m256 pos_acc2 = _mm256_setzero_ps();
        __m256 pos_acc3 = _mm256_setzero_ps();
        __m256 neg_acc0 = _mm256_setzero_ps();
        __m256 neg_acc1 = _mm256_setzero_ps();
        __m256 neg_acc2 = _mm256_setzero_ps();
        __m256 neg_acc3 = _mm256_setzero_ps();

        int index = 0;
#if defined(__GNUC__)
#pragma GCC unroll 4
#endif
        for (; index + 7 < packed_bytes; index += 8)
        {
            const __m256 x0 = _mm256_loadu_ps(activation + index * 8);
            const __m256 x1 = _mm256_loadu_ps(activation + index * 8 + 8);
            const __m256 x2 = _mm256_loadu_ps(activation + index * 8 + 16);
            const __m256 x3 = _mm256_loadu_ps(activation + index * 8 + 24);
            const __m256 x4 = _mm256_loadu_ps(activation + index * 8 + 32);
            const __m256 x5 = _mm256_loadu_ps(activation + index * 8 + 40);
            const __m256 x6 = _mm256_loadu_ps(activation + index * 8 + 48);
            const __m256 x7 = _mm256_loadu_ps(activation + index * 8 + 56);

            pos_acc0 = _mm256_add_ps(pos_acc0, _mm256_and_ps(mask_to_ps(pos_bits[index]), x0));
            pos_acc0 = _mm256_add_ps(pos_acc0, _mm256_and_ps(mask_to_ps(pos_bits[index + 1]), x1));
            pos_acc1 = _mm256_add_ps(pos_acc1, _mm256_and_ps(mask_to_ps(pos_bits[index + 2]), x2));
            pos_acc1 = _mm256_add_ps(pos_acc1, _mm256_and_ps(mask_to_ps(pos_bits[index + 3]), x3));
            pos_acc2 = _mm256_add_ps(pos_acc2, _mm256_and_ps(mask_to_ps(pos_bits[index + 4]), x4));
            pos_acc2 = _mm256_add_ps(pos_acc2, _mm256_and_ps(mask_to_ps(pos_bits[index + 5]), x5));
            pos_acc3 = _mm256_add_ps(pos_acc3, _mm256_and_ps(mask_to_ps(pos_bits[index + 6]), x6));
            pos_acc3 = _mm256_add_ps(pos_acc3, _mm256_and_ps(mask_to_ps(pos_bits[index + 7]), x7));

            neg_acc0 = _mm256_add_ps(neg_acc0, _mm256_and_ps(mask_to_ps(neg_bits[index]), x0));
            neg_acc0 = _mm256_add_ps(neg_acc0, _mm256_and_ps(mask_to_ps(neg_bits[index + 1]), x1));
            neg_acc1 = _mm256_add_ps(neg_acc1, _mm256_and_ps(mask_to_ps(neg_bits[index + 2]), x2));
            neg_acc1 = _mm256_add_ps(neg_acc1, _mm256_and_ps(mask_to_ps(neg_bits[index + 3]), x3));
            neg_acc2 = _mm256_add_ps(neg_acc2, _mm256_and_ps(mask_to_ps(neg_bits[index + 4]), x4));
            neg_acc2 = _mm256_add_ps(neg_acc2, _mm256_and_ps(mask_to_ps(neg_bits[index + 5]), x5));
            neg_acc3 = _mm256_add_ps(neg_acc3, _mm256_and_ps(mask_to_ps(neg_bits[index + 6]), x6));
            neg_acc3 = _mm256_add_ps(neg_acc3, _mm256_and_ps(mask_to_ps(neg_bits[index + 7]), x7));
        }

        float sum = horizontal_sum(pos_acc0) + horizontal_sum(pos_acc1) + horizontal_sum(pos_acc2) + horizontal_sum(pos_acc3) -
                    horizontal_sum(neg_acc0) - horizontal_sum(neg_acc1) - horizontal_sum(neg_acc2) - horizontal_sum(neg_acc3);
        for (; index < packed_bytes; ++index)
        {
            const __m256 x = _mm256_loadu_ps(activation + index * 8);
            sum += horizontal_sum(_mm256_and_ps(mask_to_ps(pos_bits[index]), x));
            sum -= horizontal_sum(_mm256_and_ps(mask_to_ps(neg_bits[index]), x));
        }
        return sum;
    }

} // namespace ternary
*/

