/* SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Copyright:
 *   2020      Evan Nemerson <evan@nemerson.com>
 *   2024      Guation <guation@guation.cn>
 */

#define SIMDE_TEST_X86_AVX512_INSN storeu

#include <test/x86/avx512/test-avx512.h>
#include <simde/x86/avx512/storeu.h>

static int
test_simde_mm256_storeu_epi8 (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int8_t a[32];
    const int8_t r[32];
  } test_vec[] = {
    { {  INT8_C(  75),  INT8_C(  44),  INT8_C( 120), -INT8_C(  77), -INT8_C(  11), -INT8_C(  54), -INT8_C( 120), -INT8_C(  90),
         INT8_C(  76), -INT8_C( 120),  INT8_C(  18),  INT8_C( 102), -INT8_C(  27),  INT8_C( 105), -INT8_C(  91), -INT8_C(  67),
        -INT8_C( 127), -INT8_C(  90), -INT8_C(  52), -INT8_C(   1), -INT8_C( 106),  INT8_C(  47), -INT8_C( 109), -INT8_C( 117),
        -INT8_C(  83), -INT8_C(  57),  INT8_C(   1),  INT8_C( 120), -INT8_C(  46),  INT8_C(  84), -INT8_C(  19),  INT8_C(  30) },
      {  INT8_C(  75),  INT8_C(  44),  INT8_C( 120), -INT8_C(  77), -INT8_C(  11), -INT8_C(  54), -INT8_C( 120), -INT8_C(  90),
         INT8_C(  76), -INT8_C( 120),  INT8_C(  18),  INT8_C( 102), -INT8_C(  27),  INT8_C( 105), -INT8_C(  91), -INT8_C(  67),
        -INT8_C( 127), -INT8_C(  90), -INT8_C(  52), -INT8_C(   1), -INT8_C( 106),  INT8_C(  47), -INT8_C( 109), -INT8_C( 117),
        -INT8_C(  83), -INT8_C(  57),  INT8_C(   1),  INT8_C( 120), -INT8_C(  46),  INT8_C(  84), -INT8_C(  19),  INT8_C(  30) } },
    { {      INT8_MIN,  INT8_C( 101), -INT8_C(  47),  INT8_C( 117),  INT8_C(  47),  INT8_C(  90),  INT8_C(  28),  INT8_C( 124),
        -INT8_C(  30),  INT8_C(  46), -INT8_C(  30), -INT8_C(  57), -INT8_C( 105), -INT8_C( 121), -INT8_C( 124),  INT8_C(  25),
         INT8_C(  45),  INT8_C(  80),  INT8_C(  24), -INT8_C(  61),      INT8_MAX, -INT8_C(  85),  INT8_C(  79),  INT8_C(  44),
         INT8_C( 114),  INT8_C(  80), -INT8_C(  92),  INT8_C(  69), -INT8_C(  91), -INT8_C( 110),  INT8_C(  99),  INT8_C(  37) },
      {      INT8_MIN,  INT8_C( 101), -INT8_C(  47),  INT8_C( 117),  INT8_C(  47),  INT8_C(  90),  INT8_C(  28),  INT8_C( 124),
        -INT8_C(  30),  INT8_C(  46), -INT8_C(  30), -INT8_C(  57), -INT8_C( 105), -INT8_C( 121), -INT8_C( 124),  INT8_C(  25),
         INT8_C(  45),  INT8_C(  80),  INT8_C(  24), -INT8_C(  61),      INT8_MAX, -INT8_C(  85),  INT8_C(  79),  INT8_C(  44),
         INT8_C( 114),  INT8_C(  80), -INT8_C(  92),  INT8_C(  69), -INT8_C(  91), -INT8_C( 110),  INT8_C(  99),  INT8_C(  37) } },
    { { -INT8_C(   9),  INT8_C(  52), -INT8_C( 101),  INT8_C(  39), -INT8_C( 114), -INT8_C(  73), -INT8_C(  93),  INT8_C( 113),
        -INT8_C(  27), -INT8_C( 123),  INT8_C(  56),  INT8_C( 124),  INT8_C(  13), -INT8_C(  67), -INT8_C( 107),  INT8_C(  58),
         INT8_C(  13), -INT8_C(  82), -INT8_C(   2), -INT8_C( 115),  INT8_C(  89),  INT8_C(  77), -INT8_C(  71), -INT8_C(  52),
        -INT8_C(  99),  INT8_C(  94),  INT8_C(  17),  INT8_C(  66), -INT8_C(  16),  INT8_C( 116),  INT8_C( 104), -INT8_C(  25) },
      { -INT8_C(   9),  INT8_C(  52), -INT8_C( 101),  INT8_C(  39), -INT8_C( 114), -INT8_C(  73), -INT8_C(  93),  INT8_C( 113),
        -INT8_C(  27), -INT8_C( 123),  INT8_C(  56),  INT8_C( 124),  INT8_C(  13), -INT8_C(  67), -INT8_C( 107),  INT8_C(  58),
         INT8_C(  13), -INT8_C(  82), -INT8_C(   2), -INT8_C( 115),  INT8_C(  89),  INT8_C(  77), -INT8_C(  71), -INT8_C(  52),
        -INT8_C(  99),  INT8_C(  94),  INT8_C(  17),  INT8_C(  66), -INT8_C(  16),  INT8_C( 116),  INT8_C( 104), -INT8_C(  25) } },
    { { -INT8_C(  88),  INT8_C(   3),  INT8_C(  14),  INT8_C(  55), -INT8_C(  70), -INT8_C(  79), -INT8_C(  88), -INT8_C(  97),
         INT8_C(  55), -INT8_C(  32),  INT8_C(  27),  INT8_C(  68), -INT8_C(  99), -INT8_C(  79),  INT8_C( 126), -INT8_C(  85),
         INT8_C(  95),  INT8_C( 124),  INT8_C(  56), -INT8_C(  72), -INT8_C(  55), -INT8_C(  15), -INT8_C( 124),  INT8_C( 103),
         INT8_C(  79), -INT8_C( 107), -INT8_C(  87),  INT8_C(  63),  INT8_C(   9),  INT8_C(  17),  INT8_C(  39), -INT8_C(  78) },
      { -INT8_C(  88),  INT8_C(   3),  INT8_C(  14),  INT8_C(  55), -INT8_C(  70), -INT8_C(  79), -INT8_C(  88), -INT8_C(  97),
         INT8_C(  55), -INT8_C(  32),  INT8_C(  27),  INT8_C(  68), -INT8_C(  99), -INT8_C(  79),  INT8_C( 126), -INT8_C(  85),
         INT8_C(  95),  INT8_C( 124),  INT8_C(  56), -INT8_C(  72), -INT8_C(  55), -INT8_C(  15), -INT8_C( 124),  INT8_C( 103),
         INT8_C(  79), -INT8_C( 107), -INT8_C(  87),  INT8_C(  63),  INT8_C(   9),  INT8_C(  17),  INT8_C(  39), -INT8_C(  78) } },
    { {  INT8_C(  20),  INT8_C(  53), -INT8_C(  23), -INT8_C(  50), -INT8_C(  25), -INT8_C( 111),  INT8_C( 109),  INT8_C(  30),
         INT8_C( 113), -INT8_C( 119),  INT8_C(  98),  INT8_C(  15),  INT8_C(  58), -INT8_C(  32), -INT8_C(  70), -INT8_C( 103),
         INT8_C(  93), -INT8_C(  14),  INT8_C(  81),  INT8_C(  38), -INT8_C(  29), -INT8_C(  42), -INT8_C( 115),  INT8_C(  51),
         INT8_C( 107),  INT8_C(  55),  INT8_C( 114),  INT8_C( 117),  INT8_C(  72), -INT8_C( 103),  INT8_C(  39),  INT8_C(  93) },
      {  INT8_C(  20),  INT8_C(  53), -INT8_C(  23), -INT8_C(  50), -INT8_C(  25), -INT8_C( 111),  INT8_C( 109),  INT8_C(  30),
         INT8_C( 113), -INT8_C( 119),  INT8_C(  98),  INT8_C(  15),  INT8_C(  58), -INT8_C(  32), -INT8_C(  70), -INT8_C( 103),
         INT8_C(  93), -INT8_C(  14),  INT8_C(  81),  INT8_C(  38), -INT8_C(  29), -INT8_C(  42), -INT8_C( 115),  INT8_C(  51),
         INT8_C( 107),  INT8_C(  55),  INT8_C( 114),  INT8_C( 117),  INT8_C(  72), -INT8_C( 103),  INT8_C(  39),  INT8_C(  93) } },
    { { -INT8_C(  49),  INT8_C(  16),  INT8_C(  43), -INT8_C(  74), -INT8_C(  95), -INT8_C( 103), -INT8_C(  44),  INT8_C(  18),
         INT8_C(  34),  INT8_C(  54),  INT8_C(  33),  INT8_C(  92),  INT8_C(  22), -INT8_C(  37), -INT8_C(  11),  INT8_C( 115),
        -INT8_C(  51),  INT8_C(  70), -INT8_C( 102), -INT8_C(  79),  INT8_C(  28),  INT8_C(  39), -INT8_C(  28), -INT8_C( 120),
         INT8_C(  94),  INT8_C(  86), -INT8_C(   3), -INT8_C(  89), -INT8_C(  16),  INT8_C(  36),  INT8_C(   4), -INT8_C(  65) },
      { -INT8_C(  49),  INT8_C(  16),  INT8_C(  43), -INT8_C(  74), -INT8_C(  95), -INT8_C( 103), -INT8_C(  44),  INT8_C(  18),
         INT8_C(  34),  INT8_C(  54),  INT8_C(  33),  INT8_C(  92),  INT8_C(  22), -INT8_C(  37), -INT8_C(  11),  INT8_C( 115),
        -INT8_C(  51),  INT8_C(  70), -INT8_C( 102), -INT8_C(  79),  INT8_C(  28),  INT8_C(  39), -INT8_C(  28), -INT8_C( 120),
         INT8_C(  94),  INT8_C(  86), -INT8_C(   3), -INT8_C(  89), -INT8_C(  16),  INT8_C(  36),  INT8_C(   4), -INT8_C(  65) } },
    { {  INT8_C(  52),  INT8_C(  47),  INT8_C( 117), -INT8_C(  43), -INT8_C(  56),  INT8_C(  73), -INT8_C(  25), -INT8_C(  22),
             INT8_MAX,  INT8_C(   9),  INT8_C(  70), -INT8_C( 107), -INT8_C(  28),  INT8_C(  59),  INT8_C(   9), -INT8_C(  78),
        -INT8_C( 126), -INT8_C(  93),  INT8_C(  99), -INT8_C(  98), -INT8_C(  54),  INT8_C(  71),  INT8_C(  38),  INT8_C(  41),
        -INT8_C(  99),  INT8_C(  35), -INT8_C(  48), -INT8_C( 115),  INT8_C(  71), -INT8_C(  44),  INT8_C(  76),  INT8_C( 123) },
      {  INT8_C(  52),  INT8_C(  47),  INT8_C( 117), -INT8_C(  43), -INT8_C(  56),  INT8_C(  73), -INT8_C(  25), -INT8_C(  22),
             INT8_MAX,  INT8_C(   9),  INT8_C(  70), -INT8_C( 107), -INT8_C(  28),  INT8_C(  59),  INT8_C(   9), -INT8_C(  78),
        -INT8_C( 126), -INT8_C(  93),  INT8_C(  99), -INT8_C(  98), -INT8_C(  54),  INT8_C(  71),  INT8_C(  38),  INT8_C(  41),
        -INT8_C(  99),  INT8_C(  35), -INT8_C(  48), -INT8_C( 115),  INT8_C(  71), -INT8_C(  44),  INT8_C(  76),  INT8_C( 123) } },
    { {  INT8_C(   3), -INT8_C(  63),  INT8_C(  80), -INT8_C(  52),  INT8_C(  10),  INT8_C(  56), -INT8_C(  74), -INT8_C( 119),
         INT8_C(  65), -INT8_C(   3),  INT8_C(  31),  INT8_C(  37),  INT8_C(  56),  INT8_C(  40), -INT8_C(  41), -INT8_C(  70),
        -INT8_C(  53),  INT8_C(  58),  INT8_C(  89), -INT8_C( 107), -INT8_C( 127),      INT8_MAX, -INT8_C(  66),  INT8_C(  31),
        -INT8_C(  93), -INT8_C( 114), -INT8_C(  84), -INT8_C(  22),  INT8_C(  98), -INT8_C(   7),  INT8_C( 102),  INT8_C( 102) },
      {  INT8_C(   3), -INT8_C(  63),  INT8_C(  80), -INT8_C(  52),  INT8_C(  10),  INT8_C(  56), -INT8_C(  74), -INT8_C( 119),
         INT8_C(  65), -INT8_C(   3),  INT8_C(  31),  INT8_C(  37),  INT8_C(  56),  INT8_C(  40), -INT8_C(  41), -INT8_C(  70),
        -INT8_C(  53),  INT8_C(  58),  INT8_C(  89), -INT8_C( 107), -INT8_C( 127),      INT8_MAX, -INT8_C(  66),  INT8_C(  31),
        -INT8_C(  93), -INT8_C( 114), -INT8_C(  84), -INT8_C(  22),  INT8_C(  98), -INT8_C(   7),  INT8_C( 102),  INT8_C( 102) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m256i a = simde_mm256_loadu_epi8(test_vec[i].a);
    int8_t r[sizeof(simde__m256i) / sizeof(int8_t)];
    simde_mm256_storeu_epi8(r, a);
    simde_assert_equal_vi8(sizeof(r) / sizeof(r[0]), r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    simde__m256i a = simde_test_x86_random_i8x32();
    int8_t r[sizeof(simde__m256i) / sizeof(int8_t)];
    simde_mm256_storeu_epi8(r, a);

    simde_test_x86_write_i8x32(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_codegen_write_vi8(2, sizeof(r) / sizeof(r[0]), r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm256_storeu_epi16 (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t a[16];
    const int16_t r[16];
  } test_vec[] = {
    { {  INT16_C( 27201), -INT16_C( 10931),  INT16_C( 26543), -INT16_C( 17697), -INT16_C(  5170), -INT16_C(  2379),  INT16_C( 24504), -INT16_C( 22914),
         INT16_C(  3457), -INT16_C( 18629), -INT16_C(  4523),  INT16_C(  1253), -INT16_C(  9921),  INT16_C( 31191),  INT16_C(  9211),  INT16_C( 15721) },
      {  INT16_C( 27201), -INT16_C( 10931),  INT16_C( 26543), -INT16_C( 17697), -INT16_C(  5170), -INT16_C(  2379),  INT16_C( 24504), -INT16_C( 22914),
         INT16_C(  3457), -INT16_C( 18629), -INT16_C(  4523),  INT16_C(  1253), -INT16_C(  9921),  INT16_C( 31191),  INT16_C(  9211),  INT16_C( 15721) } },
    { { -INT16_C( 18803),  INT16_C( 15634), -INT16_C(  3810), -INT16_C(  4873), -INT16_C( 21283), -INT16_C( 27166),  INT16_C( 24587), -INT16_C( 29381),
         INT16_C( 30573), -INT16_C( 15804),  INT16_C( 10597), -INT16_C( 23097), -INT16_C( 25086), -INT16_C(   482), -INT16_C( 30782),  INT16_C( 20283) },
      { -INT16_C( 18803),  INT16_C( 15634), -INT16_C(  3810), -INT16_C(  4873), -INT16_C( 21283), -INT16_C( 27166),  INT16_C( 24587), -INT16_C( 29381),
         INT16_C( 30573), -INT16_C( 15804),  INT16_C( 10597), -INT16_C( 23097), -INT16_C( 25086), -INT16_C(   482), -INT16_C( 30782),  INT16_C( 20283) } },
    { {  INT16_C( 19773),  INT16_C( 23436), -INT16_C( 31938),  INT16_C(  6983),  INT16_C( 10544),  INT16_C( 15281), -INT16_C(  4982), -INT16_C(  2104),
         INT16_C(  3171), -INT16_C( 13894), -INT16_C( 32458),  INT16_C( 14446), -INT16_C( 29665), -INT16_C(  7882),  INT16_C( 28947),  INT16_C( 20529) },
      {  INT16_C( 19773),  INT16_C( 23436), -INT16_C( 31938),  INT16_C(  6983),  INT16_C( 10544),  INT16_C( 15281), -INT16_C(  4982), -INT16_C(  2104),
         INT16_C(  3171), -INT16_C( 13894), -INT16_C( 32458),  INT16_C( 14446), -INT16_C( 29665), -INT16_C(  7882),  INT16_C( 28947),  INT16_C( 20529) } },
    { { -INT16_C( 16962), -INT16_C(   596), -INT16_C(  3263),  INT16_C( 28952), -INT16_C( 14051), -INT16_C( 22612),  INT16_C( 30134),  INT16_C(  6558),
         INT16_C( 22657), -INT16_C( 18462),  INT16_C( 20697), -INT16_C(  1552),  INT16_C(  9948), -INT16_C(  4134),  INT16_C(  2968),  INT16_C( 22080) },
      { -INT16_C( 16962), -INT16_C(   596), -INT16_C(  3263),  INT16_C( 28952), -INT16_C( 14051), -INT16_C( 22612),  INT16_C( 30134),  INT16_C(  6558),
         INT16_C( 22657), -INT16_C( 18462),  INT16_C( 20697), -INT16_C(  1552),  INT16_C(  9948), -INT16_C(  4134),  INT16_C(  2968),  INT16_C( 22080) } },
    { { -INT16_C(  4919),  INT16_C(  2643),  INT16_C( 27871), -INT16_C(   901),  INT16_C( 10037), -INT16_C(  5213),  INT16_C( 17052),  INT16_C(  7685),
        -INT16_C(  6246),  INT16_C( 29909), -INT16_C( 15048),  INT16_C(  5229),  INT16_C( 18412), -INT16_C( 31740),  INT16_C( 17491),  INT16_C(  7386) },
      { -INT16_C(  4919),  INT16_C(  2643),  INT16_C( 27871), -INT16_C(   901),  INT16_C( 10037), -INT16_C(  5213),  INT16_C( 17052),  INT16_C(  7685),
        -INT16_C(  6246),  INT16_C( 29909), -INT16_C( 15048),  INT16_C(  5229),  INT16_C( 18412), -INT16_C( 31740),  INT16_C( 17491),  INT16_C(  7386) } },
    { {  INT16_C( 11824),  INT16_C(  3878), -INT16_C( 24166), -INT16_C( 12532), -INT16_C( 20536),  INT16_C( 26043), -INT16_C( 16143), -INT16_C( 29565),
         INT16_C( 22695), -INT16_C(  8448),  INT16_C( 27934),  INT16_C(  2804), -INT16_C(  1868),  INT16_C(  1934),  INT16_C( 26684),  INT16_C( 27683) },
      {  INT16_C( 11824),  INT16_C(  3878), -INT16_C( 24166), -INT16_C( 12532), -INT16_C( 20536),  INT16_C( 26043), -INT16_C( 16143), -INT16_C( 29565),
         INT16_C( 22695), -INT16_C(  8448),  INT16_C( 27934),  INT16_C(  2804), -INT16_C(  1868),  INT16_C(  1934),  INT16_C( 26684),  INT16_C( 27683) } },
    { {  INT16_C( 18838),  INT16_C( 12411), -INT16_C( 30742), -INT16_C( 19712), -INT16_C( 17609),  INT16_C( 10264), -INT16_C( 25733),  INT16_C(  8884),
        -INT16_C( 19213),  INT16_C(  4354), -INT16_C(  2527), -INT16_C( 10725), -INT16_C( 22034),  INT16_C( 10973),  INT16_C(   274), -INT16_C( 22378) },
      {  INT16_C( 18838),  INT16_C( 12411), -INT16_C( 30742), -INT16_C( 19712), -INT16_C( 17609),  INT16_C( 10264), -INT16_C( 25733),  INT16_C(  8884),
        -INT16_C( 19213),  INT16_C(  4354), -INT16_C(  2527), -INT16_C( 10725), -INT16_C( 22034),  INT16_C( 10973),  INT16_C(   274), -INT16_C( 22378) } },
    { {  INT16_C(  4426),  INT16_C( 13785), -INT16_C(  9831), -INT16_C( 12056),  INT16_C(   148),  INT16_C(  4088), -INT16_C( 21093), -INT16_C( 29135),
         INT16_C( 13153), -INT16_C( 31840), -INT16_C( 17623),  INT16_C(  5977),  INT16_C( 13925),  INT16_C( 30529), -INT16_C( 10441), -INT16_C( 32225) },
      {  INT16_C(  4426),  INT16_C( 13785), -INT16_C(  9831), -INT16_C( 12056),  INT16_C(   148),  INT16_C(  4088), -INT16_C( 21093), -INT16_C( 29135),
         INT16_C( 13153), -INT16_C( 31840), -INT16_C( 17623),  INT16_C(  5977),  INT16_C( 13925),  INT16_C( 30529), -INT16_C( 10441), -INT16_C( 32225) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m256i a = simde_mm256_loadu_epi16(test_vec[i].a);
    int16_t r[sizeof(simde__m256i) / sizeof(int16_t)];
    simde_mm256_storeu_epi16(r, a);
    simde_assert_equal_vi16(sizeof(r) / sizeof(r[0]), r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    simde__m256i a = simde_test_x86_random_i16x16();
    int16_t r[sizeof(simde__m256i) / sizeof(int16_t)];
    simde_mm256_storeu_epi16(r, a);

    simde_test_x86_write_i16x16(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_codegen_write_vi16(2, sizeof(r) / sizeof(r[0]), r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm256_mask_storeu_epi16 (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const simde__mmask16 k;
    const int16_t a[16];
    const int16_t r0[16];
    const int16_t r1[16];
  } test_vec[] = {
    { UINT16_C(32467),
      { -INT16_C( 20890),  INT16_C( 13558), -INT16_C(  7888), -INT16_C( 18506),  INT16_C(  3160),  INT16_C(  2708),  INT16_C( 13770),  INT16_C( 30334),
         INT16_C( 18860),  INT16_C(  2237),  INT16_C( 20703),  INT16_C( 11606),  INT16_C( 25991), -INT16_C( 20652),  INT16_C( 10190),  INT16_C( 13357) },
      {  INT16_C(  9173),  INT16_C(  1640),  INT16_C(  7684),  INT16_C( 23741),  INT16_C( 20778), -INT16_C(  2970), -INT16_C(  6778),  INT16_C( 13163),
         INT16_C( 10286),  INT16_C(  3387), -INT16_C( 28296),  INT16_C(    58), -INT16_C( 28937), -INT16_C( 14929), -INT16_C(  9034), -INT16_C( 29703) },
      { -INT16_C( 20890),  INT16_C( 13558),  INT16_C(  7684),  INT16_C( 23741),  INT16_C(  3160), -INT16_C(  2970),  INT16_C( 13770),  INT16_C( 30334),
         INT16_C( 10286),  INT16_C(  2237),  INT16_C( 20703),  INT16_C( 11606),  INT16_C( 25991), -INT16_C( 20652),  INT16_C( 10190), -INT16_C( 29703) } },
    { UINT16_C(24832),
      {  INT16_C(  1169),  INT16_C( 20095), -INT16_C( 21919), -INT16_C( 14432),  INT16_C(  9886),  INT16_C(  2476), -INT16_C(  9639), -INT16_C( 27598),
        -INT16_C( 21784),  INT16_C(  8742),  INT16_C(  7594),  INT16_C( 22961),  INT16_C( 26594), -INT16_C(  9418),  INT16_C( 14066), -INT16_C( 31684) },
      { -INT16_C( 17606), -INT16_C( 25646),  INT16_C( 29285),  INT16_C(  1123),  INT16_C(  3993), -INT16_C(  3571),  INT16_C( 16362), -INT16_C( 11641),
        -INT16_C( 21014), -INT16_C( 27404), -INT16_C( 23094), -INT16_C( 21266),  INT16_C(  9228), -INT16_C(   121), -INT16_C( 15526), -INT16_C( 27517) },
      { -INT16_C( 17606), -INT16_C( 25646),  INT16_C( 29285),  INT16_C(  1123),  INT16_C(  3993), -INT16_C(  3571),  INT16_C( 16362), -INT16_C( 11641),
        -INT16_C( 21784), -INT16_C( 27404), -INT16_C( 23094), -INT16_C( 21266),  INT16_C(  9228), -INT16_C(  9418),  INT16_C( 14066), -INT16_C( 27517) } },
    { UINT16_C(21886),
      { -INT16_C(  7120), -INT16_C( 27704),  INT16_C( 25064), -INT16_C(  2654), -INT16_C( 29613), -INT16_C(  9675),  INT16_C(  8030),  INT16_C( 21383),
         INT16_C( 20915), -INT16_C( 24072),  INT16_C(  1533), -INT16_C( 31547),  INT16_C(  7940), -INT16_C( 30905), -INT16_C( 14668), -INT16_C(  6948) },
      { -INT16_C( 23382), -INT16_C( 28041),  INT16_C(  6405),  INT16_C( 22919), -INT16_C( 17242),  INT16_C(  1075), -INT16_C( 17445), -INT16_C( 28841),
         INT16_C( 20492),  INT16_C(  2608), -INT16_C(  2475),  INT16_C( 22926), -INT16_C( 10731), -INT16_C( 13856), -INT16_C( 17252),  INT16_C( 18093) },
      { -INT16_C( 23382), -INT16_C( 27704),  INT16_C( 25064), -INT16_C(  2654), -INT16_C( 29613), -INT16_C(  9675),  INT16_C(  8030), -INT16_C( 28841),
         INT16_C( 20915),  INT16_C(  2608),  INT16_C(  1533),  INT16_C( 22926),  INT16_C(  7940), -INT16_C( 13856), -INT16_C( 14668),  INT16_C( 18093) } },
    { UINT16_C( 9313),
      {  INT16_C( 26328),  INT16_C( 24382), -INT16_C(  6977), -INT16_C(  3300), -INT16_C(  2072),  INT16_C( 16558), -INT16_C( 17786), -INT16_C( 18544),
        -INT16_C(  6716),  INT16_C( 21421), -INT16_C( 15810),  INT16_C(  7721), -INT16_C( 14964),  INT16_C( 14810),  INT16_C( 15115), -INT16_C(  7330) },
      { -INT16_C( 25438),  INT16_C( 24898),  INT16_C( 24192),  INT16_C( 26708),  INT16_C(   598), -INT16_C(  9048),  INT16_C( 14525), -INT16_C( 32365),
         INT16_C( 16413),  INT16_C( 23508), -INT16_C(   765), -INT16_C( 28807),  INT16_C( 21698), -INT16_C( 12856),  INT16_C(  9871),  INT16_C( 12720) },
      {  INT16_C( 26328),  INT16_C( 24898),  INT16_C( 24192),  INT16_C( 26708),  INT16_C(   598),  INT16_C( 16558), -INT16_C( 17786), -INT16_C( 32365),
         INT16_C( 16413),  INT16_C( 23508), -INT16_C( 15810), -INT16_C( 28807),  INT16_C( 21698),  INT16_C( 14810),  INT16_C(  9871),  INT16_C( 12720) } },
    { UINT16_C(62402),
      {  INT16_C( 17043), -INT16_C(  6319), -INT16_C( 22613),  INT16_C( 21482), -INT16_C( 22652),  INT16_C(  6028), -INT16_C( 22232), -INT16_C(   680),
         INT16_C( 23301),  INT16_C( 32506), -INT16_C( 16918), -INT16_C( 19758),  INT16_C( 25226),  INT16_C( 15321), -INT16_C( 25709),  INT16_C(  9774) },
      {  INT16_C( 32734), -INT16_C( 30450), -INT16_C(  2009), -INT16_C( 21540),  INT16_C( 26783), -INT16_C( 14398),  INT16_C(  6674),  INT16_C(  6084),
        -INT16_C( 16523),  INT16_C( 24469),  INT16_C( 26748),  INT16_C(  1554), -INT16_C(  5174),  INT16_C( 23873),  INT16_C( 28550),  INT16_C( 25732) },
      {  INT16_C( 32734), -INT16_C(  6319), -INT16_C(  2009), -INT16_C( 21540),  INT16_C( 26783), -INT16_C( 14398), -INT16_C( 22232), -INT16_C(   680),
         INT16_C( 23301),  INT16_C( 32506),  INT16_C( 26748),  INT16_C(  1554),  INT16_C( 25226),  INT16_C( 15321), -INT16_C( 25709),  INT16_C(  9774) } },
    { UINT16_C(37615),
      {  INT16_C(  5869), -INT16_C( 13686),  INT16_C( 10689), -INT16_C( 31950),  INT16_C( 17648), -INT16_C( 19042),  INT16_C(  4955), -INT16_C(  3724),
        -INT16_C(  3981), -INT16_C( 31399),  INT16_C(  9206),  INT16_C( 14448), -INT16_C(  2432),  INT16_C(  1191), -INT16_C( 27045),  INT16_C( 18582) },
      {  INT16_C(  8364),  INT16_C( 27922),  INT16_C( 17737),  INT16_C( 15089), -INT16_C( 28791), -INT16_C(  6673),  INT16_C( 25506),  INT16_C(  5590),
         INT16_C( 12115),  INT16_C( 18842),  INT16_C(  2642), -INT16_C( 11647),  INT16_C( 10497),  INT16_C( 23767),  INT16_C( 28095),  INT16_C( 27812) },
      {  INT16_C(  5869), -INT16_C( 13686),  INT16_C( 10689), -INT16_C( 31950), -INT16_C( 28791), -INT16_C( 19042),  INT16_C(  4955), -INT16_C(  3724),
         INT16_C( 12115), -INT16_C( 31399),  INT16_C(  2642), -INT16_C( 11647), -INT16_C(  2432),  INT16_C( 23767),  INT16_C( 28095),  INT16_C( 18582) } },
    { UINT16_C(46990),
      { -INT16_C( 10279), -INT16_C( 13572), -INT16_C( 31471),  INT16_C(    89), -INT16_C(   918),  INT16_C( 16483), -INT16_C( 18927), -INT16_C( 21393),
        -INT16_C( 16128), -INT16_C( 32330), -INT16_C( 18540),  INT16_C( 27562),  INT16_C( 27155), -INT16_C( 18216),  INT16_C( 26326), -INT16_C( 20625) },
      {  INT16_C( 27454),  INT16_C( 20346), -INT16_C( 11280),  INT16_C( 23376), -INT16_C( 19505), -INT16_C(  7781),  INT16_C(  2922),  INT16_C( 27277),
         INT16_C( 17356),  INT16_C( 24811), -INT16_C( 26885),  INT16_C(  3787), -INT16_C( 23552), -INT16_C( 10554),  INT16_C( 13578),  INT16_C( 18565) },
      {  INT16_C( 27454), -INT16_C( 13572), -INT16_C( 31471),  INT16_C(    89), -INT16_C( 19505), -INT16_C(  7781),  INT16_C(  2922), -INT16_C( 21393),
        -INT16_C( 16128), -INT16_C( 32330), -INT16_C( 18540),  INT16_C(  3787),  INT16_C( 27155), -INT16_C( 18216),  INT16_C( 13578), -INT16_C( 20625) } },
    { UINT16_C(65440),
      { -INT16_C( 28264), -INT16_C(  5933), -INT16_C( 23828), -INT16_C( 30821),  INT16_C(  1411),  INT16_C(  4242),  INT16_C( 24431),  INT16_C( 23380),
         INT16_C( 20415), -INT16_C( 29711), -INT16_C(  3747),  INT16_C(  9263),  INT16_C( 14791),  INT16_C( 19545), -INT16_C(  1406),  INT16_C(  6732) },
      {  INT16_C(  8075),  INT16_C( 30466), -INT16_C( 25151),  INT16_C( 17918), -INT16_C( 28253),  INT16_C(  4693), -INT16_C( 22032), -INT16_C( 20627),
         INT16_C( 24312),  INT16_C( 22074),  INT16_C( 26959),  INT16_C(  5754), -INT16_C( 11357),  INT16_C(  9571), -INT16_C( 20531),  INT16_C( 22591) },
      {  INT16_C(  8075),  INT16_C( 30466), -INT16_C( 25151),  INT16_C( 17918), -INT16_C( 28253),  INT16_C(  4242), -INT16_C( 22032),  INT16_C( 23380),
         INT16_C( 20415), -INT16_C( 29711), -INT16_C(  3747),  INT16_C(  9263),  INT16_C( 14791),  INT16_C( 19545), -INT16_C(  1406),  INT16_C(  6732) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m256i a = simde_mm256_loadu_epi16(test_vec[i].a);
    simde__m256i r0 = simde_mm256_loadu_epi16(test_vec[i].r0);
    int16_t r1[sizeof(simde__m256i) / sizeof(int16_t)];
    simde_mm256_storeu_epi16(r1, r0);
    simde_mm256_mask_storeu_epi16(r1, test_vec[i].k, a);
    simde_assert_equal_vi16(sizeof(r1) / sizeof(r1[0]), r1, test_vec[i].r1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    simde__mmask16 k = simde_test_x86_random_mmask16();
    simde__m256i a = simde_test_x86_random_i16x16();
    simde__m256i r0 = simde_test_x86_random_i16x16();
    int16_t r1[sizeof(simde__m256i) / sizeof(int16_t)];
    simde_mm256_storeu_epi16(r1, r0);
    simde_mm256_mask_storeu_epi16(r1, k, a);

    simde_test_x86_write_mmask16(2, k, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i16x16(2, a, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_i16x16(2, r0, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_codegen_write_vi16(2, sizeof(r1) / sizeof(r1[0]), r1, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm256_storeu_epi32 (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t a[8];
    const int32_t r[8];
  } test_vec[] = {
    { { -INT32_C(  2101872407),  INT32_C(  1699913681),  INT32_C(   980699807),  INT32_C(  1506322167),  INT32_C(    64776409), -INT32_C(  1994771164), -INT32_C(  1560257429),  INT32_C(   472194867) },
      { -INT32_C(  2101872407),  INT32_C(  1699913681),  INT32_C(   980699807),  INT32_C(  1506322167),  INT32_C(    64776409), -INT32_C(  1994771164), -INT32_C(  1560257429),  INT32_C(   472194867) } },
    { { -INT32_C(   375464936),  INT32_C(   441446523),  INT32_C(   844415803),  INT32_C(  1133190249), -INT32_C(  1455003771),  INT32_C(   137519260), -INT32_C(   257215812),  INT32_C(  1762447441) },
      { -INT32_C(   375464936),  INT32_C(   441446523),  INT32_C(   844415803),  INT32_C(  1133190249), -INT32_C(  1455003771),  INT32_C(   137519260), -INT32_C(   257215812),  INT32_C(  1762447441) } },
    { {  INT32_C(   659794860), -INT32_C(   700341605), -INT32_C(   821455515),  INT32_C(   907187377), -INT32_C(  1730193156),  INT32_C(  1973424568), -INT32_C(  1788523709), -INT32_C(   939626213) },
      {  INT32_C(   659794860), -INT32_C(   700341605), -INT32_C(   821455515),  INT32_C(   907187377), -INT32_C(  1730193156),  INT32_C(  1973424568), -INT32_C(  1788523709), -INT32_C(   939626213) } },
    { { -INT32_C(  1192341220),  INT32_C(  1502490611),  INT32_C(  1982371780),  INT32_C(   682375724),  INT32_C(  1254132882), -INT32_C(   507551331), -INT32_C(   931781460), -INT32_C(  1299221354) },
      { -INT32_C(  1192341220),  INT32_C(  1502490611),  INT32_C(  1982371780),  INT32_C(   682375724),  INT32_C(  1254132882), -INT32_C(   507551331), -INT32_C(   931781460), -INT32_C(  1299221354) } },
    { { -INT32_C(  1184203066),  INT32_C(  1913846189), -INT32_C(  1125631344),  INT32_C(   115643508), -INT32_C(  1101945568), -INT32_C(  1298198522), -INT32_C(   881191627),  INT32_C(  1333594761) },
      { -INT32_C(  1184203066),  INT32_C(  1913846189), -INT32_C(  1125631344),  INT32_C(   115643508), -INT32_C(  1101945568), -INT32_C(  1298198522), -INT32_C(   881191627),  INT32_C(  1333594761) } },
    { {  INT32_C(   889841800),  INT32_C(  1906777057), -INT32_C(   902918314),  INT32_C(  1154552356), -INT32_C(  1123933513),  INT32_C(  1735434546),  INT32_C(  1077078710),  INT32_C(  2089791732) },
      {  INT32_C(   889841800),  INT32_C(  1906777057), -INT32_C(   902918314),  INT32_C(  1154552356), -INT32_C(  1123933513),  INT32_C(  1735434546),  INT32_C(  1077078710),  INT32_C(  2089791732) } },
    { {  INT32_C(  2041747608),  INT32_C(   183130548),  INT32_C(   232003817), -INT32_C(   497965781), -INT32_C(    90155833), -INT32_C(  1402924811), -INT32_C(   269708038), -INT32_C(   596935868) },
      {  INT32_C(  2041747608),  INT32_C(   183130548),  INT32_C(   232003817), -INT32_C(   497965781), -INT32_C(    90155833), -INT32_C(  1402924811), -INT32_C(   269708038), -INT32_C(   596935868) } },
    { { -INT32_C(   933946092),  INT32_C(  1624391543), -INT32_C(  2089965992),  INT32_C(   325434956),  INT32_C(   135070994),  INT32_C(   280260373),  INT32_C(  1207934979),  INT32_C(   790850075) },
      { -INT32_C(   933946092),  INT32_C(  1624391543), -INT32_C(  2089965992),  INT32_C(   325434956),  INT32_C(   135070994),  INT32_C(   280260373),  INT32_C(  1207934979),  INT32_C(   790850075) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m256i a = simde_mm256_loadu_epi32(test_vec[i].a);
    int32_t r[sizeof(simde__m256i) / sizeof(int32_t)];
    simde_mm256_storeu_epi32(r, a);
    simde_assert_equal_vi32(sizeof(r) / sizeof(r[0]), r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    simde__m256i a = simde_test_x86_random_i32x8();
    int32_t r[sizeof(simde__m256i) / sizeof(int32_t)];
    simde_mm256_storeu_epi32(r, a);

    simde_test_x86_write_i32x8(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_codegen_write_vi32(2, sizeof(r) / sizeof(r[0]), r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm256_storeu_epi64 (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[4];
    const int64_t r[4];
  } test_vec[] = {
    { { -INT64_C( 8520414791212126978),  INT64_C( 6470760737637609380),  INT64_C( 7372765298930470642),  INT64_C( 9137132392545280597) },
      { -INT64_C( 8520414791212126978),  INT64_C( 6470760737637609380),  INT64_C( 7372765298930470642),  INT64_C( 9137132392545280597) } },
    { { -INT64_C( 6181695135723249974),  INT64_C( 6582275189149771364),  INT64_C( 1378027538322451646), -INT64_C( 1793282446973829346) },
      { -INT64_C( 6181695135723249974),  INT64_C( 6582275189149771364),  INT64_C( 1378027538322451646), -INT64_C( 1793282446973829346) } },
    { {  INT64_C( 3047472147532648602),  INT64_C( 1960335077399948657),  INT64_C( 8772998174208204782), -INT64_C( 9045984628089229966) },
      {  INT64_C( 3047472147532648602),  INT64_C( 1960335077399948657),  INT64_C( 8772998174208204782), -INT64_C( 9045984628089229966) } },
    { {  INT64_C( 5166874341197089733),  INT64_C( 5643948629755485914), -INT64_C( 8302182390395426828), -INT64_C( 2733242877560954714) },
      {  INT64_C( 5166874341197089733),  INT64_C( 5643948629755485914), -INT64_C( 8302182390395426828), -INT64_C( 2733242877560954714) } },
    { {  INT64_C( 8843408995216798621),  INT64_C( 2640521362198879223), -INT64_C(   84940192658251989),  INT64_C( 2164997219794349596) },
      {  INT64_C( 8843408995216798621),  INT64_C( 2640521362198879223), -INT64_C(   84940192658251989),  INT64_C( 2164997219794349596) } },
    { {  INT64_C( 9217545211248472017), -INT64_C( 3712237534943208409),  INT64_C( 5003750052667256346),  INT64_C( 7741830259258359250) },
      {  INT64_C( 9217545211248472017), -INT64_C( 3712237534943208409),  INT64_C( 5003750052667256346),  INT64_C( 7741830259258359250) } },
    { {  INT64_C( 8501044911504360177),  INT64_C( 7336723655508676452), -INT64_C( 7375690268599065803), -INT64_C( 8265410862250190267) },
      {  INT64_C( 8501044911504360177),  INT64_C( 7336723655508676452), -INT64_C( 7375690268599065803), -INT64_C( 8265410862250190267) } },
    { {  INT64_C( 5990350384227821662),  INT64_C( 7601841099293388239), -INT64_C( 1914473353548638451),  INT64_C( 1562487073980897926) },
      {  INT64_C( 5990350384227821662),  INT64_C( 7601841099293388239), -INT64_C( 1914473353548638451),  INT64_C( 1562487073980897926) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m256i a = simde_mm256_loadu_epi64(test_vec[i].a);
    int64_t r[sizeof(simde__m256i) / sizeof(int64_t)];
    simde_mm256_storeu_epi64(r, a);
    simde_assert_equal_vi64(sizeof(r) / sizeof(r[0]), r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    simde__m256i a = simde_test_x86_random_i64x4();
    int64_t r[sizeof(simde__m256i) / sizeof(int64_t)];
    simde_mm256_storeu_epi64(r, a);

    simde_test_x86_write_i64x4(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_codegen_write_vi64(2, sizeof(r) / sizeof(r[0]), r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm512_storeu_si512 (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[8];
    const int64_t r[8];
  } test_vec[] = {
    { {  INT64_C( 5551297066613610080),  INT64_C( 2769370642455107003),  INT64_C( 8911450648572637446),  INT64_C( 8709882614583967283),
        -INT64_C(  981322687331038258),  INT64_C( 2111787002933628730), -INT64_C( 1232246065541749179),  INT64_C(  899623978740254444) },
      {  INT64_C( 5551297066613610080),  INT64_C( 2769370642455107003),  INT64_C( 8911450648572637446),  INT64_C( 8709882614583967283),
        -INT64_C(  981322687331038258),  INT64_C( 2111787002933628730), -INT64_C( 1232246065541749179),  INT64_C(  899623978740254444) } },
    { { -INT64_C(  769973994054019545),  INT64_C( 5621734825629873616), -INT64_C( 6774927093785852336),  INT64_C( 8678191640414074405),
        -INT64_C( 2312836934179320265), -INT64_C( 4615122616548195172),  INT64_C( 8117391204880657488), -INT64_C( 6646114117179192326) },
      { -INT64_C(  769973994054019545),  INT64_C( 5621734825629873616), -INT64_C( 6774927093785852336),  INT64_C( 8678191640414074405),
        -INT64_C( 2312836934179320265), -INT64_C( 4615122616548195172),  INT64_C( 8117391204880657488), -INT64_C( 6646114117179192326) } },
    { {  INT64_C( 6827313836062055377), -INT64_C( 6940056578053339775),  INT64_C( 4062328498057148836),  INT64_C( 5162613319030551445),
         INT64_C( 4180720034842622948), -INT64_C( 6604008781415577493), -INT64_C( 2393114233981895086),  INT64_C( 8855348549745855783) },
      {  INT64_C( 6827313836062055377), -INT64_C( 6940056578053339775),  INT64_C( 4062328498057148836),  INT64_C( 5162613319030551445),
         INT64_C( 4180720034842622948), -INT64_C( 6604008781415577493), -INT64_C( 2393114233981895086),  INT64_C( 8855348549745855783) } },
    { {  INT64_C( 6679442916376579007),  INT64_C(  563354252334259245),  INT64_C( 9193079773018482029), -INT64_C( 5206441577905622098),
         INT64_C( 3627896915669016587),  INT64_C( 5438714331474718397),  INT64_C( 2409027175853396630), -INT64_C( 7514213599801865463) },
      {  INT64_C( 6679442916376579007),  INT64_C(  563354252334259245),  INT64_C( 9193079773018482029), -INT64_C( 5206441577905622098),
         INT64_C( 3627896915669016587),  INT64_C( 5438714331474718397),  INT64_C( 2409027175853396630), -INT64_C( 7514213599801865463) } },
    { { -INT64_C( 8701538556783106042),  INT64_C( 7315099028607835482),  INT64_C( 2601606259657704837), -INT64_C( 6021068993116929970),
         INT64_C( 5338724821846394053),  INT64_C(  632482533526933394),  INT64_C( 3227704201039649676), -INT64_C( 7787156077433917285) },
      { -INT64_C( 8701538556783106042),  INT64_C( 7315099028607835482),  INT64_C( 2601606259657704837), -INT64_C( 6021068993116929970),
         INT64_C( 5338724821846394053),  INT64_C(  632482533526933394),  INT64_C( 3227704201039649676), -INT64_C( 7787156077433917285) } },
    { {  INT64_C( 3293600942643127930), -INT64_C( 3759566431926142904), -INT64_C( 8395766407757573007),  INT64_C( 8327362393387401775),
         INT64_C( 3686921886725223265), -INT64_C( 9163186904764597813), -INT64_C( 5322888039829059418),  INT64_C(  809596429515615951) },
      {  INT64_C( 3293600942643127930), -INT64_C( 3759566431926142904), -INT64_C( 8395766407757573007),  INT64_C( 8327362393387401775),
         INT64_C( 3686921886725223265), -INT64_C( 9163186904764597813), -INT64_C( 5322888039829059418),  INT64_C(  809596429515615951) } },
    { {  INT64_C( 8652876881953495798), -INT64_C( 8324244311700794134),  INT64_C( 7484212472235313824), -INT64_C( 3826252275745058767),
         INT64_C(  696581432120503685),  INT64_C( 2614477592715275477), -INT64_C( 2322123828474713401), -INT64_C( 1849395731149163080) },
      {  INT64_C( 8652876881953495798), -INT64_C( 8324244311700794134),  INT64_C( 7484212472235313824), -INT64_C( 3826252275745058767),
         INT64_C(  696581432120503685),  INT64_C( 2614477592715275477), -INT64_C( 2322123828474713401), -INT64_C( 1849395731149163080) } },
    { { -INT64_C( 7442546088993029746), -INT64_C(  101159233921959440), -INT64_C( 4353816406582762746),  INT64_C( 9034402200944528920),
         INT64_C( 3171062162695813541), -INT64_C(  590471889900513894),  INT64_C( 7487586559880604443), -INT64_C( 9080874382890354546) },
      { -INT64_C( 7442546088993029746), -INT64_C(  101159233921959440), -INT64_C( 4353816406582762746),  INT64_C( 9034402200944528920),
         INT64_C( 3171062162695813541), -INT64_C(  590471889900513894),  INT64_C( 7487586559880604443), -INT64_C( 9080874382890354546) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i a = simde_mm512_loadu_epi64(test_vec[i].a);
    int64_t r[sizeof(simde__m512i) / sizeof(int64_t)];
    simde_mm512_storeu_si512(r, a);
    simde_assert_equal_vi64(sizeof(r) / sizeof(r[0]), r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    simde__m512i a = simde_test_x86_random_i64x8();
    int64_t r[sizeof(simde__m512i) / sizeof(int64_t)];
    simde_mm512_storeu_si512(r, a);

    simde_test_x86_write_i64x8(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_codegen_write_vi64(2, sizeof(r) / sizeof(r[0]), r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm512_mask_storeu_epi8 (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int8_t a[64];
    simde__mmask64 k;
    const int8_t r0[64];
    const int8_t r1[64];
  } test_vec[] = {
    { {  INT8_C(  82),  INT8_C(  77), -INT8_C(  85), -INT8_C(  25), -INT8_C(  16), -INT8_C(  37), -INT8_C(  74), -INT8_C(  86),
         INT8_C(  64),  INT8_C( 102), -INT8_C(   2), -INT8_C(   4),  INT8_C(  23),  INT8_C( 115), -INT8_C(  75),  INT8_C( 123),
         INT8_C( 125),  INT8_C(   6),  INT8_C( 114), -INT8_C(  44), -INT8_C(  99), -INT8_C(   4),  INT8_C(   4),  INT8_C(  27),
        -INT8_C(  66),  INT8_C(  62), -INT8_C( 123), -INT8_C(  65), -INT8_C( 113), -INT8_C(  76), -INT8_C(   7), -INT8_C(  30),
         INT8_C(   1), -INT8_C(  92), -INT8_C(  55), -INT8_C(  15),      INT8_MAX,      INT8_MIN, -INT8_C( 101), -INT8_C(  64),
        -INT8_C(  26), -INT8_C( 103), -INT8_C(  68), -INT8_C(   3),  INT8_C(  12),  INT8_C( 114),  INT8_C( 120), -INT8_C( 118),
         INT8_C( 120), -INT8_C(  21),  INT8_C(  94),  INT8_C(  21), -INT8_C(  25),  INT8_C(  98),  INT8_C(  48), -INT8_C(  90),
        -INT8_C(  96), -INT8_C(  75),  INT8_C( 101),  INT8_C(  48),  INT8_C( 105),  INT8_C(  94),  INT8_C(  18),  INT8_C( 106) },
      UINT64_C( 4774650527367289603),
      { -INT8_C( 112), -INT8_C(   1),  INT8_C(  63), -INT8_C(  99),  INT8_C( 113), -INT8_C(  72),  INT8_C(  39), -INT8_C(  23),
        -INT8_C(  93), -INT8_C( 123), -INT8_C(   2), -INT8_C( 118), -INT8_C(  25),  INT8_C(  47),  INT8_C(  48), -INT8_C( 120),
        -INT8_C(  28), -INT8_C( 107), -INT8_C(  72),  INT8_C(  78), -INT8_C(  12), -INT8_C(  54), -INT8_C(  72), -INT8_C(   9),
        -INT8_C(  91),  INT8_C(  20),  INT8_C( 121),  INT8_C(   1),  INT8_C(  11), -INT8_C(  68),  INT8_C(  67), -INT8_C( 101),
        -INT8_C(  69), -INT8_C( 126),  INT8_C(  56),  INT8_C(  44),  INT8_C(  58),  INT8_C(  95),  INT8_C(  21), -INT8_C(  35),
        -INT8_C(  28),  INT8_C(  19),  INT8_C( 104), -INT8_C(  52),  INT8_C(  66), -INT8_C( 104),  INT8_C(  84),  INT8_C(  39),
         INT8_C(  46),  INT8_C(  12),  INT8_C( 117),  INT8_C(  34), -INT8_C(  42),  INT8_C(  45),  INT8_C(  25),  INT8_C( 123),
         INT8_C(  65), -INT8_C( 110),  INT8_C( 124),  INT8_C(  76),  INT8_C(  78), -INT8_C(  65), -INT8_C(  24),  INT8_C(   9) },
      {  INT8_C(  82),  INT8_C(  77),  INT8_C(  63), -INT8_C(  99),  INT8_C( 113), -INT8_C(  72),  INT8_C(  39), -INT8_C(  23),
         INT8_C(  64),  INT8_C( 102), -INT8_C(   2), -INT8_C(   4),  INT8_C(  23),  INT8_C(  47), -INT8_C(  75),  INT8_C( 123),
         INT8_C( 125),  INT8_C(   6), -INT8_C(  72), -INT8_C(  44), -INT8_C(  99), -INT8_C(  54),  INT8_C(   4), -INT8_C(   9),
        -INT8_C(  91),  INT8_C(  62),  INT8_C( 121),  INT8_C(   1),  INT8_C(  11), -INT8_C(  68),  INT8_C(  67), -INT8_C(  30),
         INT8_C(   1), -INT8_C(  92),  INT8_C(  56), -INT8_C(  15),      INT8_MAX,  INT8_C(  95), -INT8_C( 101), -INT8_C(  35),
        -INT8_C(  26), -INT8_C( 103), -INT8_C(  68), -INT8_C(  52),  INT8_C(  12),  INT8_C( 114),  INT8_C( 120), -INT8_C( 118),
         INT8_C(  46), -INT8_C(  21),  INT8_C( 117),  INT8_C(  34), -INT8_C(  42),  INT8_C(  45),  INT8_C(  48),  INT8_C( 123),
         INT8_C(  65), -INT8_C(  75),  INT8_C( 124),  INT8_C(  76),  INT8_C(  78), -INT8_C(  65),  INT8_C(  18),  INT8_C(   9) } },
    { {  INT8_C(  66),  INT8_C(  32),  INT8_C(  53),  INT8_C( 124),      INT8_MIN,  INT8_C(  74),  INT8_C(  90),  INT8_C( 100),
         INT8_C(  94), -INT8_C(  62),  INT8_C(  48), -INT8_C(  96),  INT8_C(  90), -INT8_C( 124), -INT8_C(  57), -INT8_C( 120),
        -INT8_C( 112),  INT8_C(  60), -INT8_C(  86),  INT8_C( 102),  INT8_C( 106), -INT8_C(  61), -INT8_C(  30), -INT8_C(  85),
         INT8_C(  86),  INT8_C(  94), -INT8_C(   8), -INT8_C(  92),  INT8_C(  30), -INT8_C(  32), -INT8_C(  82),  INT8_C(  96),
         INT8_C(   0), -INT8_C(  29), -INT8_C(  36),      INT8_MIN,  INT8_C(  46),  INT8_C(  54), -INT8_C(  27), -INT8_C( 116),
        -INT8_C(   8),  INT8_C(  21),  INT8_C(  44),  INT8_C(  83), -INT8_C( 102), -INT8_C(  12), -INT8_C(  37),  INT8_C(  42),
         INT8_C(  48), -INT8_C( 122), -INT8_C( 111), -INT8_C( 102),  INT8_C(  73),  INT8_C( 115),  INT8_C(  70), -INT8_C(  97),
        -INT8_C(  47),  INT8_C(  62),  INT8_C(  68), -INT8_C(  17),  INT8_C(  30), -INT8_C(  14),  INT8_C(  79),  INT8_C(  30) },
      UINT64_C( 6597637535623490773),
      { -INT8_C( 103), -INT8_C(  68), -INT8_C(  82),  INT8_C(  51), -INT8_C(  80), -INT8_C( 119),  INT8_C(  94), -INT8_C(  32),
         INT8_C(  15), -INT8_C(  17),  INT8_C( 123),  INT8_C(  89),  INT8_C(  98), -INT8_C(  63), -INT8_C(   8),  INT8_C(  51),
        -INT8_C(   1),  INT8_C(  60),  INT8_C(  35),  INT8_C(  29),  INT8_C(  46),  INT8_C( 114),  INT8_C(  59),  INT8_C(   4),
        -INT8_C(  98), -INT8_C(  38),  INT8_C(   7),  INT8_C(   1),  INT8_C(  94), -INT8_C( 105),  INT8_C(  92), -INT8_C(   8),
         INT8_C(  83),  INT8_C(  10),  INT8_C(  43),  INT8_C(   3), -INT8_C( 109), -INT8_C( 119), -INT8_C(  29), -INT8_C(  93),
         INT8_C( 120),  INT8_C(  94), -INT8_C(   4), -INT8_C(  38),  INT8_C(  31), -INT8_C(  12),  INT8_C(  14),  INT8_C(  30),
         INT8_C(  49),  INT8_C(  49),  INT8_C(  59),  INT8_C(  95), -INT8_C(  93),  INT8_C( 119),  INT8_C(  99),  INT8_C(  66),
         INT8_C(  81),  INT8_C( 107),  INT8_C(  67), -INT8_C(  80),  INT8_C(   2), -INT8_C(  97), -INT8_C(  88),  INT8_C(  85) },
      {  INT8_C(  66), -INT8_C(  68),  INT8_C(  53),  INT8_C(  51),      INT8_MIN, -INT8_C( 119),  INT8_C(  90),  INT8_C( 100),
         INT8_C(  15), -INT8_C(  17),  INT8_C(  48), -INT8_C(  96),  INT8_C(  98), -INT8_C( 124), -INT8_C(   8),  INT8_C(  51),
        -INT8_C( 112),  INT8_C(  60), -INT8_C(  86),  INT8_C( 102),  INT8_C( 106),  INT8_C( 114),  INT8_C(  59), -INT8_C(  85),
         INT8_C(  86),  INT8_C(  94),  INT8_C(   7),  INT8_C(   1),  INT8_C(  94), -INT8_C( 105),  INT8_C(  92), -INT8_C(   8),
         INT8_C(  83), -INT8_C(  29),  INT8_C(  43),  INT8_C(   3), -INT8_C( 109),  INT8_C(  54), -INT8_C(  27), -INT8_C(  93),
         INT8_C( 120),  INT8_C(  94),  INT8_C(  44), -INT8_C(  38),  INT8_C(  31), -INT8_C(  12),  INT8_C(  14),  INT8_C(  42),
         INT8_C(  48), -INT8_C( 122), -INT8_C( 111), -INT8_C( 102), -INT8_C(  93),  INT8_C( 119),  INT8_C(  99), -INT8_C(  97),
        -INT8_C(  47),  INT8_C(  62),  INT8_C(  67), -INT8_C(  17),  INT8_C(  30), -INT8_C(  97),  INT8_C(  79),  INT8_C(  85) } },
    { { -INT8_C(  87), -INT8_C(  45),  INT8_C(  88),  INT8_C(  60),  INT8_C(  93),  INT8_C(  59), -INT8_C(  33), -INT8_C(  43),
        -INT8_C( 102), -INT8_C(  37), -INT8_C(  80), -INT8_C(  71), -INT8_C(  48), -INT8_C(  66), -INT8_C(  40),  INT8_C(   1),
        -INT8_C(  17),  INT8_C(  19),  INT8_C(  96), -INT8_C( 110), -INT8_C( 118), -INT8_C(  60), -INT8_C(  44), -INT8_C(  36),
         INT8_C(  47),  INT8_C(  23), -INT8_C( 116),  INT8_C(  49), -INT8_C(  74),  INT8_C(  52), -INT8_C( 122),  INT8_C(  95),
         INT8_C(   7), -INT8_C(  34), -INT8_C( 100),  INT8_C( 100),  INT8_C(  25),  INT8_C( 123),  INT8_C(  58), -INT8_C(  77),
         INT8_C(  87), -INT8_C(  22),  INT8_C( 109),  INT8_C(  39), -INT8_C(  88),  INT8_C(  69),  INT8_C(  40), -INT8_C( 105),
         INT8_C(  88), -INT8_C( 120),  INT8_C(  41), -INT8_C(  29),  INT8_C(  76), -INT8_C(   2), -INT8_C(  65),  INT8_C( 123),
         INT8_C(  21),  INT8_C(  75), -INT8_C(  84), -INT8_C(  52),      INT8_MAX,  INT8_C(  50),  INT8_C(  43), -INT8_C( 122) },
      UINT64_C(11159116422107088656),
      {  INT8_C(  15),  INT8_C(  74), -INT8_C(  63), -INT8_C(  73), -INT8_C( 113), -INT8_C(  23),  INT8_C(  78), -INT8_C(  24),
         INT8_C( 113),  INT8_C( 119), -INT8_C(  53), -INT8_C(  66),  INT8_C( 117), -INT8_C( 118),  INT8_C(  57), -INT8_C( 117),
        -INT8_C(  43), -INT8_C(  26),  INT8_C(  87),  INT8_C(  84),  INT8_C(  24), -INT8_C( 126), -INT8_C(  38),  INT8_C(  41),
         INT8_C(  74), -INT8_C(  59),  INT8_C(  83), -INT8_C( 115), -INT8_C(  22),  INT8_C(  48),  INT8_C(  39), -INT8_C(   7),
         INT8_C( 123), -INT8_C(  24), -INT8_C(  80),  INT8_C(  10), -INT8_C(  47), -INT8_C(   2), -INT8_C(  14),  INT8_C(  66),
         INT8_C( 118), -INT8_C(  67),  INT8_C(   0), -INT8_C(  21),  INT8_C(  71),  INT8_C(  58),  INT8_C( 118),  INT8_C(  28),
         INT8_C(  32), -INT8_C(  51),  INT8_C( 112),  INT8_C(  56),  INT8_C(  80),  INT8_C(  75),  INT8_C(  97), -INT8_C( 102),
         INT8_C(  16), -INT8_C(  76),  INT8_C(  39), -INT8_C(   5), -INT8_C(  27),  INT8_C(  78), -INT8_C(  12),  INT8_C(  96) },
      {  INT8_C(  15),  INT8_C(  74), -INT8_C(  63), -INT8_C(  73),  INT8_C(  93), -INT8_C(  23),  INT8_C(  78), -INT8_C(  24),
        -INT8_C( 102), -INT8_C(  37), -INT8_C(  80), -INT8_C(  66),  INT8_C( 117), -INT8_C( 118), -INT8_C(  40),  INT8_C(   1),
        -INT8_C(  17),  INT8_C(  19),  INT8_C(  87), -INT8_C( 110),  INT8_C(  24), -INT8_C(  60), -INT8_C(  44), -INT8_C(  36),
         INT8_C(  74),  INT8_C(  23),  INT8_C(  83),  INT8_C(  49), -INT8_C(  22),  INT8_C(  52),  INT8_C(  39), -INT8_C(   7),
         INT8_C(   7), -INT8_C(  34), -INT8_C(  80),  INT8_C(  10), -INT8_C(  47), -INT8_C(   2),  INT8_C(  58),  INT8_C(  66),
         INT8_C(  87), -INT8_C(  67),  INT8_C( 109), -INT8_C(  21),  INT8_C(  71),  INT8_C(  69),  INT8_C( 118),  INT8_C(  28),
         INT8_C(  88), -INT8_C(  51),  INT8_C(  41), -INT8_C(  29),  INT8_C(  76),  INT8_C(  75), -INT8_C(  65),  INT8_C( 123),
         INT8_C(  16),  INT8_C(  75),  INT8_C(  39), -INT8_C(  52),      INT8_MAX,  INT8_C(  78), -INT8_C(  12), -INT8_C( 122) } },
    { {  INT8_C(  54), -INT8_C(  91),  INT8_C( 106),  INT8_C(   7), -INT8_C(  93),  INT8_C(  93),  INT8_C(  73),  INT8_C(  25),
         INT8_C(  26),  INT8_C(  74),  INT8_C(   5),  INT8_C(  98), -INT8_C( 124),  INT8_C( 123),  INT8_C( 126), -INT8_C(  92),
         INT8_C(  73), -INT8_C(  17), -INT8_C(  36), -INT8_C( 103),  INT8_C(  58),  INT8_C(  62),  INT8_C(  51),  INT8_C(  74),
        -INT8_C(  14),  INT8_C(  90),  INT8_C(  69), -INT8_C(  41), -INT8_C(  88),  INT8_C(  58),  INT8_C(  55), -INT8_C(  34),
        -INT8_C(  33), -INT8_C(  94), -INT8_C(  27), -INT8_C( 126), -INT8_C(   1),  INT8_C(  46), -INT8_C( 100),  INT8_C(  25),
         INT8_C( 120), -INT8_C(  95),  INT8_C( 123), -INT8_C(   4),  INT8_C(  28), -INT8_C(   6), -INT8_C(  96),  INT8_C( 101),
        -INT8_C(  23),  INT8_C( 125), -INT8_C(   2),  INT8_C(  35), -INT8_C(  69),  INT8_C(  49),  INT8_C( 109), -INT8_C(  83),
        -INT8_C( 117), -INT8_C(  77), -INT8_C( 123),  INT8_C(  51), -INT8_C(  19), -INT8_C(  68),  INT8_C(  17), -INT8_C(  52) },
      UINT64_C(11346795232383399518),
      { -INT8_C( 117), -INT8_C(  14), -INT8_C( 102), -INT8_C(  88), -INT8_C(  20),  INT8_C(  58),  INT8_C(  13), -INT8_C(  43),
        -INT8_C(  73),  INT8_C(  12), -INT8_C(   8),  INT8_C( 114),  INT8_C(  61),  INT8_C( 102),  INT8_C(  32), -INT8_C(  55),
         INT8_C(  25), -INT8_C(  91), -INT8_C(   4),  INT8_C(   6),  INT8_C(  97),  INT8_C(  14), -INT8_C(  46), -INT8_C(  64),
         INT8_C(   4),  INT8_C(  32),  INT8_C(  29),  INT8_C(  41),  INT8_C(  11), -INT8_C( 108), -INT8_C(  57), -INT8_C( 106),
        -INT8_C( 121),  INT8_C(  97),  INT8_C(  62),  INT8_C( 115), -INT8_C( 101),  INT8_C(  76),  INT8_C(  73),  INT8_C(  83),
         INT8_C(  88),  INT8_C(  65), -INT8_C(  59), -INT8_C( 107), -INT8_C(  89), -INT8_C(  27),  INT8_C(  94), -INT8_C(  64),
        -INT8_C( 118),  INT8_C(  91), -INT8_C(  58), -INT8_C(  20),  INT8_C( 105), -INT8_C( 104), -INT8_C(  84),  INT8_C( 109),
        -INT8_C(  71), -INT8_C(  55), -INT8_C( 105), -INT8_C(  60),  INT8_C(  94),  INT8_C(  94),  INT8_C(  90), -INT8_C(  27) },
      { -INT8_C( 117), -INT8_C(  91),  INT8_C( 106),  INT8_C(   7), -INT8_C(  93),  INT8_C(  58),  INT8_C(  73), -INT8_C(  43),
        -INT8_C(  73),  INT8_C(  74),  INT8_C(   5),  INT8_C( 114), -INT8_C( 124),  INT8_C( 123),  INT8_C( 126), -INT8_C(  92),
         INT8_C(  25), -INT8_C(  17), -INT8_C(  36), -INT8_C( 103),  INT8_C(  97),  INT8_C(  14),  INT8_C(  51), -INT8_C(  64),
        -INT8_C(  14),  INT8_C(  32),  INT8_C(  69), -INT8_C(  41), -INT8_C(  88), -INT8_C( 108),  INT8_C(  55), -INT8_C( 106),
        -INT8_C(  33),  INT8_C(  97), -INT8_C(  27),  INT8_C( 115), -INT8_C( 101),  INT8_C(  46),  INT8_C(  73),  INT8_C(  83),
         INT8_C(  88), -INT8_C(  95), -INT8_C(  59), -INT8_C(   4), -INT8_C(  89), -INT8_C(   6), -INT8_C(  96),  INT8_C( 101),
        -INT8_C(  23),  INT8_C( 125), -INT8_C(   2), -INT8_C(  20), -INT8_C(  69),  INT8_C(  49),  INT8_C( 109),  INT8_C( 109),
        -INT8_C( 117), -INT8_C(  55), -INT8_C( 123),  INT8_C(  51), -INT8_C(  19),  INT8_C(  94),  INT8_C(  90), -INT8_C(  52) } },
    { { -INT8_C(  65), -INT8_C( 103),  INT8_C(  88),  INT8_C(  90), -INT8_C(  27), -INT8_C(  95), -INT8_C(  83),  INT8_C(  61),
        -INT8_C(  29),  INT8_C( 115), -INT8_C(  46), -INT8_C( 118),  INT8_C(  88),  INT8_C(  49),  INT8_C(  75), -INT8_C(  29),
        -INT8_C( 116),  INT8_C(  17), -INT8_C(  49), -INT8_C(  11), -INT8_C(  86),  INT8_C( 123),  INT8_C(  98),  INT8_C(  99),
         INT8_C(  68), -INT8_C(   7),  INT8_C(  39), -INT8_C(  94),  INT8_C(  87), -INT8_C( 127), -INT8_C( 121),  INT8_C(  22),
         INT8_C(  26), -INT8_C(  32),  INT8_C( 113), -INT8_C(   1), -INT8_C( 127),  INT8_C(  30),  INT8_C(  60),  INT8_C( 100),
        -INT8_C( 111),  INT8_C(  15), -INT8_C(  17), -INT8_C(  22),  INT8_C(  64),  INT8_C(  58), -INT8_C(  51), -INT8_C(  52),
         INT8_C(  75), -INT8_C( 100), -INT8_C(  63), -INT8_C(  11),  INT8_C(  23),  INT8_C(  35),  INT8_C(  88),  INT8_C(  91),
         INT8_C(  29),      INT8_MAX, -INT8_C(   2),  INT8_C( 116),  INT8_C(   1), -INT8_C( 123), -INT8_C( 117),  INT8_C(  27) },
      UINT64_C(12415112570835041381),
      {  INT8_C( 102),  INT8_C(  58), -INT8_C( 106), -INT8_C(  90),  INT8_C( 116),  INT8_C(  99),  INT8_C( 114), -INT8_C(  64),
        -INT8_C(   1),  INT8_C(  51), -INT8_C(  75),  INT8_C(  22),  INT8_C(  87),  INT8_C(  14),  INT8_C( 113),  INT8_C( 116),
        -INT8_C( 115),  INT8_C( 111), -INT8_C(  24), -INT8_C( 114), -INT8_C(  11),  INT8_C( 115), -INT8_C(  86),  INT8_C(  90),
         INT8_C( 111), -INT8_C(  59),  INT8_C(  65), -INT8_C( 118),  INT8_C(  28), -INT8_C( 115),  INT8_C(  54), -INT8_C( 125),
        -INT8_C(  57), -INT8_C(  52),  INT8_C(  41),  INT8_C(  60),  INT8_C(  47), -INT8_C( 100), -INT8_C(   4),  INT8_C(  46),
        -INT8_C(  49), -INT8_C(  79),  INT8_C(  68),  INT8_C(  38), -INT8_C(  65), -INT8_C(  75), -INT8_C( 102),  INT8_C(  77),
         INT8_C(  37), -INT8_C( 125), -INT8_C(  37),  INT8_C(  26), -INT8_C(  10), -INT8_C( 123),  INT8_C( 116),  INT8_C( 102),
         INT8_C(  74), -INT8_C(  74), -INT8_C(  16),  INT8_C( 103),  INT8_C(  67),  INT8_C(  38), -INT8_C(  22),  INT8_C(  10) },
      { -INT8_C(  65),  INT8_C(  58),  INT8_C(  88), -INT8_C(  90),  INT8_C( 116), -INT8_C(  95), -INT8_C(  83), -INT8_C(  64),
        -INT8_C(   1),  INT8_C(  51), -INT8_C(  46), -INT8_C( 118),  INT8_C(  88),  INT8_C(  49),  INT8_C(  75), -INT8_C(  29),
        -INT8_C( 116),  INT8_C(  17), -INT8_C(  24), -INT8_C(  11), -INT8_C(  86),  INT8_C( 115), -INT8_C(  86),  INT8_C(  90),
         INT8_C(  68), -INT8_C(   7),  INT8_C(  39), -INT8_C( 118),  INT8_C(  28), -INT8_C( 127), -INT8_C( 121),  INT8_C(  22),
        -INT8_C(  57), -INT8_C(  32),  INT8_C(  41), -INT8_C(   1), -INT8_C( 127), -INT8_C( 100), -INT8_C(   4),  INT8_C(  46),
        -INT8_C( 111),  INT8_C(  15), -INT8_C(  17),  INT8_C(  38),  INT8_C(  64), -INT8_C(  75), -INT8_C(  51),  INT8_C(  77),
         INT8_C(  75), -INT8_C( 100), -INT8_C(  37), -INT8_C(  11), -INT8_C(  10), -INT8_C( 123),  INT8_C(  88),  INT8_C( 102),
         INT8_C(  74), -INT8_C(  74), -INT8_C(   2),  INT8_C( 116),  INT8_C(  67), -INT8_C( 123), -INT8_C(  22),  INT8_C(  27) } },
    { { -INT8_C(  14),  INT8_C(  19),  INT8_C(  70),  INT8_C(  33), -INT8_C(  81),  INT8_C(  66),  INT8_C(  79),      INT8_MAX,
        -INT8_C(  12), -INT8_C( 109), -INT8_C(  91), -INT8_C(  77),  INT8_C(  72),  INT8_C(  64),  INT8_C(   0),  INT8_C( 109),
        -INT8_C(  61), -INT8_C(  36), -INT8_C( 121), -INT8_C(  71),  INT8_C(  97), -INT8_C(   4),  INT8_C(  31), -INT8_C(  84),
        -INT8_C(  78),  INT8_C(  15),  INT8_C(  19), -INT8_C(  11),  INT8_C(  53), -INT8_C(   3), -INT8_C(   1),  INT8_C(  39),
         INT8_C(  16),  INT8_C(  70),  INT8_C(  72), -INT8_C(  64), -INT8_C( 120), -INT8_C( 105),  INT8_C(  63),  INT8_C( 124),
         INT8_C(  42), -INT8_C(  28),  INT8_C(  48),  INT8_C( 115),  INT8_C(  36),  INT8_C(  48), -INT8_C(  32), -INT8_C(  25),
         INT8_C(  12),  INT8_C( 104), -INT8_C(  95),  INT8_C( 110),  INT8_C( 100), -INT8_C(  64),  INT8_C(  26),  INT8_C(  22),
        -INT8_C(  48),  INT8_C(  45),  INT8_C(  11),  INT8_C(   5),  INT8_C(  42),  INT8_C(  10),  INT8_C(  45),  INT8_C(  58) },
      UINT64_C( 3987155776760870224),
      {  INT8_C(  30), -INT8_C( 123), -INT8_C(  86),  INT8_C(  66), -INT8_C(  74), -INT8_C( 117),  INT8_C(  42), -INT8_C(  62),
        -INT8_C(  13), -INT8_C(  53),  INT8_C(  48),  INT8_C(  87), -INT8_C( 117),  INT8_C(  74),  INT8_C( 109),  INT8_C(  91),
         INT8_C( 119),  INT8_C( 120),  INT8_C(  97), -INT8_C(  95), -INT8_C( 126), -INT8_C( 114), -INT8_C(  36), -INT8_C(  45),
         INT8_C(   3), -INT8_C(  42), -INT8_C(  84),  INT8_C(  16),  INT8_C(  16),  INT8_C(   1),  INT8_C(  72),  INT8_C(  46),
        -INT8_C( 121), -INT8_C(  14),  INT8_C( 112),  INT8_C(  61),  INT8_C( 125), -INT8_C( 102), -INT8_C(   1),  INT8_C( 112),
         INT8_C( 101),  INT8_C(  48), -INT8_C(  57), -INT8_C(  15),  INT8_C( 122),  INT8_C(  52),  INT8_C(  76), -INT8_C(  14),
        -INT8_C(  84), -INT8_C(  83), -INT8_C( 109),  INT8_C(  47),  INT8_C(  59),  INT8_C( 111),  INT8_C(   2),  INT8_C(  63),
         INT8_C(  70), -INT8_C(  82),  INT8_C(  79),  INT8_C(  86), -INT8_C(  81), -INT8_C( 105), -INT8_C( 124),  INT8_C(  54) },
      {  INT8_C(  30), -INT8_C( 123), -INT8_C(  86),  INT8_C(  66), -INT8_C(  81), -INT8_C( 117),  INT8_C(  79), -INT8_C(  62),
        -INT8_C(  12), -INT8_C(  53), -INT8_C(  91),  INT8_C(  87),  INT8_C(  72),  INT8_C(  64),  INT8_C(   0),  INT8_C(  91),
         INT8_C( 119), -INT8_C(  36),  INT8_C(  97), -INT8_C(  71),  INT8_C(  97), -INT8_C(   4),  INT8_C(  31), -INT8_C(  84),
        -INT8_C(  78), -INT8_C(  42), -INT8_C(  84), -INT8_C(  11),  INT8_C(  53),  INT8_C(   1), -INT8_C(   1),  INT8_C(  39),
         INT8_C(  16), -INT8_C(  14),  INT8_C(  72), -INT8_C(  64),  INT8_C( 125), -INT8_C( 102), -INT8_C(   1),  INT8_C( 112),
         INT8_C(  42),  INT8_C(  48), -INT8_C(  57),  INT8_C( 115),  INT8_C(  36),  INT8_C(  48),  INT8_C(  76), -INT8_C(  14),
         INT8_C(  12), -INT8_C(  83), -INT8_C(  95),  INT8_C(  47),  INT8_C( 100),  INT8_C( 111),  INT8_C(  26),  INT8_C(  63),
        -INT8_C(  48),  INT8_C(  45),  INT8_C(  11),  INT8_C(  86),  INT8_C(  42),  INT8_C(  10), -INT8_C( 124),  INT8_C(  54) } },
    { { -INT8_C( 118), -INT8_C(  12),  INT8_C( 115),  INT8_C(   7), -INT8_C( 113),  INT8_C( 115),  INT8_C( 120), -INT8_C(  12),
        -INT8_C(  93),  INT8_C(  63), -INT8_C(  27),  INT8_C(  29),  INT8_C( 116),  INT8_C(  50),  INT8_C(  15),  INT8_C(  32),
        -INT8_C(  33), -INT8_C(  93),  INT8_C(  79),  INT8_C(  27),  INT8_C(  18),  INT8_C(  81),  INT8_C(  90),  INT8_C(  88),
        -INT8_C(   1), -INT8_C(  87), -INT8_C(  82), -INT8_C(  81),  INT8_C(  65),  INT8_C(  50), -INT8_C(  27), -INT8_C(  53),
         INT8_C(  39),  INT8_C(  89), -INT8_C(  46), -INT8_C(  74), -INT8_C(  52),  INT8_C(  74), -INT8_C(  86),  INT8_C( 111),
        -INT8_C( 118), -INT8_C( 112), -INT8_C( 116), -INT8_C(   2), -INT8_C(  62), -INT8_C( 100),  INT8_C(  30), -INT8_C(  95),
         INT8_C(  63),  INT8_C( 110), -INT8_C(  68),  INT8_C(  81), -INT8_C(  65),  INT8_C(  22), -INT8_C(  86), -INT8_C(  65),
        -INT8_C(  64),  INT8_C(  88),  INT8_C( 110),  INT8_C(   1), -INT8_C( 117),  INT8_C(  83), -INT8_C(  52), -INT8_C(  78) },
      UINT64_C( 8351664827938676396),
      { -INT8_C(  94),  INT8_C( 116),  INT8_C( 113),  INT8_C( 100),  INT8_C(  16), -INT8_C( 113),  INT8_C(   6),  INT8_C(  79),
        -INT8_C(   3), -INT8_C(  62), -INT8_C(  96), -INT8_C(  67), -INT8_C(  39),  INT8_C(  74),  INT8_C( 124), -INT8_C( 103),
        -INT8_C(  93), -INT8_C(  22), -INT8_C( 102),  INT8_C(  46),  INT8_C(  61),  INT8_C( 102), -INT8_C(  32), -INT8_C(  22),
         INT8_C(   4),  INT8_C(  72),  INT8_C(  98), -INT8_C(  19),  INT8_C(  90),  INT8_C(  74),  INT8_C(  96), -INT8_C(   3),
        -INT8_C(  66), -INT8_C(  47),  INT8_C(  97), -INT8_C(  50),  INT8_C(  97),  INT8_C( 103),  INT8_C(  29),  INT8_C(  94),
         INT8_C(  42), -INT8_C(  67),  INT8_C(  27),  INT8_C(   3),  INT8_C(   8), -INT8_C( 105), -INT8_C( 100), -INT8_C(  85),
        -INT8_C( 127),  INT8_C(  54), -INT8_C(  39), -INT8_C(  65), -INT8_C( 100), -INT8_C(  71), -INT8_C(  87), -INT8_C(  96),
         INT8_C(   1),  INT8_C(  11), -INT8_C( 114),  INT8_C(  91),  INT8_C(  85), -INT8_C(  18),  INT8_C(  88),  INT8_C(  19) },
      { -INT8_C(  94),  INT8_C( 116),  INT8_C( 115),  INT8_C(   7),  INT8_C(  16),  INT8_C( 115),  INT8_C(   6), -INT8_C(  12),
        -INT8_C(   3),  INT8_C(  63), -INT8_C(  27),  INT8_C(  29),  INT8_C( 116),  INT8_C(  74),  INT8_C( 124),  INT8_C(  32),
        -INT8_C(  93), -INT8_C(  22), -INT8_C( 102),  INT8_C(  27),  INT8_C(  61),  INT8_C(  81),  INT8_C(  90), -INT8_C(  22),
         INT8_C(   4),  INT8_C(  72),  INT8_C(  98), -INT8_C(  81),  INT8_C(  65),  INT8_C(  50), -INT8_C(  27), -INT8_C(   3),
         INT8_C(  39), -INT8_C(  47),  INT8_C(  97), -INT8_C(  74),  INT8_C(  97),  INT8_C(  74), -INT8_C(  86),  INT8_C( 111),
         INT8_C(  42), -INT8_C( 112),  INT8_C(  27),  INT8_C(   3), -INT8_C(  62), -INT8_C( 105), -INT8_C( 100), -INT8_C(  85),
         INT8_C(  63),  INT8_C( 110), -INT8_C(  68), -INT8_C(  65), -INT8_C( 100),  INT8_C(  22), -INT8_C(  86), -INT8_C(  65),
        -INT8_C(  64),  INT8_C(  88), -INT8_C( 114),  INT8_C(  91), -INT8_C( 117),  INT8_C(  83), -INT8_C(  52),  INT8_C(  19) } },
    { { -INT8_C(  64), -INT8_C(  70), -INT8_C(  31),  INT8_C(  33),  INT8_C(  33), -INT8_C(   2),      INT8_MAX,  INT8_C(  75),
        -INT8_C(  68), -INT8_C( 101),  INT8_C(  78), -INT8_C(  60),  INT8_C(  50), -INT8_C(  22),  INT8_C( 111), -INT8_C(  76),
         INT8_C(  32),  INT8_C(  72),  INT8_C( 115), -INT8_C(  68),  INT8_C(   1),  INT8_C(  28),  INT8_C(  93),  INT8_C(   2),
         INT8_C(  39), -INT8_C(  21),  INT8_C(  93),  INT8_C( 125), -INT8_C(  39), -INT8_C(  74), -INT8_C( 112), -INT8_C( 103),
         INT8_C( 112),  INT8_C( 114), -INT8_C(  70), -INT8_C( 111),  INT8_C( 112),  INT8_C(  58), -INT8_C(  35),  INT8_C(  44),
        -INT8_C(  43),  INT8_C(  43), -INT8_C(  16),  INT8_C(   7),  INT8_C(  22),  INT8_C(  95), -INT8_C(  69),  INT8_C(  54),
        -INT8_C(  89),  INT8_C(  46), -INT8_C(  13), -INT8_C(  88),  INT8_C(  74),  INT8_C(  80), -INT8_C(  86),  INT8_C( 114),
         INT8_C(  59),  INT8_C(   8), -INT8_C(  17),  INT8_C(  20), -INT8_C(  66),      INT8_MAX, -INT8_C(  82),  INT8_C(  46) },
      UINT64_C( 8614995358461683953),
      { -INT8_C(  56),      INT8_MAX,      INT8_MAX, -INT8_C(  34), -INT8_C(  34),  INT8_C(  58),  INT8_C(  20), -INT8_C( 122),
         INT8_C( 105),  INT8_C(   7),  INT8_C(  46), -INT8_C(  77),  INT8_C(  87), -INT8_C(  39),  INT8_C(  37), -INT8_C( 110),
        -INT8_C(  31),  INT8_C(  20), -INT8_C(  89), -INT8_C(  97), -INT8_C( 108),  INT8_C(  85), -INT8_C(  51), -INT8_C( 123),
        -INT8_C(  67), -INT8_C( 116), -INT8_C(  25),  INT8_C(  96),  INT8_C(  41),  INT8_C( 118), -INT8_C(  41), -INT8_C(  15),
        -INT8_C(  11),  INT8_C(  86), -INT8_C(  49), -INT8_C(  45), -INT8_C( 111), -INT8_C(  29),  INT8_C(  89), -INT8_C(   6),
        -INT8_C(  21), -INT8_C( 120), -INT8_C(  83),  INT8_C(  66),  INT8_C(  97), -INT8_C(  45), -INT8_C(  43),  INT8_C(  66),
        -INT8_C(  25),  INT8_C( 124), -INT8_C(  31),  INT8_C( 123), -INT8_C(  47), -INT8_C(  82),  INT8_C(   1), -INT8_C( 114),
         INT8_C(  58), -INT8_C(  24), -INT8_C(  18),  INT8_C(  99),  INT8_C(  94), -INT8_C(  58),  INT8_C(  84),  INT8_C(  83) },
      { -INT8_C(  64),      INT8_MAX,      INT8_MAX, -INT8_C(  34),  INT8_C(  33), -INT8_C(   2),      INT8_MAX,  INT8_C(  75),
         INT8_C( 105),  INT8_C(   7),  INT8_C(  46), -INT8_C(  60),  INT8_C(  87), -INT8_C(  22),  INT8_C( 111), -INT8_C( 110),
         INT8_C(  32),  INT8_C(  72),  INT8_C( 115), -INT8_C(  68),  INT8_C(   1),  INT8_C(  28), -INT8_C(  51),  INT8_C(   2),
        -INT8_C(  67), -INT8_C(  21), -INT8_C(  25),  INT8_C(  96),  INT8_C(  41), -INT8_C(  74), -INT8_C( 112), -INT8_C(  15),
        -INT8_C(  11),  INT8_C( 114), -INT8_C(  49), -INT8_C(  45), -INT8_C( 111),  INT8_C(  58),  INT8_C(  89),  INT8_C(  44),
        -INT8_C(  21), -INT8_C( 120), -INT8_C(  16),  INT8_C(   7),  INT8_C(  22), -INT8_C(  45), -INT8_C(  43),  INT8_C(  54),
        -INT8_C(  25),  INT8_C(  46), -INT8_C(  13), -INT8_C(  88), -INT8_C(  47), -INT8_C(  82),  INT8_C(   1),  INT8_C( 114),
         INT8_C(  59),  INT8_C(   8), -INT8_C(  17),  INT8_C(  99), -INT8_C(  66),      INT8_MAX, -INT8_C(  82),  INT8_C(  83) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i a = simde_mm512_loadu_epi8(test_vec[i].a);
    simde__m512i r0 = simde_mm512_loadu_epi8(test_vec[i].r0);
    int8_t r1[sizeof(simde__m512i) / sizeof(int8_t)];
    simde_mm512_storeu_epi8(r1, r0);
    simde_mm512_mask_storeu_epi8(r1, test_vec[i].k, a);
    simde_assert_equal_vi8(sizeof(r1) / sizeof(r1[0]), r1, test_vec[i].r1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    simde__m512i a = simde_test_x86_random_i8x64();
    simde__mmask64 k = simde_test_x86_random_mmask64();
    simde__m512i r0 = simde_test_x86_random_i8x64();
    int8_t r1[sizeof(simde__m512i) / sizeof(int8_t)];
    simde_mm512_storeu_epi8(r1, r0);

    simde_mm512_mask_storeu_epi8(r1, k, a);

    simde_test_x86_write_i8x64(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_mmask64(2, k, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_i8x64(2, r0, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_codegen_write_vi8(2, sizeof(r1) / sizeof(r1[0]), r1, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm512_mask_storeu_epi16 (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const simde__mmask32 k;
    const int16_t a[32];
    const int16_t r0[32];
    const int16_t r1[32];
  } test_vec[] = {
    { UINT32_C(2529696525),
      { -INT16_C( 20772), -INT16_C(  5620), -INT16_C( 15405), -INT16_C( 10386), -INT16_C(  3361),  INT16_C( 24295),  INT16_C( 30557),  INT16_C( 24735),
        -INT16_C(  5718),  INT16_C( 25900), -INT16_C(   982), -INT16_C( 24156),  INT16_C( 27219),  INT16_C( 25079), -INT16_C( 16506),  INT16_C( 25335),
         INT16_C(   877),  INT16_C( 16460), -INT16_C( 17466), -INT16_C( 23273), -INT16_C(   339),  INT16_C(  2564), -INT16_C( 23690),  INT16_C(  8298),
        -INT16_C( 26996), -INT16_C( 18810),  INT16_C( 10899), -INT16_C(  6568),  INT16_C( 20372),  INT16_C(  6727),  INT16_C( 15886),  INT16_C( 31613) },
      { -INT16_C( 14014),  INT16_C(  2236), -INT16_C( 11388),  INT16_C( 12718), -INT16_C( 19758),  INT16_C( 18491), -INT16_C( 22955), -INT16_C(  7576),
        -INT16_C(  4548), -INT16_C( 12392), -INT16_C(  4072), -INT16_C( 21066), -INT16_C(   705),  INT16_C( 19911),  INT16_C( 17468),  INT16_C( 32457),
        -INT16_C( 31474), -INT16_C( 28026),  INT16_C( 13400),  INT16_C( 10948), -INT16_C(    26),  INT16_C( 15474), -INT16_C(  9307), -INT16_C(  7650),
        -INT16_C( 18743), -INT16_C(  7503),  INT16_C( 26535), -INT16_C(  6513),  INT16_C( 22117), -INT16_C( 24268), -INT16_C(   613), -INT16_C( 22241) },
      { -INT16_C( 20772),  INT16_C(  2236), -INT16_C( 15405), -INT16_C( 10386), -INT16_C( 19758),  INT16_C( 18491), -INT16_C( 22955), -INT16_C(  7576),
        -INT16_C(  5718),  INT16_C( 25900), -INT16_C(  4072), -INT16_C( 24156),  INT16_C( 27219),  INT16_C( 19911),  INT16_C( 17468),  INT16_C( 32457),
        -INT16_C( 31474), -INT16_C( 28026),  INT16_C( 13400), -INT16_C( 23273), -INT16_C(    26),  INT16_C( 15474), -INT16_C( 23690),  INT16_C(  8298),
        -INT16_C( 18743), -INT16_C( 18810),  INT16_C( 10899), -INT16_C(  6513),  INT16_C( 20372), -INT16_C( 24268), -INT16_C(   613),  INT16_C( 31613) } },
    { UINT32_C(3661342082),
      { -INT16_C(    38), -INT16_C( 16379),  INT16_C( 30719), -INT16_C( 23300),  INT16_C(  6738),  INT16_C(  7302),  INT16_C( 14545),  INT16_C( 30974),
        -INT16_C( 29281),  INT16_C(  1118), -INT16_C( 27933),  INT16_C( 32421), -INT16_C( 15217),  INT16_C(  4391),  INT16_C( 25450),  INT16_C( 17644),
        -INT16_C(  3742),  INT16_C( 24836),  INT16_C(   360), -INT16_C( 17658), -INT16_C( 29669), -INT16_C(  4905), -INT16_C( 10812),  INT16_C( 25700),
        -INT16_C( 15518),  INT16_C( 17768),  INT16_C(  3669), -INT16_C(  6716), -INT16_C(  5166),  INT16_C( 15606), -INT16_C(  7602), -INT16_C( 20096) },
      { -INT16_C( 31277),  INT16_C( 15378),  INT16_C(  6278), -INT16_C( 24073), -INT16_C( 12635),  INT16_C( 27022), -INT16_C(  3421),  INT16_C(  1485),
         INT16_C( 14005),  INT16_C(  2890),  INT16_C(  3652),  INT16_C(  5872), -INT16_C(  6406),  INT16_C( 18515), -INT16_C( 11319), -INT16_C( 25351),
         INT16_C(  3160), -INT16_C(  8488), -INT16_C( 12508), -INT16_C( 13952),  INT16_C(  3741),  INT16_C( 16435),  INT16_C(     0), -INT16_C( 18875),
        -INT16_C( 28618),  INT16_C( 31425), -INT16_C( 20066), -INT16_C( 26479), -INT16_C(  7017),  INT16_C( 24801), -INT16_C(  9545),  INT16_C(  4349) },
      { -INT16_C( 31277), -INT16_C( 16379),  INT16_C(  6278), -INT16_C( 24073), -INT16_C( 12635),  INT16_C( 27022), -INT16_C(  3421),  INT16_C( 30974),
        -INT16_C( 29281),  INT16_C(  2890), -INT16_C( 27933),  INT16_C(  5872), -INT16_C(  6406),  INT16_C(  4391), -INT16_C( 11319),  INT16_C( 17644),
        -INT16_C(  3742),  INT16_C( 24836), -INT16_C( 12508), -INT16_C( 17658), -INT16_C( 29669), -INT16_C(  4905),  INT16_C(     0), -INT16_C( 18875),
        -INT16_C( 28618),  INT16_C( 17768), -INT16_C( 20066), -INT16_C(  6716), -INT16_C(  5166),  INT16_C( 24801), -INT16_C(  7602), -INT16_C( 20096) } },
    { UINT32_C( 200201702),
      {  INT16_C( 28325),  INT16_C( 17108),  INT16_C(  1916),  INT16_C( 32131), -INT16_C( 14328),  INT16_C( 15923), -INT16_C(  2984), -INT16_C(  2119),
         INT16_C( 19109),  INT16_C( 15503),  INT16_C( 28718), -INT16_C(  6755), -INT16_C( 26037),  INT16_C( 12789), -INT16_C(  7057),  INT16_C(  5180),
         INT16_C(  4434), -INT16_C( 12457), -INT16_C(  9704),  INT16_C(  8268),  INT16_C( 32674), -INT16_C(  1185),  INT16_C(  6259),  INT16_C(  6386),
        -INT16_C( 32414), -INT16_C( 28588), -INT16_C(  3598),  INT16_C( 15733),  INT16_C( 27531), -INT16_C(  1170), -INT16_C( 21681), -INT16_C( 24305) },
      {  INT16_C( 26300), -INT16_C( 11152), -INT16_C( 17344), -INT16_C(  7179),  INT16_C( 21563), -INT16_C( 20770), -INT16_C( 12180), -INT16_C( 12602),
         INT16_C(  6993),  INT16_C( 17246), -INT16_C( 11508), -INT16_C( 26496), -INT16_C(  4290), -INT16_C( 29293), -INT16_C( 23910),  INT16_C( 22063),
        -INT16_C( 24823),  INT16_C( 18730),  INT16_C(  8028), -INT16_C( 26836),  INT16_C(  2675), -INT16_C(  8378),  INT16_C(  3290),  INT16_C( 11437),
         INT16_C(  2855),  INT16_C( 13423), -INT16_C(  3873),  INT16_C(  7628),  INT16_C( 24543),  INT16_C( 31147), -INT16_C(  9727),  INT16_C(  2767) },
      {  INT16_C( 26300),  INT16_C( 17108),  INT16_C(  1916), -INT16_C(  7179),  INT16_C( 21563),  INT16_C( 15923), -INT16_C(  2984), -INT16_C(  2119),
         INT16_C( 19109),  INT16_C( 17246),  INT16_C( 28718), -INT16_C( 26496), -INT16_C( 26037), -INT16_C( 29293), -INT16_C(  7057),  INT16_C(  5180),
        -INT16_C( 24823), -INT16_C( 12457), -INT16_C(  9704),  INT16_C(  8268),  INT16_C(  2675), -INT16_C(  1185),  INT16_C(  6259),  INT16_C(  6386),
        -INT16_C( 32414), -INT16_C( 28588), -INT16_C(  3873),  INT16_C( 15733),  INT16_C( 24543),  INT16_C( 31147), -INT16_C(  9727),  INT16_C(  2767) } },
    { UINT32_C(3579115897),
      { -INT16_C( 32743), -INT16_C( 29587), -INT16_C( 19573),  INT16_C( 25964),  INT16_C(  6591), -INT16_C(  6255),  INT16_C(   293),  INT16_C(  1051),
        -INT16_C(  6159), -INT16_C( 12255), -INT16_C( 13242),  INT16_C( 18249),  INT16_C(  6310),  INT16_C(  8274), -INT16_C( 23023),  INT16_C( 10997),
         INT16_C( 25126), -INT16_C( 20041),  INT16_C(  8981), -INT16_C( 10985), -INT16_C( 22468),  INT16_C( 25020), -INT16_C( 10327), -INT16_C( 26011),
        -INT16_C( 30786),  INT16_C(  1130), -INT16_C( 19629), -INT16_C(  1461), -INT16_C( 25141), -INT16_C(  8934),  INT16_C(  3907),  INT16_C( 27143) },
      { -INT16_C( 16782), -INT16_C( 30949),  INT16_C( 13025),  INT16_C(  7772),  INT16_C(  6363), -INT16_C( 31617), -INT16_C(  6673), -INT16_C( 21217),
        -INT16_C( 30356), -INT16_C( 16463), -INT16_C(   707),  INT16_C(  2233), -INT16_C( 11366), -INT16_C(  8475), -INT16_C(  4637),  INT16_C( 21832),
         INT16_C( 25515), -INT16_C( 29220),  INT16_C( 14742),  INT16_C( 29099),  INT16_C( 10833),  INT16_C( 16885),  INT16_C(  5135),  INT16_C( 31726),
        -INT16_C( 24418), -INT16_C(  9413), -INT16_C(  2915),  INT16_C( 14307), -INT16_C( 13880), -INT16_C( 21739),  INT16_C( 23990),  INT16_C( 24832) },
      { -INT16_C( 32743), -INT16_C( 30949),  INT16_C( 13025),  INT16_C( 25964),  INT16_C(  6591), -INT16_C(  6255),  INT16_C(   293), -INT16_C( 21217),
        -INT16_C(  6159), -INT16_C( 16463), -INT16_C(   707),  INT16_C( 18249),  INT16_C(  6310),  INT16_C(  8274), -INT16_C( 23023),  INT16_C( 10997),
         INT16_C( 25515), -INT16_C( 29220),  INT16_C(  8981),  INT16_C( 29099), -INT16_C( 22468),  INT16_C( 16885), -INT16_C( 10327),  INT16_C( 31726),
        -INT16_C( 30786), -INT16_C(  9413), -INT16_C( 19629),  INT16_C( 14307), -INT16_C( 25141), -INT16_C( 21739),  INT16_C(  3907),  INT16_C( 27143) } },
    { UINT32_C(1475271873),
      { -INT16_C( 26347),  INT16_C( 26568), -INT16_C( 16956), -INT16_C( 11352), -INT16_C( 26926),  INT16_C( 28751), -INT16_C( 30154), -INT16_C( 11445),
         INT16_C( 11902),  INT16_C( 17931),  INT16_C(  8439), -INT16_C( 21007), -INT16_C(  3714),  INT16_C( 16143), -INT16_C(   562), -INT16_C(  7274),
         INT16_C( 24215),  INT16_C( 23370), -INT16_C(  3557), -INT16_C(  4818),  INT16_C( 32137), -INT16_C( 16547), -INT16_C( 22521), -INT16_C( 31085),
        -INT16_C( 24873), -INT16_C( 12596), -INT16_C( 16706),  INT16_C( 15484), -INT16_C( 29777),  INT16_C( 32123),  INT16_C(  4488),  INT16_C(  8033) },
      { -INT16_C( 21649), -INT16_C( 29830), -INT16_C( 22114),  INT16_C( 10104), -INT16_C( 10714),  INT16_C( 12006),  INT16_C( 31102),  INT16_C( 21940),
        -INT16_C( 32745), -INT16_C( 10716), -INT16_C( 24514), -INT16_C(  4590), -INT16_C( 29141), -INT16_C( 19605), -INT16_C( 13153),  INT16_C(  4051),
         INT16_C( 19832),  INT16_C(  5786),  INT16_C(  4854),  INT16_C(  7485),  INT16_C(  9192),  INT16_C( 26443), -INT16_C(    99), -INT16_C( 19268),
        -INT16_C(  8065), -INT16_C( 16758), -INT16_C( 25216), -INT16_C( 21588),  INT16_C(  5931), -INT16_C( 13729),  INT16_C( 13028),  INT16_C( 23769) },
      { -INT16_C( 26347), -INT16_C( 29830), -INT16_C( 22114),  INT16_C( 10104), -INT16_C( 10714),  INT16_C( 12006), -INT16_C( 30154), -INT16_C( 11445),
        -INT16_C( 32745), -INT16_C( 10716),  INT16_C(  8439), -INT16_C( 21007), -INT16_C(  3714), -INT16_C( 19605), -INT16_C(   562), -INT16_C(  7274),
         INT16_C( 19832),  INT16_C( 23370), -INT16_C(  3557), -INT16_C(  4818),  INT16_C(  9192), -INT16_C( 16547), -INT16_C( 22521), -INT16_C( 31085),
        -INT16_C( 24873), -INT16_C( 12596), -INT16_C( 16706), -INT16_C( 21588), -INT16_C( 29777), -INT16_C( 13729),  INT16_C(  4488),  INT16_C( 23769) } },
    { UINT32_C(1987212159),
      { -INT16_C( 20602),  INT16_C( 28307), -INT16_C(  8494),  INT16_C( 28629), -INT16_C( 27939),  INT16_C( 23588), -INT16_C( 20878), -INT16_C(  3302),
        -INT16_C( 14773),  INT16_C( 30366), -INT16_C(   546), -INT16_C( 15807),  INT16_C(  6703), -INT16_C( 20706), -INT16_C( 28530),  INT16_C(  5157),
        -INT16_C( 18369),  INT16_C(  4482),  INT16_C( 22678),  INT16_C( 29569), -INT16_C( 23062),  INT16_C( 23759), -INT16_C(  5549), -INT16_C( 24753),
        -INT16_C(  4432), -INT16_C( 29163),  INT16_C( 22251),  INT16_C(  6992),  INT16_C( 28273), -INT16_C(    54), -INT16_C(  4098),  INT16_C( 15635) },
      { -INT16_C( 27225),  INT16_C( 15695), -INT16_C( 12051), -INT16_C( 10320),  INT16_C( 32629), -INT16_C( 14284), -INT16_C( 31895),  INT16_C(  6759),
         INT16_C( 32113),  INT16_C( 23976), -INT16_C(  1581),  INT16_C( 17528),  INT16_C( 16999),  INT16_C( 26179),  INT16_C( 22065), -INT16_C( 10077),
        -INT16_C(  3348), -INT16_C(  9963), -INT16_C( 14910),  INT16_C( 14257), -INT16_C(  6844), -INT16_C( 20992),  INT16_C( 26472), -INT16_C(  9528),
         INT16_C( 28900), -INT16_C( 18377), -INT16_C( 20631), -INT16_C( 11780),  INT16_C( 16625),  INT16_C(  8759), -INT16_C(  9578), -INT16_C( 32006) },
      { -INT16_C( 20602),  INT16_C( 28307), -INT16_C(  8494),  INT16_C( 28629), -INT16_C( 27939),  INT16_C( 23588), -INT16_C( 20878),  INT16_C(  6759),
        -INT16_C( 14773),  INT16_C( 30366), -INT16_C(  1581),  INT16_C( 17528),  INT16_C(  6703), -INT16_C( 20706), -INT16_C( 28530), -INT16_C( 10077),
        -INT16_C(  3348),  INT16_C(  4482), -INT16_C( 14910),  INT16_C( 14257), -INT16_C( 23062),  INT16_C( 23759), -INT16_C(  5549), -INT16_C(  9528),
         INT16_C( 28900), -INT16_C( 29163),  INT16_C( 22251), -INT16_C( 11780),  INT16_C( 28273), -INT16_C(    54), -INT16_C(  4098), -INT16_C( 32006) } },
    { UINT32_C(2405175245),
      {  INT16_C(  3540),  INT16_C(  6343), -INT16_C( 14350),  INT16_C( 23238), -INT16_C( 29138),  INT16_C(  4916),  INT16_C( 27647),  INT16_C( 26827),
        -INT16_C( 14566),  INT16_C(  2873),  INT16_C( 28679), -INT16_C( 25043),  INT16_C( 10059),  INT16_C(  6176),  INT16_C( 31798),  INT16_C(  2727),
         INT16_C( 28297),  INT16_C( 31523), -INT16_C(  5835),  INT16_C( 25814),  INT16_C(  2680),  INT16_C( 30583),  INT16_C( 17014), -INT16_C( 28449),
         INT16_C(  6409),  INT16_C(  4508), -INT16_C( 13943), -INT16_C( 11089), -INT16_C( 12303),  INT16_C( 10220), -INT16_C( 27572), -INT16_C( 10958) },
      {  INT16_C( 21762),  INT16_C( 14417),  INT16_C( 10046), -INT16_C( 18788),  INT16_C(  4913), -INT16_C( 22739),  INT16_C(  3413),  INT16_C( 24120),
        -INT16_C( 11226), -INT16_C( 20625),  INT16_C(  7837), -INT16_C( 29052),  INT16_C( 28910),  INT16_C( 15030), -INT16_C(  6140),  INT16_C(  1807),
         INT16_C( 24637),  INT16_C( 31551), -INT16_C(  9337), -INT16_C( 18126),  INT16_C( 24558),  INT16_C( 17248), -INT16_C( 26516), -INT16_C( 27999),
         INT16_C(  4460),  INT16_C(  2626), -INT16_C( 14801),  INT16_C(  7576),  INT16_C( 20022),  INT16_C( 15191),  INT16_C( 26422),  INT16_C( 29506) },
      {  INT16_C(  3540),  INT16_C( 14417), -INT16_C( 14350),  INT16_C( 23238),  INT16_C(  4913), -INT16_C( 22739),  INT16_C( 27647),  INT16_C( 26827),
        -INT16_C( 14566),  INT16_C(  2873),  INT16_C( 28679), -INT16_C( 25043),  INT16_C( 28910),  INT16_C( 15030), -INT16_C(  6140),  INT16_C(  1807),
         INT16_C( 24637),  INT16_C( 31551), -INT16_C(  5835),  INT16_C( 25814),  INT16_C(  2680),  INT16_C( 17248),  INT16_C( 17014), -INT16_C( 27999),
         INT16_C(  6409),  INT16_C(  4508), -INT16_C( 13943), -INT16_C( 11089),  INT16_C( 20022),  INT16_C( 15191),  INT16_C( 26422), -INT16_C( 10958) } },
    { UINT32_C(1341096391),
      {  INT16_C(  8540),  INT16_C( 18952),  INT16_C( 26752), -INT16_C(  4723),  INT16_C( 11777),  INT16_C( 28031), -INT16_C( 16065),  INT16_C( 28535),
         INT16_C(  4231), -INT16_C( 16756), -INT16_C(  7074), -INT16_C( 27143),  INT16_C( 15179),  INT16_C(  4616), -INT16_C(  2116),  INT16_C(  6241),
         INT16_C( 26904), -INT16_C( 26270), -INT16_C(  4142), -INT16_C( 11386),  INT16_C(  1309),  INT16_C( 23872), -INT16_C( 18233),  INT16_C( 20172),
         INT16_C( 22728),  INT16_C(  9740),  INT16_C(  1340), -INT16_C( 30789), -INT16_C( 15296), -INT16_C(   870), -INT16_C(  1093), -INT16_C( 11244) },
      {  INT16_C( 30309),  INT16_C( 14189), -INT16_C(  3227), -INT16_C( 31990),  INT16_C( 19192), -INT16_C( 16416), -INT16_C( 21502), -INT16_C( 13810),
         INT16_C(  6660),  INT16_C( 16881), -INT16_C( 21472),  INT16_C( 24776),  INT16_C( 25200),  INT16_C( 11357),  INT16_C( 29022), -INT16_C( 15616),
         INT16_C( 28136),  INT16_C( 19962),  INT16_C(  1120),  INT16_C( 22736), -INT16_C( 20402),  INT16_C( 20760),  INT16_C(  9820),  INT16_C( 24859),
         INT16_C(  3136),  INT16_C( 24738),  INT16_C( 27321),  INT16_C( 10689),  INT16_C(  7885),  INT16_C( 11093),  INT16_C( 21903),  INT16_C( 30702) },
      {  INT16_C(  8540),  INT16_C( 18952),  INT16_C( 26752), -INT16_C( 31990),  INT16_C( 19192), -INT16_C( 16416), -INT16_C( 16065),  INT16_C( 28535),
         INT16_C(  4231),  INT16_C( 16881), -INT16_C( 21472),  INT16_C( 24776),  INT16_C( 25200),  INT16_C( 11357),  INT16_C( 29022),  INT16_C(  6241),
         INT16_C( 26904), -INT16_C( 26270), -INT16_C(  4142), -INT16_C( 11386), -INT16_C( 20402),  INT16_C( 23872), -INT16_C( 18233),  INT16_C( 20172),
         INT16_C( 22728),  INT16_C(  9740),  INT16_C(  1340), -INT16_C( 30789),  INT16_C(  7885),  INT16_C( 11093), -INT16_C(  1093),  INT16_C( 30702) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i a = simde_mm512_loadu_epi16(test_vec[i].a);
    simde__m512i r0 = simde_mm512_loadu_epi16(test_vec[i].r0);
    int16_t r1[sizeof(simde__m512i) / sizeof(int16_t)];
    simde_mm512_storeu_epi16(r1, r0);
    simde_mm512_mask_storeu_epi16(r1, test_vec[i].k, a);
    simde_assert_equal_vi16(sizeof(r1) / sizeof(r1[0]), r1, test_vec[i].r1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    simde__mmask32 k = simde_test_x86_random_mmask32();
    simde__m512i a = simde_test_x86_random_i16x32();
    simde__m512i r0 = simde_test_x86_random_i16x32();
    int16_t r1[sizeof(simde__m512i) / sizeof(int16_t)];
    simde_mm512_storeu_epi16(r1, r0);
    simde_mm512_mask_storeu_epi16(r1, k, a);

    simde_test_x86_write_mmask32(2, k, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i16x32(2, a, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_i16x32(2, r0, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_codegen_write_vi16(2, sizeof(r1) / sizeof(r1[0]), r1, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm512_mask_storeu_epi32 (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const simde__mmask16 k;
    const int32_t a[16];
    const int32_t r0[16];
    const int32_t r1[16];
  } test_vec[] = {
    { UINT16_C( 6699),
      {  INT32_C(    45653270), -INT32_C(   631230043), -INT32_C(   750974957),  INT32_C(  1887093155), -INT32_C(   812493712),  INT32_C(   544183919),  INT32_C(   967216968),  INT32_C(  1414779198),
         INT32_C(   324406126),  INT32_C(  1357756221),  INT32_C(  1646471871),  INT32_C(  1540529643),  INT32_C(  1630168306),  INT32_C(  1132632827),  INT32_C(  1752967209),  INT32_C(  1740427513) },
      {  INT32_C(   410653659), -INT32_C(  1989580598),  INT32_C(  2129366162),  INT32_C(   484031786),  INT32_C(   494732322), -INT32_C(   949944418),  INT32_C(   540007463), -INT32_C(  2004357972),
        -INT32_C(   912260353), -INT32_C(    61732502), -INT32_C(  1065730666),  INT32_C(   484201722), -INT32_C(   164013736), -INT32_C(  2135058087),  INT32_C(   581037430), -INT32_C(   659937063) },
      {  INT32_C(    45653270), -INT32_C(   631230043),  INT32_C(  2129366162),  INT32_C(  1887093155),  INT32_C(   494732322),  INT32_C(   544183919),  INT32_C(   540007463), -INT32_C(  2004357972),
        -INT32_C(   912260353),  INT32_C(  1357756221), -INT32_C(  1065730666),  INT32_C(  1540529643),  INT32_C(  1630168306), -INT32_C(  2135058087),  INT32_C(   581037430), -INT32_C(   659937063) } },
    { UINT16_C(19242),
      { -INT32_C(   212560735),  INT32_C(   187755153), -INT32_C(  2040583510), -INT32_C(  2132756665),  INT32_C(  1796880813),  INT32_C(  1515753657),  INT32_C(  1568879026), -INT32_C(  1431786231),
        -INT32_C(   744620990), -INT32_C(  1847669273),  INT32_C(  1041776375), -INT32_C(  1547699979), -INT32_C(   368125647),  INT32_C(   457533033), -INT32_C(  1602697065), -INT32_C(  1219878795) },
      {  INT32_C(    76211997), -INT32_C(  1416271436), -INT32_C(  1662341721), -INT32_C(   700470875), -INT32_C(   339653246),  INT32_C(  1241908915),  INT32_C(  1139441614), -INT32_C(  1124387681),
        -INT32_C(   809466597), -INT32_C(  1787078930), -INT32_C(  1456315133), -INT32_C(  1870696178),  INT32_C(  1920680127),  INT32_C(   347963718), -INT32_C(  1604802816), -INT32_C(   144944164) },
      {  INT32_C(    76211997),  INT32_C(   187755153), -INT32_C(  1662341721), -INT32_C(  2132756665), -INT32_C(   339653246),  INT32_C(  1515753657),  INT32_C(  1139441614), -INT32_C(  1124387681),
        -INT32_C(   744620990), -INT32_C(  1847669273), -INT32_C(  1456315133), -INT32_C(  1547699979),  INT32_C(  1920680127),  INT32_C(   347963718), -INT32_C(  1602697065), -INT32_C(   144944164) } },
    { UINT16_C( 7640),
      {  INT32_C(  1114884039), -INT32_C(  1901627812), -INT32_C(  1627343585), -INT32_C(  1059078331), -INT32_C(   297720271), -INT32_C(  1835646406),  INT32_C(  1055224546),  INT32_C(   811318889),
        -INT32_C(   512569723),  INT32_C(  1685068101),  INT32_C(   318992590),  INT32_C(  1624498991),  INT32_C(  1129256201),  INT32_C(   970384727), -INT32_C(  1065895081), -INT32_C(    17706119) },
      { -INT32_C(   421502047),  INT32_C(  1263227005), -INT32_C(   278966592),  INT32_C(   978268721), -INT32_C(  1635869113), -INT32_C(   623422333), -INT32_C(  1986310385), -INT32_C(  1014526942),
         INT32_C(  1823107055),  INT32_C(  2008610231),  INT32_C(  1969624899), -INT32_C(  1850755511), -INT32_C(   667996844), -INT32_C(  1850603647),  INT32_C(  2014989654), -INT32_C(   918838823) },
      { -INT32_C(   421502047),  INT32_C(  1263227005), -INT32_C(   278966592), -INT32_C(  1059078331), -INT32_C(   297720271), -INT32_C(   623422333),  INT32_C(  1055224546),  INT32_C(   811318889),
        -INT32_C(   512569723),  INT32_C(  2008610231),  INT32_C(   318992590),  INT32_C(  1624498991),  INT32_C(  1129256201), -INT32_C(  1850603647),  INT32_C(  2014989654), -INT32_C(   918838823) } },
    { UINT16_C(58633),
      { -INT32_C(   304431051), -INT32_C(  1643897288),  INT32_C(  1112821395),  INT32_C(   242264543),  INT32_C(   857076097), -INT32_C(  1669239934),  INT32_C(   524180195),  INT32_C(  1493452579),
         INT32_C(  1061609223), -INT32_C(  1864479747),  INT32_C(  2027041433),  INT32_C(  1552302811), -INT32_C(  1232036812), -INT32_C(   346877689), -INT32_C(  1895132821), -INT32_C(   538439976) },
      { -INT32_C(   367055123),  INT32_C(   310050169),  INT32_C(   193613103), -INT32_C(  1016655473), -INT32_C(  1267009619),  INT32_C(  1956629768),  INT32_C(   906209630), -INT32_C(  1525290056),
        -INT32_C(  1819266023),  INT32_C(  1638206001), -INT32_C(   412340137), -INT32_C(   290794687), -INT32_C(   744348469),  INT32_C(  1330070257), -INT32_C(  1551545621),  INT32_C(  1330223925) },
      { -INT32_C(   304431051),  INT32_C(   310050169),  INT32_C(   193613103),  INT32_C(   242264543), -INT32_C(  1267009619),  INT32_C(  1956629768),  INT32_C(   906209630), -INT32_C(  1525290056),
         INT32_C(  1061609223),  INT32_C(  1638206001),  INT32_C(  2027041433), -INT32_C(   290794687), -INT32_C(   744348469), -INT32_C(   346877689), -INT32_C(  1895132821), -INT32_C(   538439976) } },
    { UINT16_C(55759),
      { -INT32_C(  2015165982), -INT32_C(   826852510), -INT32_C(   861800414), -INT32_C(  1980666650), -INT32_C(  2016681408), -INT32_C(  1210927566), -INT32_C(  1554905254),  INT32_C(   947659350),
        -INT32_C(  2050990301), -INT32_C(  1135380582),  INT32_C(  1451881584), -INT32_C(  1579189663), -INT32_C(  1909937572), -INT32_C(  1152976287),  INT32_C(  1482594306), -INT32_C(   577643846) },
      { -INT32_C(   731754183),  INT32_C(   949007816),  INT32_C(   193927594), -INT32_C(   257134957),  INT32_C(  2055132185), -INT32_C(   785005361),  INT32_C(   388600669), -INT32_C(  1493845395),
        -INT32_C(   730179829), -INT32_C(  1223947507), -INT32_C(  1195205852),  INT32_C(   598240778),  INT32_C(   295511618),  INT32_C(  1222824683), -INT32_C(   765522843), -INT32_C(   763865914) },
      { -INT32_C(  2015165982), -INT32_C(   826852510), -INT32_C(   861800414), -INT32_C(  1980666650),  INT32_C(  2055132185), -INT32_C(   785005361), -INT32_C(  1554905254),  INT32_C(   947659350),
        -INT32_C(  2050990301), -INT32_C(  1223947507), -INT32_C(  1195205852), -INT32_C(  1579189663), -INT32_C(  1909937572),  INT32_C(  1222824683),  INT32_C(  1482594306), -INT32_C(   577643846) } },
    { UINT16_C(62380),
      { -INT32_C(  1291929178),  INT32_C(   843981424), -INT32_C(  2103420710),  INT32_C(   413786747), -INT32_C(   689269516),  INT32_C(  1004687324), -INT32_C(  1718572767),  INT32_C(   562838651),
         INT32_C(  1708362485), -INT32_C(  2020138579), -INT32_C(   200657031),  INT32_C(   218936089),  INT32_C(   602207815),  INT32_C(  1717487173), -INT32_C(   369037713),  INT32_C(   520850474) },
      { -INT32_C(   997925097),  INT32_C(  2051742464),  INT32_C(  1819170130),  INT32_C(  1333361416), -INT32_C(  1217241743), -INT32_C(  1826762460), -INT32_C(   360833601), -INT32_C(  1056339542),
         INT32_C(  1770360424), -INT32_C(    52178775),  INT32_C(   778588454),  INT32_C(  1048502732),  INT32_C(  1677062207), -INT32_C(  2114579775), -INT32_C(   630492112),  INT32_C(  1721464062) },
      { -INT32_C(   997925097),  INT32_C(  2051742464), -INT32_C(  2103420710),  INT32_C(   413786747), -INT32_C(  1217241743),  INT32_C(  1004687324), -INT32_C(   360833601),  INT32_C(   562838651),
         INT32_C(  1708362485), -INT32_C(  2020138579),  INT32_C(   778588454),  INT32_C(  1048502732),  INT32_C(   602207815),  INT32_C(  1717487173), -INT32_C(   369037713),  INT32_C(   520850474) } },
    { UINT16_C( 8194),
      { -INT32_C(  1292849969),  INT32_C(   268703400), -INT32_C(  1007562683),  INT32_C(    62074894), -INT32_C(  1978239597),  INT32_C(  1644054262), -INT32_C(  1143604192),  INT32_C(   853268579),
         INT32_C(   753257348),  INT32_C(   658303458), -INT32_C(   924176967), -INT32_C(   238314146),  INT32_C(   142336274),  INT32_C(  1198094887), -INT32_C(   654164106), -INT32_C(  1693655785) },
      { -INT32_C(  1949830743), -INT32_C(  1817050150), -INT32_C(  1906598864),  INT32_C(  1283467065),  INT32_C(   810875656), -INT32_C(   327696779),  INT32_C(   348486397),  INT32_C(    11522391),
        -INT32_C(  1668581694), -INT32_C(  1456456327),  INT32_C(   322472921), -INT32_C(  1151354702),  INT32_C(   703312819),  INT32_C(  1846895217),  INT32_C(   880990940),  INT32_C(  1848914348) },
      { -INT32_C(  1949830743),  INT32_C(   268703400), -INT32_C(  1906598864),  INT32_C(  1283467065),  INT32_C(   810875656), -INT32_C(   327696779),  INT32_C(   348486397),  INT32_C(    11522391),
        -INT32_C(  1668581694), -INT32_C(  1456456327),  INT32_C(   322472921), -INT32_C(  1151354702),  INT32_C(   703312819),  INT32_C(  1198094887),  INT32_C(   880990940),  INT32_C(  1848914348) } },
    { UINT16_C(49319),
      {  INT32_C(  1006444555),  INT32_C(    46585802),  INT32_C(  1236957674),  INT32_C(   536636724), -INT32_C(  1417580906),  INT32_C(  1602641628), -INT32_C(   963628398), -INT32_C(  1417267040),
         INT32_C(   585532504),  INT32_C(  1160031579),  INT32_C(  1519312422), -INT32_C(   495351220), -INT32_C(   712049928), -INT32_C(   365685672), -INT32_C(   424557498),  INT32_C(  1435645948) },
      {  INT32_C(   376928443),  INT32_C(  1264294949), -INT32_C(   962205318),  INT32_C(  1823022708),  INT32_C(  1900099353), -INT32_C(  1856211637),  INT32_C(   913771834),  INT32_C(     9111876),
        -INT32_C(  1491729534),  INT32_C(   435384991), -INT32_C(   807364517), -INT32_C(   801338953),  INT32_C(   188841408),  INT32_C(   765238771), -INT32_C(   278719574), -INT32_C(  1611665635) },
      {  INT32_C(  1006444555),  INT32_C(    46585802),  INT32_C(  1236957674),  INT32_C(  1823022708),  INT32_C(  1900099353),  INT32_C(  1602641628),  INT32_C(   913771834), -INT32_C(  1417267040),
        -INT32_C(  1491729534),  INT32_C(   435384991), -INT32_C(   807364517), -INT32_C(   801338953),  INT32_C(   188841408),  INT32_C(   765238771), -INT32_C(   424557498),  INT32_C(  1435645948) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i a = simde_mm512_loadu_epi32(test_vec[i].a);
    simde__m512i r0 = simde_mm512_loadu_epi32(test_vec[i].r0);
    int32_t r1[sizeof(simde__m512i) / sizeof(int32_t)];
    simde_mm512_storeu_epi32(r1, r0);
    simde_mm512_mask_storeu_epi32(r1, test_vec[i].k, a);
    simde_assert_equal_vi32(sizeof(r1) / sizeof(r1[0]), r1, test_vec[i].r1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    simde__mmask16 k = simde_test_x86_random_mmask16();
    simde__m512i a = simde_test_x86_random_i32x16();
    simde__m512i r0 = simde_test_x86_random_i32x16();
    int32_t r1[sizeof(simde__m512i) / sizeof(int32_t)];
    simde_mm512_storeu_epi32(r1, r0);
    simde_mm512_mask_storeu_epi32(r1, k, a);

    simde_test_x86_write_mmask16(2, k, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i32x16(2, a, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_i32x16(2, r0, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_codegen_write_vi32(2, sizeof(r1) / sizeof(r1[0]), r1, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm512_mask_storeu_epi64 (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const simde__mmask8 k;
    const int64_t a[8];
    const int64_t r0[8];
    const int64_t r1[8];
  } test_vec[] = {
    { UINT8_C(116),
      {  INT64_C( 7821136857127087205), -INT64_C( 3086715412299973979),  INT64_C( 3804833830256720094), -INT64_C( 6779582939625596871),
        -INT64_C( 7460854953210433407),  INT64_C( 4482921526137428145), -INT64_C( 4311144913225969983),  INT64_C( 1436969966937734420) },
      { -INT64_C( 8691785833942081835),  INT64_C(  776904232638398250), -INT64_C( 8684298580950665428), -INT64_C( 5886546827589208663),
         INT64_C( 2829059901202263014),  INT64_C( 8496174055210166934), -INT64_C( 4671737947774719696), -INT64_C( 5772407693822459556) },
      { -INT64_C( 8691785833942081835),  INT64_C(  776904232638398250),  INT64_C( 3804833830256720094), -INT64_C( 5886546827589208663),
        -INT64_C( 7460854953210433407),  INT64_C( 4482921526137428145), -INT64_C( 4311144913225969983), -INT64_C( 5772407693822459556) } },
    { UINT8_C(180),
      { -INT64_C( 2898109596814972325),  INT64_C( 8661150570052101769),  INT64_C(   55817804788454850),  INT64_C( 4343291800871107656),
         INT64_C( 6514234555907307747), -INT64_C( 2242353778674333477),  INT64_C(  388454972989796196), -INT64_C( 8562786932942754291) },
      { -INT64_C( 6340925555082684202),  INT64_C( 5766846043464490249),  INT64_C( 8080893019910591074),  INT64_C( 8104097352976940103),
         INT64_C( 1915744057269129249), -INT64_C(  166270779552323468), -INT64_C( 4220362687602924133),  INT64_C( 8649491115070967294) },
      { -INT64_C( 6340925555082684202),  INT64_C( 5766846043464490249),  INT64_C(   55817804788454850),  INT64_C( 8104097352976940103),
         INT64_C( 6514234555907307747), -INT64_C( 2242353778674333477), -INT64_C( 4220362687602924133), -INT64_C( 8562786932942754291) } },
    { UINT8_C(186),
      { -INT64_C( 6508573386203175136),  INT64_C( 2525462478712737064),  INT64_C( 5087773971783486043),  INT64_C( 6845213315809752090),
        -INT64_C( 2801446013656183940), -INT64_C(  490515647562096560),  INT64_C( 6133675350625276768),  INT64_C( 5621224410600619110) },
      {  INT64_C( 3238061589546965348), -INT64_C( 1090315789711820473),  INT64_C(  923212250356479039),  INT64_C( 7405239410954889636),
        -INT64_C( 4948493423222158261),  INT64_C( 4045780548568780196),  INT64_C( 8909670200818986739),  INT64_C( 6004623253148095979) },
      {  INT64_C( 3238061589546965348),  INT64_C( 2525462478712737064),  INT64_C(  923212250356479039),  INT64_C( 6845213315809752090),
        -INT64_C( 2801446013656183940), -INT64_C(  490515647562096560),  INT64_C( 8909670200818986739),  INT64_C( 5621224410600619110) } },
    { UINT8_C( 65),
      {  INT64_C( 6692191602363169744),  INT64_C( 2847274667785655312),  INT64_C( 1705591860769987647), -INT64_C( 4800146790523365013),
        -INT64_C( 7486997402512517138),  INT64_C( 4128862949717011947),  INT64_C( 3503982773641300679),  INT64_C( 2443399830199975998) },
      { -INT64_C( 6635219829452316469), -INT64_C( 8964673733981105565),  INT64_C( 8454291377827385694),  INT64_C( 2504433877244814445),
        -INT64_C(  520905092053492118),  INT64_C(   50082347400727921), -INT64_C( 2870070132529185153), -INT64_C( 4569964678440498150) },
      {  INT64_C( 6692191602363169744), -INT64_C( 8964673733981105565),  INT64_C( 8454291377827385694),  INT64_C( 2504433877244814445),
        -INT64_C(  520905092053492118),  INT64_C(   50082347400727921),  INT64_C( 3503982773641300679), -INT64_C( 4569964678440498150) } },
    { UINT8_C(215),
      {  INT64_C( 3880524699107385865),  INT64_C( 8585785832501545146), -INT64_C( 3131767520459825927), -INT64_C( 7474873054805689550),
        -INT64_C( 3627128247784788020),  INT64_C( 5698068749848182754), -INT64_C( 2065946406161484169),  INT64_C( 3849048715227463235) },
      {  INT64_C( 2891401418481691456), -INT64_C( 6379393745304644142), -INT64_C( 4440131970103036211), -INT64_C( 9108249590064893959),
         INT64_C( 7410943205808882026),  INT64_C( 4711334154792246410), -INT64_C(  180953205323101892),  INT64_C( 8976509763770651343) },
      {  INT64_C( 3880524699107385865),  INT64_C( 8585785832501545146), -INT64_C( 3131767520459825927), -INT64_C( 9108249590064893959),
        -INT64_C( 3627128247784788020),  INT64_C( 4711334154792246410), -INT64_C( 2065946406161484169),  INT64_C( 3849048715227463235) } },
    { UINT8_C( 57),
      { -INT64_C( 6852452214191084441), -INT64_C( 8357743008756227647),  INT64_C( 5621493144824626708),  INT64_C(  945717056158831854),
        -INT64_C( 2112663076595962615),  INT64_C( 3135257509509055218), -INT64_C( 8146058858874334614),  INT64_C( 4118986268563755894) },
      { -INT64_C( 3061334063181294719),  INT64_C( 4825692594288405686),  INT64_C( 3498975552439080799),  INT64_C( 1772296845881312047),
        -INT64_C( 8520179564595782135),  INT64_C( 4216582949960125801),  INT64_C( 8615887775330888871),  INT64_C( 6807875099733450443) },
      { -INT64_C( 6852452214191084441),  INT64_C( 4825692594288405686),  INT64_C( 3498975552439080799),  INT64_C(  945717056158831854),
        -INT64_C( 2112663076595962615),  INT64_C( 3135257509509055218),  INT64_C( 8615887775330888871),  INT64_C( 6807875099733450443) } },
    { UINT8_C( 76),
      { -INT64_C( 7216269573415285964),  INT64_C( 1006185773633248256), -INT64_C( 4071591709004965910), -INT64_C( 8116195322409197406),
         INT64_C( 2697896032800373958), -INT64_C( 9176810371150387465),  INT64_C( 7670668961976089539), -INT64_C( 3351849051435134831) },
      {  INT64_C( 5521156224548502356),  INT64_C( 4761020036613936380), -INT64_C( 8739689221168675707),  INT64_C( 6687543119559406066),
        -INT64_C( 8788082484109216664), -INT64_C(  195015463840190839), -INT64_C( 8220757502408902289), -INT64_C( 5650230511440077775) },
      {  INT64_C( 5521156224548502356),  INT64_C( 4761020036613936380), -INT64_C( 4071591709004965910), -INT64_C( 8116195322409197406),
        -INT64_C( 8788082484109216664), -INT64_C(  195015463840190839),  INT64_C( 7670668961976089539), -INT64_C( 5650230511440077775) } },
    { UINT8_C(205),
      {  INT64_C( 7452013144196017772), -INT64_C(  607742141013199881),  INT64_C( 4204427865645904935), -INT64_C( 7579219347113798741),
        -INT64_C(   35217322617132637),  INT64_C( 7352884226097701461),  INT64_C( 7619781325499506085),  INT64_C( 1817552603889615481) },
      { -INT64_C( 1769994288957640600), -INT64_C( 1820011425482240231), -INT64_C( 5996217232036049758), -INT64_C( 2451497007277197718),
         INT64_C(   50310249588797240),  INT64_C( 6084952102157266251),  INT64_C( 8017108844846373987),  INT64_C( 5838716913853698485) },
      {  INT64_C( 7452013144196017772), -INT64_C( 1820011425482240231),  INT64_C( 4204427865645904935), -INT64_C( 7579219347113798741),
         INT64_C(   50310249588797240),  INT64_C( 6084952102157266251),  INT64_C( 7619781325499506085),  INT64_C( 1817552603889615481) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i a = simde_mm512_loadu_epi64(test_vec[i].a);
    simde__m512i r0 = simde_mm512_loadu_epi64(test_vec[i].r0);
    int64_t r1[sizeof(simde__m512i) / sizeof(int64_t)];
    simde_mm512_storeu_epi64(r1, r0);
    simde_mm512_mask_storeu_epi64(r1, test_vec[i].k, a);
    simde_assert_equal_vi64(sizeof(r1) / sizeof(r1[0]), r1, test_vec[i].r1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    simde__mmask8 k = simde_test_x86_random_mmask8();
    simde__m512i a = simde_test_x86_random_i64x8();
    simde__m512i r0 = simde_test_x86_random_i64x8();
    int64_t r1[sizeof(simde__m512i) / sizeof(int64_t)];
    simde_mm512_storeu_epi64(r1, r0);
    simde_mm512_mask_storeu_epi64(r1, k, a);

    simde_test_x86_write_mmask8(2, k, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i64x8(2, a, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_i64x8(2, r0, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_codegen_write_vi64(2, sizeof(r1) / sizeof(r1[0]), r1, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm512_storeu_ps (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const simde_float32 a[16];
    const simde_float32 r[16];
  } test_vec[] = {
    { { SIMDE_FLOAT32_C(   397.85), SIMDE_FLOAT32_C(   280.50), SIMDE_FLOAT32_C(  -482.10), SIMDE_FLOAT32_C(  -764.38),
        SIMDE_FLOAT32_C(   375.26), SIMDE_FLOAT32_C(  -613.57), SIMDE_FLOAT32_C(    56.03), SIMDE_FLOAT32_C(   417.16),
        SIMDE_FLOAT32_C(  -424.36), SIMDE_FLOAT32_C(    64.48), SIMDE_FLOAT32_C(     0.84), SIMDE_FLOAT32_C(   101.24),
        SIMDE_FLOAT32_C(  -965.83), SIMDE_FLOAT32_C(   916.49), SIMDE_FLOAT32_C(   799.09), SIMDE_FLOAT32_C(   628.08) },
      { SIMDE_FLOAT32_C(   397.85), SIMDE_FLOAT32_C(   280.50), SIMDE_FLOAT32_C(  -482.10), SIMDE_FLOAT32_C(  -764.38),
        SIMDE_FLOAT32_C(   375.26), SIMDE_FLOAT32_C(  -613.57), SIMDE_FLOAT32_C(    56.03), SIMDE_FLOAT32_C(   417.16),
        SIMDE_FLOAT32_C(  -424.36), SIMDE_FLOAT32_C(    64.48), SIMDE_FLOAT32_C(     0.84), SIMDE_FLOAT32_C(   101.24),
        SIMDE_FLOAT32_C(  -965.83), SIMDE_FLOAT32_C(   916.49), SIMDE_FLOAT32_C(   799.09), SIMDE_FLOAT32_C(   628.08) } },
    { { SIMDE_FLOAT32_C(  -588.70), SIMDE_FLOAT32_C(   688.61), SIMDE_FLOAT32_C(   202.01), SIMDE_FLOAT32_C(  -610.64),
        SIMDE_FLOAT32_C(   838.07), SIMDE_FLOAT32_C(  -733.40), SIMDE_FLOAT32_C(  -127.00), SIMDE_FLOAT32_C(   993.35),
        SIMDE_FLOAT32_C(  -249.66), SIMDE_FLOAT32_C(   -45.23), SIMDE_FLOAT32_C(   849.71), SIMDE_FLOAT32_C(   -85.52),
        SIMDE_FLOAT32_C(   193.59), SIMDE_FLOAT32_C(  -257.46), SIMDE_FLOAT32_C(   827.23), SIMDE_FLOAT32_C(  -408.56) },
      { SIMDE_FLOAT32_C(  -588.70), SIMDE_FLOAT32_C(   688.61), SIMDE_FLOAT32_C(   202.01), SIMDE_FLOAT32_C(  -610.64),
        SIMDE_FLOAT32_C(   838.07), SIMDE_FLOAT32_C(  -733.40), SIMDE_FLOAT32_C(  -127.00), SIMDE_FLOAT32_C(   993.35),
        SIMDE_FLOAT32_C(  -249.66), SIMDE_FLOAT32_C(   -45.23), SIMDE_FLOAT32_C(   849.71), SIMDE_FLOAT32_C(   -85.52),
        SIMDE_FLOAT32_C(   193.59), SIMDE_FLOAT32_C(  -257.46), SIMDE_FLOAT32_C(   827.23), SIMDE_FLOAT32_C(  -408.56) } },
    { { SIMDE_FLOAT32_C(  -976.96), SIMDE_FLOAT32_C(  -654.87), SIMDE_FLOAT32_C(  -172.94), SIMDE_FLOAT32_C(   398.29),
        SIMDE_FLOAT32_C(  -268.45), SIMDE_FLOAT32_C(   883.09), SIMDE_FLOAT32_C(  -184.55), SIMDE_FLOAT32_C(   307.20),
        SIMDE_FLOAT32_C(   -52.43), SIMDE_FLOAT32_C(   816.29), SIMDE_FLOAT32_C(  -591.56), SIMDE_FLOAT32_C(   -18.26),
        SIMDE_FLOAT32_C(   732.78), SIMDE_FLOAT32_C(  -792.48), SIMDE_FLOAT32_C(  -390.18), SIMDE_FLOAT32_C(  -855.92) },
      { SIMDE_FLOAT32_C(  -976.96), SIMDE_FLOAT32_C(  -654.87), SIMDE_FLOAT32_C(  -172.94), SIMDE_FLOAT32_C(   398.29),
        SIMDE_FLOAT32_C(  -268.45), SIMDE_FLOAT32_C(   883.09), SIMDE_FLOAT32_C(  -184.55), SIMDE_FLOAT32_C(   307.20),
        SIMDE_FLOAT32_C(   -52.43), SIMDE_FLOAT32_C(   816.29), SIMDE_FLOAT32_C(  -591.56), SIMDE_FLOAT32_C(   -18.26),
        SIMDE_FLOAT32_C(   732.78), SIMDE_FLOAT32_C(  -792.48), SIMDE_FLOAT32_C(  -390.18), SIMDE_FLOAT32_C(  -855.92) } },
    { { SIMDE_FLOAT32_C(   896.13), SIMDE_FLOAT32_C(   811.83), SIMDE_FLOAT32_C(  -466.56), SIMDE_FLOAT32_C(   734.20),
        SIMDE_FLOAT32_C(  -921.57), SIMDE_FLOAT32_C(   406.44), SIMDE_FLOAT32_C(   727.55), SIMDE_FLOAT32_C(  -171.23),
        SIMDE_FLOAT32_C(  -638.79), SIMDE_FLOAT32_C(   577.26), SIMDE_FLOAT32_C(   743.25), SIMDE_FLOAT32_C(   554.80),
        SIMDE_FLOAT32_C(  -680.21), SIMDE_FLOAT32_C(   570.48), SIMDE_FLOAT32_C(  -853.75), SIMDE_FLOAT32_C(  -657.17) },
      { SIMDE_FLOAT32_C(   896.13), SIMDE_FLOAT32_C(   811.83), SIMDE_FLOAT32_C(  -466.56), SIMDE_FLOAT32_C(   734.20),
        SIMDE_FLOAT32_C(  -921.57), SIMDE_FLOAT32_C(   406.44), SIMDE_FLOAT32_C(   727.55), SIMDE_FLOAT32_C(  -171.23),
        SIMDE_FLOAT32_C(  -638.79), SIMDE_FLOAT32_C(   577.26), SIMDE_FLOAT32_C(   743.25), SIMDE_FLOAT32_C(   554.80),
        SIMDE_FLOAT32_C(  -680.21), SIMDE_FLOAT32_C(   570.48), SIMDE_FLOAT32_C(  -853.75), SIMDE_FLOAT32_C(  -657.17) } },
    { { SIMDE_FLOAT32_C(   915.61), SIMDE_FLOAT32_C(   -26.70), SIMDE_FLOAT32_C(   741.12), SIMDE_FLOAT32_C(  -352.84),
        SIMDE_FLOAT32_C(  -143.61), SIMDE_FLOAT32_C(  -443.43), SIMDE_FLOAT32_C(   954.36), SIMDE_FLOAT32_C(   803.96),
        SIMDE_FLOAT32_C(  -627.14), SIMDE_FLOAT32_C(  -637.21), SIMDE_FLOAT32_C(  -214.30), SIMDE_FLOAT32_C(  -894.36),
        SIMDE_FLOAT32_C(  -429.68), SIMDE_FLOAT32_C(   395.52), SIMDE_FLOAT32_C(  -750.28), SIMDE_FLOAT32_C(  -533.55) },
      { SIMDE_FLOAT32_C(   915.61), SIMDE_FLOAT32_C(   -26.70), SIMDE_FLOAT32_C(   741.12), SIMDE_FLOAT32_C(  -352.84),
        SIMDE_FLOAT32_C(  -143.61), SIMDE_FLOAT32_C(  -443.43), SIMDE_FLOAT32_C(   954.36), SIMDE_FLOAT32_C(   803.96),
        SIMDE_FLOAT32_C(  -627.14), SIMDE_FLOAT32_C(  -637.21), SIMDE_FLOAT32_C(  -214.30), SIMDE_FLOAT32_C(  -894.36),
        SIMDE_FLOAT32_C(  -429.68), SIMDE_FLOAT32_C(   395.52), SIMDE_FLOAT32_C(  -750.28), SIMDE_FLOAT32_C(  -533.55) } },
    { { SIMDE_FLOAT32_C(   207.35), SIMDE_FLOAT32_C(  -216.84), SIMDE_FLOAT32_C(  -799.36), SIMDE_FLOAT32_C(   285.78),
        SIMDE_FLOAT32_C(  -810.40), SIMDE_FLOAT32_C(   928.19), SIMDE_FLOAT32_C(  -885.45), SIMDE_FLOAT32_C(  -449.19),
        SIMDE_FLOAT32_C(   505.45), SIMDE_FLOAT32_C(   857.81), SIMDE_FLOAT32_C(  -894.39), SIMDE_FLOAT32_C(   825.24),
        SIMDE_FLOAT32_C(   428.29), SIMDE_FLOAT32_C(  -748.14), SIMDE_FLOAT32_C(  -831.93), SIMDE_FLOAT32_C(   343.89) },
      { SIMDE_FLOAT32_C(   207.35), SIMDE_FLOAT32_C(  -216.84), SIMDE_FLOAT32_C(  -799.36), SIMDE_FLOAT32_C(   285.78),
        SIMDE_FLOAT32_C(  -810.40), SIMDE_FLOAT32_C(   928.19), SIMDE_FLOAT32_C(  -885.45), SIMDE_FLOAT32_C(  -449.19),
        SIMDE_FLOAT32_C(   505.45), SIMDE_FLOAT32_C(   857.81), SIMDE_FLOAT32_C(  -894.39), SIMDE_FLOAT32_C(   825.24),
        SIMDE_FLOAT32_C(   428.29), SIMDE_FLOAT32_C(  -748.14), SIMDE_FLOAT32_C(  -831.93), SIMDE_FLOAT32_C(   343.89) } },
    { { SIMDE_FLOAT32_C(   225.16), SIMDE_FLOAT32_C(   909.19), SIMDE_FLOAT32_C(   991.05), SIMDE_FLOAT32_C(  -918.45),
        SIMDE_FLOAT32_C(  -534.23), SIMDE_FLOAT32_C(   945.41), SIMDE_FLOAT32_C(   885.51), SIMDE_FLOAT32_C(  -161.37),
        SIMDE_FLOAT32_C(  -691.80), SIMDE_FLOAT32_C(  -328.80), SIMDE_FLOAT32_C(   -55.73), SIMDE_FLOAT32_C(  -121.48),
        SIMDE_FLOAT32_C(  -933.28), SIMDE_FLOAT32_C(   193.99), SIMDE_FLOAT32_C(   344.96), SIMDE_FLOAT32_C(   274.08) },
      { SIMDE_FLOAT32_C(   225.16), SIMDE_FLOAT32_C(   909.19), SIMDE_FLOAT32_C(   991.05), SIMDE_FLOAT32_C(  -918.45),
        SIMDE_FLOAT32_C(  -534.23), SIMDE_FLOAT32_C(   945.41), SIMDE_FLOAT32_C(   885.51), SIMDE_FLOAT32_C(  -161.37),
        SIMDE_FLOAT32_C(  -691.80), SIMDE_FLOAT32_C(  -328.80), SIMDE_FLOAT32_C(   -55.73), SIMDE_FLOAT32_C(  -121.48),
        SIMDE_FLOAT32_C(  -933.28), SIMDE_FLOAT32_C(   193.99), SIMDE_FLOAT32_C(   344.96), SIMDE_FLOAT32_C(   274.08) } },
    { { SIMDE_FLOAT32_C(   977.14), SIMDE_FLOAT32_C(   545.61), SIMDE_FLOAT32_C(  -440.14), SIMDE_FLOAT32_C(  -833.26),
        SIMDE_FLOAT32_C(   473.80), SIMDE_FLOAT32_C(  -325.59), SIMDE_FLOAT32_C(  -282.45), SIMDE_FLOAT32_C(   -20.75),
        SIMDE_FLOAT32_C(  -467.78), SIMDE_FLOAT32_C(  -176.84), SIMDE_FLOAT32_C(  -195.51), SIMDE_FLOAT32_C(   960.51),
        SIMDE_FLOAT32_C(    75.02), SIMDE_FLOAT32_C(   -27.44), SIMDE_FLOAT32_C(   304.40), SIMDE_FLOAT32_C(  -699.82) },
      { SIMDE_FLOAT32_C(   977.14), SIMDE_FLOAT32_C(   545.61), SIMDE_FLOAT32_C(  -440.14), SIMDE_FLOAT32_C(  -833.26),
        SIMDE_FLOAT32_C(   473.80), SIMDE_FLOAT32_C(  -325.59), SIMDE_FLOAT32_C(  -282.45), SIMDE_FLOAT32_C(   -20.75),
        SIMDE_FLOAT32_C(  -467.78), SIMDE_FLOAT32_C(  -176.84), SIMDE_FLOAT32_C(  -195.51), SIMDE_FLOAT32_C(   960.51),
        SIMDE_FLOAT32_C(    75.02), SIMDE_FLOAT32_C(   -27.44), SIMDE_FLOAT32_C(   304.40), SIMDE_FLOAT32_C(  -699.82) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512 a = simde_mm512_loadu_ps(test_vec[i].a);
    simde_float32 r[sizeof(simde__m512) / sizeof(simde_float32)];
    simde_mm512_storeu_ps(r, a);
    simde_assert_equal_vf32(sizeof(r) / sizeof(r[0]), r, test_vec[i].r, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    simde__m512 a = simde_test_x86_random_f32x16(SIMDE_FLOAT32_C(-1000.0), SIMDE_FLOAT32_C(1000.0));
    simde_float32 r[sizeof(simde__m512) / sizeof(simde_float32)];
    simde_mm512_storeu_ps(r, a);

    simde_test_x86_write_f32x16(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_codegen_write_vf32(2, sizeof(r) / sizeof(r[0]), r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

#if defined(SIMDE_FLOAT16_IS_SCALAR)
static int
test_simde_mm512_storeu_ph (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const simde_float16 a[32];
    const simde_float16 r[32];
  } test_vec[] = {
    { { SIMDE_FLOAT16_VALUE(   614.50), SIMDE_FLOAT16_VALUE(   397.50), SIMDE_FLOAT16_VALUE(   743.00), SIMDE_FLOAT16_VALUE(   963.00),
        SIMDE_FLOAT16_VALUE(   996.00), SIMDE_FLOAT16_VALUE(  -396.25), SIMDE_FLOAT16_VALUE(   558.50), SIMDE_FLOAT16_VALUE(  -368.25),
        SIMDE_FLOAT16_VALUE(   209.25), SIMDE_FLOAT16_VALUE(   440.00), SIMDE_FLOAT16_VALUE(  -763.50), SIMDE_FLOAT16_VALUE(   731.00),
        SIMDE_FLOAT16_VALUE(  -291.50), SIMDE_FLOAT16_VALUE(  -477.50), SIMDE_FLOAT16_VALUE(   439.25), SIMDE_FLOAT16_VALUE(  -934.50),
        SIMDE_FLOAT16_VALUE(   999.50), SIMDE_FLOAT16_VALUE(  -494.75), SIMDE_FLOAT16_VALUE(    79.06), SIMDE_FLOAT16_VALUE(   249.88),
        SIMDE_FLOAT16_VALUE(  -358.00), SIMDE_FLOAT16_VALUE(   242.00), SIMDE_FLOAT16_VALUE(  -751.50), SIMDE_FLOAT16_VALUE(   483.25),
        SIMDE_FLOAT16_VALUE(  -873.00), SIMDE_FLOAT16_VALUE(   974.50), SIMDE_FLOAT16_VALUE(  -845.50), SIMDE_FLOAT16_VALUE(   801.00),
        SIMDE_FLOAT16_VALUE(  -563.50), SIMDE_FLOAT16_VALUE(    43.72), SIMDE_FLOAT16_VALUE(  -969.50), SIMDE_FLOAT16_VALUE(  -949.00) },
      { SIMDE_FLOAT16_VALUE(   614.50), SIMDE_FLOAT16_VALUE(   397.50), SIMDE_FLOAT16_VALUE(   743.00), SIMDE_FLOAT16_VALUE(   963.00),
        SIMDE_FLOAT16_VALUE(   996.00), SIMDE_FLOAT16_VALUE(  -396.25), SIMDE_FLOAT16_VALUE(   558.50), SIMDE_FLOAT16_VALUE(  -368.25),
        SIMDE_FLOAT16_VALUE(   209.25), SIMDE_FLOAT16_VALUE(   440.00), SIMDE_FLOAT16_VALUE(  -763.50), SIMDE_FLOAT16_VALUE(   731.00),
        SIMDE_FLOAT16_VALUE(  -291.50), SIMDE_FLOAT16_VALUE(  -477.50), SIMDE_FLOAT16_VALUE(   439.25), SIMDE_FLOAT16_VALUE(  -934.50),
        SIMDE_FLOAT16_VALUE(   999.50), SIMDE_FLOAT16_VALUE(  -494.75), SIMDE_FLOAT16_VALUE(    79.06), SIMDE_FLOAT16_VALUE(   249.88),
        SIMDE_FLOAT16_VALUE(  -358.00), SIMDE_FLOAT16_VALUE(   242.00), SIMDE_FLOAT16_VALUE(  -751.50), SIMDE_FLOAT16_VALUE(   483.25),
        SIMDE_FLOAT16_VALUE(  -873.00), SIMDE_FLOAT16_VALUE(   974.50), SIMDE_FLOAT16_VALUE(  -845.50), SIMDE_FLOAT16_VALUE(   801.00),
        SIMDE_FLOAT16_VALUE(  -563.50), SIMDE_FLOAT16_VALUE(    43.72), SIMDE_FLOAT16_VALUE(  -969.50), SIMDE_FLOAT16_VALUE(  -949.00) } },
    { { SIMDE_FLOAT16_VALUE(  -559.00), SIMDE_FLOAT16_VALUE(   773.50), SIMDE_FLOAT16_VALUE(  -985.50), SIMDE_FLOAT16_VALUE(  -562.50),
        SIMDE_FLOAT16_VALUE(  -622.50), SIMDE_FLOAT16_VALUE(   572.50), SIMDE_FLOAT16_VALUE(    69.19), SIMDE_FLOAT16_VALUE(   586.50),
        SIMDE_FLOAT16_VALUE(    12.58), SIMDE_FLOAT16_VALUE(   305.75), SIMDE_FLOAT16_VALUE(   317.75), SIMDE_FLOAT16_VALUE(   721.00),
        SIMDE_FLOAT16_VALUE(   828.50), SIMDE_FLOAT16_VALUE(  -243.00), SIMDE_FLOAT16_VALUE(   786.50), SIMDE_FLOAT16_VALUE(   828.00),
        SIMDE_FLOAT16_VALUE(   262.25), SIMDE_FLOAT16_VALUE(  -134.50), SIMDE_FLOAT16_VALUE(    77.88), SIMDE_FLOAT16_VALUE(   904.00),
        SIMDE_FLOAT16_VALUE(  -892.50), SIMDE_FLOAT16_VALUE(   326.00), SIMDE_FLOAT16_VALUE(   387.50), SIMDE_FLOAT16_VALUE(  -765.50),
        SIMDE_FLOAT16_VALUE(   300.50), SIMDE_FLOAT16_VALUE(   542.00), SIMDE_FLOAT16_VALUE(  -965.00), SIMDE_FLOAT16_VALUE(   737.00),
        SIMDE_FLOAT16_VALUE(  -414.00), SIMDE_FLOAT16_VALUE(  -934.50), SIMDE_FLOAT16_VALUE(   788.00), SIMDE_FLOAT16_VALUE(    27.16) },
      { SIMDE_FLOAT16_VALUE(  -559.00), SIMDE_FLOAT16_VALUE(   773.50), SIMDE_FLOAT16_VALUE(  -985.50), SIMDE_FLOAT16_VALUE(  -562.50),
        SIMDE_FLOAT16_VALUE(  -622.50), SIMDE_FLOAT16_VALUE(   572.50), SIMDE_FLOAT16_VALUE(    69.19), SIMDE_FLOAT16_VALUE(   586.50),
        SIMDE_FLOAT16_VALUE(    12.58), SIMDE_FLOAT16_VALUE(   305.75), SIMDE_FLOAT16_VALUE(   317.75), SIMDE_FLOAT16_VALUE(   721.00),
        SIMDE_FLOAT16_VALUE(   828.50), SIMDE_FLOAT16_VALUE(  -243.00), SIMDE_FLOAT16_VALUE(   786.50), SIMDE_FLOAT16_VALUE(   828.00),
        SIMDE_FLOAT16_VALUE(   262.25), SIMDE_FLOAT16_VALUE(  -134.50), SIMDE_FLOAT16_VALUE(    77.88), SIMDE_FLOAT16_VALUE(   904.00),
        SIMDE_FLOAT16_VALUE(  -892.50), SIMDE_FLOAT16_VALUE(   326.00), SIMDE_FLOAT16_VALUE(   387.50), SIMDE_FLOAT16_VALUE(  -765.50),
        SIMDE_FLOAT16_VALUE(   300.50), SIMDE_FLOAT16_VALUE(   542.00), SIMDE_FLOAT16_VALUE(  -965.00), SIMDE_FLOAT16_VALUE(   737.00),
        SIMDE_FLOAT16_VALUE(  -414.00), SIMDE_FLOAT16_VALUE(  -934.50), SIMDE_FLOAT16_VALUE(   788.00), SIMDE_FLOAT16_VALUE(    27.16) } },
    { { SIMDE_FLOAT16_VALUE(   839.00), SIMDE_FLOAT16_VALUE(   802.50), SIMDE_FLOAT16_VALUE(   464.50), SIMDE_FLOAT16_VALUE(  -783.50),
        SIMDE_FLOAT16_VALUE(   375.00), SIMDE_FLOAT16_VALUE(  -466.25), SIMDE_FLOAT16_VALUE(   803.00), SIMDE_FLOAT16_VALUE(  -612.50),
        SIMDE_FLOAT16_VALUE(   839.50), SIMDE_FLOAT16_VALUE(   120.81), SIMDE_FLOAT16_VALUE(  -891.00), SIMDE_FLOAT16_VALUE(   668.00),
        SIMDE_FLOAT16_VALUE(   878.00), SIMDE_FLOAT16_VALUE(   895.00), SIMDE_FLOAT16_VALUE(   495.75), SIMDE_FLOAT16_VALUE(   140.00),
        SIMDE_FLOAT16_VALUE(  -239.25), SIMDE_FLOAT16_VALUE(  -426.50), SIMDE_FLOAT16_VALUE(    44.31), SIMDE_FLOAT16_VALUE(  -131.75),
        SIMDE_FLOAT16_VALUE(   899.50), SIMDE_FLOAT16_VALUE(  -568.00), SIMDE_FLOAT16_VALUE(   102.56), SIMDE_FLOAT16_VALUE(   200.00),
        SIMDE_FLOAT16_VALUE(   974.00), SIMDE_FLOAT16_VALUE(   137.75), SIMDE_FLOAT16_VALUE(   -62.97), SIMDE_FLOAT16_VALUE(  -440.00),
        SIMDE_FLOAT16_VALUE(   203.38), SIMDE_FLOAT16_VALUE(  -275.00), SIMDE_FLOAT16_VALUE(   587.00), SIMDE_FLOAT16_VALUE(    42.59) },
      { SIMDE_FLOAT16_VALUE(   839.00), SIMDE_FLOAT16_VALUE(   802.50), SIMDE_FLOAT16_VALUE(   464.50), SIMDE_FLOAT16_VALUE(  -783.50),
        SIMDE_FLOAT16_VALUE(   375.00), SIMDE_FLOAT16_VALUE(  -466.25), SIMDE_FLOAT16_VALUE(   803.00), SIMDE_FLOAT16_VALUE(  -612.50),
        SIMDE_FLOAT16_VALUE(   839.50), SIMDE_FLOAT16_VALUE(   120.81), SIMDE_FLOAT16_VALUE(  -891.00), SIMDE_FLOAT16_VALUE(   668.00),
        SIMDE_FLOAT16_VALUE(   878.00), SIMDE_FLOAT16_VALUE(   895.00), SIMDE_FLOAT16_VALUE(   495.75), SIMDE_FLOAT16_VALUE(   140.00),
        SIMDE_FLOAT16_VALUE(  -239.25), SIMDE_FLOAT16_VALUE(  -426.50), SIMDE_FLOAT16_VALUE(    44.31), SIMDE_FLOAT16_VALUE(  -131.75),
        SIMDE_FLOAT16_VALUE(   899.50), SIMDE_FLOAT16_VALUE(  -568.00), SIMDE_FLOAT16_VALUE(   102.56), SIMDE_FLOAT16_VALUE(   200.00),
        SIMDE_FLOAT16_VALUE(   974.00), SIMDE_FLOAT16_VALUE(   137.75), SIMDE_FLOAT16_VALUE(   -62.97), SIMDE_FLOAT16_VALUE(  -440.00),
        SIMDE_FLOAT16_VALUE(   203.38), SIMDE_FLOAT16_VALUE(  -275.00), SIMDE_FLOAT16_VALUE(   587.00), SIMDE_FLOAT16_VALUE(    42.59) } },
    { { SIMDE_FLOAT16_VALUE(  -472.50), SIMDE_FLOAT16_VALUE(    51.56), SIMDE_FLOAT16_VALUE(   259.00), SIMDE_FLOAT16_VALUE(   902.50),
        SIMDE_FLOAT16_VALUE(   585.00), SIMDE_FLOAT16_VALUE(    62.31), SIMDE_FLOAT16_VALUE(  -710.00), SIMDE_FLOAT16_VALUE(   424.75),
        SIMDE_FLOAT16_VALUE(  -817.00), SIMDE_FLOAT16_VALUE(  -601.00), SIMDE_FLOAT16_VALUE(    92.44), SIMDE_FLOAT16_VALUE(  -939.00),
        SIMDE_FLOAT16_VALUE(  -706.00), SIMDE_FLOAT16_VALUE(  -411.75), SIMDE_FLOAT16_VALUE(   200.88), SIMDE_FLOAT16_VALUE(    54.81),
        SIMDE_FLOAT16_VALUE(   161.75), SIMDE_FLOAT16_VALUE(  -755.00), SIMDE_FLOAT16_VALUE(   923.00), SIMDE_FLOAT16_VALUE(    61.38),
        SIMDE_FLOAT16_VALUE(  -323.00), SIMDE_FLOAT16_VALUE(    25.62), SIMDE_FLOAT16_VALUE(  -738.50), SIMDE_FLOAT16_VALUE(  -349.00),
        SIMDE_FLOAT16_VALUE(  -836.50), SIMDE_FLOAT16_VALUE(   198.50), SIMDE_FLOAT16_VALUE(   211.00), SIMDE_FLOAT16_VALUE(   366.75),
        SIMDE_FLOAT16_VALUE(   923.50), SIMDE_FLOAT16_VALUE(  -201.88), SIMDE_FLOAT16_VALUE(  -590.50), SIMDE_FLOAT16_VALUE(  -549.00) },
      { SIMDE_FLOAT16_VALUE(  -472.50), SIMDE_FLOAT16_VALUE(    51.56), SIMDE_FLOAT16_VALUE(   259.00), SIMDE_FLOAT16_VALUE(   902.50),
        SIMDE_FLOAT16_VALUE(   585.00), SIMDE_FLOAT16_VALUE(    62.31), SIMDE_FLOAT16_VALUE(  -710.00), SIMDE_FLOAT16_VALUE(   424.75),
        SIMDE_FLOAT16_VALUE(  -817.00), SIMDE_FLOAT16_VALUE(  -601.00), SIMDE_FLOAT16_VALUE(    92.44), SIMDE_FLOAT16_VALUE(  -939.00),
        SIMDE_FLOAT16_VALUE(  -706.00), SIMDE_FLOAT16_VALUE(  -411.75), SIMDE_FLOAT16_VALUE(   200.88), SIMDE_FLOAT16_VALUE(    54.81),
        SIMDE_FLOAT16_VALUE(   161.75), SIMDE_FLOAT16_VALUE(  -755.00), SIMDE_FLOAT16_VALUE(   923.00), SIMDE_FLOAT16_VALUE(    61.38),
        SIMDE_FLOAT16_VALUE(  -323.00), SIMDE_FLOAT16_VALUE(    25.62), SIMDE_FLOAT16_VALUE(  -738.50), SIMDE_FLOAT16_VALUE(  -349.00),
        SIMDE_FLOAT16_VALUE(  -836.50), SIMDE_FLOAT16_VALUE(   198.50), SIMDE_FLOAT16_VALUE(   211.00), SIMDE_FLOAT16_VALUE(   366.75),
        SIMDE_FLOAT16_VALUE(   923.50), SIMDE_FLOAT16_VALUE(  -201.88), SIMDE_FLOAT16_VALUE(  -590.50), SIMDE_FLOAT16_VALUE(  -549.00) } },
    { { SIMDE_FLOAT16_VALUE(   849.50), SIMDE_FLOAT16_VALUE(   668.50), SIMDE_FLOAT16_VALUE(  -646.50), SIMDE_FLOAT16_VALUE(   435.00),
        SIMDE_FLOAT16_VALUE(  -269.25), SIMDE_FLOAT16_VALUE(  -356.25), SIMDE_FLOAT16_VALUE(  -140.38), SIMDE_FLOAT16_VALUE(   -86.06),
        SIMDE_FLOAT16_VALUE(    42.69), SIMDE_FLOAT16_VALUE(   952.00), SIMDE_FLOAT16_VALUE(   -25.16), SIMDE_FLOAT16_VALUE(   336.75),
        SIMDE_FLOAT16_VALUE(  -459.75), SIMDE_FLOAT16_VALUE(  -824.00), SIMDE_FLOAT16_VALUE(  -608.50), SIMDE_FLOAT16_VALUE(   702.00),
        SIMDE_FLOAT16_VALUE(  -579.00), SIMDE_FLOAT16_VALUE(  -685.50), SIMDE_FLOAT16_VALUE(  -236.75), SIMDE_FLOAT16_VALUE(    98.06),
        SIMDE_FLOAT16_VALUE(   340.25), SIMDE_FLOAT16_VALUE(    24.69), SIMDE_FLOAT16_VALUE(   749.00), SIMDE_FLOAT16_VALUE(   503.75),
        SIMDE_FLOAT16_VALUE(  -777.00), SIMDE_FLOAT16_VALUE(   -39.78), SIMDE_FLOAT16_VALUE(  -129.50), SIMDE_FLOAT16_VALUE(  -853.00),
        SIMDE_FLOAT16_VALUE(   758.50), SIMDE_FLOAT16_VALUE(   280.00), SIMDE_FLOAT16_VALUE(  -402.25), SIMDE_FLOAT16_VALUE(   608.00) },
      { SIMDE_FLOAT16_VALUE(   849.50), SIMDE_FLOAT16_VALUE(   668.50), SIMDE_FLOAT16_VALUE(  -646.50), SIMDE_FLOAT16_VALUE(   435.00),
        SIMDE_FLOAT16_VALUE(  -269.25), SIMDE_FLOAT16_VALUE(  -356.25), SIMDE_FLOAT16_VALUE(  -140.38), SIMDE_FLOAT16_VALUE(   -86.06),
        SIMDE_FLOAT16_VALUE(    42.69), SIMDE_FLOAT16_VALUE(   952.00), SIMDE_FLOAT16_VALUE(   -25.16), SIMDE_FLOAT16_VALUE(   336.75),
        SIMDE_FLOAT16_VALUE(  -459.75), SIMDE_FLOAT16_VALUE(  -824.00), SIMDE_FLOAT16_VALUE(  -608.50), SIMDE_FLOAT16_VALUE(   702.00),
        SIMDE_FLOAT16_VALUE(  -579.00), SIMDE_FLOAT16_VALUE(  -685.50), SIMDE_FLOAT16_VALUE(  -236.75), SIMDE_FLOAT16_VALUE(    98.06),
        SIMDE_FLOAT16_VALUE(   340.25), SIMDE_FLOAT16_VALUE(    24.69), SIMDE_FLOAT16_VALUE(   749.00), SIMDE_FLOAT16_VALUE(   503.75),
        SIMDE_FLOAT16_VALUE(  -777.00), SIMDE_FLOAT16_VALUE(   -39.78), SIMDE_FLOAT16_VALUE(  -129.50), SIMDE_FLOAT16_VALUE(  -853.00),
        SIMDE_FLOAT16_VALUE(   758.50), SIMDE_FLOAT16_VALUE(   280.00), SIMDE_FLOAT16_VALUE(  -402.25), SIMDE_FLOAT16_VALUE(   608.00) } },
    { { SIMDE_FLOAT16_VALUE(   -51.50), SIMDE_FLOAT16_VALUE(   -48.50), SIMDE_FLOAT16_VALUE(    42.84), SIMDE_FLOAT16_VALUE(   679.50),
        SIMDE_FLOAT16_VALUE(   595.50), SIMDE_FLOAT16_VALUE(   902.50), SIMDE_FLOAT16_VALUE(  -406.75), SIMDE_FLOAT16_VALUE(  -362.00),
        SIMDE_FLOAT16_VALUE(   854.50), SIMDE_FLOAT16_VALUE(   568.00), SIMDE_FLOAT16_VALUE(   975.00), SIMDE_FLOAT16_VALUE(  -605.50),
        SIMDE_FLOAT16_VALUE(   744.00), SIMDE_FLOAT16_VALUE(  -633.50), SIMDE_FLOAT16_VALUE(  -903.50), SIMDE_FLOAT16_VALUE(  -835.00),
        SIMDE_FLOAT16_VALUE(  -319.00), SIMDE_FLOAT16_VALUE(  -140.25), SIMDE_FLOAT16_VALUE(   263.00), SIMDE_FLOAT16_VALUE(  -979.00),
        SIMDE_FLOAT16_VALUE(   884.50), SIMDE_FLOAT16_VALUE(    12.23), SIMDE_FLOAT16_VALUE(   525.00), SIMDE_FLOAT16_VALUE(  -892.50),
        SIMDE_FLOAT16_VALUE(   972.50), SIMDE_FLOAT16_VALUE(  -604.50), SIMDE_FLOAT16_VALUE(  -745.50), SIMDE_FLOAT16_VALUE(   731.00),
        SIMDE_FLOAT16_VALUE(   675.50), SIMDE_FLOAT16_VALUE(  -147.75), SIMDE_FLOAT16_VALUE(   338.75), SIMDE_FLOAT16_VALUE(  -376.00) },
      { SIMDE_FLOAT16_VALUE(   -51.50), SIMDE_FLOAT16_VALUE(   -48.50), SIMDE_FLOAT16_VALUE(    42.84), SIMDE_FLOAT16_VALUE(   679.50),
        SIMDE_FLOAT16_VALUE(   595.50), SIMDE_FLOAT16_VALUE(   902.50), SIMDE_FLOAT16_VALUE(  -406.75), SIMDE_FLOAT16_VALUE(  -362.00),
        SIMDE_FLOAT16_VALUE(   854.50), SIMDE_FLOAT16_VALUE(   568.00), SIMDE_FLOAT16_VALUE(   975.00), SIMDE_FLOAT16_VALUE(  -605.50),
        SIMDE_FLOAT16_VALUE(   744.00), SIMDE_FLOAT16_VALUE(  -633.50), SIMDE_FLOAT16_VALUE(  -903.50), SIMDE_FLOAT16_VALUE(  -835.00),
        SIMDE_FLOAT16_VALUE(  -319.00), SIMDE_FLOAT16_VALUE(  -140.25), SIMDE_FLOAT16_VALUE(   263.00), SIMDE_FLOAT16_VALUE(  -979.00),
        SIMDE_FLOAT16_VALUE(   884.50), SIMDE_FLOAT16_VALUE(    12.23), SIMDE_FLOAT16_VALUE(   525.00), SIMDE_FLOAT16_VALUE(  -892.50),
        SIMDE_FLOAT16_VALUE(   972.50), SIMDE_FLOAT16_VALUE(  -604.50), SIMDE_FLOAT16_VALUE(  -745.50), SIMDE_FLOAT16_VALUE(   731.00),
        SIMDE_FLOAT16_VALUE(   675.50), SIMDE_FLOAT16_VALUE(  -147.75), SIMDE_FLOAT16_VALUE(   338.75), SIMDE_FLOAT16_VALUE(  -376.00) } },
    { { SIMDE_FLOAT16_VALUE(   803.50), SIMDE_FLOAT16_VALUE(  -618.50), SIMDE_FLOAT16_VALUE(  -697.00), SIMDE_FLOAT16_VALUE(   399.00),
        SIMDE_FLOAT16_VALUE(  -716.00), SIMDE_FLOAT16_VALUE(  -103.50), SIMDE_FLOAT16_VALUE(  -963.00), SIMDE_FLOAT16_VALUE(  -861.50),
        SIMDE_FLOAT16_VALUE(  -535.50), SIMDE_FLOAT16_VALUE(  -988.50), SIMDE_FLOAT16_VALUE(  -467.00), SIMDE_FLOAT16_VALUE(  -791.50),
        SIMDE_FLOAT16_VALUE(  -622.00), SIMDE_FLOAT16_VALUE(  -370.50), SIMDE_FLOAT16_VALUE(  -626.50), SIMDE_FLOAT16_VALUE(    59.03),
        SIMDE_FLOAT16_VALUE(   489.25), SIMDE_FLOAT16_VALUE(   636.50), SIMDE_FLOAT16_VALUE(    80.25), SIMDE_FLOAT16_VALUE(   373.75),
        SIMDE_FLOAT16_VALUE(  -351.25), SIMDE_FLOAT16_VALUE(  -394.75), SIMDE_FLOAT16_VALUE(   481.25), SIMDE_FLOAT16_VALUE(  -378.75),
        SIMDE_FLOAT16_VALUE(     0.61), SIMDE_FLOAT16_VALUE(   735.50), SIMDE_FLOAT16_VALUE(  -648.00), SIMDE_FLOAT16_VALUE(  -324.00),
        SIMDE_FLOAT16_VALUE(  -412.25), SIMDE_FLOAT16_VALUE(   690.50), SIMDE_FLOAT16_VALUE(   299.75), SIMDE_FLOAT16_VALUE(  -608.50) },
      { SIMDE_FLOAT16_VALUE(   803.50), SIMDE_FLOAT16_VALUE(  -618.50), SIMDE_FLOAT16_VALUE(  -697.00), SIMDE_FLOAT16_VALUE(   399.00),
        SIMDE_FLOAT16_VALUE(  -716.00), SIMDE_FLOAT16_VALUE(  -103.50), SIMDE_FLOAT16_VALUE(  -963.00), SIMDE_FLOAT16_VALUE(  -861.50),
        SIMDE_FLOAT16_VALUE(  -535.50), SIMDE_FLOAT16_VALUE(  -988.50), SIMDE_FLOAT16_VALUE(  -467.00), SIMDE_FLOAT16_VALUE(  -791.50),
        SIMDE_FLOAT16_VALUE(  -622.00), SIMDE_FLOAT16_VALUE(  -370.50), SIMDE_FLOAT16_VALUE(  -626.50), SIMDE_FLOAT16_VALUE(    59.03),
        SIMDE_FLOAT16_VALUE(   489.25), SIMDE_FLOAT16_VALUE(   636.50), SIMDE_FLOAT16_VALUE(    80.25), SIMDE_FLOAT16_VALUE(   373.75),
        SIMDE_FLOAT16_VALUE(  -351.25), SIMDE_FLOAT16_VALUE(  -394.75), SIMDE_FLOAT16_VALUE(   481.25), SIMDE_FLOAT16_VALUE(  -378.75),
        SIMDE_FLOAT16_VALUE(     0.61), SIMDE_FLOAT16_VALUE(   735.50), SIMDE_FLOAT16_VALUE(  -648.00), SIMDE_FLOAT16_VALUE(  -324.00),
        SIMDE_FLOAT16_VALUE(  -412.25), SIMDE_FLOAT16_VALUE(   690.50), SIMDE_FLOAT16_VALUE(   299.75), SIMDE_FLOAT16_VALUE(  -608.50) } },
    { { SIMDE_FLOAT16_VALUE(  -927.50), SIMDE_FLOAT16_VALUE(   603.00), SIMDE_FLOAT16_VALUE(   790.50), SIMDE_FLOAT16_VALUE(  -644.00),
        SIMDE_FLOAT16_VALUE(  -500.50), SIMDE_FLOAT16_VALUE(   827.50), SIMDE_FLOAT16_VALUE(  -505.25), SIMDE_FLOAT16_VALUE(   -35.69),
        SIMDE_FLOAT16_VALUE(   839.00), SIMDE_FLOAT16_VALUE(    27.69), SIMDE_FLOAT16_VALUE(   172.88), SIMDE_FLOAT16_VALUE(  -783.00),
        SIMDE_FLOAT16_VALUE(   657.00), SIMDE_FLOAT16_VALUE(   546.50), SIMDE_FLOAT16_VALUE(   276.25), SIMDE_FLOAT16_VALUE(   146.50),
        SIMDE_FLOAT16_VALUE(   183.00), SIMDE_FLOAT16_VALUE(  -643.50), SIMDE_FLOAT16_VALUE(  -479.75), SIMDE_FLOAT16_VALUE(   832.00),
        SIMDE_FLOAT16_VALUE(   -38.28), SIMDE_FLOAT16_VALUE(  -998.50), SIMDE_FLOAT16_VALUE(  -547.00), SIMDE_FLOAT16_VALUE(   962.50),
        SIMDE_FLOAT16_VALUE(   737.00), SIMDE_FLOAT16_VALUE(  -194.88), SIMDE_FLOAT16_VALUE(  -361.75), SIMDE_FLOAT16_VALUE(  -675.00),
        SIMDE_FLOAT16_VALUE(  -504.00), SIMDE_FLOAT16_VALUE(   938.00), SIMDE_FLOAT16_VALUE(  -283.50), SIMDE_FLOAT16_VALUE(  -431.75) },
      { SIMDE_FLOAT16_VALUE(  -927.50), SIMDE_FLOAT16_VALUE(   603.00), SIMDE_FLOAT16_VALUE(   790.50), SIMDE_FLOAT16_VALUE(  -644.00),
        SIMDE_FLOAT16_VALUE(  -500.50), SIMDE_FLOAT16_VALUE(   827.50), SIMDE_FLOAT16_VALUE(  -505.25), SIMDE_FLOAT16_VALUE(   -35.69),
        SIMDE_FLOAT16_VALUE(   839.00), SIMDE_FLOAT16_VALUE(    27.69), SIMDE_FLOAT16_VALUE(   172.88), SIMDE_FLOAT16_VALUE(  -783.00),
        SIMDE_FLOAT16_VALUE(   657.00), SIMDE_FLOAT16_VALUE(   546.50), SIMDE_FLOAT16_VALUE(   276.25), SIMDE_FLOAT16_VALUE(   146.50),
        SIMDE_FLOAT16_VALUE(   183.00), SIMDE_FLOAT16_VALUE(  -643.50), SIMDE_FLOAT16_VALUE(  -479.75), SIMDE_FLOAT16_VALUE(   832.00),
        SIMDE_FLOAT16_VALUE(   -38.28), SIMDE_FLOAT16_VALUE(  -998.50), SIMDE_FLOAT16_VALUE(  -547.00), SIMDE_FLOAT16_VALUE(   962.50),
        SIMDE_FLOAT16_VALUE(   737.00), SIMDE_FLOAT16_VALUE(  -194.88), SIMDE_FLOAT16_VALUE(  -361.75), SIMDE_FLOAT16_VALUE(  -675.00),
        SIMDE_FLOAT16_VALUE(  -504.00), SIMDE_FLOAT16_VALUE(   938.00), SIMDE_FLOAT16_VALUE(  -283.50), SIMDE_FLOAT16_VALUE(  -431.75) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512h a = simde_mm512_loadu_ph(test_vec[i].a);
    simde_float16 r[sizeof(simde__m512) / sizeof(simde_float16)];
    simde_mm512_storeu_ph(r, a);
    simde_assert_equal_vf16(sizeof(r) / sizeof(r[0]), r, test_vec[i].r, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    simde__m512h a = simde_test_x86_random_f16x32(SIMDE_FLOAT16_VALUE(-1000.0), SIMDE_FLOAT16_VALUE(1000.0));
    simde_float16 r[sizeof(simde__m512) / sizeof(simde_float16)];
    simde_mm512_storeu_ph(r, a);

    simde_test_x86_write_f16x32(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_codegen_write_vf16(2, sizeof(r) / sizeof(r[0]), r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}
#endif

static int
test_simde_mm512_mask_storeu_ps (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const simde__mmask16 k;
    const simde_float32 a[16];
    const simde_float32 r0[16];
    const simde_float32 r1[16];
  } test_vec[] = {
    { UINT32_C(     29279),
      { SIMDE_FLOAT32_C(   404.96), SIMDE_FLOAT32_C(  -827.14), SIMDE_FLOAT32_C(    55.49), SIMDE_FLOAT32_C(  -491.16),
        SIMDE_FLOAT32_C(  -950.43), SIMDE_FLOAT32_C(   741.83), SIMDE_FLOAT32_C(   461.70), SIMDE_FLOAT32_C(  -423.16),
        SIMDE_FLOAT32_C(   -64.62), SIMDE_FLOAT32_C(   757.35), SIMDE_FLOAT32_C(   336.54), SIMDE_FLOAT32_C(  -363.50),
        SIMDE_FLOAT32_C(  -804.37), SIMDE_FLOAT32_C(  -884.59), SIMDE_FLOAT32_C(   512.84), SIMDE_FLOAT32_C(   554.50) },
      { SIMDE_FLOAT32_C(  -313.26), SIMDE_FLOAT32_C(   466.54), SIMDE_FLOAT32_C(   708.20), SIMDE_FLOAT32_C(  -626.25),
        SIMDE_FLOAT32_C(  -260.54), SIMDE_FLOAT32_C(   848.96), SIMDE_FLOAT32_C(  -785.70), SIMDE_FLOAT32_C(  -788.45),
        SIMDE_FLOAT32_C(  -637.99), SIMDE_FLOAT32_C(   655.94), SIMDE_FLOAT32_C(   -52.13), SIMDE_FLOAT32_C(   593.08),
        SIMDE_FLOAT32_C(  -997.92), SIMDE_FLOAT32_C(   354.66), SIMDE_FLOAT32_C(   937.65), SIMDE_FLOAT32_C(   407.04) },
      { SIMDE_FLOAT32_C(   404.96), SIMDE_FLOAT32_C(  -827.14), SIMDE_FLOAT32_C(    55.49), SIMDE_FLOAT32_C(  -491.16),
        SIMDE_FLOAT32_C(  -950.43), SIMDE_FLOAT32_C(   848.96), SIMDE_FLOAT32_C(   461.70), SIMDE_FLOAT32_C(  -788.45),
        SIMDE_FLOAT32_C(  -637.99), SIMDE_FLOAT32_C(   757.35), SIMDE_FLOAT32_C(   -52.13), SIMDE_FLOAT32_C(   593.08),
        SIMDE_FLOAT32_C(  -804.37), SIMDE_FLOAT32_C(  -884.59), SIMDE_FLOAT32_C(   512.84), SIMDE_FLOAT32_C(   407.04) } },
    { UINT32_C(     20173),
      { SIMDE_FLOAT32_C(   915.88), SIMDE_FLOAT32_C(   577.09), SIMDE_FLOAT32_C(  -265.03), SIMDE_FLOAT32_C(   377.58),
        SIMDE_FLOAT32_C(  -846.08), SIMDE_FLOAT32_C(   670.36), SIMDE_FLOAT32_C(   134.93), SIMDE_FLOAT32_C(   490.47),
        SIMDE_FLOAT32_C(  -693.14), SIMDE_FLOAT32_C(   330.56), SIMDE_FLOAT32_C(   605.88), SIMDE_FLOAT32_C(   819.70),
        SIMDE_FLOAT32_C(  -114.94), SIMDE_FLOAT32_C(  -707.38), SIMDE_FLOAT32_C(   286.24), SIMDE_FLOAT32_C(  -406.74) },
      { SIMDE_FLOAT32_C(  -333.63), SIMDE_FLOAT32_C(  -974.30), SIMDE_FLOAT32_C(  -557.78), SIMDE_FLOAT32_C(  -119.33),
        SIMDE_FLOAT32_C(  -762.75), SIMDE_FLOAT32_C(  -195.76), SIMDE_FLOAT32_C(  -463.39), SIMDE_FLOAT32_C(   185.12),
        SIMDE_FLOAT32_C(  -602.68), SIMDE_FLOAT32_C(  -461.32), SIMDE_FLOAT32_C(  -460.22), SIMDE_FLOAT32_C(  -665.03),
        SIMDE_FLOAT32_C(   945.72), SIMDE_FLOAT32_C(  -932.70), SIMDE_FLOAT32_C(   328.11), SIMDE_FLOAT32_C(   861.60) },
      { SIMDE_FLOAT32_C(   915.88), SIMDE_FLOAT32_C(  -974.30), SIMDE_FLOAT32_C(  -265.03), SIMDE_FLOAT32_C(   377.58),
        SIMDE_FLOAT32_C(  -762.75), SIMDE_FLOAT32_C(  -195.76), SIMDE_FLOAT32_C(   134.93), SIMDE_FLOAT32_C(   490.47),
        SIMDE_FLOAT32_C(  -602.68), SIMDE_FLOAT32_C(   330.56), SIMDE_FLOAT32_C(   605.88), SIMDE_FLOAT32_C(   819.70),
        SIMDE_FLOAT32_C(   945.72), SIMDE_FLOAT32_C(  -932.70), SIMDE_FLOAT32_C(   286.24), SIMDE_FLOAT32_C(   861.60) } },
    { UINT32_C(     12188),
      { SIMDE_FLOAT32_C(   239.18), SIMDE_FLOAT32_C(   798.32), SIMDE_FLOAT32_C(   733.44), SIMDE_FLOAT32_C(  -625.88),
        SIMDE_FLOAT32_C(   288.78), SIMDE_FLOAT32_C(  -959.69), SIMDE_FLOAT32_C(   704.67), SIMDE_FLOAT32_C(  -105.34),
        SIMDE_FLOAT32_C(   860.01), SIMDE_FLOAT32_C(  -410.27), SIMDE_FLOAT32_C(   187.28), SIMDE_FLOAT32_C(   146.25),
        SIMDE_FLOAT32_C(   183.00), SIMDE_FLOAT32_C(   853.64), SIMDE_FLOAT32_C(   171.95), SIMDE_FLOAT32_C(   625.22) },
      { SIMDE_FLOAT32_C(  -265.69), SIMDE_FLOAT32_C(   409.21), SIMDE_FLOAT32_C(  -570.55), SIMDE_FLOAT32_C(   270.92),
        SIMDE_FLOAT32_C(  -405.67), SIMDE_FLOAT32_C(  -173.23), SIMDE_FLOAT32_C(   809.60), SIMDE_FLOAT32_C(   134.11),
        SIMDE_FLOAT32_C(   161.74), SIMDE_FLOAT32_C(   755.32), SIMDE_FLOAT32_C(   201.41), SIMDE_FLOAT32_C(  -510.15),
        SIMDE_FLOAT32_C(   616.93), SIMDE_FLOAT32_C(  -154.20), SIMDE_FLOAT32_C(  -447.06), SIMDE_FLOAT32_C(  -143.89) },
      { SIMDE_FLOAT32_C(  -265.69), SIMDE_FLOAT32_C(   409.21), SIMDE_FLOAT32_C(   733.44), SIMDE_FLOAT32_C(  -625.88),
        SIMDE_FLOAT32_C(   288.78), SIMDE_FLOAT32_C(  -173.23), SIMDE_FLOAT32_C(   809.60), SIMDE_FLOAT32_C(  -105.34),
        SIMDE_FLOAT32_C(   860.01), SIMDE_FLOAT32_C(  -410.27), SIMDE_FLOAT32_C(   187.28), SIMDE_FLOAT32_C(   146.25),
        SIMDE_FLOAT32_C(   616.93), SIMDE_FLOAT32_C(   853.64), SIMDE_FLOAT32_C(  -447.06), SIMDE_FLOAT32_C(  -143.89) } },
    { UINT32_C(     44913),
      { SIMDE_FLOAT32_C(   230.23), SIMDE_FLOAT32_C(   932.90), SIMDE_FLOAT32_C(  -673.31), SIMDE_FLOAT32_C(   -65.10),
        SIMDE_FLOAT32_C(  -172.44), SIMDE_FLOAT32_C(  -813.30), SIMDE_FLOAT32_C(   524.64), SIMDE_FLOAT32_C(  -985.16),
        SIMDE_FLOAT32_C(   332.95), SIMDE_FLOAT32_C(  -292.37), SIMDE_FLOAT32_C(   868.49), SIMDE_FLOAT32_C(  -495.10),
        SIMDE_FLOAT32_C(  -667.15), SIMDE_FLOAT32_C(  -397.20), SIMDE_FLOAT32_C(   914.10), SIMDE_FLOAT32_C(  -237.69) },
      { SIMDE_FLOAT32_C(   873.72), SIMDE_FLOAT32_C(  -491.57), SIMDE_FLOAT32_C(   589.08), SIMDE_FLOAT32_C(   683.32),
        SIMDE_FLOAT32_C(   642.54), SIMDE_FLOAT32_C(  -249.18), SIMDE_FLOAT32_C(   438.64), SIMDE_FLOAT32_C(  -156.05),
        SIMDE_FLOAT32_C(   240.68), SIMDE_FLOAT32_C(    55.57), SIMDE_FLOAT32_C(   689.75), SIMDE_FLOAT32_C(   793.61),
        SIMDE_FLOAT32_C(   911.68), SIMDE_FLOAT32_C(  -666.13), SIMDE_FLOAT32_C(  -920.00), SIMDE_FLOAT32_C(   141.91) },
      { SIMDE_FLOAT32_C(   230.23), SIMDE_FLOAT32_C(  -491.57), SIMDE_FLOAT32_C(   589.08), SIMDE_FLOAT32_C(   683.32),
        SIMDE_FLOAT32_C(  -172.44), SIMDE_FLOAT32_C(  -813.30), SIMDE_FLOAT32_C(   524.64), SIMDE_FLOAT32_C(  -156.05),
        SIMDE_FLOAT32_C(   332.95), SIMDE_FLOAT32_C(  -292.37), SIMDE_FLOAT32_C(   868.49), SIMDE_FLOAT32_C(  -495.10),
        SIMDE_FLOAT32_C(   911.68), SIMDE_FLOAT32_C(  -397.20), SIMDE_FLOAT32_C(  -920.00), SIMDE_FLOAT32_C(  -237.69) } },
    { UINT32_C(     59071),
      { SIMDE_FLOAT32_C(  -923.19), SIMDE_FLOAT32_C(    94.34), SIMDE_FLOAT32_C(  -406.62), SIMDE_FLOAT32_C(   601.45),
        SIMDE_FLOAT32_C(   109.18), SIMDE_FLOAT32_C(   926.33), SIMDE_FLOAT32_C(  -690.92), SIMDE_FLOAT32_C(   -22.33),
        SIMDE_FLOAT32_C(  -568.77), SIMDE_FLOAT32_C(  -358.07), SIMDE_FLOAT32_C(   580.47), SIMDE_FLOAT32_C(  -654.67),
        SIMDE_FLOAT32_C(   404.24), SIMDE_FLOAT32_C(   454.18), SIMDE_FLOAT32_C(  -146.24), SIMDE_FLOAT32_C(    -6.68) },
      { SIMDE_FLOAT32_C(   137.50), SIMDE_FLOAT32_C(  -503.70), SIMDE_FLOAT32_C(   744.14), SIMDE_FLOAT32_C(  -423.86),
        SIMDE_FLOAT32_C(   340.25), SIMDE_FLOAT32_C(   -15.18), SIMDE_FLOAT32_C(   631.70), SIMDE_FLOAT32_C(    30.00),
        SIMDE_FLOAT32_C(  -221.57), SIMDE_FLOAT32_C(   543.38), SIMDE_FLOAT32_C(   363.87), SIMDE_FLOAT32_C(  -141.57),
        SIMDE_FLOAT32_C(  -314.72), SIMDE_FLOAT32_C(   630.64), SIMDE_FLOAT32_C(   265.11), SIMDE_FLOAT32_C(  -237.91) },
      { SIMDE_FLOAT32_C(  -923.19), SIMDE_FLOAT32_C(    94.34), SIMDE_FLOAT32_C(  -406.62), SIMDE_FLOAT32_C(   601.45),
        SIMDE_FLOAT32_C(   109.18), SIMDE_FLOAT32_C(   926.33), SIMDE_FLOAT32_C(   631.70), SIMDE_FLOAT32_C(   -22.33),
        SIMDE_FLOAT32_C(  -221.57), SIMDE_FLOAT32_C(  -358.07), SIMDE_FLOAT32_C(   580.47), SIMDE_FLOAT32_C(  -141.57),
        SIMDE_FLOAT32_C(  -314.72), SIMDE_FLOAT32_C(   454.18), SIMDE_FLOAT32_C(  -146.24), SIMDE_FLOAT32_C(    -6.68) } },
    { UINT32_C(     36628),
      { SIMDE_FLOAT32_C(  -636.46), SIMDE_FLOAT32_C(   834.16), SIMDE_FLOAT32_C(   784.82), SIMDE_FLOAT32_C(  -327.38),
        SIMDE_FLOAT32_C(  -188.18), SIMDE_FLOAT32_C(  -783.95), SIMDE_FLOAT32_C(   314.55), SIMDE_FLOAT32_C(  -607.71),
        SIMDE_FLOAT32_C(  -438.62), SIMDE_FLOAT32_C(  -281.21), SIMDE_FLOAT32_C(   846.47), SIMDE_FLOAT32_C(   415.14),
        SIMDE_FLOAT32_C(   712.11), SIMDE_FLOAT32_C(   -16.03), SIMDE_FLOAT32_C(   911.44), SIMDE_FLOAT32_C(   456.25) },
      { SIMDE_FLOAT32_C(   560.11), SIMDE_FLOAT32_C(   251.69), SIMDE_FLOAT32_C(  -558.93), SIMDE_FLOAT32_C(   191.81),
        SIMDE_FLOAT32_C(  -718.30), SIMDE_FLOAT32_C(   219.51), SIMDE_FLOAT32_C(  -264.81), SIMDE_FLOAT32_C(   645.57),
        SIMDE_FLOAT32_C(  -922.06), SIMDE_FLOAT32_C(   420.47), SIMDE_FLOAT32_C(   276.21), SIMDE_FLOAT32_C(   343.05),
        SIMDE_FLOAT32_C(  -817.43), SIMDE_FLOAT32_C(  -998.81), SIMDE_FLOAT32_C(   201.55), SIMDE_FLOAT32_C(  -453.89) },
      { SIMDE_FLOAT32_C(   560.11), SIMDE_FLOAT32_C(   251.69), SIMDE_FLOAT32_C(   784.82), SIMDE_FLOAT32_C(   191.81),
        SIMDE_FLOAT32_C(  -188.18), SIMDE_FLOAT32_C(   219.51), SIMDE_FLOAT32_C(  -264.81), SIMDE_FLOAT32_C(   645.57),
        SIMDE_FLOAT32_C(  -438.62), SIMDE_FLOAT32_C(  -281.21), SIMDE_FLOAT32_C(   846.47), SIMDE_FLOAT32_C(   415.14),
        SIMDE_FLOAT32_C(  -817.43), SIMDE_FLOAT32_C(  -998.81), SIMDE_FLOAT32_C(   201.55), SIMDE_FLOAT32_C(   456.25) } },
    { UINT32_C(     51689),
      { SIMDE_FLOAT32_C(   218.73), SIMDE_FLOAT32_C(  -352.83), SIMDE_FLOAT32_C(   202.42), SIMDE_FLOAT32_C(  -466.72),
        SIMDE_FLOAT32_C(    39.46), SIMDE_FLOAT32_C(   763.80), SIMDE_FLOAT32_C(   252.07), SIMDE_FLOAT32_C(  -114.07),
        SIMDE_FLOAT32_C(   178.94), SIMDE_FLOAT32_C(   -35.82), SIMDE_FLOAT32_C(   869.90), SIMDE_FLOAT32_C(    90.39),
        SIMDE_FLOAT32_C(  -579.56), SIMDE_FLOAT32_C(   430.01), SIMDE_FLOAT32_C(  -657.92), SIMDE_FLOAT32_C(  -138.49) },
      { SIMDE_FLOAT32_C(  -378.18), SIMDE_FLOAT32_C(  -376.22), SIMDE_FLOAT32_C(  -918.98), SIMDE_FLOAT32_C(   357.01),
        SIMDE_FLOAT32_C(  -730.66), SIMDE_FLOAT32_C(  -841.04), SIMDE_FLOAT32_C(  -222.52), SIMDE_FLOAT32_C(   545.55),
        SIMDE_FLOAT32_C(   502.01), SIMDE_FLOAT32_C(   -39.95), SIMDE_FLOAT32_C(   546.75), SIMDE_FLOAT32_C(  -296.45),
        SIMDE_FLOAT32_C(   506.16), SIMDE_FLOAT32_C(   382.09), SIMDE_FLOAT32_C(   689.92), SIMDE_FLOAT32_C(  -275.12) },
      { SIMDE_FLOAT32_C(   218.73), SIMDE_FLOAT32_C(  -376.22), SIMDE_FLOAT32_C(  -918.98), SIMDE_FLOAT32_C(  -466.72),
        SIMDE_FLOAT32_C(  -730.66), SIMDE_FLOAT32_C(   763.80), SIMDE_FLOAT32_C(   252.07), SIMDE_FLOAT32_C(  -114.07),
        SIMDE_FLOAT32_C(   178.94), SIMDE_FLOAT32_C(   -39.95), SIMDE_FLOAT32_C(   546.75), SIMDE_FLOAT32_C(    90.39),
        SIMDE_FLOAT32_C(   506.16), SIMDE_FLOAT32_C(   382.09), SIMDE_FLOAT32_C(  -657.92), SIMDE_FLOAT32_C(  -138.49) } },
    { UINT32_C(     65417),
      { SIMDE_FLOAT32_C(   258.16), SIMDE_FLOAT32_C(    68.72), SIMDE_FLOAT32_C(  -343.85), SIMDE_FLOAT32_C(  -489.76),
        SIMDE_FLOAT32_C(   954.65), SIMDE_FLOAT32_C(   835.09), SIMDE_FLOAT32_C(   474.42), SIMDE_FLOAT32_C(   824.55),
        SIMDE_FLOAT32_C(   -74.52), SIMDE_FLOAT32_C(   894.86), SIMDE_FLOAT32_C(   254.55), SIMDE_FLOAT32_C(   267.56),
        SIMDE_FLOAT32_C(  -243.63), SIMDE_FLOAT32_C(   876.37), SIMDE_FLOAT32_C(   891.34), SIMDE_FLOAT32_C(  -162.61) },
      { SIMDE_FLOAT32_C(   233.37), SIMDE_FLOAT32_C(  -839.32), SIMDE_FLOAT32_C(    -3.65), SIMDE_FLOAT32_C(  -989.14),
        SIMDE_FLOAT32_C(   706.23), SIMDE_FLOAT32_C(  -501.65), SIMDE_FLOAT32_C(   -29.10), SIMDE_FLOAT32_C(   252.98),
        SIMDE_FLOAT32_C(   201.91), SIMDE_FLOAT32_C(  -522.94), SIMDE_FLOAT32_C(  -364.93), SIMDE_FLOAT32_C(  -108.17),
        SIMDE_FLOAT32_C(   201.94), SIMDE_FLOAT32_C(  -335.67), SIMDE_FLOAT32_C(   784.17), SIMDE_FLOAT32_C(  -539.89) },
      { SIMDE_FLOAT32_C(   258.16), SIMDE_FLOAT32_C(  -839.32), SIMDE_FLOAT32_C(    -3.65), SIMDE_FLOAT32_C(  -489.76),
        SIMDE_FLOAT32_C(   706.23), SIMDE_FLOAT32_C(  -501.65), SIMDE_FLOAT32_C(   -29.10), SIMDE_FLOAT32_C(   824.55),
        SIMDE_FLOAT32_C(   -74.52), SIMDE_FLOAT32_C(   894.86), SIMDE_FLOAT32_C(   254.55), SIMDE_FLOAT32_C(   267.56),
        SIMDE_FLOAT32_C(  -243.63), SIMDE_FLOAT32_C(   876.37), SIMDE_FLOAT32_C(   891.34), SIMDE_FLOAT32_C(  -162.61) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512 a = simde_mm512_loadu_ps(test_vec[i].a);
    simde__m512 r0 = simde_mm512_loadu_ps(test_vec[i].r0);
    simde_float32 r1[sizeof(simde__m512) / sizeof(simde_float32)];
    simde_mm512_storeu_ps(r1, r0);
    simde_mm512_mask_storeu_ps(r1, test_vec[i].k, a);
    simde_assert_equal_vf32(sizeof(r1) / sizeof(r1[0]), r1, test_vec[i].r1, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    simde__mmask16 k = simde_test_x86_random_mmask16();
    simde__m512 a = simde_test_x86_random_f32x16(SIMDE_FLOAT32_C(-1000.0), SIMDE_FLOAT32_C(1000.0));
    simde__m512 r0 = simde_test_x86_random_f32x16(SIMDE_FLOAT32_C(-1000.0), SIMDE_FLOAT32_C(1000.0));
    simde_float32 r1[sizeof(simde__m512) / sizeof(simde_float32)];
    simde_mm512_storeu_ps(r1, r0);
    simde_mm512_mask_storeu_ps(r1, k, a);

    simde_test_x86_write_mmask32(2, k, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_f32x16(2, a, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_f32x16(2, r0, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_codegen_write_vf32(2, sizeof(r1) / sizeof(r1[0]), r1, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm512_storeu_pd (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const simde_float64 a[8];
    const simde_float64 r[8];
  } test_vec[] = {
    { { SIMDE_FLOAT64_C(  -762.61), SIMDE_FLOAT64_C(  -133.81), SIMDE_FLOAT64_C(   837.91), SIMDE_FLOAT64_C(   816.67),
        SIMDE_FLOAT64_C(  -129.22), SIMDE_FLOAT64_C(   495.46), SIMDE_FLOAT64_C(   682.26), SIMDE_FLOAT64_C(   786.79) },
      { SIMDE_FLOAT64_C(  -762.61), SIMDE_FLOAT64_C(  -133.81), SIMDE_FLOAT64_C(   837.91), SIMDE_FLOAT64_C(   816.67),
        SIMDE_FLOAT64_C(  -129.22), SIMDE_FLOAT64_C(   495.46), SIMDE_FLOAT64_C(   682.26), SIMDE_FLOAT64_C(   786.79) } },
    { { SIMDE_FLOAT64_C(   715.88), SIMDE_FLOAT64_C(   951.12), SIMDE_FLOAT64_C(  -156.68), SIMDE_FLOAT64_C(   371.76),
        SIMDE_FLOAT64_C(   -20.75), SIMDE_FLOAT64_C(  -315.62), SIMDE_FLOAT64_C(   321.24), SIMDE_FLOAT64_C(  -279.66) },
      { SIMDE_FLOAT64_C(   715.88), SIMDE_FLOAT64_C(   951.12), SIMDE_FLOAT64_C(  -156.68), SIMDE_FLOAT64_C(   371.76),
        SIMDE_FLOAT64_C(   -20.75), SIMDE_FLOAT64_C(  -315.62), SIMDE_FLOAT64_C(   321.24), SIMDE_FLOAT64_C(  -279.66) } },
    { { SIMDE_FLOAT64_C(   314.91), SIMDE_FLOAT64_C(   523.90), SIMDE_FLOAT64_C(  -427.37), SIMDE_FLOAT64_C(   378.61),
        SIMDE_FLOAT64_C(   929.35), SIMDE_FLOAT64_C(  -291.71), SIMDE_FLOAT64_C(   -31.14), SIMDE_FLOAT64_C(  -905.89) },
      { SIMDE_FLOAT64_C(   314.91), SIMDE_FLOAT64_C(   523.90), SIMDE_FLOAT64_C(  -427.37), SIMDE_FLOAT64_C(   378.61),
        SIMDE_FLOAT64_C(   929.35), SIMDE_FLOAT64_C(  -291.71), SIMDE_FLOAT64_C(   -31.14), SIMDE_FLOAT64_C(  -905.89) } },
    { { SIMDE_FLOAT64_C(   607.66), SIMDE_FLOAT64_C(   -95.89), SIMDE_FLOAT64_C(  -309.91), SIMDE_FLOAT64_C(  -953.40),
        SIMDE_FLOAT64_C(   228.74), SIMDE_FLOAT64_C(   282.70), SIMDE_FLOAT64_C(  -516.01), SIMDE_FLOAT64_C(   466.13) },
      { SIMDE_FLOAT64_C(   607.66), SIMDE_FLOAT64_C(   -95.89), SIMDE_FLOAT64_C(  -309.91), SIMDE_FLOAT64_C(  -953.40),
        SIMDE_FLOAT64_C(   228.74), SIMDE_FLOAT64_C(   282.70), SIMDE_FLOAT64_C(  -516.01), SIMDE_FLOAT64_C(   466.13) } },
    { { SIMDE_FLOAT64_C(  -851.11), SIMDE_FLOAT64_C(  -678.10), SIMDE_FLOAT64_C(   282.80), SIMDE_FLOAT64_C(    19.67),
        SIMDE_FLOAT64_C(   817.36), SIMDE_FLOAT64_C(   -34.94), SIMDE_FLOAT64_C(  -193.55), SIMDE_FLOAT64_C(   533.24) },
      { SIMDE_FLOAT64_C(  -851.11), SIMDE_FLOAT64_C(  -678.10), SIMDE_FLOAT64_C(   282.80), SIMDE_FLOAT64_C(    19.67),
        SIMDE_FLOAT64_C(   817.36), SIMDE_FLOAT64_C(   -34.94), SIMDE_FLOAT64_C(  -193.55), SIMDE_FLOAT64_C(   533.24) } },
    { { SIMDE_FLOAT64_C(   -83.82), SIMDE_FLOAT64_C(   649.77), SIMDE_FLOAT64_C(   -95.00), SIMDE_FLOAT64_C(   895.42),
        SIMDE_FLOAT64_C(  -665.85), SIMDE_FLOAT64_C(  -773.76), SIMDE_FLOAT64_C(  -384.24), SIMDE_FLOAT64_C(   649.05) },
      { SIMDE_FLOAT64_C(   -83.82), SIMDE_FLOAT64_C(   649.77), SIMDE_FLOAT64_C(   -95.00), SIMDE_FLOAT64_C(   895.42),
        SIMDE_FLOAT64_C(  -665.85), SIMDE_FLOAT64_C(  -773.76), SIMDE_FLOAT64_C(  -384.24), SIMDE_FLOAT64_C(   649.05) } },
    { { SIMDE_FLOAT64_C(   750.15), SIMDE_FLOAT64_C(   188.39), SIMDE_FLOAT64_C(    27.67), SIMDE_FLOAT64_C(   679.49),
        SIMDE_FLOAT64_C(   896.68), SIMDE_FLOAT64_C(   996.53), SIMDE_FLOAT64_C(   773.60), SIMDE_FLOAT64_C(   504.34) },
      { SIMDE_FLOAT64_C(   750.15), SIMDE_FLOAT64_C(   188.39), SIMDE_FLOAT64_C(    27.67), SIMDE_FLOAT64_C(   679.49),
        SIMDE_FLOAT64_C(   896.68), SIMDE_FLOAT64_C(   996.53), SIMDE_FLOAT64_C(   773.60), SIMDE_FLOAT64_C(   504.34) } },
    { { SIMDE_FLOAT64_C(   -99.37), SIMDE_FLOAT64_C(  -536.31), SIMDE_FLOAT64_C(   550.94), SIMDE_FLOAT64_C(  -870.63),
        SIMDE_FLOAT64_C(   746.39), SIMDE_FLOAT64_C(  -965.07), SIMDE_FLOAT64_C(   595.50), SIMDE_FLOAT64_C(   895.28) },
      { SIMDE_FLOAT64_C(   -99.37), SIMDE_FLOAT64_C(  -536.31), SIMDE_FLOAT64_C(   550.94), SIMDE_FLOAT64_C(  -870.63),
        SIMDE_FLOAT64_C(   746.39), SIMDE_FLOAT64_C(  -965.07), SIMDE_FLOAT64_C(   595.50), SIMDE_FLOAT64_C(   895.28) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512d a = simde_mm512_loadu_pd(test_vec[i].a);
    simde_float64 r[sizeof(simde__m512d) / sizeof(simde_float64)];
    simde_mm512_storeu_pd(r, a);
    simde_assert_equal_vf64(sizeof(r) / sizeof(r[0]), r, test_vec[i].r, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    simde__m512d a = simde_test_x86_random_f64x8(SIMDE_FLOAT64_C(-1000.0), SIMDE_FLOAT64_C(1000.0));
    SIMDE_ALIGN_LIKE_64(simde__m512d) simde_float64 r[sizeof(simde__m512d) / sizeof(simde_float64)];
    simde_mm512_storeu_pd(r, a);

    simde_test_x86_write_f64x8(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_codegen_write_vf64(2, sizeof(r) / sizeof(r[0]), r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm512_mask_storeu_pd (SIMDE_MUNIT_TEST_ARGS) {
#if 2
  static const struct {
    const simde_float64 a[8];
    const simde__mmask8 k;
    const simde_float64 r0[8];
    const simde_float64 r1[8];
  } test_vec[] = {
    { { SIMDE_FLOAT64_C(   663.49), SIMDE_FLOAT64_C(  -509.49), SIMDE_FLOAT64_C(  -315.08), SIMDE_FLOAT64_C(  -203.71),
        SIMDE_FLOAT64_C(   862.19), SIMDE_FLOAT64_C(   236.89), SIMDE_FLOAT64_C(   827.24), SIMDE_FLOAT64_C(  -159.93) },
      UINT8_C(110),
      { SIMDE_FLOAT64_C(  -304.62), SIMDE_FLOAT64_C(   961.42), SIMDE_FLOAT64_C(  -231.64), SIMDE_FLOAT64_C(  -560.84),
        SIMDE_FLOAT64_C(  -811.80), SIMDE_FLOAT64_C(   -65.76), SIMDE_FLOAT64_C(   609.84), SIMDE_FLOAT64_C(   425.76) },
      { SIMDE_FLOAT64_C(  -304.62), SIMDE_FLOAT64_C(  -509.49), SIMDE_FLOAT64_C(  -315.08), SIMDE_FLOAT64_C(  -203.71),
        SIMDE_FLOAT64_C(  -811.80), SIMDE_FLOAT64_C(   236.89), SIMDE_FLOAT64_C(   827.24), SIMDE_FLOAT64_C(   425.76) } },
    { { SIMDE_FLOAT64_C(    33.90), SIMDE_FLOAT64_C(   307.41), SIMDE_FLOAT64_C(    41.68), SIMDE_FLOAT64_C(  -686.43),
        SIMDE_FLOAT64_C(  -175.80), SIMDE_FLOAT64_C(   -59.45), SIMDE_FLOAT64_C(   175.99), SIMDE_FLOAT64_C(   688.67) },
      UINT8_C(180),
      { SIMDE_FLOAT64_C(    -6.81), SIMDE_FLOAT64_C(   714.62), SIMDE_FLOAT64_C(  -131.60), SIMDE_FLOAT64_C(  -343.32),
        SIMDE_FLOAT64_C(  -794.86), SIMDE_FLOAT64_C(   553.32), SIMDE_FLOAT64_C(   452.97), SIMDE_FLOAT64_C(  -932.68) },
      { SIMDE_FLOAT64_C(    -6.81), SIMDE_FLOAT64_C(   714.62), SIMDE_FLOAT64_C(    41.68), SIMDE_FLOAT64_C(  -343.32),
        SIMDE_FLOAT64_C(  -175.80), SIMDE_FLOAT64_C(   -59.45), SIMDE_FLOAT64_C(   452.97), SIMDE_FLOAT64_C(   688.67) } },
    { { SIMDE_FLOAT64_C(  -209.79), SIMDE_FLOAT64_C(   280.21), SIMDE_FLOAT64_C(   -92.61), SIMDE_FLOAT64_C(  -278.83),
        SIMDE_FLOAT64_C(   825.09), SIMDE_FLOAT64_C(   602.77), SIMDE_FLOAT64_C(  -317.41), SIMDE_FLOAT64_C(  -406.55) },
      UINT8_C(211),
      { SIMDE_FLOAT64_C(   527.69), SIMDE_FLOAT64_C(   651.78), SIMDE_FLOAT64_C(  -703.44), SIMDE_FLOAT64_C(  -438.41),
        SIMDE_FLOAT64_C(   -40.82), SIMDE_FLOAT64_C(   338.24), SIMDE_FLOAT64_C(  -124.84), SIMDE_FLOAT64_C(   783.39) },
      { SIMDE_FLOAT64_C(  -209.79), SIMDE_FLOAT64_C(   280.21), SIMDE_FLOAT64_C(  -703.44), SIMDE_FLOAT64_C(  -438.41),
        SIMDE_FLOAT64_C(   825.09), SIMDE_FLOAT64_C(   338.24), SIMDE_FLOAT64_C(  -317.41), SIMDE_FLOAT64_C(  -406.55) } },
    { { SIMDE_FLOAT64_C(  -721.21), SIMDE_FLOAT64_C(  -948.85), SIMDE_FLOAT64_C(   472.06), SIMDE_FLOAT64_C(   758.45),
        SIMDE_FLOAT64_C(   776.23), SIMDE_FLOAT64_C(  -534.75), SIMDE_FLOAT64_C(   473.08), SIMDE_FLOAT64_C(  -355.37) },
      UINT8_C(194),
      { SIMDE_FLOAT64_C(  -802.05), SIMDE_FLOAT64_C(  -425.11), SIMDE_FLOAT64_C(   745.54), SIMDE_FLOAT64_C(   -11.84),
        SIMDE_FLOAT64_C(   855.09), SIMDE_FLOAT64_C(  -347.06), SIMDE_FLOAT64_C(   709.33), SIMDE_FLOAT64_C(   680.18) },
      { SIMDE_FLOAT64_C(  -802.05), SIMDE_FLOAT64_C(  -948.85), SIMDE_FLOAT64_C(   745.54), SIMDE_FLOAT64_C(   -11.84),
        SIMDE_FLOAT64_C(   855.09), SIMDE_FLOAT64_C(  -347.06), SIMDE_FLOAT64_C(   473.08), SIMDE_FLOAT64_C(  -355.37) } },
    { { SIMDE_FLOAT64_C(  -744.29), SIMDE_FLOAT64_C(  -608.08), SIMDE_FLOAT64_C(  -726.37), SIMDE_FLOAT64_C(  -702.35),
        SIMDE_FLOAT64_C(   262.72), SIMDE_FLOAT64_C(   801.32), SIMDE_FLOAT64_C(   949.42), SIMDE_FLOAT64_C(   559.27) },
      UINT8_C(226),
      { SIMDE_FLOAT64_C(  -102.49), SIMDE_FLOAT64_C(   238.06), SIMDE_FLOAT64_C(  -308.00), SIMDE_FLOAT64_C(   176.29),
        SIMDE_FLOAT64_C(   289.21), SIMDE_FLOAT64_C(  -835.95), SIMDE_FLOAT64_C(   -65.25), SIMDE_FLOAT64_C(    65.44) },
      { SIMDE_FLOAT64_C(  -102.49), SIMDE_FLOAT64_C(  -608.08), SIMDE_FLOAT64_C(  -308.00), SIMDE_FLOAT64_C(   176.29),
        SIMDE_FLOAT64_C(   289.21), SIMDE_FLOAT64_C(   801.32), SIMDE_FLOAT64_C(   949.42), SIMDE_FLOAT64_C(   559.27) } },
    { { SIMDE_FLOAT64_C(  -370.70), SIMDE_FLOAT64_C(  -592.17), SIMDE_FLOAT64_C(   710.08), SIMDE_FLOAT64_C(   751.22),
        SIMDE_FLOAT64_C(  -913.95), SIMDE_FLOAT64_C(   908.03), SIMDE_FLOAT64_C(  -673.90), SIMDE_FLOAT64_C(   831.59) },
      UINT8_C(178),
      { SIMDE_FLOAT64_C(  -515.48), SIMDE_FLOAT64_C(  -394.48), SIMDE_FLOAT64_C(   861.38), SIMDE_FLOAT64_C(  -259.76),
        SIMDE_FLOAT64_C(    -2.55), SIMDE_FLOAT64_C(  -864.99), SIMDE_FLOAT64_C(    37.88), SIMDE_FLOAT64_C(  -739.84) },
      { SIMDE_FLOAT64_C(  -515.48), SIMDE_FLOAT64_C(  -592.17), SIMDE_FLOAT64_C(   861.38), SIMDE_FLOAT64_C(  -259.76),
        SIMDE_FLOAT64_C(  -913.95), SIMDE_FLOAT64_C(   908.03), SIMDE_FLOAT64_C(    37.88), SIMDE_FLOAT64_C(   831.59) } },
    { { SIMDE_FLOAT64_C(   936.32), SIMDE_FLOAT64_C(   -12.69), SIMDE_FLOAT64_C(   819.43), SIMDE_FLOAT64_C(  -700.78),
        SIMDE_FLOAT64_C(   895.92), SIMDE_FLOAT64_C(  -283.06), SIMDE_FLOAT64_C(   537.28), SIMDE_FLOAT64_C(  -412.09) },
      UINT8_C( 47),
      { SIMDE_FLOAT64_C(  -248.04), SIMDE_FLOAT64_C(  -172.01), SIMDE_FLOAT64_C(   891.93), SIMDE_FLOAT64_C(   381.26),
        SIMDE_FLOAT64_C(   235.81), SIMDE_FLOAT64_C(   602.01), SIMDE_FLOAT64_C(   132.48), SIMDE_FLOAT64_C(   321.86) },
      { SIMDE_FLOAT64_C(   936.32), SIMDE_FLOAT64_C(   -12.69), SIMDE_FLOAT64_C(   819.43), SIMDE_FLOAT64_C(  -700.78),
        SIMDE_FLOAT64_C(   235.81), SIMDE_FLOAT64_C(  -283.06), SIMDE_FLOAT64_C(   132.48), SIMDE_FLOAT64_C(   321.86) } },
    { { SIMDE_FLOAT64_C(   510.03), SIMDE_FLOAT64_C(   458.58), SIMDE_FLOAT64_C(   153.45), SIMDE_FLOAT64_C(  -593.78),
        SIMDE_FLOAT64_C(   639.78), SIMDE_FLOAT64_C(   637.97), SIMDE_FLOAT64_C(    11.75), SIMDE_FLOAT64_C(   501.16) },
      UINT8_C(237),
      { SIMDE_FLOAT64_C(   636.16), SIMDE_FLOAT64_C(   416.09), SIMDE_FLOAT64_C(  -730.64), SIMDE_FLOAT64_C(   572.48),
        SIMDE_FLOAT64_C(  -596.60), SIMDE_FLOAT64_C(  -911.21), SIMDE_FLOAT64_C(   871.70), SIMDE_FLOAT64_C(  -700.69) },
      { SIMDE_FLOAT64_C(   510.03), SIMDE_FLOAT64_C(   416.09), SIMDE_FLOAT64_C(   153.45), SIMDE_FLOAT64_C(  -593.78),
        SIMDE_FLOAT64_C(  -596.60), SIMDE_FLOAT64_C(   637.97), SIMDE_FLOAT64_C(    11.75), SIMDE_FLOAT64_C(   501.16) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512d a = simde_mm512_loadu_pd(test_vec[i].a);
    simde__m512d r0 = simde_mm512_loadu_pd(test_vec[i].r0);
    simde_float64 r1[sizeof(simde__m512d) / sizeof(simde_float64)];
    simde_mm512_storeu_pd(r1, r0);
    simde_mm512_mask_storeu_pd(r1, test_vec[i].k, a);
    simde_assert_equal_vf64(sizeof(r1) / sizeof(r1[0]), r1, test_vec[i].r1, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    simde__m512d a = simde_test_x86_random_f64x8(SIMDE_FLOAT64_C(-1000.0), SIMDE_FLOAT64_C(1000.0));
    simde__mmask8 k = simde_test_x86_random_mmask16();
    simde__m512d r0 = simde_test_x86_random_f64x8(SIMDE_FLOAT64_C(-1000.0), SIMDE_FLOAT64_C(1000.0));
    simde_float64 r1[sizeof(simde__m512d) / sizeof(simde_float64)];
    simde_mm512_storeu_pd(r1, r0);
    simde_mm512_mask_storeu_pd(r1, k, a);

    simde_test_x86_write_f64x8(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_mmask8(2, k, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_f64x8(2, r0, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_codegen_write_vf64(2, sizeof(r1) / sizeof(r1[0]), r1, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

SIMDE_TEST_FUNC_LIST_BEGIN
  SIMDE_TEST_FUNC_LIST_ENTRY(mm256_storeu_epi8)
  SIMDE_TEST_FUNC_LIST_ENTRY(mm256_storeu_epi16)
  SIMDE_TEST_FUNC_LIST_ENTRY(mm256_mask_storeu_epi16)
  SIMDE_TEST_FUNC_LIST_ENTRY(mm256_storeu_epi32)
  SIMDE_TEST_FUNC_LIST_ENTRY(mm256_storeu_epi64)
  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_storeu_si512) // alias mm512_storeu_epi8 mm512_storeu_epi16 mm512_storeu_epi32 mm512_storeu_epi64
  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_mask_storeu_epi8)
  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_mask_storeu_epi16)
  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_mask_storeu_epi32)
  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_mask_storeu_epi64)
  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_storeu_ps)
  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_mask_storeu_ps)
  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_storeu_pd)
  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_mask_storeu_pd)
  #if defined(SIMDE_FLOAT16_IS_SCALAR)
    SIMDE_TEST_FUNC_LIST_ENTRY(mm512_storeu_ph)
  #endif
SIMDE_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
