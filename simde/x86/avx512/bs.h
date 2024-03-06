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
 *   2024      Guation <guation@guation.cn>
 */

#if !defined(SIMDE_X86_AVX512_BS_H)
#define SIMDE_X86_AVX512_BS_H

#include "types.h"
#include "../avx2.h"

HEDLEY_DIAGNOSTIC_PUSH
SIMDE_DISABLE_UNWANTED_DIAGNOSTICS
SIMDE_BEGIN_DECLS_

SIMDE_FUNCTION_ATTRIBUTES
simde__m512i
simde_mm512_bslli_epi128 (simde__m512i a, int imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  simde__m512i_private a_ = simde__m512i_to_private(a);
  simde__m512i_private r_;

  int imm8_ = (imm8 > 15 ? 16 : imm8) * 8;
  SIMDE_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.u128) / sizeof(r_.u128[0])) ; i++) {
    r_.u128[i] = a_.u128[i] << imm8_;
  }

  return simde__m512i_from_private(r_);
}
#if defined(SIMDE_X86_AVX512BW_NATIVE)
  #define simde_mm512_bslli_epi128(a, imm8) _mm512_bslli_epi128(a, imm8)
#elif defined(SIMDE_X86_AVX2_NATIVE)
  #define simde_mm512_bslli_epi128(a, imm8) ({simde__m512i_private a_ = simde__m512i_to_private(a); simde_x_mm512_set_m256i(simde_mm256_bslli_epi128(a_.m256i[1], (imm8)), simde_mm256_bslli_epi128(a_.m256i[0], (imm8)));})
#elif defined(SIMDE_X86_SSE2_NATIVE)
  #define simde_mm512_bslli_epi128(a, imm8) ({simde__m512i_private a_ = simde__m512i_to_private(a); simde_x_mm512_set_m128i(simde_mm_bslli_si128(a_.m128i[3], (imm8)), simde_mm_bslli_si128(a_.m128i[2], (imm8)), simde_mm_bslli_si128(a_.m128i[1], (imm8)), simde_mm_bslli_si128(a_.m128i[0], (imm8)));})
#endif
#if defined(SIMDE_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_bslli_epi128
  #define _mm512_bslli_epi128(a, imm8) simde_mm512_bslli_epi128((a), (imm8))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde__m512i
simde_mm512_bsrli_epi128 (simde__m512i a, int imm8)
    SIMDE_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  simde__m512i_private a_ = simde__m512i_to_private(a);
  simde__m512i_private r_;

  int imm8_ = (imm8 > 15 ? 16 : imm8) * 8;
  SIMDE_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.u128) / sizeof(r_.u128[0])) ; i++) {
    r_.u128[i] = a_.u128[i] >> imm8_;
  }

  return simde__m512i_from_private(r_);
}
#if defined(SIMDE_X86_AVX512BW_NATIVE)
  #define simde_mm512_bsrli_epi128(a, imm8) _mm512_bsrli_epi128(a, imm8)
#elif defined(SIMDE_X86_AVX2_NATIVE)
  #define simde_mm512_bsrli_epi128(a, imm8) ({simde__m512i_private a_ = simde__m512i_to_private(a); simde_x_mm512_set_m256i(simde_mm256_bsrli_epi128(a_.m256i[1], (imm8)), simde_mm256_bsrli_epi128(a_.m256i[0], (imm8)));})
#elif defined(SIMDE_X86_SSE2_NATIVE)
  #define simde_mm512_bsrli_epi128(a, imm8) ({simde__m512i_private a_ = simde__m512i_to_private(a); simde_x_mm512_set_m128i(simde_mm_bsrli_si128(a_.m128i[3], (imm8)), simde_mm_bsrli_si128(a_.m128i[2], (imm8)), simde_mm_bsrli_si128(a_.m128i[1], (imm8)), simde_mm_bsrli_si128(a_.m128i[0], (imm8)));})
#endif
#if defined(SIMDE_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_bsrli_epi128
  #define _mm512_bsrli_epi128(a, imm8) simde_mm512_bsrli_epi128((a), (imm8))
#endif

SIMDE_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(SIMDE_X86_AVX512_BS_H) */
