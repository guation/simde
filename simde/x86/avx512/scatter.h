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

#if !defined(SIMDE_X86_AVX512_SCATTER_H)
#define SIMDE_X86_AVX512_SCATTER_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
SIMDE_DISABLE_UNWANTED_DIAGNOSTICS
SIMDE_BEGIN_DECLS_

SIMDE_FUNCTION_ATTRIBUTES
void
simde_mm512_i32scatter_epi32 (void* base_addr, simde__m512i vindex, simde__m512i a, int scale)
    SIMDE_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  simde__m512i_private vindex_ = simde__m512i_to_private(vindex);
  simde__m512i_private a_ = simde__m512i_to_private(a);
  uint8_t* base_addr_ = HEDLEY_REINTERPRET_CAST(uint8_t*, base_addr);
  SIMDE_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
    uint8_t* dst = base_addr_ + (HEDLEY_STATIC_CAST(size_t, vindex_.i32[i]) * HEDLEY_STATIC_CAST(size_t, scale));
    int32_t src = a_.i32[i];
    simde_memcpy(dst, &src, sizeof(src));
  }
}
#if defined(SIMDE_X86_AVX512F_NATIVE)
  #define simde_mm512_i32scatter_epi32(base_addr, vindex, a, scale) _mm512_i32scatter_epi32((base_addr), (vindex), (a), (scale))
#endif
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_i32scatter_epi32
  #define _mm512_i32scatter_epi32(base_addr, vindex, a, scale) simde_mm512_i32scatter_epi32((base_addr), (vindex), (a), (scale))
#endif

SIMDE_FUNCTION_ATTRIBUTES
void
simde_mm512_mask_i32scatter_epi32 (void* base_addr, simde__mmask16 k, simde__m512i vindex, simde__m512i a, int scale)
    SIMDE_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  simde__m512i_private vindex_ = simde__m512i_to_private(vindex);
  simde__m512i_private a_ = simde__m512i_to_private(a);
  uint8_t* base_addr_ = HEDLEY_REINTERPRET_CAST(uint8_t*, base_addr);
  SIMDE_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
    if (k & (UINT16_C(1) << i)) {
      uint8_t* dst = base_addr_ + (HEDLEY_STATIC_CAST(size_t, vindex_.i32[i]) * HEDLEY_STATIC_CAST(size_t, scale));
      int32_t src = a_.i32[i];
      simde_memcpy(dst, &src, sizeof(src));
    }
  }
}
#if defined(SIMDE_X86_AVX512F_NATIVE)
  #define simde_mm512_mask_i32scatter_epi32(base_addr, k, vindex, a, scale) _mm512_mask_i32scatter_epi32((base_addr), (k), (vindex), (a), (scale))
#endif
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_i32scatter_epi32
  #define _mm512_mask_i32scatter_epi32(base_addr, k, vindex, a, scale) simde_mm512_mask_i32scatter_epi32((base_addr), (k), (vindex), (a), (scale))
#endif

SIMDE_FUNCTION_ATTRIBUTES
void
simde_mm512_i32scatter_ps (void* base_addr, simde__m512i vindex, simde__m512 a, int scale)
    SIMDE_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  simde__m512i_private vindex_ = simde__m512i_to_private(vindex);
  simde__m512_private a_ = simde__m512_to_private(a);
  uint8_t* base_addr_ = HEDLEY_REINTERPRET_CAST(uint8_t*, base_addr);
  SIMDE_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
    uint8_t* dst = base_addr_ + (HEDLEY_STATIC_CAST(size_t, vindex_.i32[i]) * HEDLEY_STATIC_CAST(size_t, scale));
    simde_float32 src = a_.f32[i];
    simde_memcpy(dst, &src, sizeof(src));
  }
}
#if defined(SIMDE_X86_AVX512F_NATIVE)
  #define simde_mm512_i32scatter_ps(base_addr, vindex, a, scale) _mm512_i32scatter_ps((base_addr), (vindex), (a), (scale))
#endif
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_i32scatter_ps
  #define _mm512_i32scatter_ps(base_addr, vindex, a, scale) simde_mm512_i32scatter_ps((base_addr), (vindex), (a), (scale))
#endif

SIMDE_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(SIMDE_X86_AVX512_SCATTER_H) */
