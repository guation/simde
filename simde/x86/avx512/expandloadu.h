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

#if !defined(SIMDE_X86_AVX512_EXPANDLOADU_H)
#define SIMDE_X86_AVX512_EXPANDLOADU_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
SIMDE_DISABLE_UNWANTED_DIAGNOSTICS
SIMDE_BEGIN_DECLS_

SIMDE_FUNCTION_ATTRIBUTES
simde__m512i
simde_mm512_mask_expandloadu_epi32 (simde__m512i src, simde__mmask16 k, void const* mem_addr) {
  #if defined(SIMDE_X86_AVX512F_NATIVE)
    return _mm512_mask_expandloadu_epi32(src, k, mem_addr);
  #else
    simde__m512i_private src_ = simde__m512i_to_private(src);
    const int32_t* mem_addr_ = HEDLEY_STATIC_CAST(const int32_t*, mem_addr);
    simde__m512i_private r_;
    SIMDE_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      if (k & (UINT16_C(1) << i)) {
        r_.i32[i] = *mem_addr_;
        mem_addr_++;
      } else {
        r_.i32[i] = src_.i32[i];
      }
    }
    return simde__m512i_from_private(r_);
  #endif
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_expandloadu_epi32
  #define _mm512_mask_expandloadu_epi32(src, k, mem_addr) simde_mm512_mask_expandloadu_epi32((src), (k), (mem_addr))
#endif

SIMDE_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(SIMDE_X86_AVX512_EXPANDLOADU_H) */
