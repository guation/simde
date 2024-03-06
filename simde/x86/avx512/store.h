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

#if !defined(SIMDE_X86_AVX512_STORE_H)
#define SIMDE_X86_AVX512_STORE_H

#include "types.h"
#include "../avx2.h"
#include "movm.h"

HEDLEY_DIAGNOSTIC_PUSH
SIMDE_DISABLE_UNWANTED_DIAGNOSTICS
SIMDE_BEGIN_DECLS_

SIMDE_FUNCTION_ATTRIBUTES
void
simde_mm512_store_ps (void * mem_addr, simde__m512 a) {
  #if defined(SIMDE_X86_AVX512F_NATIVE)
    _mm512_store_ps(mem_addr, a);
  #else
    simde_memcpy(SIMDE_ALIGN_ASSUME_LIKE(mem_addr, simde__m512), &a, sizeof(a));
  #endif
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_store_ps
  #define _mm512_store_ps(mem_addr, a) simde_mm512_store_ps(mem_addr, a)
#endif

SIMDE_FUNCTION_ATTRIBUTES
void
simde_mm512_mask_store_ps (void * mem_addr, simde__mmask16 k, simde__m512 a) {
  #if defined(SIMDE_X86_AVX512F_NATIVE)
    _mm512_mask_store_ps(mem_addr, k, a);
  #else
    simde__m512_private a_ = simde__m512_to_private(a);
    simde_float32* mem_addr_ = HEDLEY_STATIC_CAST(simde_float32*, mem_addr);
    #if defined(SIMDE_X86_AVX_NATIVE)
      SIMDE_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.m256) / sizeof(a_.m256[0])) ; i++) {
        simde_mm256_maskstore_ps(&mem_addr_[i * sizeof(simde__m256) / sizeof(simde_float32)], simde_mm256_movm_epi32(k >> i * 8 & 0xff), a_.m256[i]);
      }
    #else
      SIMDE_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
        if (k & (UINT16_C(1) << i)) {
          mem_addr_[i] = a_.f32[i];
        }
      }
    #endif
  #endif
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_store_ps
  #define _mm512_mask_store_ps(mem_addr, k, a) simde_mm512_mask_store_ps(mem_addr, k, a)
#endif

SIMDE_FUNCTION_ATTRIBUTES
void
simde_mm512_store_pd (void * mem_addr, simde__m512d a) {
  #if defined(SIMDE_X86_AVX512F_NATIVE)
    _mm512_store_pd(mem_addr, a);
  #else
    simde_memcpy(SIMDE_ALIGN_ASSUME_LIKE(mem_addr, simde__m512d), &a, sizeof(a));
  #endif
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_store_pd
  #define _mm512_store_pd(mem_addr, a) simde_mm512_store_pd(mem_addr, a)
#endif

SIMDE_FUNCTION_ATTRIBUTES
void
simde_mm512_mask_store_pd (void * mem_addr, simde__mmask8 k, simde__m512d a) {
  #if defined(SIMDE_X86_AVX512F_NATIVE)
    _mm512_mask_store_pd(mem_addr, k, a);
  #else
    simde__m512d_private a_ = simde__m512d_to_private(a);
    simde_float64* mem_addr_ = HEDLEY_STATIC_CAST(simde_float64*, mem_addr);
    #if defined(SIMDE_X86_AVX_NATIVE)
      SIMDE_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.m256d) / sizeof(a_.m256d[0])) ; i++) {
        simde_mm256_maskstore_pd(&mem_addr_[i * sizeof(simde__m256d) / sizeof(simde_float64)], simde_mm256_movm_epi64(k >> i * 4 & 0xf), a_.m256d[i]);
      }
    #else
      SIMDE_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; i++) {
        if (k & (UINT8_C(1) << i)) {
          mem_addr_[i] = a_.f64[i];
        }
      }
    #endif
  #endif
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_store_pd
  #define _mm512_mask_store_pd(mem_addr, k, a) simde_mm512_mask_store_pd(mem_addr, k, a)
#endif

SIMDE_FUNCTION_ATTRIBUTES
void
simde_mm512_store_si512 (void * mem_addr, simde__m512i a) {
  #if defined(SIMDE_X86_AVX512F_NATIVE)
    _mm512_store_si512(HEDLEY_REINTERPRET_CAST(void*, mem_addr), a);
  #else
    simde_memcpy(SIMDE_ALIGN_ASSUME_LIKE(mem_addr, simde__m512i), &a, sizeof(a));
  #endif
}
#if defined(SIMDE_X86_AVX512F_NATIVE)
  // #define simde_mm512_store_epi8(mem_addr, a) _mm512_store_epi8(mem_addr, a)
  // #define simde_mm512_store_epi16(mem_addr, a) _mm512_store_epi16(mem_addr, a)
  #define simde_mm512_store_epi32(mem_addr, a) _mm512_store_epi32(mem_addr, a)
  #define simde_mm512_store_epi64(mem_addr, a) _mm512_store_epi64(mem_addr, a)
#else
  // #define simde_mm512_store_epi8(mem_addr, a) simde_mm512_store_si512(mem_addr, a)
  // #define simde_mm512_store_epi16(mem_addr, a) simde_mm512_store_si512(mem_addr, a)
  #define simde_mm512_store_epi32(mem_addr, a) simde_mm512_store_si512(mem_addr, a)
  #define simde_mm512_store_epi64(mem_addr, a) simde_mm512_store_si512(mem_addr, a)
#endif
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  // #undef _mm512_store_epi8
  // #undef _mm512_store_epi16
  #undef _mm512_store_epi32
  #undef _mm512_store_epi64
  #undef _mm512_store_si512
  #define _mm512_store_si512(mem_addr, a) simde_mm512_store_si512(mem_addr, a)
  // #define _mm512_store_epi8(mem_addr, a) simde_mm512_store_si512(mem_addr, a)
  // #define _mm512_store_epi16(mem_addr, a) simde_mm512_store_si512(mem_addr, a)
  #define _mm512_store_epi32(mem_addr, a) simde_mm512_store_si512(mem_addr, a)
  #define _mm512_store_epi64(mem_addr, a) simde_mm512_store_si512(mem_addr, a)
#endif

SIMDE_FUNCTION_ATTRIBUTES
void
simde_mm512_mask_store_epi32 (void * mem_addr, simde__mmask16 k, simde__m512i a) {
  #if defined(SIMDE_X86_AVX512F_NATIVE)
    _mm512_mask_store_epi32(mem_addr, k, a);
  #else
    simde__m512i_private a_ = simde__m512i_to_private(a);
    int32_t* mem_addr_ = HEDLEY_STATIC_CAST(int32_t*, mem_addr);
    #if defined(SIMDE_X86_AVX2_NATIVE)
      SIMDE_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
        simde_mm256_maskstore_epi32(&mem_addr_[i * sizeof(simde__m256i) / sizeof(int32_t)], simde_mm256_movm_epi32(k >> i * 8 & 0xff), a_.m256i[i]);
      }
    #else
      SIMDE_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
        if (k & (UINT16_C(1) << i)) {
          mem_addr_[i] = a_.i32[i];
        }
      }
    #endif
  #endif
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_store_epi32
  #define _mm512_mask_store_epi32(mem_addr, k, a) simde_mm512_mask_store_epi32(mem_addr, k, a)
#endif

SIMDE_FUNCTION_ATTRIBUTES
void
simde_mm512_mask_store_epi64 (void * mem_addr, simde__mmask8 k, simde__m512i a) {
  #if defined(SIMDE_X86_AVX512F_NATIVE)
    _mm512_mask_store_epi64(mem_addr, k, a);
  #else
    simde__m512i_private a_ = simde__m512i_to_private(a);
    int64_t* mem_addr_ = HEDLEY_STATIC_CAST(int64_t*, mem_addr);
    #if defined(SIMDE_X86_AVX2_NATIVE)
      SIMDE_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
        simde_mm256_maskstore_epi64(&mem_addr_[i * sizeof(simde__m256i) / sizeof(int64_t)], simde_mm256_movm_epi64(k >> i * 4 & 0xf), a_.m256i[i]);
      }
    #else
      SIMDE_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
        if (k & (UINT8_C(1) << i)) {
          mem_addr_[i] = a_.i64[i];
        }
      }
    #endif
  #endif
}
#if defined(SIMDE_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_store_epi64
  #define _mm512_mask_store_epi64(mem_addr, k, a) simde_mm512_mask_store_epi64(mem_addr, k, a)
#endif

SIMDE_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(SIMDE_X86_AVX512_STORE_H) */
