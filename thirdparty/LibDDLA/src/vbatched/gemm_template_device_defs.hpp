/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates
       @author Ahmad Abdelfattah
       @author Azzam Haidar

*/

#ifndef GEMM_TEMPLATE_DEVICE_DEFS_H
#define GEMM_TEMPLATE_DEVICE_DEFS_H

// =============================================================================
// conjugation -- double complex
template<const int conjugate>
__device__ inline
DdlaDoubleComplex conj(DdlaDoubleComplex &x) {return DDLA_Z_CONJ(x);}

template<>
__device__ inline
DdlaDoubleComplex conj<0>(DdlaDoubleComplex &x) {return x;}

// conjugation -- single complex
template<const int conjugate>
__device__ inline
DdlaFloatComplex conj(DdlaFloatComplex &x) {return DDLA_C_CONJ(x);}

template<>
__device__ inline
DdlaFloatComplex conj<0>(DdlaFloatComplex &x) {return x;}

// conjugation -- real single & double
template<const int conjugate>
__device__ static inline
double conj(double &x) {return x;}

template<const int conjugate>
__device__ static inline
float conj(float &x) {return x;}


// =============================================================================
#define fetch(A, m, n, bound)                                                \
    offs_d##A[((ptrdiff_t)(n*LD##A + m) < (bound))                           \
                  ? (ptrdiff_t)(n*LD##A + m)                                 \
                  : (bound)]

// =============================================================================
#if defined(PRECISION_z)
    #define add(A, B)        DDLA_Z_ADD(A, B)
    #define mul(A, B)        DDLA_Z_MUL(A, B)
    #define div(A, B)        DDLA_Z_DIV(A, B)
    #define fma(A, B, C) C = ddlaComplexFma(A, B, C)
    #define make_FloatingPoint(x, y) DDLA_Z_MAKE(x, y)
#elif defined(PRECISION_c)
    #define add(A, B)        DDLA_C_ADD(A, B)
    #define mul(A, B)        DDLA_C_MUL(A, B)
    #define div(A, B)        DDLA_C_DIV(A, B)
    #define fma(A, B, C) C = ddlaComplexFma(A, B, C)
    #define make_FloatingPoint(x, y) DDLA_C_MAKE(x, y)
#elif defined(PRECISION_h)
    #define add(A, B)         (A+B)
    #define mul(A, B)         (A*B)
    #define div(A, B)         (A/B)
    #define fma(A, B, C) C += (A*B)
    #define make_FloatingPoint(x, y) ((__half)x)
#else
    #define add(A, B)         (A+B)
    #define mul(A, B)         (A*B)
    #define div(A, B)         (A/B)
    #define fma(A, B, C) C += (A*B)
    #define make_FloatingPoint(x, y) (x)
#endif

#if defined(PRECISION_z)
    #define ddla_internal_atomic_add ddla_internal_zatomic_add
#elif defined(PRECISION_c)
    #define ddla_internal_atomic_add ddla_internal_catomic_add
#elif defined(PRECISION_d)
    #define ddla_internal_atomic_add ddla_internal_datomic_add
#else
    #define ddla_internal_atomic_add ddla_internal_satomic_add
#endif

#endif // GEMM_TEMPLATE_DEVICE_DEFS_H
