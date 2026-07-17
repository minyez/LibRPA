#include <ddla/ddla.h>
#include <cassert>
#include <ddla/ddla_stream.h>

namespace ddla {

/**
 * @brief Distributed LU solve without pivoting.
 *
 * Solves A * X = B using the factors from pgetrf_nopiv.  Because no
 * pivoting was performed, the solve is simply two triangular solves:
 *   1) L * Y = B  (forward, lower triangular, unit diagonal)
 *   2) U * X = Y  (backward, upper triangular, non-unit diagonal)
 *
 * Only trans='N' is supported.
 *
 * @tparam T   Scalar type.
 * @param trans   'N' -- no transpose (only 'N' supported).
 * @param n       Order of matrix A.
 * @param nrhs    Number of right-hand sides.
 * @param d_A     Device pointer to LU factors (from pgetrf_nopiv).
 * @param array_descA  DdlaDesc for A.
 * @param d_B     Device pointer to RHS / solution B (input/output).
 * @param array_descB  DdlaDesc for B.
 */
template<typename T>
void pgetrs_nopiv(
    const char& trans, const int& n, const int& nrhs,
    T* d_A, const DdlaDesc& array_descA,
    T* d_B, const DdlaDesc& array_descB
)
{
    assert(trans == 'N');

    // Forward solve: L * Y = B (L lower, unit diagonal)
    ptrtrs('L', 'L', 'N', 'U', n, nrhs,
           d_A, array_descA,
           d_B, array_descB);

    // Backward solve: U * X = Y (U upper, non-unit diagonal)
    ptrtrs('L', 'U', 'N', 'N', n, nrhs,
           d_A, array_descA,
           d_B, array_descB);
}

template void pgetrs_nopiv<float>(
    const char& trans, const int& n, const int& nrhs,
    float* d_A, const DdlaDesc& array_descA,
    float* d_B, const DdlaDesc& array_descB
);
template void pgetrs_nopiv<double>(
    const char& trans, const int& n, const int& nrhs,
    double* d_A, const DdlaDesc& array_descA,
    double* d_B, const DdlaDesc& array_descB
);
template void pgetrs_nopiv<std::complex<float>>(
    const char& trans, const int& n, const int& nrhs,
    std::complex<float>* d_A, const DdlaDesc& array_descA,
    std::complex<float>* d_B, const DdlaDesc& array_descB
);
template void pgetrs_nopiv<std::complex<double>>(
    const char& trans, const int& n, const int& nrhs,
    std::complex<double>* d_A, const DdlaDesc& array_descA,
    std::complex<double>* d_B, const DdlaDesc& array_descB
);

} // namespace ddla
