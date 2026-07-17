#include <ddla/ddla.h>
#include <cassert>
#include <ddla/ddla_stream.h>

namespace ddla {

/**
 * @brief Distributed linear-system solver without pivoting (driver).
 *
 * Convenience wrapper: pgetrf_nopiv (LU) + pgetrs_nopiv (solve).
 * Solves A * X = B without pivoting.
 *
 * @tparam T   Scalar type.
 * @param n       Order of square matrix A.
 * @param nrhs    Number of right-hand sides.
 * @param d_A     Device pointer to A (input: coefficient; output: LU factors).
 * @param array_descA  DdlaDesc for A.
 * @param d_B     Device pointer to RHS / solution B (input/output).
 * @param array_descB  DdlaDesc for B.
 * @throws std::runtime_error if LU factorization fails (info != 0).
 */
template <typename T>
void pgesv_nopiv(
    const int& n, const int& nrhs,
    T* d_A, const DdlaDesc& array_descA,
    T* d_B, const DdlaDesc& array_descB
)
{
    int info = 1;
    pgetrf_nopiv(n, n, d_A, array_descA, info);
    if (info != 0) {
        printf("Error in pgetrf_nopiv, info = %d\n", info);
        throw std::runtime_error("info != 0\n");
    }
    pgetrs_nopiv('N', n, nrhs, d_A, array_descA, d_B, array_descB);
}

template void pgesv_nopiv<float>(
    const int& n, const int& nrhs,
    float* d_A, const DdlaDesc& array_descA,
    float* d_B, const DdlaDesc& array_descB
);
template void pgesv_nopiv<double>(
    const int& n, const int& nrhs,
    double* d_A, const DdlaDesc& array_descA,
    double* d_B, const DdlaDesc& array_descB
);
template void pgesv_nopiv<std::complex<float>>(
    const int& n, const int& nrhs,
    std::complex<float>* d_A, const DdlaDesc& array_descA,
    std::complex<float>* d_B, const DdlaDesc& array_descB
);
template void pgesv_nopiv<std::complex<double>>(
    const int& n, const int& nrhs,
    std::complex<double>* d_A, const DdlaDesc& array_descA,
    std::complex<double>* d_B, const DdlaDesc& array_descB
);

} // namespace ddla
