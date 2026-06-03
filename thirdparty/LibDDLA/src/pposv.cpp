#include <ddla/ddla.h>

namespace ddla{

template <typename T>
void pposv(
    const char& side, const char& uplo, const char& trans,
    const int & n, const int& nrhs,
    T* d_A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    T* d_B, const int& ib, const int& jb, const DdlaDesc& array_descB,
    int& info, // host pointer
    bool is_head, int location
)
{
    bool is_nega = ppotrf(uplo, n, d_A, ia, ja, array_descA, info, is_head, location);
    if(info == 0 && !is_nega)
        ppotrs(side, uplo, trans, n, nrhs, d_A, array_descA, d_B, array_descB, is_nega, location);
    return;
}

template void pposv<std::complex<float>>(
    const char& side, const char& uplo, const char& trans,
    const int & n, const int& nrhs,
    std::complex<float>* d_A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    std::complex<float>* d_B, const int& ib, const int& jb, const DdlaDesc& array_descB,
    int& info, // host pointer
    bool is_head, int location
);

template void pposv<std::complex<double>>(
    const char& side, const char& uplo, const char& trans,
    const int & n, const int& nrhs,
    std::complex<double>* d_A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    std::complex<double>* d_B, const int& ib, const int& jb, const DdlaDesc& array_descB,
    int& info, // host pointer
    bool is_head, int location
);

} // namespace DDLA