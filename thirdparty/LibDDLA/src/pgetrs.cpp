#include <ddla/ddla.h>
#include <cassert>
#include <vector>
namespace ddla{

template<typename T>
void pgetrs(
    const char& trans, const int& n, const int& nrhs,
    T* d_A, const DdlaDesc& array_descA,
    const int* ipiv, // host
    T* d_B, const DdlaDesc& array_descB
)
{
    assert(trans == 'N');
    char direc = 'F';
    char rowcol = 'R';
    char pivroc='C';
    plapiv(
        direc, rowcol, pivroc,
        n, nrhs,
        d_B, array_descB,
        ipiv, array_descA,
        nullptr
    );
    ptrtrs(
        'L', 'L', 'N', 'U', n, nrhs,
        d_A, array_descA,
        d_B, array_descB
    );
    ptrtrs(
        'L', 'U', 'N', 'N', n, nrhs,
        d_A, array_descA,
        d_B, array_descB
    );
}

template void pgetrs<double>(
    const char& trans, const int& n, const int& nrhs,
    double* d_A, const DdlaDesc& array_descA,
    const int* ipiv, // host
    double* d_B, const DdlaDesc& array_descB
);

template void pgetrs<float>(
    const char& trans, const int& n, const int& nrhs,
    float* d_A, const DdlaDesc& array_descA,
    const int* ipiv, // host
    float* d_B, const DdlaDesc& array_descB
);

template void pgetrs<std::complex<double>>(
    const char& trans, const int& n, const int& nrhs,
    std::complex<double>* d_A, const DdlaDesc& array_descA,
    const int* ipiv, // host
    std::complex<double>* d_B, const DdlaDesc& array_descB
);

template void pgetrs<std::complex<float>>(
    const char& trans, const int& n, const int& nrhs,
    std::complex<float>* d_A, const DdlaDesc& array_descA,
    const int* ipiv, // host
    std::complex<float>* d_B, const DdlaDesc& array_descB
);


} // namespace ddla
