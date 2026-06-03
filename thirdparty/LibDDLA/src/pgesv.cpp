#include <ddla/ddla.h>
#include <cassert>
#include <vector>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>

namespace ddla{

template <typename T>
void pgesv(
    const int& n, const int& nrhs,
    T* d_A, const DdlaDesc& array_descA,
    T* d_B, const DdlaDesc& array_descB
)
{
    std::vector<int> ipiv(array_descA.m_loc());
    int info = 1;
    pgetrf(
        n, n,
        d_A, array_descA,
        ipiv.data(),
        info
    );
    if(info !=0){
        printf("Error in pzgetrf, info = %d\n", info);
        throw std::runtime_error("info !=0\n");
    }
    pgetrs(
        'N', n, nrhs,
        d_A, array_descA,
        ipiv.data(),
        d_B, array_descB
    );
}

template void pgesv<float>(
    const int& n, const int& nrhs,
    float* d_A, const DdlaDesc& array_descA,
    float* d_B, const DdlaDesc& array_descB
);

template void pgesv<double>(
    const int& n, const int& nrhs,
    double* d_A, const DdlaDesc& array_descA,
    double* d_B, const DdlaDesc& array_descB
);

template void pgesv<std::complex<float>>(
    const int& n, const int& nrhs,
    std::complex<float>* d_A, const DdlaDesc& array_descA,
    std::complex<float>* d_B, const DdlaDesc& array_descB
);

template void pgesv<std::complex<double>>(
    const int& n, const int& nrhs,
    std::complex<double>* d_A, const DdlaDesc& array_descA,
    std::complex<double>* d_B, const DdlaDesc& array_descB
);

}