#include <ddla/ddla.h>
#include <cassert>
#include <vector>
#include <ddla/ddla_stream.h>
namespace ddla{

template<typename T>
void pgetrs(
    const char& trans, const int& n, const int& nrhs,
    T* d_A, const DdlaDesc& array_descA,
    const int* ipiv, // host
    T* d_B, const DdlaDesc& array_descB
)
{
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();
    
    assert(trans == 'N');
    char direc = 'F';
    char rowcol = 'R';
    char pivroc='C';
    printf("myid:%d, start pzlapiv\n",ddla_handle->myid);
    double start_time_swap = MPI_Wtime();
    plapiv(
        direc, rowcol, pivroc,
        n, nrhs,
        d_B, array_descB,
        ipiv, array_descA,
        nullptr
    );
    printf("myid:%d, pzlapiv time:%lf\n",ddla_handle->myid,MPI_Wtime()-start_time_swap);
    double start_time = MPI_Wtime();
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
    printf("myid:%d 2xtrtrs time:%lf\n",ddla_handle->myid,MPI_Wtime()-start_time);

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