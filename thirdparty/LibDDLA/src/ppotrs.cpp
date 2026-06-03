#include <ddla/ddla.h>
#include <cassert>
#include <vector>
#include <ddla/ddla_stream.h>
namespace ddla{

template <typename T>
void ppotrs(
    const char& side, const char& uplo, const char& trans,
    const int& n, const int& nrhs,
    T* d_A, const DdlaDesc& array_descA,
    T* d_B, const DdlaDesc& array_descB,
    bool is_nega, int location
)
{
    assert(location == -1);
    assert(trans == 'N');
    assert(side == 'L');
    double start_time = MPI_Wtime();
    // if(is_nega){
    //     int i_loc = array_descA.indx_g2l_r(n - 1);
    //     int j_loc = array_descA.indx_g2l_c(n - 1);
    //     if(i_loc >= 0 && j_loc >= 0){
    //         DdlaHandle_t ddla_handle = array_descA.ddla_handle();
    //         DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    //         T correction;
    //         DEVICE_CHECK(deviceMemcpy(&correction, d_A + i_loc + j_loc * array_descA.lld(), sizeof(T), deviceMemcpyDeviceToHost));
    //         correction = -correction;
    //         DEVICE_CHECK(deviceMemcpy(d_A + i_loc + j_loc * array_descA.lld(), &correction, sizeof(T), deviceMemcpyHostToDevice));
    //     }
    // }
    ptrtrs(
        side, uplo, uplo == 'L' ? 'N' : 'C', 'N', n, nrhs,
        d_A, array_descA,
        d_B, array_descB
    );
    if(is_nega){
        int i_loc = array_descA.indx_g2l_r(n - 1);
        int j_loc = array_descA.indx_g2l_c(n - 1);
        if(i_loc >= 0 && j_loc >= 0){
            DdlaHandle_t ddla_handle = array_descA.ddla_handle();
            DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
            T correction;
            DEVICE_CHECK(deviceMemcpy(&correction, d_A + i_loc + j_loc * array_descA.lld(), sizeof(T), deviceMemcpyDeviceToHost));
            correction = -correction;
            DEVICE_CHECK(deviceMemcpy(d_A + i_loc + j_loc * array_descA.lld(), &correction, sizeof(T), deviceMemcpyHostToDevice));
        }
    }
    ptrtrs(
        side, uplo, uplo == 'L' ? 'C' : 'N', 'N', n, nrhs,
        d_A, array_descA,
        d_B, array_descB
    );
}

template void ppotrs<std::complex<double>>(
    const char& side, const char& uplo, const char& trans,
    const int& n, const int& nrhs,
    std::complex<double>* d_A, const DdlaDesc& array_descA,
    std::complex<double>* d_B, const DdlaDesc& array_descB,
    bool is_nega, int location
);

template void ppotrs<std::complex<float>>(
    const char& side, const char& uplo, const char& trans,
    const int& n, const int& nrhs,
    std::complex<float>* d_A, const DdlaDesc& array_descA,
    std::complex<float>* d_B, const DdlaDesc& array_descB,
    bool is_nega, int location
);


}