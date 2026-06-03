#include <ddla/ddla.h>
#include <cassert>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <ddla/swap.h>
#include <ddla/ddla_comm.h>

namespace ddla{

template <typename T>
void plapiv(
    const char& direc, const char& rowcol, const char& pivroc,
    const int& m, const int& n,
    T* d_A,const DdlaDesc& array_descA,
    const int* ipiv, const DdlaDesc& array_descIP,
    int* iwork
)
{
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();
    
    assert(direc=='F');
    assert(rowcol=='R');
    assert(pivroc=='C');
    assert(m<=array_descA.m());
    assert(n==array_descA.n());

    int nprows = array_descA.nprows();
    int myprow = array_descA.myprow();

    int mb = array_descA.mb();
    int lldA = array_descA.lld();

    T*temp_A_target;
    DEVICE_CHECK(deviceMallocAsync(&temp_A_target, sizeof(T)*array_descA.n_loc(), ddla_handle->stream));

    deviceStream_t stream = ddla_handle->stream;
    deblasHandle_t blasH = ddla_handle->blasH;
    

    // 初始化 NCCL
    MPI_Comm col_comm = ddla_handle->col_comm;
    #ifdef DDLA_USE_CCL
    ncclComm_t col_nccl_comm=ddla_handle->nccl_col_comm;
    #else
    MPI_Comm col_nccl_comm=ddla_handle->col_comm;
    #endif
    int i_loc;
    int owner_row;
    int target_row;
    int target_i_global,target_i_loc;
    for(int i=0; i<m; i++){
        i_loc = array_descA.indx_g2l_r(i);
        owner_row = indxg2p(i, mb, array_descA.irsrc(), nprows);
        if(i_loc>=0){
            target_i_global = ipiv[i_loc] - 1;
        }
        MPI_Bcast(&target_i_global, 1, MPI_INT, owner_row, col_comm);
        if(target_i_global == i)
            continue;
        target_row = indxg2p(target_i_global,array_descIP.mb(),array_descA.irsrc(),nprows);
        target_i_loc = array_descIP.indx_g2l_r(target_i_global);
        // if(i_loc>=0)
        //     printf("myid:%d, i:%d, i_loc:%d, target_i_global:%d, target_i_loc:%d, owner_row:%d, target_row:%d\n",mpi_comm_global_h.myid,i,i_loc,target_i_global,target_i_loc,owner_row,target_row);
        // else if(target_i_loc>=0)
        //     printf("myid:%d, i:%d, i_loc:%d, target_i_global:%d, target_i_loc:%d, owner_row:%d, target_row:%d\n",mpi_comm_global_h.myid,i,i_loc,target_i_global,target_i_loc,owner_row,target_row);
        if(target_row==owner_row){
            if(myprow==owner_row)
                BLAS_CHECK(deblasSwap(blasH, array_descA.n_loc(), d_A + i_loc, lldA, d_A + target_i_loc, lldA));
        }else{
            if(myprow==target_row){
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    temp_A_target, 1 * sizeof(T),
                    d_A + target_i_loc, lldA * sizeof(T),
                    1 * sizeof(T), array_descA.n_loc(),
                    deviceMemcpyDeviceToDevice, stream
                ));
                CCL_CHECK(cclSend(temp_A_target, array_descA.n_loc(), owner_row, col_nccl_comm, stream));
                CCL_CHECK(cclRecv(temp_A_target, array_descA.n_loc(), owner_row, col_nccl_comm, stream));
                BLAS_CHECK(deblasSwap(blasH, array_descA.n_loc(), d_A + target_i_loc, lldA, temp_A_target, 1));
            }else if(myprow==owner_row){
                CCL_CHECK(cclRecv(temp_A_target, array_descA.n_loc(), target_row, col_nccl_comm, stream));
                BLAS_CHECK(deblasSwap(blasH, array_descA.n_loc(), d_A + i_loc, lldA, temp_A_target, 1));
                CCL_CHECK(cclSend(temp_A_target, array_descA.n_loc(), target_row, col_nccl_comm, stream));
            }
        }
        // pswap(
        //     n,
        //     d_A, i + 1, 1, array_descA, array_descA.m(),
        //     d_A, target_i_global + 1, 1, array_descA, array_descA.m()
        // );
    }

    DEVICE_CHECK(deviceFreeAsync(temp_A_target, stream));
    DEVICE_CHECK(deviceStreamSynchronize(stream));

}

template void plapiv<std::complex<double>>(
    const char& direc, const char& rowcol, const char& pivroc,
    const int& m, const int& n,
    std::complex<double>* d_A,const DdlaDesc& array_descA,
    const int* ipiv, const DdlaDesc& array_descIP,
    int* iwork
);

template void plapiv<std::complex<float>>(
    const char& direc, const char& rowcol, const char& pivroc,
    const int& m, const int& n,
    std::complex<float>* d_A,const DdlaDesc& array_descA,
    const int* ipiv, const DdlaDesc& array_descIP,
    int* iwork
);

template void plapiv<float>(
    const char& direc, const char& rowcol, const char& pivroc,
    const int& m, const int& n,
    float* d_A,const DdlaDesc& array_descA,
    const int* ipiv, const DdlaDesc& array_descIP,
    int* iwork
);

template void plapiv<double>(
    const char& direc, const char& rowcol, const char& pivroc,
    const int& m, const int& n,
    double* d_A,const DdlaDesc& array_descA,
    const int* ipiv, const DdlaDesc& array_descIP,
    int* iwork
);


}