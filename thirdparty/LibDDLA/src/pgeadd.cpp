#include <ddla/ddla.h>
#include <cassert>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <vector>
#include <ddla/transport_block.h>
#include <ddla/geam.h>
#include <ddla/ddla_comm.h>

namespace ddla{

template <typename T>
void pgeadd(
    const char& transa, const char& transb,
    const int& m, const int& n,
    const T& alpha,
    const T* d_A, const DdlaDesc& array_descA,
    const T& beta,
    const T* d_B, const DdlaDesc& array_descB,
    T* d_C, const DdlaDesc& array_descC
)
{
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();

    if(transa != 'N' || transb != 'N')
    {
        if(array_descA.nprows() != array_descA.npcols()){
            throw std::runtime_error("the trans multiplication is now implemented for mxn(m!=n) mpi grid");
        }
    }
    {
        int mbA, nbA, mbB, nbB, mbC, nbC;
        mbC = array_descC.mb();
        nbC = array_descC.nb();
        if(transa == 'N'){
            mbA = array_descA.mb();
            nbA = array_descA.nb();
        }else{
            mbA = array_descA.nb();
            nbA = array_descA.mb();
        }

        if(transb == 'N'){
            mbB = array_descB.mb();
            nbB = array_descB.nb();
        }else{
            mbB = array_descB.nb();
            nbB = array_descB.mb();
        }
        assert(mbA == mbB && mbA == mbC);
        assert(nbA == nbB && nbA == nbC);
    }

    #ifdef DDLA_USE_CCL
    ncclComm_t nccl_comm = ddla_handle->nccl_comm;
    ncclComm_t row_nccl_comm = ddla_handle->nccl_row_comm;
    ncclComm_t col_nccl_comm = ddla_handle->nccl_col_comm;
    #else
    MPI_Comm nccl_comm = ddla_handle->comm;
    MPI_Comm row_nccl_comm = ddla_handle->row_comm;
    MPI_Comm col_nccl_comm = ddla_handle->col_comm;
    #endif
    int nprows = array_descC.nprows();
    int npcols = array_descC.npcols();
    int myprow = array_descC.myprow();
    int mypcol = array_descC.mypcol();

    int m_loc_C = num_loc(m, array_descC.mb(), array_descC.myprow(), array_descC.irsrc(), array_descC.nprows());
    int n_loc_C = num_loc(n, array_descC.nb(), array_descC.mypcol(), array_descC.icsrc(), array_descC.npcols());

    deviceStream_t stream = ddla_handle->stream;

    deblasOperation_t opA = (transa == 'N') ? DEBLAS_OP_N :
                            (transa == 'T') ? DEBLAS_OP_T : DEBLAS_OP_C;
    deblasOperation_t opB = (transb == 'N') ? DEBLAS_OP_N :
                            (transb == 'T') ? DEBLAS_OP_T : DEBLAS_OP_C;

    if(transa == 'N' && transb == 'N'){
        BLAS_CHECK(deblasGeam(
            ddla_handle->blasH, opA, opB,
            m_loc_C, n_loc_C,
            alpha, 
            d_A, array_descA.lld(),
            beta,
            d_B, array_descB.lld(),
            d_C, array_descC.lld()
        ));
        
    }else if (transa != 'N' && transb != 'N'){
        if(myprow == mypcol){
            BLAS_CHECK(deblasGeam(
                ddla_handle->blasH, opA, opB,
                m_loc_C, n_loc_C,
                alpha, 
                d_A, array_descA.lld(),
                beta,
                d_B, array_descB.lld(),
                d_C, array_descC.lld()
            ));
        }else{
            T* d_temp;
            DEVICE_CHECK(deviceMallocAsync((void**)&d_temp, m_loc_C * n_loc_C * sizeof(T), stream));
            BLAS_CHECK(deblasGeam(
                ddla_handle->blasH, opA, opB,
                n_loc_C, m_loc_C,
                alpha,
                d_A, array_descA.lld(),
                beta,
                d_B, array_descB.lld(),
                d_temp, n_loc_C
            ));
            int trans_rank = ddla_handle->rc_to_rank(mypcol, myprow);
            if(myprow > mypcol){
                CCL_CHECK(cclSend(d_temp, m_loc_C * n_loc_C, trans_rank, nccl_comm, stream));
                CCL_CHECK(cclRecv(d_C, m_loc_C * n_loc_C, trans_rank, nccl_comm, stream));
            }else{
                CCL_CHECK(cclRecv(d_C, m_loc_C * n_loc_C, trans_rank, nccl_comm, stream));
                CCL_CHECK(cclSend(d_temp, m_loc_C * n_loc_C, trans_rank, nccl_comm, stream));
            }
            DEVICE_CHECK(deviceFreeAsync(d_temp, stream));
        }
    }else{
        if(myprow == mypcol){
            BLAS_CHECK(deblasGeam(
                ddla_handle->blasH, opA, opB,
                m_loc_C, n_loc_C,
                alpha, 
                d_A, array_descA.lld(),
                beta,
                d_B, array_descB.lld(),
                d_C, array_descC.lld()
            ));
        }else{
            T* d_temp;
            DEVICE_CHECK(deviceMallocAsync((void**)&d_temp, m_loc_C * n_loc_C * sizeof(T), stream));
            const T* d_comm = transa != 'N' ? d_A : d_B;
            const T* d_nt = transa == 'N' ? d_A : d_B;
            deblasOperation_t op_trans = transa != 'N' ? opA : opB;
            int trans_rank = ddla_handle->rc_to_rank(mypcol, myprow);
            if(myprow > mypcol){
                CCL_CHECK(cclSend(d_comm, m_loc_C * n_loc_C, trans_rank, nccl_comm, stream));
                CCL_CHECK(cclRecv(d_temp, m_loc_C * n_loc_C, trans_rank, nccl_comm, stream));
            }else{
                CCL_CHECK(cclRecv(d_temp, m_loc_C * n_loc_C, trans_rank, nccl_comm, stream));
                CCL_CHECK(cclSend(d_comm, m_loc_C * n_loc_C, trans_rank, nccl_comm, stream));
            }
            BLAS_CHECK(deblasGeam(
                ddla_handle->blasH, op_trans, DEBLAS_OP_N,
                m_loc_C, n_loc_C,
                alpha,
                d_temp, n_loc_C,
                beta,
                d_nt, m_loc_C,
                d_C, m_loc_C
            ));
            DEVICE_CHECK(deviceFreeAsync(d_temp, stream));
        }
    }

    return;
    
}

template void pgeadd<float>(
    const char& transa, const char& transb,
    const int& m, const int& n,
    const float& alpha,
    const float* d_A, const DdlaDesc& array_descA,
    const float& beta,
    const float* d_B, const DdlaDesc& array_descB,
    float* d_C, const DdlaDesc& array_descC
);

template void pgeadd<double>(
    const char& transa, const char& transb,
    const int& m, const int& n,
    const double& alpha,
    const double* d_A, const DdlaDesc& array_descA,
    const double& beta,
    const double* d_B, const DdlaDesc& array_descB,
    double* d_C, const DdlaDesc& array_descC
);

template void pgeadd<std::complex<float>>(
    const char& transa, const char& transb,
    const int& m, const int& n,
    const std::complex<float>& alpha,
    const std::complex<float>* d_A, const DdlaDesc& array_descA,
    const std::complex<float>& beta,
    const std::complex<float>* d_B, const DdlaDesc& array_descB,
    std::complex<float>* d_C, const DdlaDesc& array_descC
);

template void pgeadd<std::complex<double>>(
    const char& transa, const char& transb,
    const int& m, const int& n,
    const std::complex<double>& alpha,
    const std::complex<double>* d_A, const DdlaDesc& array_descA,
    const std::complex<double>& beta,
    const std::complex<double>* d_B, const DdlaDesc& array_descB,
    std::complex<double>* d_C, const DdlaDesc& array_descC
);


}