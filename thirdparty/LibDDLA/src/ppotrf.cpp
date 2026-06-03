#include <ddla/ddla.h>
#include <cassert>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <vector>
#include <type_traits>
#include <cmath>
#include <ddla/trsm.h>
#include <ddla/potrf.h>
#include <ddla/gemmBatched.h>
#include <ddla/herk.h>
#include <ddla/gemm.h>
#include <ddla/ddla_comm.h>

namespace ddla{

template<typename T>
bool ppotrf(
    const char& uplo, const int& n,
    T* A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    int& info, // host pointer
    bool is_head, int location
)
{
    bool is_nega = false;
    assert(uplo == 'L');
    assert(array_descA.mb() == array_descA.nb());
    assert(n > 0);
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();
    if(is_head)
    if(location != -1 && location != n){
        pswap(
            n,
            A, location, 1, array_descA, array_descA.m(),
            A, n, 1, array_descA, array_descA.m()
        );
        pswap(
            n,
            A, 1, location, array_descA, 1,
            A, 1, location, array_descA, 1
        );
    }

    int nb = array_descA.mb();
    int lldA = array_descA.lld();

    int nprows = array_descA.nprows();
    int npcols = array_descA.npcols();
    int myprow = array_descA.myprow();
    int mypcol = array_descA.mypcol();

    // 初始化 NCCL  
    #ifdef DDLA_USE_CCL
    ncclComm_t row_comm=ddla_handle->nccl_row_comm;
    ncclComm_t col_comm=ddla_handle->nccl_col_comm;
    #else
    MPI_Comm row_comm=ddla_handle->row_comm;
    MPI_Comm col_comm=ddla_handle->col_comm;
    #endif
    deviceStream_t stream=ddla_handle->stream;
    deblasHandle_t blasH=ddla_handle->blasH;
    desolverHandle_t solverH=ddla_handle->solverH;

    deblasFillMode_t uplo_device = (uplo == 'U') ? DEBLAS_FILL_MODE_UPPER : DEBLAS_FILL_MODE_LOWER;
    deblasDiagType_t diag_device = DEBLAS_DIAG_NON_UNIT;
    deblasOperation_t trans_device = DEBLAS_OP_C;
    deblasSideMode_t side_device = (uplo == 'U') ?DEBLAS_SIDE_LEFT : DEBLAS_SIDE_RIGHT;

    T* d_block_diag, *d_block_row, *d_block_col;
    DEVICE_CHECK(deviceMallocAsync((void**)&d_block_diag, nb * nb * sizeof(T), stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_block_row, nb * array_descA.n_loc() * sizeof(T), stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_block_col, nb * array_descA.m_loc() * sizeof(T), stream));
    int *d_info;
    DEVICE_CHECK(deviceMallocAsync((void**)&d_info, sizeof(int), stream));

    #ifdef DDLA_USE_GPU_CPU_TUNNEL
    std::vector<T> h_temp(nb * std::max(array_descA.n_loc(), array_descA.m_loc()));
    #endif

    int owner_row, owner_col;
    int mm_row_start, mm_col_start;
    int nb_real;

    int num_row_block = array_descA.m_loc() / nb;
    int num_col_block = array_descA.n_loc() / nb;
    int batchCount = num_row_block * num_col_block;

    T** d_A_array, ** d_B_array, ** d_C_array;
    std::vector<T*> h_A_array(batchCount), h_B_array(batchCount), h_C_array(batchCount);

    DEVICE_CHECK(deviceMallocAsync((void**)&d_A_array, batchCount * sizeof(T*), stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_B_array, batchCount * sizeof(T*), stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_C_array, batchCount * sizeof(T*), stream));
    int h_info;
    int i_batch_count, row_s, col_s, row_remain, col_remain, length_row, length_col;
    for(int n_s = 0; n_s < array_descA.m(); n_s += nb)
    {
        nb_real = std::min(nb, array_descA.m() - n_s);
        // printf("myid:%d, n_s:%d, nb_real:%d\n",ddla_handle->myid, n_s, nb_real);
        mm_row_start = num_loc(n_s, nb, myprow, array_descA.irsrc(), nprows);
        mm_col_start = num_loc(n_s, nb, mypcol, array_descA.icsrc(), npcols);

        owner_row = indxg2p(n_s, nb, array_descA.irsrc(), nprows);
        owner_col = indxg2p(n_s, nb, array_descA.icsrc(), npcols);

        if(myprow == owner_row && mypcol == owner_col)
        {
            if(n_s + nb_real == array_descA.m() && is_head){
                if(nb_real > 1){
                    SOLVER_CHECK(desolverPotrf(solverH, uplo_device, nb_real - 1, A + mm_row_start + mm_col_start * lldA, lldA, d_info));
                    BLAS_CHECK(deblasTrsm(
                        blasH, side_device, uplo_device, trans_device, diag_device,
                        1, nb_real - 1, (T)1.0, 
                        A + mm_row_start + mm_col_start * lldA, lldA,
                        A + mm_row_start + nb_real - 1 + mm_col_start * lldA, lldA
                    ));
                    BLAS_CHECK(deblasHerk(
                        blasH, uplo_device, DEBLAS_OP_N,
                        1, nb_real - 1,
                        -1.0, A + mm_row_start + nb_real - 1 + mm_col_start * lldA, lldA,
                        1.0, A + mm_row_start + nb_real - 1 + (mm_col_start + nb_real - 1) * lldA, lldA
                    ));
                }
                T last_value;
                DEVICE_CHECK(deviceMemcpyAsync(&last_value, A + mm_row_start + nb_real - 1 + (mm_col_start + nb_real - 1) * lldA, sizeof(T), deviceMemcpyDeviceToHost, stream));
                is_nega = false;
                if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>){
                    if(last_value < 0){
                        is_nega = true;
                        last_value = -last_value;
                    }
                }else if constexpr (std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>>){
                    if(last_value.real() < 0){
                        is_nega = true;
                        last_value = -last_value;
                    }                
                }else{
                    throw std::runtime_error("unsupported template type\n");
                }
                last_value = std::sqrt(last_value);
                DEVICE_CHECK(deviceMemcpyAsync(A + mm_row_start + nb_real - 1 + (mm_col_start + nb_real - 1) * lldA, &last_value, sizeof(T), deviceMemcpyHostToDevice, stream));
            }else
                SOLVER_CHECK(desolverPotrf(solverH, uplo_device, nb_real, A + mm_row_start + mm_col_start * lldA, lldA, d_info));
            DEVICE_CHECK(deviceStreamSynchronize(stream));
            DEVICE_CHECK(deviceMemcpy(&info, d_info, sizeof(int), deviceMemcpyDeviceToHost));
            DEVICE_CHECK(deviceMemcpy2DAsync(
                d_block_diag, nb_real * sizeof(T),
                A + mm_row_start + mm_col_start * lldA, lldA * sizeof(T),
                nb_real * sizeof(T), nb_real,
                deviceMemcpyDeviceToDevice, stream
            ));
        }
        if(n_s + nb_real == array_descA.m())
            MPI_CHECK(MPI_Bcast(&is_nega, 1, MPI_CXX_BOOL, ddla_handle->rc_to_rank(owner_row, owner_col), ddla_handle->comm));
        MPI_CHECK(MPI_Bcast(&info, 1, MPI_INT, ddla_handle->rc_to_rank(owner_row, owner_col), ddla_handle->comm));
        if(info != 0){
            info = info + n_s;
            printf("the matrix is not positive definite myid:%d, info:%d\n", ddla_handle->myid, info);
            return false;
        }
        if(myprow == owner_row)
            mm_row_start += nb_real;
        length_row = array_descA.m_loc() - mm_row_start;
        if(mypcol == owner_col){
            #ifdef DDLA_USE_GPU_CPU_TUNNEL
            MPI_CHECK(cclBcast(h_temp.data(), d_block_diag, nb_real * nb_real, owner_row, ddla_handle->col_comm, ddla_handle->stream));
            #else
            CCL_CHECK(cclBcast(d_block_diag, nb_real * nb_real, owner_row, col_comm, stream));
            #endif
            if(length_row > 0){
                BLAS_CHECK(deblasTrsm(
                    blasH, side_device, uplo_device, trans_device, diag_device,
                    length_row, nb_real, (T)1.0, 
                    d_block_diag, nb_real,
                    A + mm_row_start + mm_col_start * lldA, lldA
                ));
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_block_col, length_row * sizeof(T),
                    A + mm_row_start + mm_col_start * lldA, lldA * sizeof(T),
                    length_row * sizeof(T), nb_real,
                    deviceMemcpyDeviceToDevice, stream
                ));
            }
        }
        if(mypcol == owner_col)
            mm_col_start += nb_real;
        length_col = array_descA.n_loc() - mm_col_start;
        if(length_row > 0){
            #ifdef DDLA_USE_GPU_CPU_TUNNEL
            MPI_CHECK(cclBcast(h_temp.data(), d_block_col, length_row * nb_real, owner_col, ddla_handle->row_comm, ddla_handle->stream));
            #else
            CCL_CHECK(cclBcast(d_block_col, length_row * nb_real, owner_col, row_comm, stream));
            #endif
        }
        if(myprow == mypcol){
            if(length_col > 0)
                DEVICE_CHECK(deviceMemcpyAsync(d_block_row, d_block_col, length_col * nb_real * sizeof(T), deviceMemcpyDeviceToDevice, stream));
        }
        if(length_col > 0){
            #ifdef DDLA_USE_GPU_CPU_TUNNEL
            MPI_CHECK(cclBcast(h_temp.data(), d_block_row, nb_real * length_col, mypcol, ddla_handle->col_comm, ddla_handle->stream));
            #else
            CCL_CHECK(cclBcast(d_block_row, nb_real * length_col, mypcol, col_comm, stream));
            #endif
        }
        if(myprow == mypcol){
            if(length_row > 0)
                BLAS_CHECK(deblasHerk(
                    blasH, uplo_device, DEBLAS_OP_N,
                    length_row, nb_real,
                    -1.0, d_block_col, length_row,
                    1.0, A + mm_row_start + mm_col_start * lldA, lldA
                ));
        }else{
            // the first approach in which the unused block will be polluted
            // if(length_row > 0 && length_col > 0)
            //     BLAS_CHECK(deblasGemm(
            //         blasH, DEBLAS_OP_N, DEBLAS_OP_T,
            //         length_row, length_col, nb_real,
            //         (T)-1.0,
            //         d_block_col, length_row,
            //         d_block_row, length_col,
            //         (T)1.0,
            //         A + mm_row_start + mm_col_start * lldA, lldA
            //     ));
            // the second method is to use the batched gemm which will not pollute the unused block
            if(length_row <= 0 || length_col <= 0)
                continue;
            num_col_block = length_col / nb;
            num_row_block = length_row / nb;
            if(length_col % nb !=0)
                num_col_block++;
            if(length_row % nb !=0)
                num_row_block++;
            
            i_batch_count = 0;
            
            col_s = nb;
            row_remain = length_row % nb;
            col_remain = length_col % nb;
            row_s = nb + row_remain;
            if(row_remain != 0){
                int g_row_s = array_descA.indx_l2g_r(array_descA.m_loc() - row_remain);
                int g_col_s;
                int length_col_real =  length_col;
                do{
                    length_col_real -= nb;
                    g_col_s = array_descA.indx_l2g_c(mm_col_start + length_col_real);
                }while(g_row_s < g_col_s);
                length_col_real += nb;
                if(length_col_real > 0)
                    BLAS_CHECK(deblasGemm(
                        blasH, DEBLAS_OP_N, DEBLAS_OP_C,
                        row_remain, length_col_real, nb_real, (T)-1.0,
                        d_block_col + length_row - row_remain, length_row,
                        d_block_row, length_col,
                        (T)1.0, A + mm_row_start + mm_col_start * lldA + (length_row - row_remain), lldA
                    ));
            }
            if(col_remain != 0){
                int g_col_s = array_descA.indx_l2g_c(array_descA.n_loc() - col_remain);
                int g_row_s;
                int length_row_real = length_row + nb;
                do{
                    length_row_real -= nb;
                    g_row_s = array_descA.indx_l2g_r(mm_row_start + length_row - length_row_real);
                }while(g_row_s < g_col_s);
                if(length_row_real > 0)
                    BLAS_CHECK(deblasGemm(
                        blasH, DEBLAS_OP_N, DEBLAS_OP_C,
                        length_row_real, col_remain, nb, (T)-1.0,
                        d_block_col + length_row - length_row_real, length_row,
                        d_block_row + length_col - col_remain, length_col,
                        (T)1.0, A + mm_row_start + mm_col_start * lldA + (length_row - length_row_real) + (length_col - col_remain) * lldA, lldA
                    ));
            }
            // printf("1-myid:%d, length_row:%d, length_col:%d, i_batch_count:%d\n", ddla_handle->myid, length_row, length_col, i_batch_count);
            for(;row_s <= num_row_block * nb; row_s += nb){
                int g_row_s = array_descA.indx_l2g_r(array_descA.m_loc() - row_s);
                int g_col_s;
                col_s = col_remain;
                do{
                    col_s += nb;
                    g_col_s = array_descA.indx_l2g_c(array_descA.n_loc() - col_s);
                }while(g_row_s < g_col_s);
                // printf("myid:%d, col_s:%d\n", ddla_handle->myid, col_s);
                for(; col_s <= num_col_block * nb; col_s += nb){
                    // printf("myid:%d, before h_A\n", ddla_handle->myid);
                    h_A_array[i_batch_count] = d_block_col + length_row - row_s;
                    // printf("myid:%d, before h_B\n", ddla_handle->myid);
                    h_B_array[i_batch_count] = d_block_row + length_col - col_s;
                    // printf("myid:%d, before h_C\n", ddla_handle->myid);
                    h_C_array[i_batch_count] = A + array_descA.m_loc() - row_s + (array_descA.n_loc() - col_s) * lldA;
                    i_batch_count++;
                }
            }
            // printf("2-myid:%d, length_row:%d, length_col:%d, i_batch_count:%d\n", ddla_handle->myid, length_row, length_col, i_batch_count);
            if(i_batch_count == 0) continue;
            DEVICE_CHECK(deviceMemcpyAsync(d_A_array, h_A_array.data(), i_batch_count * sizeof(T*), deviceMemcpyHostToDevice, stream));
            DEVICE_CHECK(deviceMemcpyAsync(d_B_array, h_B_array.data(), i_batch_count * sizeof(T*), deviceMemcpyHostToDevice, stream));
            DEVICE_CHECK(deviceMemcpyAsync(d_C_array, h_C_array.data(), i_batch_count * sizeof(T*), deviceMemcpyHostToDevice, stream));
            BLAS_CHECK(deblasGemmBatched(
                blasH, DEBLAS_OP_N, DEBLAS_OP_C,
                nb, nb, nb_real, -1.0,
                d_A_array, length_row,
                d_B_array, length_col,
                1.0, d_C_array, lldA,
                i_batch_count
            ));

            
            DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
        }
        DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    }
    // printf("myid:%d, end\n", ddla_handle->myid);
    DEVICE_CHECK(deviceFreeAsync(d_A_array, stream));
    DEVICE_CHECK(deviceFreeAsync(d_B_array, stream));
    DEVICE_CHECK(deviceFreeAsync(d_C_array, stream));
    DEVICE_CHECK(deviceFreeAsync(d_block_diag, stream));
    DEVICE_CHECK(deviceFreeAsync(d_block_row, stream));
    DEVICE_CHECK(deviceFreeAsync(d_block_col, stream));
    DEVICE_CHECK(deviceFreeAsync(d_info, stream));
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    return is_nega;

}

template bool ppotrf<std::complex<float>>(
    const char& uplo, const int& n,
    std::complex<float>* A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    int& info, // host pointer
    bool is_head, int location
);

template bool ppotrf<std::complex<double>>(
    const char& uplo, const int& n,
    std::complex<double>* A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    int& info, // host pointer
    bool is_head, int location
);


}