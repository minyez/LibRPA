#include <cassert>
#include <cmath>
#include <mpi.h>
#include <time.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <complex>
#include <string>
#include <ddla.h>
#include <ddla_connector.h>
#include <random>
#include <ddla_stream.h>
using namespace DDLA;

#include <ddla.h>
#include <cassert>
#include <ddla_connector.h>
#include <ddla_utils.h>
#include <ddla_stream.h>
#include <vector>
namespace DDLA{

}
void check_pzgetrf(int n, const DdlaHandle_t& ddla_handle)
{

    DDLA::DdlaDesc matrix_desc(ddla_handle);
    matrix_desc.init_square_blk(n, n, 0, 0);
    int nb = std::min(10, matrix_desc.mb());
    matrix_desc.init(n, n, nb, nb, 0, 0);

    int myid = matrix_desc.mypcol() + matrix_desc.myprow()*matrix_desc.npcols();
    printf("myid:%d, m_loc:%d, n_loc:%d, mb:%d, nb:%d, m:%d, n:%d\n", myid, matrix_desc.m_loc(), matrix_desc.n_loc(), matrix_desc.mb(), matrix_desc.nb(), matrix_desc.m(), matrix_desc.n());
    bool verbose = true;
    
    std::vector<std::complex<double>> a(matrix_desc.m_loc()*matrix_desc.n_loc());

    std::complex<double>* d_A,* d_A_copy,* d_identity;
    std::vector<std::complex<double>> h_identity(matrix_desc.m_loc()*matrix_desc.n_loc());
    memset(h_identity.data(),0,sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc());
    for(int i=0;i<matrix_desc.m();i++){
        int i_loc = matrix_desc.indx_g2l_r(i);
        if(i_loc<0) continue;
        int j_loc = matrix_desc.indx_g2l_c(i);
        if(j_loc<0) continue;
        h_identity[i_loc+j_loc*matrix_desc.lld()] = {1.0, 0.0};
    }
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    ddla_handle->check_memory();
    MPI_Barrier(MPI_COMM_WORLD);

    const size_t size = matrix_desc.m_loc()*matrix_desc.n_loc()*sizeof(std::complex<double>);

    DEVICE_CHECK(deviceMallocAsync((void**)&d_A, size, ddla_handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_A_copy, size, ddla_handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_identity, size, ddla_handle->stream));
    
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    ddla_handle->check_memory();
    MPI_Barrier(MPI_COMM_WORLD);
    DDLA::random_generator(d_A, matrix_desc.m_loc()*matrix_desc.n_loc(),DEVICE_C_64F);
    
    DEVICE_CHECK(deviceMemcpyAsync(d_A_copy, d_A, sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc(), deviceMemcpyDeviceToDevice, ddla_handle->stream));
    DEVICE_CHECK(deviceMemcpyAsync(a.data(), d_A, matrix_desc.m_loc() * matrix_desc.n_loc()* sizeof(std::complex<double>), deviceMemcpyDeviceToHost, ddla_handle->stream));
    DEVICE_CHECK(deviceMemcpyAsync(d_identity, h_identity.data(), sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc(), deviceMemcpyHostToDevice, ddla_handle->stream));
    
    std::vector<int> ipiv(n);

    if(verbose)
    {
        std::string filename = "before_gemm_myid_";
        filename += std::to_string(myid);
        filename += ".txt";
        DDLA::write_matrix(a.data(), matrix_desc.m_loc(), matrix_desc.n_loc(), filename.c_str());
    }
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);
    printf("myid:%d, start gemm:\n",myid);
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time_gemm = MPI_Wtime();
    pgemm(
        'N', 'C',
        n, n, n,
        {1.0,0.0},
        d_identity, matrix_desc,
        d_A_copy, matrix_desc,
        {0.0,0.0},
        d_A, matrix_desc
    );
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    double end_time_gemm = MPI_Wtime();
    printf("myid:%d, pzgemm time:%lf\n",myid,end_time_gemm-start_time_gemm);
    if(verbose)
    { 
        DEVICE_CHECK(deviceMemcpyAsync(a.data(), d_A, sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc(), deviceMemcpyDeviceToHost, ddla_handle->stream));
        DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
        std::string filename = "after_gemm_myid_";
        filename += std::to_string(myid);
        filename += ".txt";
        DDLA::write_matrix(a.data(), matrix_desc.m_loc(), matrix_desc.n_loc(), filename.c_str());
    }
    DEVICE_CHECK(deviceFreeAsync(d_identity, ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_A, ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_A_copy, ddla_handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
}
int main(int argc, char* argv[]) {  
    MPI_Init(&argc, &argv);
    printf("before stream init\n");
    DdlaHandle_t ddla_handle;
    ddla_init(ddla_handle);
    ddla_set(ddla_handle);

    printf("after stream init\n");
    check_pzgetrf(100, ddla_handle);
    // for(int i=10000;i<=20000;i+=10000){
    //     DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     printf("testing matrix size: %d\n",i);
    //     check_pzgetrf(i,ddla_handle);
    // }
    ddla_destroy(ddla_handle);
    MPI_Finalize();
    return 0;
}