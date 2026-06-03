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


void check_ppotrf(int n, const DdlaHandle_t& ddla_handle)
{

    DDLA::DdlaDesc matrix_desc(ddla_handle);
    matrix_desc.init_square_blk(n, n, 0, 0);
    int nb = std::min(128, matrix_desc.mb());
    matrix_desc.init(n, n, nb, nb, 0, 0);

    int myid = matrix_desc.mypcol() + matrix_desc.myprow()*matrix_desc.npcols();
    printf("myid:%d, m_loc:%d, n_loc:%d, mb:%d, nb:%d, m:%d, n:%d\n", myid, matrix_desc.m_loc(), matrix_desc.n_loc(), matrix_desc.mb(), matrix_desc.nb(), matrix_desc.m(), matrix_desc.n());
    bool verbose = false;

    std::complex<double>* d_A;

    MPI_Barrier(MPI_COMM_WORLD);
    ddla_handle->check_memory();
    MPI_Barrier(MPI_COMM_WORLD);

    const size_t size = matrix_desc.m_loc()*matrix_desc.n_loc()*sizeof(std::complex<double>);
    DEVICE_CHECK(deviceMallocAsync((void**)&d_A, size, ddla_handle->stream));
    
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    ddla_handle->check_memory();
    MPI_Barrier(MPI_COMM_WORLD);
    DDLA::random_generator(d_A, matrix_desc.m_loc()*matrix_desc.n_loc(),DEVICE_C_64F);
    
    std::complex<double> ten = 10000.0;
    std::complex<double> one = -1.0;
    for(int i=0;i<matrix_desc.m();i++){
        int i_loc = matrix_desc.indx_g2l_r(i);
        if(i_loc<0) continue;
        int j_loc = matrix_desc.indx_g2l_c(i);
        if(j_loc<0) continue;
        DEVICE_CHECK(deviceMemcpyAsync(d_A+i_loc+j_loc*matrix_desc.lld(), &ten, sizeof(std::complex<double>), deviceMemcpyHostToDevice, ddla_handle->stream));
        if(i == n - 1)
        DEVICE_CHECK(deviceMemcpyAsync(d_A+i_loc+j_loc*matrix_desc.lld(), &one, sizeof(std::complex<double>), deviceMemcpyHostToDevice, ddla_handle->stream));
    }
    // if(verbose)
    // {
    //     std::vector<std::complex<double>> a(matrix_desc.m_loc()*matrix_desc.n_loc());
    //     DEVICE_CHECK(deviceMemcpyAsync(a.data(), d_A, sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc(), deviceMemcpyDeviceToHost, ddla_handle->stream));
    //     DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    //     std::string filename = "before_potrf_myid_";
    //     filename += std::to_string(myid);
    //     filename += ".txt";
    //     DDLA::write_matrix(a.data(), matrix_desc.m_loc(), matrix_desc.n_loc(), filename.c_str());
    // }
    std::complex<double>* d_A_copy;
    DEVICE_CHECK(deviceMallocAsync(&d_A_copy, size, ddla_handle->stream));
    DEVICE_CHECK(deviceMemcpyAsync(d_A_copy, d_A, size, deviceMemcpyDeviceToDevice, ddla_handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);
    int info;
    printf("myid:%d, start ppotrf\n",myid);
    double start_time_sv = MPI_Wtime();
    ppotrf(
        'L', n,
        d_A, 1, 1, matrix_desc,
        info
    );
    printf("info = %d\n", info);
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    printf("myid:%d, ppotrf time:%lf\n",myid, MPI_Wtime()-start_time_sv);
    MPI_Barrier(MPI_COMM_WORLD);
    if(verbose)
    { 
        std::vector<std::complex<double>> a(matrix_desc.m_loc()*matrix_desc.n_loc());
        DEVICE_CHECK(deviceMemcpyAsync(a.data(), d_A, sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc(), deviceMemcpyDeviceToHost, ddla_handle->stream));
        DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
        std::string filename = "ppotrf_myid_";
        filename += std::to_string(myid);
        filename += ".txt";
        DDLA::write_matrix(a.data(), matrix_desc.m_loc(), matrix_desc.n_loc(), filename.c_str());
    }
    
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    double start_time_potrf = MPI_Wtime();
    bool is_nega = ppotrf(
        'L', n,
        d_A_copy, 1, 1, matrix_desc,
        info,
        true, -1
    );
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    printf("myid:%d, ppotrf_head time:%lf\n",myid, MPI_Wtime() - start_time_potrf);
    printf("is_nega: %d\n", is_nega);
    if(verbose)
    { 
        std::vector<std::complex<double>> a(matrix_desc.m_loc()*matrix_desc.n_loc());
        DEVICE_CHECK(deviceMemcpyAsync(a.data(), d_A_copy, sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc(), deviceMemcpyDeviceToHost, ddla_handle->stream));
        DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
        std::string filename = "ppotrf_head_myid_";
        filename += std::to_string(myid);
        filename += ".txt";
        DDLA::write_matrix(a.data(), matrix_desc.m_loc(), matrix_desc.n_loc(), filename.c_str());
    }
    {
        printf("myid:%d, start check potrf\n");
        std::vector<std::complex<double>> a(matrix_desc.m_loc()*matrix_desc.n_loc());
        std::vector<std::complex<double>> b(matrix_desc.m_loc()*matrix_desc.n_loc());
        DEVICE_CHECK(deviceMemcpyAsync(a.data(), d_A, sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc(), deviceMemcpyDeviceToHost, ddla_handle->stream));
        DEVICE_CHECK(deviceMemcpyAsync(b.data(), d_A_copy, sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc(), deviceMemcpyDeviceToHost, ddla_handle->stream));
        DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
        for(int i=0;i<matrix_desc.m();i++){
            int i_loc = matrix_desc.indx_g2l_r(i);
            if(i_loc < 0)
                continue;
            for(int j=0;j<matrix_desc.n();j++){
                int j_loc = matrix_desc.indx_g2l_c(j);
                if(j_loc < 0)
                    continue;
                int offset = i_loc + j_loc * matrix_desc.lld();
                auto diff = a[offset]-b[offset];
                if(std::abs(diff)>1e-6){
                    printf("myid:%d, i:%d, j:%d, a:(%lf,%lf), b:(%lf,%lf)\n", myid, i, j, a[offset].real(), a[offset].imag(), b[offset].real(), b[offset].imag());
                }
            }
        }
        printf("myid:%d, check potrf pass\n");
    }
    DEVICE_CHECK(deviceFreeAsync(d_A_copy, ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_A, ddla_handle->stream));
}
int main(int argc, char* argv[]) {  
    MPI_Init(&argc, &argv);
    printf("before stream init\n");
    DdlaHandle_t ddla_handle = nullptr;
    ddla_init(ddla_handle);
    ddla_set(ddla_handle);
    // DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    printf("after stream init\n");
    check_ppotrf(5000, ddla_handle);
    for(int i=5000;i<=20000;i+=5000){
        DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
        MPI_Barrier(MPI_COMM_WORLD);
        printf("testing matrix size: %d\n",i);
        check_ppotrf(i,ddla_handle);
    }
    ddla_destroy(ddla_handle);
    MPI_Finalize();
    return 0;
}