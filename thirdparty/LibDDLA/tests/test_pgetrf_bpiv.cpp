#include <cassert>
#include <cmath>
#include <mpi.h>
#include <time.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <complex>
#include <string>
#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/scal.h>
#include <random>
#include <ddla/ddla_stream.h>

using namespace ddla;


void check_pzgetrf(int n, const DdlaHandle_t& ddla_handle)
{

    DdlaDesc matrix_desc(ddla_handle);
    matrix_desc.init_square_blk(n, n, 0, 0);
    int nb = std::min(128, matrix_desc.mb());
    matrix_desc.init(n, n, nb, nb, 0, 0);

    int myid = matrix_desc.mypcol() + matrix_desc.myprow()*matrix_desc.npcols();
    printf("myid:%d, m_loc:%d, n_loc:%d, mb:%d, nb:%d, m:%d, n:%d\n", myid, matrix_desc.m_loc(), matrix_desc.n_loc(), matrix_desc.mb(), matrix_desc.nb(), matrix_desc.m(), matrix_desc.n());
    bool verbose = false;

    std::complex<double> *d_A, *d_A_copy;
    int *d_ipiv;
    
    std::vector<int>ipiv(matrix_desc.m_loc());
    ddla_handle->check_memory();
    MPI_Barrier(MPI_COMM_WORLD);

    const size_t size = matrix_desc.m_loc()*matrix_desc.n_loc()*sizeof(std::complex<double>);

    DEVICE_CHECK(deviceMallocAsync((void**)&d_A, size, ddla_handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_A_copy, size, ddla_handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_ipiv, matrix_desc.m_loc() * sizeof(int), ddla_handle->stream));
    
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    ddla_handle->check_memory();
    MPI_Barrier(MPI_COMM_WORLD);
    random_generator(d_A, matrix_desc.m_loc()*matrix_desc.n_loc(),DEVICE_C_64F);
    BLAS_CHECK(deblasScal(ddla_handle->blasH, matrix_desc.m_loc()*matrix_desc.n_loc(), 0.01, d_A, 1));
    std::complex<double> cons_i = 2.0;
    for(int i = 0; i < matrix_desc.m(); i++){
        int i_loc = matrix_desc.indx_g2l_r(i);
        if(i_loc < 0 ) continue;
        int j_loc = matrix_desc.indx_g2l_c(i);
        if(j_loc < 0 ) continue;
        DEVICE_CHECK(deviceMemcpy(d_A + i_loc + j_loc * matrix_desc.lld(), &cons_i, sizeof(std::complex<double>), deviceMemcpyHostToDevice));
    }
    
    DEVICE_CHECK(deviceMemcpyAsync(d_A_copy, d_A, sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc(), deviceMemcpyDeviceToDevice, ddla_handle->stream));

    if(verbose)
    {
        std::vector<std::complex<double>> a(matrix_desc.m_loc()*matrix_desc.n_loc());
        DEVICE_CHECK(deviceMemcpyAsync(a.data(), d_A, sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc(), deviceMemcpyDeviceToHost, ddla_handle->stream));
        DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
        std::string filename = "before_trf_myid_";
        filename += std::to_string(myid);
        filename += ".txt";
        write_matrix(a.data(), matrix_desc.m_loc(), matrix_desc.n_loc(), filename.c_str());
    }
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);
    printf("myid:%d, start sv\n",myid);
    int info = -1;
    double start_time_trf = MPI_Wtime();
    pgetrf(n, n, d_A, matrix_desc, ipiv.data(), info);
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time_trf = MPI_Wtime();
    assert(info == 0);
    printf("myid:%d, start bpiv\n", myid);
    info = -1;
    pgetrf_bpiv(n, n, d_A_copy, matrix_desc, d_ipiv, info);
    printf("myid:%d, piv of lu time:%lf, bpiv of lu time:%lf\n", myid, end_time_trf - start_time_trf, MPI_Wtime() - end_time_trf);
    assert(info == 0);
    if(verbose)
    {
        std::vector<std::complex<double>> a(matrix_desc.m_loc()*matrix_desc.n_loc());
        DEVICE_CHECK(deviceMemcpyAsync(a.data(), d_A, sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc(), deviceMemcpyDeviceToHost, ddla_handle->stream));
        DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
        std::string filename = "identity_myid_";
        filename += std::to_string(myid);
        filename += ".txt";
        write_matrix(a.data(), matrix_desc.m_loc(), matrix_desc.n_loc(), filename.c_str());
    }
    {
        std::vector<std::complex<double>> a(matrix_desc.m_loc()*matrix_desc.n_loc());
        std::vector<std::complex<double>> b(matrix_desc.m_loc()*matrix_desc.n_loc());
        DEVICE_CHECK(deviceMemcpyAsync(a.data(), d_A, sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc(), deviceMemcpyDeviceToHost, ddla_handle->stream));
        DEVICE_CHECK(deviceMemcpyAsync(b.data(), d_A_copy, sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc(), deviceMemcpyDeviceToHost, ddla_handle->stream));
        printf("myid:%d, start check trf result\n", myid);
        int loc_pi;
        std::complex<double> tmp;
        std::complex<double> ln_det_loc_a = 0.0;
        std::complex<double> ln_det_all_a = 0.0;
        std::complex<double> ln_det_loc_b = 0.0;
        std::complex<double> ln_det_all_b = 0.0;
        for (int ig = 0; ig != n; ig++)
        {
            int locr = matrix_desc.indx_g2l_r(ig);
            int locc = matrix_desc.indx_g2l_c(ig);
            if (locr >= 0 && locc >= 0)
            {
                tmp = a[locr + locc * matrix_desc.lld()];
                ln_det_loc_a += tmp.real() > 0 ? std::log(tmp) : std::log(-tmp);
                tmp = b[locr + locc * matrix_desc.lld()];
                ln_det_loc_b += tmp.real() > 0 ? std::log(tmp) : std::log(-tmp);
            }
        }
        MPI_Allreduce(&ln_det_loc_a,&ln_det_all_a,1,MPI_DOUBLE_COMPLEX,MPI_SUM, ddla_handle->comm);
        MPI_Allreduce(&ln_det_loc_b,&ln_det_all_b,1,MPI_DOUBLE_COMPLEX,MPI_SUM, ddla_handle->comm);
        MPI_Barrier(ddla_handle->comm);
        printf("myid:%d, ln_det_a:%lf+i%lf, ln_det_b:%lf+i%lf\n", myid, ln_det_all_a.real(), ln_det_all_a.imag(), ln_det_all_b.real(), ln_det_all_b.imag());
    }
    DEVICE_CHECK(deviceFreeAsync(d_ipiv, ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_A, ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_A_copy, ddla_handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
}
int main(int argc, char* argv[]) {  
    MPI_Init(&argc, &argv);
    printf("before stream init\n");
    DdlaHandle_t ddla_handle = nullptr;
    ddla_init(ddla_handle);
    ddla_set(ddla_handle);
    // DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    printf("after stream init\n");
    check_pzgetrf(5000, ddla_handle);
    for(int i = 5000; i <= 4 * 5000; i += 5000){
        DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
        MPI_Barrier(MPI_COMM_WORLD);
        printf("testing matrix size: %d\n",i);
        check_pzgetrf(i, ddla_handle);
    }
    ddla_destroy(ddla_handle);
    MPI_Finalize();
    return 0;
}