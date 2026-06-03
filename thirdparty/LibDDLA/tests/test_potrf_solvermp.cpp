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
#include <cusolverMp.h>
#include "helpers.h"
#include "potrf.h"
using namespace DDLA;

void check_ppotrf(int n, const DdlaHandle_t& ddla_handle)
{

    DDLA::DdlaDesc matrix_desc(ddla_handle);
    matrix_desc.init_square_blk(n, n, 0, 0);
    int nb = std::min(128, matrix_desc.mb());
    matrix_desc.init(n, n, nb, nb, 0, 0);

    printf("myid:%d, m_loc:%d, n_loc:%d, mb:%d, nb:%d, m:%d, n:%d\n", ddla_handle->myid, matrix_desc.m_loc(), matrix_desc.n_loc(), matrix_desc.mb(), matrix_desc.nb(), matrix_desc.m(), matrix_desc.n());
    bool verbose = false;

    std::complex<double>* d_A;
    std::complex<double>* d_A_copy;
    
    MPI_Barrier(MPI_COMM_WORLD);
    ddla_handle->check_memory();
    MPI_Barrier(MPI_COMM_WORLD);

    const size_t size = matrix_desc.m_loc()*matrix_desc.n_loc()*sizeof(std::complex<double>);
    DEVICE_CHECK(deviceMallocAsync((void**)&d_A, size, ddla_handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_A_copy, size, ddla_handle->stream));
    
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    ddla_handle->check_memory();
    MPI_Barrier(MPI_COMM_WORLD);
    DDLA::random_generator(d_A, matrix_desc.m_loc()*matrix_desc.n_loc(),DEVICE_C_64F);
    std::complex<double> ten = 1000.0;
    for(int i=0;i<matrix_desc.m();i++){
        int i_loc = matrix_desc.indx_g2l_r(i);
        if(i_loc<0) continue;
        int j_loc = matrix_desc.indx_g2l_c(i);
        if(j_loc<0) continue;
        DEVICE_CHECK(deviceMemcpyAsync(d_A+i_loc+j_loc*matrix_desc.lld(), &ten, sizeof(std::complex<double>), deviceMemcpyHostToDevice, ddla_handle->stream));
    }
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time_geadd = MPI_Wtime();
    pgeadd(
        'C', 'N',
        n, n,
        {1.0, 0.0},
        d_A, matrix_desc,
        {1.0, 0.0},
        d_A, matrix_desc,
        d_A_copy, matrix_desc
    );
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    printf("myid:%d, pgeadd time:%lf\n", ddla_handle->myid, MPI_Wtime() - start_time_geadd);
    DEVICE_CHECK(deviceMemcpyAsync(d_A, d_A_copy, size, deviceMemcpyDeviceToDevice, ddla_handle->stream));
    std::complex<double>* h_A = NULL;
    if (ddla_handle->myid == 0)
    {
        h_A = (std::complex<double>*)malloc(matrix_desc.m() * matrix_desc.n() * sizeof(std::complex<double>));
        memset(h_A, 0xFF, matrix_desc.m() * matrix_desc.n() * sizeof(std::complex<double>));
        generate_diagonal_dominant_symmetric_matrix(matrix_desc.m(), h_A, matrix_desc.n());
    }

    if(verbose)
    {
        std::vector<std::complex<double>> a(matrix_desc.m_loc()*matrix_desc.n_loc());
        DEVICE_CHECK(deviceMemcpyAsync(a.data(), d_A, sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc(), deviceMemcpyDeviceToHost, ddla_handle->stream));
        DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
        std::string filename = "before_potrf_myid_";
        filename += std::to_string(ddla_handle->myid);
        filename += ".txt";
        DDLA::write_matrix(a.data(), matrix_desc.m_loc(), matrix_desc.n_loc(), filename.c_str());
    }
    
    // DEVICE_CHECK(deviceMemcpyAsync(d_A_copy, d_A, size, deviceMemcpyDeviceToDevice, ddla_handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    
    int * d_info;
    DEVICE_CHECK(deviceMallocAsync((void**)&d_info, sizeof(int), ddla_handle->stream));
    
    if(verbose)
    { 
        std::vector<std::complex<double>> a(matrix_desc.m_loc()*matrix_desc.n_loc());
        DEVICE_CHECK(deviceMemcpyAsync(a.data(), d_A_copy, sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc(), deviceMemcpyDeviceToHost, ddla_handle->stream));
        DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
        std::string filename = "ppotrf_myid_";
        filename += std::to_string(ddla_handle->myid);
        filename += ".txt";
        DDLA::write_matrix(a.data(), matrix_desc.m_loc(), matrix_desc.n_loc(), filename.c_str());
    }
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    {
        cusolverMpHandle_t cusolverMpHandle = NULL;
        cal_comm_t         cal_comm = NULL;

        cusolverStatus_t cusolverStat;
        calError_t       calStat;

        cal_comm_create_params_t params;
        params.allgather    = allgather;
        params.req_test     = request_test;
        params.req_free     = request_free;
        params.data         = (void*)(MPI_COMM_WORLD);
        params.rank         = ddla_handle->myid;
        params.nranks       = ddla_handle->nprocs;
        params.local_device = DdlaStream::local_device;

        calStat = cal_comm_create(params, &cal_comm);
        assert(calStat == CAL_OK);

        SOLVER_CHECK(cusolverMpCreate(&cusolverMpHandle, DdlaStream::local_device, ddla_handle->stream));

        cusolverMpGrid_t grid = NULL;
        cusolverMpMatrixDescriptor_t descr = NULL;
        void* d_work = NULL;
        void* h_work = NULL;
        size_t workspaceInBytesOnDevice = 0;
        size_t workspaceInBytesOnHost = 0;

        cusolverStat = cusolverMpCreateDeviceGrid(
            cusolverMpHandle, &grid, cal_comm, matrix_desc.nprows(), matrix_desc.npcols(), CUSOLVERMP_GRID_MAPPING_ROW_MAJOR);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* Create matrix descriptor */
        cusolverStat = cusolverMpCreateMatrixDesc(
            &descr, grid, CUDA_C_64F, matrix_desc.m(), matrix_desc.n(), matrix_desc.mb(), matrix_desc.nb(), matrix_desc.irsrc(), matrix_desc.icsrc(), matrix_desc.lld());
        printf("myid:%d,m:%d,n:%d,mb:%d,nb:%d,irsrc:%d,icsrc:%d,lld:%d\n",ddla_handle->myid,matrix_desc.m(),matrix_desc.n(),matrix_desc.mb(),matrix_desc.nb(),matrix_desc.irsrc(),matrix_desc.icsrc(),matrix_desc.lld());
        
        // SOLVER_CHECK(cusolverMpMatrixScatterH2D(
        //     cusolverMpHandle, matrix_desc.m(), matrix_desc.n(), 
        //     (void*)d_A_copy, 1, 1, descr,
        //     0, (void*)h_A, matrix_desc.m()
        // ));
        cusolverStat = cusolverMpPotrf_bufferSize(
            cusolverMpHandle, DEBLAS_FILL_MODE_LOWER, matrix_desc.m(),
            d_A_copy, 1, 1, descr, CUDA_C_64F,
            &workspaceInBytesOnDevice, &workspaceInBytesOnHost
        );
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        printf("getrf rank:%d, workspaceInBytesOnDevice:%lf GiB, workspaceInBytesOnHost:%lf GiB\n",ddla_handle->myid,workspaceInBytesOnDevice/(1024.*1024.*1024.),workspaceInBytesOnHost/(1024.*1024.*1024.));

        if(workspaceInBytesOnDevice > 0)
            DEVICE_CHECK(cudaMalloc((void**)&d_work, workspaceInBytesOnDevice));
        
        h_work = (void*)malloc(workspaceInBytesOnHost);
        assert(h_work != NULL);

        calStat = cal_stream_sync(cal_comm, ddla_handle->stream);
        assert(calStat == CAL_OK);
        DEVICE_CHECK(deviceMemcpyAsync(d_A, d_A_copy, matrix_desc.m_loc() * matrix_desc.n_loc() * sizeof(std::complex<double>), deviceMemcpyDeviceToDevice, ddla_handle->stream));
        DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time=MPI_Wtime();
        cusolverStat = cusolverMpPotrf(
            cusolverMpHandle, DEBLAS_FILL_MODE_LOWER, matrix_desc.m(),
            d_A_copy, 1, 1, descr, CUDA_C_64F,
            d_work, workspaceInBytesOnDevice,
            h_work, workspaceInBytesOnHost,
            d_info
        );
        // SOLVER_CHECK(desolverPotrf(
        //     ddla_handle->solverH, DEBLAS_FILL_MODE_LOWER, matrix_desc.m(),
        //     d_A_copy, matrix_desc.lld(), d_info
        // ));
        if(cusolverStat != CUSOLVER_STATUS_SUCCESS){
            printf("myid:%d, cusolverStat:%d\n",ddla_handle->myid,cusolverStat);
        }
        DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
        printf("myid:%d, cusolverMpPotrf time:%lf\n", ddla_handle->myid, MPI_Wtime()-start_time);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        MPI_Barrier(MPI_COMM_WORLD);
        printf("myid:%d, start ppotrf\n",ddla_handle->myid);
        double start_time_ppotrf = MPI_Wtime();
        ppotrf(
            'L', n,
            d_A, 1, 1, matrix_desc,
            d_info
        );
        printf("1234\n");
        DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
        printf("myid:%d, ppotrf time:%lf\n",ddla_handle->myid,MPI_Wtime()-start_time_ppotrf);
        MPI_Barrier(MPI_COMM_WORLD);

        SOLVER_CHECK(cusolverMpDestroy(cusolverMpHandle));
        SOLVER_CHECK(cusolverMpDestroyMatrixDesc(descr));
        SOLVER_CHECK(cusolverMpDestroyGrid(grid));

        calStat = cal_comm_barrier(cal_comm, ddla_handle->stream);
        assert(calStat == CAL_OK);

        /* destroy CAL communicator */
        calStat = cal_comm_destroy(cal_comm);
        assert(calStat == CAL_OK);

        if (d_work != NULL)
        {
            DEVICE_CHECK(deviceFreeAsync(d_work, ddla_handle->stream));
            d_work = NULL;
        }
        if (h_work != NULL) 
        {
            free(h_work);
            h_work = NULL;
        } 
    }
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    if(verbose)
    {
        std::vector<std::complex<double>> a(matrix_desc.m_loc()*matrix_desc.n_loc());
        DEVICE_CHECK(deviceMemcpyAsync(a.data(), d_A_copy, sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc(), deviceMemcpyDeviceToHost, ddla_handle->stream));
        DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
        std::string filename = "potrf_myid_";
        filename += std::to_string(ddla_handle->myid);
        filename += ".txt";
        DDLA::write_matrix(a.data(), matrix_desc.m_loc(), matrix_desc.n_loc(), filename.c_str());
    }
    {
        printf("myid:%d, start check potrf\n", ddla_handle->myid);
        std::vector<std::complex<double>> a(matrix_desc.m_loc()*matrix_desc.n_loc());
        std::vector<std::complex<double>> b(matrix_desc.m_loc()*matrix_desc.n_loc());
        DEVICE_CHECK(deviceMemcpyAsync(a.data(), d_A, sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc(), deviceMemcpyDeviceToHost, ddla_handle->stream));
        DEVICE_CHECK(deviceMemcpyAsync(b.data(), d_A_copy, sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc(), deviceMemcpyDeviceToHost, ddla_handle->stream));
        DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
        for(int i=0;i<matrix_desc.m();i++){
            int i_loc = matrix_desc.indx_g2l_r(i);
            if(i_loc < 0) continue;
            for(int j = 0; j < matrix_desc.n(); j++){
                int j_loc = matrix_desc.indx_g2l_c(j);
                if(j_loc < 0) continue;
                auto diff = a[i_loc + j_loc * matrix_desc.lld()]-b[i_loc + j_loc * matrix_desc.lld()];
                if(std::abs(diff)>1e-10){
                    printf("myid:%d, i:%d, j:%d, diff:(%lf,%lf), b:%lf\n",ddla_handle->myid,i,j,diff.real(),diff.imag(),1e-6);
                    break;
                }
            }
        }
        printf("myid:%d, check potrf pass\n", ddla_handle->myid);
    }
    if(h_A){
        free(h_A);
        h_A = nullptr;
    }
    DEVICE_CHECK(deviceFreeAsync(d_A_copy, ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_A, ddla_handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_info, ddla_handle->stream));

    
}
int main(int argc, char* argv[]) {  
    MPI_Init(&argc, &argv);
    printf("before stream init\n");
    DdlaHandle_t ddla_handle = nullptr;
    ddla_init(ddla_handle);
    ddla_set(ddla_handle, MPI_COMM_WORLD, 'R');
    // DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    printf("after stream init\n");
    check_ppotrf(5000, ddla_handle);
    for(int i = 1000; i <= 1000 + 10 * 100; i += 100){
        DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
        MPI_Barrier(MPI_COMM_WORLD);
        printf("testing matrix size: %d\n",i);
        check_ppotrf(i,ddla_handle);
    }
    ddla_destroy(ddla_handle);
    MPI_Finalize();
    return 0;
}