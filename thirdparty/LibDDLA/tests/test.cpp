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


void check_mpi_aware(const DdlaHandle_t& ddla_handle)
{
    int n = 10;
    int myid = ddla_handle->myid;
    std::vector<int> a(n);
    memset(a.data(), 0, n*sizeof(int));
    int* d_A;
    if(myid == 0)
    for(int i=0;i<n;i++){
        a[i] = i;
    }
    ddla_handle->check_memory();
    MPI_Barrier(ddla_handle->comm);

    DEVICE_CHECK(deviceMallocAsync((void**)&d_A, n * sizeof(int), ddla_handle->stream));
    
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    ddla_handle->check_memory();
    MPI_Barrier(ddla_handle->comm);
    
    DEVICE_CHECK(deviceMemcpyAsync(d_A, a.data(), n * sizeof(int) , deviceMemcpyHostToDevice, ddla_handle->stream));

    
    {
        printf("before myid:%d,a[n]: ",myid);
        for(int i=0;i<n;i++)
            printf("%d ",a[i]);
        printf("\n");
    }
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(ddla_handle->comm);
    printf("myid:%d, start Bcast\n",myid);
    double start_time = MPI_Wtime();
    MPI_Bcast(d_A, n, MPI_INT, 0, ddla_handle->comm);
    printf("myid:%d, Bcast time:%lf\n",myid,MPI_Wtime()-start_time);
    DEVICE_CHECK(deviceMemcpyAsync(a.data(), d_A, n * sizeof(int) , deviceMemcpyDeviceToHost, ddla_handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(ddla_handle->comm);
    if(myid != 0)
    {
        printf("after myid:%d,a[n]: ",myid);
        for(int i=0;i<n;i++)
            printf("%d ",a[i]);
        printf("\n");
    }
    DEVICE_CHECK(deviceFreeAsync(d_A, ddla_handle->stream));
}
int main(int argc, char* argv[]) {  
    MPI_Init(&argc, &argv);
    printf("before stream init\n");
    DdlaHandle_t ddla_handle;
    ddla_init(ddla_handle);
    ddla_set(ddla_handle);
    printf("after stream init\n");
    check_mpi_aware(ddla_handle);
    ddla_destroy(ddla_handle);
    MPI_Finalize();
    return 0;
}