#include <cassert>
#include <cmath>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <complex>
#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <ddla/ptran.h>

using namespace ddla;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    DdlaHandle_t handle;
    ddla_init(handle);
    ddla_set(handle, MPI_COMM_WORLD, 2, 3);

    int m = 12, n = 12;
    int nb = 4;

    DdlaDesc descA(handle);
    descA.init(m, n, nb, nb, 0, 0);
    DdlaDesc descAT(handle);
    descAT.init(n, m, nb, nb, 0, 0);

    int loc_size_A = descA.m_loc() * descA.n_loc();
    int loc_size_AT = descAT.m_loc() * descAT.n_loc();

    std::vector<std::complex<double>> h_A(loc_size_A);
    for(int i=0;i<loc_size_A;i++){
        h_A[i] = std::complex<double>(i, -i);
    }

    std::complex<double>* d_A;
    std::complex<double>* d_AT;
    DEVICE_CHECK(deviceMalloc(&d_A, sizeof(std::complex<double>) * loc_size_A));
    DEVICE_CHECK(deviceMalloc(&d_AT, sizeof(std::complex<double>) * loc_size_AT));
    DEVICE_CHECK(deviceMemcpy(d_A, h_A.data(), sizeof(std::complex<double>) * loc_size_A, deviceMemcpyHostToDevice));
    std::vector<std::complex<double>> h_zero_AT(loc_size_AT);
    DEVICE_CHECK(deviceMemcpy(d_AT, h_zero_AT.data(), sizeof(std::complex<double>) * loc_size_AT, deviceMemcpyHostToDevice));

    ptran(d_A, descA, d_AT, descAT);

    std::vector<std::complex<double>> h_AT(loc_size_AT);
    DEVICE_CHECK(deviceMemcpy(h_AT.data(), d_AT, sizeof(std::complex<double>) * loc_size_AT, deviceMemcpyDeviceToHost));

    // Gather global A and AT on host
    auto gather = [&](const DdlaDesc& desc, const std::vector<std::complex<double>>& local){
        std::vector<int> recvcounts(nprocs), displs(nprocs);
        int sz = local.size();
        MPI_Allgather(&sz, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        displs[0] = 0;
        for(int i=1;i<nprocs;i++) displs[i] = displs[i-1] + recvcounts[i-1];
        std::vector<std::complex<double>> all(displs[nprocs-1] + recvcounts[nprocs-1]);
        MPI_Allgatherv(local.data(), sz, MPI_C_DOUBLE_COMPLEX,
                       all.data(), recvcounts.data(), displs.data(), MPI_C_DOUBLE_COMPLEX,
                       MPI_COMM_WORLD);
        std::vector<std::complex<double>> global(desc.m() * desc.n());
        int npcols = desc.npcols();
        for(int src=0; src<nprocs; src++){
            int prow = src / npcols;
            int pcol = src % npcols;
            int off = displs[src];
            int ml = num_loc(desc.m(), desc.mb(), prow, desc.irsrc(), desc.nprows());
            int nl = num_loc(desc.n(), desc.nb(), pcol, desc.icsrc(), desc.npcols());
            if(ml * nl != recvcounts[src]){
                std::cerr << "size mismatch src=" << src << " ml*nl=" << ml*nl << " recv=" << recvcounts[src] << std::endl;
                continue;
            }
            for(int j=0;j<nl;j++){
                int gj = indxl2g(j, desc.nb(), pcol, desc.icsrc(), desc.npcols());
                for(int i=0;i<ml;i++){
                    int gi = indxl2g(i, desc.mb(), prow, desc.irsrc(), desc.nprows());
                    global[gi + gj * desc.m()] = all[off + i + j * ml];
                }
            }
        }
        return global;
    };

    auto g_A = gather(descA, h_A);
    auto g_AT = gather(descAT, h_AT);

    double max_err = 0;
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){
            std::complex<double> ref = g_A[j + i * n]; // A^T[i,j] = A[j,i]
            std::complex<double> diff = g_AT[i + j * m] - ref;
            max_err = std::max(max_err, std::abs(diff));
        }
    }

    double global_max;
    MPI_Reduce(&max_err, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(descAT.myprow() == 0 && descAT.mypcol() == 0){
        std::cout << "ptran max_err " << global_max << std::endl;
    }

    DEVICE_CHECK(deviceFree(d_A));
    DEVICE_CHECK(deviceFree(d_AT));
    ddla_destroy(handle);
    MPI_Finalize();
    return 0;
}
