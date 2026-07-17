#include <cassert>
#include <cmath>
#include <mpi.h>
#include <time.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <complex>
#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <random>
#include <ddla/ddla_stream.h>

using namespace ddla;

namespace {

std::complex<double> op_value(char trans, const std::vector<std::complex<double>>& mat,
                              int m, int n, int i, int j)
{
    // mat is stored column-major, m rows x n cols
    if(trans == 'N'){
        return mat[i + j * m];
    }else if(trans == 'T'){
        return mat[j + i * m];
    }else{ // 'C'
        return std::conj(mat[j + i * m]);
    }
}

void check_pgemm(char transa, char transb,
                 int m, int n, int k, int nb,
                 const DdlaHandle_t& ddla_handle)
{
    DdlaDesc descA(ddla_handle);
    descA.init(m, k, nb, nb, 0, 0);
    DdlaDesc descB(ddla_handle);
    descB.init(k, n, nb, nb, 0, 0);
    DdlaDesc descC(ddla_handle);
    descC.init(m, n, nb, nb, 0, 0);

    int myid = descC.mypcol() + descC.myprow() * descC.npcols();

    std::vector<std::complex<double>> h_A(descA.m_loc() * descA.n_loc());
    std::vector<std::complex<double>> h_B(descB.m_loc() * descB.n_loc());
    std::vector<std::complex<double>> h_C(descC.m_loc() * descC.n_loc());

    std::mt19937 gen(42 + myid);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for(auto& v : h_A) v = std::complex<double>(dist(gen), dist(gen));
    for(auto& v : h_B) v = std::complex<double>(dist(gen), dist(gen));
    for(auto& v : h_C) v = std::complex<double>(dist(gen), dist(gen));

    std::complex<double>* d_A;
    std::complex<double>* d_B;
    std::complex<double>* d_C;
    DEVICE_CHECK(deviceMalloc(&d_A, sizeof(std::complex<double>) * h_A.size()));
    DEVICE_CHECK(deviceMalloc(&d_B, sizeof(std::complex<double>) * h_B.size()));
    DEVICE_CHECK(deviceMalloc(&d_C, sizeof(std::complex<double>) * h_C.size()));

    DEVICE_CHECK(deviceMemcpy(d_A, h_A.data(), sizeof(std::complex<double>) * h_A.size(), deviceMemcpyHostToDevice));
    DEVICE_CHECK(deviceMemcpy(d_B, h_B.data(), sizeof(std::complex<double>) * h_B.size(), deviceMemcpyHostToDevice));
    DEVICE_CHECK(deviceMemcpy(d_C, h_C.data(), sizeof(std::complex<double>) * h_C.size(), deviceMemcpyHostToDevice));

    std::complex<double> alpha(1.0, 0.0);
    std::complex<double> beta(0.0, 0.0);

    double start = MPI_Wtime();
    pgemm(transa, transb, m, n, k, alpha, d_A, descA, d_B, descB, beta, d_C, descC);
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    double elapsed = MPI_Wtime() - start;

    std::vector<std::complex<double>> h_C_out(descC.m_loc() * descC.n_loc());
    DEVICE_CHECK(deviceMemcpy(h_C_out.data(), d_C, sizeof(std::complex<double>) * h_C_out.size(), deviceMemcpyDeviceToHost));

    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    auto gather_global = [&](const DdlaDesc& desc, const std::vector<std::complex<double>>& local)->std::vector<std::complex<double>>{
        int mg = desc.m();
        int ng = desc.n();
        std::vector<std::complex<double>> global(mg * ng);
        std::vector<int> recvcounts(nprocs);
        std::vector<int> displs(nprocs);
        int loc_size = local.size();
        MPI_Allgather(&loc_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        displs[0] = 0;
        for(int i=1;i<nprocs;i++) displs[i] = displs[i-1] + recvcounts[i-1];
        std::vector<std::complex<double>> all_local(displs[nprocs-1] + recvcounts[nprocs-1]);
        MPI_Allgatherv(local.data(), loc_size, MPI_C_DOUBLE_COMPLEX,
                       all_local.data(), recvcounts.data(), displs.data(), MPI_C_DOUBLE_COMPLEX,
                       MPI_COMM_WORLD);
        int npcols = desc.npcols();
        for(int src=0; src<nprocs; src++){
            int prow = src / npcols;
            int pcol = src % npcols;
            int offset = displs[src];
            int count = recvcounts[src];
            if(count == 0) continue;
            int m_loc = num_loc(mg, desc.mb(), prow, desc.irsrc(), desc.nprows());
            int n_loc = num_loc(ng, desc.nb(), pcol, desc.icsrc(), desc.npcols());
            if(m_loc * n_loc != count) continue;
            for(int j_loc=0; j_loc<n_loc; j_loc++){
                int j_g = indxl2g(j_loc, desc.nb(), pcol, desc.icsrc(), desc.npcols());
                for(int i_loc=0; i_loc<m_loc; i_loc++){
                    int i_g = indxl2g(i_loc, desc.mb(), prow, desc.irsrc(), desc.nprows());
                    global[i_g + j_g * mg] = all_local[offset + i_loc + j_loc * m_loc];
                }
            }
        }
        return global;
    };

    std::vector<std::complex<double>> g_A = gather_global(descA, h_A);
    std::vector<std::complex<double>> g_B = gather_global(descB, h_B);
    std::vector<std::complex<double>> g_C = gather_global(descC, h_C_out);

    int ma = (transa == 'N') ? m : k;
    int na = (transa == 'N') ? k : m;
    int mb_ = (transb == 'N') ? k : n;
    int nb_ = (transb == 'N') ? n : k;

    double max_err = 0.0;
    for(int j=0; j<n; j++){
        for(int i=0; i<m; i++){
            std::complex<double> ref(0.0, 0.0);
            for(int l=0; l<k; l++){
                ref += alpha * op_value(transa, g_A, ma, na, i, l) * op_value(transb, g_B, mb_, nb_, l, j);
            }
            std::complex<double> diff = g_C[i + j * m] - ref;
            max_err = std::max(max_err, std::abs(diff));
        }
    }

    double global_max_err;
    MPI_Reduce(&max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(myid == 0){
        std::cout << "pgemm(" << transa << "," << transb << ") "
                  << m << "x" << n << "x" << k
                  << " grid " << descC.nprows() << "x" << descC.npcols()
                  << " time " << elapsed << "s"
                  << " max_err " << global_max_err << std::endl;
    }

    if(global_max_err > 1e-10){
        std::cerr << "FAIL: pgemm(" << transa << "," << transb << ") error too large" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    DEVICE_CHECK(deviceFree(d_A));
    DEVICE_CHECK(deviceFree(d_B));
    DEVICE_CHECK(deviceFree(d_C));
}

} // anonymous namespace

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    std::vector<std::pair<int,int>> grids;
    if(nprocs == 4){
        grids = {{1,4}, {2,2}, {4,1}};
    }else if(nprocs == 6){
        grids = {{2,3}, {3,2}};
    }else if(nprocs == 8){
        grids = {{2,4}, {4,2}};
    }else{
        grids = {{-1,-1}};
    }

    for(const auto& grid : grids){
        DdlaHandle_t ddla_handle;
        ddla_init(ddla_handle);
        if(grid.first < 0){
            ddla_set(ddla_handle);
        }else{
            ddla_set(ddla_handle, MPI_COMM_WORLD, grid.first, grid.second);
        }

        int m = 100, n = 100, k = 100;
        int nb = 16;
        std::vector<char> trans_opts = {'N', 'T', 'C'};
        for(char transa : trans_opts){
            for(char transb : trans_opts){
                check_pgemm(transa, transb, m, n, k, nb, ddla_handle);
            }
        }

        ddla_destroy(ddla_handle);
    }

    MPI_Finalize();
    return 0;
}
