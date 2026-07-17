#include <algorithm>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <mpi.h>

#include "benchmark_grid_options.h"

#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <ddla/scal.h>

using namespace ddla;

using Complex = std::complex<double>;

constexpr unsigned long long kRandomSeed = 20260710ULL;

void fill_matrix(int n, const DdlaDesc& desc, Complex* d_A, const DdlaHandle_t& handle)
{
    const size_t nelem = static_cast<size_t>(desc.m_loc()) * desc.n_loc();
    derandGenerator_t generator;
    DERAND_CHECK(derandCreateGenerator(&generator, DERAND_RNG_PSEUDO_DEFAULT));
    DERAND_CHECK(derandSetPseudoRandomGeneratorSeed(
        generator, kRandomSeed + static_cast<unsigned long long>(handle->myid)));
    DERAND_CHECK(derandGenerateUniformDouble(
        generator, reinterpret_cast<double*>(d_A), nelem * 2));
    DERAND_CHECK(derandDestroyGenerator(generator));
    BLAS_CHECK(deblasScal(handle->blasH, nelem, Complex(1.0e-4, 0.0), d_A, 1));

    const Complex diag(2.0, 0.0);
    for(int i = 0; i < n; ++i){
        const int iloc = desc.indx_g2l_r(i);
        const int jloc = desc.indx_g2l_c(i);
        if(iloc >= 0 && jloc >= 0){
            DEVICE_CHECK(deviceMemcpyAsync(d_A + iloc + jloc * desc.lld(), &diag,
                                          sizeof(Complex), deviceMemcpyHostToDevice,
                                          handle->stream));
        }
    }
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
}

double benchmark_pgetrf(int n, const DdlaHandle_t& handle,
                        const benchmark_cli::Options& options, bool warmup)
{
    const int nb = std::min(128, n);
    DdlaDesc desc(handle);
    desc.init(n, n, nb, nb, 0, 0);

    const size_t nelem = static_cast<size_t>(desc.m_loc()) * desc.n_loc();
    Complex* d_A = nullptr;
    DEVICE_CHECK(deviceMallocAsync(reinterpret_cast<void**>(&d_A),
                                  nelem * sizeof(Complex), handle->stream));
    fill_matrix(n, desc, d_A, handle);

    std::vector<int> ipiv(desc.m_loc());
    int info = -1;

    MPI_Barrier(handle->comm);
    const double start = MPI_Wtime();
    pgetrf(n, n, d_A, desc, ipiv.data(), info);
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
    MPI_Barrier(handle->comm);
    const double elapsed = MPI_Wtime() - start;

    int global_info = 0;
    MPI_Allreduce(&info, &global_info, 1, MPI_INT, MPI_MAX, handle->comm);
    if(global_info != 0){
        if(handle->myid == 0){
            std::cerr << "pgetrf failed for n=" << n << ", info=" << global_info << std::endl;
        }
        MPI_Abort(handle->comm, 1);
    }

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, handle->comm);

    DEVICE_CHECK(deviceFreeAsync(d_A, handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

    if(handle->myid == 0){
        std::cout << (warmup ? "WARMUP" : "RESULT") << " n=" << n
                  << " type=complex<double>"
                  << " grid=" << benchmark_cli::grid_name(options)
                  << " ranks=" << handle->nprocs
                  << " nb=" << nb
                  << " seed=" << kRandomSeed
                  << " time_s=" << std::fixed << std::setprecision(6)
                  << max_elapsed
                  << std::endl;
    }
    return max_elapsed;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int nprocs = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    benchmark_cli::Options options;
    std::string option_error;
    if(!benchmark_cli::parse(argc, argv, false, 1, options, option_error)){
        if(rank == 0){
            std::cerr << "Error: " << option_error << std::endl;
            std::cerr << benchmark_cli::usage(argv[0], false) << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if(nprocs != options.nprows * options.npcols){
        if(rank == 0){
            std::cerr << "--grid " << benchmark_cli::grid_name(options)
                      << " requires " << options.nprows * options.npcols
                      << " MPI ranks, but this run has " << nprocs << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    DdlaHandle_t handle = nullptr;
    ddla_init(handle);
    ddla_set(handle, MPI_COMM_WORLD, options.nprows, options.npcols);

    if(handle->myid == 0){
        std::cout << "=== pgetrf benchmark: complex<double>, " << nprocs
                  << " MPI ranks, " << benchmark_cli::grid_name(options)
                  << " grid, seed=" << kRandomSeed << " ===" << std::endl;
    }

    for(size_t i = 0; i < options.sizes.size(); ++i){
        const bool warmup = i == 0 && options.sizes[i] == 500;
        benchmark_pgetrf(options.sizes[i], handle, options, warmup);
    }

    ddla_destroy(handle);
    MPI_Finalize();
    return 0;
}
