#include <algorithm>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <mpi.h>

#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>

using namespace ddla;

using Complex = std::complex<double>;

Complex lower_value(int i, int j, int n)
{
    if(i < j) return Complex(0.0, 0.0);
    if(i == j) return Complex(2.0 + 0.0001 * n + 0.005 * i, 0.0);
    return Complex(0.003 * ((2 * i + j + n) % 7 - 3),
                   0.002 * ((i + 3 * j) % 5 - 2));
}

Complex rhs_value(int i, int j, int n)
{
    return Complex(0.01 * ((i + 2 * j + n) % 11 - 5),
                   0.008 * ((3 * i + j) % 13 - 6));
}

template <typename Fn>
void fill_local(int rows, int cols, const DdlaDesc& desc, Complex* d_A,
                const DdlaHandle_t& handle, Fn value)
{
    std::vector<Complex> local(static_cast<size_t>(desc.lld()) * desc.n_loc(),
                               Complex(0.0, 0.0));
    for(int jloc = 0; jloc < desc.n_loc(); ++jloc){
        const int j = desc.indx_l2g_c(jloc);
        if(j >= cols) continue;
        for(int iloc = 0; iloc < desc.m_loc(); ++iloc){
            const int i = desc.indx_l2g_r(iloc);
            if(i >= rows) continue;
            local[iloc + jloc * desc.lld()] = value(i, j);
        }
    }
    DEVICE_CHECK(deviceMemcpyAsync(d_A, local.data(), local.size() * sizeof(Complex),
                                  deviceMemcpyHostToDevice, handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
}

double benchmark_ptrtrs(int n, int nrhs, const DdlaHandle_t& handle)
{
    const int nb = std::min(128, n);
    DdlaDesc descA(handle), descB(handle);
    descA.init(n, n, nb, nb, 0, 0);
    descB.init(n, nrhs, nb, nb, 0, 0);

    const size_t a_nelem = static_cast<size_t>(descA.lld()) * descA.n_loc();
    const size_t b_nelem = static_cast<size_t>(descB.lld()) * descB.n_loc();
    Complex* d_A = nullptr;
    Complex* d_B = nullptr;
    DEVICE_CHECK(deviceMallocAsync(reinterpret_cast<void**>(&d_A),
                                  std::max<size_t>(1, a_nelem) * sizeof(Complex),
                                  handle->stream));
    DEVICE_CHECK(deviceMallocAsync(reinterpret_cast<void**>(&d_B),
                                  std::max<size_t>(1, b_nelem) * sizeof(Complex),
                                  handle->stream));
    fill_local(n, n, descA, d_A, handle, [&](int i, int j){ return lower_value(i, j, n); });
    fill_local(n, nrhs, descB, d_B, handle, [&](int i, int j){ return rhs_value(i, j, n); });

    MPI_Barrier(handle->comm);
    const double start = MPI_Wtime();
    ptrtrs('L', 'L', 'N', 'N', n, nrhs, d_A, descA, d_B, descB);
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
    MPI_Barrier(handle->comm);
    const double elapsed = MPI_Wtime() - start;

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, handle->comm);

    DEVICE_CHECK(deviceFreeAsync(d_A, handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_B, handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

    if(handle->myid == 0){
        std::cout << "RESULT n=" << n
                  << " nrhs=" << nrhs
                  << " type=complex<double>"
                  << " op=ptrtrs(L,L,N,N)"
                  << " grid=2x2"
                  << " ranks=4"
                  << " nb=" << nb
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
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(nprocs != 4){
        if(rank == 0){
            std::cerr << "benchmark_ptrtrs requires exactly 4 MPI ranks for a 2x2 grid"
                      << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    DdlaHandle_t handle = nullptr;
    ddla_init(handle);
    ddla_set(handle, MPI_COMM_WORLD, 2, 2);

    std::vector<int> sizes = {500, 5000, 10000, 15000};
    if(argc > 1){
        sizes.clear();
        for(int i = 1; i < argc; ++i){
            sizes.push_back(std::atoi(argv[i]));
        }
    }

    if(handle->myid == 0){
        std::cout << "=== ptrtrs benchmark: complex<double>, 4 MPI ranks, 2x2 grid, nrhs=n ==="
                  << std::endl;
    }

    for(int n : sizes){
        benchmark_ptrtrs(n, n, handle);
    }

    ddla_destroy(handle);
    MPI_Finalize();
    return 0;
}
