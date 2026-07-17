#include <mpi.h>
#include <iostream>
#include <vector>
#include <complex>
#include <iomanip>
#include <algorithm>
#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>

using namespace ddla;

/**
 * Benchmark pgemm with complex<double> on a 2x2 process grid (1 node, 4 GPU).
 * Compares 4 transpose combinations: (N,N), (C,N), (N,C), (C,C)
 * for matrix dimensions 100, 5000, 10000, 15000.
 */

struct BenchResult {
    int n;
    char transa, transb;
    double time;
};

void benchmark_pgemm(int n, char transa, char transb,
                     const DdlaHandle_t& ddla_handle,
                     std::vector<BenchResult>& results)
{
    int nb = std::min(128, n);
    DdlaDesc descA(ddla_handle);
    descA.init(n, n, nb, nb, 0, 0);
    DdlaDesc descB(ddla_handle);
    descB.init(n, n, nb, nb, 0, 0);
    DdlaDesc descC(ddla_handle);
    descC.init(n, n, nb, nb, 0, 0);

    int myid = descC.mypcol() + descC.myprow() * descC.npcols();
    const size_t nelem = static_cast<size_t>(descA.m_loc()) * descA.n_loc();
    const size_t size = nelem * sizeof(std::complex<double>);

    std::complex<double>* d_A = nullptr;
    std::complex<double>* d_B = nullptr;
    std::complex<double>* d_C = nullptr;
    DEVICE_CHECK(deviceMalloc(&d_A, size));
    DEVICE_CHECK(deviceMalloc(&d_B, size));
    DEVICE_CHECK(deviceMalloc(&d_C, size));

    random_generator(d_A, nelem, DEVICE_C_64F);
    random_generator(d_B, nelem, DEVICE_C_64F);
    random_generator(d_C, nelem, DEVICE_C_64F);

    std::complex<double> alpha(1.0, 0.0);
    std::complex<double> beta(0.0, 0.0);

    // Warm up
    pgemm(transa, transb, n, n, n, alpha, d_A, descA, d_B, descB, beta, d_C, descC);
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);

    // Timed runs: more iterations for small matrices
    int niter = (n <= 1000) ? 10 : (n <= 5000 ? 3 : 1);
    double start = MPI_Wtime();
    for (int iter = 0; iter < niter; iter++) {
        pgemm(transa, transb, n, n, n, alpha, d_A, descA, d_B, descB, beta, d_C, descC);
    }
    DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = (MPI_Wtime() - start) / niter;

    double max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, ddla_handle->comm);

    if (myid == 0) {
        std::cout << "n=" << std::setw(6) << n
                  << "  opA=" << transa << " opB=" << transb
                  << "  time=" << std::fixed << std::setprecision(4)
                  << max_elapsed << "s"
                  << "  (iters=" << niter << ")"
                  << std::endl;
        results.push_back({n, transa, transb, max_elapsed});
    }

    DEVICE_CHECK(deviceFree(d_A));
    DEVICE_CHECK(deviceFree(d_B));
    DEVICE_CHECK(deviceFree(d_C));
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int myid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    DdlaHandle_t ddla_handle = nullptr;
    ddla_init(ddla_handle);
    ddla_set(ddla_handle, MPI_COMM_WORLD, 2, 2);

    if (myid == 0) {
        std::cout << "=== pgemm benchmark (complex<double>, " << nprocs
                  << " MPI, 2x2 grid, V100) ===" << std::endl;
    }

    std::vector<int> sizes = {500, 5000, 10000, 15000};
    if (argc > 1) {
        sizes.clear();
        for (int i = 1; i < argc; ++i)
            sizes.push_back(std::atoi(argv[i]));
    }

    struct OpPair { char a, b; const char* label; };
    std::vector<OpPair> ops = {
        {'N','N',"(N,N)"},
        {'C','N',"(C,N)"},
        {'N','C',"(N,C)"},
        {'C','C',"(C,C)"},
    };

    std::vector<BenchResult> results;

    for (int n : sizes) {
        for (auto& op : ops) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (myid == 0)
                std::cout << "  running n=" << n << " " << op.label << " ..." << std::flush;
            benchmark_pgemm(n, op.a, op.b, ddla_handle, results);
            if (myid == 0)
                std::cout << " done" << std::endl;
        }
    }

    // Print summary table
    if (myid == 0) {
        std::cout << std::endl;
        std::cout << "=== Summary (time in seconds) ===" << std::endl;
        std::cout << std::setw(8) << "n";
        for (auto& op : ops)
            std::cout << std::setw(12) << op.label;
        std::cout << std::endl;
        std::cout << std::string(8 + 12*ops.size(), '-') << std::endl;

        for (int n : sizes) {
            std::cout << std::setw(8) << n;
            for (auto& op : ops) {
                double t = -1;
                for (auto& r : results) {
                    if (r.n == n && r.transa == op.a && r.transb == op.b) {
                        t = r.time;
                        break;
                    }
                }
                if (t >= 0)
                    std::cout << std::setw(12) << std::fixed << std::setprecision(4) << t;
                else
                    std::cout << std::setw(12) << "N/A";
            }
            std::cout << std::endl;
        }
    }

    ddla_destroy(ddla_handle);
    MPI_Finalize();
    return 0;
}
