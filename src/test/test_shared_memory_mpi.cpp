#include <cassert>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../mpi/mpi_shm.h"

using namespace librpa_int;

namespace
{

void test_shared_a_with_local_b_multiply(MPI_Comm comm)
{
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    constexpr int owner = 0;
    constexpr int m = 2;
    constexpr int kdim = 3;
    constexpr int ncol = 2;
    constexpr int nmat = 2;

    ShmMpiHandler<double> a(comm, rank == owner ? m * kdim : 0);

    if (rank == owner)
    {
        for (int row = 0; row != m; ++row)
            for (int k = 0; k != kdim; ++k) a.local_data()[row * kdim + k] = 10.0 * row + k + 1.0;
    }

    a.sync();

    assert(a.local_count() == (rank == owner ? m * kdim : 0));
    assert(a.counts()[owner] == m * kdim);
    for (int src = 0; src != size; ++src)
        if (src != owner) assert(a.count(src) == 0);

    std::vector<double> b(nmat * kdim * ncol);
    std::vector<double> c(nmat * m * ncol, 0.0);

    for (int imat = 0; imat != nmat; ++imat)
        for (int k = 0; k != kdim; ++k)
            for (int col = 0; col != ncol; ++col)
                b[(imat * kdim + k) * ncol + col] =
                    100.0 * rank + 10.0 * imat + 2.0 * k + col + 1.0;

    const double *shared_a = a.data(owner);
    for (int imat = 0; imat != nmat; ++imat)
    {
        for (int row = 0; row != m; ++row)
        {
            for (int col = 0; col != ncol; ++col)
            {
                double value = 0.0;
                for (int k = 0; k != kdim; ++k)
                    value += shared_a[row * kdim + k] * b[(imat * kdim + k) * ncol + col];
                c[(imat * m + row) * ncol + col] = value;
            }
        }
    }

    for (int imat = 0; imat != nmat; ++imat)
    {
        for (int row = 0; row != m; ++row)
        {
            for (int col = 0; col != ncol; ++col)
            {
                double ref = 0.0;
                for (int k = 0; k != kdim; ++k)
                {
                    const double a_ref = 10.0 * row + k + 1.0;
                    const double b_ref = 100.0 * rank + 10.0 * imat + 2.0 * k + col + 1.0;
                    ref += a_ref * b_ref;
                }
                assert(std::abs(c[(imat * m + row) * ncol + col] - ref) < 1e-12);
            }
        }
    }
}

void test_move_keeps_window(MPI_Comm comm)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    ShmMpiHandler<int> window(comm, 1);
    window.local_data()[0] = rank + 7;
    window.sync();

    ShmMpiHandler<int> moved(std::move(window));
    moved.sync();
    assert(moved.local_data()[0] == rank + 7);
}

}  // namespace

int main(int argc, char *argv[])
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    MPI_Comm shared_comm = MPI_COMM_NULL;
    int ierr =
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shared_comm);
    if (ierr != MPI_SUCCESS) throw std::runtime_error("MPI_Comm_split_type failed");

    test_shared_a_with_local_b_multiply(shared_comm);
    test_move_keeps_window(shared_comm);

    MPI_Comm_free(&shared_comm);
    MPI_Finalize();
    return 0;
}
