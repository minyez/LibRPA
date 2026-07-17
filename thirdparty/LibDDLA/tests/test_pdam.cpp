#include <mpi.h>
#include <iostream>
#include <vector>
#include <complex>
#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>

using namespace ddla;

template <typename T1, typename T2>
void check_pdam(int n, const DdlaHandle_t& handle, const T1& alpha)
{
    DdlaDesc desc(handle);
    desc.init_square_blk(n, n, 0, 0);

    const int m_loc = desc.m_loc();
    const int n_loc = desc.n_loc();
    const int lld = desc.lld();
    const size_t nelem = static_cast<size_t>(n_loc) * lld;

    T2* d_A = nullptr;
    if (nelem > 0) {
        DEVICE_CHECK(deviceMallocAsync(reinterpret_cast<void**>(&d_A), nelem * sizeof(T2), handle->stream));
        DEVICE_CHECK(deviceMemsetAsync(d_A, 0, nelem * sizeof(T2), handle->stream));
    }

    pdam(alpha, d_A, desc);
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

    std::vector<T2> h_A(nelem);
    if (nelem > 0) {
        DEVICE_CHECK(deviceMemcpyAsync(h_A.data(), d_A, nelem * sizeof(T2), deviceMemcpyDeviceToHost, handle->stream));
        DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
    }

    const T2 expected = static_cast<T2>(alpha);

    int local_ok = 1;
    for (int i = 0; i < n; ++i) {
        const int ilo = desc.indx_g2l_r(i);
        const int jlo = desc.indx_g2l_c(i);
        if (ilo >= 0 && jlo >= 0) {
            if (h_A[ilo + jlo * lld] != expected) {
                local_ok = 0;
                break;
            }
        }
    }

    if (local_ok) {
        for (int iloc = 0; iloc < m_loc; ++iloc) {
            const int g_row = desc.indx_l2g_r(iloc);
            for (int jloc = 0; jloc < n_loc; ++jloc) {
                const int g_col = desc.indx_l2g_c(jloc);
                if (g_row == g_col) continue;
                if (h_A[iloc + jloc * lld] != T2(0)) {
                    local_ok = 0;
                    break;
                }
            }
            if (!local_ok) break;
        }
    }

    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (handle->myprow_ == 0 && handle->mypcol_ == 0) {
        std::cout << "pdam check n=" << n << " : " << (global_ok ? "OK" : "FAIL") << std::endl;
    }

    if (nelem > 0) {
        DEVICE_CHECK(deviceFreeAsync(d_A, handle->stream));
    }

    if (global_ok == 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    DdlaHandle_t handle = nullptr;
    ddla_init(handle);
    ddla_set(handle, MPI_COMM_WORLD);

    // Real scalar added to real matrix.
    check_pdam<double, double>(100, handle, 3.14);
    check_pdam<double, double>(257, handle, -1.0);

    // Real scalar added to complex matrix.
    check_pdam<double, std::complex<double>>(100, handle, 2.71);
    check_pdam<float, std::complex<float>>(50, handle, 1.5f);

    // Complex scalar added to complex matrix.
    check_pdam<std::complex<double>, std::complex<double>>(100, handle, std::complex<double>(1.0, 2.0));

    ddla_destroy(handle);
    MPI_Finalize();
    return 0;
}
