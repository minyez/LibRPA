#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_pgetrf(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    const int n = square_size(handle, base);
    const int nrhs = nrhs_size(base);
    ddla::DdlaDesc descA(handle), descB(handle);
    descA.init(n, n, nb, nb, 0, 0);
    descB.init(n, nrhs, nb, nb, 0, 0);

    {
        const int singular_index = std::min(nb + 1, n - 1);
        auto h_singular = make_local<Complex>(descA, [=](int i, int j){
            if(i != j) return Complex(0.0, 0.0);
            return Complex(i == singular_index ? 0.0 : 1.0, 0.0);
        });
        DeviceBuffer<Complex> d_singular(handle, h_singular.size());
        upload(handle, d_singular.ptr, h_singular);
        DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

        std::vector<int> singular_ipiv(descA.m_loc(), -1);
        int singular_info = -1;
        ddla::pgetrf(n, n, d_singular.ptr, descA, singular_ipiv.data(), singular_info);
        require_close(handle, "pgetrf(singular info)",
                      std::abs(singular_info - (singular_index + 1)), 0.0);
    }

    auto h_A = make_local<Complex>(descA, [=](int i, int j){ return dominant_value(i, j, n); });
    auto h_B = build_rhs(descB, n, dominant_value, n);

    DeviceBuffer<Complex> d_A(handle, h_A.size());
    DeviceBuffer<Complex> d_B(handle, h_B.size());
    upload(handle, d_A.ptr, h_A);
    upload(handle, d_B.ptr, h_B);
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

    std::vector<int> ipiv(descA.m_loc());
    int info = -1;
    ddla::pgetrf(n, n, d_A.ptr, descA, ipiv.data(), info);
    if(info != 0) MPI_Abort(handle->comm, 1);
    ddla::pgetrs('N', n, nrhs, d_A.ptr, descA, ipiv.data(), d_B.ptr, descB);
    check_solution(handle, descB, d_B.ptr, h_B.size(), "pgetrf", 5e-9);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pgetrf", check_pgetrf);
}
