#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_pdam(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    const int n = square_size(handle, base);
    ddla::DdlaDesc desc(handle);
    desc.init(n, n, nb, nb, 0, 0);

    std::vector<Complex> h_A(local_size(desc), Complex(0.0, 0.0));
    DeviceBuffer<Complex> d_A(handle, h_A.size());
    upload(handle, d_A.ptr, h_A);
    const Complex alpha(1.25, -0.5);
    ddla::pdam(alpha, d_A.ptr, desc);
    auto out = download(handle, d_A.ptr, h_A.size());

    const double err = local_max_error<Complex>(desc, out, [&](int i, int j){
        return i == j ? alpha : Complex(0.0, 0.0);
    });
    require_close(handle, "pdam(complex)", err, 1e-12);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pdam", check_pdam);
}
