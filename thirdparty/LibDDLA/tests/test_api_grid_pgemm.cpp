#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_pgemm(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    const int m = round_up_for_grid(base.m, nb, handle->nprows_);
    const int n = round_up_for_grid(base.n, nb, handle->npcols_);
    const int k = std::max(base.k, nb * std::max(handle->nprows_, handle->npcols_) + 1);
    const Complex alpha(0.8, -0.2);
    const Complex beta(-0.3, 0.1);

    for(char transa : {'N', 'T', 'C'}){
        for(char transb : {'N', 'T', 'C'}){
            const int a_rows = transa == 'N' ? m : k;
            const int a_cols = transa == 'N' ? k : m;
            const int b_rows = transb == 'N' ? k : n;
            const int b_cols = transb == 'N' ? n : k;

            ddla::DdlaDesc descA(handle), descB(handle), descC(handle);
            descA.init(a_rows, a_cols, nb, nb, 0, 0);
            descB.init(b_rows, b_cols, nb, nb, 0, 0);
            descC.init(m, n, nb, nb, 0, 0);

            auto h_A = make_local<Complex>(descA, [](int i, int j){ return general_value(i, j, 1); });
            auto h_B = make_local<Complex>(descB, [](int i, int j){ return general_value(i, j, 2); });
            auto h_C = make_local<Complex>(descC, [](int i, int j){ return general_value(i, j, 3); });

            DeviceBuffer<Complex> d_A(handle, h_A.size());
            DeviceBuffer<Complex> d_B(handle, h_B.size());
            DeviceBuffer<Complex> d_C(handle, h_C.size());
            upload(handle, d_A.ptr, h_A);
            upload(handle, d_B.ptr, h_B);
            upload(handle, d_C.ptr, h_C);
            DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

            ddla::pgemm(transa, transb, m, n, k, alpha, d_A.ptr, descA, d_B.ptr, descB,
                        beta, d_C.ptr, descC);
            auto out = download(handle, d_C.ptr, h_C.size());

            const double err = local_max_error<Complex>(descC, out, [&](int i, int j){
                Complex ref = beta * general_value(i, j, 3);
                for(int l = 0; l < k; ++l){
                    ref += alpha * op_value(transa, a_rows, a_cols, i, l, general_value, 1)
                         * op_value(transb, b_rows, b_cols, l, j, general_value, 2);
                }
                return ref;
            });
            std::string name = std::string("pgemm(") + transa + "," + transb + ")";
            require_close(handle, name, err, 2e-10);
        }
    }
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pgemm", check_pgemm);
}
