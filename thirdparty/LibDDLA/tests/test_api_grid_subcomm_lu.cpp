#include "api_grid_test_common.h"

using namespace api_grid_test;

namespace {

template <typename Factorization>
void run_factorization_case(const ddla::DdlaHandle_t& handle,
                            const std::string& name,
                            Factorization factorize)
{
    const int n = 6;
    const int nb = 2;
    ddla::DdlaDesc descA(handle);
    descA.init(n, n, nb, nb, 0, 0);

    auto h_A = make_local<Complex>(descA, [=](int i, int j){
        return dominant_value(i, j, n);
    });
    DeviceBuffer<Complex> d_A(handle, h_A.size());
    upload(handle, d_A.ptr, h_A);
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

    int info = -1;
    factorize(d_A.ptr, descA, info);
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
    require_close(handle, name + "(info)", std::abs(info), 0.0);
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if(world_size < 2){
        if(world_rank == 0){
            std::cout << "test_api_grid_subcomm_lu skipped: requires at least 2 MPI ranks"
                      << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                  MPI_INFO_NULL, &local_comm));
    int local_rank = 0;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CHECK(MPI_Comm_free(&local_comm));

    const int active_size = world_size / 2;
    const bool is_active = world_rank < active_size;
    MPI_Comm role_comm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, is_active ? 0 : 1,
                             world_rank, &role_comm));

    constexpr int completion_tag = 7101;
    int completion = 0;

    if(is_active){
        ddla::DdlaHandle_t handle = nullptr;
        ddla::ddla_init(handle);

        int device_count = 0;
        DEVICE_CHECK(deviceGetDeviceCount(&device_count));
        if(device_count <= 0){
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        handle->set_local_device(local_rank % device_count);
        ddla::ddla_set(handle, role_comm, 1, active_size);

        run_factorization_case(handle, "subcomm pgetrf",
            [](Complex* d_A, const ddla::DdlaDesc& descA, int& info){
                std::vector<int> ipiv(std::max(1, descA.m_loc()));
                ddla::pgetrf(descA.m(), descA.n(), d_A, descA, ipiv.data(), info);
            });

        run_factorization_case(handle, "subcomm pgetrf_bpiv",
            [&](Complex* d_A, const ddla::DdlaDesc& descA, int& info){
                DeviceBuffer<int> d_ipiv(handle, std::max(1, descA.m_loc()));
                ddla::pgetrf_bpiv(descA.m(), descA.n(), d_A, descA,
                                  d_ipiv.ptr, info);
            });

        run_factorization_case(handle, "subcomm pgetrf_nopiv",
            [](Complex* d_A, const ddla::DdlaDesc& descA, int& info){
                ddla::pgetrf_nopiv(descA.m(), descA.n(), d_A, descA, info);
            });

        DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
        ddla::ddla_destroy(handle);

        if(world_rank == 0){
            completion = 1;
            MPI_CHECK(MPI_Send(&completion, 1, MPI_INT, active_size,
                               completion_tag, MPI_COMM_WORLD));
        }
    }else{
        if(world_rank == active_size){
            MPI_CHECK(MPI_Recv(&completion, 1, MPI_INT, 0, completion_tag,
                               MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        }
        MPI_CHECK(MPI_Bcast(&completion, 1, MPI_INT, 0, role_comm));
        if(completion != 1){
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_CHECK(MPI_Comm_free(&role_comm));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if(world_rank == 0){
        std::cout << "test_api_grid_subcomm_lu passed" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
