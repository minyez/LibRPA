#include "api_grid_test_common.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace api_grid_test;

namespace {

template <typename T>
constexpr bool is_complex_v = std::is_same_v<T, std::complex<float>>
                           || std::is_same_v<T, std::complex<double>>;

template <typename T>
struct real_type {
    using type = T;
};

template <typename Real>
struct real_type<std::complex<Real>> {
    using type = Real;
};

template <typename T>
using real_type_t = typename real_type<T>::type;

template <typename T>
T scalar(double real, double imag = 0.0)
{
    if constexpr (is_complex_v<T>){
        using Real = typename T::value_type;
        return T(static_cast<Real>(real), static_cast<Real>(imag));
    }else{
        (void)imag;
        return static_cast<T>(real);
    }
}

template <typename T>
T conjugate(const T& value)
{
    if constexpr (is_complex_v<T>){
        return std::conj(value);
    }else{
        return value;
    }
}

template <typename T>
const char* scalar_name();

template <>
const char* scalar_name<float>() { return "float"; }

template <>
const char* scalar_name<double>() { return "double"; }

template <>
const char* scalar_name<std::complex<float>>() { return "complex<float>"; }

template <>
const char* scalar_name<std::complex<double>>() { return "complex<double>"; }

template <typename T>
T upper_factor_value(int i, int j)
{
    if(i > j) return T{};
    if(i == j) return scalar<T>(1.75 + 0.003 * (i % 17));

    const double real = 0.004 * (((3 * i + 5 * j) % 11) - 5);
    const double imag = 0.003 * (((7 * i + 2 * j) % 9) - 4);
    return scalar<T>(real, imag);
}

template <typename T>
T upper_product_value(int i, int j, int n)
{
    T sum{};
    for(int k = std::max(i, j); k < n; ++k){
        sum += upper_factor_value<T>(i, k)
             * conjugate(upper_factor_value<T>(j, k));
    }
    return sum;
}

template <typename T>
T lower_factor_value(int i, int j)
{
    if(i < j) return T{};
    return conjugate(upper_factor_value<T>(j, i));
}

template <typename T>
T lower_product_value(int i, int j, int n)
{
    T sum{};
    for(int k = std::max(i, j); k < n; ++k){
        sum += conjugate(lower_factor_value<T>(k, i))
             * lower_factor_value<T>(k, j);
    }
    return sum;
}

template <typename T>
T opposite_triangle_sentinel(int i, int j, int n)
{
    return scalar<T>(-41.0 - 0.031 * i - 0.007 * n,
                     17.0 + 0.019 * j);
}

template <typename T>
T padding_sentinel()
{
    return scalar<T>(-913.0, 37.0);
}

template <typename T>
double factor_tolerance(int n)
{
    return 512.0 * static_cast<double>(std::numeric_limits<real_type_t<T>>::epsilon())
         * std::max(1, n);
}

inline void update_error(double& error, double value)
{
    if(!std::isfinite(value)){
        error = std::numeric_limits<double>::infinity();
    }else{
        error = std::max(error, value);
    }
}

template <typename T>
struct LocalMatrix {
    std::vector<T> values;
    std::vector<unsigned char> valid;
};

template <typename T, typename Fn>
LocalMatrix<T> make_local_with_padding(const ddla::DdlaDesc& desc, Fn value)
{
    const size_t logical_count = static_cast<size_t>(desc.lld()) * desc.n_loc();
    const size_t allocation_count = std::max<size_t>(1, logical_count);
    LocalMatrix<T> local{
        std::vector<T>(allocation_count, padding_sentinel<T>()),
        std::vector<unsigned char>(allocation_count, 0)
    };

    for(int jloc = 0; jloc < desc.n_loc(); ++jloc){
        const int j = desc.indx_l2g_c(jloc);
        for(int iloc = 0; iloc < desc.m_loc(); ++iloc){
            const int i = desc.indx_l2g_r(iloc);
            const size_t offset = iloc + static_cast<size_t>(jloc) * desc.lld();
            local.values[offset] = value(i, j);
            local.valid[offset] = 1;
        }
    }
    return local;
}

std::string case_name(const ddla::DdlaHandle_t& handle, const char* type,
                      char uplo, int n, int nb, int irsrc, int icsrc)
{
    std::ostringstream os;
    os << "ppotrf_bottom_right<" << type << ">"
       << " uplo=" << uplo
       << " grid=" << grid_name(handle)
       << " n=" << n << " nb=" << nb
       << " src=(" << irsrc << "," << icsrc << ")";
    return os.str();
}

template <typename T>
void check_success_case(const ddla::DdlaHandle_t& handle, char uplo, int n, int nb,
                        int irsrc, int icsrc)
{
    ddla::DdlaDesc desc(handle);
    desc.init(n, n, nb, nb, irsrc, icsrc);

    auto input = make_local_with_padding<T>(desc, [=](int i, int j){
        if(uplo == 'U'){
            return i <= j ? upper_product_value<T>(i, j, n)
                          : opposite_triangle_sentinel<T>(i, j, n);
        }
        return i >= j ? lower_product_value<T>(i, j, n)
                      : opposite_triangle_sentinel<T>(i, j, n);
    });
    DeviceBuffer<T> d_A(handle, input.values.size());
    upload(handle, d_A.ptr, input.values);
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

    int info = -1;
    ddla::ppotrf_bottom_right(uplo, n, d_A.ptr, desc, info);
    DEVICE_CHECK(deviceGetLastError());
    auto output = download(handle, d_A.ptr, input.values.size());

    double factor_error = 0.0;
    double sentinel_error = 0.0;
    double padding_error = 0.0;
    for(size_t offset = 0; offset < output.size(); ++offset){
        if(!input.valid[offset]){
            update_error(padding_error,
                         static_cast<double>(std::abs(output[offset] - padding_sentinel<T>())));
        }
    }
    for(int jloc = 0; jloc < desc.n_loc(); ++jloc){
        const int j = desc.indx_l2g_c(jloc);
        for(int iloc = 0; iloc < desc.m_loc(); ++iloc){
            const int i = desc.indx_l2g_r(iloc);
            const size_t offset = iloc + static_cast<size_t>(jloc) * desc.lld();
            const bool in_factor = uplo == 'U' ? i <= j : i >= j;
            if(in_factor){
                const T expected = uplo == 'U'
                    ? upper_factor_value<T>(i, j)
                    : lower_factor_value<T>(i, j);
                update_error(factor_error,
                             static_cast<double>(std::abs(output[offset] - expected)));
            }else{
                update_error(sentinel_error,
                             static_cast<double>(std::abs(
                                 output[offset]
                                 - opposite_triangle_sentinel<T>(i, j, n))));
            }
        }
    }

    const std::string name = case_name(
        handle, scalar_name<T>(), uplo, n, nb, irsrc, icsrc);
    require_close(handle, name + " info", std::abs(info), 0.0);
    require_close(handle, name + " factor", factor_error, factor_tolerance<T>(n));
    require_close(handle, name + " opposite sentinel", sentinel_error, 0.0);
    require_close(handle, name + " padding", padding_error, 0.0);
}

template <typename T>
void check_failure_case(const ddla::DdlaHandle_t& handle, char uplo, int n, int nb,
                        int irsrc, int icsrc, int failed_pivot)
{
    ddla::DdlaDesc desc(handle);
    desc.init(n, n, nb, nb, irsrc, icsrc);

    auto input = make_local_with_padding<T>(desc, [=](int i, int j){
        const bool opposite_triangle = uplo == 'U' ? i > j : i < j;
        if(opposite_triangle) return opposite_triangle_sentinel<T>(i, j, n);
        if(i != j) return T{};
        return scalar<T>(i == failed_pivot ? -1.0 : 4.0);
    });
    DeviceBuffer<T> d_A(handle, input.values.size());
    upload(handle, d_A.ptr, input.values);
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

    int info = -1;
    ddla::ppotrf_bottom_right(uplo, n, d_A.ptr, desc, info);
    DEVICE_CHECK(deviceGetLastError());
    auto output = download(handle, d_A.ptr, input.values.size());

    int info_min = 0;
    int info_max = 0;
    MPI_Allreduce(&info, &info_min, 1, MPI_INT, MPI_MIN, handle->comm);
    MPI_Allreduce(&info, &info_max, 1, MPI_INT, MPI_MAX, handle->comm);
    const int expected_info = failed_pivot + 1;
    double info_error = static_cast<double>(std::abs(info - expected_info));
    info_error = std::max(info_error, static_cast<double>(info_max - info_min));

    double sentinel_error = 0.0;
    double padding_error = 0.0;
    for(size_t offset = 0; offset < output.size(); ++offset){
        if(!input.valid[offset]){
            update_error(padding_error,
                         static_cast<double>(std::abs(output[offset] - padding_sentinel<T>())));
        }
    }
    for(int jloc = 0; jloc < desc.n_loc(); ++jloc){
        const int j = desc.indx_l2g_c(jloc);
        for(int iloc = 0; iloc < desc.m_loc(); ++iloc){
            const int i = desc.indx_l2g_r(iloc);
            const bool opposite_triangle = uplo == 'U' ? i > j : i < j;
            if(!opposite_triangle) continue;
            const size_t offset = iloc + static_cast<size_t>(jloc) * desc.lld();
            update_error(sentinel_error,
                         static_cast<double>(std::abs(
                             output[offset]
                             - opposite_triangle_sentinel<T>(i, j, n))));
        }
    }

    std::string name = case_name(
        handle, scalar_name<T>(), uplo, n, nb, irsrc, icsrc);
    name += failed_pivot == 0 ? " fail-left" : " fail-right";
    require_close(handle, name + " info consensus", info_error, 0.0);
    require_close(handle, name + " opposite sentinel", sentinel_error, 0.0);
    require_close(handle, name + " padding", padding_error, 0.0);
}

template <typename T>
void run_type_cases(const ddla::DdlaHandle_t& handle,
                    const std::vector<int>& sizes, int nb,
                    const std::vector<std::pair<int, int>>& sources)
{
    for(char uplo : {'U', 'L'}){
        for(const auto& [irsrc, icsrc] : sources){
            for(int n : sizes){
                check_success_case<T>(handle, uplo, n, nb, irsrc, icsrc);
            }

            const int failure_n = 2 * nb + 1;
            check_failure_case<T>(handle, uplo, failure_n, nb, irsrc, icsrc, 0);
            check_failure_case<T>(
                handle, uplo, failure_n, nb, irsrc, icsrc, failure_n - 1);
        }
    }
}

void check_zero_local_blocks(const ddla::DdlaHandle_t& handle, int nb,
                             int irsrc, int icsrc)
{
    if(handle->nprows_ == 1) return;

    ddla::DdlaDesc desc(handle);
    desc.init(nb - 1, nb - 1, nb, nb, irsrc, icsrc);
    const int local_has_zero = desc.m_loc() == 0 || desc.n_loc() == 0 ? 1 : 0;
    int zero_ranks = 0;
    MPI_Allreduce(&local_has_zero, &zero_ranks, 1, MPI_INT, MPI_SUM, handle->comm);
    require_close(handle, "ppotrf_bottom_right zero-local-block coverage",
                  zero_ranks > 0 ? 0.0 : 1.0, 0.0);
}

struct Options {
    bool quick = false;
    bool grid_set = false;
    int nprows = 0;
    int npcols = 0;
};

bool parse_test_options(int argc, char** argv, Options& options, std::string& error)
{
    for(int i = 1; i < argc; ++i){
        const std::string arg(argv[i]);
        if(arg == "--quick"){
            options.quick = true;
        }else if(arg == "--grid"){
            if(i + 1 >= argc){
                error = "--grid requires a value like 2x2";
                return false;
            }
            if(options.grid_set){
                error = "--grid was provided more than once";
                return false;
            }
            options.grid_set = true;
            if(!parse_grid_spec(argv[++i], options.nprows, options.npcols)){
                error = "invalid --grid value: " + std::string(argv[i]);
                return false;
            }
        }else if(arg.rfind("--grid=", 0) == 0){
            if(options.grid_set){
                error = "--grid was provided more than once";
                return false;
            }
            options.grid_set = true;
            if(!parse_grid_spec(arg.substr(7), options.nprows, options.npcols)){
                error = "invalid --grid value: " + arg.substr(7);
                return false;
            }
        }else{
            error = "unknown option: " + arg;
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int nprocs = 0;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Options options;
    std::string error;
    if(!parse_test_options(argc, argv, options, error)){
        if(rank == 0){
            std::cerr << "Error: " << error << std::endl;
            std::cerr << "Usage: " << argv[0]
                      << " [--grid 1x1|2x2|3x3] [--quick]" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int grid_dim = 0;
    if(options.grid_set){
        if(options.nprows != options.npcols
           || options.nprows * options.npcols != nprocs){
            if(rank == 0){
                std::cerr << "ppotrf_bottom_right tests require a square process grid "
                          << "matching the MPI rank count" << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        grid_dim = options.nprows;
    }else if(nprocs == 1){
        grid_dim = 1;
    }else if(nprocs == 4){
        grid_dim = 2;
    }else if(nprocs == 9){
        grid_dim = 3;
    }else{
        if(rank == 0){
            std::cout << "test_api_grid_ppotrf_bottom_right skipped: use 1, 4, or 9 MPI ranks"
                      << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    if(grid_dim != 1 && grid_dim != 2 && grid_dim != 3){
        if(rank == 0){
            std::cout << "test_api_grid_ppotrf_bottom_right skipped: only 1x1, 2x2, and 3x3 grids "
                      << "are covered" << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    ddla::DdlaHandle_t handle = nullptr;
    ddla::ddla_init(handle);
    ddla::ddla_set(handle, MPI_COMM_WORLD, grid_dim, grid_dim);

    constexpr int nb = 16;
    const bool use_quick_sizes = options.quick || grid_dim == 3;
    const std::vector<int> sizes = use_quick_sizes
        ? std::vector<int>{nb - 1, nb + 1, 2 * nb + 1}
        : std::vector<int>{nb - 1, nb, nb + 1, 2 * nb + 1, 3 * nb - 1};
    std::vector<std::pair<int, int>> sources;
    if(grid_dim == 1){
        sources.emplace_back(0, 0);
    }else if(grid_dim == 2){
        sources = {{0, 0}, {1, 1}, {0, 1}};
    }else{
        for(int irsrc = 0; irsrc < grid_dim; ++irsrc){
            for(int icsrc = 0; icsrc < grid_dim; ++icsrc){
                sources.emplace_back(irsrc, icsrc);
            }
        }
    }

    check_zero_local_blocks(handle, nb, sources.back().first, sources.back().second);
    run_type_cases<float>(handle, sizes, nb, sources);
    run_type_cases<double>(handle, sizes, nb, sources);
    run_type_cases<std::complex<float>>(handle, sizes, nb, sources);
    run_type_cases<std::complex<double>>(handle, sizes, nb, sources);

    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
    ddla::ddla_destroy(handle);

    if(rank == 0){
        std::cout << "test_api_grid_ppotrf_bottom_right passed" << std::endl;
    }
    MPI_Finalize();
    return 0;
}
