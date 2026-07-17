#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

#include <mpi.h>

#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>

#include "potrf_bottom_right_internal.h"

using namespace ddla;

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
T opposite_triangle_sentinel(int row, int column)
{
    return scalar<T>(-73.0 - 0.013 * row, 19.0 + 0.017 * column);
}

template <typename T>
T padding_sentinel()
{
    return scalar<T>(-911.0, 37.0);
}

bool in_referenced_triangle(char uplo, int row, int column)
{
    return uplo == 'U' ? row <= column : row >= column;
}

template <typename T>
std::vector<T> make_factor(char uplo, int n, int lda)
{
    std::vector<T> factor(static_cast<std::size_t>(lda) * n, T{});
    for(int column = 0; column < n; ++column){
        for(int row = 0; row < n; ++row){
            if(!in_referenced_triangle(uplo, row, column)){
                continue;
            }
            if(row == column){
                factor[row + static_cast<std::size_t>(column) * lda] =
                    scalar<T>(1.5 + 0.002 * (row % 23));
            }else{
                const double real =
                    0.004 * (((3 * row + 5 * column) % 11) - 5);
                const double imag =
                    0.003 * (((7 * row + 2 * column) % 9) - 4);
                factor[row + static_cast<std::size_t>(column) * lda] =
                    scalar<T>(real, imag);
            }
        }
    }
    return factor;
}

template <typename T>
std::vector<T> form_product(
    char uplo, const std::vector<T>& factor, int n, int lda)
{
    std::vector<T> matrix(
        static_cast<std::size_t>(lda) * n, padding_sentinel<T>());
    for(int column = 0; column < n; ++column){
        for(int row = 0; row < n; ++row){
            if(!in_referenced_triangle(uplo, row, column)){
                matrix[row + static_cast<std::size_t>(column) * lda] =
                    opposite_triangle_sentinel<T>(row, column);
                continue;
            }
            T sum{};
            if(uplo == 'U'){
                for(int k = column; k < n; ++k){
                    sum += factor[row + static_cast<std::size_t>(k) * lda]
                         * conjugate(factor[
                               column + static_cast<std::size_t>(k) * lda]);
                }
            }else{
                for(int k = row; k < n; ++k){
                    sum += conjugate(factor[
                               k + static_cast<std::size_t>(row) * lda])
                         * factor[
                               k + static_cast<std::size_t>(column) * lda];
                }
            }
            matrix[row + static_cast<std::size_t>(column) * lda] = sum;
        }
    }
    return matrix;
}

template <typename T>
double tolerance(int n)
{
    return 512.0
         * static_cast<double>(std::numeric_limits<real_type_t<T>>::epsilon())
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
bool run_success_case(char uplo, int n, const DdlaHandle_t& handle)
{
    constexpr int global_offset = 23;
    const int lda = n + 3;
    const auto expected = make_factor<T>(uplo, n, lda);
    const auto input = form_product(uplo, expected, n, lda);
    const std::size_t matrix_bytes = input.size() * sizeof(T);

    T* d_A = nullptr;
    T* d_work = nullptr;
    int* d_info = nullptr;
    DEVICE_CHECK(deviceMallocAsync(
        reinterpret_cast<void**>(&d_A), matrix_bytes, handle->stream));
    DEVICE_CHECK(deviceMallocAsync(
        reinterpret_cast<void**>(&d_work),
        static_cast<std::size_t>(n) * n * sizeof(T), handle->stream));
    DEVICE_CHECK(deviceMallocAsync(
        reinterpret_cast<void**>(&d_info), sizeof(int), handle->stream));
    DEVICE_CHECK(deviceMemcpyAsync(
        d_A, input.data(), matrix_bytes,
        deviceMemcpyHostToDevice, handle->stream));

    int info = -1;
    detail::potrf_bottom_right_block(
        uplo, n, d_A, lda, global_offset,
        d_work, d_info, info, handle);
    DEVICE_CHECK(deviceGetLastError());

    std::vector<T> output(input.size());
    DEVICE_CHECK(deviceMemcpyAsync(
        output.data(), d_A, matrix_bytes,
        deviceMemcpyDeviceToHost, handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

    double factor_error = 0.0;
    double opposite_triangle_error = 0.0;
    double padding_error = 0.0;
    for(int column = 0; column < n; ++column){
        for(int row = 0; row < n; ++row){
            const T actual =
                output[row + static_cast<std::size_t>(column) * lda];
            const bool referenced =
                in_referenced_triangle(uplo, row, column);
            const T reference = referenced
                ? expected[row + static_cast<std::size_t>(column) * lda]
                : opposite_triangle_sentinel<T>(row, column);
            if(referenced){
                update_error(factor_error,
                             static_cast<double>(std::abs(actual - reference)));
            }else{
                update_error(opposite_triangle_error,
                             static_cast<double>(std::abs(actual - reference)));
            }
        }
        for(int row = n; row < lda; ++row){
            update_error(
                padding_error,
                static_cast<double>(std::abs(
                    output[row + static_cast<std::size_t>(column) * lda]
                    - padding_sentinel<T>())));
        }
    }

    DEVICE_CHECK(deviceFreeAsync(d_A, handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_work, handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_info, handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

    const double tol = tolerance<T>(n);
    const bool passed = info == 0
                     && factor_error <= tol
                     && opposite_triangle_error == 0.0
                     && padding_error == 0.0;
    std::cout << (passed ? "PASS" : "FAIL")
              << " potrf_bottom_right_block<" << scalar_name<T>() << ">"
              << " uplo=" << uplo
              << " n=" << n
              << " info=" << info
              << " factor_error=" << factor_error
              << " tolerance=" << tol
              << " opposite_triangle_error=" << opposite_triangle_error
              << " padding_error=" << padding_error
              << std::endl;
    return passed;
}

template <typename T>
bool run_failure_case(
    char uplo, int n, int failed_pivot, const DdlaHandle_t& handle)
{
    constexpr int global_offset = 23;
    const int lda = n + 3;
    std::vector<T> input(
        static_cast<std::size_t>(lda) * n, padding_sentinel<T>());
    for(int column = 0; column < n; ++column){
        for(int row = 0; row < n; ++row){
            if(!in_referenced_triangle(uplo, row, column)){
                input[row + static_cast<std::size_t>(column) * lda] =
                    opposite_triangle_sentinel<T>(row, column);
            }else if(row == column){
                input[row + static_cast<std::size_t>(column) * lda] =
                    scalar<T>(row == failed_pivot ? -1.0 : 4.0);
            }else{
                input[row + static_cast<std::size_t>(column) * lda] = T{};
            }
        }
    }

    T* d_A = nullptr;
    T* d_work = nullptr;
    int* d_info = nullptr;
    DEVICE_CHECK(deviceMallocAsync(
        reinterpret_cast<void**>(&d_A), input.size() * sizeof(T),
        handle->stream));
    DEVICE_CHECK(deviceMallocAsync(
        reinterpret_cast<void**>(&d_work),
        static_cast<std::size_t>(n) * n * sizeof(T), handle->stream));
    DEVICE_CHECK(deviceMallocAsync(
        reinterpret_cast<void**>(&d_info), sizeof(int), handle->stream));
    DEVICE_CHECK(deviceMemcpyAsync(
        d_A, input.data(), input.size() * sizeof(T),
        deviceMemcpyHostToDevice, handle->stream));

    int info = -1;
    detail::potrf_bottom_right_block(
        uplo, n, d_A, lda, global_offset,
        d_work, d_info, info, handle);
    DEVICE_CHECK(deviceGetLastError());

    DEVICE_CHECK(deviceFreeAsync(d_A, handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_work, handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_info, handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

    const int expected_info = global_offset + failed_pivot + 1;
    const bool passed = info == expected_info;
    std::cout << (passed ? "PASS" : "FAIL")
              << " potrf_bottom_right_block<" << scalar_name<T>() << ">"
              << " uplo=" << uplo
              << " failed_pivot=" << failed_pivot + 1
              << " info=" << info
              << " expected_info=" << expected_info
              << std::endl;
    return passed;
}

template <typename T>
bool run_type(const DdlaHandle_t& handle)
{
    bool passed = true;
    for(const char uplo : {'U', 'L'}){
        for(const int n : {1, 7, 31, 32, 127, 128, 192, 256}){
            passed = run_success_case<T>(uplo, n, handle) && passed;
        }
        passed = run_failure_case<T>(uplo, 256, 0, handle) && passed;
        passed = run_failure_case<T>(uplo, 256, 255, handle) && passed;
    }
    return passed;
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int nprocs = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    if(nprocs != 1){
        std::cerr << "test_potrf_bottom_right_block requires exactly one MPI rank"
                  << std::endl;
        MPI_Finalize();
        return 2;
    }

    DdlaHandle_t handle = nullptr;
    ddla_init(handle);
    ddla_set(handle, MPI_COMM_WORLD);

    bool passed = true;
    passed = run_type<float>(handle) && passed;
    passed = run_type<double>(handle) && passed;
    passed = run_type<std::complex<float>>(handle) && passed;
    passed = run_type<std::complex<double>>(handle) && passed;

    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
    ddla_destroy(handle);
    MPI_Finalize();
    return passed ? 0 : 1;
}
