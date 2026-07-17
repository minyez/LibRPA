#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <mpi.h>

#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>

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
T padding_value()
{
    return scalar<T>(-911.0, 37.0);
}

template <typename T>
T opposite_triangle_value(int row, int column)
{
    return scalar<T>(
        -73.0 - 0.013 * row,
        19.0 + 0.017 * column);
}

bool in_referenced_triangle(char uplo, int row, int column)
{
    return uplo == 'U' ? row <= column : row >= column;
}

template <typename T>
std::vector<T> make_factor(char uplo, int n, int lda)
{
    std::vector<T> factor(static_cast<size_t>(lda) * n, T{});
    for(int j = 0; j < n; ++j){
        for(int i = 0; i < n; ++i){
            if(!in_referenced_triangle(uplo, i, j)){
                continue;
            }
            if(i == j){
                factor[i + static_cast<size_t>(j) * lda] =
                    scalar<T>(1.5 + 0.002 * (i % 23));
            }else{
                const double real = 0.004 * (((3 * i + 5 * j) % 11) - 5);
                const double imag = 0.003 * (((7 * i + 2 * j) % 9) - 4);
                factor[i + static_cast<size_t>(j) * lda] = scalar<T>(real, imag);
            }
        }
    }
    return factor;
}

template <typename T>
std::vector<T> form_product(
    char uplo, const std::vector<T>& factor, int n, int lda)
{
    std::vector<T> matrix(static_cast<size_t>(lda) * n, padding_value<T>());
    for(int j = 0; j < n; ++j){
        for(int i = 0; i < n; ++i){
            if(!in_referenced_triangle(uplo, i, j)){
                matrix[i + static_cast<size_t>(j) * lda] =
                    opposite_triangle_value<T>(i, j);
                continue;
            }
            T sum{};
            if(uplo == 'U'){
                for(int k = j; k < n; ++k){
                    sum += factor[i + static_cast<size_t>(k) * lda]
                         * conjugate(
                               factor[j + static_cast<size_t>(k) * lda]);
                }
            }else{
                for(int k = i; k < n; ++k){
                    sum += conjugate(
                               factor[k + static_cast<size_t>(i) * lda])
                         * factor[k + static_cast<size_t>(j) * lda];
                }
            }
            matrix[i + static_cast<size_t>(j) * lda] = sum;
        }
    }
    return matrix;
}

template <typename T>
double factor_tolerance(int n)
{
    return 128.0 * static_cast<double>(std::numeric_limits<real_type_t<T>>::epsilon())
         * std::max(1, n);
}

template <typename T>
double factor_error(
    char uplo, const std::vector<T>& actual, const std::vector<T>& expected,
    int n, int lda)
{
    double error = 0.0;
    for(int j = 0; j < n; ++j){
        for(int i = 0; i < n; ++i){
            if(!in_referenced_triangle(uplo, i, j)){
                continue;
            }
            error = std::max(
                error,
                static_cast<double>(std::abs(
                    actual[i + static_cast<size_t>(j) * lda]
                    - expected[i + static_cast<size_t>(j) * lda])));
        }
    }
    return error;
}

template <typename T>
double opposite_triangle_error(
    char uplo, const std::vector<T>& matrix, int n, int lda)
{
    double error = 0.0;
    for(int j = 0; j < n; ++j){
        for(int i = 0; i < n; ++i){
            if(in_referenced_triangle(uplo, i, j)){
                continue;
            }
            error = std::max(
                error,
                static_cast<double>(std::abs(
                    matrix[i + static_cast<size_t>(j) * lda]
                    - opposite_triangle_value<T>(i, j))));
        }
    }
    return error;
}

template <typename T>
double padding_error(const std::vector<T>& matrix, int n, int lda)
{
    double error = 0.0;
    const T expected = padding_value<T>();
    for(int j = 0; j < n; ++j){
        for(int i = n; i < lda; ++i){
            error = std::max(
                error,
                static_cast<double>(std::abs(
                    matrix[i + static_cast<size_t>(j) * lda] - expected)));
        }
    }
    return error;
}

template <typename T>
bool run_success_case(char uplo, int n, const ddla::DdlaHandle_t& handle)
{
    const int lda = n + 3;
    const auto expected = make_factor<T>(uplo, n, lda);
    const auto input = form_product(uplo, expected, n, lda);
    const size_t bytes = input.size() * sizeof(T);

    T* d_A = nullptr;
    DEVICE_CHECK(deviceMallocAsync(
        reinterpret_cast<void**>(&d_A), bytes, handle->stream));
    DEVICE_CHECK(deviceMemcpyAsync(
        d_A, input.data(), bytes, deviceMemcpyHostToDevice, handle->stream));

    int info = -1;
    ddla::potrf_bottom_right(uplo, n, d_A, lda, info, handle);
    DEVICE_CHECK(deviceGetLastError());

    std::vector<T> actual(input.size());
    DEVICE_CHECK(deviceMemcpyAsync(
        actual.data(), d_A, bytes, deviceMemcpyDeviceToHost, handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_A, handle->stream));

    const double error = factor_error(uplo, actual, expected, n, lda);
    const double triangle_error =
        opposite_triangle_error(uplo, actual, n, lda);
    const double pad_error = padding_error(actual, n, lda);
    const double tolerance = factor_tolerance<T>(n);
    const bool passed = info == 0
                     && error <= tolerance
                     && triangle_error == 0.0
                     && pad_error == 0.0;
    std::cout << (passed ? "PASS" : "FAIL")
              << " potrf_bottom_right<" << scalar_name<T>() << ">"
              << " uplo=" << uplo
              << " n=" << n << " lda=" << lda
              << " info=" << info
              << " factor_error=" << error
              << " tolerance=" << tolerance
              << " opposite_triangle_error=" << triangle_error
              << " padding_error=" << pad_error << std::endl;
    return passed;
}

template <typename T>
std::vector<T> make_failure_matrix(
    char uplo, int n, int lda, int failed_pivot)
{
    std::vector<T> matrix(static_cast<size_t>(lda) * n, padding_value<T>());
    for(int j = 0; j < n; ++j){
        for(int i = 0; i < n; ++i){
            matrix[i + static_cast<size_t>(j) * lda] =
                in_referenced_triangle(uplo, i, j)
                    ? T{}
                    : opposite_triangle_value<T>(i, j);
        }
        matrix[j + static_cast<size_t>(j) * lda] = scalar<T>(4.0);
    }
    matrix[failed_pivot + static_cast<size_t>(failed_pivot) * lda] = scalar<T>(-1.0);
    return matrix;
}

template <typename T>
bool run_failure_case(
    char uplo, int failed_pivot, const ddla::DdlaHandle_t& handle)
{
    constexpr int n = 129;
    constexpr int lda = n + 5;
    const auto input = make_failure_matrix<T>(uplo, n, lda, failed_pivot);
    const size_t bytes = input.size() * sizeof(T);

    T* d_A = nullptr;
    DEVICE_CHECK(deviceMallocAsync(
        reinterpret_cast<void**>(&d_A), bytes, handle->stream));
    DEVICE_CHECK(deviceMemcpyAsync(
        d_A, input.data(), bytes, deviceMemcpyHostToDevice, handle->stream));

    int info = -1;
    ddla::potrf_bottom_right(uplo, n, d_A, lda, info, handle);
    DEVICE_CHECK(deviceGetLastError());
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_A, handle->stream));

    const int expected_info = failed_pivot + 1;
    const bool passed = info == expected_info;
    std::cout << (passed ? "PASS" : "FAIL")
              << " potrf_bottom_right<" << scalar_name<T>() << ">"
              << " uplo=" << uplo
              << " failed_pivot=" << failed_pivot + 1
              << " info=" << info
              << " expected_info=" << expected_info << std::endl;
    return passed;
}

template <typename T>
bool run_type(const ddla::DdlaHandle_t& handle)
{
    bool passed = true;
    for(const char uplo : {'U', 'L'}){
        for(const int n : {1, 70, 127, 128, 129, 257}){
            const bool case_passed = run_success_case<T>(uplo, n, handle);
            passed = case_passed && passed;
        }
        passed = run_failure_case<T>(uplo, 128, handle) && passed;
        passed = run_failure_case<T>(uplo, 0, handle) && passed;
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
        std::cerr << "test_potrf_bottom_right requires exactly one MPI rank" << std::endl;
        MPI_Finalize();
        return 2;
    }

    ddla::DdlaHandle_t handle = nullptr;
    ddla::ddla_init(handle);
    ddla::ddla_set(handle, MPI_COMM_WORLD);

    bool passed = true;
    passed = run_type<float>(handle) && passed;
    passed = run_type<double>(handle) && passed;
    passed = run_type<std::complex<float>>(handle) && passed;
    passed = run_type<std::complex<double>>(handle) && passed;

    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
    ddla::ddla_destroy(handle);
    MPI_Finalize();
    return passed ? 0 : 1;
}
