#include <cassert>
#include <functional>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include "../gpu/la_connector.h"
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
#include <ddla/ddla_connector.h>
#endif
#include "elpa_connector.h"
#ifdef LIBRPA_USE_LIBRI
#include <RI/global/Tensor.h>
#else
#include "../utils/libri_stub.h"
#endif
#include "../math/scalapack_connector.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../utils/profiler.h"
#include "../core/utils_atomic_basis_blacs.h"
#include <vector>

namespace librpa_int
{

namespace ElpaConnector
{

/*!
 * @brief Compute power of Hermitian matrix using BLACS
 *
 * @param  [in,out]  A_local       Process-local part of global matrix A to be powered
 * @param  [in]      ad_A          Array descriptor of A
 * @param  [out]     Z_local       Process-local part of the eigenvector matrix (Z) of A
 * @param  [in]      ad_Z          Array descriptor of Z
 * @param  [out]     n_filtered    Array descriptor of Z
 * @param  [out]     W             Eigenvalues of A, including those smaller than threshold
 * @param  [in]      power         Power to perform
 * @param  [in]      threshold     The threshold to filter the eigenvalues
 *
 * @retval           scale_Z       Eigenvectors scaled by the power of eigenvalues, using ad_Z
 */
template <typename T>
matrix_m<std::complex<T>> power_hemat_blacs(matrix_m<std::complex<T>> &A_local,
                                            const ArrayDesc &ad_A,
                                            matrix_m<std::complex<T>> &Z_local,
                                            const ArrayDesc &ad_Z,
                                            size_t &n_filtered, T *W, T power,
                                            const T &threshold)
{
    using global::ofs_myid;
    using global::profiler;

    profiler.start(__FUNCTION__);

    assert (A_local.is_col_major() && Z_local.is_col_major());
    const bool is_int_power = fabs(power - int(power)) < 1e-4;
    const size_t n = ad_A.m();
    const char jobz = 'V';
    const char uplo = 'U';

    // temporary array for heev with optmized block size
    const int blocksize_row_opt = std::min(ad_A.mb(), 128);
    const int blocksize_col_opt = std::min(ad_A.nb(), 128);

    // initialize descriptor of array A for optimized block size
    ArrayDesc ad_A_opt(ad_A.ictxt());
    ad_A_opt.init(n, n, blocksize_row_opt, blocksize_col_opt, 0, 0);
    auto A_local_opt = init_local_mat<std::complex<T>>(ad_A_opt, MAJOR::COL);
    // NOTE: imply A and Z should be in the same context
    ScalapackConnector::pgemr2d_f(n, n, A_local.ptr(), 1, 1, ad_A.desc,
                                  A_local_opt.ptr(), 1, 1, ad_A_opt.desc, ad_A.ictxt());

    // initialize descriptor of array Z for optimized block size
    if (ad_A.ictxt() != ad_Z.ictxt())
    {
        ofs_myid << "Warning(power_hemat_blacs): input contexts of A and Z are different!" << std::endl;
    }
    ArrayDesc ad_Z_opt(ad_Z.ictxt());
    ad_Z_opt.init(n, n, blocksize_row_opt, blocksize_col_opt, 0, 0);
    auto Z_local_opt = init_local_mat<std::complex<T>>(ad_Z_opt, MAJOR::COL);
    // printf("Z_local_opt size: %d\n", Z_local_opt.size());

    global::profiler.start("power_hemat_blacs_1");
#ifndef ENABLE_ELPA
    int lwork = -1, lrwork = -1, info = 0;
    std::complex<T> *work;
    T *rwork;
    {
        work  = new std::complex<T>[1];
        rwork = new T[1];
        // query the optimal lwork and lrwork
        T *Wquery = new T[1];
        // librpa_int::global::lib_printf("power_hemat_blacs descA %s\n", ad_A.info_desc().c_str());
        ScalapackConnector::pheev_f(jobz, uplo,
                n, A_local_opt.ptr(), 1, 1, ad_A_opt.desc,
                Wquery, Z_local_opt.ptr(), 1, 1, ad_A_opt.desc, work, lwork, rwork, lrwork, info);
        lwork = std::max(int(work[0].real()), 1);
        lrwork = std::max(int(rwork[0]), 1);
        delete [] work;
        delete [] Wquery;
        delete [] rwork;
    }
#endif
    global::profiler.stop("power_hemat_blacs_1");

    global::profiler.start("power_hemat_blacs_2");
    std::complex<T> *A, *Z;
    T* W_uni;
#ifndef ENABLE_ELPA
    work = new std::complex<T> [lwork];
    rwork = new T [lrwork];
    ScalapackConnector::pheev_f(jobz, uplo,
            n, A_local_opt.ptr(), 1, 1, ad_A_opt.desc,
            W, Z_local_opt.ptr(), 1, 1, ad_Z_opt.desc, work, lwork, rwork, lrwork, info);
    delete [] work;
    delete [] rwork;
#else
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    auto ddla_handle = ad_A.ddla_desc().ddla_handle();
    ad_Z_opt.set_ddla_desc(ddla_handle);
    ad_A_opt.set_ddla_desc(ddla_handle);
    std::complex<T>* d_A, * d_Z;
    T* d_W;
    ddla::DEVICE_CHECK(deviceMallocAsync((void**)&d_A, A_local_opt.size() * sizeof(std::complex<T>), ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceMallocAsync((void**)&d_Z, Z_local_opt.size() * sizeof(std::complex<T>), ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceMallocAsync((void**)&d_W, n * sizeof(T), ddla_handle->stream));

    ddla::DEVICE_CHECK(deviceMemcpyAsync(d_A, A_local_opt.ptr(), A_local_opt.size() * sizeof(std::complex<T>), ddla::deviceMemcpyHostToDevice, ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceMemcpyAsync(d_Z, Z_local_opt.ptr(), Z_local_opt.size() * sizeof(std::complex<T>), ddla::deviceMemcpyHostToDevice, ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceMemcpyAsync(d_W, W, n * sizeof(T), ddla::deviceMemcpyHostToDevice, ddla_handle->stream));
    A = d_A;
    Z = d_Z;
    W_uni = d_W;
#else
    A = A_local_opt.ptr();
    Z = Z_local_opt.ptr();
    W_uni = W;
#endif
    int error;
    elpa_eigenvectors(ad_A.elpa_handle(), A, W_uni, Z, &error);
    if(error != ELPA_OK){
        throw std::runtime_error("elpa eigenvectors error\n");
    }
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    ddla::DEVICE_CHECK(deviceMemcpyAsync(W, W_uni, n * sizeof(T), ddla::deviceMemcpyDeviceToHost, ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceMemcpyAsync(Z_local_opt.ptr(), Z, Z_local_opt.size()*sizeof(std::complex<T>), ddla::deviceMemcpyDeviceToHost, ddla_handle->stream));
    ddla::DEVICE_CHECK(ddla::deviceStreamSynchronize(ddla_handle->stream));
#endif
#endif
    global::profiler.stop("power_hemat_blacs_2");
    // Optimized A no longer used
    profiler.start("power_hemat_blacs_3");
    A_local_opt.clear();
    // send back the eigenvector matrix
    ScalapackConnector::pgemr2d_f(n, n, Z_local_opt.ptr(), 1, 1, ad_Z_opt.desc,
                                  Z_local.ptr(), 1, 1, ad_Z.desc, ad_Z.ictxt());
    profiler.stop("power_hemat_blacs_3");

    // check the number of non-singular eigenvalues,
    // using the fact that W is in ascending order
    n_filtered = n;
    for (size_t i = 0; i != n; i++)
        if (W[i] >= threshold)
        {
            n_filtered = i;
            break;
        }

    // filter and scale the eigenvalues, store in a temp array
    profiler.start("power_hemat_blacs_4");
    std::vector<T> W_temp(n);
    for (size_t i = 0; i < n_filtered; i++) W_temp[i] = 0.0;
    for (size_t i = n_filtered; i != n; i++)
    {
        if (W[i] < 0 && !is_int_power)
        {
            global::lib_printf(
                "Warning! unfiltered negative eigenvalue with non-integer power: # %d ev = %f , "
                "pow = %f\n",
                i, W[i], power);
        }
        if (fabs(W[i]) < 1e-10 && power < 0)
        {
            global::lib_printf(
                "Warning! unfiltered nearly-singular eigenvalue with negative power: # %d ev = %f "
                ", pow = %f\n",
                i, W[i], power);
        }
        W_temp[i] = std::pow(W[i], power);
    }
    profiler.stop("power_hemat_blacs_4");
    // debug print
    // for (int i = 0; i != n; i++)
    // {
    //     librpa_int::global::lib_printf("%d %f %f\n", i, W[i], W_temp[i]);
    // }

    profiler.start("power_hemat_blacs_5");
    // create scaled eigenvectors
    auto scaled_opt = Z_local_opt.copy();
    std::complex<T> *C;
    #if defined(ENABLE_ELPA) && (defined(ENABLE_HIP) || defined(ENABLE_CUDA))
    std::complex<T>* d_C;
    ddla::DEVICE_CHECK(deviceMallocAsync((void**)&d_C, Z_local_opt.size() * sizeof(std::complex<T>), ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceMemcpyAsync(d_C, Z_local_opt.ptr(), Z_local_opt.size() * sizeof(std::complex<T>), ddla::deviceMemcpyHostToDevice, ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceMemcpyAsync(d_A, d_Z, Z_local_opt.size() * sizeof(std::complex<T>), ddla::deviceMemcpyDeviceToDevice, ddla_handle->stream));
    C = d_C;
    #else 
    Z = Z_local_opt.ptr();
    A = scaled_opt.ptr();
    C = A_local.ptr();
    #endif
    for (int i = 0; i != n; i++)
    {
        // ScalapackConnector::pscal_f(n, W_temp[i], scaled_opt.ptr(), 1, 1 + i, ad_Z_opt.desc, 1);
        int j_loc = ad_Z_opt.indx_g2l_c(i);
        if(j_loc>=0)
            LaConnector::scal(ad_Z_opt.m_loc(), W_temp[i], A + ad_Z_opt.lld() * j_loc, 1, ad_Z_opt);
    }
    LaConnector::pgemm('N', 'C', n, n, n, {1.0, 0.0}, Z, 1, 1, ad_Z_opt, A, 1, 1, ad_Z_opt, {0.0, 0.0}, C, 1, 1, ad_A);
#if defined(ENABLE_ELPA) && (defined(ENABLE_HIP) || defined(ENABLE_CUDA))
    ddla::DEVICE_CHECK(deviceMemcpyAsync(scaled_opt.ptr(), A, scaled_opt.size() * sizeof(std::complex<T>), ddla::deviceMemcpyDeviceToHost, ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceMemcpyAsync(A_local.ptr(), C, A_local.size() * sizeof(std::complex<T>), ddla::deviceMemcpyDeviceToHost, ddla_handle->stream));
    ddla::DEVICE_CHECK(ddla::deviceStreamSynchronize(ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceFreeAsync(d_C, ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceFreeAsync(d_A, ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceFreeAsync(d_Z, ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceFreeAsync(d_W, ddla_handle->stream));
#endif
    auto scaled = Z_local.copy();
    // send back the scaled eigenvector matrix with descriptor using optimized block size to that
    // with input descriptor
    ScalapackConnector::pgemr2d_f(n, n, scaled_opt.ptr(), 1, 1, ad_Z_opt.desc, scaled.ptr(), 1, 1,
                                  ad_Z.desc, ad_Z.ictxt());
    profiler.stop("power_hemat_blacs_5");
    profiler.stop(__FUNCTION__);

    return scaled;
}


template <typename T>
matrix_m<std::complex<T>> power_hemat_blacs_real(matrix_m<std::complex<T>> &A_local,
                                                 const ArrayDesc &ad_A,
                                                 matrix_m<std::complex<T>> &Z_local,
                                                 const ArrayDesc &ad_Z, size_t &n_filtered,
                                                 T *W, T power, const T &threshold)
{
    using global::ofs_myid;
    using global::profiler;

    profiler.start("use elpa to sqrt matrix");
    profiler.start(__FUNCTION__);
    // Step 1: Extract real part of the complex matrix
    auto A_local_real = A_local.get_real();
    A_local_real *= -1.0;
    assert(A_local.is_col_major() && Z_local.is_col_major());
    const bool is_int_power = fabs(power - int(power)) < 1e-4;
    const int n = ad_A.m();
    

    // temporary array for syev with optimized block size
    const int blocksize_row_opt = std::min(ad_A.mb(), 128);
    const int blocksize_col_opt = std::min(ad_A.nb(), 128);

    // initialize descriptor of array A for optimized block size
    ArrayDesc ad_A_opt(ad_A.ictxt());
    ad_A_opt.init(n, n, blocksize_row_opt, blocksize_col_opt, 0, 0);
    // Initialize as real matrix for diagonalization
    auto A_local_opt = init_local_mat<T>(ad_A_opt, MAJOR::COL);
    ScalapackConnector::pgemr2d_f(n, n, A_local_real.ptr(), 1, 1, ad_A.desc, A_local_opt.ptr(), 1,
                                  1, ad_A_opt.desc, ad_A.ictxt());

    // initialize descriptor of array Z for optimized block size
    if (ad_A.ictxt() != ad_Z.ictxt())
    {
        ofs_myid << "Warning(power_hemat_blacs): input contexts of A and Z are different!\n";
    }
    ArrayDesc ad_Z_opt(ad_Z.ictxt());
    ad_Z_opt.init(n, n, blocksize_row_opt, blocksize_col_opt, 0, 0);
    // Initialize Z as real matrix initially
    auto Z_local_opt = init_local_mat<T>(ad_Z_opt, MAJOR::COL);

    profiler.start("power_hemat_blacs_1");
#ifndef ENABLE_ELPA
    const char jobz = 'V';
    const char uplo = 'U';
    int lwork = -1, lrwork = -1, info = 0;

    // Query optimal workspace using the provided psyev_f interface
    T work_query, rwork_query, Wquery;
    ScalapackConnector::psyev_f(jobz, uplo, n, A_local_opt.ptr(), 1, 1, ad_A_opt.desc, &Wquery,
                                Z_local_opt.ptr(), 1, 1, ad_A_opt.desc, &work_query, lwork, &rwork_query,
                                lrwork, info);
    lwork = std::max(as_int(work_query), 1);
    lrwork = std::max(as_int(rwork_query), 1);
    std::vector<T> work(lwork);
    std::vector<T> rwork(lrwork);
#endif
    profiler.stop("power_hemat_blacs_1");
    
    profiler.start("power_hemat_blacs_2");
    T *A, *Z, *W_uni;
#ifndef ENABLE_ELPA
    // Perform real symmetric diagonalization using the provided interface
    ofs_myid << lwork << " " << lrwork << " " << n << std::endl;
    ScalapackConnector::psyev_f(jobz, uplo, n, A_local_opt.ptr(), 1, 1, ad_A_opt.desc, W,
                                Z_local_opt.ptr(), 1, 1, ad_Z_opt.desc, work.data(), lwork, rwork.data(), lrwork,
                                info);
    work.clear();
    rwork.clear();
#else
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    auto ddla_handle = ad_A.ddla_desc().ddla_handle();
    ad_Z_opt.set_ddla_desc(ddla_handle);
    ad_A_opt.set_ddla_desc(ddla_handle);
    T* d_A, *d_Z, *d_W;
    ddla::DEVICE_CHECK(deviceMallocAsync((void**)&d_A, A_local_opt.size() * sizeof(T), ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceMallocAsync((void**)&d_Z, Z_local_opt.size() * sizeof(T), ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceMallocAsync((void**)&d_W, n * sizeof(T), ddla_handle->stream));

    ddla::DEVICE_CHECK(deviceMemcpyAsync(d_A, A_local_opt.ptr(), A_local_opt.size() * sizeof(T), ddla::deviceMemcpyHostToDevice, ddla_handle->stream));
    A = d_A;
    Z = d_Z;
    W_uni = d_W;
#else
    A = A_local_opt.ptr();
    Z = Z_local_opt.ptr();
    W_uni = W;
#endif
    int error;
    elpa_eigenvectors(ad_A.elpa_handle(), A, W_uni, Z, &error);
    if(error != ELPA_OK){
        throw std::runtime_error("elpa eigenvectors error\n");
    }
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    ddla::DEVICE_CHECK(deviceMemcpyAsync(W, W_uni, n * sizeof(T), ddla::deviceMemcpyDeviceToHost, ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceMemcpyAsync(Z_local_opt.ptr(), Z, Z_local_opt.size()*sizeof(T), ddla::deviceMemcpyDeviceToHost, ddla_handle->stream));
    ddla::DEVICE_CHECK(ddla::deviceStreamSynchronize(ddla_handle->stream));
#endif
#endif
    for (int i = 0; i < n; i++)
    {
        W[i] *= -1.0;
    }
    // A_local_opt.clear();

    // Convert real eigenvectors to complex (with zero imaginary part)
    auto Z_local_opt_complex = Z_local_opt.to_complex();
    // Transfer using complex matrix descriptor
    ScalapackConnector::pgemr2d_f(n, n, Z_local_opt_complex.ptr(), 1, 1, ad_Z_opt.desc,
                                  Z_local.ptr(), 1, 1, ad_Z.desc, ad_Z.ictxt());
    profiler.stop("power_hemat_blacs_2");

    // std::cout << "Z_local_opt:" << std::endl << Z_local_opt << std::endl;

    // Check number of non-singular eigenvalues
    n_filtered = n;
    for (int i = 0; i < n; i++)
    {
        if (W[n - 1 - i] >= threshold)
        {
            n_filtered = i;
            break;
        }
    }

    // Filter and scale eigenvalues
    profiler.start("power_hemat_blacs_3");
    std::vector<T> W_temp(n, T(0));
    for (int i = 0; i < n - n_filtered; i++)
    {
        if (W[i] < 0 && !is_int_power)
        {
            global::lib_printf(
                "Warning! unfiltered negative eigenvalue with non-integer power: # %d ev = %f , "
                "pow = %f\n",
                i, W[i], power);
        }
        if (fabs(W[i]) < 1e-10 && power < 0)
        {
            global::lib_printf(
                "Warning! unfiltered nearly-singular eigenvalue with negative power: # %d ev = %f "
                ", pow = %f\n",
                i, W[i], power);
        }
        W_temp[i] = std::pow(W[i], power);
    }
    profiler.stop("power_hemat_blacs_3");

    profiler.start("power_hemat_blacs_4");
    // Scale eigenvectors using complex matrix operations
    // auto scaled_opt = Z_local_opt_complex.copy();
    auto scaled_opt = Z_local_opt.copy();
    T* C;
#if defined(ENABLE_ELPA) && (defined(ENABLE_HIP) || defined(ENABLE_CUDA))
    T* d_C;
    ddla::DEVICE_CHECK(deviceMallocAsync((void**)&d_C, Z_local_opt.size() * sizeof(T), ddla_handle->stream));
    // ddla::DEVICE_CHECK(deviceMemcpyAsync(d_C, Z_local_opt.ptr(), Z_local_opt.size() * sizeof(T), ddla::deviceMemcpyHostToDevice, ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceMemcpyAsync(d_A, d_Z, Z_local_opt.size() * sizeof(T), ddla::deviceMemcpyDeviceToDevice, ddla_handle->stream));
    C = d_C;
#else
    Z = Z_local_opt.ptr();
    A = scaled_opt.ptr();
    C = A_local_opt.ptr();
#endif
    // printf("before scal\n");
    for (int i = 0; i < n; i++)
    {
        // Use ScaLAPACK scaling function with complex matrix
        // ScalapackConnector::pscal_f(n, W_temp[i], scaled_opt.ptr(), 1, 1 + i, ad_Z_opt.desc, 1);
        int j_loc = ad_Z_opt.indx_g2l_c(i);
        if(j_loc>=0)
            LaConnector::scal(ad_Z_opt.m_loc(), W_temp[i], A + ad_Z_opt.lld() * j_loc, 1, ad_Z_opt);
    }

    // Compute Z * diag(W_temp) * Z^H
    // ScalapackConnector::pgemm_f('N', 'C', n, n, n, 1.0, Z_local_opt_complex.ptr(), 1, 1,
    //                             ad_Z_opt.desc, scaled_opt.ptr(), 1, 1, ad_Z_opt.desc, 0.0,
    //                             A_local.ptr(), 1, 1, ad_A.desc);
    // printf("before pgemm\n");
    LaConnector::pgemm('N', 'C', n, n, n, 1.0, Z, 1, 1, ad_Z_opt, A, 1, 1, ad_Z_opt, 0.0, C, 1, 1, ad_A);
#if defined(ENABLE_ELPA) && (defined(ENABLE_HIP) || defined(ENABLE_CUDA))
    ddla::DEVICE_CHECK(deviceMemcpyAsync(scaled_opt.ptr(), A, scaled_opt.size() * sizeof(T), ddla::deviceMemcpyDeviceToHost, ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceMemcpyAsync(A_local_opt.ptr(), C, A_local.size() * sizeof(T), ddla::deviceMemcpyDeviceToHost, ddla_handle->stream));
    ddla::DEVICE_CHECK(ddla::deviceStreamSynchronize(ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceFreeAsync(d_C, ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceFreeAsync(d_A, ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceFreeAsync(d_Z, ddla_handle->stream));
    ddla::DEVICE_CHECK(deviceFreeAsync(d_W, ddla_handle->stream));
#endif
    A_local = A_local_opt.to_complex();
    auto scaled = Z_local.copy();

    Z_local_opt_complex = scaled_opt.to_complex();
    // Transfer back to original complex matrix
    ScalapackConnector::pgemr2d_f(n, n, Z_local_opt_complex.ptr(), 1, 1, ad_Z_opt.desc, scaled.ptr(), 1, 1,
                                  ad_Z.desc, ad_Z.ictxt());
    profiler.stop("power_hemat_blacs_4");
    profiler.stop(__FUNCTION__);
    profiler.stop("use elpa to sqrt matrix");

    return scaled;
}

} // namespace ElpaConnector


} // namespace librpa_int