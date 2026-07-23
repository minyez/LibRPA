/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah

*/
#include "gemm_vbatched_compat.h"
#include <stdexcept>
#define PRECISION_d
#include "gemm_template_kernel_vbatched.hpp"

#include "gemm_config/dgemm_param_nn.h"
#include "gemm_config/dgemm_param_nt.h"
#include "gemm_config/dgemm_param_tn.h"
#include "gemm_config/dgemm_param_tt.h"

#define version(s,v) s ## _V_ ## v

struct Config
{
    int DIM_X;
    int DIM_Y;
    int BLK_M;
    int BLK_N;
    int BLK_K;
    int dim_vec;
    int DIM_XA;
    int DIM_YA;
    int DIM_XB;
    int DIM_YB;
};

extern "C" Config
ddla_internal_dgemm_vbatched_config(
    ddla::deblasOperation_t transA, ddla::deblasOperation_t transB,
    int max_m, int max_n, int max_k)
{
    int shape = 0;
    if      (transA == ddla::DEBLAS_OP_N   && transB == ddla::DEBLAS_OP_N)   { shape = 0; } // nn
    else if (transA == ddla::DEBLAS_OP_N   && transB == ddla::DEBLAS_OP_T)     { shape = 1; } // nt
    else if (transA == ddla::DEBLAS_OP_N   && transB == ddla::DEBLAS_OP_C) { shape = 2; } // nc
    else if (transA == ddla::DEBLAS_OP_T     && transB == ddla::DEBLAS_OP_N)   { shape = 3; } // tn
    else if (transA == ddla::DEBLAS_OP_T     && transB == ddla::DEBLAS_OP_T)     { shape = 4; } // tt
    else if (transA == ddla::DEBLAS_OP_T     && transB == ddla::DEBLAS_OP_C) { shape = 5; } // tc
    else if (transA == ddla::DEBLAS_OP_C && transB == ddla::DEBLAS_OP_N)   { shape = 6; } // cn
    else if (transA == ddla::DEBLAS_OP_C && transB == ddla::DEBLAS_OP_T)     { shape = 7; } // ct
    else if (transA == ddla::DEBLAS_OP_C && transB == ddla::DEBLAS_OP_C) { shape = 8; } // cc

    switch(shape)
    {
        case 0: // nn
            {
                if(max_k < 32)
                {
                    if(max_k==8 && max_n==24)   return Config {version(NN,32)};
                    else if (max_n<32)          return Config {version(NN,49)};
                    else                        return Config {version(NN,111)};
                }
                else
                {
                    if(max_m < 80)	return Config {version(NN,93)};
                    else	        return Config {version(NN,111)};
                }
            }
            break;
        case 1: // nt
            {
                if(max_k < 128)	    return Config {version(NT,160)};
                else
                {
                    if(max_m < 256)	return Config {version(NT,160)};
                    else	        return Config {version(NT,190)};
                }
            }
            break;
        case 2: // nc
            {
                if(max_k < 128)	    return Config {version(NT,160)};
                else
                {
                    if(max_m < 256)	return Config {version(NT,160)};
                    else	        return Config {version(NT,190)};
                }
            }
            break;
        case 3: // tn
            {
                if(max_k < 64)	    return Config {version(TN,207)};
                else
                {
                    if(max_m < 256)	return Config {version(TN,207)};
                    else	        return Config {version(TN,209)};
                }
            }
            break;
        case 6: // cn
            {
                if(max_k < 64)	    return Config {version(TN,207)};
                else
                {
                    if(max_m < 256)	return Config {version(TN,207)};
                    else	        return Config {version(TN,209)};
                }
            }
            break;
        case 4: // tt
            {
                if(max_k < 128)	    return Config {version(TT,81)};
                else
                {
                    if(max_m < 256)	return Config {version(TT,81)};
                    else	        return Config {version(TT,85)};
                }
            }
            break;
        case 5: // tc
            {
                if(max_k < 128)	    return Config {version(TT,81)};
                else
                {
                    if(max_m < 256)	return Config {version(TT,81)};
                    else	        return Config {version(TT,85)};
                }
            }
            break;
        case 7: // ct
            {
                if(max_k < 128)	    return Config {version(TT,81)};
                else
                {
                    if(max_m < 256)	return Config {version(TT,81)};
                    else	        return Config {version(TT,85)};
                }
            }
            break;
        case 8: // cc
            {
                if(max_k < 128)	    return Config {version(TT,81)};
                else
                {
                    if(max_m < 256)	return Config {version(TT,81)};
                    else	        return Config {version(TT,85)};
                }
            }
            break;
        default:; // propose something
    }
    throw std::invalid_argument(
        "ddla_internal_dgemm_vbatched_config: invalid transpose operation");
}


extern "C" Func_gemm_template_vbatched_kernel<double>
ddla_internal_dgemm_vbatched_function(
    ddla::deblasOperation_t transA, ddla::deblasOperation_t transB,
    int max_m, int max_n, int max_k)
{
    int shape = 0;
    if      (transA == ddla::DEBLAS_OP_N   && transB == ddla::DEBLAS_OP_N)   { shape = 0; } // nn
    else if (transA == ddla::DEBLAS_OP_N   && transB == ddla::DEBLAS_OP_T)     { shape = 1; } // nt
    else if (transA == ddla::DEBLAS_OP_N   && transB == ddla::DEBLAS_OP_C) { shape = 2; } // nc
    else if (transA == ddla::DEBLAS_OP_T     && transB == ddla::DEBLAS_OP_N)   { shape = 3; } // tn
    else if (transA == ddla::DEBLAS_OP_T     && transB == ddla::DEBLAS_OP_T)     { shape = 4; } // tt
    else if (transA == ddla::DEBLAS_OP_T     && transB == ddla::DEBLAS_OP_C) { shape = 5; } // tc
    else if (transA == ddla::DEBLAS_OP_C && transB == ddla::DEBLAS_OP_N)   { shape = 6; } // cn
    else if (transA == ddla::DEBLAS_OP_C && transB == ddla::DEBLAS_OP_T)     { shape = 7; } // ct
    else if (transA == ddla::DEBLAS_OP_C && transB == ddla::DEBLAS_OP_C) { shape = 8; } // cc

    switch(shape)
    {
        case 0: // nn
            {
                if(max_k < 32)
                {
                    if(max_k==8 && max_n==24)   return gemm_template_vbatched_nn_kernel<double, version(NN,32), 0, 0>;
                    else if (max_n<32)          return gemm_template_vbatched_nn_kernel<double, version(NN,49), 0, 0>;
                    else                        return gemm_template_vbatched_nn_kernel<double, version(NN,111), 0, 0>;
                }
                else
                {
                    if(max_m < 80)	return gemm_template_vbatched_nn_kernel<double, version(NN,93), 0, 0>;
                    else	        return gemm_template_vbatched_nn_kernel<double, version(NN,111), 0, 0>;
                }
            }
            break;
        case 1: // nt
            {
                if(max_k < 128)	    return gemm_template_vbatched_nt_kernel<double, version(NT,160), 0, 0>;
                else
                {
                    if(max_m < 256)	return gemm_template_vbatched_nt_kernel<double, version(NT,160), 0, 0>;
                    else	        return gemm_template_vbatched_nt_kernel<double, version(NT,190), 0, 0>;
                }
            }
            break;
        case 2: // nc
            {
                if(max_k < 128)	    return gemm_template_vbatched_nt_kernel<double, version(NT,160), 0, 1>;
                else
                {
                    if(max_m < 256)	return gemm_template_vbatched_nt_kernel<double, version(NT,160), 0, 1>;
                    else	        return gemm_template_vbatched_nt_kernel<double, version(NT,190), 0, 1>;
                }
            }
            break;
        case 3: // tn
            {
                if(max_k < 64)	    return gemm_template_vbatched_tn_kernel<double, version(TN,207), 0, 0>;
                else
                {
                    if(max_m < 256)	return gemm_template_vbatched_tn_kernel<double, version(TN,207), 0, 0>;
                    else	        return gemm_template_vbatched_tn_kernel<double, version(TN,209), 0, 0>;
                }
            }
            break;
        case 6: // cn
            {
                if(max_k < 64)	    return gemm_template_vbatched_tn_kernel<double, version(TN,207), 1, 0>;
                else
                {
                    if(max_m < 256)	return gemm_template_vbatched_tn_kernel<double, version(TN,207), 1, 0>;
                    else	        return gemm_template_vbatched_tn_kernel<double, version(TN,209), 1, 0>;
                }
            }
            break;
        case 4: // tt
            {
                if(max_k < 128)	    return gemm_template_vbatched_tt_kernel<double, version(TT,81), 0, 0>;
                else
                {
                    if(max_m < 256)	return gemm_template_vbatched_tt_kernel<double, version(TT,81), 0, 0>;
                    else	        return gemm_template_vbatched_tt_kernel<double, version(TT,85), 0, 0>;
                }
            }
            break;
        case 5: // tc
            {
                if(max_k < 128)	    return gemm_template_vbatched_tt_kernel<double, version(TT,81), 0, 1>;
                else
                {
                    if(max_m < 256)	return gemm_template_vbatched_tt_kernel<double, version(TT,81), 0, 1>;
                    else	        return gemm_template_vbatched_tt_kernel<double, version(TT,85), 0, 1>;
                }
            }
            break;
        case 7: // ct
            {
                if(max_k < 128)	    return gemm_template_vbatched_tt_kernel<double, version(TT,81), 1, 0>;
                else
                {
                    if(max_m < 256)	return gemm_template_vbatched_tt_kernel<double, version(TT,81), 1, 0>;
                    else	        return gemm_template_vbatched_tt_kernel<double, version(TT,85), 1, 0>;
                }
            }
            break;
        case 8: // cc
            {
                if(max_k < 128)	    return gemm_template_vbatched_tt_kernel<double, version(TT,81), 1, 1>;
                else
                {
                    if(max_m < 256)	return gemm_template_vbatched_tt_kernel<double, version(TT,81), 1, 1>;
                    else	        return gemm_template_vbatched_tt_kernel<double, version(TT,85), 1, 1>;
                }
            }
            break;
        default:; // propose something
    }
    throw std::invalid_argument(
        "ddla_internal_dgemm_vbatched_function: invalid transpose operation");
}


/******************************************************************************/
extern "C" void
ddla_internal_dgemm_vbatched_core(
    ddla::deblasOperation_t transA, ddla::deblasOperation_t transB,
    int max_m, int max_n, int max_k,
    int* m, int* n, int* k,
    double alpha,
    double const * const * dA_array, int Ai, int Aj, int* ldda,
    double const * const * dB_array, int Bi, int Bj, int* lddb,
    double beta,
    double              ** dC_array, int Ci, int Cj, int* lddc,
    int batchCount, ddla::deviceStream_t stream )
{
    if (max_m <= 0 || max_n <= 0 || max_k < 0) return;

    Func_gemm_template_vbatched_kernel<double> func_gemm_template_vbatched_kernel
        = ddla_internal_dgemm_vbatched_function(transA, transB, max_m, max_n, max_k);

    Config config = ddla_internal_dgemm_vbatched_config(transA, transB, max_m, max_n, max_k);
    const int DIM_X=config.DIM_X, DIM_Y=config.DIM_Y, BLK_M=config.BLK_M, BLK_N=config.BLK_N, BLK_K=config.BLK_K;

    gemm_template_vbatched(func_gemm_template_vbatched_kernel,
        max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, batchCount, stream);
}


extern "C" void
ddla_internal_dgemm_vbatched_core_2s(
    ddla::deblasOperation_t transA_0, ddla::deblasOperation_t transB_0,
    int max_m_0, int max_n_0, int max_k_0,
    int* m_0, int* n_0, int* k_0,
    double alpha_0,
    double const * const * dA_array_0, int Ai_0, int Aj_0, int* ldda_0,
    double const * const * dB_array_0, int Bi_0, int Bj_0, int* lddb_0,
    double beta_0,
    double              ** dC_array_0, int Ci_0, int Cj_0, int* lddc_0,
    ddla::deblasOperation_t transA_1, ddla::deblasOperation_t transB_1,
    int max_m_1, int max_n_1, int max_k_1,
    int* m_1, int* n_1, int* k_1,
    double alpha_1,
    double const * const * dAB_array_1, int Ai_1, int Aj_1, int* ldda_1,
                                        int Bi_1, int Bj_1, int* lddb_1,
    double beta_1,
    double              ** dC_array_1, int Ci_1, int Cj_1, int* lddc_1,
    bool C0_left,
    int batchCount, int const * segment_sizes, ddla::deviceStream_t stream )
{
    if (max_m_0 <= 0 || max_n_0 <= 0 || max_k_0 < 0) return;

    Func_gemm_template_vbatched_kernel<double> func_gemm_template_vbatched_kernel_0 = ddla_internal_dgemm_vbatched_function(transA_0, transB_0, max_m_0, max_n_0, max_k_0);
    Func_gemm_template_vbatched_kernel<double> func_gemm_template_vbatched_kernel_1 = ddla_internal_dgemm_vbatched_function(transA_1, transB_1, max_m_1, max_n_1, max_k_1);

    Config config_0 = ddla_internal_dgemm_vbatched_config(transA_0, transB_0, max_m_0, max_n_0, max_k_0);
    Config config_1 = ddla_internal_dgemm_vbatched_config(transA_1, transB_1, max_m_1, max_n_1, max_k_1);

    gemm_template_vbatched_2s(
        func_gemm_template_vbatched_kernel_0,
        max_m_0, max_n_0, max_k_0, m_0, n_0, k_0, alpha_0, dA_array_0, Ai_0, Aj_0, ldda_0, dB_array_0, Bi_0, Bj_0, lddb_0, beta_0, dC_array_0, Ci_0, Cj_0, lddc_0, config_0.DIM_X, config_0.DIM_Y, config_0.BLK_M, config_0.BLK_N, config_0.BLK_K,
        func_gemm_template_vbatched_kernel_1,
        max_m_1, max_n_1, max_k_1, m_1, n_1, k_1, alpha_1, dAB_array_1, Ai_1, Aj_1, ldda_1,            Bi_1, Bj_1, lddb_1, beta_1, dC_array_1, Ci_1, Cj_1, lddc_1, config_1.DIM_X, config_1.DIM_Y, config_1.BLK_M, config_1.BLK_N, config_1.BLK_K,
        C0_left,
        batchCount, segment_sizes, stream);
}
