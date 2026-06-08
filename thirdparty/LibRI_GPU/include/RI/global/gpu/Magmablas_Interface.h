// ===================
//  Author: Laiyuan Yang
//  date: 2026.2.1
// ===================

#pragma once

#include <magma.h>

namespace RI
{
	inline void magmablas_gemm_vbatched_2(
		magma_trans_t transA_0, magma_trans_t transB_0,
		magma_int_t *m_0, magma_int_t *n_0, magma_int_t *k_0,
		float alpha_0,
		float const *const *dA_array_0, magma_int_t *ldda_0,
		float const *const *dB_array_0, magma_int_t *lddb_0,
		float beta_0,
		float **dC_array_0, magma_int_t *lddc_0,
		magma_trans_t transA_1, magma_trans_t transB_1,
		magma_int_t *m_1, magma_int_t *n_1, magma_int_t *k_1,
		float alpha_1,
		float const *const *dAB_array_1, 
		magma_int_t *ldda_1, magma_int_t *lddb_1,
		float beta_1,
		float **dC_array_1, magma_int_t *lddc_1,
		bool C0_left,
		magma_int_t batchCount, magma_int_t max_batchCount, magma_queue_t queue)
	{
		magmablas_sgemm_vbatched_2(
			transA_0, transB_0,
			m_0, n_0, k_0,
			alpha_0,
			dA_array_0, ldda_0,
			dB_array_0, lddb_0,
			beta_0,
			dC_array_0, lddc_0,
			transA_1, transB_1,
			m_1, n_1, k_1,
			alpha_1,
			dAB_array_1, 
			ldda_1, lddb_1,
			beta_1,
			dC_array_1, lddc_1,
			C0_left,
			batchCount, max_batchCount, queue);
	}

	inline void magmablas_gemm_vbatched_2(
		magma_trans_t transA_0, magma_trans_t transB_0,
		magma_int_t *m_0, magma_int_t *n_0, magma_int_t *k_0,
		double alpha_0,
		double const *const *dA_array_0, magma_int_t *ldda_0,
		double const *const *dB_array_0, magma_int_t *lddb_0,
		double beta_0,
		double **dC_array_0, magma_int_t *lddc_0,
		magma_trans_t transA_1, magma_trans_t transB_1,
		magma_int_t *m_1, magma_int_t *n_1, magma_int_t *k_1,
		double alpha_1,
		double const *const *dAB_array_1, 
		magma_int_t *ldda_1, magma_int_t *lddb_1,
		double beta_1,
		double **dC_array_1, magma_int_t *lddc_1,
		bool C0_left,
		magma_int_t batchCount, magma_int_t max_batchCount, magma_queue_t queue)
	{
		magmablas_dgemm_vbatched_2(
			transA_0, transB_0, 
			m_0, n_0, k_0, 
			alpha_0, 
			dA_array_0, ldda_0, 
			dB_array_0, lddb_0, 
			beta_0, 
			dC_array_0, lddc_0,
			transA_1, transB_1,
			m_1, n_1, k_1,
			alpha_1,
			dAB_array_1, 
			ldda_1, lddb_1,
			beta_1,
			dC_array_1, lddc_1,
			C0_left,
			batchCount, max_batchCount, queue);
	}

	inline void magmablas_gemm_vbatched_2(
		magma_trans_t transA_0, magma_trans_t transB_0,
		magma_int_t *m_0, magma_int_t *n_0, magma_int_t *k_0,
		magmaFloatComplex alpha_0,
		magmaFloatComplex const *const *dA_array_0, magma_int_t *ldda_0,
		magmaFloatComplex const *const *dB_array_0, magma_int_t *lddb_0,
		magmaFloatComplex beta_0,
		magmaFloatComplex **dC_array_0, magma_int_t *lddc_0,
		magma_trans_t transA_1, magma_trans_t transB_1,
		magma_int_t *m_1, magma_int_t *n_1, magma_int_t *k_1,
		magmaFloatComplex alpha_1,
		magmaFloatComplex const *const *dAB_array_1, 
		magma_int_t *ldda_1, magma_int_t *lddb_1,
		magmaFloatComplex beta_1,
		magmaFloatComplex **dC_array_1, magma_int_t *lddc_1,
		bool C0_left,
		magma_int_t batchCount, magma_int_t max_batchCount, magma_queue_t queue)
	{
		magmablas_cgemm_vbatched_2(
			transA_0, transB_0,
			m_0, n_0, k_0,
			alpha_0,
			dA_array_0, ldda_0,
			dB_array_0, lddb_0,
			beta_0,
			dC_array_0, lddc_0,
			transA_1, transB_1,
			m_1, n_1, k_1,
			alpha_1,
			dAB_array_1, 
			ldda_1, lddb_1,
			beta_1,
			dC_array_1, lddc_1,
			C0_left,
			batchCount, max_batchCount, queue);
	}

	inline void magmablas_gemm_vbatched_2(
		magma_trans_t transA_0, magma_trans_t transB_0,
		magma_int_t *m_0, magma_int_t *n_0, magma_int_t *k_0,
		magmaDoubleComplex alpha_0,
		magmaDoubleComplex const *const *dA_array_0, magma_int_t *ldda_0,
		magmaDoubleComplex const *const *dB_array_0, magma_int_t *lddb_0,
		magmaDoubleComplex beta_0,
		magmaDoubleComplex **dC_array_0, magma_int_t *lddc_0,
		magma_trans_t transA_1, magma_trans_t transB_1,
		magma_int_t *m_1, magma_int_t *n_1, magma_int_t *k_1,
		magmaDoubleComplex alpha_1,
		magmaDoubleComplex const *const *dAB_array_1, 
		magma_int_t *ldda_1, magma_int_t *lddb_1,
		magmaDoubleComplex beta_1,
		magmaDoubleComplex **dC_array_1, magma_int_t *lddc_1,
		bool C0_left,
		magma_int_t batchCount, magma_int_t max_batchCount, magma_queue_t queue)
	{
		magmablas_zgemm_vbatched_2(
			transA_0, transB_0,
			m_0, n_0, k_0,
			alpha_0,
			dA_array_0, ldda_0,
			dB_array_0, lddb_0,
			beta_0,
			dC_array_0, lddc_0,
			transA_1, transB_1,
			m_1, n_1, k_1,
			alpha_1,
			dAB_array_1, 
			ldda_1, lddb_1,
			beta_1,
			dC_array_1, lddc_1,
			C0_left,
			batchCount, max_batchCount, queue);
	}

	inline void magmablas_gemm_vbatched_2s(
		const magma_trans_t transA_0, const magma_trans_t transB_0,
		magma_int_t*const m_0, magma_int_t*const n_0, magma_int_t*const k_0,
		const float alpha_0, const float*const*const dA_array_0, magma_int_t*const ldda_0,
		                     const float*const*const dB_array_0, magma_int_t*const lddb_0,
		const float beta_0,        float*     *const dC_array_0, magma_int_t*const lddc_0,
		magma_trans_t transA_1, magma_trans_t transB_1,
		magma_int_t*const m_1, magma_int_t*const n_1, magma_int_t*const k_1,
		const float alpha_1, const float*const*const dAB_array_1, 
		                                                          magma_int_t*const ldda_1,
															      magma_int_t*const lddb_1,
		const float beta_1,        float*     *const dC_array_1,  magma_int_t*const lddc_1,
		const bool C0_left,
		const magma_int_t batchCount, const magma_int_t*const max_batchCount, magma_queue_t queue)
	{
		magmablas_sgemm_vbatched_2s(
			transA_0, transB_0,
			m_0, n_0, k_0,
			alpha_0,
			dA_array_0, ldda_0,
			dB_array_0, lddb_0,
			beta_0,
			dC_array_0, lddc_0,
			transA_1, transB_1,
			m_1, n_1, k_1,
			alpha_1,
			dAB_array_1, 
			ldda_1, lddb_1,
			beta_1,
			dC_array_1, lddc_1,
			C0_left,
			batchCount, max_batchCount, queue);
	}

	inline void magmablas_gemm_vbatched_2s(
		const magma_trans_t transA_0, const magma_trans_t transB_0,
		magma_int_t*const m_0, magma_int_t*const n_0, magma_int_t*const k_0,
		const double alpha_0, const double*const*const dA_array_0, magma_int_t*const ldda_0,
		                      const double*const*const dB_array_0, magma_int_t*const lddb_0,
		const double beta_0,        double*     *const dC_array_0, magma_int_t*const lddc_0,
		magma_trans_t transA_1, magma_trans_t transB_1,
		magma_int_t*const m_1, magma_int_t*const n_1, magma_int_t*const k_1,
		const double alpha_1, const double*const*const dAB_array_1, 
		                                                            magma_int_t*const ldda_1,
															        magma_int_t*const lddb_1,
		const double beta_1,        double*     *const dC_array_1,  magma_int_t*const lddc_1,
		const bool C0_left,
		const magma_int_t batchCount, const magma_int_t*const max_batchCount, magma_queue_t queue)
	{
		magmablas_dgemm_vbatched_2s(
			transA_0, transB_0, 
			m_0, n_0, k_0, 
			alpha_0, 
			dA_array_0, ldda_0, 
			dB_array_0, lddb_0, 
			beta_0, 
			dC_array_0, lddc_0,
			transA_1, transB_1,
			m_1, n_1, k_1,
			alpha_1,
			dAB_array_1, 
			ldda_1, lddb_1,
			beta_1,
			dC_array_1, lddc_1,
			C0_left,
			batchCount, max_batchCount, queue);
	}

	inline void magmablas_gemm_vbatched_2s(
		const magma_trans_t transA_0, const magma_trans_t transB_0,
		magma_int_t*const m_0, magma_int_t*const n_0, magma_int_t*const k_0,
		const magmaFloatComplex alpha_0, const magmaFloatComplex*const*const dA_array_0, magma_int_t*const ldda_0,
		                                 const magmaFloatComplex*const*const dB_array_0, magma_int_t*const lddb_0,
		const magmaFloatComplex beta_0,        magmaFloatComplex*     *const dC_array_0, magma_int_t*const lddc_0,
		magma_trans_t transA_1, magma_trans_t transB_1,
		magma_int_t*const m_1, magma_int_t*const n_1, magma_int_t*const k_1,
		const magmaFloatComplex alpha_1, const magmaFloatComplex*const*const dAB_array_1, 
		                                                                                  magma_int_t*const ldda_1,
															                              magma_int_t*const lddb_1,
		const magmaFloatComplex beta_1,        magmaFloatComplex*     *const dC_array_1,  magma_int_t*const lddc_1,
		const bool C0_left,
		const magma_int_t batchCount, const magma_int_t*const max_batchCount, magma_queue_t queue)
	{
		magmablas_cgemm_vbatched_2s(
			transA_0, transB_0,
			m_0, n_0, k_0,
			alpha_0,
			dA_array_0, ldda_0,
			dB_array_0, lddb_0,
			beta_0,
			dC_array_0, lddc_0,
			transA_1, transB_1,
			m_1, n_1, k_1,
			alpha_1,
			dAB_array_1, 
			ldda_1, lddb_1,
			beta_1,
			dC_array_1, lddc_1,
			C0_left,
			batchCount, max_batchCount, queue);
	}

	inline void magmablas_gemm_vbatched_2s(
		const magma_trans_t transA_0, const magma_trans_t transB_0,
		magma_int_t*const m_0, magma_int_t*const n_0, magma_int_t*const k_0,
		const magmaDoubleComplex alpha_0, const magmaDoubleComplex*const*const dA_array_0, magma_int_t*const ldda_0,
		                                  const magmaDoubleComplex*const*const dB_array_0, magma_int_t*const lddb_0,
		const magmaDoubleComplex beta_0,        magmaDoubleComplex*     *const dC_array_0, magma_int_t*const lddc_0,
		magma_trans_t transA_1, magma_trans_t transB_1,
		magma_int_t*const m_1, magma_int_t*const n_1, magma_int_t*const k_1,
		const magmaDoubleComplex alpha_1, const magmaDoubleComplex*const*const dAB_array_1, 
		                                                                                    magma_int_t*const ldda_1,
															                                magma_int_t*const lddb_1,
		const magmaDoubleComplex beta_1,        magmaDoubleComplex*     *const dC_array_1,  magma_int_t*const lddc_1,
		const bool C0_left,
		const magma_int_t batchCount, const magma_int_t*const max_batchCount, magma_queue_t queue)
	{
		magmablas_zgemm_vbatched_2s(
			transA_0, transB_0,
			m_0, n_0, k_0,
			alpha_0,
			dA_array_0, ldda_0,
			dB_array_0, lddb_0,
			beta_0,
			dC_array_0, lddc_0,
			transA_1, transB_1,
			m_1, n_1, k_1,
			alpha_1,
			dAB_array_1, 
			ldda_1, lddb_1,
			beta_1,
			dC_array_1, lddc_1,
			C0_left,
			batchCount, max_batchCount, queue);
	}

	inline void magmablas_gemm_vbatched(
		magma_trans_t transA, magma_trans_t transB,
		magma_int_t *m, magma_int_t *n, magma_int_t *k,
		float alpha,
		float const *const *dA_array, magma_int_t *ldda,
		float const *const *dB_array, magma_int_t *lddb,
		float beta,
		float **dC_array, magma_int_t *lddc,
		magma_int_t batchCount, magma_queue_t queue)
	{
		magmablas_sgemm_vbatched(
			transA, transB,
			m, n, k,
			alpha,
			dA_array, ldda,
			dB_array, lddb,
			beta,
			dC_array, lddc,
			batchCount, queue);
	}

	inline void magmablas_gemm_vbatched(
		magma_trans_t transA, magma_trans_t transB,
		magma_int_t *m, magma_int_t *n, magma_int_t *k,
		double alpha,
		double const *const *dA_array, magma_int_t *ldda,
		double const *const *dB_array, magma_int_t *lddb,
		double beta,
		double **dC_array, magma_int_t *lddc,
		magma_int_t batchCount, magma_queue_t queue)
	{
		magmablas_dgemm_vbatched(
			transA, transB,
			m, n, k,
			alpha,
			dA_array, ldda,
			dB_array, lddb,
			beta,
			dC_array, lddc,
			batchCount, queue);
	}

	inline void magmablas_gemm_vbatched(
		magma_trans_t transA, magma_trans_t transB,
		magma_int_t *m, magma_int_t *n, magma_int_t *k,
		magmaFloatComplex alpha,
		magmaFloatComplex const *const *dA_array, magma_int_t *ldda,
		magmaFloatComplex const *const *dB_array, magma_int_t *lddb,
		magmaFloatComplex beta,
		magmaFloatComplex **dC_array, magma_int_t *lddc,
		magma_int_t batchCount, magma_queue_t queue)
	{
		magmablas_cgemm_vbatched(
			transA, transB,
			m, n, k,
			alpha,
			dA_array, ldda,
			dB_array, lddb,
			beta,
			dC_array, lddc,
			batchCount, queue);
	}

	inline void magmablas_gemm_vbatched(
		magma_trans_t transA, magma_trans_t transB,
		magma_int_t *m, magma_int_t *n, magma_int_t *k,
		magmaDoubleComplex alpha,
		magmaDoubleComplex const *const *dA_array, magma_int_t *ldda,
		magmaDoubleComplex const *const *dB_array, magma_int_t *lddb,
		magmaDoubleComplex beta,
		magmaDoubleComplex **dC_array, magma_int_t *lddc,
		magma_int_t batchCount, magma_queue_t queue)
	{
		magmablas_zgemm_vbatched(
			transA, transB,
			m, n, k,
			alpha,
			dA_array, ldda,
			dB_array, lddb,
			beta,
			dC_array, lddc,
			batchCount, queue);
	}
}