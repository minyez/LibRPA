// ===================
//  Author: Laiyuan Yang
//  date: 2026.2.1
// ===================

#pragma once

#include "Magmablas_Interface.h"

namespace RI
{
	template <typename T>
	inline void magmablas_gemm_vbatched(
		magma_trans_t transA, magma_trans_t transB,
		magma_int_t *m, magma_int_t *n, magma_int_t *k,
		T alpha,
		T const *const *dA_array,
		T const *const *dB_array,
		T beta,
		T **dC_array,
		magma_int_t batchCount, magma_queue_t queue)
	{
		magma_int_t*& ldda = (transA==MagmaNoTrans) ? k : m;
		magma_int_t*& lddb = (transB==MagmaNoTrans) ? n : k;
		magma_int_t*& lddc = n;
		magmablas_gemm_vbatched(
			transB, transA,
			n, m, k,
			alpha,
			dB_array, lddb,
			dA_array, ldda,
			beta,
			dC_array, lddc,
			batchCount, queue);
	}

	template <typename T>
	inline void magmablas_gemm_vbatched_2(
		magma_trans_t transA_0, magma_trans_t transB_0,
		magma_int_t *m_0, magma_int_t *n_0, magma_int_t *k_0,
		T alpha_0,
		T const *const *dA_array_0,
		T const *const *dB_array_0,
		T beta_0,
		T **dC_array_0,
		magma_trans_t transA_1, magma_trans_t transB_1,
		magma_int_t *m_1, magma_int_t *n_1, magma_int_t *k_1,
		T alpha_1,
		T const *const *dAB_array_1,
		T beta_1,
		T **dC_array_1,
		bool C0_left,
		magma_int_t batchCount, magma_int_t max_batchCount, magma_queue_t queue)
	{
		magma_int_t*& ldda_0 = (transA_0==MagmaNoTrans) ? k_0 : m_0;
		magma_int_t*& lddb_0 = (transB_0==MagmaNoTrans) ? n_0 : k_0;
		magma_int_t*& lddc_0 = n_0;
		magma_int_t*& ldda_1 = (transA_1==MagmaNoTrans) ? k_1 : m_1;
		magma_int_t*& lddb_1 = (transB_1==MagmaNoTrans) ? n_1 : k_1;
		magma_int_t*& lddc_1 = n_1;
		magmablas_gemm_vbatched_2(
			transB_0, transA_0,
			n_0, m_0, k_0,
			alpha_0,
			dB_array_0, lddb_0,
			dA_array_0, ldda_0,
			beta_0,
			dC_array_0, lddc_0,
			transB_1, transA_1,
			n_1, m_1, k_1,
			alpha_1,
			dAB_array_1, 
			lddb_1, ldda_1,
			beta_1,
			dC_array_1, lddc_1,
			!C0_left,
			batchCount, max_batchCount, queue);
	}

	template <typename T>
	inline void magmablas_gemm_vbatched_2s(
		const magma_trans_t transA_0, const magma_trans_t transB_0,
		magma_int_t*const m_0, magma_int_t*const n_0, magma_int_t*const k_0,
		const T alpha_0, const T*const*const dA_array_0,
		                 const T*const*const dB_array_0,
		const T beta_0,        T*     *const dC_array_0,
		magma_trans_t transA_1, magma_trans_t transB_1,
		magma_int_t*const m_1, magma_int_t*const n_1, magma_int_t*const k_1,
		const T alpha_1, const T*const*const dAB_array_1,
		const T beta_1,        T*     *const dC_array_1,
		const bool C0_left,
		const magma_int_t batchCount, const magma_int_t*const max_batchCount, magma_queue_t queue)
	{
		magma_int_t*const ldda_0 = (transA_0==MagmaNoTrans) ? k_0 : m_0;
		magma_int_t*const lddb_0 = (transB_0==MagmaNoTrans) ? n_0 : k_0;
		magma_int_t*const lddc_0 = n_0;
		magma_int_t*const ldda_1 = (transA_1==MagmaNoTrans) ? k_1 : m_1;
		magma_int_t*const lddb_1 = (transB_1==MagmaNoTrans) ? n_1 : k_1;
		magma_int_t*const lddc_1 = n_1;
		magmablas_gemm_vbatched_2s(
			transB_0, transA_0,
			n_0, m_0, k_0,
			alpha_0,
			dB_array_0, lddb_0,
			dA_array_0, ldda_0,
			beta_0,
			dC_array_0, lddc_0,
			transB_1, transA_1,
			n_1, m_1, k_1,
			alpha_1,
			dAB_array_1, 
			lddb_1, ldda_1,
			beta_1,
			dC_array_1, lddc_1,
			!C0_left,
			batchCount, max_batchCount, queue);
	}
}