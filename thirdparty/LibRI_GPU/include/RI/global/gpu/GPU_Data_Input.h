#pragma once

#include "GPU_Data_Pack.h"
#include <omp.h>

namespace RI
{

namespace GPU_Data
{

template<typename TA, typename TAC, typename Tdata>
class Input
{
  public:
	Input()
	{
		this->h_data.resize(omp_get_max_threads());
		this->tensor_insert.resize(omp_get_max_threads(), false);
	}

	const Pack & insert(const TA &Aa, const TAC &Ab, const Tensor<Tdata> &tensor)
	{
		Pack &pack = this->ptrList[Aa][Ab];
		if(!pack.exist)
		{
			pack.exist = true;
			pack.pos = this->h_data[omp_get_thread_num()].size();
			this->tensor_insert[omp_get_thread_num()] = true;
			pack.shape = tensor.shape;
			pack.thread_num = omp_get_thread_num();
		}
		else
		{
			this->tensor_insert[omp_get_thread_num()] = false;
		}
		this->h_array.push_back(pack);
		return pack;
	}

	void insert_data(const Tensor<Tdata> &tensor)
	{
		if(this->tensor_insert[omp_get_thread_num()])
			this->h_data[omp_get_thread_num()].insert(
				this->h_data[omp_get_thread_num()].end(),
				tensor.ptr(),
				tensor.ptr() + tensor.shape.get_shape_all());
	}

	// 将对应的 C, V, D 放在 d_Cs, d_Vs, d_Ds 上
	void upload(magma_queue_t &queue)
	{
		std::vector<std::size_t> h_data_begin(this->h_data.size()+1, 0);
		for(std::size_t i=1; i<h_data_begin.size(); ++i)
			h_data_begin[i] = h_data_begin[i-1] + this->h_data[i-1].size();

		TESTING_CHECK(magma_malloc((void **)&this->d_data, h_data_begin.back() * sizeof(Tdata)));
		for(std::size_t i=0; i<this->h_data.size(); ++i)
			magma_setvector_async(this->h_data[i].size(), sizeof(Tdata), this->h_data[i].data(), 1, this->d_data+h_data_begin[i], 1, queue);

		const std::size_t batchCount = this->h_array.size();
		std::vector<Tdata*> d_array_(batchCount);							// 记录每个batch的 d_data 指针（CPU）
		for(std::size_t i=0; i<batchCount; ++i)
			d_array_[i] = this->d_data + this->h_array[i].pos + h_data_begin[this->h_array[i].thread_num];
		TESTING_CHECK(magma_malloc((void **)&this->d_array, batchCount * sizeof(Tdata *)));
		magma_setvector_async(batchCount, sizeof(Tdata*), d_array_.data(), 1, this->d_array, 1, queue);
	}

	~Input()
	{
		if(this->d_data)	TESTING_CHECK(magma_free(this->d_data));
		if(this->d_array)	TESTING_CHECK(magma_free(this->d_array));
	}

	std::vector<std::vector<Tdata>> h_data;		// 存储数据（CPU）
	Tdata *d_data = nullptr;					// 存储数据（GPU）
	Tdata **d_array = nullptr;					// 记录每个batch的 d_data 指针（GPU）
	std::vector<Pack> h_array;					// 记录每个batch的Pack
	std::map<TA, std::map<TAC, Pack>> ptrList;	// 记录每个原子对的Pack
	std::vector<bool> tensor_insert;			// 记录是否当前张量需要insert_data
};

}

}