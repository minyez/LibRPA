#pragma once

#include "GPU_Data_Pack.h"

namespace RI
{

namespace GPU_Data
{

template<typename TA, typename TAC, typename Tdata>
class Output
{
  public:
	const Pack &insert(const TA &Aa, const TAC &Ab, const Shape_Vector &shape)
	{
		Pack &pack = this->ptrList[Aa][Ab];
		if(!pack.exist)
		{
			pack.exist = true;
			pack.pos = this->totalSize;
			pack.shape = shape;
			this->totalSize += shape.get_shape_all();
		}
		this->h_array.push_back(pack);
		return pack;
	}

	void upload(magma_queue_t &queue)
	{
		TESTING_CHECK(magma_malloc((void **)&this->d_data, this->totalSize * sizeof(Tdata)));
		GPU_Wrapper::GPUMemset(this->d_data, 0, totalSize * sizeof(Tdata)); // 初始化

		const std::size_t batchCount = this->h_array.size();
		std::vector<Tdata*> d_array_(batchCount);							// 记录每个batch的 d_data 指针（CPU）
		for (std::size_t i = 0; i < batchCount; i++)
			d_array_[i] = this->d_data + this->h_array[i].pos;
		TESTING_CHECK(magma_malloc((void **)&this->d_array, batchCount * sizeof(Tdata *)));
		magma_setvector_async(batchCount, sizeof(Tdata*), d_array_.data(), 1, this->d_array, 1, queue);
	}

	void download(std::map<TA, std::map<TAC, Tensor<Tdata>>> &Ds_result, const magma_queue_t queue) const
	{
		std::vector<Tdata> h_data(this->totalSize);
		magma_getvector_async(this->totalSize, sizeof(Tdata), this->d_data, 1, h_data.data(), 1, queue);
		magma_queue_sync(queue);
		for(const auto &ptrList_A : this->ptrList)
		{
			for(const auto &ptrList_B : ptrList_A.second)
			{
				const Pack &pack = ptrList_B.second;
				Tensor<Tdata> D_output(pack.shape);
				std::memcpy(D_output.ptr(), h_data.data() + pack.pos, pack.shape.get_shape_all()*sizeof(Tdata));

				Tensor<Tdata> &D_result = Ds_result[ptrList_A.first][ptrList_B.first];
				if(D_result.empty())
					D_result = std::move(D_output);
				else
					D_result += D_output;
			}
		}
	}

	~Output()
	{
		if(this->d_data)	TESTING_CHECK(magma_free(this->d_data));
		if(this->d_array)	TESTING_CHECK(magma_free(this->d_array));
	}

	std::size_t totalSize = 0;                  // 总的数据数量
	Tdata *d_data = nullptr;					// 存储数据（GPU）
	std::vector<Pack> h_array;					// 记录每个batch的Pack
	Tdata **d_array = nullptr;					// 记录每个batch的 d_data 指针（GPU）
	std::map<TA, std::map<TAC, Pack>> ptrList;	// 记录每个原子对的Pack
};

}

}