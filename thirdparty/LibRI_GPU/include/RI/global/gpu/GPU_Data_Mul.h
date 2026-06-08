#pragma once

#include "GPU_Data_Pack.h"

namespace RI
{

namespace GPU_Data
{

template<typename TA, typename TAC, typename Tdata>
class Mul
{
  public:
	const Pack &insert_1st(const TA &Aa, const TAC &Ab, const Shape_Vector &shape)
	{
		Pack &pack = this->ptrList[Aa][Ab];
		if(!pack.exist)
		{
			pack.exist = true;
			pack.pos = this->totalSize;
			pack.shape = shape;
			this->totalSize += shape.get_shape_all();
		}
		this->h_array_1.push_back(pack);
		return pack;
	}

	void upload_1st(magma_queue_t queue)
	{
		TESTING_CHECK(magma_malloc((void **)&this->d_data, this->totalSize * sizeof(Tdata)));
		GPU_Wrapper::GPUMemset(this->d_data, 0, totalSize * sizeof(Tdata)); // 初始化

		const std::size_t batchCount = this->h_array_1.size();
		std::vector<Tdata*> d_array_(batchCount);							// 记录每个batch的 d_data 指针（CPU）
		for (std::size_t i = 0; i < batchCount; i++)
			d_array_[i] = this->d_data + this->h_array_1[i].pos;
		TESTING_CHECK(magma_malloc((void **)&this->d_array_1, batchCount * sizeof(Tdata *)));
		magma_setvector_async(batchCount, sizeof(Tdata*), d_array_.data(), 1, this->d_array_1, 1, queue);
	}

	const Pack &find_2nd(const TA &Aa, const TAC &Ab)
	{
		return Global_Func::find(this->ptrList, Aa, Ab);
	}

	const Pack &insert_2nd(const Pack &pack)
	{
		this->h_array_2.push_back(pack);
		return this->h_array_2.back();
	}

	// 在 GPU 上分配空间，并获取指针
	void upload_2nd(magma_queue_t queue)
	{
		const std::size_t batchCount = this->h_array_2.size();
		std::vector<Tdata*> d_array_2_(batchCount);							// 记录每个batch的 d_data 指针（CPU）
		for (std::size_t i = 0; i < batchCount; i++)
			d_array_2_[i] = this->d_data + this->h_array_2[i].pos;
		TESTING_CHECK(magma_malloc((void **)&this->d_array_2, batchCount * sizeof(Tdata *)));
		magma_setvector_async(batchCount, sizeof(Tdata*), d_array_2_.data(), 1, this->d_array_2, 1, queue);
	}

	~Mul()
	{
		if(this->d_data)	TESTING_CHECK(magma_free(this->d_data));
		if(this->d_array_1)	TESTING_CHECK(magma_free(this->d_array_1));
		if(this->d_array_2)	TESTING_CHECK(magma_free(this->d_array_2));
	}

	std::size_t totalSize = 0;                  // 总的数据数量
	Tdata *d_data = nullptr;					// 存储数据（GPU）
	std::vector<Pack> h_array_1;				// 记录每个batch的Pack
	std::vector<Pack> h_array_2;				// 记录每个batch的Pack
	Tdata **d_array_1 = nullptr;				// 记录每个batch的 d_data 指针（GPU）
	Tdata **d_array_2 = nullptr;				// 记录每个batch的 d_data 指针（GPU）
	std::map<TA, std::map<TAC, Pack>> ptrList;	// 记录每个原子对的Pack
};

}

}