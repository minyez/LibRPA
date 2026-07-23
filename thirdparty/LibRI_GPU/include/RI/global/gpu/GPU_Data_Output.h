#pragma once

#include "GPU_Backend.h"
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

	void upload(GPU_Backend::Queue queue)
	{
		GPU_Backend::allocate(&this->d_data, this->totalSize);
		GPU_Backend::memset(
			this->d_data, 0, totalSize * sizeof(Tdata), queue);

		const std::size_t batchCount = this->h_array.size();
		std::vector<Tdata*> d_array_(batchCount);							// 记录每个batch的 d_data 指针（CPU）
		for (std::size_t i = 0; i < batchCount; i++)
			d_array_[i] = this->d_data + this->h_array[i].pos;
		GPU_Backend::allocate(&this->d_array, batchCount);
		GPU_Backend::upload(batchCount, d_array_.data(), this->d_array, queue);
	}

	void download(
		std::map<TA, std::map<TAC, Tensor<Tdata>>> &Ds_result,
		GPU_Backend::Queue queue) const
	{
		std::vector<Tdata> h_data(this->totalSize);
		GPU_Backend::download(
			this->totalSize, this->d_data, h_data.data(), queue);
		GPU_Backend::sync(queue);
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
		GPU_Backend::free(this->d_data);
		GPU_Backend::free(this->d_array);
	}

	std::size_t totalSize = 0;                  // 总的数据数量
	Tdata *d_data = nullptr;					// 存储数据（GPU）
	std::vector<Pack> h_array;					// 记录每个batch的Pack
	Tdata **d_array = nullptr;					// 记录每个batch的 d_data 指针（GPU）
	std::map<TA, std::map<TAC, Pack>> ptrList;	// 记录每个原子对的Pack
};

}

}
