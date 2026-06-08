#pragma once

#include "GPU_Wrapper.h"
#include <RI/global/Map_Operator.h>
#include <numeric>

namespace RI
{
    // 记录 X, H, Xtmp 数据
    template <typename TA, typename TAC, typename Tdata>
    class dynArr
    {
    public:
        // ~dynArr()
        // {
        //     magma_free(data);
        //     magma_free(d_array);
        //     magma_free_cpu(d_array_);
        //     h_array.clear();
        // }

        // 检查 map 中是否已经存在，并记录其偏移量
        void findInMap(const TA &index1, const TAC &index2, std::vector<std::size_t> &matrixShapes)
        {
            if (RI::Global_Func::find(mapExist, index1, index2))
            {
                // 已经存在
                h_array.push_back(mapBias[index1][index2]);
            }
            else
            {
                // 不存在，则插入 matrixSize 个 0，并记录偏移量
                mapExist[index1][index2] = true;
                shapes[index1][index2] = matrixShapes;
                mapBias[index1][index2] = totalSize;
                h_array.push_back(totalSize);
                totalSize += std::accumulate(matrixShapes.begin(), matrixShapes.end(), 1, std::multiplies<std::size_t>());
            }
        }

        // 在 GPU 上分配空间，并获取指针
        void allocate(int batchCount, magma_queue_t queue)
        {
            assert(batchCount == h_array.size());
            TESTING_CHECK(magma_malloc((void **)&data, totalSize * sizeof(Tdata)));
            TESTING_CHECK(magma_malloc_cpu((void **)&d_array_, batchCount * sizeof(Tdata *)));
            TESTING_CHECK(magma_malloc((void **)&d_array, batchCount * sizeof(Tdata *)));
            GPU_Wrapper::GPUMemset(data, 0, totalSize * sizeof(Tdata)); // 初始化
            for (std::size_t i = 0; i < batchCount; i++)
            {
                d_array_[i] = data + h_array[i];
            }
            magma_setvector_async(batchCount, sizeof(Tdata *), d_array_, 1, d_array, 1, queue);
        }

        // 将数组内的所有数据全部输出到 output_data 中
        void returnData(Tdata *output_data, magma_queue_t queue)
        {
            magma_getvector_async(totalSize, sizeof(Tdata), data, 1, output_data, 1, queue);
        }

        // 返回总的数据数量
        std::size_t getTotalSize()
        {
            return totalSize;
        }

    protected:
        std::size_t totalSize = 0;                  // 总的数据数量
        Tdata *data;                                // 数据（GPU）
        std::vector<std::size_t> h_array;           // 记录偏移量
        Tdata **d_array_;                           // 记录指针（CPU）
        std::map<TA, std::map<TAC, bool>> mapExist; // 记录指针是否存在 //todo: 有望被优化掉，但我又不想让它甚至mapBias成为全局变量
    public:
        std::map<TA, std::map<TAC, std::vector<std::size_t>>> shapes; // 记录维度
        Tdata **d_array;                                              // 记录指针（GPU）
        std::map<TA, std::map<TAC, std::size_t>> mapBias; // 记录指针位置以及它的偏移量
    };

    // 为 Xtmp 设计
    template <typename TA, typename TAC, typename Tdata>
    class dynArr_Xtmp : public dynArr<TA, TAC, Tdata>
    {
    public:
        dynArr_Xtmp(const magma_int_t maxBatchCount_in)
            : maxBatchCount(maxBatchCount_in) {}
        // ~dynArr_Xtmp()
        // {
        //     magma_free(data);
        //     magma_free(d_array);
        //     magma_free_cpu(d_array_);
        //     h_array.clear();
        // }

        // 在数组末尾插入 matrixSize 个 0，并记录偏移量
        // Xtmp 的 h_array 是重复指向一处的
        void insertZeros(std::size_t matrixSize, magma_int_t currentBatchCount)
        {
            if (currentBatchCount < maxBatchCount)
            {
                // 没有达到上限，直接插入
                h_array.push_back(totalSize);
                totalSize += matrixSize;
            }
            else
            {
                // 达到上限，重复利用空间
                h_array.push_back(h_array[currentBatchCount % maxBatchCount]);
            }
        }

        // 在 GPU 上分配空间，并返回指针
        // Xtmp 的 data 不需要提前初始化的
        void allocate(int batchCount, magma_queue_t queue)
        {
            assert(batchCount == h_array.size());
            TESTING_CHECK(magma_malloc((void **)&data, totalSize * sizeof(Tdata)));
            TESTING_CHECK(magma_malloc_cpu((void **)&d_array_, batchCount * sizeof(Tdata *)));
            TESTING_CHECK(magma_malloc((void **)&d_array, batchCount * sizeof(Tdata *)));
            for (std::size_t i = 0; i < batchCount; i++)
            {
                d_array_[i] = data + h_array[i];
            }
            magma_setvector_async(batchCount, sizeof(Tdata *), d_array_, 1, d_array, 1, queue);
        }

    protected:
        using dynArr<TA, TAC, Tdata>::totalSize;
        using dynArr<TA, TAC, Tdata>::h_array;
        using dynArr<TA, TAC, Tdata>::data;
        using dynArr<TA, TAC, Tdata>::d_array_;
        // GPU 一次性同时处理的 batch 是有上限的
        magma_int_t maxBatchCount = 65534;

    public:
        using dynArr<TA, TAC, Tdata>::d_array;
        using dynArr<TA, TAC, Tdata>::shapes; // 记录维度
    };

    // 为 X 设计
    template <typename TA, typename TAC, typename Tdata>
    class dynArr_X : public dynArr<TA, TAC, Tdata>
    {
    public:
        // 为计算 H 时的第二次筛选设计的
        bool findInMap2(const TA &index1, const TAC &index2) const
        {
            return RI::Global_Func::find(mapExist, index1, index2);
        }

        void h_array2_push_back(const TA &index1, const TAC &index2)
        {
            h_array2.push_back(mapBias[index1][index2]);
        }

        // 在 GPU 上分配空间，并获取指针
        void allocate2(int batchCount, magma_queue_t queue)
        {
            assert(batchCount == h_array2.size());
            TESTING_CHECK(magma_malloc_cpu((void **)&d_array2_, batchCount * sizeof(Tdata *)));
            TESTING_CHECK(magma_malloc((void **)&d_array2, batchCount * sizeof(Tdata *)));
            for (std::size_t i = 0; i < batchCount; i++)
            {
                d_array2_[i] = data + h_array2[i];
            }
            magma_setvector_async(batchCount, sizeof(Tdata *), d_array2_, 1, d_array2, 1, queue);
        }

    protected:
        using dynArr<TA, TAC, Tdata>::totalSize;
        using dynArr<TA, TAC, Tdata>::h_array;
        using dynArr<TA, TAC, Tdata>::mapExist;
        using dynArr<TA, TAC, Tdata>::mapBias;
        using dynArr<TA, TAC, Tdata>::data;
        //* 一定要为 X 专门写一个 array2。因为第二次利用 X 的时候，h_array 与 h_array2 不一定是相同的
        std::vector<std::size_t> h_array2; // 记录偏移量
        Tdata **d_array2_;                 // 记录指针（CPU）
    public:
        Tdata **d_array2; // 记录指针（GPU）
        using dynArr<TA, TAC, Tdata>::shapes; // 记录维度
    };
}