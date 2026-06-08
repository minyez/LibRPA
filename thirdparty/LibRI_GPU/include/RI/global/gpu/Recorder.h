#pragma once

#include "GPU_Wrapper.h"

namespace RI
{
    /*
    h_Cs，d_Cs：	这两个是用来存储数据的；
    h_C_array：		记录 h_Cs 位置的，
    d_C_array_：	记录 d_Cs 指针的 (magma_malloc_cpu)，
    d_C_array：		这个是用来提交计算任务的（magma_malloc)；
    CptrList：		用来剔除重复的；
    CfirstAppearance:记录 C 在 h_Cs 初次出现的位置；
    shapeCs：		记录矩阵大小
    */
    // 记录 C, V, D 相关的指针
    template <typename TA, typename TC, typename Tdata>
    struct Recorder
    {
        using TAC = std::pair<TA, TC>;
        using Ttensor = RI::Tensor<Tdata>;

    public:

        // 将对应的 C, V, D 放在 h_Cs, h_Vs, h_Ds 上
        void input(const Ttensor &tensor, const std::pair<TA, TAC> &pos)
        {
            const auto it = ptrList.find(pos);
            if (it == ptrList.end())
            {
                // 不在，则添加
                const std::size_t position = ptrList[pos] = h_data.size();
                h_data.insert(h_data.end(), tensor.ptr(), tensor.ptr() + tensor.shape.get_shape_all());
                h_array.push_back(position);
            }
            else
                h_array.push_back(it->second);
        }

        // 将对应的 C, V, D 放在 d_Cs, d_Vs, d_Ds 上
        void upload(int batchCount, magma_queue_t queue)
        {
            // std::cout << h_data.size() << std::endl;
            TESTING_CHECK(magma_malloc((void **)&d_data, h_data.size() * sizeof(Tdata)));
            TESTING_CHECK(magma_malloc_cpu((void **)&d_array_, batchCount * sizeof(Tdata *)));
            TESTING_CHECK(magma_malloc((void **)&d_array, batchCount * sizeof(Tdata *)));
            magma_setvector_async(h_data.size(), sizeof(Tdata), h_data.data(), 1, d_data, 1, queue);
            // TESTING_CHECK(magma_setvector_async(h_data.size(), sizeof(Tdata), h_data.data(), 1, d_data, 1, queue));
            // array 的填充
            for (int i = 0; i < batchCount; i++)
            {
                d_array_[i] = d_data + h_array[i];
            }
            magma_setvector_async(batchCount, sizeof(double *), d_array_, 1, d_array, 1, queue);
        }

    protected:
        std::vector<Tdata> h_data;                         // 存储数据（CPU）
        Tdata *d_data;                                     // 存储数据（GPU）
        std::vector<std::size_t> h_array;                  // 记录 h_data 位置
        Tdata **d_array_;                                  // 记录 d_data 指针（CPU）
        std::map<std::pair<TA, TAC>, std::size_t> ptrList; // 去重，记录小矩阵位置
    public:
        Tdata **d_array;                                   // 记录 d_data 指针（GPU）
    };
} // namespace RI