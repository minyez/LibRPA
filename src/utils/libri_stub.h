/*
 * @file      libri_stub.h
 * @brief     Stubbing classes and functions for LibRI
 * @author    Min-Ye Zhang
 * @date      2024-07-08
 */
#pragma once
#ifndef LIBRPA_USE_LIBRI
#include <vector>
#include <memory>
#include <set>
#include <map>
#include <valarray>
#include <cassert>
#include <utility>
#include <mpi.h>

namespace RI
{

template <typename Tdata>
class Tensor
{
private:
    std::vector<std::size_t> shape;

    static std::size_t shape_size_(const std::vector<std::size_t> &shape_in)
    {
        std::size_t size = 1;
        for (const auto dim : shape_in)
            size *= dim;
        return shape_in.empty() ? 0 : size;
    }

    static std::vector<std::size_t> normalize_shape_(const std::vector<int> &dimension)
    {
        std::vector<std::size_t> shape_out;
        shape_out.reserve(dimension.size());
        for (const auto dim : dimension)
        {
            assert(dim >= 0);
            shape_out.push_back(static_cast<std::size_t>(dim));
        }
        return shape_out;
    }

    void reset_(std::vector<std::size_t> shape_in, std::shared_ptr<std::valarray<Tdata>> data_in)
    {
        assert(data_in);
        assert(shape_size_(shape_in) == data_in->size());
        shape = std::move(shape_in);
        data = std::move(data_in);
    }

public:
    std::shared_ptr<std::valarray<Tdata>> data;

    Tensor(): data(std::make_shared<std::valarray<Tdata>>())
    {};

    Tensor(const std::vector<int> &dimension)
    {
        shape = normalize_shape_(dimension);
        if(!this->shape.empty())
        {
            const auto size = shape_size_(shape);
			data = std::make_shared<std::valarray<Tdata>>(0, size);
        }
    }

    Tensor(const std::initializer_list<std::size_t> &dimension)
    {
        shape = std::vector<std::size_t>(dimension);
        if(!this->shape.empty())
        {
            const auto size = shape_size_(shape);
			data = std::make_shared<std::valarray<Tdata>>(0, size);
        }
    }

    Tensor(const std::vector<int> &dimension, std::shared_ptr<std::valarray<Tdata>> data_in)
    {
        reset_(normalize_shape_(dimension), std::move(data_in));
    };

    Tensor(const std::initializer_list<std::size_t> &dimension, std::shared_ptr<std::valarray<Tdata>> data_in)
    {
        reset_(std::vector<std::size_t>(dimension), std::move(data_in));
    };

    inline Tdata& operator() (const std::size_t i0, const std::size_t i1) const
    {
        assert(shape.size() == 2);
        assert(i0 < shape[0] && i1 < shape[1]);
        return (*data)[i0 * shape[1] + i1];
    };

    inline Tdata& operator() (const std::size_t i0, const std::size_t i1, const std::size_t i2) const
    {
        assert(shape.size() == 3);
        assert(i0 < shape[0] && i1 < shape[1] && i2 < shape[2]);
        return (*data)[(i0 * shape[1] + i1) * shape[2] + i2];
    };

    inline Tdata& operator() (const std::size_t i0, const std::size_t i1)
    {
        return const_cast<const Tensor *>(this)->operator()(i0, i1);
    };

    inline Tdata& operator() (const std::size_t i0, const std::size_t i1, const std::size_t i2)
    {
        return const_cast<const Tensor *>(this)->operator()(i0, i1, i2);
    };

    inline std::size_t get_shape_all() const { return shape_size_(shape); };

    Tdata* ptr() const { return data->size() > 0 ? &(*data)[0] : nullptr; }
    void clear()
    {
        shape.clear();
        data = std::make_shared<std::valarray<Tdata>>();
    };
};


namespace Communicate_Tensors_Map_Judge
{

template <typename TA, typename TAC, typename Tdata>
std::map<TA, std::map<TAC, Tensor<Tdata>>> comm_map2_first(
    const MPI_Comm &mpi_comm, const std::map<TA, std::map<TAC, Tensor<Tdata>>> &Ds_in,
    const std::set<TA> &s0, const std::set<TA> &s1)
{
    std::map<TA, std::map<TAC, Tensor<Tdata>>> dummy;
    return dummy;
}

} /* end of namespace Communicate_Tensors_Map_Judge */

} /* end of namespace RI */
#endif
