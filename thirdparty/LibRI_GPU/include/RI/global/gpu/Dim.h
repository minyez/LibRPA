// ===================
//  Author: Peize Lin
//  date: 2026.02.18
// ===================

#pragma once

#include <magma.h>
#include <vector>

class Dim
{
  public:
	void input(const int shape)
	{
		this->shapes.push_back(shape);
	}
	void input(const int shape1, const int shape2)
	{
		this->shapes.push_back(shape1 * shape2);
	}

	void upload(const magma_queue_t &queue)
	{
		TESTING_CHECK(magma_imalloc(&this->d_m, this->shapes.size() + 1));
		magma_isetvector(this->shapes.size(), this->shapes.data(), 1, this->d_m, 1, queue);
	}

	int* data() const
	{
		return this->d_m;
	}

	std::vector<int> shapes;
	int* d_m;
};


class Dim_mnk
{
  public:
	Dim m;
	Dim n;
	Dim k;

	void input(const int shape_m, const int shape_n, const int shape_k)
	{
		m.input(shape_m);
		n.input(shape_n);
		k.input(shape_k);
	}

	void input(const std::pair<int,int> shape_m, const int shape_n, const int shape_k)
	{
		m.input(std::get<0>(shape_m), std::get<1>(shape_m));
		n.input(shape_n);
		k.input(shape_k);
	}

	void input(const int shape_m, const std::pair<int,int> shape_n, const int shape_k)
	{
		m.input(shape_m);
		n.input(std::get<0>(shape_n), std::get<1>(shape_n));
		k.input(shape_k);
	}

	void input(const int shape_m, const int shape_n, const std::pair<int,int> shape_k)
	{
		m.input(shape_m);
		n.input(shape_n);
		k.input(std::get<0>(shape_k), std::get<1>(shape_k));
	}

	void upload(const magma_queue_t &queue)
	{
		m.upload(queue);
		n.upload(queue);
		k.upload(queue);
	}
};