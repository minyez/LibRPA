#pragma once

namespace RI
{

namespace GPU_Data
{

struct Pack
{
	bool exist = false;
	std::size_t pos = -1;
	Shape_Vector shape;
	int thread_num = 0;
};

}

}