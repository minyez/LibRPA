// ===================
//  Author: Peize Lin
//  date: 2025.02.10
// ===================

#pragma once

#include <magma.h>

namespace RI
{

namespace Magma_Wrapper
{
	inline magma_device_t magma_get_rank(const MPI_Comm &mpi_comm)
	{
		magma_device_t dev_id;
		magma_getdevice(&dev_id);
		return dev_id;
	}

	inline magma_int_t magma_get_size(const magma_int_t dev_size_uplim=100)
	{
		magma_int_t dev_size;
		std::vector<magma_device_t> devs_id(dev_size_uplim);
		magma_getdevices(devs_id.data(), dev_size_uplim, &dev_size);
		return dev_size;
	}
}

}