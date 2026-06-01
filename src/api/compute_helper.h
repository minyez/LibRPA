#pragma once

#include <vector>

#include "../mpi/base_mpi.h"

namespace librpa_int::api
{

//! Collect the union of k-point indices requested by all ranks in an extraction API call.
//! The result gives a compact, deterministic k-list for collective result handoff.
std::vector<int> collect_requested_iks(const MpiCommHandler &comm_h, int n_kpts_this,
                                       const int *iks_this, int n_kpoints);

}
