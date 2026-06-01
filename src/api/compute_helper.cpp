#include "compute_helper.h"

#include <algorithm>
#include <set>
#include <string>

#include "../utils/error.h"

namespace librpa_int::api
{

// Build a deduplicated list of requested k-points across ranks. This lets
// extraction APIs exchange only the k-resolved quantities callers asked for.
std::vector<int> collect_requested_iks(const MpiCommHandler &comm_h, const int n_kpts_this,
                                       const int *iks_this, const int n_kpoints)
{
    std::vector<int> nks_all(comm_h.nprocs, 0);
    nks_all[comm_h.myid] = n_kpts_this;
    comm_h.allreduce(MPI_IN_PLACE, nks_all.data(), comm_h.nprocs, MPI_SUM);

    const int nk_max = *std::max_element(nks_all.cbegin(), nks_all.cend());
    if (nk_max == 0) return {};

    std::vector<int> iks_all(comm_h.nprocs * nk_max, -1);
    int bad_ik_local = 0;
    int bad_ik_value_local = -1;
    for (int i = 0; i != n_kpts_this; ++i)
    {
        const int ik = iks_this[i];
        if (ik < 0 || ik >= n_kpoints)
        {
            bad_ik_local = 1;
            bad_ik_value_local = ik;
            continue;
        }
        iks_all[comm_h.myid * nk_max + i] = ik;
    }

    int bad_ik = 0;
    int bad_ik_value = -1;
    comm_h.allreduce(&bad_ik_local, &bad_ik, 1, MPI_MAX);
    comm_h.allreduce(&bad_ik_value_local, &bad_ik_value, 1, MPI_MAX);
    if (bad_ik)
        throw LIBRPA_RUNTIME_ERROR("requested k-point index out of range: " +
                                  std::to_string(bad_ik_value));

    comm_h.allreduce(MPI_IN_PLACE, iks_all.data(), static_cast<int>(iks_all.size()), MPI_MAX);

    std::set<int> unique_iks;
    for (int pid = 0; pid != comm_h.nprocs; ++pid)
    {
        for (int i = 0; i != nks_all[pid]; ++i)
        {
            unique_iks.insert(iks_all[pid * nk_max + i]);
        }
    }
    return {unique_iks.cbegin(), unique_iks.cend()};
}

}
