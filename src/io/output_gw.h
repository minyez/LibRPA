#pragma once
#include "../core/gw.h"

#include <string>
#include <vector>

namespace librpa_int
{

//! Collectively export a square submatrix of a distributed KS-basis matrix.
void write_ks_matrix_binary_parallel(const Matz &mat_loc, const ArrayDesc &desc,
                                     int istate_start, int istate_end,
                                     const std::string &fn);

//! Collective over G0W0::comm_h. Rank 0 writes KS-diagonal SigC(iw) data.
void write_self_energy_omega(const char *fn, const G0W0& s_g0w0, int n_kpts, int n_bands);
void write_self_energy_omega(const char *fn, const G0W0& s_g0w0,
                             const std::vector<int> &iks, int n_bands);
void write_self_energy_omega_kpoints(const char *fn, const G0W0& s_g0w0,
                                     const std::vector<int> &iks);

}
