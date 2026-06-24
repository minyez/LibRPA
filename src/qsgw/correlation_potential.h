#pragma once

#include <complex>
#include <map>
#include <vector>

#include "../core/meanfield.h"
#include "../math/matrix_m.h"

namespace librpa_int
{
namespace qsgw
{

using SigmaRealAxisBlocks = std::vector<std::vector<std::vector<cplxdb>>>;

std::vector<std::vector<cplxdb>> build_G0(
    const MeanField& meanfield,
    const std::vector<double>& freq_nodes,
    int ispin,
    int ikpt,
    int n_states);

Matz build_correlation_potential_spin_k(
    const SigmaRealAxisBlocks& sigc_spin_k,
    int n_states);

Matz build_correlation_potential_spin_k_modeA(
    const SigmaRealAxisBlocks& sigc_spin_k,
    int n_states);

SigmaRealAxisBlocks build_sigma_real_axis_blocks_qsgw(
    const MeanField& meanfield,
    const std::vector<double>& freq_nodes,
    const std::map<double, Matz>& sigc_spin_k,
    int ispin,
    int ikpt,
    int n_states,
    int n_params_anacon);

Matz build_correlation_potential_spin_k_modeB_ac(
    const MeanField& meanfield,
    const std::vector<double>& freq_nodes,
    const std::map<double, Matz>& sigc_spin_k,
    int ispin,
    int ikpt,
    int n_states,
    int n_params_anacon);

Matz build_correlation_potential_spin_k_modeA_ac(
    const MeanField& meanfield,
    const std::vector<double>& freq_nodes,
    const std::map<double, Matz>& sigc_spin_k,
    int ispin,
    int ikpt,
    int n_states,
    int n_params_anacon);

Matz calculate_scRPA_exchange_correlation(
    const MeanField& meanfield,
    const std::vector<double>& freq_nodes,
    const std::vector<double>& freq_weights,
    const std::map<double, Matz>& sigc_spin_k,
    const SigmaRealAxisBlocks& sigc_sk_mat,
    const std::vector<std::vector<cplxdb>>& G0,
    int ispin,
    int ikpt,
    int n_states,
    double temperature);

} // namespace qsgw
} // namespace librpa_int
