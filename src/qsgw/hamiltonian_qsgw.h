#pragma once

#include <map>
#include <vector>

#include "../core/meanfield.h"
#include "../math/complexmatrix.h"
#include "../math/matrix_m.h"

#if __has_include("qsgw_state.h")
#include "qsgw_state.h"
#define LIBRPA_QSGW_HAS_STATE_HEADER 1
#else
#define LIBRPA_QSGW_HAS_STATE_HEADER 0
#endif

namespace librpa_int
{
namespace qsgw
{

using SpinKMatrixMap = std::map<int, std::map<int, Matz>>;
using WfcMap = std::map<int, std::map<int, std::map<int, ComplexMatrix>>>;
using VelocityMatrix = std::vector<std::vector<std::vector<ComplexMatrix>>>;

#if !LIBRPA_QSGW_HAS_STATE_HEADER
struct QsgwState;
#endif

SpinKMatrixMap construct_H0_GW(
    const SpinKMatrixMap& H_KS_all,
    const SpinKMatrixMap& vxc_all,
    const SpinKMatrixMap& Hexx_all,
    const SpinKMatrixMap& Vc_all,
    int n_spins,
    int n_kpoints,
    int n_bands);

SpinKMatrixMap construct_H0_GW_fermi_window(
    const MeanField& meanfield,
    const SpinKMatrixMap& H_KS_all,
    const SpinKMatrixMap& vxc_all,
    const SpinKMatrixMap& Hexx_all,
    const SpinKMatrixMap& Vc_all,
    int n_spins,
    int n_kpoints,
    int n_bands,
    int bands_above_fermi,
    double diag_shift_ev);

SpinKMatrixMap construct_H0_GW_cut(
    const MeanField& meanfield,
    const SpinKMatrixMap& H_KS_all,
    const SpinKMatrixMap& vxc_all,
    const SpinKMatrixMap& Hexx_all,
    const SpinKMatrixMap& Vc_all,
    int n_spins,
    int n_kpoints,
    int n_bands);

SpinKMatrixMap construct_H0_GW_new_basis(
    const MeanField& meanfield,
    const SpinKMatrixMap& H_KS_all,
    const SpinKMatrixMap& H_DFT_nao,
    const SpinKMatrixMap& Hexx_all,
    const SpinKMatrixMap& Vc_all,
    int n_spins,
    int n_kpoints,
    int n_bands);

SpinKMatrixMap construct_H0_HF(
    const SpinKMatrixMap& H_KS_all,
    const SpinKMatrixMap& vxc_all,
    const SpinKMatrixMap& Hexx_all,
    int n_spins,
    int n_kpoints);

void diagonalize_and_store(
    MeanField& meanfield,
    const SpinKMatrixMap& H0_GW_all,
    int n_spins,
    int n_kpoints,
    int dimension,
    VelocityMatrix* velocity = nullptr);

#if LIBRPA_QSGW_HAS_STATE_HEADER
void diagonalize_and_store_fixed_basis(
    MeanField& meanfield,
    const SpinKMatrixMap& H0_GW_all,
    const QsgwState& state,
    int n_spins,
    int n_kpoints,
    int dimension,
    VelocityMatrix* velocity = nullptr);
#endif

void diagonalize_and_store_fixed_basis(
    MeanField& meanfield,
    const SpinKMatrixMap& H0_GW_all,
    const WfcMap& wfc0,
    const VelocityMatrix* velocity0,
    int n_spins,
    int n_kpoints,
    int dimension,
    VelocityMatrix* velocity = nullptr);

Matz collect_wfc_rows(
    const MeanField& meanfield,
    int ispin,
    int ikpt,
    const WfcMap* fixed_wfc = nullptr);

void store_rotated_wfc(
    MeanField& meanfield,
    int ispin,
    int ikpt,
    const Matz& eigvec_nao);

void rotate_velocity_to_qp_basis(
    const Matz& eigvec_ks,
    const VelocityMatrix& velocity_source,
    int ispin,
    int ikpt,
    VelocityMatrix& velocity_target);

void apply_qsgw_hround(Matz& mat, int n_bands);

double qsgw_hround_scale();

} // namespace qsgw
} // namespace librpa_int
