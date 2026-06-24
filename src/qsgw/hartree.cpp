/*!
 * @file hartree.cpp
 * @brief Hartree (classical Coulomb) kernel for QSGW — the H3 carrier.
 *
 * See hartree.h for the migration rationale. This file reconstructs the legacy
 * LIBRPA::Hartree class (7a7ff17f:src/hartree.cpp, 618 lines) on the new
 * LibRPA architecture.
 *
 * Implementation status (line refs are to legacy hartree.cpp @ 7a7ff17f):
 *   - DONE, aligned to legacy: constructor, get_dmat_cplx_k_global (:88-100,
 *     n_soc -> n_spinor), extract_dmat_cplx_IJblock (:102-117, via the new
 *     AtomicBasis::get_global_indices), reset_rspace/reset_kspace, and the
 *     build_KS_kgrid0/kgrid/band wavefunction wiring (QsgwState::wfc0 anchor
 *     for H2, MeanField::get_eigenvectors for the current wfc, kfrac from
 *     pbc.kfrac_list).
 *   - PENDING leader decision (H3): build() real-space kernel. The legacy core
 *     hartree_libri.lri.cal_cvcd_k_hartree (:243) is a localized-RI multi-center
 *     Coulomb contraction whose .hpp is unreachable; the bundled LibRI has
 *     dropped RI::Hartree. Options: (C) restore libRI RI::Hartree (preferred,
 *     near-verbatim migration) or (A) hand-written LRI-J contraction (high
 *     risk of silent numerical error). See the report sent to the team lead.
 *   - PENDING: build_KS() BLACS projection (mirror legacy :292-410 on the new
 *     ScaLAPACK API, using the Exx::build_KS_blacs pattern; simpler than Exx
 *     because HHartree_IJR is spin-summed — a single Fourier+rotation pass per
 *     (ispin,ispinor)), and nearest_R (use the existing find_nearest_bvk_cell).
 */
#include "hartree.h"

#include <stdexcept>

namespace librpa_int
{
namespace qsgw
{

Hartree::Hartree(const MeanField &mf_in,
                 const AtomicBasis &atbasis_wfc_in,
                 const PeriodicBoundaryData &pbc_in,
                 const KPointBlacsParallelContext &kblacs_ctxt_in,
                 const ArrayDesc &desc_wfc_in)
    : mf(mf_in),
      desc_wfc(desc_wfc_in),
      atbasis_wfc(atbasis_wfc_in),
      pbc(pbc_in),
      comm_h(kblacs_ctxt_in.comm_global_h),
      kblacs_ctxt(kblacs_ctxt_in)
{
}

ComplexMatrix Hartree::get_dmat_cplx_k_global(int ik) const
{
    // Mirrors legacy get_dmat_cplx_k_global (legacy hartree.cpp:88-100):
    // sum the density matrix over spin and spinor channels (n_soc -> n_spinor),
    // since the Hartree kernel couples only to the total charge density.
    const auto nspins = mf.get_n_spins();
    const auto nspinor = mf.get_n_spinor();
    ComplexMatrix dmat(mf.get_n_aos(), mf.get_n_aos());
    for (int ispin = 0; ispin < nspins; ++ispin)
    {
        for (int ispinor = 0; ispinor < nspinor; ++ispinor)
        {
            dmat += mf.get_dmat_cplx(ispin, ispinor, ispinor, ik);
        }
    }
    return dmat;
}

ComplexMatrix Hartree::extract_dmat_cplx_IJblock(const ComplexMatrix &dmat,
                                                 atom_t I, atom_t J) const
{
    // Mirrors legacy extract_dmat_cplx_IJblock (legacy hartree.cpp:102-117),
    // using the new AtomicBasis::get_global_indices in place of the legacy
    // atom_iw_loc2glo global-index helper.
    const auto I_idx = atbasis_wfc.get_global_indices(I);
    const auto J_idx = atbasis_wfc.get_global_indices(J);
    ComplexMatrix block(I_idx.size(), J_idx.size());
    for (std::size_t i = 0; i < I_idx.size(); ++i)
    {
        for (std::size_t j = 0; j < J_idx.size(); ++j)
        {
            block(i, j) = dmat(I_idx[i], J_idx[j]);
        }
    }
    return block;
}

std::array<int, 3> Hartree::nearest_R(atom_t I, atom_t J,
                                      const std::array<int, 3> &R) const
{
    // TODO(H3): mirror legacy nearest_R (legacy hartree.cpp:120-150) by calling
    // the existing find_nearest_bvk_cell (pbc.h:120), which requires the atomic
    // fractional coordinates. Resolve the new-architecture source of per-atom
    // fractional coords (AtomicBasis / PeriodicBoundaryData) when wiring build().
    (void)I;
    (void)J;
    (void)R;
    throw std::runtime_error("Hartree::nearest_R: not implemented yet (H3)");
}

void Hartree::build(const AtomicBasis &atbasis_abf,
                    const Cs_LRI &Cs,
                    const atpair_R_mat_t &coul_mat)
{
    // TODO(H3, pending leader decision): mirror legacy build (legacy
    // hartree.cpp:130-289). The legacy data flow is fully understood:
    //   1. set_parallel(comm, atoms_pos, lat, period)
    //   2. distribute_atom_pair_and_k -> list_I/list_J/list_k
    //   3. set_Cs(Cs as complex, 3-center aux_I x nao_I x nao_J)
    //   4. set_Vs(coul_mat as complex, 2-center aux_I x aux_J)
    //   5. dmat_libri[I][J][k] = extract_dmat_cplx_IJblock(get_dmat_cplx_k_global(k))
    //   6. CORE: HHartree_k = cal_cvcd_k_hartree(dmat, kfrac, I, J, IJ)   <-- legacy :243
    //   7. HHartree_k -> HHartree_IJR via sum_k exp(-i2pi k.R_bvk)/nk, nearest_R
    // Step 6 is the H3 decision point: RI::Hartree/cal_cvcd_k_hartree was
    // removed from the bundled LibRI. Restore libRI RI::Hartree (option C,
    // near-verbatim) or hand-write the LRI-J contraction (option A, high risk).
    (void)atbasis_abf;
    (void)Cs;
    (void)coul_mat;
    throw std::runtime_error("Hartree::build: pending H3 kernel-path decision");
}

void Hartree::build_KS(const WfcMap &wfc_target,
                       const std::vector<Vector3_Order<double>> &kfrac_target)
{
    // TODO: mirror legacy build_KS (legacy hartree.cpp:292-410) on the new
    // ScaLAPACK API, following the Exx::build_KS_blacs pattern. Simpler than
    // Exx because HHartree_IJR is spin-summed: for each (ik) Fourier-transform
    // HHartree_IJR to the k-point NAO block, then for each (ispin, ispinor)
    // rotate wfc^dagger . HHartree_nao_nao(k) . wfc into the n_bands basis and
    // accumulate into Hartree_is_ik_KS / EHartree.
    (void)wfc_target;
    (void)kfrac_target;
    throw std::runtime_error("Hartree::build_KS: not implemented yet");
}

void Hartree::build_KS_kgrid0(const QsgwState &state)
{
    // H2 fixed-basis anchor: project using the step-0 wavefunctions cached in
    // QsgwState::wfc0 (replaces the legacy meanfield.get_eigenvectors0()).
    build_KS(state.wfc0, pbc.kfrac_list);
}

void Hartree::build_KS_kgrid()
{
    // Current wavefunctions (same behavior as legacy build_KS_kgrid).
    build_KS(mf.get_eigenvectors(), pbc.kfrac_list);
}

void Hartree::build_KS_band(const WfcMap &wfc_band,
                            const std::vector<Vector3_Order<double>> &kfrac_band)
{
    // Band wavefunctions supplied by the caller (band path does not use wfc0).
    build_KS(wfc_band, kfrac_band);
}

void Hartree::reset_rspace()
{
    HHartree_IJR.clear();
    is_rspace_built_ = false;
}

void Hartree::reset_kspace()
{
    Hartree_is_ik_KS.clear();
    Hartree_is_ik_nao.clear();
    EHartree.clear();
    is_kspace_built_ = false;
}

} // namespace qsgw
} // namespace librpa_int
