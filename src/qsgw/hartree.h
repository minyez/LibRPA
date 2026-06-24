/*!
 * @file hartree.h
 * @brief Hartree (classical Coulomb) kernel for QSGW — the H3 carrier.
 *
 * Reconstructs the legacy LIBRPA::Hartree class (7a7ff17f:src/hartree.{h,cpp},
 * 618 lines) on the new LibRPA architecture. The new src/core/ has no Hartree
 * class, and the bundled thirdparty/LibRI has dropped the RI::Hartree physics
 * class together with its lri.cal_cvcd_k_hartree contraction (verified: only
 * Exx / Exx_Post_2D / GW / RPA remain under include/RI/physics/, and a
 * full-tree grep for cal_cvcd_k_hartree / RI::Hartree returns zero matches).
 * Exx.cal_Hs is an exchange kernel and cannot reproduce the Coulomb kernel.
 *
 * The real-space Hartree matrix is therefore rebuilt from the density matrix
 * (HHartree ~ Cs_LRI . V . rho): the density is taken from
 * MeanField::get_dmat_cplx (meanfield.h:123, spin+spinor summed, mirroring the
 * legacy get_dmat_cplx_k_global) and the Coulomb matrix from FT_Vq
 * (coulmat.cpp:439), reproducing the legacy build(Cs, Rlist, VR) data flow
 * (legacy hartree.cpp:130-289).
 *
 * The KS-basis projection (build_KS_*) rotates the NAO-space Hartree matrix to
 * the n_bands x n_bands KS basis. build_KS_kgrid0 consumes QsgwState::wfc0 as
 * the fixed step-0 anchor (H2), build_KS_kgrid uses MeanField::get_eigenvectors,
 * and build_KS_band takes band wavefunctions directly.
 *
 * SOC (H8): legacy n_soc -> new n_spinor; wavefunctions are indexed
 * [ispin][ispinor][ikpt] (MeanField::get_eigenvectors, 3-key map), replacing
 * the legacy vector[ispin][isoc][ikpt] layout.
 */
#ifndef QSGW_HARTREE_H
#define QSGW_HARTREE_H

#include <array>
#include <map>
#include <vector>

#include "../core/atom.h"                     // atom_t
#include "../core/atomic_basis.h"             // AtomicBasis
#include "../core/meanfield.h"                // MeanField
#include "../core/pbc.h"                      // PeriodicBoundaryData
#include "../core/ri.h"                       // Cs_LRI, atpair_R_mat_t, libri_types, RI::Tensor
#include "../math/complexmatrix.h"            // ComplexMatrix
#include "../math/matrix_m.h"                 // Matz, Matd
#include "../math/vector3_order.h"            // Vector3_Order
#include "../mpi/base_blacs.h"                // ArrayDesc
#include "../mpi/kpoint_blacs_parallel_context.h" // KPointBlacsParallelContext

#include "qsgw_state.h"                       // QsgwState (H2 wfc0 anchor)

namespace librpa_int
{
namespace qsgw
{

//! (ispin, ik) -> n_bands x n_bands complex matrix (a KS-basis operator).
using SpinKMatrixMap = std::map<int, std::map<int, Matz>>;
//! [ispin][ispinor][ikpt] -> (n_bands x n_aos) wavefunction block.
using WfcMap = std::map<int, std::map<int, std::map<int, ComplexMatrix>>>;

/*!
 * @brief Hartree (Coulomb) operator rebuilt from the density matrix for QSGW.
 *
 * Mirrors the legacy LIBRPA::Hartree class but on the new MeanField / libRI
 * APIs. The constructor signature follows the new Exx class (exx.h:60); the
 * MPI handler is taken from kblacs_ctxt.comm_global_h exactly as in Exx::Exx
 * (exx.cpp:219), so no separate comm argument is required.
 */
class Hartree
{
public:
    //! Reference to the MeanField providing the density matrix.
    const MeanField &mf;
    //! ScaLAPACK descriptor of the MeanField wavefunctions.
    const ArrayDesc &desc_wfc;
    //! Atomic-basis description of the NAO wavefunction basis.
    const AtomicBasis &atbasis_wfc;
    //! Periodic-boundary data (lattice vectors, Rlist, BvK period).
    const PeriodicBoundaryData &pbc;
    //! Global MPI communicator handler (kblacs_ctxt.comm_global_h).
    const MpiCommHandler &comm_h;
    //! k-point BLACS parallel context.
    const KPointBlacsParallelContext &kblacs_ctxt;

    //! Real-space Hartree matrix, dim (I, (J,R), nao_I, nao_J); spin+spinor
    //! summed since the Hartree kernel couples only to the total charge density.
    //! Mirrors legacy HHartree_libri (legacy hartree.h).
    std::map<int, std::map<libri_types<int, int>::TAC, RI::Tensor<std::complex<double>>>> HHartree_IJR;
    //! Hartree operator in the KS basis, dim (ispin, ik, n_bands, n_bands).
    SpinKMatrixMap Hartree_is_ik_KS;
    //! Hartree operator in the NAO basis, dim (ispin, ik, n_aos, n_aos).
    SpinKMatrixMap Hartree_is_ik_nao;
    //! Per-state Hartree energy (diagonal of Hartree_is_ik_KS),
    //! dim (ispin, ik, ib).
    std::map<int, std::map<int, std::map<int, double>>> EHartree;

    Hartree(const MeanField &mf_in,
            const AtomicBasis &atbasis_wfc_in,
            const PeriodicBoundaryData &pbc_in,
            const KPointBlacsParallelContext &kblacs_ctxt_in,
            const ArrayDesc &desc_wfc_in);

    //! Build the real-space Hartree matrix HHartree_IJR from the density.
    /*! Mirrors legacy Hartree::build(Cs, Rlist, VR) (legacy hartree.cpp:130-289).
     *  @param atbasis_abf  atomic-basis description of the auxiliary (ABF) basis.
     *  @param Cs            localized-RI 3-center coefficients (Cs_LRI).
     *  @param coul_mat      ABF Coulomb matrix in real space (e.g. from FT_Vq).
     */
    void build(const AtomicBasis &atbasis_abf,
               const Cs_LRI &Cs,
               const atpair_R_mat_t &coul_mat);

    //! Project onto the KS basis using the fixed step-0 anchor wfc0 (H2).
    void build_KS_kgrid0(const QsgwState &state);
    //! Project onto the KS basis using the current MeanField wavefunctions.
    void build_KS_kgrid();
    //! Project onto the KS basis using externally supplied band wavefunctions.
    void build_KS_band(const WfcMap &wfc_band,
                       const std::vector<Vector3_Order<double>> &kfrac_band);

    void reset_rspace();
    void reset_kspace();

private:
    bool is_rspace_built_ = false;
    bool is_kspace_built_ = false;

    //! Total (spin+spinor summed) k-space density matrix, mirroring the legacy
    //! get_dmat_cplx_k_global (legacy hartree.cpp:88-100) with n_soc -> n_spinor.
    ComplexMatrix get_dmat_cplx_k_global(int ik) const;
    //! Extract the (I,J) atom-pair block from a global NAO density matrix,
    //! mirroring the legacy extract_dmat_cplx_IJblock (legacy hartree.cpp:102-117).
    ComplexMatrix extract_dmat_cplx_IJblock(const ComplexMatrix &dmat,
                                            atom_t I, atom_t J) const;
    //! Nearest periodic image of R for the atom pair (I,J) in the BvK cell,
    //! mirroring the legacy nearest_R (legacy hartree.cpp:120-150).
    std::array<int, 3> nearest_R(atom_t I, atom_t J,
                                 const std::array<int, 3> &R) const;

    //! Shared KS-basis projection (NAO -> n_bands rotation), mirroring the
    //! legacy build_KS (legacy hartree.cpp:292-410) on the new BLACS API.
    void build_KS(const WfcMap &wfc_target,
                  const std::vector<Vector3_Order<double>> &kfrac_target);
};

} // namespace qsgw
} // namespace librpa_int

#endif // QSGW_HARTREE_H
