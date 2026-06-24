/*!
 @file qsgw_state.h
 @brief QSGW self-consistency state cache — the H2 carrier.

 Holds the step-0 (fixed-basis) anchors wfc0 / velocity0 / wg0 that replace the
 legacy MeanField::get_eigenvectors0 / get_velocity0 / get_weight0 getters which
 no longer exist on the new MeanField (LEADER_AUDIT hard-fact #4, §3 H2).

 Contract (H2, LEADER_AUDIT §3):
   - The driver snapshots  wfc0 = mf.get_eigenvectors()  *before* the SCF loop
     first mutates the wavefunctions; the SCF loop then reads wfc0 read-only.
   - Task B (collect_wfc_rows(use_fixed_basis=true) / store_rotated_wfc /
     rotate_velocity) and Task C (Hartree.build_KS_kgrid0) consume
     QsgwState::wfc0 directly.
 */
#ifndef QSGW_STATE_H
#define QSGW_STATE_H

#include <map>
#include <vector>

#include "../math/complexmatrix.h" // ComplexMatrix
#include "../math/matrix.h"        // matrix (occupation weights)
#include "../core/meanfield.h"     // MeanField

namespace librpa_int {

//! Step-0 (fixed-basis) anchors for QSGW self-consistency.
struct QsgwState
{
    //! KS eigenvectors at SCF start, indexed [ispin][ispinor][ikpt].
    //! Same layout as MeanField::get_eigenvectors().
    std::map<int, std::map<int, std::map<int, ComplexMatrix>>> wfc0;

    //! Occupation weights at SCF start, [ispin](ikpt, ib).
    //! Same layout as MeanField::get_weight().
    std::vector<matrix> wg0;

    //! Velocity/momentum matrix at SCF start, [ispin][ikpt][cartesian].
    //! Identical to velocity_matrix_t from dielecmodel.h; kept as the explicit
    //! nested type so this header does not pull in dielecmodel.h.
    std::vector<std::vector<std::vector<ComplexMatrix>>> velocity0;

    bool has_wfc0 = false;
    bool has_wg0 = false;
    bool has_velocity0 = false;

    //! Snapshot wfc0 = mf.get_eigenvectors() (deep copy of the 3-key map).
    void snapshot_wfc0(const MeanField &mf);
    //! Snapshot wg0 = mf.get_weight().
    void snapshot_wg0(const MeanField &mf);
    //! Snapshot velocity0 from the Dataset velocity matrix (velocity_matrix_t).
    void snapshot_velocity0(
        const std::vector<std::vector<std::vector<ComplexMatrix>>> &velocity);

    //! Convenience: snapshot all three anchors at once.
    void snapshot(
        const MeanField &mf,
        const std::vector<std::vector<std::vector<ComplexMatrix>>> &velocity);

    //! Read-only accessor for a single wfc0 block; returns nullptr if absent
    //! (mirrors MeanField::find_wfc semantics).
    const ComplexMatrix *find_wfc0(int ispin, int ispinor, int ikpt) const noexcept;

    //! Drop all anchors (e.g. on SCF restart).
    void clear();
};

} // namespace librpa_int

#endif // QSGW_STATE_H
