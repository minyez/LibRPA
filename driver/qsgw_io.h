/*!
 @file qsgw_io.h
 @brief QSGW pyatb / Hamiltonian I/O bundle export (Task F).

 Faithful port of legacy driver/driver_utils.cpp:91/341/374 (7a7ff17f).
 HR source is the QSGW Hamiltonian H0_GW_all in KS-orthonormal band space.

 Implementation status (see LEADER_AUDIT Sec.5 Task F + red line #5):
  - export_qsgw_hamiltonian_bundle : PORTED (the acceptance core: writes the
    HR = H0_GW_all bundle -- ELSI CSC + Matrix Market per (spin,kpoint), a
    latest-iteration marker, and manifest.json). Pure output, no read-back.
  - export_pyatb_state_bundle / refresh_pyatb_headwing_bundle : DECLARED but
    NOT YET IMPLEMENTED. They depend on write_pyatb_bundle,
    future driver-level QSGW pyatb export/rebuild controls,
    get_iterative_pyatb_headwing_bundle_dir and maybe_run_pyatb_rebuild_command,
    none of which exist on the new arch yet (QSGW driver Task D / pyatb-export
    subsystem). They throw std::runtime_error if called. To be filled when that
    infrastructure lands.
 */
#pragma once

#include <map>
#include <string>
#include <vector>

#include "../src/core/meanfield.h"     // librpa_int::MeanField
#include "../src/math/matrix_m.h"      // librpa_int::Matz
#include "../src/math/vector3_order.h" // librpa_int::Vector3_Order

namespace librpa_int {

//! Write the QSGW Hamiltonian bundle (HR source = H0_GW_all, KS-band space).
/*! Under <bundle_dir>/qsgw_hamiltonian/iter_XXXXXX/: per-(spin,kpoint) ELSI CSC
 *  + Matrix Market files, a <state_label>_latest_iteration.txt marker, and a
 *  manifest.json (dims, efermi, k-points). Pure output. Returns iteration dir.
 *
 *  Aligned to legacy driver/driver_utils.cpp:91-158. Two new-arch adaptations:
 *   - manifest "n_soc" -> "n_spinor" (MeanField::get_n_soc removed, hard-fact #4);
 *   - print_matrix_elsi_csc ported locally (Matz) since the new arch lacks it.
 */
std::string export_qsgw_hamiltonian_bundle(
    const std::string &bundle_dir, const MeanField &mf,
    const std::vector<Vector3_Order<double>> &kfrac,
    const std::map<int, std::map<int, Matz>> &hamiltonians, int iteration,
    const std::string &state_label);

//! (stub) Export pyatb state bundle + optional Hamiltonian. Blocked on Task D.
void export_pyatb_state_bundle(
    const MeanField &mf, const std::vector<Vector3_Order<double>> &kfrac,
    const std::string &bundle_dir = "",
    const std::map<int, std::map<int, Matz>> *hamiltonians = nullptr,
    int iteration = -1, const std::string &state_label = "kgrid",
    bool run_rebuild_command = false);

//! (stub) Refresh pyatb headwing bundle. Blocked on Task D.
void refresh_pyatb_headwing_bundle(
    const MeanField &mf, const std::vector<Vector3_Order<double>> &kfrac,
    const std::string &bundle_dir = "",
    const std::map<int, std::map<int, Matz>> *hamiltonians = nullptr,
    int iteration = -1, const std::string &state_label = "kgrid");

} // namespace librpa_int
