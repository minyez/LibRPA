// QSGW self-consistent driver (Task D skeleton, for leader review).
//
// Single-file driver mirroring driver/tasks/g0w0.cpp. Implements the SCF loop
// integrating the ported QSGW modules (Tasks A/B/C/F) per qsgw_driver_design.md.
// Status: SKELETON for review — SCF structure + H1..H5/Hartree/convergence/
// checkpoint integration points are wired with the real module APIs; details
// marked `TODO(D)` are deferred for the refine pass after review. Not all paths
// compile yet (some assembly helpers / knob readers are stubs).
//
// Hard constraints (LEADER_AUDIT §3): H1 reset cached kernels each step;
// H2 QsgwState wfc0 anchor snapshot before first wfc mutation; H3 Hartree
// build + Hartree_0(iter1)/Hartree_i_delta(iter>1); H4 vxc0 = vxc_dft +
// hf_rotated; H5 full non-diagonal sigc via pds->p_g0w0->sigc_is_ik_f_KS.
//
// Legacy reference: driver/task_qsgw.cpp @ 7a7ff17f (line refs in design note).

#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <librpa_enums.h>

#include "../../src/api/instance_manager.h" // get_dataset_instance
#include "../../src/core/input_symmetry.h"
#include "../../src/core/coulmat.h"      // FT_Vq (Hartree Coulomb, H3)
#include "../../src/io/fs.h"
#include "../../src/io/global_io.h"
#include "../../src/io/input_elsi.h"   // load_matrix_cplx (hf_exchange CSC reader, H4)
#include "../../src/utils/constants.h" // HA2EV
#include "../../src/utils/profiler.h"
#include "../driver.h"
#include "../read_data.h"
#include "../task.h"
#include "task_helper.h"

// Ported QSGW modules (Tasks A/B/C/F)
#include "../../src/qsgw/qsgw_state.h"                 // QsgwState (H2)
#include "../../src/qsgw/mixing.h"                     // PulayMixer
#include "../../src/qsgw/fermi_energy_occupation.h"    // calculate_fermi_energy / update / calculate_total_weight
#include "../../src/qsgw/hamiltonian_qsgw.h"           // construct_H0_GW / diagonalize_and_store_fixed_basis / apply_qsgw_hround
#include "../../src/qsgw/correlation_potential.h"      // build_correlation_potential_spin_k
#include "../../src/qsgw/hartree.h"                    // Hartree (H3)
#include "../qsgw_io.h"                                // export_qsgw_hamiltonian_bundle (Task F)

void driver::task_qsgw()
{
    using std::cout;
    using std::endl;
    using std::map;
    using namespace librpa_int;
    using namespace librpa_int::global;
    using namespace librpa_int::qsgw;

    profiler.start("qsgw", "QSGW self-consistent quasi-particle calculation");

    // ---- dimensions / dataset (g0w0.cpp pattern) ----
    auto pds = librpa_int::api::get_dataset_instance(h);
    auto &mf = pds->mf;
    const auto &kfrac_list = pds->pbc.kfrac_list;
    const auto &symmetry_context = pds->symmetry_context;
    const int n_spins = mf.get_n_spins();
    const int n_kpoints = mf.get_n_kpoints();
    const int n_bands = mf.get_n_bands();
    const int n_spinor = mf.get_n_spinor(); // H8: n_soc -> n_spinor

    // ========================================================================
    // ONE-TIME SETUP (legacy task_qsgw.cpp L595-L1140)
    // ========================================================================
    profiler.start("qsgw_setup", "QSGW one-time setup");

    // --- read truncated Coulomb (g0w0.cpp pattern) ---
    profiler.start("read_vq_cut", "Load truncated Coulomb");
    auto routing = opts.parallel_routing;
    if (routing == LIBRPA_ROUTING_AUTO)
        routing = librpa_int::decide_auto_routing(n_atoms, n_kpoints * opts.nfreq);
    if (routing == LIBRPA_ROUTING_RTAU)
        read_Vq_full(driver_params.input_dir, driver_params.prefix_coul_cut, true,
                     driver_params.version_coul_reader,
                     driver::get_bool(driver::opts.use_shrink_abfs));
    else
        read_Vq_row(driver_params.input_dir, driver_params.prefix_coul_cut, opts.vq_threshold,
                    local_atpair, true, driver_params.version_coul_reader,
                    driver::get_bool(driver::opts.use_shrink_abfs));
    profiler.stop("read_vq_cut");

    // --- DFT xc: loaded per (spin,kpt) as the FULL xc_matr CSC below (H4 #4 fix,
    //     coremath review: H0_GW is a full matrix, off-diagonal xc must be kept;
    //     the diagonal read_vxc path drops off-diag). hf added in the same loop. ---

    // --- H2: snapshot step-0 anchors BEFORE the loop mutates wfc ---
    QsgwState qsgw_state;
    qsgw_state.snapshot_wfc0(mf);
    qsgw_state.snapshot_wg0(mf);
    // velocity0 snapshot deferred until velocity source is wired (TODO(D)).

    // --- QSGW knobs (legacy read them via local parser, NOT Params;
    //     Params lacks these — qsgw_driver_design.md §6) ---
    // TODO(D): wire InputParser (driver/inputfile.h) for: max_iter, mixing_history,
    //          mixing_beta, linear_mixing_steps, eigenvalue_diff_tolerance,
    //          temperature, eigdiff_focus_nbands, hamiltonian_cut_above_fermi,
    //          hamiltonian_cut_diag_shift_ev, qsgw_checkpoint_every, qsgw_restart_dir.
    const int max_iterations = 500;
    const int mixing_history = 12;
    const double mixing_beta = 0.2;
    const double temperature = 0.0;            // TODO(D): knob
    const int eigdiff_focus_nbands = 10;       // convergence focus window
    const double eigenvalue_diff_tolerance = 1e-5;
    const int hamiltonian_cut_above_fermi = -1; // <0 disables fermi_window variant
    const double hamiltonian_cut_diag_shift_ev = 0.0;

    double efermi = mf.get_efermi();
    const double total_electrons = calculate_total_weight(mf); // replaces get_total_weight

    // --- H_KS0 + H4 vxc0 assembly (legacy task_qsgw.cpp L882-L905) ---
    // H_KS0[ispin][ikpt] = diag(eigvals) (KS orthonormal band space, n_bands x n_bands).
    // vxc0 = vxc_dft + hf_ks, hf_ks = conj(wfc)*hf_nao*transpose(wfc), hf_nao read
    // from hf_exchange_*.csc (hard fact #7; construct_H0_GW does H_KS - vxc0 + Hexx + Vc).
    const int n_aos = mf.get_n_aos();
    SpinKMatrixMap vxc0;
    SpinKMatrixMap H_KS0;
    for (int ispin = 0; ispin < n_spins; ++ispin)
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
        {
            // H_KS0 = diag(eigvals) (KS orthonormal band space).
            Matz H_KS0_sk(n_bands, n_bands, MAJOR::COL);
            for (int ib = 0; ib < n_bands; ++ib)
                H_KS0_sk(ib, ib) = mf.get_eigenvals()[ispin](ikpt, ib);

            // KS wfc (n_bands x n_aos), non-SOC (ispinor=0).
            // SOC spinor folding (H8) TODO(D-review): wfc1(ib, iao*n_spinor+ispinor)
            // per legacy L688, plus spinor-summed density for Hartree.
            Matz wfc1(n_bands, n_aos, MAJOR::COL);
            for (int ib = 0; ib < n_bands; ++ib)
                for (int iao = 0; iao < n_aos; ++iao)
                    wfc1(ib, iao) = mf.get_eigenvectors()[ispin].at(0).at(ikpt)(ib, iao);

            // #4 (coremath): load the FULL xc_matr (KS band space; H0_GW is a full
            // matrix so off-diagonal xc must be kept — the diagonal read_vxc path
            // drops it). Pattern: xc_matr_spin_0{S}_kpt_{000006K}.csc, S=ispin+1.
            std::ostringstream oss_xc;
            oss_xc << driver_params.input_dir << "xc_matr_spin_0" << (ispin + 1)
                   << "_kpt_" << std::setw(6) << std::setfill('0') << (ikpt + 1) << ".csc";
            Matz vxc0_sk = load_matrix_cplx(oss_xc.str()); // full n_bands x n_bands (KS)

            // H4: hf rotation. hf_exchange_*.csc is NAO (n_aos x n_aos) -> KS via wfc1;
            // vxc0 = xc + hf (hard fact #7). (#5 TODO(D): HF mandatory -> hard error if
            // absent; interim: skip when the file is missing.)
            std::ostringstream oss_hf;
            oss_hf << driver_params.input_dir << "hf_exchange_spin_0" << (ispin + 1)
                   << "_kpt_" << std::setw(6) << std::setfill('0') << (ikpt + 1) << ".csc";
            if (librpa_int::path_exists(oss_hf.str().c_str()))
            {
                Matz hf_nao = load_matrix_cplx(oss_hf.str());
                vxc0_sk = vxc0_sk + conj(wfc1) * hf_nao * transpose(wfc1);
            }
            vxc0[ispin][ikpt] = vxc0_sk;
            H_KS0[ispin][ikpt] = H_KS0_sk;
        }

    // --- H3: Hartree object (Task C) — holds const MeanField& so it reads the
    // live mf; constructed once, build() called per iteration to recompute from
    // the updated density. Recipe from hartree, mirroring Exx (compute_g0w0.cpp:
    // 628-645). basis/Cs/VR are iteration-invariant; coul uses vq_cut for spike
    // ground-truth alignment (final parity may switch to pds->vq, coremath review).
    const bool use_shrink_h = opts.use_shrink_abfs == LIBRPA_SWITCH_ON;
    const auto &basis_aux_h = use_shrink_h ? pds->basis_aux_shrink : pds->basis_aux;
    const auto &cs_h = use_shrink_h ? pds->cs_data_shrink : pds->cs_data;
    const auto &coul_h = pds->vq_cut;
    profiler.start("hartree_ft_vq", "Fourier transform Coulomb for Hartree");
    const auto VR_h = librpa_int::FT_Vq(basis_aux_h, pds->symmetry_context, coul_h,
                                        pds->pbc, true,
                                        opts.use_symmetry_exx == LIBRPA_SWITCH_ON);
    profiler.stop("hartree_ft_vq");
    qsgw::Hartree hartree(pds->mf, pds->basis_wfc, pds->pbc, pds->symmetry_context,
                          pds->scfk_blacs_ctxt, pds->desc_wfc_kb_full);
    SpinKMatrixMap Hartree_0; // iter-1 anchor (legacy task_qsgw.cpp:416-428)

    // --- mixing (Task A) ---
    PulayMixer mixer(mixing_history, mixing_beta);

    // --- checkpoint restart (legacy L1069-L1093) ---
    int iteration = 0;
    bool converged = false;
    // TODO(D): if (Params::qsgw_restart) load_qsgw_checkpoint(...) -> restore
    //          H0_GW_all/Hartree_0/efermi/mixer + diagonalize_and_store_fixed_basis.

    profiler.stop("qsgw_setup");

    // ========================================================================
    // SCF LOOP (legacy task_qsgw.cpp L1141 `while(!converged && iteration<max)`)
    // ========================================================================
    while (!converged && iteration < max_iterations)
    {
        iteration++;
        profiler.start("qsgw_iter", "QSGW SCF iteration", true);

        // ---- H1: reset cached kernels so they rebuild with the updated mf ----
        // p_exx has a real guard (compute_g0w0.cpp:628); p_headwing caches an mf
        // copy. p_chi0/p_g0w0 rebuild unconditionally. (LEADER_AUDIT §3 H1)
        pds->p_exx.reset();
        pds->p_headwing.reset();

        // ---- recompute G0W0 self-energy with updated mf (g0w0.cpp pattern) ----
        h.build_g0w0_sigma(opts);

        // #1/#2 (coremath review): build_g0w0_sigma only does build_spacetime; it does
        // NOT fill sigc_is_ik_f_KS. Enable the KS-matrix storage flag (#2, gated default
        // off at gw.cpp:1798) then run the KS rotation (#1) which populates it.
        pds->p_g0w0->output_sigc_ks_mat_kf = true;
        pds->p_g0w0->build_sigc_matrix_KS_kgrid_blacs(pds->blacs_h);

        // ---- H5 + Vc: full non-diagonal sigc -> correlation potential (mode B) ----
        // sigc_is_ik_f_KS : [ispin][ikpt][freq] -> Matz (gw.h:80). For each (spin,kpt)
        // build SigmaRealAxisBlocks (Pade AC via the coremath Task B helper) then Vc.
        // (hard fact #5; legacy built sigcmat inline at task_qsgw.cpp:1370-1490.)
        // #6 (coremath review): sigc_is_ik_f_KS is BLACS-distributed (gw.cpp:2096 stores
        // the local block); build_sigma_real_axis_blocks_qsgw indexes i,j over n_bands so
        // it needs the FULL matrix. Correct as-is for serial / 1-proc BLACS (local==full);
        // parallel needs a gather (pdgemr2d to a 1x1 grid) or root-collect + bcast Vc —
        // TODO(D-review) with coremath on the preferred utility.
        const auto freq_nodes = pds->tfg.get_freq_nodes();
        SpinKMatrixMap Vc_all;
        for (int ispin = 0; ispin < n_spins; ++ispin)
            for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
            {
                const auto &sigc_spin_k = pds->p_g0w0->sigc_is_ik_f_KS.at(ispin).at(ikpt);
                const auto sigc_blocks = build_sigma_real_axis_blocks_qsgw(
                    mf, freq_nodes, sigc_spin_k, ispin, ikpt, n_bands, opts.n_params_anacon);
                Vc_all[ispin][ikpt] = build_correlation_potential_spin_k(sigc_blocks, n_bands);
            }

        // ---- H3: Hartree (Task C) — recompute from the updated density ----
        profiler.start("hartree_build", "Build Hartree kernel + KS projection");
        hartree.build(basis_aux_h, cs_h, VR_h);              // 3 params, no routing
        hartree.build_KS_kgrid0(qsgw_state);                 // H2 wfc0 anchor
        profiler.stop("hartree_build");
        const SpinKMatrixMap &Hartree_i = hartree.Hartree_is_ik_KS;
        SpinKMatrixMap Hartree_i_delta;
        for (int ispin = 0; ispin < n_spins; ++ispin)
            for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
            {
                if (iteration == 1)
                    Hartree_0[ispin][ikpt] = Hartree_i.at(ispin).at(ikpt); // iter-1 anchor
                else
                    Hartree_i_delta[ispin][ikpt] =
                        Hartree_i.at(ispin).at(ikpt) - Hartree_0.at(ispin).at(ikpt); // legacy L430
            }

        // iter>1: Vc includes the Hartree delta (legacy task_qsgw.cpp:725-728)
        if (iteration > 1)
            for (int ispin = 0; ispin < n_spins; ++ispin)
                for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
                    Vc_all[ispin][ikpt] = Vc_all[ispin][ikpt] + Hartree_i_delta[ispin][ikpt];

        // ---- Hexx (exchange) from the rebuilt p_exx (Exx::exx_KS, exx.h:56) ----
        const SpinKMatrixMap &Hexx_all = pds->p_exx->exx_KS;

        // ---- construct H0_GW (Task B; HROUND inside; H4 vxc0 + H8 via n_spinor) ----
        // NOTE: new signature drops the legacy meanfield param (qsgw_driver_design §3).
        SpinKMatrixMap H0_GW_all;
        if (hamiltonian_cut_above_fermi >= 0)
            H0_GW_all = construct_H0_GW_fermi_window(mf, H_KS0, vxc0, Hexx_all, Vc_all,
                                                     n_spins, n_kpoints, n_bands,
                                                     hamiltonian_cut_above_fermi,
                                                     hamiltonian_cut_diag_shift_ev);
        else
            H0_GW_all = construct_H0_GW(H_KS0, vxc0, Hexx_all, Vc_all,
                                        n_spins, n_kpoints, n_bands);

        // ---- Pulay mixing (Task A): pack H0_GW_all -> real matrix, mix, unpack ----
        // Legacy pack: all (spin,kpt) into one real matrix, width doubled [Re|Im]
        // (task_qsgw.cpp:1740-1800).
        // TODO(D): pack/unpack helpers; on iter<=linear_mixing_steps rely on the
        //          mixer's internal linear fallback (mixing.h: history_size<=1).
        if (!mixer.get_history_size()) // first step: initialize
        {
            // matrix mixed_input = pack(H0_GW_all); // TODO(D)
            // mixer.initialize(mixed_input);
        }
        else
        {
            // matrix mixed_input = pack(H0_GW_all);
            // matrix mixed_output = mixer.mix(mixed_input);
            // H0_GW_all = unpack(mixed_output);
        }

        // ---- H2: diagonalize using the fixed wfc0 anchor (Task B) ----
        diagonalize_and_store_fixed_basis(mf, H0_GW_all, qsgw_state,
                                          n_spins, n_kpoints, n_bands);

        // ---- fermi / occupations (Task A) ----
        efermi = calculate_fermi_energy(mf, temperature, total_electrons);
        update_fermi_energy_and_occupations(mf, temperature, efermi);

        // ---- convergence: focus-window sorted eigenvalue diff (legacy L1872-L1988) ----
        double eigdiff_for_conv = 0.0; // TODO(D): compute (focus window if eigdiff_focus_nbands>0)
        converged = (eigdiff_for_conv < eigenvalue_diff_tolerance);
        if (mpi_comm_global_h.is_root() && should_output())
            cout << "QSGW iter " << iteration << " eigdiff_for_conv = " << eigdiff_for_conv
                 << " eV" << (converged ? "  CONVERGED" : "") << endl;

        // ---- checkpoint save (legacy L2006) ----
        // TODO(D): if ((iter % qsgw_checkpoint_every == 0) || converged || iter==max)
        //   write_qsgw_checkpoint(..., H0_GW_all, efermi, &Hartree_0, &mixer);

        profiler.stop("qsgw_iter");

        // ---- MPI sync (legacy L2022-L2028) ----
        mpi_comm_global_h.barrier();
        mpi_comm_global_h.bcast(&converged, 1, 0); // MpiCommHandler::bcast (base_mpi.h:84)
        mpi_comm_global_h.barrier();
        // TODO(D): broadcast updated mf across ranks (legacy meanfield.broadcast(...))
        mpi_comm_global_h.barrier();
        if (converged || iteration == max_iterations) break;
    }

    // ---- optional HR bundle export (Task F, pure output) ----
    // SpinKMatrixMap H0_GW_all_final; // TODO(D): keep last H0_GW_all above scope
    // export_qsgw_hamiltonian_bundle(driver_params.output_dir + "qsgw_bundles/", mf,
    //                                kfrac_list, H0_GW_all_final, iteration, "kgrid");

    // ========================================================================
    // BAND SECTION (TODO(D)): mirror g0w0.cpp:257 pattern; read QSGW-band content
    // from legacy task_qsgw_band.cpp. efermi override: mf_band.efermi = efermi.
    // ========================================================================

    profiler.stop("qsgw");
}
