// QSGW self-consistent driver (Task D skeleton, for leader review).
//
// Single-file driver mirroring driver/tasks/g0w0.cpp. Implements the SCF loop
// integrating the ported QSGW modules (Tasks A/B/C/F) per qsgw_driver_design.md.
// Status: iter-1 driver skeleton — SCF structure + H1..H5/Hartree/SigC/EXX
// integration points are wired with real module APIs; mixing/convergence/
// checkpoint/velocity0/MPI-bcast details marked `TODO(D)` are deferred for the
// refine pass after review.
//
// Hard constraints (LEADER_AUDIT §3): H1 reset cached kernels each step;
// H2 QsgwState wfc0 anchor snapshot before first wfc mutation; H3 Hartree
// build + Hartree_0(iter1)/Hartree_i_delta(iter>1); H4 vxc0 = vxc_dft +
// hf_rotated; H5 full non-diagonal sigc via pds->p_g0w0->sigc_is_ik_f_KS.
//
// Legacy reference: driver/task_qsgw.cpp @ 7a7ff17f (line refs in design note).

#include <cmath>
#include <cstdlib>
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
#include "../../src/api/dataset_helper.h"  // initialize_ds_exx
#include "../../src/core/input_symmetry.h"
#include "../../src/core/coulmat.h"      // FT_Vq (Hartree Coulomb, H3)
#include "../../src/io/fs.h"
#include "../../src/io/global_io.h"
#include "../../src/io/input_elsi.h"   // load_matrix_cplx (full xc_matr CSC reader, H4)
#include "../../src/utils/constants.h" // HA2EV
#include "../../src/utils/profiler.h"
#include "../driver.h"
#include "../read_data.h"
#include "../task.h"
#include "task_helper.h"

// Ported QSGW modules (Tasks A/B/C/F)
#include "../../src/qsgw/qsgw_state.h"                 // QsgwState (H2)
#include "../../src/qsgw/fermi_energy_occupation.h"    // calculate_fermi_energy / update / calculate_total_weight
#include "../../src/qsgw/hamiltonian_qsgw.h"           // construct_H0_GW / diagonalize_and_store_fixed_basis / apply_qsgw_hround
#include "../../src/qsgw/correlation_potential.h"      // build_correlation_potential_spin_k
#include "../../src/qsgw/hartree.h"                    // Hartree (H3)
#include "../qsgw_io.h"                                // export_qsgw_hamiltonian_bundle (Task F)

namespace
{
using librpa_int::Matz;

struct MatrixChecksum
{
    int nr = 0, nc = 0;
    int major = 0;
    size_t nnz = 0;
    double sum_real = 0.0, sum_imag = 0.0;
    double sum_abs = 0.0, sum_sq = 0.0, max_abs = 0.0;
};

MatrixChecksum compute_checksum(const Matz &m, double threshold = 1e-15)
{
    MatrixChecksum c;
    c.nr = m.nr();
    c.nc = m.nc();
    c.major = static_cast<int>(m.major());
    const size_t n = m.size();
    for (size_t i = 0; i < n; ++i)
    {
        const auto v = m.ptr()[i];
        const double a = std::abs(v);
        if (a > threshold) c.nnz++;
        c.sum_real += v.real();
        c.sum_imag += v.imag();
        c.sum_abs += a;
        c.sum_sq += a * a;
        c.max_abs = std::max(c.max_abs, a);
    }
    return c;
}

void write_checksum_file(const MatrixChecksum &c, const std::string &fn,
                         const std::string &label)
{
    std::ofstream fs(fn);
    if (!fs)
        throw LIBRPA_RUNTIME_ERROR("QSGW dump: failed to open checksum file " + fn);
    fs << "# " << label << "\n";
    fs << "shape " << c.nr << " " << c.nc << "\n";
    fs << "major " << c.major << "\n";
    fs << "nnz " << c.nnz << "\n";
    fs << std::scientific << std::setprecision(15);
    fs << "sum_real " << c.sum_real << "\n";
    fs << "sum_imag " << c.sum_imag << "\n";
    fs << "sum_abs " << c.sum_abs << "\n";
    fs << "sum_sq " << c.sum_sq << "\n";
    fs << "max_abs " << c.max_abs << "\n";
    fs << "frobenius " << std::sqrt(c.sum_sq) << "\n";
}

void dump_matz(const Matz &m, const std::string &fn_base, const std::string &label)
{
    librpa_int::print_matrix_mm_file(m, fn_base + ".mm", label, 1e-15, true);
}
} // unnamed namespace

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
    const double mixing_beta = 0.2;
    const double temperature = 0.0;            // TODO(D): knob
    const int eigdiff_focus_nbands = 10;       // convergence focus window
    const double eigenvalue_diff_tolerance = 1e-5;
    const int hamiltonian_cut_above_fermi = -1; // <0 disables fermi_window variant
    const double hamiltonian_cut_diag_shift_ev = 0.0;

    // --- minimal env knob to force >1 iterations for molecule diagnostics ---
    int qsgw_min_iterations = 1;
    {
        const char *env_min_iter = std::getenv("QSGW_MIN_ITERATIONS");
        if (env_min_iter != nullptr)
        {
            try {
                qsgw_min_iterations = std::stoi(std::string(env_min_iter));
            } catch (...) {
                if (mpi_comm_global_h.is_root())
                    std::cerr << "[QSGW] Warning: invalid QSGW_MIN_ITERATIONS='"
                              << env_min_iter << "', keep default 1" << std::endl;
            }
        }
        if (qsgw_min_iterations <= 1)
            qsgw_min_iterations = 1;
        if (mpi_comm_global_h.is_root())
            std::cout << "[QSGW] min_iterations=" << qsgw_min_iterations << std::endl;
    }

    // --- env-gated iter-1 diagnostic dump (default off) ---
    bool qsgw_dump_iter1 = false;
    std::string qsgw_dump_dir;
    {
        const char *env_dump = std::getenv("QSGW_DUMP_ITER1");
        qsgw_dump_iter1 = (env_dump != nullptr && std::string(env_dump) == "1");
        const char *env_dir = std::getenv("QSGW_DUMP_DIR");
        if (env_dir != nullptr)
            qsgw_dump_dir = librpa_int::path_as_directory(std::string(env_dir));
        else
            qsgw_dump_dir = librpa_int::path_as_directory(std::string(opts.output_dir)) +
                            "qsgw_dump/";
        if (qsgw_dump_iter1)
        {
            const bool is_serial_blacs =
                (mpi_comm_global_h.nprocs == 1) &&
                (pds->blacs_h.nprocs == 1) &&
                (pds->blacs_h.nprows == 1) &&
                (pds->blacs_h.npcols == 1);
            if (!is_serial_blacs)
                throw LIBRPA_RUNTIME_ERROR(
                    "QSGW_DUMP_ITER1 requires serial / 1x1 BLACS execution; "
                    "parallel SigC gather is not yet implemented.");
        }
        if (qsgw_dump_iter1 && mpi_comm_global_h.is_root())
        {
            std::cout << "[QSGW] iter-1 diagnostic dump enabled: " << qsgw_dump_dir << std::endl;
            librpa_int::create_directories(qsgw_dump_dir.c_str(), 0);
        }
    }

    // --- env-gated Hartree-only diagnostic harness (default off) ---
    // Motivation: FHI-aims only writes IBZ xc_matr files, but Hartree only needs
    // the density matrix and fitted Coulomb vertices.  This path builds the new
    // qsgw::Hartree kernel for all full-BZ kpts and dumps Hartree_is_ik_KS
    // without reading xc_matr / SigC / Exx or entering the SCF loop.
    bool qsgw_hartree_only = false;
    std::string qsgw_hartree_only_dir;
    {
        const char *env_ho = std::getenv("QSGW_HARTREE_ONLY");
        qsgw_hartree_only = (env_ho != nullptr && std::string(env_ho) == "1");
        const char *env_ho_dir = std::getenv("QSGW_HARTREE_ONLY_DUMP_DIR");
        if (env_ho_dir != nullptr)
            qsgw_hartree_only_dir = librpa_int::path_as_directory(std::string(env_ho_dir));
        else
            qsgw_hartree_only_dir =
                librpa_int::path_as_directory(std::string(opts.output_dir)) +
                "hartree_only_dump/";
        if (qsgw_hartree_only)
        {
            const bool is_serial_blacs =
                (mpi_comm_global_h.nprocs == 1) &&
                (pds->blacs_h.nprocs == 1) &&
                (pds->blacs_h.nprows == 1) &&
                (pds->blacs_h.npcols == 1);
            if (!is_serial_blacs)
                throw LIBRPA_RUNTIME_ERROR(
                    "QSGW_HARTREE_ONLY requires serial / 1x1 BLACS execution.");
            if (mpi_comm_global_h.is_root())
            {
                std::cout << "[QSGW] Hartree-only harness enabled: "
                          << qsgw_hartree_only_dir << std::endl;
                librpa_int::create_directories(qsgw_hartree_only_dir.c_str(), 0);
            }
        }
    }
    if (qsgw_hartree_only)
    {
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
        profiler.start("hartree_build", "Build Hartree kernel + KS projection");
        hartree.build(basis_aux_h, cs_h, VR_h);
        hartree.build_KS_kgrid0(qsgw_state);
        profiler.stop("hartree_build");

        const SpinKMatrixMap &Hartree_i = hartree.Hartree_is_ik_KS;
        if (mpi_comm_global_h.is_root())
        {
            for (int ispin = 0; ispin < n_spins; ++ispin)
                for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
                {
                    std::ostringstream oss;
                    oss << qsgw_hartree_only_dir << "hartree_only_spin" << ispin
                        << "_kpt" << std::setw(6) << std::setfill('0') << (ikpt + 1);
                    dump_matz(Hartree_i.at(ispin).at(ikpt), oss.str(),
                              "Hartree_only spin=" + std::to_string(ispin) +
                              " kpt=" + std::to_string(ikpt));
                }
        }
        profiler.stop("qsgw_setup");
        profiler.stop("qsgw");
        return;
    }

    double efermi = mf.get_efermi();
    const double total_electrons = calculate_total_weight(mf); // replaces get_total_weight

    // --- (B)-fixed H4: DFT HF exchange from the kernel (no hf_exchange file) ---
    // coremath APPROVED with guardrails. Si G0W0 testcase has no hf_exchange_*.csc,
    // so source the DFT HF exchange from the new-arch Exx kernel instead. Build Exx
    // once at setup with the SAME settings as the SCF loop (compute_g0w0.cpp:628-645);
    // the mf here is the DFT (step-0) wfc, so exx_KS = DFT HF exchange in KS basis =
    // legacy hf_ks (sign/factor parity pending coremath bookkeeping + spike).
    initialize_ds_exx(*pds, opts);
    const bool use_shrink_exx = opts.use_shrink_abfs == LIBRPA_SWITCH_ON;
    const auto &basis_aux_exx = use_shrink_exx ? pds->basis_aux_shrink : pds->basis_aux;
    const auto &cs_data_exx = use_shrink_exx ? pds->cs_data_shrink : pds->cs_data;
    const auto &coul_exx = opts.use_fullcoul_exx ? pds->vq : pds->vq_cut;
    const auto VR_exx = librpa_int::FT_Vq(basis_aux_exx, pds->symmetry_context, coul_exx,
                                          pds->pbc, true,
                                          opts.use_symmetry_exx == LIBRPA_SWITCH_ON);
    pds->p_exx->build(routing, basis_aux_exx, cs_data_exx, VR_exx);
    pds->p_exx->build_KS_kgrid_blacs(pds->blacs_h);
    // Deep-copy exx_KS into hf0_ks (owning). matrix_m's copy ctor is shallow (shared
    // storage), and H1 resets/rebuilds p_exx each SCF step, so a shallow copy would be
    // invalidated. copy() detaches. (coremath guardrail: hf0_ks survives p_exx reset.)
    SpinKMatrixMap hf0_ks;
    for (const auto &sp : pds->p_exx->exx_KS)
        for (const auto &kp : sp.second)
            hf0_ks[sp.first][kp.first] = kp.second.copy();
    if (qsgw_dump_iter1 && mpi_comm_global_h.is_root())
    {
        const std::string base = qsgw_dump_dir + "hf0_ks_spin0_kpt0";
        dump_matz(hf0_ks.at(0).at(0), base, "hf0_ks spin=0 kpt=0");
    }

    // --- H_KS0 + vxc0 assembly (legacy task_qsgw.cpp L882-L905) ---
    // H_KS0[ispin][ikpt] = diag(eigvals) (KS band space). vxc0 = xc + hf0_ks (fixed
    // DFT HF). construct_H0_GW does H_KS - vxc0 + Hexx_iter + Vc; Hexx_iter comes from
    // the per-iteration Exx KS rotation in the loop (separate from this fixed hf0_ks).
    SpinKMatrixMap vxc0;
    SpinKMatrixMap H_KS0;
    for (int ispin = 0; ispin < n_spins; ++ispin)
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
        {
            Matz H_KS0_sk(n_bands, n_bands, MAJOR::COL);
            for (int ib = 0; ib < n_bands; ++ib)
                H_KS0_sk(ib, ib) = mf.get_eigenvals()[ispin](ikpt, ib);

            // #4 (coremath): full xc_matr (KS band space; H0_GW is full so off-diag xc
            // is kept). Pattern: xc_matr_spin_{S}_kpt_{000006K}.csc (no leading zero).
            std::ostringstream oss_xc;
            oss_xc << driver_params.input_dir << "xc_matr_spin_" << (ispin + 1)
                   << "_kpt_" << std::setw(6) << std::setfill('0') << (ikpt + 1) << ".csc";
            // (B)-fixed: vxc0 = xc + hf0_ks (fixed DFT HF from kernel, deep-copied).
            Matz xc_matr = load_matrix_cplx(oss_xc.str());
            Matz vxc0_sk = xc_matr + hf0_ks.at(ispin).at(ikpt);

            if (qsgw_dump_iter1 && ispin == 0 && ikpt == 0 && mpi_comm_global_h.is_root())
            {
                const std::string base = qsgw_dump_dir + "xc_matr_raw_spin0_kpt0";
                dump_matz(xc_matr, base, "xc_matr_raw spin=0 kpt=0");
                write_checksum_file(compute_checksum(xc_matr), base + ".chk",
                                    "xc_matr_raw spin=0 kpt=0");
                dump_matz(vxc0_sk, qsgw_dump_dir + "vxc0_spin0_kpt0", "vxc0 spin=0 kpt=0");
                dump_matz(H_KS0_sk, qsgw_dump_dir + "H_KS0_spin0_kpt0", "H_KS0 spin=0 kpt=0");
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

    // --- mixing (Task A): owning copy of previous mixed H0_GW for linear mix ---
    SpinKMatrixMap prev_H0_GW_all;

    // --- checkpoint restart (legacy L1069-L1093) ---
    int iteration = 0;
    bool converged = false;
    bool use_fixed_basis_iter_gt1 = true;
    if (const char *env_fb = std::getenv("QSGW_USE_FIXED_BASIS_ITER_GT1"))
    {
        if (std::string(env_fb) == "0")
            use_fixed_basis_iter_gt1 = false;
    }
    std::string qsgw_vc_mode = "B";
    if (const char *env_vc_mode = std::getenv("QSGW_VC_MODE"))
    {
        const std::string mode(env_vc_mode);
        if (mode == "A" || mode == "a")
            qsgw_vc_mode = "A";
    }
    if (mpi_comm_global_h.is_root())
        std::cout << "[QSGW] Vc mode=" << qsgw_vc_mode << std::endl;
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
        if (iteration == 1 || !use_fixed_basis_iter_gt1)
            pds->p_g0w0->build_sigc_matrix_KS_kgrid_blacs(pds->blacs_h);
        else
            pds->p_g0w0->build_sigc_matrix_KS_kgrid0_blacs(qsgw_state, pds->blacs_h);

        // Exx KS rotation (parallel to the sigc #1 fix; coremath final review):
        // build_g0w0_sigma builds Exx real-space only — compute_g0w0.cpp:644
        // build_KS_kgrid_blacs is commented out, so p_exx->exx_KS stays empty and
        // construct_H0_GW's Hexx_all (= p_exx->exx_KS) would be empty/out_of_range.
        // Project to KS here. iter-1 current=wfc0, so the current-basis call is right.
        // For iter>1, keep Hexx_iter in the fixed KS0 basis used by H_KS0/vxc0.
        if (iteration == 1 || !use_fixed_basis_iter_gt1)
            pds->p_exx->build_KS_kgrid_blacs(pds->blacs_h);
        else
            pds->p_exx->build_KS_kgrid0_blacs(qsgw_state, pds->blacs_h);

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
                Vc_all[ispin][ikpt] =
                    (qsgw_vc_mode == "A")
                        ? build_correlation_potential_spin_k_modeA(sigc_blocks, n_bands)
                        : build_correlation_potential_spin_k(sigc_blocks, n_bands);
                if (qsgw_dump_iter1 && iteration == 1 && ispin == 0 && ikpt == 0 &&
                    mpi_comm_global_h.is_root())
                {
                    dump_matz(Vc_all[ispin][ikpt], qsgw_dump_dir + "Vc_all_spin0_kpt0",
                              "Vc_all spin=0 kpt=0");
                }
            }

        // ---- H3: Hartree (Task C) — recompute from the updated density ----
        // #7 (coremath): Hartree::build has an is_rspace_built_ cache guard
        // (hartree.cpp:217); without reset, iter>1 build() returns early and
        // Hartree_i goes stale -> Hartree_i_delta silently 0. Reset both spaces
        // each step so build() + build_KS_kgrid0 recompute from the live density.
        hartree.reset_rspace();
        hartree.reset_kspace();
        profiler.start("hartree_build", "Build Hartree kernel + KS projection");
        hartree.build(basis_aux_h, cs_h, VR_h);              // 3 params, no routing
        hartree.build_KS_kgrid0(qsgw_state);                 // H2 wfc0 anchor
        profiler.stop("hartree_build");
        const SpinKMatrixMap &Hartree_i = hartree.Hartree_is_ik_KS;
        if (qsgw_dump_iter1 && iteration == 1 && mpi_comm_global_h.is_root())
        {
            dump_matz(Hartree_i.at(0).at(0), qsgw_dump_dir + "Hartree_i_spin0_kpt0",
                      "Hartree_i spin=0 kpt=0");
        }
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
        if (qsgw_dump_iter1 && iteration == 1 && mpi_comm_global_h.is_root())
        {
            dump_matz(Hexx_all.at(0).at(0), qsgw_dump_dir + "Hexx_iter_spin0_kpt0",
                      "Hexx_iter spin=0 kpt=0");
        }

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
        if (qsgw_dump_iter1 && iteration == 1 && mpi_comm_global_h.is_root())
        {
            dump_matz(H0_GW_all.at(0).at(0), qsgw_dump_dir + "H0_GW_all_spin0_kpt0",
                      "H0_GW_all spin=0 kpt=0");
        }

        // ---- Linear mixing (Task A): H_new = H_old + beta * (H_current - H_old) ----
        double max_abs_delta = 0.0;
        if (iteration == 1)
        {
            // iter-1: store current H0_GW_all as the mixing anchor, no modification.
            for (int ispin = 0; ispin < n_spins; ++ispin)
                for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
                    prev_H0_GW_all[ispin][ikpt] = H0_GW_all[ispin][ikpt].copy();
        }
        else
        {
            for (int ispin = 0; ispin < n_spins; ++ispin)
                for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
                {
                    Matz &H = H0_GW_all[ispin][ikpt];
                    const Matz &H_prev = prev_H0_GW_all[ispin][ikpt];
                    for (int i = 0; i < n_bands; ++i)
                        for (int j = 0; j < n_bands; ++j)
                        {
                            const cplxdb delta = H(i, j) - H_prev(i, j);
                            max_abs_delta = std::max(max_abs_delta, std::abs(delta));
                            H(i, j) = H_prev(i, j) + mixing_beta * delta;
                        }
                    prev_H0_GW_all[ispin][ikpt] = H.copy();
                }
            if (mpi_comm_global_h.is_root() && should_output())
                cout << "QSGW iter " << iteration << " linear mix beta=" << mixing_beta
                     << " max_abs_delta_H=" << max_abs_delta << endl;
        }

        // ---- H2: diagonalize using the fixed wfc0 anchor (Task B) ----
        diagonalize_and_store_fixed_basis(mf, H0_GW_all, qsgw_state,
                                          n_spins, n_kpoints, n_bands);

        // ---- fermi / occupations (Task A) ----
        efermi = calculate_fermi_energy(mf, temperature, total_electrons);
        update_fermi_energy_and_occupations(mf, temperature, efermi);

        // ---- convergence: focus-window sorted eigenvalue diff (legacy L1872-L1988) ----
        double eigdiff_for_conv = 0.0; // TODO(D): compute (focus window if eigdiff_focus_nbands>0)
        converged = (iteration >= qsgw_min_iterations) && (eigdiff_for_conv < eigenvalue_diff_tolerance);
        if (mpi_comm_global_h.is_root() && should_output())
            cout << "QSGW iter " << iteration << " min_iter=" << qsgw_min_iterations
                 << " eigdiff_for_conv = " << eigdiff_for_conv
                 << " eV" << (converged ? "  CONVERGED" : "") << endl;

        // ---- checkpoint save (legacy L2006) ----
        // TODO(D): if ((iter % qsgw_checkpoint_every == 0) || converged || iter==max)
        //   write_qsgw_checkpoint(..., H0_GW_all, efermi, &Hartree_0, &mixer);

        profiler.stop("qsgw_iter");

        // ---- MPI sync (legacy L2022-L2028) ----
        mpi_comm_global_h.barrier();
        // bcast int, not bool: mpi_datatype has no bool specialization (base_mpi.h:86,
        // caught by Fisherd compile-test). int has the specialization.
        int conv_int = converged ? 1 : 0;
        mpi_comm_global_h.bcast(&conv_int, 1, 0);
        converged = conv_int;
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
