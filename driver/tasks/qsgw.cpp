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

#include <algorithm>
#include <cmath>
#include <cctype>
#include <complex>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <librpa_enums.h>

#include "../../src/api/instance_manager.h" // get_dataset_instance
#include "../../src/api/dataset_helper.h"  // initialize_ds_exx
#include "../../src/core/coulmat.h"      // FT_Vq (Hartree Coulomb, H3)
#include "../../src/io/fs.h"
#include "../../src/io/global_io.h"
#include "../../src/io/input_elsi.h"   // load_matrix_cplx (full xc_matr CSC reader, H4)
#include "../../src/io/stl_io_helper.h"
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
#include "../../src/qsgw/mixing.h"                     // PulayMixer
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

void dump_complex_matrix(const librpa_int::ComplexMatrix &m,
                         const std::string &fn_base,
                         const std::string &label)
{
    Matz out(m.nr, m.nc, librpa_int::MAJOR::COL);
    for (int ir = 0; ir < m.nr; ++ir)
        for (int ic = 0; ic < m.nc; ++ic)
            out(ir, ic) = m(ir, ic);
    dump_matz(out, fn_base, label);
}

bool read_env_bool(const char *name, const bool default_value)
{
    const char *env = std::getenv(name);
    if (env == nullptr)
        return default_value;
    const std::string value(env);
    if (value == "0" || value == "false" || value == "FALSE" ||
        value == "off" || value == "OFF")
        return false;
    return true;
}

double read_env_double(const char *name, const double default_value)
{
    const char *env = std::getenv(name);
    if (env == nullptr)
        return default_value;
    try {
        return std::stod(std::string(env));
    } catch (...) {
        return default_value;
    }
}

int read_env_int(const char *name, const int default_value)
{
    const char *env = std::getenv(name);
    if (env == nullptr)
        return default_value;
    try {
        return std::stoi(std::string(env));
    } catch (...) {
        return default_value;
    }
}

Matz scaled_matz(const Matz &m, const double scale)
{
    Matz out = m.copy();
    for (size_t i = 0; i < out.size(); ++i)
        out.ptr()[i] *= scale;
    return out;
}

librpa_int::qsgw::SpinKMatrixMap scaled_spin_k_map(
    const librpa_int::qsgw::SpinKMatrixMap &in,
    const double scale)
{
    librpa_int::qsgw::SpinKMatrixMap out;
    for (const auto &sp : in)
        for (const auto &kp : sp.second)
            out[sp.first][kp.first] = scaled_matz(kp.second, scale);
    return out;
}

void scale_spin_k_map_inplace(librpa_int::qsgw::SpinKMatrixMap &in, const double scale)
{
    if (scale == 1.0)
        return;
    for (auto &sp : in)
        for (auto &kp : sp.second)
            kp.second = scaled_matz(kp.second, scale);
}

void apply_active_band_window(
    librpa_int::qsgw::SpinKMatrixMap &H0_GW_all,
    const librpa_int::qsgw::SpinKMatrixMap &H_KS0,
    const int n_spins,
    const int n_kpoints,
    const int n_bands,
    const int active_band_min,
    const int active_band_max)
{
    if (active_band_min <= 0 && active_band_max >= n_bands - 1)
        return;

    for (int ispin = 0; ispin < n_spins; ++ispin)
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
        {
            Matz &H0 = H0_GW_all.at(ispin).at(ikpt);
            const Matz &H_KS = H_KS0.at(ispin).at(ikpt);
            for (int i = 0; i < n_bands; ++i)
                for (int j = 0; j < n_bands; ++j)
                {
                    const bool active_i = (i >= active_band_min && i <= active_band_max);
                    const bool active_j = (j >= active_band_min && j <= active_band_max);
                    if (active_i && active_j)
                        continue;
                    H0(i, j) = (i == j) ? H_KS(i, i) : librpa_int::cplxdb(0.0, 0.0);
                }
        }
}

std::string normalized_qsgw_mixer(std::string mode)
{
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (mode == "diis")
        mode = "pulay";
    if (mode != "linear" && mode != "pulay")
    {
        throw LIBRPA_RUNTIME_ERROR(
            "Unknown qsgw_mixer (" + mode + "). Available values: linear, pulay/diis.");
    }
    return mode;
}

librpa_int::matrix pack_h0_gw_for_mixing(
    const librpa_int::qsgw::SpinKMatrixMap &H0_GW_all,
    const int n_spins,
    const int n_kpoints,
    const int n_bands)
{
    librpa_int::matrix mixed_input(n_spins * n_kpoints * n_bands,
                                   2 * n_bands, true);
    int row_offset = 0;
    for (int ispin = 0; ispin < n_spins; ++ispin)
    {
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
        {
            const Matz &mat = H0_GW_all.at(ispin).at(ikpt);
            for (int i = 0; i < n_bands; ++i)
                for (int j = 0; j < n_bands; ++j)
                {
                    const auto value = mat(i, j);
                    mixed_input(row_offset + i, j) = value.real();
                    mixed_input(row_offset + i, j + n_bands) = value.imag();
                }
            row_offset += n_bands;
        }
    }
    return mixed_input;
}

void unpack_h0_gw_from_mixing(
    const librpa_int::matrix &mixed_output,
    librpa_int::qsgw::SpinKMatrixMap &H0_GW_all,
    const int n_spins,
    const int n_kpoints,
    const int n_bands)
{
    int row_offset = 0;
    for (int ispin = 0; ispin < n_spins; ++ispin)
    {
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
        {
            Matz &mat = H0_GW_all.at(ispin).at(ikpt);
            for (int i = 0; i < n_bands; ++i)
                for (int j = 0; j < n_bands; ++j)
                    mat(i, j) = librpa_int::cplxdb(
                        mixed_output(row_offset + i, j),
                        mixed_output(row_offset + i, j + n_bands));
            row_offset += n_bands;
        }
    }
}

librpa_int::matrix linear_mix_h0(
    const librpa_int::matrix &previous_input,
    const librpa_int::matrix &current_output,
    const double beta)
{
    if (previous_input.nr != current_output.nr ||
        previous_input.nc != current_output.nc)
        throw LIBRPA_RUNTIME_ERROR("QSGW linear mixing matrix dimensions mismatch");

    librpa_int::matrix mixed_output(current_output.nr, current_output.nc, true);
    for (int i = 0; i < current_output.nr; ++i)
        for (int j = 0; j < current_output.nc; ++j)
            mixed_output(i, j) =
                (1.0 - beta) * previous_input(i, j) + beta * current_output(i, j);
    return mixed_output;
}

double max_abs_matrix_delta(
    const librpa_int::matrix &a,
    const librpa_int::matrix &b)
{
    if (a.nr != b.nr || a.nc != b.nc)
        throw LIBRPA_RUNTIME_ERROR("QSGW mixing delta matrix dimensions mismatch");

    double max_abs_delta = 0.0;
    for (int i = 0; i < a.nr; ++i)
        for (int j = 0; j < a.nc; ++j)
            max_abs_delta = std::max(max_abs_delta, std::abs(a(i, j) - b(i, j)));
    return max_abs_delta;
}

bool load_abacus_vxck_nao_matrix(
    const std::string &file_path,
    const int n_spins,
    Matz &matrix)
{
    if (!librpa_int::file_exists(file_path))
        return false;

    std::ifstream input(file_path);
    if (!input)
        throw LIBRPA_RUNTIME_ERROR("Cannot open ABACUS Vxc matrix " + file_path);

    int rows = -1;
    int cols = -1;
    int current_row = -1;
    int current_col = -1;
    std::vector<int> entry_rows;
    std::vector<int> entry_cols;
    std::vector<librpa_int::cplxdb> entry_values;
    std::vector<librpa_int::cplxdb> dense_values;
    std::string line;
    while (std::getline(input, line))
    {
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos)
            continue;
        const std::string content = line.substr(first);

        if (content.rfind("# rows", 0) == 0)
        {
            std::istringstream iss(content);
            std::string hash;
            std::string key;
            iss >> hash >> key >> rows;
            continue;
        }
        if (content.rfind("# columns", 0) == 0)
        {
            std::istringstream iss(content);
            std::string hash;
            std::string key;
            iss >> hash >> key >> cols;
            continue;
        }
        if (content[0] == '#')
            continue;
        if (content.rfind("Row ", 0) == 0)
        {
            std::istringstream iss(content);
            std::string row_label;
            int row = 0;
            iss >> row_label >> row;
            current_row = row - 1;
            current_col = current_row;
            continue;
        }

        size_t pos = 0;
        while (true)
        {
            const auto lb = content.find('(', pos);
            if (lb == std::string::npos)
                break;
            const auto comma = content.find(',', lb + 1);
            const auto rb = content.find(')', comma == std::string::npos ? lb + 1 : comma + 1);
            if (comma == std::string::npos || rb == std::string::npos)
                throw LIBRPA_RUNTIME_ERROR("Malformed ABACUS complex value in " + file_path);

            const double real = std::stod(content.substr(lb + 1, comma - lb - 1));
            const double imag = std::stod(content.substr(comma + 1, rb - comma - 1));
            if (current_row >= 0)
            {
                entry_rows.emplace_back(current_row);
                entry_cols.emplace_back(current_col);
                entry_values.emplace_back(real, imag);
                ++current_col;
            }
            else
            {
                dense_values.emplace_back(real, imag);
            }
            pos = rb + 1;
        }
    }

    if (rows <= 0 || cols <= 0)
        throw LIBRPA_RUNTIME_ERROR("ABACUS Vxc matrix header is incomplete in " + file_path);

    matrix = Matz(rows, cols, librpa_int::MAJOR::COL);
    const double scale = 1.0 / static_cast<double>(std::max(1, n_spins));
    if (!entry_values.empty())
    {
        const size_t expected_triangular =
            rows == cols ? static_cast<size_t>(rows) * static_cast<size_t>(rows + 1) / 2 : 0;
        if (entry_values.size() != expected_triangular)
        {
            std::ostringstream oss;
            oss << "ABACUS triangular Vxc matrix entry count mismatch in " << file_path
                << ": got " << entry_values.size() << " values, expected "
                << expected_triangular;
            throw LIBRPA_RUNTIME_ERROR(oss.str());
        }
        for (size_t i = 0; i < entry_values.size(); ++i)
        {
            const int ir = entry_rows[i];
            const int ic = entry_cols[i];
            if (ir < 0 || ir >= rows || ic < 0 || ic >= cols)
            {
                std::ostringstream oss;
                oss << "ABACUS Vxc triangular index out of range in " << file_path
                    << ": (" << (ir + 1) << ", " << (ic + 1) << ") for "
                    << rows << "x" << cols;
                throw LIBRPA_RUNTIME_ERROR(oss.str());
            }
            const auto value = entry_values[i] * scale;
            matrix(ir, ic) = value;
            if (ir != ic)
                matrix(ic, ir) = std::conj(value);
        }
    }
    else if (dense_values.size() == static_cast<size_t>(rows) * static_cast<size_t>(cols))
    {
        for (int ir = 0; ir < rows; ++ir)
            for (int ic = 0; ic < cols; ++ic)
                matrix(ir, ic) = dense_values[static_cast<size_t>(ir) * cols + ic] * scale;
    }
    else
    {
        std::ostringstream oss;
        oss << "ABACUS Vxc matrix entry count mismatch in " << file_path
            << ": got " << dense_values.size() << " dense values for "
            << rows << "x" << cols;
        throw LIBRPA_RUNTIME_ERROR(oss.str());
    }

    return true;
}

void write_homo_lumo_iteration(
    const librpa_int::MeanField &mf,
    const double efermi,
    const int iteration)
{
    double homo = -std::numeric_limits<double>::infinity();
    double lumo = std::numeric_limits<double>::infinity();
    const double occupied_threshold =
        1.0 / static_cast<double>(mf.get_n_spins() * mf.get_n_kpoints());

    for (int ispin = 0; ispin < mf.get_n_spins(); ++ispin)
    {
        for (int ikpt = 0; ikpt < mf.get_n_kpoints(); ++ikpt)
        {
            int homo_level = -1;
            for (int ib = 0; ib < mf.get_n_bands(); ++ib)
            {
                if (mf.get_weight()[ispin](ikpt, ib) >= occupied_threshold)
                    homo_level = ib;
            }

            if (homo_level < 0)
                continue;

            homo = std::max(homo, mf.get_eigenvals()[ispin](ikpt, homo_level));
            if (homo_level + 1 < mf.get_n_bands())
            {
                lumo = std::min(
                    lumo,
                    mf.get_eigenvals()[ispin](ikpt, homo_level + 1));
            }
        }
    }

    if (!std::isfinite(homo) || !std::isfinite(lumo))
    {
        throw LIBRPA_RUNTIME_ERROR(
            "QSGW failed to locate finite HOMO/LUMO levels for iteration output");
    }

    std::ofstream file(
        "homo_lumo_vs_iterations.dat",
        iteration == 0 ? std::ios::trunc : std::ios::app);
    file << std::setprecision(12)
         << iteration << " "
         << homo * librpa_int::HA2EV << " "
         << lumo * librpa_int::HA2EV << " "
         << efermi * librpa_int::HA2EV << std::endl;

    std::cout << (iteration == 0 ? "Initial" : "Iteration")
              << " " << iteration
              << ": HOMO = " << homo * librpa_int::HA2EV << " eV, "
              << "LUMO = " << lumo * librpa_int::HA2EV << " eV, "
              << "Efermi = " << efermi * librpa_int::HA2EV << " eV"
              << std::endl;
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
    const bool qsgw_serial_blacs =
        (mpi_comm_global_h.nprocs == 1) &&
        (pds->blacs_h.nprocs == 1) &&
        (pds->blacs_h.nprows == 1) &&
        (pds->blacs_h.npcols == 1);
    if (!qsgw_serial_blacs)
    {
        throw LIBRPA_RUNTIME_ERROR(
            "QSGW driver currently requires serial / 1x1 BLACS execution. "
            "The Vc/H0 path consumes full KS matrices, while parallel "
            "SigC gather/broadcast is not implemented yet.");
    }

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

    const auto file_df = driver_params.input_dir + driver_params.fn_dielfunc;
    const bool compute_headwing =
        driver::get_bool(opts.replace_w_head) &&
        (opts.option_dielect_func == 3 || opts.option_dielect_func == 4);
    if (compute_headwing)
    {
        read_headwing_input(driver_params.input_dir, opts.option_dielect_func == 3);
    }
    else if (driver::get_bool(opts.replace_w_head) && librpa_int::path_exists(file_df.c_str()))
    {
        if (mpi_comm_global_h.is_root() && should_output())
            std::cout << "Reading dielectric function for head correction" << std::endl;
        std::vector<double> omegas_dielect;
        std::vector<double> dielect_func;
        read_dielec_func(file_df, omegas_dielect, dielect_func);
        ofs_myid << "Dielectric functions read:" << std::endl;
        ofs_myid << "omegas_dielect: " << omegas_dielect << std::endl;
        ofs_myid << "dielect_func:   " << dielect_func << std::endl;
        h.set_dielect_func_imagfreq(omegas_dielect, dielect_func);
    }

    // --- DFT xc: loaded per (spin,kpt) as the FULL xc_matr CSC below (H4 #4 fix,
    //     coremath review: H0_GW is a full matrix, off-diagonal xc must be kept;
    //     the diagonal read_vxc path drops off-diag). hf added in the same loop. ---

    // --- H2: snapshot step-0 anchors BEFORE the loop mutates wfc ---
    QsgwState qsgw_state;
    qsgw_state.snapshot_wfc0(mf);
    qsgw_state.snapshot_wg0(mf);
    if (!pds->velocity_matrix.empty())
        qsgw_state.snapshot_velocity0(pds->velocity_matrix);

    // --- QSGW knobs parsed from librpa.in into DriverParams; env overrides below are
    //     retained only for ad-hoc diagnostics. ---
    std::string qsgw_mixer_mode = normalized_qsgw_mixer(driver_params.qsgw_mixer);
    double mixing_beta = driver_params.qsgw_mixing_beta;
    int mixing_history = driver_params.qsgw_mixing_history;
    int linear_mixing_steps = driver_params.qsgw_linear_mixing_steps;
    const double temperature = 0.0;            // TODO(D): knob
    const int eigdiff_focus_nbands = 10;       // convergence focus window
    const double eigenvalue_diff_tolerance = 1e-5;
    const int hamiltonian_cut_above_fermi = -1; // <0 disables fermi_window variant
    const double hamiltonian_cut_diag_shift_ev = 0.0;

    // Environment overrides are retained only for ad-hoc diagnostics.  CI and
    // regression tests should set these in librpa.in.
    if (const char *env_mixer = std::getenv("LIBRPA_QSGW_MIXER"))
        qsgw_mixer_mode = normalized_qsgw_mixer(env_mixer);
    mixing_beta = read_env_double("LIBRPA_QSGW_MIXING_BETA", mixing_beta);
    mixing_history = read_env_int("LIBRPA_QSGW_MIXING_HISTORY", mixing_history);
    linear_mixing_steps =
        read_env_int("LIBRPA_QSGW_LINEAR_MIXING_STEPS", linear_mixing_steps);

    if (!(mixing_beta > 0.0))
        throw LIBRPA_RUNTIME_ERROR("qsgw_mixing_beta must be positive");
    if (mixing_history < 1)
        throw LIBRPA_RUNTIME_ERROR("qsgw_mixing_history must be at least 1");
    if (linear_mixing_steps < 0)
        throw LIBRPA_RUNTIME_ERROR("qsgw_linear_mixing_steps must be non-negative");
    if (mpi_comm_global_h.is_root())
        std::cout << "[QSGW] mixer=" << qsgw_mixer_mode
                  << " history=" << mixing_history
                  << " beta=" << mixing_beta
                  << " linear_warmup_steps=" << linear_mixing_steps << std::endl;

    // --- minimal env knob to force >1 iterations for molecule diagnostics ---
    int qsgw_min_iterations = driver_params.qsgw_min_iter;
    {
        const char *env_min_iter = std::getenv("QSGW_MIN_ITERATIONS");
        if (env_min_iter != nullptr)
        {
            try {
                qsgw_min_iterations = std::stoi(std::string(env_min_iter));
            } catch (...) {
                if (mpi_comm_global_h.is_root())
                    std::cerr << "[QSGW] Warning: invalid QSGW_MIN_ITERATIONS='"
                              << env_min_iter << "', keep "
                              << qsgw_min_iterations << std::endl;
            }
        }
        if (qsgw_min_iterations <= 1)
            qsgw_min_iterations = 1;
        if (mpi_comm_global_h.is_root())
            std::cout << "[QSGW] min_iterations=" << qsgw_min_iterations << std::endl;
    }
    int qsgw_max_iterations = driver_params.qsgw_max_iter;
    qsgw_max_iterations = read_env_int("QSGW_MAX_ITERATIONS", qsgw_max_iterations);
    if (qsgw_max_iterations < qsgw_min_iterations)
    {
        if (mpi_comm_global_h.is_root())
            std::cerr << "[QSGW] Warning: QSGW_MAX_ITERATIONS="
                      << qsgw_max_iterations
                      << " is smaller than min_iterations; raising to "
                      << qsgw_min_iterations << std::endl;
        qsgw_max_iterations = qsgw_min_iterations;
    }
    if (mpi_comm_global_h.is_root())
        std::cout << "[QSGW] max_iterations=" << qsgw_max_iterations << std::endl;

    // --- iter-1 diagnostic dump, used by QSGW regression tests ---
    bool qsgw_dump_iter1 = driver_params.qsgw_dump_iter1;
    bool qsgw_dump_sigc_ks = false;
    std::string qsgw_dump_dir;
    {
        const char *env_dump = std::getenv("QSGW_DUMP_ITER1");
        if (env_dump != nullptr)
            qsgw_dump_iter1 = (std::string(env_dump) == "1");
        qsgw_dump_sigc_ks = read_env_bool("QSGW_SIGC_KS_DUMP", false);
        const char *env_dir = std::getenv("QSGW_DUMP_DIR");
        if (env_dir != nullptr)
            qsgw_dump_dir = librpa_int::path_as_directory(std::string(env_dir));
        else if (!driver_params.qsgw_dump_dir.empty())
            qsgw_dump_dir = librpa_int::path_as_directory(driver_params.qsgw_dump_dir);
        else
            qsgw_dump_dir = librpa_int::path_as_directory(std::string(opts.output_dir)) +
                            "qsgw_dump/";
        if (qsgw_dump_iter1)
        {
            if (!qsgw_serial_blacs)
                throw LIBRPA_RUNTIME_ERROR(
                    "QSGW_DUMP_ITER1 requires serial / 1x1 BLACS execution; "
                    "parallel SigC gather is not yet implemented.");
        }
        if (qsgw_dump_iter1 && mpi_comm_global_h.is_root())
        {
            std::cout << "[QSGW] iter-1 diagnostic dump enabled: " << qsgw_dump_dir << std::endl;
            librpa_int::create_directories(qsgw_dump_dir.c_str(), 0);
            if (const auto *wfc0 = qsgw_state.find_wfc0(0, 0, 0))
                dump_complex_matrix(*wfc0, qsgw_dump_dir + "wfc0_spin0_kpt0",
                                    "wfc0 spin=0 spinor=0 kpt=0");
        }
    }

    // --- Hartree-only diagnostic harness disabled on this branch ---
    bool qsgw_hartree_only = false;
    {
        const char *env_ho = std::getenv("QSGW_HARTREE_ONLY");
        qsgw_hartree_only = (env_ho != nullptr && std::string(env_ho) == "1");
        if (qsgw_hartree_only)
        {
            throw LIBRPA_RUNTIME_ERROR(
                "QSGW_HARTREE_ONLY is disabled while the QSGW driver is testing the "
                "non-Hartree path.");
        }
    }

    double efermi = mf.get_efermi();
    const double total_electrons = calculate_total_weight(mf); // replaces get_total_weight
    if (mpi_comm_global_h.is_root())
        write_homo_lumo_iteration(mf, efermi, 0);

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

    bool qsgw_vxc0_with_hf = driver_params.qsgw_vxc0_with_hf;
    qsgw_vxc0_with_hf = read_env_bool("QSGW_VXC0_WITH_HF", qsgw_vxc0_with_hf);
    if (mpi_comm_global_h.is_root())
        std::cout << "[QSGW] vxc0_with_hf=" << (qsgw_vxc0_with_hf ? 1 : 0) << std::endl;

    std::vector<matrix> diag_vxc_scf;
    const std::string diag_vxc_scf_path = driver_params.input_dir + driver_params.fn_vxc_scf;
    const int flag_read_diag_vxc_scf = read_vxc(diag_vxc_scf_path, diag_vxc_scf);
    bool diag_vxc_scf_available = (flag_read_diag_vxc_scf == 0) &&
                                  (static_cast<int>(diag_vxc_scf.size()) >= n_spins);
    if (diag_vxc_scf_available)
    {
        for (int ispin = 0; ispin < n_spins; ++ispin)
        {
            if (diag_vxc_scf[ispin].nr < n_kpoints || diag_vxc_scf[ispin].nc < n_bands)
            {
                diag_vxc_scf_available = false;
                break;
            }
        }
    }
    if (flag_read_diag_vxc_scf == 0 && !diag_vxc_scf_available &&
        mpi_comm_global_h.is_root())
    {
        std::cerr << "[QSGW] Warning: diagonal Vxc fallback file "
                  << diag_vxc_scf_path
                  << " has dimensions incompatible with QSGW matrix shape; "
                  << "full xc_matr_spin_* input will still be required." << std::endl;
    }
    bool diag_vxc_scf_used = false;
    bool abacus_vxck_nao_used = false;

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
            Matz xc_matr;
            if (librpa_int::file_exists(oss_xc.str()))
            {
                xc_matr = load_matrix_cplx(oss_xc.str());
            }
            else
            {
                std::ostringstream oss_abacus_vxc;
                oss_abacus_vxc << driver_params.input_dir << "OUT.ABACUS/vxck"
                               << (ikpt + 1) << "s" << (ispin + 1) << "_nao.txt";
                if (load_abacus_vxck_nao_matrix(oss_abacus_vxc.str(), n_spins, xc_matr))
                {
                    if (!abacus_vxck_nao_used && mpi_comm_global_h.is_root())
                    {
                        std::cout << "[QSGW] using ABACUS full Vxc matrices from "
                                  << driver_params.input_dir
                                  << "OUT.ABACUS/vxck{k}s{s}_nao.txt" << std::endl;
                    }
                    abacus_vxck_nao_used = true;
                }
            }
            if (xc_matr.size() == 0 && diag_vxc_scf_available)
            {
                xc_matr = Matz(n_bands, n_bands, MAJOR::COL);
                for (int ib = 0; ib < n_bands; ++ib)
                    xc_matr(ib, ib) = librpa_int::cplxdb(diag_vxc_scf[ispin](ikpt, ib), 0.0);
                if (!diag_vxc_scf_used && mpi_comm_global_h.is_root())
                {
                    std::cout << "[QSGW] Warning: using diagonal Vxc fallback from "
                              << diag_vxc_scf_path
                              << "; full QSGW matrix input is preferred." << std::endl;
                }
                diag_vxc_scf_used = true;
            }
            if (xc_matr.size() == 0)
            {
                throw LIBRPA_RUNTIME_ERROR(
                    "QSGW Vxc input not found: missing " + oss_xc.str() +
                    ", no ABACUS full Vxc matrix, and no compatible diagonal fallback " +
                    diag_vxc_scf_path);
            }
            if (xc_matr.nr() != n_bands || xc_matr.nc() != n_bands)
            {
                std::ostringstream oss;
                oss << "QSGW xc_matr dimension mismatch for " << oss_xc.str()
                    << " (spin=" << ispin << ", kpt=" << ikpt << "): got "
                    << xc_matr.nr() << "x" << xc_matr.nc()
                    << ", expected " << n_bands << "x" << n_bands;
                throw std::runtime_error(oss.str());
            }
            Matz vxc0_sk =
                qsgw_vxc0_with_hf ? xc_matr + hf0_ks.at(ispin).at(ikpt) : xc_matr.copy();

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
    if (mpi_comm_global_h.is_root())
    {
        std::cout << "[QSGW] Hartree update disabled; testing H_KS0 - vxc0 + Hexx + Vc"
                  << std::endl;
    }

    // --- mixing (Task A): owning copy of previous mixed H0_GW ---
    librpa_int::matrix previous_mixed_h0;
    bool previous_mixed_h0_initialized = false;
    librpa_int::PulayMixer mixer(mixing_history, mixing_beta);
    bool mixer_initialized = false;

    // --- checkpoint restart (legacy L1069-L1093) ---
    int iteration = 0;
    bool converged = false;
    bool use_fixed_basis_qsgw = true;
    if (const char *env_fb = std::getenv("QSGW_USE_FIXED_BASIS_ITER_GT1"))
    {
        if (std::string(env_fb) == "0")
            use_fixed_basis_qsgw = false;
    }
    std::string qsgw_vc_mode = "B";
    if (const char *env_vc_mode = std::getenv("QSGW_VC_MODE"))
    {
        const std::string mode(env_vc_mode);
        if (mode == "A" || mode == "a")
            qsgw_vc_mode = "A";
    }
    const double qsgw_exx_scale = read_env_double("QSGW_EXX_SCALE", 1.0);
    const double qsgw_vc_scale = read_env_double("QSGW_VC_SCALE", 1.0);
    const int active_band_min_1b = read_env_int("QSGW_ACTIVE_BAND_MIN", 1);
    const int active_band_max_1b = read_env_int("QSGW_ACTIVE_BAND_MAX", n_bands);
    if (active_band_min_1b < 1 || active_band_max_1b > n_bands ||
        active_band_min_1b > active_band_max_1b)
    {
        std::ostringstream oss;
        oss << "Invalid QSGW active band window: QSGW_ACTIVE_BAND_MIN="
            << active_band_min_1b << " QSGW_ACTIVE_BAND_MAX=" << active_band_max_1b
            << ", valid range is 1.." << n_bands;
        throw LIBRPA_RUNTIME_ERROR(oss.str());
    }
    const int active_band_min = active_band_min_1b - 1;
    const int active_band_max = active_band_max_1b - 1;
    const bool qsgw_uses_band_window =
        (active_band_min != 0 || active_band_max != n_bands - 1);
    if (qsgw_uses_band_window && !use_fixed_basis_qsgw)
    {
        throw LIBRPA_RUNTIME_ERROR(
            "QSGW active band window requires QSGW_USE_FIXED_BASIS_ITER_GT1=1 "
            "because the window is applied in the fixed KS0 band basis");
    }
    if (mpi_comm_global_h.is_root())
    {
        std::cout << "[QSGW] Vc mode=" << qsgw_vc_mode << std::endl;
        std::cout << "[QSGW] exx_scale=" << qsgw_exx_scale
                  << " vc_scale=" << qsgw_vc_scale << std::endl;
        if (qsgw_uses_band_window)
            std::cout << "[QSGW] active_band_window=" << active_band_min_1b
                      << ".." << active_band_max_1b
                      << " (bands outside this range stay at KS0 diagonal)" << std::endl;
    }
    // TODO(D): if (qsgw_restart) load_qsgw_checkpoint(...) -> restore
    //          H0_GW_all/Hartree_0/efermi/mixer + diagonalize_and_store_fixed_basis.

    profiler.stop("qsgw_setup");

    // ========================================================================
    // SCF LOOP (legacy task_qsgw.cpp L1141 `while(!converged && iteration<max)`)
    // ========================================================================
    while (!converged && iteration < qsgw_max_iterations)
    {
        iteration++;
        profiler.start("qsgw_iter", "QSGW SCF iteration", true);

        // ---- H1: reset cached kernels so they rebuild with the updated mf ----
        // p_exx has a real guard (compute_g0w0.cpp:628); p_headwing caches an mf
        // copy. p_chi0/p_g0w0 rebuild unconditionally. (LEADER_AUDIT §3 H1)
        pds->p_exx.reset();
        if (compute_headwing)
        {
            read_headwing_input(driver_params.input_dir, opts.option_dielect_func == 3);
        }
        else
        {
            pds->p_headwing.reset();
            pds->epsmacs_imagfreq.clear();
            pds->omegas_imagfreq.clear();
        }

        // ---- recompute G0W0 self-energy with updated mf (g0w0.cpp pattern) ----
        h.build_g0w0_sigma(opts);

        // #1/#2 (coremath review): build_g0w0_sigma only does build_spacetime; it does
        // NOT fill sigc_is_ik_f_KS. Enable the KS-matrix storage flag (#2, gated default
        // off at gw.cpp:1798) then run the KS rotation (#1) which populates it.
        pds->p_g0w0->output_sigc_ks_mat_kf = true;
        pds->p_g0w0->output_sigc_mat_kf =
            read_env_bool("QSGW_SIGC_AO_DUMP", false);
        if (use_fixed_basis_qsgw)
            pds->p_g0w0->build_sigc_matrix_KS_kgrid0_blacs(qsgw_state, pds->blacs_h);
        else
            pds->p_g0w0->build_sigc_matrix_KS_kgrid_blacs(pds->blacs_h);

        // Exx KS rotation (parallel to the sigc #1 fix; coremath final review):
        // build_g0w0_sigma builds Exx real-space only — compute_g0w0.cpp:644
        // build_KS_kgrid_blacs is commented out, so p_exx->exx_KS stays empty and
        // construct_H0_GW's Hexx_all (= p_exx->exx_KS) would be empty/out_of_range.
        // Project to the same fixed KS0 basis used by H_KS0/vxc0. Legacy
        // qsgw_band0 used build_KS_kgrid0() starting at iteration 1, so do not
        // special-case iter-1 through the live-basis BLACS path.
        if (use_fixed_basis_qsgw)
            pds->p_exx->build_KS_kgrid0_blacs(qsgw_state, pds->blacs_h);
        else
            pds->p_exx->build_KS_kgrid_blacs(pds->blacs_h);

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
                if (qsgw_dump_iter1 && qsgw_dump_sigc_ks &&
                    iteration == 1 && ispin == 0 && ikpt == 0 &&
                    mpi_comm_global_h.is_root())
                {
                    for (const auto &freq : freq_nodes)
                    {
                        const int ifreq = pds->tfg.get_freq_index(freq);
                        std::ostringstream stem;
                        stem << qsgw_dump_dir << "SigcKS_spin0_kpt0_ifreq_"
                             << std::setw(3) << std::setfill('0') << ifreq;
                        std::ostringstream label;
                        label << "SigcKS spin=0 kpt=0 ifreq=" << ifreq;
                        dump_matz(sigc_spin_k.at(freq), stem.str(), label.str());
                    }
                }
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
        scale_spin_k_map_inplace(Vc_all, qsgw_vc_scale);

        // ---- H3: Hartree (Task C) — recompute from the updated density ----
        // #7 (coremath): Hartree::build has an is_rspace_built_ cache guard
        // (hartree.cpp:217); without reset, iter>1 build() returns early and
        // Hartree_i goes stale -> Hartree_i_delta silently 0. Reset both spaces
        // each step so build() + build_KS_kgrid0 recompute from the live density.
        // Hartree update intentionally disabled for this diagnostic branch.
        // Vc_all is left as the correlation potential only; no Hartree_i_delta
        // is added for iter>1.

        // ---- Hexx (exchange) from the rebuilt p_exx (Exx::exx_KS, exx.h:56) ----
        const SpinKMatrixMap &Hexx_all = pds->p_exx->exx_KS;
        const SpinKMatrixMap Hexx_scaled =
            (qsgw_exx_scale == 1.0) ? SpinKMatrixMap() : scaled_spin_k_map(Hexx_all, qsgw_exx_scale);
        const SpinKMatrixMap &Hexx_for_h0 =
            (qsgw_exx_scale == 1.0) ? Hexx_all : Hexx_scaled;
        if (qsgw_dump_iter1 && iteration == 1 && mpi_comm_global_h.is_root())
        {
            dump_matz(Hexx_for_h0.at(0).at(0), qsgw_dump_dir + "Hexx_iter_spin0_kpt0",
                      "Hexx_iter spin=0 kpt=0");
        }

        // ---- construct H0_GW (Task B; HROUND inside; H4 vxc0 + H8 via n_spinor) ----
        // NOTE: new signature drops the legacy meanfield param (qsgw_driver_design §3).
        SpinKMatrixMap H0_GW_all;
        if (hamiltonian_cut_above_fermi >= 0)
            H0_GW_all = construct_H0_GW_fermi_window(mf, H_KS0, vxc0, Hexx_for_h0, Vc_all,
                                                     n_spins, n_kpoints, n_bands,
                                                     hamiltonian_cut_above_fermi,
                                                     hamiltonian_cut_diag_shift_ev);
        else
            H0_GW_all = construct_H0_GW(H_KS0, vxc0, Hexx_for_h0, Vc_all,
                                        n_spins, n_kpoints, n_bands);
        if (qsgw_uses_band_window)
            apply_active_band_window(H0_GW_all, H_KS0, n_spins, n_kpoints, n_bands,
                                     active_band_min, active_band_max);
        if (qsgw_dump_iter1 && iteration == 1 && mpi_comm_global_h.is_root())
        {
            dump_matz(H0_GW_all.at(0).at(0), qsgw_dump_dir + "H0_GW_all_spin0_kpt0",
                      "H0_GW_all spin=0 kpt=0");
        }

        // ---- QSGW H0 mixing ----
        double max_abs_delta = 0.0;
        const matrix mixed_input = pack_h0_gw_for_mixing(
            H0_GW_all, n_spins, n_kpoints, n_bands);
        if (!previous_mixed_h0_initialized)
        {
            // iter-1: store current H0_GW_all as the mixing anchor, no modification.
            previous_mixed_h0 = mixed_input;
            previous_mixed_h0_initialized = true;
        }
        else
        {
            matrix mixed_output;
            max_abs_delta = max_abs_matrix_delta(mixed_input, previous_mixed_h0);
            const bool use_manual_linear =
                (qsgw_mixer_mode == "linear") ||
                (qsgw_mixer_mode == "pulay" && iteration <= linear_mixing_steps);

            if (use_manual_linear)
            {
                mixed_output = linear_mix_h0(previous_mixed_h0, mixed_input, mixing_beta);
                mixer_initialized = false;
                if (mpi_comm_global_h.is_root() && should_output())
                {
                    cout << "QSGW iter " << iteration << " linear mix beta="
                         << mixing_beta << " max_abs_delta_H=" << max_abs_delta;
                    if (qsgw_mixer_mode == "pulay")
                        cout << " pulay_warmup_until=" << linear_mixing_steps;
                    cout << endl;
                }
            }
            else
            {
                if (!mixer_initialized)
                {
                    mixer.initialize(previous_mixed_h0);
                    mixer_initialized = true;
                    if (mpi_comm_global_h.is_root() && should_output())
                        cout << "QSGW Pulay mixer initialized history="
                             << mixing_history << " beta=" << mixer.get_mixing_beta()
                             << endl;
                }
                mixed_output = mixer.mix(mixed_input);
                if (mpi_comm_global_h.is_root() && should_output())
                    cout << "QSGW iter " << iteration << " pulay mix beta="
                         << mixer.get_mixing_beta()
                         << " max_abs_delta_H=" << max_abs_delta << endl;
            }

            unpack_h0_gw_from_mixing(
                mixed_output, H0_GW_all, n_spins, n_kpoints, n_bands);
            previous_mixed_h0 = mixed_output;
            previous_mixed_h0_initialized = true;
        }

        // ---- H2: diagonalize using the fixed wfc0 anchor (Task B) ----
        diagonalize_and_store_fixed_basis(mf, H0_GW_all, qsgw_state,
                                          n_spins, n_kpoints, n_bands);

        // ---- fermi / occupations (Task A) ----
        efermi = calculate_fermi_energy(mf, temperature, total_electrons);
        update_fermi_energy_and_occupations(mf, temperature, efermi);
        if (mpi_comm_global_h.is_root())
            write_homo_lumo_iteration(mf, efermi, iteration);

        // ---- convergence: focus-window sorted eigenvalue diff (legacy L1872-L1988) ----
        const double h0diff_for_conv =
            (iteration > 1) ? max_abs_delta * HA2EV : std::numeric_limits<double>::infinity();
        converged = (iteration >= qsgw_min_iterations) && (iteration > 1) &&
                    (h0diff_for_conv < eigenvalue_diff_tolerance);
        if (mpi_comm_global_h.is_root() && should_output())
            cout << "QSGW iter " << iteration << " min_iter=" << qsgw_min_iterations
                 << " h0diff_for_conv = " << h0diff_for_conv
                 << " eV" << (converged ? "  CONVERGED" : "")
                 << " (H0 proxy; eigdiff TODO, focus_nbands="
                 << eigdiff_focus_nbands << ")" << endl;

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
        if (converged || iteration == qsgw_max_iterations) break;
    }
    if (!converged && mpi_comm_global_h.is_root() && should_output())
    {
        cout << "QSGW stopped at max_iterations=" << qsgw_max_iterations
             << " without convergence; this is a diagnostic stop, not a converged QSGW result"
             << endl;
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
