#include "qsgw_io.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "../src/utils/constants.h" // HA2EV

namespace librpa_int {

namespace {

// ---- self-contained path/dir helpers (ported from legacy driver_utils.cpp) ----

std::string ensure_trailing_slash(const std::string &dir_path)
{
    if (dir_path.empty() || dir_path.back() == '/')
    {
        return dir_path;
    }
    return dir_path + "/";
}

std::string shell_single_quote(const std::string &value)
{
    std::string quoted;
    quoted.reserve(value.size() + 8);
    quoted.push_back('\'');
    for (const char ch : value)
    {
        if (ch == '\'')
        {
            quoted += "'\"'\"'";
        }
        else
        {
            quoted.push_back(ch);
        }
    }
    quoted.push_back('\'');
    return quoted;
}

void ensure_dir_exists_shell(const std::string &dir_path)
{
    const auto normalized_dir = ensure_trailing_slash(dir_path);
    const auto command = "mkdir -p " + shell_single_quote(normalized_dir);
    const int status = std::system(command.c_str());
    if (status != 0)
    {
        throw std::runtime_error("Failed to create directory: " + normalized_dir);
    }
}

std::string qsgw_hamiltonian_export_root(const std::string &bundle_dir)
{
    return ensure_trailing_slash(bundle_dir) + "qsgw_hamiltonian/";
}

std::string qsgw_hamiltonian_iteration_dir(const std::string &bundle_dir, const int iteration)
{
    std::ostringstream oss;
    oss << qsgw_hamiltonian_export_root(bundle_dir) << "iter_" << std::setw(5)
        << std::setfill('0') << std::max(iteration, 0) << "/";
    return oss.str();
}

std::string qsgw_hamiltonian_file_prefix(const std::string &iteration_dir,
                                         const std::string &state_label, const int ispin,
                                         const int ikpt)
{
    std::ostringstream oss;
    oss << ensure_trailing_slash(iteration_dir) << "Hqsgw_" << state_label << "_spin_"
        << std::setw(2) << std::setfill('0') << (ispin + 1) << "_k_" << std::setw(6)
        << std::setfill('0') << (ikpt + 1);
    return oss.str();
}

} // namespace

// Faithful port of legacy driver/driver_utils.cpp:91-158 (7a7ff17f).
// New-arch adaptations:
//  - manifest "n_soc" -> "n_spinor" (MeanField::get_n_soc removed, hard-fact #4);
//  - print_matrix_elsi_csc -> write_matrix_elsi_csc (renamed in new matrix_m.h);
//  - print_matrix_mm_file 3rd arg is now header_comment (string), threshold moved
//    to 4th, so an empty comment is passed explicitly.
std::string export_qsgw_hamiltonian_bundle(
    const std::string &bundle_dir, const MeanField &mf,
    const std::vector<Vector3_Order<double>> &kfrac,
    const std::map<int, std::map<int, Matz>> &hamiltonians, const int iteration,
    const std::string &state_label)
{
    const auto iteration_dir = qsgw_hamiltonian_iteration_dir(bundle_dir, iteration);
    ensure_dir_exists_shell(qsgw_hamiltonian_export_root(bundle_dir));
    ensure_dir_exists_shell(iteration_dir);

    for (const auto &spin_entry : hamiltonians)
    {
        for (const auto &k_entry : spin_entry.second)
        {
            const auto prefix = qsgw_hamiltonian_file_prefix(iteration_dir, state_label,
                                                             spin_entry.first, k_entry.first);
            write_matrix_elsi_csc(k_entry.second, prefix + ".csc", 1e-15);
            print_matrix_mm_file(k_entry.second, prefix + ".mtx", "", 1e-15);
        }
    }

    {
        std::ofstream latest(qsgw_hamiltonian_export_root(bundle_dir) + state_label
                             + "_latest_iteration.txt");
        if (!latest.good())
        {
            throw std::runtime_error("Failed to write latest iteration marker under "
                                     + qsgw_hamiltonian_export_root(bundle_dir));
        }
        latest << iteration << std::endl;
    }

    {
        std::ofstream manifest(iteration_dir + "manifest.json");
        if (!manifest.good())
        {
            throw std::runtime_error("Failed to write manifest under " + iteration_dir);
        }

        manifest << std::fixed << std::setprecision(16);
        manifest << "{\n";
        manifest << "  \"iteration\": " << iteration << ",\n";
        manifest << "  \"state_label\": \"" << state_label << "\",\n";
        manifest << "  \"bundle_dir\": \"" << ensure_trailing_slash(bundle_dir) << "\",\n";
        manifest << "  \"n_spins\": " << mf.get_n_spins() << ",\n";
        manifest << "  \"n_kpoints\": " << mf.get_n_kpoints() << ",\n";
        manifest << "  \"n_bands\": " << mf.get_n_bands() << ",\n";
        manifest << "  \"n_aos\": " << mf.get_n_aos() << ",\n";
        manifest << "  \"n_spinor\": " << mf.get_n_spinor() << ",\n";
        manifest << "  \"efermi_ha\": " << mf.get_efermi() << ",\n";
        manifest << "  \"efermi_ev\": " << mf.get_efermi() * HA2EV << ",\n";
        manifest << "  \"basis\": \"KS_orthonormal_band_space\",\n";
        manifest << "  \"hamiltonian_formats\": [\"elsi_csc\", \"matrix_market\"],\n";
        manifest << "  \"velocity_source\": \"libRPA_meanfield_seed\",\n";
        manifest << "  \"kpoints\": [\n";
        for (size_t ik = 0; ik < kfrac.size(); ++ik)
        {
            manifest << "    {\"index\": " << (ik + 1) << ", \"kx\": " << kfrac[ik].x
                     << ", \"ky\": " << kfrac[ik].y << ", \"kz\": " << kfrac[ik].z << "}"
                     << (ik + 1 == kfrac.size() ? "\n" : ",\n");
        }
        manifest << "  ]\n";
        manifest << "}\n";
    }

    return iteration_dir;
}

// ---- Stubs: blocked on QSGW driver (Task D) / pyatb-export subsystem. ----------
// These wrap export_qsgw_hamiltonian_bundle with pyatb-state writing +
// optional rebuild-command orchestration. They need write_pyatb_bundle,
// future driver-level QSGW pyatb export/rebuild controls,
// get_iterative_pyatb_headwing_bundle_dir and maybe_run_pyatb_rebuild_command,
// none of which exist on the new arch yet. Implemented when that lands.

void export_pyatb_state_bundle(
    const MeanField & /*mf*/, const std::vector<Vector3_Order<double>> & /*kfrac*/,
    const std::string & /*bundle_dir*/,
    const std::map<int, std::map<int, Matz>> * /*hamiltonians*/, int /*iteration*/,
    const std::string & /*state_label*/, bool /*run_rebuild_command*/)
{
    throw std::runtime_error(
        "export_pyatb_state_bundle not yet ported to the new arch: blocked on "
        "write_pyatb_bundle + driver-level QSGW pyatb export/rebuild controls "
        "+ pyatb headwing bundle-dir helpers "
        "(QSGW driver Task D / pyatb-export subsystem).");
}

void refresh_pyatb_headwing_bundle(
    const MeanField &mf, const std::vector<Vector3_Order<double>> &kfrac,
    const std::string &bundle_dir,
    const std::map<int, std::map<int, Matz>> *hamiltonians, int iteration,
    const std::string &state_label)
{
    // Legacy body was: export_pyatb_state_bundle(..., run_rebuild_command=true).
    export_pyatb_state_bundle(mf, kfrac, bundle_dir, hamiltonians, iteration, state_label,
                              true);
}

} // namespace librpa_int
