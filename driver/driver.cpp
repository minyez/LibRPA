#include "driver.h"

#include <ios>
#include <stdexcept>
#include <sstream>
#include <string>
#include "librpa_enums.h"

namespace driver
{

DriverParams::DriverParams():
    task("unset"),
    constants_choice("internal"),
    input_dir("./"),
    use_spinor_wfc(false),
    prefix_ri_coeff("Cs_data"),
    prefix_ri_coeff_shrink("Cs_shrinked_data"),
    prefix_coul_full("coulomb_mat"),
    prefix_coul_cut("coulomb_cut"),
    fn_stru("stru_out"),
    fn_bz_sampling("bz_sampling_out"),
    fn_basis("basis_out"),
    fn_eigocc_scf("band_out"),
    version_coul_reader(0),
    cs_threshold(1e-6),
    output_energy_qp(false),
    i_state_low(0),
    i_state_high(default_i_state_high),
    output_hamgnn(false),
    use_pyatb(false),
    output_gw_spec_func(false),
    sf_omega_start(0.0),
    sf_omega_end(1.0),
    sf_omega_step(0.1),
    sf_gf_omega_shift(0.01),
    sf_sigc_omega_shift(0.01),
    sf_state_start(0),
    sf_state_end(10000)
{
}

std::string DriverParams::format()
{
    std::stringstream ss;

#define normal_pair(name) {#name, name}
    const std::vector<std::pair<std::string, std::string>> str_params
        {
            normal_pair(task),
            normal_pair(constants_choice),
            normal_pair(input_dir),
            normal_pair(prefix_ri_coeff),
            normal_pair(prefix_ri_coeff_shrink),
            normal_pair(prefix_coul_full),
            normal_pair(prefix_coul_cut),
            normal_pair(fn_stru),
            normal_pair(fn_bz_sampling),
            normal_pair(fn_basis),
            normal_pair(fn_eigocc_scf),
        };
    for (const auto &[k, v]: str_params)
        ss << k << " = " << v << std::endl;

    const std::vector<std::pair<std::string, bool>> bool_params
        {
            normal_pair(use_spinor_wfc),
            normal_pair(output_energy_qp),
            normal_pair(output_gw_spec_func),
            normal_pair(output_hamgnn),
            normal_pair(use_pyatb),
        };
    for (const auto &[k, v]: bool_params)
        ss << k << " = " << std::boolalpha << v << std::endl;
#undef normal_pair

    ss << "version_coul_reader = " << version_coul_reader << std::endl;
    ss << "cs_R_threshold = " << cs_threshold << std::endl;
    ss << "i_state_low = " << i_state_low << std::endl;
    ss << "i_state_high = " << i_state_high << std::endl;
    if (output_gw_spec_func)
    {
        ss << "sf_omega_start = " << sf_omega_start << std::endl;
        ss << "sf_omega_end   = " << sf_omega_end << std::endl;
        ss << "sf_omega_step  = " << sf_omega_step << std::endl;
    }
    return ss.str();
}

DriverParams driver_params;

const std::string input_filename = "librpa.in";

std::vector<int> atom_types;
size_t n_atoms;

int n_spins = 0;
int n_kpoints = 0;
int n_kpoints_band = 0;
int n_ibz_kpoints = 0;
int n_states = 0;
int n_basis_wfc = 0;
int n_basis_ao = 0;
int n_spinor = 1;

std::vector<int> iks_eigvec_this;
std::vector<int> iks_band_eigvec_this;

// Used to read Coulomb matrix data.
// Should be consistent with the internal `atpairs_local` of the Dataset object
std::vector<std::pair<size_t, size_t>> local_atpair;

std::vector<librpa_int::Vector3_Order<double>> ibz_kpoints;
std::vector<librpa_int::Vector3_Order<double>> kfrac_band;

librpa::Handler h;

librpa::Options opts;

#define normal_pair(name) {#name, opts.name}
#define bool_pair(name) {#name, get_bool(opts.name)}

std::string format_runtime_options(const librpa::Options &opts) noexcept
{
    using std::endl;
    using std::boolalpha;

    std::stringstream ss;
    const std::vector<std::pair<std::string, double>> double_params
        {
            {"gf_R_threshold", opts.gf_threshold},
            normal_pair(vq_threshold),
            normal_pair(minimax_emin),
            normal_pair(minimax_emax),
            normal_pair(minimax_regulation),
            normal_pair(sqrt_coulomb_threshold),
            normal_pair(libri_chi0_threshold_C),
            normal_pair(libri_chi0_threshold_G),
            normal_pair(libri_exx_threshold_C),
            normal_pair(libri_exx_threshold_D),
            normal_pair(libri_exx_threshold_V),
            normal_pair(libri_g0w0_threshold_C),
            normal_pair(libri_g0w0_threshold_G),
            normal_pair(libri_g0w0_threshold_Wc),
            normal_pair(qpe_solver_thres),
            normal_pair(qpe_solver_damp_factor),
        };

    const std::vector<std::pair<std::string, int>> int_params
        {
            normal_pair(nfreq),
            normal_pair(n_bands_chi0),
            normal_pair(n_bands_sigc),
            normal_pair(n_params_anacon),
            normal_pair(option_dielect_func),
            normal_pair(qpe_solver_n_iter_max),
        };

    const std::vector<std::pair<std::string, std::string>> str_params
        {
            {"output_dir", opts.output_dir},
            {"output_level", get_verbose_string(opts.output_level)},
            {"tfgrids_type", get_tfgrid_string(opts.tfgrids_type)},
            {"parallel_routing", get_routing_string(opts.parallel_routing)},
        };

    const std::vector<std::pair<std::string, bool>> bool_params
        {
            {"debug", opts.output_level >= LIBRPA_VERBOSE_DEBUG},
            bool_pair(replace_w_head),
            bool_pair(use_fullcoul_eps),
            bool_pair(use_fullcoul_exx),
            bool_pair(use_fullcoul_wc),
            bool_pair(use_shrink_abfs),
            bool_pair(use_shrink_chi),
            bool_pair(use_scalapack_ecrpa),
            bool_pair(use_scalapack_gw_wc),
            bool_pair(use_cholesky_gw_wc),
            bool_pair(use_kpara_scf_eigvec),
            bool_pair(output_gw_sigc_mat),
            bool_pair(output_gw_sigc_mat_rf),
            bool_pair(output_gw_sigc_mat_rt),
            bool_pair(use_qpe_adaptive_damp),
            bool_pair(override_qpe_solver_nan),
        };

    for (const auto &[k, v] : str_params) ss << k << " = " << v << endl;

    for (const auto &[k, v] : int_params) ss << k << " = " << v << endl;

    for (const auto &[k, v] : double_params) ss << k << " = " << v << endl;

    for (const auto &[k, v] : bool_params) ss << k << " = " << boolalpha << v << endl;

    return ss.str();
}

#undef normal_pair
#undef bool_pair

LibrpaTimeFreqGrid get_tfgrid_type(const std::string& grid_str)
{
    if (grid_str == "unset")
        return LIBRPA_TFGRID_UNSET;
    if (grid_str == "GL")
        return LIBRPA_TFGRID_GAUSS_LEGENDRE;
    if (grid_str == "GC-I")
        return LIBRPA_TFGRID_GAUSS_CHEBYSHEV_I;
    if (grid_str == "GL-II")
        return LIBRPA_TFGRID_GAUSS_CHEBYSHEV_II;
    if (grid_str == "minimax")
        return LIBRPA_TFGRID_MINIMAX;
    if (grid_str == "evenspaced")
        return LIBRPA_TFGRID_EVEN_SPACED;
    if (grid_str == "evenspaced_tf")
        return LIBRPA_TFGRID_EVEN_SPACED_TF;
    throw std::runtime_error("Unknown time-frequency grid string: " + grid_str);
}

std::string get_tfgrid_string(const LibrpaTimeFreqGrid& grid_type) noexcept
{
    if (grid_type == LIBRPA_TFGRID_GAUSS_LEGENDRE)
        return "GL";
    if (grid_type == LIBRPA_TFGRID_GAUSS_CHEBYSHEV_I)
        return "GC-I";
    if (grid_type == LIBRPA_TFGRID_GAUSS_CHEBYSHEV_II)
        return "GL-II";
    if (grid_type == LIBRPA_TFGRID_MINIMAX)
        return "minimax";
    if (grid_type == LIBRPA_TFGRID_EVEN_SPACED)
        return "evenspaced";
    if (grid_type == LIBRPA_TFGRID_EVEN_SPACED_TF)
        return "evenspaced_tf";
    return "unset";
}

LibrpaParallelRouting get_parallel_routing(const std::string& routing_str_low)
{
    if (routing_str_low == "auto") return LIBRPA_ROUTING_AUTO;
    if (routing_str_low == "rtau") return LIBRPA_ROUTING_RTAU;
    if (routing_str_low == "atompair") return LIBRPA_ROUTING_ATOMPAIR;
    if (routing_str_low == "libri") return LIBRPA_ROUTING_LIBRI;
    throw std::runtime_error("Unknown parallel routing string: " + routing_str_low);
}

std::string get_routing_string(LibrpaParallelRouting routing)
{
    if (routing == LIBRPA_ROUTING_AUTO) return "auto";
    if (routing == LIBRPA_ROUTING_ATOMPAIR) return "atompair";
    if (routing == LIBRPA_ROUTING_RTAU) return "rtau";
    if (routing == LIBRPA_ROUTING_LIBRI) return "libri";
    return "unset";
}

std::string get_verbose_string(LibrpaVerbose verbose)
{
    if (verbose == LIBRPA_VERBOSE_DEBUG) return "debug";
    if (verbose == LIBRPA_VERBOSE_WARN) return "warn";
    if (verbose == LIBRPA_VERBOSE_INFO) return "info";
    if (verbose == LIBRPA_VERBOSE_CRITICAL) return "critical";
    return "silent";
}

LibrpaVerbose get_verbose(const std::string& verbose_str_low)
{
    if (verbose_str_low == "debug") return LIBRPA_VERBOSE_DEBUG;
    if (verbose_str_low == "warn") return LIBRPA_VERBOSE_WARN;
    if (verbose_str_low == "warning") return LIBRPA_VERBOSE_WARN;
    if (verbose_str_low == "info") return LIBRPA_VERBOSE_INFO;
    if (verbose_str_low == "critical") return LIBRPA_VERBOSE_CRITICAL;
    if (verbose_str_low == "silent") return LIBRPA_VERBOSE_SILENT;
    throw std::runtime_error("Unknown verbose level string: " + verbose_str_low);
}

}
