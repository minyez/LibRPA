// Public headers (prefixed by librpa)
#include "librpa_enums.h"
#include "librpa_options.h"

#include "../io/fs.h"
#include "../utils/error.h"

#include <string>
#include <cstring>

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
#include <ddla/ddla_connector.h>
#endif

// C APIs
void librpa_init_options(LibrpaOptions *opts)
{
    librpa_set_output_dir(opts, "librpa.d");

    opts->parallel_routing = LIBRPA_ROUTING_AUTO;
    opts->output_level = LIBRPA_VERBOSE_INFO;
    opts->vq_threshold = 0.0e0;
    opts->use_kpara_scf_eigvec = LIBRPA_SWITCH_OFF;

    opts->tfgrids_type = LIBRPA_TFGRID_UNSET;
    opts->nfreq = 6;
    opts->tfgrids_freq_min = 0.005;
    opts->tfgrids_freq_interval = 0.0;
    opts->tfgrids_freq_max = 1000.0;
    opts->tfgrids_time_min = 0.005;
    opts->tfgrids_time_interval = 0.0;

    opts->minimax_emin = -1.0;
    opts->minimax_emax = -1.0;
    opts->minimax_regulation = 0.0;

    opts->use_fullcoul_eps = LIBRPA_SWITCH_ON;
    opts->use_fullcoul_exx = LIBRPA_SWITCH_OFF;
    opts->use_fullcoul_wc = LIBRPA_SWITCH_OFF;

    opts->n_bands_chi0 = -1;
    opts->n_bands_sigc = -1;

    opts->gf_threshold = 0.0e0;
    opts->use_scalapack_ecrpa = LIBRPA_SWITCH_ON;

    opts->use_shrink_abfs = LIBRPA_SWITCH_OFF;
    opts->use_shrink_chi = LIBRPA_SWITCH_OFF;

    opts->n_params_anacon = -1;
    opts->option_qpe_solver = 0;
    opts->qpe_solver_thres = 1.0e-5;
    opts->qpe_solver_n_iter_max = 200;
    opts->qpe_solver_damp_factor = 0.1;
    opts->use_qpe_adaptive_damp = LIBRPA_SWITCH_OFF;
    opts->override_qpe_solver_nan = LIBRPA_SWITCH_OFF;
    opts->use_scalapack_gw_wc = LIBRPA_SWITCH_ON;
    opts->use_cholesky_gw_wc = LIBRPA_SWITCH_OFF;
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
    int deviceCount = 0;
    ddla::DEVICE_CHECK(ddla::deviceGetDeviceCount(&deviceCount));
    if(deviceCount > 0)
        opts->use_gpu_gw_wc = LIBRPA_SWITCH_ON;
    else
        opts->use_gpu_gw_wc = LIBRPA_SWITCH_OFF;
#else
    opts->use_gpu_gw_wc = LIBRPA_SWITCH_OFF;
#endif
#ifdef ENABLE_ELPA
    opts->use_elpa_sqrt_coulomb = LIBRPA_SWITCH_ON;
#else
    opts->use_elpa_sqrt_coulomb = LIBRPA_SWITCH_OFF;
#endif
    opts->replace_w_head = LIBRPA_SWITCH_OFF;
    opts->option_dielect_func = 0;
    opts->use_2d_dielectric = LIBRPA_SWITCH_OFF;
    opts->sqrt_coulomb_threshold = 0.0e0;
    opts->load_sigc_from_file = LIBRPA_SWITCH_OFF;

    opts->libri_chi0_threshold_C = 0.0e0;
    opts->libri_chi0_threshold_G = 0.0e0;
    opts->libri_exx_threshold_C = 0.0e0;
    opts->libri_exx_threshold_D = 0.0e0;
    opts->libri_exx_threshold_V = 0.0e0;
    opts->libri_g0w0_threshold_C = 0.0e0;
    opts->libri_g0w0_threshold_G = 0.0e0;
    opts->libri_g0w0_threshold_Wc = 0.0e0;

    opts->output_gw_sigc_mat = LIBRPA_SWITCH_OFF;
    opts->output_gw_sigc_mat_rt = LIBRPA_SWITCH_OFF;
    opts->output_gw_sigc_mat_rf = LIBRPA_SWITCH_OFF;
    opts->option_output_Wc_Rf_mat = 0;
}

void librpa_set_output_dir(LibrpaOptions *opts, const char *output_dir)
{
    if (output_dir == nullptr)
    {
        throw LIBRPA_RUNTIME_ERROR("output_dir is null");
    }

    std::string output_dir_s = librpa_int::path_as_directory(output_dir);
    if (output_dir_s.size() >= LIBRPA_MAX_STRLEN)
    {
        throw LIBRPA_RUNTIME_ERROR(
            "output_dir is too long; maximum length is "
            + std::to_string(LIBRPA_MAX_STRLEN - 1)
            + " characters including the appended trailing slash");
    }

    std::memcpy(opts->output_dir, output_dir_s.c_str(), output_dir_s.size() + 1);
}
