#pragma once
/**
 * @file librpa_compute.h
 * @brief Computing data APIs for LibRPA.
 */

#include "librpa_handler.h"
#include "librpa_options.h"

// C APIs
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Construct and return frequency grids for numerical integration.
 *
 * Generates the frequency points and quadrature weights based on the grid type
 * specified in the options (e.g., Gauss-Legendre, Minimax, etc.).
 *
 * @param[in]  h        Handler.
 * @param[in]  p_opts   Runtime options.
 * @param[out] omegas  Array of frequency points (size: opts->nfreq).
 * @param[out] weights Array of quadrature weights (size: opts->nfreq).
 */
void librpa_get_imaginary_frequency_grids(LibrpaHandler *h, const LibrpaOptions *p_opts,
                                          double *omegas, double *weights);

/**
 * @brief Compute RPA correlation energy.
 *
 * Calculates the RPA correlation energy using the input data (wavefunctions,
 * Coulomb matrices, etc.) that has been set via the input parsing functions.
 *
 * @param[in]  h                           Handler.
 * @param[in]  p_opts                      Runtime options.
 * @param[in]  n_ibz_kpoints                Number of irreducible k-points.
 * @param[out] rpa_corr_ibzk_contrib_re   Real part of correlation energy per IBZ k-point.
 * @param[out] rpa_corr_ibzk_contrib_im   Imaginary part of correlation energy per IBZ k-point.
 * @return Total RPA correlation energy (real part).
 */
double librpa_get_rpa_correlation_energy(LibrpaHandler *h, const LibrpaOptions *p_opts,
                                         int n_ibz_kpoints, double *rpa_corr_ibzk_contrib_re,
                                         double *rpa_corr_ibzk_contrib_im);

//! Build exact-exchange matrix
/**
 * @param[in]  h                Pointer to LibRPA handler.
 * @param[in]  p_opts           Pointer to runtime options.
 */
void librpa_build_exx(LibrpaHandler *h, const LibrpaOptions *p_opts);

//! Obtain exact-exchange potential for selected states.
/**
 * @param[in]  h                Pointer to LibRPA handler.
 * @param[in]  p_opts           Pointer to runtime options.
 * @param[in]  n_spins          Number of spin channels.
 * @param[in]  n_kpts_this      Number of k-points to compute on this process.
 * @param[in]  iks_this         (Global) index of k-points that this process compute.
 *                              Each process can have different indices.
 *                              Must be a subset of k-points at which the eigenvetors are parsed.
 * @param[in]  i_state_low      Index of the first state to compute the potential (inclusive)
 * @param[in]  i_state_high     Index of the last state to compute the potential (exclusive)
 * @param[out] vexx             Exact-exchange potential for selected states.
 *                              It should be at least as long as n_spins * n_kpts_local * (i_state_high - i_state_low).
 */
void librpa_get_exx_pot_kgrid(LibrpaHandler *h, const LibrpaOptions *p_opts,
                              const int n_spins, const int n_kpts_this, const int *iks_this,
                              int i_state_low, int i_state_high, double *vexx);

//! Obtain exact-exchange potential for selected states at band k-points.
/**
 * @param[in]  h                Pointer to LibRPA handler.
 * @param[in]  p_opts           Pointer to runtime options.
 * @param[in]  n_spins          Number of spin channels.
 * @param[in]  n_kpts_band_this Number of k-points to compute on this process.
 * @param[in]  iks_band_this    (Global) index of k-points that this process compute.
 *                              Each process can have different indices.
 *                              Must be a subset of band k-points at which the eigenvetors are parsed.
 * @param[in]  i_state_low      Index of the first state to compute the potential (inclusive)
 * @param[in]  i_state_high     Index of the last state to compute the potential (exclusive)
 * @param[out] vexx_band        Exact-exchange potential for selected states at band k-points.
 *                              It should be at least as long as n_spins * n_kpts_band_this * (i_state_high - i_state_low).
 */
void librpa_get_exx_pot_band_k(LibrpaHandler *h, const LibrpaOptions *p_opts,
                               const int n_spins, const int n_kpts_band_this, const int *iks_band_this,
                               int i_state_low, int i_state_high, double *vexx_band);

//! Build self-energy matrix of G0W0, including the correlation and exchange contributions.
/**
 * @param[in]  h                Pointer to LibRPA handler.
 * @param[in]  p_opts           Pointer to runtime options.
 */
void librpa_build_g0w0_sigma(LibrpaHandler *h, const LibrpaOptions *p_opts);

//! Obtain correlation self-energies for selected states.
/**
 * @param[in]  h                Pointer to LibRPA handler.
 * @param[in]  p_opts           Pointer to runtime options.
 * @param[in]  n_spins          Number of spin channels.
 * @param[in]  n_kpts_this      Number of k-points to compute on this process.
 * @param[in]  iks_this         (Global) index of k-points that this process compute.
 *                              Each process can have different indices.
 *                              Must be a subset of k-points at which the eigenvetors are parsed.
 * @param[in]  i_state_low      Index of the first state to compute the potential (inclusive)
 * @param[in]  i_state_high     Index of the last state to compute the potential (exclusive)
 * @param[in]  vxc              exchange-correlation potential of the selected states.
 * @param[in]  vexx             Exact-exchange potential for the selected states.
 *                              It should be at least as long as n_spins * n_kpoints_local * (i_state_high - i_state_low).
 *                              It can be obtained using librpa_get_exx_pot_kgrid.
 * @param[out] sigc_re          Real-part of the correlation self-energy for the selected states.
 *                              For option_qpe_solver=2, this is the perturbative effective real
 *                              contribution that reconstructs the perturbative QP energy.
 *                              It should be at least as long as n_spins * n_kpoints_local * (i_state_high - i_state_low).
 * @param[out] sigc_im          Same as sigc_re, but for the imaginary part.
 */
void librpa_get_g0w0_sigc_kgrid(LibrpaHandler *h, const LibrpaOptions *p_opts,
                                const int n_spins, const int n_kpts_this,
                                const int *iks_this, int i_state_low, int i_state_high,
                                const double *vxc, const double *vexx, double *sigc_re, double *sigc_im);

//! Obtain correlation self-energies and QP energies for selected states.
/**
 * Same inputs and SigC outputs as librpa_get_g0w0_sigc_kgrid.
 *
 * @param[out] eqp              Quasi-particle energy solved by the selected QPE solver.
 *                              It should be at least as long as sigc_re/sigc_im.
 */
void librpa_get_g0w0_qpe_kgrid(LibrpaHandler *h, const LibrpaOptions *p_opts,
                               const int n_spins, const int n_kpts_this,
                               const int *iks_this, int i_state_low, int i_state_high,
                               const double *vxc, const double *vexx, double *sigc_re,
                               double *sigc_im, double *eqp);

//! Obtain spectral functions for selected k-grid states.
/**
 * Computes spectral functions under the diagonal approximation of the Green's
 * function:
 * \f$A_{n\mathbf{k}}(\omega) = -\pi^{-1}\mathrm{Im}\,G_{n\mathbf{k}}(\omega)\f$.
 *
 * @param[in]  h                       Pointer to LibRPA handler.
 * @param[in]  p_opts                  Pointer to runtime options.
 * @param[in]  n_spins                 Number of spin channels.
 * @param[in]  n_kpts_this             Number of k-points to compute on this process.
 * @param[in]  iks_this                Global k-point indices computed on this process.
 * @param[in]  i_state_low             First state index (inclusive).
 * @param[in]  i_state_high            Last state index (exclusive).
 * @param[in]  n_omegas                Number of real-frequency points.
 * @param[in]  omegas                  Real-frequency points, in Hartree.
 * @param[in]  vxc                     XC potential for selected states.
 * @param[in]  vexx                    Exact-exchange potential for selected states.
 * @param[out] spectral_function       Spectral functions. This output array should be at least as long as
 *                                     n_spins * n_kpts_this * (i_state_high - i_state_low) * n_omegas.
 *                                     The order is [spin][k-point][state][omega].
 * @param[out] sigc                    Optional continued correlation self-energy. Pass nullptr if this output is
 *                                     not needed. If present, this is a packed complex<double> buffer with the
 *                                     same [spin][k-point][state][omega] order as spectral_function.
 */
void librpa_get_g0w0_spectral_function_kgrid(
    LibrpaHandler *h, const LibrpaOptions *p_opts, const int n_spins,
    const int n_kpts_this, const int *iks_this, int i_state_low, int i_state_high,
    const int n_omegas, const double *omegas, const double *vxc, const double *vexx,
    double *spectral_function, double *sigc);

//! Obtain correlation self-energies for selected states at band k-points.
/**
 * @param[in]  h                Pointer to LibRPA handler.
 * @param[in]  p_opts           Pointer to runtime options.
 * @param[in]  n_spins          Number of spin channels.
 * @param[in]  n_kpts_band_this Number of k-points to compute on this process.
 * @param[in]  iks_band_this    (Global) index of k-points that this process compute.
 *                              Each process can have different indices.
 *                              Must be a subset of k-points at which the eigenvetors are parsed.
 * @param[in]  i_state_low      Index of the first state to compute the potential (inclusive)
 * @param[in]  i_state_high     Index of the last state to compute the potential (exclusive)
 * @param[in]  vxc_band         exchange-correlation potential of the selected states at band k-points.
 * @param[in]  vexx_band        Exact-exchange potential for the selected states at band k-points.
 *                              It should be at least as long as n_spins * n_kpts_band_this * (i_state_high - i_state_low).
 *                              It can be obtained using librpa_get_exx_pot_kgrid.
 * @param[out] sigc_band_re     Real-part of the correlation self-energy for the selected states.
 *                              For option_qpe_solver=2, this is the perturbative effective real
 *                              contribution that reconstructs the perturbative QP energy.
 *                              It should be at least as long as n_spins * n_kpts_band_this * (i_state_high - i_state_low).
 * @param[out] sigc_band_im     Same as sigc_band_re, but for the imaginary part.
 */
void librpa_get_g0w0_sigc_band_k(LibrpaHandler *h, const LibrpaOptions *p_opts,
                                 const int n_spins, const int n_kpts_band_this,
                                 const int *iks_band_this, int i_state_low, int i_state_high,
                                 const double *vxc_band, const double *vexx_band, double *sigc_band_re, double *sigc_band_im);

//! Obtain correlation self-energies and QP energies for selected band-k states.
/**
 * Same inputs and SigC outputs as librpa_get_g0w0_sigc_band_k.
 *
 * @param[out] eqp_band         Quasi-particle energy solved by the selected QPE solver.
 *                              It should be at least as long as sigc_band_re/sigc_band_im.
 */
void librpa_get_g0w0_qpe_band_k(LibrpaHandler *h, const LibrpaOptions *p_opts,
                                const int n_spins, const int n_kpts_band_this,
                                const int *iks_band_this, int i_state_low, int i_state_high,
                                const double *vxc_band, const double *vexx_band,
                                double *sigc_band_re, double *sigc_band_im,
                                double *eqp_band);

//! Obtain spectral functions for selected band-k states.
/**
 * Same convention as librpa_get_g0w0_spectral_function_kgrid, but evaluates
 * states at band k-points.
 *
 * @param[in]  h                       Pointer to LibRPA handler.
 * @param[in]  p_opts                  Pointer to runtime options.
 * @param[in]  n_spins                 Number of spin channels.
 * @param[in]  n_kpts_band_this        Number of band k-points to compute on this process.
 * @param[in]  iks_band_this           Global band-k indices computed on this process.
 * @param[in]  i_state_low             First state index (inclusive).
 * @param[in]  i_state_high            Last state index (exclusive).
 * @param[in]  n_omegas                Number of real-frequency points.
 * @param[in]  omegas                  Real-frequency points, in Hartree.
 * @param[in]  vxc_band                XC potential for selected band states.
 * @param[in]  vexx_band               Exact-exchange potential for selected band states.
 * @param[out] spectral_function_band  Spectral functions. This output array should be at least as long as
 *                                     n_spins * n_kpts_band_this * (i_state_high - i_state_low) * n_omegas.
 *                                     The order is [spin][band k-point][state][omega].
 * @param[out] sigc_band               Optional continued correlation self-energy. Pass nullptr if this output is
 *                                     not needed. If present, this is a packed complex<double> buffer with the
 *                                     same [spin][band k-point][state][omega] order as spectral_function_band.
 */
void librpa_get_g0w0_spectral_function_band_k(
    LibrpaHandler *h, const LibrpaOptions *p_opts, const int n_spins,
    const int n_kpts_band_this, const int *iks_band_this, int i_state_low,
    int i_state_high, const int n_omegas, const double *omegas,
    const double *vxc_band, const double *vexx_band, double *spectral_function_band,
    double *sigc_band);

#ifdef __cplusplus
}
#endif
