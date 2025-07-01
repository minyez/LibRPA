#pragma once
#include <functional>
#include <vector>

#include "atomic_basis.h"
#include "complexmatrix.h"
#include "constants.h"
#include "envs_blacs.h"
#include "envs_io.h"
#include "envs_mpi.h"
#include "libri_stub.h"
#include "matrix_m_parallel_utils.h"
#include "meanfield.h"
#include "parallel_mpi.h"
#include "params.h"
#include "pbc.h"
#include "ri.h"
#include "utils_blacs.h"
#include "utils_io.h"
#include "vec.h"

//! double-dispersion Havriliak-Negami model
struct DoubleHavriliakNegami
{
    static const int d_npar;
    static const std::function<double(double, const std::vector<double> &)> func_imfreq;
    static const std::function<void(std::vector<double> &, double, const std::vector<double> &)>
        grad_imfreq;
};

// All calculation in unit: Bohr and Ha.
class diele_func
{
   private:
    // ( omega, alpha * beta  )
    std::vector<matrix_m<std::complex<double>>> head;
    // ( omega, mu:n_abfs * alpha ) for comparison with FHI-aims
    std::vector<matrix_m<std::complex<double>>> wing_mu;
    // ( omega, lambda:n_lambda * alpha)
    std::vector<matrix_m<std::complex<double>>> wing;
    // std::vector<std::vector<std::vector<std::complex<double>>>> wing;

    // ( i:n_lambda, j:n_lambda )
    matrix_m<std::complex<double>> body_inv;
    // ( i:3, j:3 )
    matrix_m<std::complex<double>> Lind;
    // ( i:n_lambda, j:n_lambda )
    matrix_m<std::complex<double>> chi0;
    // ( lambda: n_nonsingular-1, mu: n_abfs)
    std::vector<std::vector<std::complex<double>>> Coul_vector;
    // ( lambda: n_nonsingular-1 )
    std::vector<std::complex<double>> Coul_value;
    // ( mu: n_abfs, m: n_bands, n: n_bands, k )
    std::vector<std::vector<std::vector<std::map<Vector3_Order<double>, std::complex<double>>>>>
        Ctri_mn;
    // ( mu: n_abfs@I, i: i atom basis, j: j atom basis, k, I atom, J atom, R cell  )
    // Ctri_ij.data_libri[I][{J, k_array}](mu, i, j)
    Cs_LRI_clx Ctri_ij;

    MeanField &meanfield_df;
    std::vector<double> omega;
    std::vector<Vector3_Order<double>> &kfrac_band;
    int n_basis, n_states, n_spin, n_abf, nk;
    size_t n_nonsingular;
    // lebedev-quadrature, qw has absorbed 4Pi.
    std::vector<double> qx_leb, qy_leb, qz_leb, qw_leb;
    // gamma reciprocal lattice vector, (27-1)*3
    std::vector<Vector3_Order<double>> g_enclosing_gamma;
    std::vector<double> q_gamma;
    double vol_gamma;

   public:
    diele_func(MeanField &mf, std::vector<Vector3_Order<double>> &kfrac,
               std::vector<double> frequencies_target, int nbasis, int nstates, int nspin)
        : meanfield_df(mf),
          kfrac_band(kfrac),
          omega(frequencies_target),
          n_basis(nbasis),
          n_states(nstates),
          n_spin(nspin)
    {
        init();
    };
    diele_func() : meanfield_df(meanfield), kfrac_band(kfrac_list) {};
    ~diele_func() {};
    void init();
    void init_Cs();
    void set(MeanField &mf, std::vector<Vector3_Order<double>> &kfrac,
             std::vector<double> frequencies_target, int nbasis, int nstates, int nspin);

    void cal_head();
    double cal_factor(string name);
    void test_head();
    std::vector<double> get_head_vec();

    void cal_wing();
    // compute wing in ABF representation
    std::complex<double> compute_wing(int alpha, int iomega, int mu);
    // transform wing from ABF to Coulomb representation
    void wing_mu_to_lambda(matrix_m<std::complex<double>> &sqrtveig_blacs);
    // tranform Cs_ij(R) to Cs_ij(k)
    void FT_R2k();
    std::complex<double> compute_Cijk(Cs_LRI &Cs_in, int mu, int I, int i, int J, int j, int ik);
    void Cs_ij2mn();
    std::complex<double> compute_Cs_ij2mn(int mu, int m, int n, int ik);
    // diagonalize real Vq_cut(q=0)
    void get_Xv_real();
    // diagonalize complex Vq_cut(q=0)
    void get_Xv_cpl();
    void test_wing();
    // set wing=0 for debug
    void set_0_wing();

    void get_body_inv(matrix_m<std::complex<double>> &chi0_block);
    void construct_L(const int ifreq);
    // Lebedev-Laikov quadrature
    void get_Leb_points();
    void get_g_enclosing_gamma();
    void calculate_q_gamma();
    void cal_eps(const int ifreq);
    std::complex<double> compute_chi0_inv_00(const int ifreq);
    std::complex<double> compute_chi0_inv_ij(const int ifreq, int i, int j);
    void rewrite_eps(matrix_m<std::complex<double>> &chi0_block, const int ifreq);
};

extern diele_func df_headwing;
