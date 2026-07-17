#include <array>
#include <cassert>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <valarray>

#include "../core/chi0.h"
#include "../core/dielecmodel.h"
#include "../core/epsilon.h"
#include "../core/qpoint_view.h"
#include "../core/meanfield_mpi.h"
#include "../io/global_io.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../mpi/base_blacs.h"
#include "../mpi/base_mpi.h"
#include "../mpi/kpoint_blacs_parallel_context.h"
#include "../utils/constants.h"

using librpa_int::ArrayDesc;
using librpa_int::AtomicBasis;
using librpa_int::atpair_k_cplx_mat_t;
using librpa_int::BlacsCtxtHandler;
using librpa_int::C_ONE;
using librpa_int::ComplexMatrix;
using librpa_int::diele_func;
using librpa_int::init_local_mat;
using librpa_int::KPointBlacsParallelContext;
using librpa_int::KPointBlacsProcessShape;
using librpa_int::MAJOR;
using librpa_int::Matrix3;
using librpa_int::Matz;
using librpa_int::matrix_m;
using librpa_int::MeanField;
using librpa_int::PeriodicBoundaryData;
using librpa_int::SpeciesBasisLayout;
using librpa_int::SymmetryContext;
using librpa_int::SymmetryKAtomRotation;
using librpa_int::SymmetryKStarMember;
using librpa_int::SymmetryOperation;
using librpa_int::SymmetryQPointRestoreMode;
using librpa_int::TFGrids;
using librpa_int::Vector3_Order;
using librpa_int::atom_mapping;
using librpa_int::atom_t;
using librpa_int::build_symmetry_qpoint_view;

namespace
{

void test_rspace_symmetry_requires_complete_band_space()
{
    MeanField truncated(1, 1, 2, 3);
    assert(!librpa_int::rspace_symmetry_has_complete_band_space(truncated, -1));
    assert(!librpa_int::rspace_symmetry_has_complete_band_space(truncated, 3));

    MeanField complete(1, 1, 3, 3);
    assert(librpa_int::rspace_symmetry_has_complete_band_space(complete, -1));
    assert(!librpa_int::rspace_symmetry_has_complete_band_space(complete, 2));
}

void assert_complex_close(const std::complex<double> &actual, const std::complex<double> &expected,
                          const double tolerance)
{
    if (std::abs(actual - expected) >= tolerance)
    {
        std::cerr << "actual=" << actual << " expected=" << expected
                  << " diff=" << std::abs(actual - expected) << std::endl;
        assert(false);
    }
}

void require_double_close(const double actual, const double expected, const double tolerance)
{
    if (std::abs(actual - expected) >= tolerance)
    {
        std::cerr << "actual=" << actual << " expected=" << expected
                  << " diff=" << std::abs(actual - expected) << std::endl;
        std::abort();
    }
}

RI::Tensor<double> make_scalar_cs_tensor(const double value)
{
    auto data = std::make_shared<std::valarray<double>>(1);
    (*data)[0] = value;
    return RI::Tensor<double>({1UL, 1UL, 1UL}, data);
}

void test_kpoint_coordinate_mapping_selects_active_klist_from_full_source()
{
    const std::vector<Vector3_Order<double>> pyatb_full_kpoints{
        {0.0, 0.0, 0.0}, {0.125, 0.0, 0.0}, {0.25, 0.0, 0.0}, {0.375, 0.0, 0.0},
        {0.0, 0.125, 0.0}, {0.125, 0.125, 0.0}, {0.875, 0.0, 0.0}};
    const std::vector<Vector3_Order<double>> active_kpoints{
        {0.0, 0.0, 0.0}, {0.25, 0.0, 0.0}, {0.125, 0.125, 0.0}};

    const auto mapping = librpa_int::map_kpoints_by_coordinates(active_kpoints, pyatb_full_kpoints);

    assert((mapping == std::vector<int>{0, 2, 5}));

    const std::vector<Vector3_Order<double>> wrapped_active_kpoints{{-0.125, 0.0, 0.0}};
    const auto wrapped_mapping =
        librpa_int::map_kpoints_by_coordinates(wrapped_active_kpoints, pyatb_full_kpoints);
    assert((wrapped_mapping == std::vector<int>{6}));
}

void test_kstar_velocity_mapping_preserves_member_order_and_periodic_gauge()
{
    SymmetryContext ctx;
    librpa_int::SymmetryKStar gamma;
    gamma.k_ibz = {0.0, 0.0, 0.0};
    librpa_int::SymmetryKStarMember gamma_first;
    gamma_first.k_bz = {0.0, 0.0, 0.0};
    librpa_int::SymmetryKStarMember gamma_second;
    gamma_second.k_bz = {0.5, 0.0, 0.0};
    gamma.members = {gamma_first, gamma_second};
    librpa_int::SymmetryKStar quarter;
    quarter.k_ibz = {0.25, 0.0, 0.0};
    librpa_int::SymmetryKStarMember quarter_first;
    quarter_first.k_bz = {0.25, 0.0, 0.0};
    librpa_int::SymmetryKStarMember quarter_second;
    quarter_second.k_bz = {-0.25, 0.0, 0.0};
    quarter.members = {quarter_first, quarter_second};
    ctx.kstars = {gamma, quarter};

    const std::vector<Vector3_Order<double>> ibz_kpoints{{0.25, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    const std::vector<Vector3_Order<double>> full_bz_kpoints{
        {0.5, 0.0, 0.0}, {0.75, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.25, 0.0, 0.0}};

    const auto mapping = librpa_int::map_symmetry_kstar_members_to_source_kpoints(
        ctx, ibz_kpoints, full_bz_kpoints);

    assert((mapping == std::vector<std::vector<int>>{{3, 1}, {2, 0}}));
}

void test_replace_rpa_response_headwing_replaces_only_singular_channels(
    const BlacsCtxtHandler &blacs_h)
{
    ArrayDesc desc(blacs_h);
    desc.init_square_blk(4, 4, 0, 0);

    auto response = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    for (int i = 0; i != 4; ++i)
    {
        const int ilo = desc.indx_g2l_r(i);
        if (ilo < 0) continue;
        for (int j = 0; j != 4; ++j)
        {
            const int jlo = desc.indx_g2l_c(j);
            if (jlo < 0) continue;
            response(ilo, jlo) =
                std::complex<double>(0.01 * (i + 1) + 0.02 * (j + 1), 0.001 * (i - j));
        }
    }

    const matrix_m<std::complex<double>> head(
        std::vector<std::vector<std::complex<double>>>{
            {std::complex<double>{2.0, 0.0}, std::complex<double>{0.2, 0.1},
             std::complex<double>{0.3, -0.1}},
            {std::complex<double>{0.2, -0.1}, std::complex<double>{2.2, 0.0},
             std::complex<double>{0.4, 0.2}},
            {std::complex<double>{0.3, 0.1}, std::complex<double>{0.4, -0.2},
             std::complex<double>{2.4, 0.0}}},
        MAJOR::COL);
    const matrix_m<std::complex<double>> wing(
        std::vector<std::vector<std::complex<double>>>{
            {std::complex<double>{0.11, 0.01}, std::complex<double>{0.12, 0.02},
             std::complex<double>{0.13, 0.03}},
            {std::complex<double>{0.21, 0.04}, std::complex<double>{0.22, 0.05},
             std::complex<double>{0.23, 0.06}},
            {std::complex<double>{0.31, 0.07}, std::complex<double>{0.32, 0.08},
             std::complex<double>{0.33, 0.09}}},
        MAJOR::COL);

    librpa_int::replace_rpa_response_headwing(response, head, wing, desc);

    const int head_row = desc.indx_g2l_r(0);
    const int head_col = desc.indx_g2l_c(0);
    if (head_row >= 0 && head_col >= 0)
    {
        assert_complex_close(response(head_row, head_col), std::complex<double>{2.2, 0.0}, 1e-12);
    }

    for (int lambda = 1; lambda != 4; ++lambda)
    {
        std::complex<double> expected_wing = 0.0;
        for (int alpha = 0; alpha != 3; ++alpha)
        {
            expected_wing += wing(lambda - 1, alpha);
        }
        expected_wing /= 3.0;

        const int row_body = desc.indx_g2l_r(lambda);
        const int col_head = desc.indx_g2l_c(0);
        if (row_body >= 0 && col_head >= 0)
        {
            assert_complex_close(response(row_body, col_head), expected_wing, 1e-12);
        }

        const int row_head = desc.indx_g2l_r(0);
        const int col_body = desc.indx_g2l_c(lambda);
        if (row_head >= 0 && col_body >= 0)
        {
            assert_complex_close(response(row_head, col_body), std::conj(expected_wing), 1e-12);
        }
    }

    const int body_row = desc.indx_g2l_r(2);
    const int body_col = desc.indx_g2l_c(3);
    if (body_row >= 0 && body_col >= 0)
    {
        assert_complex_close(response(body_row, body_col), std::complex<double>{0.11, -0.001},
                             1e-12);
    }
}

void test_replace_rpa_response_head_only_keeps_numeric_wings(const BlacsCtxtHandler &blacs_h)
{
    ArrayDesc desc(blacs_h);
    desc.init_square_blk(4, 4, 0, 0);

    auto response = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    matrix_m<std::complex<double>> original(4, 4, MAJOR::COL);
    for (int i = 0; i != 4; ++i)
    {
        const int ilo = desc.indx_g2l_r(i);
        for (int j = 0; j != 4; ++j)
        {
            const auto value =
                std::complex<double>(0.1 * (i + 1) + 0.01 * (j + 1), 0.001 * (i - j));
            original(i, j) = value;
            if (ilo < 0) continue;
            const int jlo = desc.indx_g2l_c(j);
            if (jlo < 0) continue;
            response(ilo, jlo) = value;
        }
    }

    const matrix_m<std::complex<double>> chi0v_head(
        std::vector<std::vector<std::complex<double>>>{
            {std::complex<double>{0.21, 0.0}, std::complex<double>{0.01, 0.02},
             std::complex<double>{-0.03, 0.04}},
            {std::complex<double>{0.05, -0.01}, std::complex<double>{0.24, 0.0},
             std::complex<double>{0.07, 0.03}},
            {std::complex<double>{-0.02, -0.04}, std::complex<double>{0.08, -0.03},
             std::complex<double>{0.27, 0.0}}},
        MAJOR::COL);

    librpa_int::replace_rpa_response_head_only(response, chi0v_head, desc);

    const auto expected_head = (chi0v_head(0, 0) + chi0v_head(1, 1) + chi0v_head(2, 2)) / 3.0;
    for (int i = 0; i != 4; ++i)
    {
        const int ilo = desc.indx_g2l_r(i);
        if (ilo < 0) continue;
        for (int j = 0; j != 4; ++j)
        {
            const int jlo = desc.indx_g2l_c(j);
            if (jlo < 0) continue;
            const auto expected = (i == 0 && j == 0) ? expected_head : original(i, j);
            assert_complex_close(response(ilo, jlo), expected, 1e-12);
        }
    }
}

void test_head_only_trace_logdet_can_use_reduced_response(const BlacsCtxtHandler &blacs_h)
{
    ArrayDesc desc(blacs_h);
    desc.init_square_blk(3, 3, 0, 0);

    auto response = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    for (int i = 0; i != 3; ++i)
    {
        const int ilo = desc.indx_g2l_r(i);
        if (ilo < 0) continue;
        for (int j = 0; j != 3; ++j)
        {
            const int jlo = desc.indx_g2l_c(j);
            if (jlo < 0) continue;
            response(ilo, jlo) =
                std::complex<double>{0.02 * (i + 1) + 0.01 * (j + 1), 0.002 * (i - j)};
        }
    }

    const auto actual = librpa_int::compute_rpa_response_trace_logdet_blacs_2d(response, desc);

    const matrix_m<std::complex<double>> dense_response(
        std::vector<std::vector<std::complex<double>>>{
            {{0.03, 0.0}, {0.04, -0.002}, {0.05, -0.004}},
            {{0.05, 0.002}, {0.06, 0.0}, {0.07, -0.002}},
            {{0.07, 0.004}, {0.08, 0.002}, {0.09, 0.0}}},
        MAJOR::COL);
    std::complex<double> trace = 0.0;
    for (int i = 0; i != 3; ++i)
    {
        trace += dense_response(i, i);
    }
    const auto a = 1.0 - dense_response(0, 0);
    const auto b = -dense_response(0, 1);
    const auto c = -dense_response(0, 2);
    const auto d = -dense_response(1, 0);
    const auto e = 1.0 - dense_response(1, 1);
    const auto f = -dense_response(1, 2);
    const auto g = -dense_response(2, 0);
    const auto h = -dense_response(2, 1);
    const auto i = 1.0 - dense_response(2, 2);
    const auto det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    const auto expected = trace + std::log(det);

    assert_complex_close(actual, expected, 1e-12);

    ArrayDesc desc_full(blacs_h);
    desc_full.init_square_blk(5, 5, 0, 0);
    auto full_response = init_local_mat<std::complex<double>>(desc_full, MAJOR::COL);
    for (int i = 0; i != 3; ++i)
    {
        const int ilo = desc_full.indx_g2l_r(i);
        if (ilo < 0) continue;
        for (int j = 0; j != 3; ++j)
        {
            const int jlo = desc_full.indx_g2l_c(j);
            if (jlo < 0) continue;
            full_response(ilo, jlo) = dense_response(i, j);
        }
    }
    const auto full_actual =
        librpa_int::compute_rpa_response_trace_logdet_blacs_2d(full_response, desc_full);

    assert_complex_close(full_actual, actual, 1e-12);
}

void test_rpa_trace_log_average_uses_directional_head_and_wing()
{
    const matrix_m<std::complex<double>> head(
        std::vector<std::vector<std::complex<double>>>{
            {std::complex<double>{0.2, 0.0}, std::complex<double>{0.0, 0.0},
             std::complex<double>{0.0, 0.0}},
            {std::complex<double>{0.0, 0.0}, std::complex<double>{0.3, 0.0},
             std::complex<double>{0.0, 0.0}},
            {std::complex<double>{0.0, 0.0}, std::complex<double>{0.0, 0.0},
             std::complex<double>{0.4, 0.0}}},
        MAJOR::COL);
    const std::complex<double> body{0.1, 0.0};
    const std::array<std::complex<double>, 3> wing{std::complex<double>{0.05, 0.0},
                                                   std::complex<double>{0.02, 0.0},
                                                   std::complex<double>{0.01, 0.0}};
    const std::complex<double> body_inv = 1.0 / (1.0 - body);
    const matrix_m<std::complex<double>> schur_l(
        std::vector<std::vector<std::complex<double>>>{
            {1.0 - head(0, 0) - std::conj(wing[0]) * body_inv * wing[0],
             -std::conj(wing[0]) * body_inv * wing[1], -std::conj(wing[0]) * body_inv * wing[2]},
            {-std::conj(wing[1]) * body_inv * wing[0],
             1.0 - head(1, 1) - std::conj(wing[1]) * body_inv * wing[1],
             -std::conj(wing[1]) * body_inv * wing[2]},
            {-std::conj(wing[2]) * body_inv * wing[0], -std::conj(wing[2]) * body_inv * wing[1],
             1.0 - head(2, 2) - std::conj(wing[2]) * body_inv * wing[2]}},
        MAJOR::COL);
    const std::vector<double> qx{1.0, 0.0};
    const std::vector<double> qy{0.0, 1.0};
    const std::vector<double> qz{0.0, 0.0};
    const std::vector<double> weights{0.2, 0.6};
    const std::complex<double> trace_body = body;
    const std::complex<double> logdet_body = std::log(1.0 - body);

    const auto actual = librpa_int::compute_rpa_chi0v_headwing_trace_log_average(
        head, schur_l, trace_body, logdet_body, qx, qy, qz, weights);
    const auto direct_trace_log = [&](const double nx, const double ny, const double nz)
    {
        const auto directional_head = nx * (nx * head(0, 0) + ny * head(0, 1) + nz * head(0, 2)) +
                                      ny * (nx * head(1, 0) + ny * head(1, 1) + nz * head(1, 2)) +
                                      nz * (nx * head(2, 0) + ny * head(2, 1) + nz * head(2, 2));
        const auto directional_wing = nx * wing[0] + ny * wing[1] + nz * wing[2];
        const auto direct_det = (1.0 - directional_head) * (1.0 - body) -
                                std::conj(directional_wing) * directional_wing;
        return directional_head + body + std::log(direct_det);
    };
    const auto expected = weights[0] * direct_trace_log(qx[0], qy[0], qz[0]) +
                          weights[1] * direct_trace_log(qx[1], qy[1], qz[1]);

    assert_complex_close(actual, expected, 1e-12);
}

void test_rpa_headwing_regular_body_start_channel()
{
    librpa_int::RpaHeadwingSettings settings;

    settings.use_2d_dielectric = false;
    settings.rpa_headwing_body_start = 0;
    assert(librpa_int::rpa_headwing_regular_body_start_channel(settings) == 1);

    settings.use_2d_dielectric = true;
    settings.rpa_headwing_body_start = 0;
    assert(librpa_int::rpa_headwing_regular_body_start_channel(settings) == 1);

    settings.use_2d_dielectric = false;
    settings.rpa_headwing_body_start = 1;
    assert(librpa_int::rpa_headwing_regular_body_start_channel(settings) == 1);

    settings.use_2d_dielectric = true;
    settings.rpa_headwing_body_start = 4;
    assert(librpa_int::rpa_headwing_regular_body_start_channel(settings) == 4);
}

void test_rpa_headwing_gamma_cell_volume_uses_reciprocal_lattice()
{
    librpa_int::PeriodicBoundaryData pbc;
    pbc.latvec = librpa_int::Matrix3(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0);
    pbc.G = librpa_int::Matrix3(0.5, 0.0, 0.0, 0.0, 1.0 / 3.0, 0.0, 0.0, 0.0, 0.2);

    const double vol_3d = librpa_int::rpa_headwing_reciprocal_cell_volume(pbc, false);
    require_double_close(vol_3d, std::abs(pbc.G.Det()), 1e-14);

    const double vol_2d = librpa_int::rpa_headwing_reciprocal_cell_volume(pbc, true);
    const double expected_2d = std::abs(pbc.G.e11 * pbc.G.e22 - pbc.G.e12 * pbc.G.e21);
    require_double_close(vol_2d, expected_2d, 1e-14);

    pbc.set_period(4, 4, 4);
    require_double_close(librpa_int::rpa_headwing_gamma_cell_volume(pbc, false),
                         vol_3d / 64.0, 1e-14);
    require_double_close(librpa_int::rpa_headwing_gamma_cell_volume(pbc, true),
                         vol_2d / 64.0, 1e-14);
}

void test_rpa_chi0v_wing_desc_uses_global_rows(const BlacsCtxtHandler &blacs_h)
{
    ArrayDesc desc_body(blacs_h);
    desc_body.init_square_blk(10, 10, 0, 0);

    ArrayDesc desc_full_wing(blacs_h);
    desc_full_wing.init(11, 3, desc_body.mb(), 1, 0, 0);

    const auto desc_wing = librpa_int::make_rpa_chi0v_wing_desc(
        desc_body, 1, desc_full_wing.m_loc(), desc_full_wing.n_loc());

    if (desc_body.nprows() > 1)
    {
        assert(desc_full_wing.m_loc() < desc_full_wing.m());
    }
    assert(desc_wing.m() == 11);
    assert(desc_wing.n() == 3);
    assert(desc_wing.mb() == desc_body.mb());
    assert(desc_wing.nb() == 1);
    assert(desc_wing.m_loc() == desc_full_wing.m_loc());
    assert(desc_wing.n_loc() == desc_full_wing.n_loc());
}

void test_headwing_spin_weights()
{
    assert(std::abs(librpa_int::headwing_transition_weight(1.0, 0.25, 2, false) - 0.75) < 1e-12);
    assert(std::abs(librpa_int::headwing_spin_prefactor(2, false) - 1.0) < 1e-12);

    assert(std::abs(librpa_int::headwing_transition_weight(1.0, 0.25, 1, true) - 0.75) < 1e-12);
    assert(std::abs(librpa_int::headwing_spin_prefactor(1, true) - 1.0) < 1e-12);

    assert(std::abs(librpa_int::headwing_transition_weight(1.0, 0.25, 1, false) - 0.375) < 1e-12);
    assert(std::abs(librpa_int::headwing_spin_prefactor(1, false) - 2.0) < 1e-12);
}

void test_wing_cartesian_gram_is_invariant_under_row_phases()
{
    ComplexMatrix wing(2, 3);
    wing(0, 0) = {1.0, 1.0};
    wing(0, 1) = {2.0, -1.0};
    wing(0, 2) = {0.0, -1.0};
    wing(1, 0) = {0.5, -2.0};
    wing(1, 1) = {-1.0, 0.25};
    wing(1, 2) = {3.0, 0.5};

    ComplexMatrix phased = wing;
    for (int alpha = 0; alpha != 3; ++alpha)
    {
        phased(0, alpha) *= std::complex<double>{0.0, 1.0};
        phased(1, alpha) *= std::complex<double>{-1.0, 0.0};
    }

    const auto gram = librpa_int::compute_wing_cartesian_gram(wing);
    const auto phased_gram = librpa_int::compute_wing_cartesian_gram(phased);
    for (int alpha = 0; alpha != 3; ++alpha)
    {
        for (int beta = 0; beta != 3; ++beta)
        {
            assert_complex_close(gram.at(alpha).at(beta), phased_gram.at(alpha).at(beta), 1e-12);
        }
    }
    assert_complex_close(gram.at(0).at(0), {6.25, 0.0}, 1e-12);
}

void test_velocity_matrix_initialization()
{
    librpa_int::velocity_matrix_t velocity;
    librpa_int::initialize_velocity_matrix(velocity, 2, 3, 4);

    assert(velocity.size() == 2);
    for (int ispin = 0; ispin != 2; ++ispin)
    {
        assert(velocity[ispin].size() == 3);
        for (int ik = 0; ik != 3; ++ik)
        {
            assert(velocity[ispin][ik].size() == 3);
            for (int alpha = 0; alpha != 3; ++alpha)
            {
                assert(velocity[ispin][ik][alpha].nr == 4);
                assert(velocity[ispin][ik][alpha].nc == 4);
            }
        }
    }
}

void test_headwing_local_kpoints_prefers_kpoint_blacs_context()
{
    const auto all_k = librpa_int::headwing_local_kpoints(4, nullptr);
    assert((all_k == std::vector<int>{0, 1, 2, 3}));

    if (librpa_int::get_mpi_size(MPI_COMM_WORLD) != 4) return;

    KPointBlacsProcessShape shape(2, 2, true);
    KPointBlacsParallelContext kctx(shape, MPI_COMM_WORLD, 4);
    const auto local_k = librpa_int::headwing_local_kpoints(4, &kctx);

    if (kctx.kpoint_group_id() == 0)
        assert((local_k == std::vector<int>{0, 2}));
    else
        assert((local_k == std::vector<int>{1, 3}));

    const auto mismatched = librpa_int::headwing_local_kpoints(5, &kctx);
    assert((mismatched == std::vector<int>{0, 1, 2, 3, 4}));
}

void test_headwing_world_fourier_uses_all_R_blocks_at_nonzero_k()
{
    AtomicBasis basis_wfc({1});
    AtomicBasis basis_abf({1});
    librpa_int::Cs_LRI Cs_data;
    Cs_data.use_libri = true;
    Cs_data.data_libri[0][{0, {0, 0, 0}}] = make_scalar_cs_tensor(2.0);
    Cs_data.data_libri[0][{0, {1, 0, 0}}] = make_scalar_cs_tensor(3.0);
    Cs_data.data_libri[0][{0, {2, 0, 0}}] = make_scalar_cs_tensor(-1.0);

    const auto targets = librpa_int::build_headwing_full_bz_fourier_targets(
        {{0.0, 0.0, 0.0}, {0.25, 0.0, 0.0}});
    const auto Cs_IJ_k =
        librpa_int::fourier_headwing_cs_to_ijk(Cs_data, basis_wfc, basis_abf, targets);

    const auto &blocks = Cs_IJ_k.at(0);
    assert_complex_close(blocks.at({0, 0})(0, 0, 0), {4.0, 0.0}, 1e-12);
    assert_complex_close(blocks.at({0, 1})(0, 0, 0), {3.0, 3.0}, 1e-12);
}

void test_headwing_symmetry_fourier_target_ids_are_deterministic()
{
    librpa_int::SymmetryContext ctx;
    ctx.set_available();

    librpa_int::SymmetryKStar first;
    first.star_index = 0;
    first.k_ibz = {0.0, 0.0, 0.0};
    first.members.resize(2);
    first.members[0].k_bz = {0.0, 0.0, 0.0};
    first.members[1].k_bz = {0.5, 0.0, 0.0};
    ctx.kstars.push_back(first);

    librpa_int::SymmetryKStar second;
    second.star_index = 1;
    second.k_ibz = {0.25, 0.0, 0.0};
    second.members.resize(3);
    second.members[0].k_bz = {0.25, 0.0, 0.0};
    second.members[1].k_bz = {0.0, 0.25, 0.0};
    second.members[2].k_bz = {0.0, 0.0, 0.25};
    ctx.kstars.push_back(second);

    PeriodicBoundaryData pbc;
    const auto flattened = librpa_int::build_headwing_symmetry_fourier_targets(
        ctx, pbc, {first.k_ibz, second.k_ibz});

    assert((flattened.target_ids_by_ibz_member[0] == std::vector<int>{0, 1}));
    assert((flattened.target_ids_by_ibz_member[1] == std::vector<int>{2, 3, 4}));
    assert(flattened.targets.size() == 5);
    for (int target_id = 0; target_id != 5; ++target_id)
        assert(flattened.targets[target_id].target_id == target_id);
    assert(flattened.targets[0].owner_ik == 0);
    assert(flattened.targets[1].owner_ik == 0);
    assert(flattened.targets[2].owner_ik == 1);
    assert(flattened.targets[4].kfrac == second.members[2].k_bz);
}

void test_headwing_ijk_redistribution_is_owner_group_local()
{
    if (librpa_int::get_mpi_size(MPI_COMM_WORLD) != 4) return;

    KPointBlacsProcessShape shape(2, 2, true);
    KPointBlacsParallelContext kctx(shape, MPI_COMM_WORLD, 4);
    const auto desc_nao_nao = kctx.create_array_desc(2, 2, 1, 1);
    const AtomicBasis basis_wfc(std::vector<std::size_t>{1, 1});
    const AtomicBasis basis_abf(std::vector<std::size_t>{1, 1});
    const auto targets = librpa_int::build_headwing_full_bz_fourier_targets(
        {{0.0, 0.0, 0.0}, {0.25, 0.0, 0.0}, {0.5, 0.0, 0.0}, {0.75, 0.0, 0.0}});
    const std::set<int> local_iks(kctx.kpoints_local().begin(), kctx.kpoints_local().end());
    const auto requests = librpa_int::build_headwing_cs_ijk_requests(
        basis_wfc, targets, kctx.kpoints_local(), desc_nao_nao);

    for (const auto &[J, target_id] : requests.second)
    {
        (void)J;
        assert(local_iks.count(targets.at(target_id).owner_ik) == 1);
    }

#ifdef LIBRPA_USE_LIBRI
    librpa_int::Cs_LRI Cs_data;
    Cs_data.use_libri = true;
    if (librpa_int::get_mpi_rank(MPI_COMM_WORLD) == 0)
    {
        for (int I = 0; I != 2; ++I)
            for (int J = 0; J != 2; ++J)
                Cs_data.data_libri[I][{J, {0, 0, 0}}] =
                    make_scalar_cs_tensor(1.0 + 2.0 * I + J);
    }
    const auto Cs_IJ_k = librpa_int::redistribute_headwing_cs_ijk(
        Cs_data, basis_wfc, basis_abf, targets, kctx.kpoints_local(), desc_nao_nao,
        librpa_int::global::mpi_comm_global_h);
    for (const auto &[I, Jtargets] : Cs_IJ_k)
    {
        assert(requests.first.count(I) == 1);
        for (const auto &[Jtarget, tensor] : Jtargets)
        {
            assert(requests.second.count(Jtarget) == 1);
            assert(local_iks.count(targets.at(Jtarget.second).owner_ik) == 1);
            assert(std::abs(tensor(0, 0, 0)) > 0.0);
        }
    }
#endif
}

void test_accumulate_wing_mu_for_pair_matches_original_formula()
{
    const std::vector<double> omega{0.5, 1.25};
    const std::array<std::complex<double>, 3> velocity{std::complex<double>{0.2, -0.1},
                                                       std::complex<double>{-0.3, 0.4},
                                                       std::complex<double>{0.15, 0.05}};
    const std::complex<double> c_mn{0.7, -0.2};
    const double egap = 1.8;
    const double factor1 = 0.6;
    const double factor2 = 0.125;

    std::array<std::complex<double>, 6> accumulated{};
    librpa_int::accumulate_wing_mu_for_pair(omega, velocity, c_mn, egap, factor1, factor2,
                                            accumulated.data());

    for (std::size_t iomega = 0; iomega != omega.size(); ++iomega)
    {
        for (int alpha = 0; alpha != 3; ++alpha)
        {
            const auto denom = omega[iomega] * omega[iomega] + egap * egap;
            const auto expected = factor1 * std::conj(c_mn * velocity[alpha]) / denom +
                                  factor2 * c_mn * velocity[alpha] / denom;
            assert_complex_close(accumulated[iomega * 3 + alpha], expected, 1e-12);
        }
    }
}

SymmetryKStarMember make_headwing_wfc_atom_swap_member(
    const std::complex<double> &rot_0, const std::complex<double> &rot_1)
{
    SymmetryKStarMember member;
    member.spatial_isym = 0;
    member.k_bz = {0.0, 0.0, 0.0};

    SymmetryKAtomRotation atom_0;
    atom_0.atom_from = 0;
    atom_0.atom_to = 1;
    atom_0.atom_type = 0;
    atom_0.lmax = 0;
    atom_0.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
    atom_0.bloch_rsh_rotations[0](0, 0) = rot_0;

    SymmetryKAtomRotation atom_1;
    atom_1.atom_from = 1;
    atom_1.atom_to = 0;
    atom_1.atom_type = 0;
    atom_1.lmax = 0;
    atom_1.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
    atom_1.bloch_rsh_rotations[0](0, 0) = rot_1;

    member.atom_rotations = {atom_0, atom_1};
    return member;
}

void test_headwing_wfc_restore_applies_atom_permutation()
{
    SymmetryContext ctx;
    ctx.set_available();
    ctx.atom_to_type = {{0, 0}, {1, 0}};
    ctx.input_coord_frac = {{0, {0.0, 0.0, 0.0}}, {1, {0.5, 0.0, 0.0}}};

    SymmetryOperation identity_operation;
    identity_operation.rotation.Identity();
    identity_operation.translation = {0.0, 0.0, 0.0};
    ctx.rspace_operations.push_back(identity_operation);

    SpeciesBasisLayout layout;
    layout.label = "X";
    layout.set({0});
    const std::vector<SpeciesBasisLayout> layouts{layout};
    const std::map<librpa_int::atom_t, size_t> atom_nw{{0, 1}, {1, 1}};
    const auto member = make_headwing_wfc_atom_swap_member(
        {2.0, 0.5}, {3.0, -0.25});

    ComplexMatrix wfc_ibz(1, 2);
    wfc_ibz(0, 0) = {0.7, -0.2};
    wfc_ibz(0, 1) = {-0.4, 0.6};

    const auto wfc_bz = librpa_int::rotate_headwing_wfc_to_kstar_member(
        ctx, member, layouts, atom_nw, {0.0, 0.0, 0.0}, wfc_ibz, nullptr);

    assert_complex_close(wfc_bz(0, 0), wfc_ibz(0, 1) * std::complex<double>{3.0, -0.25},
                         1e-12);
    assert_complex_close(wfc_bz(0, 1), wfc_ibz(0, 0) * std::complex<double>{2.0, 0.5},
                         1e-12);
}

void test_headwing_wfc_restore_applies_time_reversal()
{
    SymmetryContext ctx;
    ctx.set_available();
    ctx.atom_to_type = {{0, 0}};
    ctx.input_coord_frac = {{0, {0.0, 0.0, 0.0}}};

    SymmetryOperation identity_operation;
    identity_operation.rotation.Identity();
    identity_operation.translation = {0.0, 0.0, 0.0};
    ctx.rspace_operations.push_back(identity_operation);

    SpeciesBasisLayout layout;
    layout.label = "X";
    layout.set({0});
    const std::vector<SpeciesBasisLayout> layouts{layout};
    const std::map<librpa_int::atom_t, size_t> atom_nw{{0, 1}};

    SymmetryKStarMember member;
    member.spatial_isym = 0;
    member.k_bz = {0.0, 0.0, 0.0};
    member.time_reversal = true;
    SymmetryKAtomRotation atom;
    atom.atom_from = 0;
    atom.atom_to = 0;
    atom.atom_type = 0;
    atom.lmax = 0;
    atom.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
    atom.bloch_rsh_rotations[0](0, 0) = {0.25, 0.75};
    member.atom_rotations.push_back(atom);

    ComplexMatrix wfc_ibz(1, 1);
    wfc_ibz(0, 0) = {0.6, -0.35};

    const auto wfc_bz = librpa_int::rotate_headwing_wfc_to_kstar_member(
        ctx, member, layouts, atom_nw, {0.0, 0.0, 0.0}, wfc_ibz, nullptr);

    assert_complex_close(wfc_bz(0, 0), std::conj(wfc_ibz(0, 0)) * std::complex<double>{0.25, 0.75},
                         1e-12);
}

void test_headwing_velocity_restore_uses_inverse_spatial_route()
{
    SymmetryContext ctx;
    ctx.lattice_vectors.Identity();
    SymmetryOperation operation;
    operation.rotation = Matrix3(0.0, -1.0, 0.0,
                                 1.0,  0.0, 0.0,
                                 0.0,  0.0, 1.0);
    ctx.rspace_operations = {operation};

    SymmetryKStarMember member;
    member.spatial_isym = 0;
    std::array<ComplexMatrix, 3> velocity_ibz;
    for (auto &component : velocity_ibz) component.create(1, 1);
    velocity_ibz[0](0, 0) = {1.0, 2.0};
    velocity_ibz[1](0, 0) = {3.0, -1.0};
    velocity_ibz[2](0, 0) = {-0.5, 0.25};

    const auto velocity_bz = librpa_int::rotate_headwing_velocity_to_kstar_member(
        ctx, member, velocity_ibz, 1, false);
    assert_complex_close(velocity_bz[0](0, 0), -velocity_ibz[1](0, 0), 1e-12);
    assert_complex_close(velocity_bz[1](0, 0), velocity_ibz[0](0, 0), 1e-12);
    assert_complex_close(velocity_bz[2](0, 0), velocity_ibz[2](0, 0), 1e-12);

    member.time_reversal = true;
    const auto velocity_bz_tr = librpa_int::rotate_headwing_velocity_to_kstar_member(
        ctx, member, velocity_ibz, 1, true);
    assert_complex_close(velocity_bz_tr[0](0, 0), std::conj(velocity_ibz[1](0, 0)), 1e-12);
    assert_complex_close(velocity_bz_tr[1](0, 0), -std::conj(velocity_ibz[0](0, 0)), 1e-12);
    assert_complex_close(velocity_bz_tr[2](0, 0), -std::conj(velocity_ibz[2](0, 0)), 1e-12);
}

void test_headwing_direct_full_bz_velocity_selects_kstar_member()
{
    librpa_int::velocity_matrix_t velocity_full;
    librpa_int::initialize_velocity_matrix(velocity_full, 1, 2, 1);
    velocity_full[0][0][0](0, 0) = {1.0, 0.0};
    velocity_full[0][1][0](0, 0) = {2.0, 0.0};
    velocity_full[0][1][1](0, 0) = {3.0, 0.0};
    velocity_full[0][1][2](0, 0) = {4.0, 0.0};

    const std::vector<std::vector<int>> member_source_ik{{1}};
    const auto &velocity = librpa_int::direct_full_bz_velocity_for_kstar_member(
        velocity_full, member_source_ik, 0, 0, 0);
    assert_complex_close(velocity[0](0, 0), {2.0, 0.0}, 1e-12);
    assert_complex_close(velocity[1](0, 0), {3.0, 0.0}, 1e-12);
    assert_complex_close(velocity[2](0, 0), {4.0, 0.0}, 1e-12);
}

void test_headwing_direct_full_bz_wfc_selects_same_kstar_member()
{
    MeanField wfc_full(1, 2, 1, 1);
    auto &wfc_k0 = wfc_full.get_eigenvectors()[0][0][0];
    wfc_k0.create(1, 1);
    wfc_k0(0, 0) = {1.0, 0.0};
    auto &wfc_k1 = wfc_full.get_eigenvectors()[0][0][1];
    wfc_k1.create(1, 1);
    wfc_k1(0, 0) = {0.0, 1.0};

    const std::vector<std::vector<int>> member_source_ik{{1}};
    const auto &wfc = librpa_int::direct_full_bz_wfc_for_kstar_member(
        wfc_full, member_source_ik, 0, 0, 0, 0);
    assert_complex_close(wfc(0, 0), {0.0, 1.0}, 1e-12);
}

void test_headwing_direct_full_bz_wfc_local_block(const BlacsCtxtHandler &blacs_h)
{
    ComplexMatrix wfc_full(5, 7);
    for (int iband = 0; iband != wfc_full.nr; ++iband)
        for (int iao = 0; iao != wfc_full.nc; ++iao)
            wfc_full(iband, iao) = {100.0 * iband + iao, iband - 0.1 * iao};

    ArrayDesc desc_wfc(blacs_h);
    desc_wfc.init(7, 5, 2, 3, 0, 0);
    const auto wfc_local = librpa_int::localize_direct_full_bz_wfc(wfc_full, desc_wfc);
    assert(wfc_local.nr == desc_wfc.n_loc());
    assert(wfc_local.nc == desc_wfc.m_loc());
    for (int jloc = 0; jloc != desc_wfc.n_loc(); ++jloc)
    {
        const int iband = desc_wfc.indx_l2g_c(jloc);
        for (int iloc = 0; iloc != desc_wfc.m_loc(); ++iloc)
        {
            const int iao = desc_wfc.indx_l2g_r(iloc);
            assert_complex_close(wfc_local(jloc, iloc), wfc_full(iband, iao), 1e-12);
        }
    }
}

RI::Tensor<double> make_single_value_tensor(const double value)
{
    auto data = std::make_shared<std::valarray<double>>(1);
    (*data)[0] = value;
    return RI::Tensor<double>({1UL, 1UL, 1UL}, data);
}

void compare_local_blacs_matrices(
    const std::pair<ArrayDesc, matrix_m<std::complex<double>>> &actual,
    const std::pair<ArrayDesc, matrix_m<std::complex<double>>> &expected,
    const double tolerance)
{
    assert(actual.first.m() == expected.first.m());
    assert(actual.first.n() == expected.first.n());
    assert(actual.first.m_loc() == expected.first.m_loc());
    assert(actual.first.n_loc() == expected.first.n_loc());
    for (int i = 0; i != actual.first.m_loc(); ++i)
    {
        for (int j = 0; j != actual.first.n_loc(); ++j)
        {
            assert_complex_close(actual.second(i, j), expected.second(i, j), tolerance);
        }
    }
}

void test_kblacs_transform_with_restored_wfc_matches_full_bz_atom_permutation(
    const BlacsCtxtHandler &blacs_h)
{
    SymmetryContext ctx;
    ctx.set_available();
    ctx.atom_to_type = {{0, 0}, {1, 0}};
    ctx.input_coord_frac = {{0, {0.0, 0.0, 0.0}}, {1, {0.5, 0.0, 0.0}}};

    SymmetryOperation identity_operation;
    identity_operation.rotation.Identity();
    identity_operation.translation = {0.0, 0.0, 0.0};
    ctx.rspace_operations.push_back(identity_operation);

    SpeciesBasisLayout layout;
    layout.label = "X";
    layout.set({0});
    const std::vector<SpeciesBasisLayout> layouts{layout};
    const std::map<librpa_int::atom_t, size_t> atom_nw{{0, 1}, {1, 1}};

    auto member = make_headwing_wfc_atom_swap_member({0.0, 1.0}, {-1.0, 0.0});
    member.k_bz = {0.5, 0.0, 0.0};

    MeanField mf_ibz(1, 1, 2, 2, 1);
    auto &wfc_ibz = mf_ibz.get_eigenvectors()[0][0][0];
    wfc_ibz.create(2, 2);
    wfc_ibz(0, 0) = {0.7, -0.2};
    wfc_ibz(0, 1) = {-0.4, 0.6};
    wfc_ibz(1, 0) = {0.3, 0.5};
    wfc_ibz(1, 1) = {-0.8, -0.1};

    const auto wfc_bz = librpa_int::rotate_headwing_wfc_to_kstar_member(
        ctx, member, layouts, atom_nw, {0.0, 0.0, 0.0}, wfc_ibz, &member.k_bz);

    MeanField mf_full(1, 1, 2, 2, 1);
    auto &wfc_full = mf_full.get_eigenvectors()[0][0][0];
    wfc_full.create(2, 2);
    for (int ib = 0; ib != 2; ++ib)
    {
        for (int iao = 0; iao != 2; ++iao)
        {
            wfc_full(ib, iao) = wfc_bz(ib, iao);
        }
    }

    librpa_int::velocity_matrix_t velocity;
    librpa_int::initialize_velocity_matrix(velocity, 1, 1, 2);
    AtomicBasis basis_wfc(std::vector<size_t>{1, 1});
    AtomicBasis basis_abf(std::vector<size_t>{1, 1});
    PeriodicBoundaryData pbc;
    const std::vector<Vector3_Order<double>> kfrac_ibz{{0.0, 0.0, 0.0}};
    const std::vector<double> omega{0.5};

    diele_func df_ibz(mf_ibz, velocity, kfrac_ibz, basis_wfc, basis_abf, omega, 2, 2, 1, 1,
                      pbc, librpa_int::global::mpi_comm_global_h, blacs_h);
    diele_func df_full(mf_full, velocity, {member.k_bz}, basis_wfc, basis_abf, omega, 2, 2, 1, 1,
                       pbc, librpa_int::global::mpi_comm_global_h, blacs_h);

    std::map<int, std::map<librpa_int::libri_types<int, int>::TAC, RI::Tensor<double>>> Cs_IJ;
    Cs_IJ[0][{0, {0, 0, 0}}] = make_single_value_tensor(1.0);
    Cs_IJ[0][{1, {0, 0, 0}}] = make_single_value_tensor(-0.35);
    Cs_IJ[0][{1, {1, 0, 0}}] = make_single_value_tensor(0.42);

    std::vector<std::vector<const ComplexMatrix *>> restored_wfc_ptrs(
        1, std::vector<const ComplexMatrix *>(1, &wfc_bz));
    const auto restored = df_ibz.transform_Cs2mnk_kblacs(
        0, 0, Cs_IJ, blacs_h, member.k_bz, &restored_wfc_ptrs);
    const auto full = df_full.transform_Cs2mnk_kblacs(
        0, 0, Cs_IJ, blacs_h, member.k_bz);

    compare_local_blacs_matrices(restored, full, 1e-12);
}

void test_kblacs_transform_matches_original_transform(const BlacsCtxtHandler &blacs_h)
{
    const int nprocs = librpa_int::get_mpi_size(MPI_COMM_WORLD);
    KPointBlacsProcessShape shape(1, nprocs, true);
    KPointBlacsParallelContext kctx(shape, MPI_COMM_WORLD, 1);
    const auto desc_wfc = kctx.create_array_desc(2, 2, 2, 2);

    MeanField mf(1, 1, 2, 2, 1);
    auto &wfc = mf.get_eigenvectors()[0][0][0];
    wfc.create(2, 2);
    wfc(0, 0) = {0.7, 0.2};
    wfc(0, 1) = {-0.3, 0.4};
    wfc(1, 0) = {0.5, -0.1};
    wfc(1, 1) = {0.9, 0.3};

    librpa_int::velocity_matrix_t velocity;
    librpa_int::initialize_velocity_matrix(velocity, 1, 1, 2);
    AtomicBasis basis_wfc({2});
    AtomicBasis basis_abf({1});
    PeriodicBoundaryData pbc;
    const std::vector<Vector3_Order<double>> kfrac{{0.25, 0.0, 0.0}};
    const std::vector<double> omega{0.5};

    diele_func df(mf, velocity, kfrac, basis_wfc, basis_abf, omega, 2, 2, 1, 1, pbc,
                  librpa_int::global::mpi_comm_global_h, blacs_h, true, &kctx, &desc_wfc);

    auto tensor_R0 = std::make_shared<std::valarray<double>>(4);
    (*tensor_R0)[0] = 1.0;
    (*tensor_R0)[1] = 0.2;
    (*tensor_R0)[2] = -0.4;
    (*tensor_R0)[3] = 0.8;
    auto tensor_R1 = std::make_shared<std::valarray<double>>(4);
    (*tensor_R1)[0] = 0.3;
    (*tensor_R1)[1] = -0.1;
    (*tensor_R1)[2] = 0.2;
    (*tensor_R1)[3] = 0.5;
    std::map<int, std::map<librpa_int::libri_types<int, int>::TAC, RI::Tensor<double>>> Cs_IJ;
    Cs_IJ[0][{0, {0, 0, 0}}] = RI::Tensor<double>({1UL, 2UL, 2UL}, tensor_R0);
    Cs_IJ[0][{0, {1, 0, 0}}] = RI::Tensor<double>({1UL, 2UL, 2UL}, tensor_R1);

    librpa_int::Cs_LRI Cs_data;
    Cs_data.use_libri = true;
    Cs_data.data_libri = Cs_IJ;
    const auto targets = librpa_int::build_headwing_full_bz_fourier_targets(kfrac);
    const auto Cs_IJ_k =
        librpa_int::fourier_headwing_cs_to_ijk(Cs_data, basis_wfc, basis_abf, targets);

    auto original = df.transform_Cs2mnk(0, 0, Cs_IJ);
    auto kblacs = df.transform_Cs2mnk_kblacs(0, 0, Cs_IJ, kctx.blacs_h, kfrac[0]);
    auto prefourier = df.transform_Cs2mnk_kblacs(0, 0, 0, Cs_IJ_k, kctx.blacs_h);

    assert(original.first.m() == kblacs.first.m());
    assert(original.first.n() == kblacs.first.n());
    assert(original.first.m_loc() == kblacs.first.m_loc());
    assert(original.first.n_loc() == kblacs.first.n_loc());
    for (int i = 0; i != original.first.m_loc(); ++i)
    {
        for (int j = 0; j != original.first.n_loc(); ++j)
        {
            assert_complex_close(kblacs.second(i, j), original.second(i, j), 1e-12);
            assert_complex_close(prefourier.second(i, j), kblacs.second(i, j), 1e-12);
        }
    }

    std::vector<std::vector<ComplexMatrix>> override_storage(1);
    override_storage[0].resize(1);
    std::vector<std::vector<const ComplexMatrix *>> override_ptrs(1);
    override_ptrs[0].assign(1, nullptr);
    if (desc_wfc.is_src())
    {
        auto &override_wfc = override_storage[0][0];
        override_wfc.create(2, 2);
        override_wfc(0, 0) = {0.2, -0.4};
        override_wfc(0, 1) = {0.6, 0.1};
        override_wfc(1, 0) = {-0.5, 0.3};
        override_wfc(1, 1) = {0.4, -0.2};
        override_ptrs[0][0] = &override_wfc;
    }
    const auto real_override = df.transform_Cs2mnk_kblacs(
        0, 0, Cs_IJ, kctx.blacs_h, kfrac[0], &override_ptrs);
    const auto prefourier_override = df.transform_Cs2mnk_kblacs(
        0, 0, 0, Cs_IJ_k, kctx.blacs_h, &override_ptrs);
    int override_changed_local = 0;
    for (int i = 0; i != real_override.first.m_loc(); ++i)
    {
        for (int j = 0; j != real_override.first.n_loc(); ++j)
        {
            assert_complex_close(prefourier_override.second(i, j), real_override.second(i, j),
                                 1e-12);
            if (std::abs(real_override.second(i, j) - kblacs.second(i, j)) > 1e-12)
                override_changed_local = 1;
        }
    }
    int override_changed = 0;
    MPI_Allreduce(&override_changed_local, &override_changed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    assert(override_changed == 1);
}

void test_kblacs_transform_uses_rectangular_opt_128_blocks()
{
    constexpr int n_basis = 320;
    constexpr int n_states = 300;
    constexpr int n_ao_Mu = 160;

    const int nprocs = librpa_int::get_mpi_size(MPI_COMM_WORLD);
    KPointBlacsProcessShape shape(1, nprocs, true);
    KPointBlacsParallelContext kctx(shape, MPI_COMM_WORLD, 1);
    const auto desc_wfc_full =
        kctx.create_array_desc(n_basis, n_states, n_basis, n_states);
    const int block_ao =
        librpa_int::get_capped_blacs_block_size(
            n_basis, librpa_int::wfc_gemm_block_size_opt, kctx.blacs_h);
    const int block_band =
        librpa_int::get_capped_blacs_block_size(
            n_states, librpa_int::wfc_gemm_block_size_opt, kctx.blacs_h);
    const auto desc_wfc =
        kctx.create_array_desc(n_basis, n_states, block_ao, block_band);

    MeanField mf(1, 1, n_states, n_basis, 1);
    if (desc_wfc_full.is_src())
    {
        auto &wfc = mf.get_eigenvectors()[0][0][0];
        wfc.create(n_states, n_basis);
        wfc.zero_out();
        for (int ib = 0; ib != n_states; ++ib) wfc(ib, ib) = {1.0, 0.0};
    }
    librpa_int::redistribute_meanfield_eigvecs_kblacs(
        mf, kctx, desc_wfc_full, desc_wfc, "test headwing");

    librpa_int::velocity_matrix_t velocity;
    librpa_int::initialize_velocity_matrix(velocity, 1, 1, n_states);
    AtomicBasis basis_wfc(std::vector<std::size_t>{n_ao_Mu, n_basis - n_ao_Mu});
    AtomicBasis basis_abf(std::vector<std::size_t>{1, 1});
    PeriodicBoundaryData pbc;
    const std::vector<Vector3_Order<double>> kfrac{{0.0, 0.0, 0.0}};
    const std::vector<double> omega{0.5};

    diele_func df(mf, velocity, kfrac, basis_wfc, basis_abf, omega, n_basis, n_states, 1,
                  2, pbc, librpa_int::global::mpi_comm_global_h, kctx.blacs_h, true, &kctx,
                  &desc_wfc);

    auto tensor_data = std::make_shared<std::valarray<std::complex<double>>>(
        std::complex<double>{0.0, 0.0}, static_cast<std::size_t>(n_ao_Mu) * n_ao_Mu);
    for (int i = 0; i != n_ao_Mu; ++i)
        (*tensor_data)[static_cast<std::size_t>(i) * n_ao_Mu + i] = {1.0, 0.0};

    librpa_int::HeadwingCsIJKMap Cs_IJ_k;
    Cs_IJ_k[0][{0, 0}] = RI::Tensor<std::complex<double>>(
        {1UL, static_cast<std::size_t>(n_ao_Mu), static_cast<std::size_t>(n_ao_Mu)},
        tensor_data);

    const auto transformed =
        df.transform_Cs2mnk_kblacs(0, 0, 0, Cs_IJ_k, kctx.blacs_h);
    const auto &desc = transformed.first;
    const auto &matrix = transformed.second;
    assert(desc.m() == n_states && desc.n() == n_states);
    assert(desc.mb() == librpa_int::wfc_gemm_block_size_opt &&
           desc.nb() == librpa_int::wfc_gemm_block_size_opt);

    for (int ilo = 0; ilo != desc.m_loc(); ++ilo)
    {
        const int i = desc.indx_l2g_r(ilo);
        for (int jlo = 0; jlo != desc.n_loc(); ++jlo)
        {
            const int j = desc.indx_l2g_c(jlo);
            const std::complex<double> expected =
                i == j && i < n_ao_Mu ? std::complex<double>{2.0, 0.0}
                                       : std::complex<double>{0.0, 0.0};
            assert_complex_close(matrix(ilo, jlo), expected, 1e-12);
        }
    }
}

void test_transform_Cs2mnk_can_keep_spin_channels_separate(const BlacsCtxtHandler &blacs_h)
{
    const int nprocs = librpa_int::get_mpi_size(MPI_COMM_WORLD);
    KPointBlacsProcessShape shape(1, nprocs, true);
    KPointBlacsParallelContext kctx(shape, MPI_COMM_WORLD, 1);
    const auto desc_wfc = kctx.create_array_desc(2, 2, 2, 2);

    MeanField mf(2, 1, 2, 2, 1);
    auto &wfc_up = mf.get_eigenvectors()[0][0][0];
    auto &wfc_dn = mf.get_eigenvectors()[1][0][0];
    wfc_up.create(2, 2);
    wfc_dn.create(2, 2);
    wfc_up(0, 0) = {0.7, 0.2};
    wfc_up(0, 1) = {-0.3, 0.4};
    wfc_up(1, 0) = {0.5, -0.1};
    wfc_up(1, 1) = {0.9, 0.3};
    wfc_dn(0, 0) = {0.2, -0.6};
    wfc_dn(0, 1) = {0.8, 0.1};
    wfc_dn(1, 0) = {-0.4, 0.5};
    wfc_dn(1, 1) = {0.6, -0.2};

    librpa_int::velocity_matrix_t velocity;
    librpa_int::initialize_velocity_matrix(velocity, 2, 1, 2);
    AtomicBasis basis_wfc({2});
    AtomicBasis basis_abf({1});
    PeriodicBoundaryData pbc;
    const std::vector<Vector3_Order<double>> kfrac{{0.0, 0.0, 0.0}};
    const std::vector<double> omega{0.5};

    diele_func df(mf, velocity, kfrac, basis_wfc, basis_abf, omega, 2, 2, 2, 1, pbc,
                  librpa_int::global::mpi_comm_global_h, blacs_h, true, &kctx, &desc_wfc);

    auto tensor_data = std::make_shared<std::valarray<double>>(4);
    (*tensor_data)[0] = 1.0;
    (*tensor_data)[1] = 0.2;
    (*tensor_data)[2] = -0.4;
    (*tensor_data)[3] = 0.8;
    std::map<int, std::map<librpa_int::libri_types<int, int>::TAC, RI::Tensor<double>>> Cs_IJ;
    Cs_IJ[0][{0, {0, 0, 0}}] = RI::Tensor<double>({1UL, 2UL, 2UL}, tensor_data);
    librpa_int::Cs_LRI Cs_data;
    Cs_data.use_libri = true;
    Cs_data.data_libri = Cs_IJ;
    const auto Cs_IJ_k = librpa_int::fourier_headwing_cs_to_ijk(
        Cs_data, basis_wfc, basis_abf,
        librpa_int::build_headwing_full_bz_fourier_targets(kfrac));

    const auto all_spin = df.transform_Cs2mnk(0, 0, Cs_IJ);
    const auto spin_up = df.transform_Cs2mnk(0, 0, Cs_IJ, 0);
    const auto spin_dn = df.transform_Cs2mnk(0, 0, Cs_IJ, 1);
    const auto all_spin_k = df.transform_Cs2mnk_kblacs(0, 0, 0, Cs_IJ_k, kctx.blacs_h);
    const auto spin_up_k = df.transform_Cs2mnk_kblacs(0, 0, 0, Cs_IJ_k, kctx.blacs_h, nullptr, 0);
    const auto spin_dn_k = df.transform_Cs2mnk_kblacs(0, 0, 0, Cs_IJ_k, kctx.blacs_h, nullptr, 1);

    int spin_channels_differ_local = 0;
    for (int i = 0; i != all_spin.first.m_loc(); ++i)
    {
        for (int j = 0; j != all_spin.first.n_loc(); ++j)
        {
            assert_complex_close(all_spin.second(i, j), spin_up.second(i, j) + spin_dn.second(i, j),
                                 1e-12);
            assert_complex_close(all_spin_k.second(i, j),
                                 spin_up_k.second(i, j) + spin_dn_k.second(i, j), 1e-12);
            if (std::abs(spin_up_k.second(i, j) - spin_dn_k.second(i, j)) > 1e-12)
                spin_channels_differ_local = 1;
        }
    }
    int spin_channels_differ = 0;
    MPI_Allreduce(&spin_channels_differ_local, &spin_channels_differ, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    assert(spin_channels_differ == 1);
}

void test_head_initialization_does_not_require_coulomb_diagonalization(
    const BlacsCtxtHandler &blacs_h)
{
    MeanField mf(1, 1, 2, 1);
    mf.get_eigenvals()[0](0, 0) = -0.5;
    mf.get_eigenvals()[0](0, 1) = 0.5;
    mf.get_weight()[0](0, 0) = 2.0;
    mf.get_weight()[0](0, 1) = 0.0;

    librpa_int::velocity_matrix_t velocity;
    librpa_int::initialize_velocity_matrix(velocity, 1, 1, 2);
    for (int alpha = 0; alpha != 3; ++alpha)
    {
        velocity[0][0][alpha](1, 0) = std::complex<double>{0.1 * (alpha + 1), 0.0};
        velocity[0][0][alpha](0, 1) = std::complex<double>{0.1 * (alpha + 1), 0.0};
    }

    AtomicBasis basis_wfc({1});
    AtomicBasis basis_abf({1});
    PeriodicBoundaryData pbc;
    const std::vector<Vector3_Order<double>> kfrac{{0.0, 0.0, 0.0}};
    const std::vector<double> omega{0.5};
    const atpair_k_cplx_mat_t empty_vq;

    diele_func df(mf, velocity, kfrac, basis_wfc, basis_abf, omega, 1, 2, 1, 1, pbc,
                  librpa_int::global::mpi_comm_global_h, blacs_h);

    df.init(0.0, empty_vq);
    df.cal_head();
    assert(df.get_head_vec().size() == 1);
}

void add_scalar_wq_block(
    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        &wq,
    const atom_t atom_i,
    const atom_t atom_j,
    const Vector3_Order<double> &q,
    const std::complex<double> value)
{
    auto &block = wq[atom_i][atom_j][q];
    block = matrix_m<std::complex<double>>(1, 1, MAJOR::ROW);
    block(0, 0) = value;
}

librpa_int::symmetry_atom_block_matrix_map_t scalar_wq_to_blocks(
    const std::map<atom_t, std::map<atom_t, std::complex<double>>> &values)
{
    librpa_int::symmetry_atom_block_matrix_map_t blocks;
    for (const auto &[atom_i, row] : values)
    {
        for (const auto &[atom_j, value] : row)
        {
            blocks[atom_i][atom_j] = ComplexMatrix(1, 1);
            blocks[atom_i][atom_j](0, 0) = value;
        }
    }
    return blocks;
}

void add_scalar_wq_blocks(
    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        &wq,
    const Vector3_Order<double> &q,
    const librpa_int::symmetry_atom_block_matrix_map_t &blocks)
{
    for (const auto &[atom_i, row] : blocks)
    {
        for (const auto &[atom_j, block] : row)
        {
            add_scalar_wq_block(wq, atom_i, atom_j, q, block(0, 0));
        }
    }
}

PeriodicBoundaryData make_wq_full_pbc()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    const std::vector<double> kvecs{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 3.0, 0.0, 0.0,
        librpa_int::TWO_PI * 2.0 / 3.0, 0.0, 0.0};
    pbc.set_kgrids_kvec(3, 1, 1, kvecs);
    return pbc;
}

PeriodicBoundaryData make_wq_reduced_pbc()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    const std::vector<double> kvecs_ibz{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 3.0, 0.0, 0.0};
    const std::vector<std::vector<Vector3_Order<double>>> full_kstars{
        {{0.0, 0.0, 0.0}},
        {{1.0 / 3.0, 0.0, 0.0}, {-1.0 / 3.0, 0.0, 0.0}}};
    pbc.set_irreducible_kgrids_kvec(3, 1, 1, kvecs_ibz, full_kstars);
    return pbc;
}

SymmetryContext make_two_atom_inversion_context(const PeriodicBoundaryData &pbc)
{
    SymmetryContext ctx;
    const Matrix3 lattice(1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0);
    ctx.set_crystal_structure(
        lattice, lattice,
        {{0, 0}, {1, 0}},
        {{0, {0.25, 0.0, 0.0}}, {1, {0.75, 0.0, 0.0}}});

    SymmetryOperation identity;
    identity.rotation.Identity();
    identity.translation = {0.0, 0.0, 0.0};

    SymmetryOperation inversion;
    inversion.rotation = Matrix3(-1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0);
    inversion.translation = {0.0, 0.0, 0.0};

    ctx.set_rspace_operations({identity, inversion});
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);
    ctx.build_rsh_rotations({-1,
                             0,
                             LIBRPA_ANGULAR_ORDER_NATURAL,
                             LIBRPA_RSH_COEFF_1_M,
                             LIBRPA_RSH_COEFF_1_M},
                            0);
    ctx.build_kstar_member_rotations(0);
    return ctx;
}

void assert_wq_rspace_maps_close(
    const atom_mapping<std::map<Vector3_Order<int>, matrix_m<std::complex<double>>>>::pair_t_old
        &actual,
    const atom_mapping<std::map<Vector3_Order<int>, matrix_m<std::complex<double>>>>::pair_t_old
        &expected)
{
    for (const auto &[atom_i, expected_row] : expected)
    {
        assert(actual.count(atom_i) != 0);
        for (const auto &[atom_j, expected_Rs] : expected_row)
        {
            assert(actual.at(atom_i).count(atom_j) != 0);
            for (const auto &[R, expected_block] : expected_Rs)
            {
                assert(actual.at(atom_i).at(atom_j).count(R) != 0);
                const auto &actual_block = actual.at(atom_i).at(atom_j).at(R);
                if (std::abs(actual_block(0, 0) - expected_block(0, 0)) >= 1e-12)
                {
                    std::cerr << "atom_pair=(" << atom_i << "," << atom_j << ") R=("
                              << R.x << "," << R.y << "," << R.z << ")" << std::endl;
                }
                assert_complex_close(actual_block(0, 0), expected_block(0, 0), 1e-12);
            }
        }
    }
}

void test_wq_to_wr_symmetry_reduced_q_matches_full_bz()
{
    const auto pbc_full = make_wq_full_pbc();
    const auto pbc_sym = make_wq_reduced_pbc();
    auto ctx = make_two_atom_inversion_context(pbc_sym);

    AtomicBasis basis_abf(std::vector<std::size_t>{1, 1});
    basis_abf.set_l_shells({{0}, {0}});
    const auto layouts = basis_abf.build_species_basis_layouts(ctx.atom_to_type);
    const std::map<atom_t, size_t> atom_nabf{{0, 1}, {1, 1}};

    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_sym;
    const auto q_gamma_sym = pbc_sym.klist.at(0);
    const auto q_rep_sym = pbc_sym.klist.at(1);
    const auto gamma_blocks = scalar_wq_to_blocks({
        {0, {{0, {1.5, 0.0}}, {1, {0.4, 0.0}}}},
        {1, {{0, {0.4, 0.0}}, {1, {1.5, 0.0}}}}});
    const auto rep_blocks = scalar_wq_to_blocks({
        {0, {{0, {2.1, 0.0}}, {1, {-0.7, 0.5}}}},
        {1, {{0, {-0.7, -0.5}}, {1, {1.4, 0.0}}}}});
    add_scalar_wq_blocks(wq_sym, q_gamma_sym, gamma_blocks);
    add_scalar_wq_blocks(wq_sym, q_rep_sym, rep_blocks);
    const double symmetry_collective_scale =
        1.0 / static_cast<double>(librpa_int::global::mpi_comm_global_h.nprocs);
    for (auto &[atom_i, row] : wq_sym)
    {
        for (auto &[atom_j, q_blocks] : row)
        {
            for (auto &[q, block] : q_blocks)
            {
                block *= symmetry_collective_scale;
            }
        }
    }

    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_full;
    add_scalar_wq_blocks(wq_full, pbc_full.klist.at(0), gamma_blocks);
    add_scalar_wq_blocks(wq_full, pbc_full.klist.at(1), rep_blocks);
    const auto inversion_minus_blocks = scalar_wq_to_blocks({
        {0, {{0, {1.4, 0.0}}, {1, {-0.7, -0.5}}}},
        {1, {{0, {-0.7, 0.5}}, {1, {2.1, 0.0}}}}});
    add_scalar_wq_blocks(wq_full, pbc_full.klist.at(2), inversion_minus_blocks);

    const TFGrids dummy_tfg;
    SymmetryContext no_symmetry;
    const auto expected = librpa_int::FT_Wc_q2R(
        librpa_int::global::mpi_comm_global_h, basis_abf, no_symmetry, wq_full,
        dummy_tfg, pbc_full, pbc_full.Rlist, false, "", false);
    const auto actual = librpa_int::FT_Wc_q2R(
        librpa_int::global::mpi_comm_global_h, basis_abf, ctx, wq_sym,
        dummy_tfg, pbc_sym, pbc_sym.Rlist, false, "", true);

    assert_wq_rspace_maps_close(actual, expected);
}

void test_wq_to_wr_symmetry_collective_handles_empty_local_rank()
{
    const auto pbc_full = make_wq_full_pbc();
    const auto pbc_sym = make_wq_reduced_pbc();
    auto ctx = make_two_atom_inversion_context(pbc_sym);

    AtomicBasis basis_abf(std::vector<std::size_t>{1, 1});
    basis_abf.set_l_shells({{0}, {0}});

    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_sym;
    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_full;
    if (librpa_int::global::mpi_comm_global_h.is_root())
    {
        const auto gamma_blocks = scalar_wq_to_blocks({
            {0, {{0, {1.5, 0.0}}, {1, {0.4, 0.0}}}},
            {1, {{0, {0.4, 0.0}}, {1, {1.5, 0.0}}}}});
        const auto rep_blocks = scalar_wq_to_blocks({
            {0, {{0, {2.1, 0.0}}, {1, {-0.7, 0.5}}}},
            {1, {{0, {-0.7, -0.5}}, {1, {1.4, 0.0}}}}});
        const auto inversion_minus_blocks = scalar_wq_to_blocks({
            {0, {{0, {1.4, 0.0}}, {1, {-0.7, -0.5}}}},
            {1, {{0, {-0.7, 0.5}}, {1, {2.1, 0.0}}}}});

        add_scalar_wq_blocks(wq_sym, pbc_sym.klist.at(0), gamma_blocks);
        add_scalar_wq_blocks(wq_sym, pbc_sym.klist.at(1), rep_blocks);
        add_scalar_wq_blocks(wq_full, pbc_full.klist.at(0), gamma_blocks);
        add_scalar_wq_blocks(wq_full, pbc_full.klist.at(1), rep_blocks);
        add_scalar_wq_blocks(wq_full, pbc_full.klist.at(2), inversion_minus_blocks);
    }

    const TFGrids dummy_tfg;
    SymmetryContext no_symmetry;
    const auto expected = librpa_int::FT_Wc_q2R(
        librpa_int::global::mpi_comm_global_h, basis_abf, no_symmetry, wq_full,
        dummy_tfg, pbc_full, pbc_full.Rlist, false, "", false);
    const auto actual = librpa_int::FT_Wc_q2R(
        librpa_int::global::mpi_comm_global_h, basis_abf, ctx, wq_sym,
        dummy_tfg, pbc_sym, pbc_sym.Rlist, false, "", true);

    assert_wq_rspace_maps_close(actual, expected);
    if (!librpa_int::global::mpi_comm_global_h.is_root())
    {
        assert(expected.empty());
        assert(actual.empty());
    }
}

Matz dense_wq_from_scalar_blocks(const librpa_int::symmetry_atom_block_matrix_map_t &blocks,
                                 const ArrayDesc &desc)
{
    Matz mat(desc.m_loc(), desc.n_loc(), MAJOR::COL);
    for (int i_local = 0; i_local < desc.m_loc(); ++i_local)
    {
        const int atom_i = desc.indx_l2g_r(i_local);
        for (int j_local = 0; j_local < desc.n_loc(); ++j_local)
        {
            const int atom_j = desc.indx_l2g_c(j_local);
            mat(i_local, j_local) = blocks.at(static_cast<atom_t>(atom_i))
                                        .at(static_cast<atom_t>(atom_j))(0, 0);
        }
    }
    return mat;
}

void assert_dense_wq_rspace_maps_close(
    const std::map<double, std::map<Vector3_Order<int>, Matz>> &actual,
    const std::map<double, std::map<Vector3_Order<int>, Matz>> &expected)
{
    for (const auto &[freq, expected_Rs] : expected)
    {
        assert(actual.count(freq) != 0);
        for (const auto &[R, expected_mat] : expected_Rs)
        {
            assert(actual.at(freq).count(R) != 0);
            const auto diff = actual.at(freq).at(R) - expected_mat;
            double max_abs = 0.0;
            for (int i = 0; i < diff.nr(); ++i)
            {
                for (int j = 0; j < diff.nc(); ++j)
                {
                    max_abs = std::max(max_abs, std::abs(diff(i, j)));
                }
            }
            if (max_abs >= 1e-12)
            {
                std::cerr << "freq=" << freq << " R=(" << R.x << "," << R.y << "," << R.z
                          << ") max_abs=" << max_abs << std::endl;
                for (int i = 0; i < diff.nr(); ++i)
                {
                    for (int j = 0; j < diff.nc(); ++j)
                    {
                        std::cerr << "  (" << i << "," << j << ") actual="
                                  << actual.at(freq).at(R)(i, j) << " expected="
                                  << expected_mat(i, j) << " diff=" << diff(i, j) << std::endl;
                    }
                }
            }
            assert(max_abs < 1e-12);
        }
    }
}

void test_dense_wq_to_wr_symmetry_reduced_q_matches_full_bz(const BlacsCtxtHandler &blacs_h)
{
    const auto pbc_full = make_wq_full_pbc();
    const auto pbc_sym = make_wq_reduced_pbc();
    auto ctx = make_two_atom_inversion_context(pbc_sym);
    const auto qpoint_view = build_symmetry_qpoint_view(ctx, pbc_sym, true);
    assert(qpoint_view.restore_mode == SymmetryQPointRestoreMode::FULL_CRYSTAL);

    AtomicBasis basis_abf(std::vector<std::size_t>{1, 1});
    basis_abf.set_l_shells({{0}, {0}});
    const auto layouts = basis_abf.build_species_basis_layouts(ctx.atom_to_type);
    const std::map<atom_t, size_t> atom_nabf{{0, 1}, {1, 1}};
    ArrayDesc ad_Wc(blacs_h);
    ad_Wc.init(2, 2, 2, 2, 0, 0);

    const auto gamma_blocks = scalar_wq_to_blocks({
        {0, {{0, {1.5, 0.0}}, {1, {0.4, 0.0}}}},
        {1, {{0, {0.4, 0.0}}, {1, {1.5, 0.0}}}}});
    const auto rep_blocks = scalar_wq_to_blocks({
        {0, {{0, {2.1, 0.0}}, {1, {-0.7, 0.5}}}},
        {1, {{0, {-0.7, -0.5}}, {1, {1.4, 0.0}}}}});

    constexpr double freq = 0.25;
    std::map<double, std::map<Vector3_Order<double>, Matz>> wq_sym;
    wq_sym[freq][pbc_sym.klist.at(0)] = dense_wq_from_scalar_blocks(gamma_blocks, ad_Wc);
    wq_sym[freq][pbc_sym.klist.at(1)] = dense_wq_from_scalar_blocks(rep_blocks, ad_Wc);

    std::map<double, std::map<Vector3_Order<double>, Matz>> wq_full;
    wq_full[freq][pbc_full.klist.at(0)] = dense_wq_from_scalar_blocks(gamma_blocks, ad_Wc);
    wq_full[freq][pbc_full.klist.at(1)] = dense_wq_from_scalar_blocks(rep_blocks, ad_Wc);
    const auto inversion_minus_blocks = scalar_wq_to_blocks({
        {0, {{0, {1.4, 0.0}}, {1, {-0.7, -0.5}}}},
        {1, {{0, {-0.7, 0.5}}, {1, {2.1, 0.0}}}}});
    wq_full[freq][pbc_full.klist.at(2)] =
        dense_wq_from_scalar_blocks(inversion_minus_blocks, ad_Wc);

    const auto expected = librpa_int::FT_Wc_freq_q(
        librpa_int::global::mpi_comm_global_h, wq_full, pbc_full, false);
    const auto actual = librpa_int::FT_Wc_freq_q(
        librpa_int::global::mpi_comm_global_h, wq_sym, pbc_sym, false,
        &qpoint_view, &ctx, &basis_abf, &ad_Wc);

    assert_dense_wq_rspace_maps_close(actual, expected);
}

void test_dense_wc_real_ct_matches_legacy_real_part()
{
    const auto pbc = make_wq_full_pbc();
    TFGrids tfg(2);
    tfg.generate_evenspaced_tf(0.1, 0.2, 0.3, 0.4);

    std::map<double, std::map<Vector3_Order<double>, Matz>> input;
    const auto freq_nodes = tfg.get_freq_nodes();
    for (std::size_t ifreq = 0; ifreq != freq_nodes.size(); ++ifreq)
    {
        for (std::size_t iq = 0; iq != pbc.klist.size(); ++iq)
        {
            Matz mat(2, 2, MAJOR::COL);
            for (std::size_t i = 0; i != mat.size(); ++i)
            {
                const double real = 1.0 + ifreq + 0.1 * i;
                const double imag = 0.2 + 0.05 * i;
                if (iq == 0)
                    mat.ptr()[i] = {real, 0.0};
                else if (iq == 1)
                    mat.ptr()[i] = {real, imag};
                else
                    mat.ptr()[i] = {real, -imag};
            }
            input[freq_nodes[ifreq]][pbc.klist[iq]] = std::move(mat);
        }
    }

    auto legacy_input = input;
    auto real_input = input;
    const auto legacy = librpa_int::CT_FT_Wc_freq_q(
        librpa_int::global::mpi_comm_global_h, legacy_input, pbc, tfg, true);
    const auto actual = librpa_int::CT_FT_Wc_freq_q_real(
        librpa_int::global::mpi_comm_global_h, real_input, pbc, tfg);
    assert(legacy_input.empty());
    assert(real_input.empty());

    for (const auto &[tau, expected_Rs] : legacy)
    {
        assert(actual.count(tau) != 0);
        for (const auto &[R, expected] : expected_Rs)
        {
            assert(actual.at(tau).count(R) != 0);
            const auto &result = actual.at(tau).at(R);
            assert(result.major() == expected.major());
            assert(result.size() == expected.size());
            for (std::size_t i = 0; i != result.size(); ++i)
                assert(std::abs(result.ptr()[i] - expected.ptr()[i].real()) < 1e-12);
        }
    }
}

void test_dense_wc_real_ct_rejects_large_imaginary_residual()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    pbc.set_kgrids_kvec(1, 1, 1, {0.0, 0.0, 0.0});
    TFGrids tfg(1);
    tfg.generate_evenspaced_tf(0.1, 0.1, 0.2, 0.1);

    std::map<double, std::map<Vector3_Order<double>, Matz>> input;
    Matz gamma(1, 1, MAJOR::COL);
    gamma(0, 0) = {1.0, 1e-4};
    input[tfg.get_freq_nodes().front()][pbc.klist.front()] = std::move(gamma);

    bool rejected = false;
    try
    {
        static_cast<void>(librpa_int::CT_FT_Wc_freq_q_real(
            librpa_int::global::mpi_comm_global_h, input, pbc, tfg));
    }
    catch (const std::runtime_error &error)
    {
        rejected = std::string(error.what()).find("imaginary residual") != std::string::npos;
    }
    assert(rejected);
}

}  // namespace

int main(int argc, char *argv[])
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    librpa_int::global::init_global_mpi(MPI_COMM_WORLD);
    librpa_int::global::init_global_io(false, "stdout", false);

    {
        BlacsCtxtHandler blacs_h(MPI_COMM_WORLD);
        blacs_h.init();
        blacs_h.set_square_grid();

        test_replace_rpa_response_headwing_replaces_only_singular_channels(blacs_h);
        test_rspace_symmetry_requires_complete_band_space();
        test_kpoint_coordinate_mapping_selects_active_klist_from_full_source();
        test_kstar_velocity_mapping_preserves_member_order_and_periodic_gauge();
        test_replace_rpa_response_head_only_keeps_numeric_wings(blacs_h);
        test_head_only_trace_logdet_can_use_reduced_response(blacs_h);
        test_rpa_trace_log_average_uses_directional_head_and_wing();
        test_rpa_headwing_regular_body_start_channel();
        test_rpa_headwing_gamma_cell_volume_uses_reciprocal_lattice();
        test_rpa_chi0v_wing_desc_uses_global_rows(blacs_h);
        test_headwing_spin_weights();
        test_wing_cartesian_gram_is_invariant_under_row_phases();
        test_velocity_matrix_initialization();
        test_headwing_local_kpoints_prefers_kpoint_blacs_context();
        test_headwing_world_fourier_uses_all_R_blocks_at_nonzero_k();
        test_headwing_symmetry_fourier_target_ids_are_deterministic();
        test_headwing_ijk_redistribution_is_owner_group_local();
        test_accumulate_wing_mu_for_pair_matches_original_formula();
        test_headwing_wfc_restore_applies_atom_permutation();
        test_headwing_wfc_restore_applies_time_reversal();
        test_headwing_velocity_restore_uses_inverse_spatial_route();
        test_headwing_direct_full_bz_velocity_selects_kstar_member();
        test_headwing_direct_full_bz_wfc_selects_same_kstar_member();
        test_headwing_direct_full_bz_wfc_local_block(blacs_h);
        test_kblacs_transform_with_restored_wfc_matches_full_bz_atom_permutation(blacs_h);
        test_kblacs_transform_matches_original_transform(blacs_h);
        test_kblacs_transform_uses_rectangular_opt_128_blocks();
        test_transform_Cs2mnk_can_keep_spin_channels_separate(blacs_h);
        test_head_initialization_does_not_require_coulomb_diagonalization(blacs_h);
        test_wq_to_wr_symmetry_reduced_q_matches_full_bz();
        test_wq_to_wr_symmetry_collective_handles_empty_local_rank();
        test_dense_wq_to_wr_symmetry_reduced_q_matches_full_bz(blacs_h);
        test_dense_wc_real_ct_matches_legacy_real_part();
        test_dense_wc_real_ct_rejects_large_imaginary_residual();
    }

    librpa_int::global::finalize_global_io();
    librpa_int::global::finalize_global_mpi();
    MPI_Finalize();
    return 0;
}
