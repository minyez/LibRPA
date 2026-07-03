#include "../core/symmetry_context.h"
#include "../core/pbc.h"
#include "../core/qpoint_view.h"
#include "../math/rsh.h"
#include "../utils/constants.h"

#include <array>
#include <cassert>
#include <complex>
#include <map>
#include <stdexcept>
#include <vector>

#include "testutils.h"
#include "../io/stl_io_helper.h"

namespace {

using namespace std;
using namespace librpa_int;

void assert_matrix_close(const ComplexMatrix& actual,
                         const ComplexMatrix& expected,
                         const double thres = 1e-12)
{
    assert(actual.nr == expected.nr);
    assert(actual.nc == expected.nc);
    for (int row = 0; row < actual.nr; ++row)
    {
        for (int col = 0; col < actual.nc; ++col)
        {
            assert(fequal(actual(row, col),
                          expected(row, col),
                          std::complex<double>(thres, 0.0)));
        }
    }
}

void test_kspace_shell_rotations_use_direct_rotation()
{
    const BasisConvention basis_convention{-1,
                                           0,
                                           LIBRPA_ANGULAR_ORDER_NATURAL,
                                           LIBRPA_RSH_COEFF_1_M,
                                           LIBRPA_RSH_COEFF_1_M};
    const Matrix3 cartesian_rotation(0.0, -1.0, 0.0,
                                     1.0, 0.0, 0.0,
                                     0.0, 0.0, 1.0);
    const Matrix3 skew_lattice(2.0, 0.0, 0.0,
                               0.5, 1.5, 0.0,
                               0.2, 0.3, 2.0);
    const Matrix3 fractional_rotation =
        skew_lattice * cartesian_rotation.Transpose() * skew_lattice.Inverse();
    const SpaceGroupSymOp op{fractional_rotation, {0.0, 0.0, 0.0}};

    const auto shell_rotations =
        build_symmetry_shell_rotations_from_direct_rotation(
            op, skew_lattice, 1, basis_convention);
    assert(shell_rotations.size() == 2);

    const auto expected_p_rotation =
        real_spherical_harmonic_rotation_matrix(cartesian_rotation,
                                                1,
                                                basis_convention.order,
                                                basis_convention.coeff_m_negative,
                                                basis_convention.coeff_m_positive);
    assert_matrix_close(shell_rotations.at(1), expected_p_rotation);

    const Vector3_Order<double> k_source{0.25, 0.0, 0.0};
    const Vector3_Order<double> k_target{0.25, 0.0, 0.0};
    const Vector3_Order<double> atom_from{0.0, 0.0, 0.0};
    const Vector3_Order<double> atom_to{0.0, 0.0, 0.0};
    const Vector3_Order<int> return_lattice{1, 0, 0};
    const auto phase = build_symmetry_kspace_phase(
        k_source, k_target, atom_from, atom_to, return_lattice, basis_convention);
    assert(fequal(phase, std::complex<double>(0.0, -1.0), std::complex<double>(1e-12, 0.0)));

    const auto phased_shell_rotations =
        build_symmetry_kspace_shell_rotations(op,
                                                    skew_lattice,
                                                    0,
                                                    basis_convention,
                                                    k_source,
                                                    k_target,
                                                    atom_from,
                                                    atom_to,
                                                    return_lattice);
    assert(phased_shell_rotations.at(0).nr == 1);
    assert(phased_shell_rotations.at(0).nc == 1);
    assert(fequal(phased_shell_rotations.at(0)(0, 0),
                  phase,
                  std::complex<double>(1e-12, 0.0)));
}

void test_kstar_member_return_lattice_preserves_input_fractional_representative()
{
    SymmetryContext ctx;
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    ctx.set_crystal_structure(pbc.latvec,
                              pbc.G,
                              {{0, 0}, {1, 1}},
                              {{0, {0.0, 0.0, 0.0}}, {1, {-0.25, -0.25, -0.25}}});
    ctx.basis_convention = {-1,
                            0,
                            LIBRPA_ANGULAR_ORDER_NATURAL,
                            LIBRPA_RSH_COEFF_1_M,
                            LIBRPA_RSH_COEFF_1_M};

    SymmetryOperation op;
    op.rotation = {0, 0, -1,
                   1, 0, -1,
                   0, 1, -1};
    op.translation = {0.0, 0.0, 0.0};
    op.use_row_convention = true;
    ctx.rspace_operations.push_back(op);

    SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = {0.0, 0.0, 0.0};
    star.members.resize(1);
    star.members[0].spatial_isym = 0;
    star.members[0].k_bz = {0.0, 0.0, 0.0};
    ctx.kstars.push_back(star);

    ctx.build_kstar_member_rotations(0);
    assert(ctx.kspace_return_lattice.at({1, 0}) == Vector3_Order<int>(0, 0, 1));
}

void test_species_basis_layout_keeps_shell_order()
{
    SpeciesBasisLayout layout;
    assert(!layout.is_shell_available());
    layout.label = "X";
    layout.set({1, 0});
    assert(layout.is_shell_available());
    assert(layout.n_ao == 4);
    assert(layout.shell_counts.at(1) == 1);
    assert(layout.shell_indices.at(1).front() == 0);
    assert(layout.shell_indices.at(0).front() == 1);

    ComplexMatrix p_rotation(3, 3);
    p_rotation.zero_out();
    p_rotation(0, 0) = {2.0, 0.0};
    p_rotation(1, 1) = {3.0, 0.0};
    p_rotation(2, 2) = {4.0, 0.0};
    ComplexMatrix s_rotation(1, 1);
    s_rotation.zero_out();
    s_rotation(0, 0) = {7.0, 0.0};

    const auto rotation =
        build_symmetry_rotation_matrix(layout, {{1, p_rotation}, {0, s_rotation}});

    ComplexMatrix expected(4, 4);
    expected.zero_out();
    expected(0, 0) = {2.0, 0.0};
    expected(1, 1) = {3.0, 0.0};
    expected(2, 2) = {4.0, 0.0};
    expected(3, 3) = {7.0, 0.0};
    assert_matrix_close(rotation, expected);
}

void test_species_layouts_match_basis_dimensions()
{
    const std::map<atom_t, int> atom_to_type{{0, 0}, {1, 0}};

    SpeciesBasisLayout full_layout;
    full_layout.label = "full";
    full_layout.set({1, 0});

    SpeciesBasisLayout shrink_layout;
    shrink_layout.label = "shrink";
    shrink_layout.set({0});

    assert(symmetry_species_layouts_match_atom_counts(
        {full_layout}, atom_to_type, {{0, 4}, {1, 4}}));
    assert(!symmetry_species_layouts_match_atom_counts(
        {full_layout}, atom_to_type, {{0, 1}, {1, 1}}));
    assert(symmetry_species_layouts_match_atom_counts(
        {shrink_layout}, atom_to_type, {{0, 1}, {1, 1}}));
}

void test_atomic_basis_builds_symmetry_species_layouts()
{
    AtomicBasis basis(std::vector<std::size_t>{1, 3});
    basis.label = "WFC";
    basis.set_l_shells({{0}, {1}});

    const auto layouts = basis.build_species_basis_layouts({{0, 0}, {1, 1}});

    assert(get_symmetry_species_layout(layouts, 0).n_ao == 1);
    assert(get_symmetry_species_layout(layouts, 1).n_ao == 3);
}

void test_periodic_mappings_store_full_kpoint_members()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({2.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    const std::vector<double> kvecs_ibz{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 6.0, 0.0, 0.0,
    };
    const std::vector<std::vector<Vector3_Order<double>>> full_kstars{
        {{0.0, 0.0, 0.0}},
        {{1.0 / 6.0, 0.0, 0.0}, {-1.0 / 6.0, 0.0, 0.0}},
    };
    pbc.set_irreducible_kgrids_kvec(3, 1, 1, kvecs_ibz, full_kstars);

    SymmetryContext ctx;
    ctx.set_crystal_structure(pbc.latvec,
                              pbc.G,
                              {{0, 0}},
                              {{0, {0.0, 0.0, 0.0}}});
    ctx.set_rspace_operations({SpaceGroupSymOp::IDENTITY});
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);

    assert(ctx.kstars.size() == 2);
    assert(ctx.kstar_grid_mapping.size() == 2);
    assert(ctx.full_kpoint_members.size() == 3);
    assert(ctx.full_kpoint_members[0].ik_ibz == 0);
    assert(ctx.full_kpoint_members[1].ik_ibz == 1);
    assert(ctx.full_kpoint_members[2].ik_ibz == 1);
    std::size_t restored_rspace_members = 0;
    for (const auto& pair_stars : ctx.rspace_sector_stars)
    {
        for (const auto& R_star : pair_stars.second)
        {
            restored_rspace_members += R_star.second.size();
        }
    }
    assert(restored_rspace_members == pbc.Rlist.size());
}

void test_periodic_mappings_accept_kq_reduced_coulomb_grid()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    const std::vector<double> kvecs{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 3.0, 0.0, 0.0,
        2.0 * librpa_int::TWO_PI / 3.0, 0.0, 0.0,
    };
    pbc.set_kgrids_kvec(3, 1, 1, kvecs);
    pbc.set_kq_mapping({0, 1, 1});

    SymmetryContext ctx;
    ctx.set_crystal_structure(pbc.latvec,
                              pbc.G,
                              {{0, 0}},
                              {{0, {0.0, 0.0, 0.0}}});
    ctx.set_rspace_operations({SpaceGroupSymOp::IDENTITY});
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);

    assert(ctx.kstars.size() == 2);
    assert(ctx.kstar_grid_mapping.size() == 3);
    assert(ctx.kstar_grid_mapping[0].iq_ibz == 0);
    assert(ctx.kstar_grid_mapping[1].iq_ibz == 1);
    assert(ctx.kstar_grid_mapping[2].iq_ibz == 2);
    assert(ctx.kstar_grid_mapping[0].star_list_index == 0);
    assert(ctx.kstar_grid_mapping[1].star_list_index == 1);
    assert(ctx.kstar_grid_mapping[2].star_list_index == 1);
    assert(ctx.kstar_grid_mapping[0].member_q_bz_keys.size() == 1);
    assert(ctx.kstar_grid_mapping[1].member_q_bz_keys.size() == 2);
    assert(ctx.kstar_grid_mapping[2].member_q_bz_keys.size() == 2);
}

void test_qpoint_view_uses_pbc_without_symmetry()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    const std::vector<double> kvecs{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 3.0, 0.0, 0.0,
        2.0 * librpa_int::TWO_PI / 3.0, 0.0, 0.0,
    };
    pbc.set_kgrids_kvec(3, 1, 1, kvecs);

    SymmetryContext ctx;
    const auto view = build_symmetry_qpoint_view(ctx, pbc, false);
    assert(view.restore_mode == SymmetryQPointRestoreMode::NONE);
    assert(view.representatives.size() == 3);
    assert(std::abs(view.weights.at(pbc.klist_coul[0]) - 1.0 / 3.0) < 1e-12);
}

void test_qpoint_view_preserves_time_reversal_mapping_without_crystal_symmetry()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    const std::vector<double> kvecs{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 3.0, 0.0, 0.0,
        2.0 * librpa_int::TWO_PI / 3.0, 0.0, 0.0,
    };
    pbc.set_kgrids_kvec(3, 1, 1, kvecs);
    pbc.set_kq_mapping({0, 1, 1});

    SymmetryContext ctx;
    const auto view = build_symmetry_qpoint_view(ctx, pbc, false);
    assert(view.restore_mode == SymmetryQPointRestoreMode::TIME_REVERSAL);
    assert(view.representatives.size() == 2);
    assert(view.members.at(pbc.klist_coul[1]).size() == 2);
    assert(std::abs(view.weights.at(pbc.klist_coul[1]) - 2.0 / 3.0) < 1e-12);
}

void test_qpoint_view_preserves_time_reversal_q_reduction_with_crystal_symmetry()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    const std::vector<double> kvecs{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 3.0, 0.0, 0.0,
        2.0 * librpa_int::TWO_PI / 3.0, 0.0, 0.0,
    };
    pbc.set_kgrids_kvec(3, 1, 1, kvecs);
    pbc.set_kq_mapping({0, 1, 1});

    SymmetryContext ctx;
    ctx.set_crystal_structure(pbc.latvec,
                              pbc.G,
                              {{0, 0}},
                              {{0, {0.0, 0.0, 0.0}}});
    ctx.set_rspace_operations({SpaceGroupSymOp::IDENTITY});
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);

    const auto view = build_symmetry_qpoint_view(ctx, pbc, true);
    assert(view.restore_mode == SymmetryQPointRestoreMode::TIME_REVERSAL);
    assert(view.representatives == pbc.klist_coul);
    assert(view.members.at(pbc.klist_coul[1]).size() == 2);
    assert(std::abs(view.weights.at(pbc.klist_coul[1]) - 2.0 / 3.0) < 1e-12);
}

void test_qpoint_view_rejects_reduced_input_without_symmetry()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({2.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    const std::vector<double> kvecs_ibz{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 6.0, 0.0, 0.0,
    };
    const std::vector<std::vector<Vector3_Order<double>>> full_kstars{
        {{0.0, 0.0, 0.0}},
        {{1.0 / 6.0, 0.0, 0.0}, {-1.0 / 6.0, 0.0, 0.0}},
    };
    pbc.set_irreducible_kgrids_kvec(3, 1, 1, kvecs_ibz, full_kstars);

    bool threw = false;
    try
    {
        SymmetryContext ctx;
        (void)build_symmetry_qpoint_view(ctx, pbc, false);
    }
    catch (const std::runtime_error&)
    {
        threw = true;
    }
    assert(threw);
}

void test_qpoint_view_accepts_reduced_input_with_symmetry()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({2.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    const std::vector<double> kvecs_ibz{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 6.0, 0.0, 0.0,
    };
    const std::vector<std::vector<Vector3_Order<double>>> full_kstars{
        {{0.0, 0.0, 0.0}},
        {{1.0 / 6.0, 0.0, 0.0}, {-1.0 / 6.0, 0.0, 0.0}},
    };
    pbc.set_irreducible_kgrids_kvec(3, 1, 1, kvecs_ibz, full_kstars);

    SymmetryContext ctx;
    ctx.set_crystal_structure(pbc.latvec,
                              pbc.G,
                              {{0, 0}},
                              {{0, {0.0, 0.0, 0.0}}});
    ctx.set_rspace_operations({SpaceGroupSymOp::IDENTITY});
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);

    const auto view = build_symmetry_qpoint_view(ctx, pbc, true);
    assert(view.restore_mode == SymmetryQPointRestoreMode::FULL_CRYSTAL);
    assert(view.representatives == pbc.klist_coul);
    assert(view.members.at(pbc.klist_coul[1]).size() == 2);
    assert(std::abs(view.weights.at(pbc.klist_coul[1]) - 2.0 / 3.0) < 1e-12);
}

void test_qpoint_view_keeps_full_input_grid_without_parsed_crystal_symmetry()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    const std::vector<double> kvecs{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 3.0, 0.0, 0.0,
        2.0 * librpa_int::TWO_PI / 3.0, 0.0, 0.0,
    };
    pbc.set_kgrids_kvec(3, 1, 1, kvecs);

    SymmetryContext ctx;
    ctx.set_crystal_structure(pbc.latvec,
                              pbc.G,
                              {{0, 0}},
                              {{0, {0.0, 0.0, 0.0}}});
    ctx.set_rspace_operations({SpaceGroupSymOp::IDENTITY});
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);

    const auto view = build_symmetry_qpoint_view(ctx, pbc, true);
    assert(view.restore_mode == SymmetryQPointRestoreMode::NONE);
    assert(view.representatives == pbc.klist_coul);
    assert(view.representatives.size() == 3);
    assert(std::abs(view.weights.at(pbc.klist_coul[1]) - 1.0 / 3.0) < 1e-12);
}

void add_irreducible_sector_entry(symmetry_irreducible_sector_t& sector,
                                  const atom_t atom_i,
                                  const atom_t atom_j,
                                  const symmetry_R_t& R)
{
    sector[{atom_i, atom_j}].insert(R);
}

SymmetryOperation make_row_symmetry_operation(const std::array<int, 9>& rotation)
{
    SymmetryOperation op;
    op.rotation = Matrix3(rotation[0], rotation[1], rotation[2],
                                      rotation[3], rotation[4], rotation[5],
                                      rotation[6], rotation[7], rotation[8]);
    op.translation = {0.0, 0.0, 0.0};
    op.use_row_convention = true;
    return op;
}

void test_symmetry_context_saves_fractional_row_operations()
{
    SymmetryContext ctx;
    const Matrix3 col_rotation(0.0, -1.0, 0.0,
                               1.0, 0.0, 0.0,
                               0.0, 0.0, 1.0);
    const Vector3_Order<double> translation{0.25, -0.5, 1.0 / 3.0};
    SymmetryOperation op;
    op.rotation = col_rotation;
    op.translation = translation;
    op.use_row_convention = false;

    const SpaceGroupSymOp original_op{col_rotation, translation, false};
    const Vector3_Order<double> coord{0.2, 0.3, 0.4};
    const auto expected = apply_space_group_symmetry_operation(original_op, coord);

    ctx.add_rspace_operation(op);
    assert(!ctx.available);
    ctx.set_available();
    assert(ctx.available);
    assert(ctx.rspace_operations.size() == 1);

    const auto& saved = ctx.rspace_operations[0];
    assert(saved.use_row_convention);
    assert(saved.rotation == col_rotation.Transpose());
    assert(fequal(saved.translation.x, translation.x));
    assert(fequal(saved.translation.y, translation.y));
    assert(fequal(saved.translation.z, translation.z));

    const auto actual = apply_space_group_symmetry_operation(saved, coord);
    assert(fequal(actual.x, expected.x));
    assert(fequal(actual.y, expected.y));
    assert(fequal(actual.z, expected.z));
}

SymmetryContext make_bn_shrink_symmetry_context()
{
    SymmetryContext ctx;
    ctx.atom_to_type = {{0, 0}, {1, 1}};
    ctx.input_coord_frac = {{0, {0.0, 0.0, 0.0}}, {1, {0.25, 0.25, 0.25}}};

    const std::vector<std::array<int, 9>> rotations = {
        { 1,  0,  0,  0,  1,  0,  0,  0,  1},
        { 0, -1,  0,  1, -1,  0,  0, -1,  1},
        {-1,  1,  0, -1,  0,  0, -1,  0,  1},
        {-1,  0,  1, -1,  0,  0, -1,  1,  0},
        { 1,  0,  0,  0,  0,  1,  0,  1,  0},
        { 0,  0, -1,  1,  0, -1,  0,  1, -1},
        { 0,  1, -1,  1,  0, -1,  0,  0, -1},
        {-1,  0,  1, -1,  1,  0, -1,  0,  0},
        { 1, -1,  0,  0, -1,  1,  0, -1,  0},
        { 0, -1,  0,  0, -1,  1,  1, -1,  0},
        { 0,  1, -1,  0,  0, -1,  1,  0, -1},
        { 0,  0,  1,  0,  1,  0,  1,  0,  0},
        { 1,  0, -1,  0,  0, -1,  0,  1, -1},
        { 0,  0,  1,  1,  0,  0,  0,  1,  0},
        {-1,  0,  0, -1,  0,  1, -1,  1,  0},
        {-1,  1,  0, -1,  0,  1, -1,  0,  0},
        { 1,  0, -1,  0,  1, -1,  0,  0, -1},
        { 0, -1,  1,  1, -1,  0,  0, -1,  0},
        { 0,  0, -1,  0,  1, -1,  1,  0, -1},
        { 0, -1,  1,  0, -1,  0,  1, -1,  0},
        { 0,  1,  0,  0,  0,  1,  1,  0,  0},
        {-1,  0,  0, -1,  1,  0, -1,  0,  1},
        { 0,  1,  0,  1,  0,  0,  0,  0,  1},
        { 1, -1,  0,  0, -1,  0,  0, -1,  1},
    };
    for (const auto& rotation : rotations)
    {
        ctx.rspace_operations.push_back(make_row_symmetry_operation(rotation));
    }
    return ctx;
}

void add_mgo_fractional_symmetry_operations(SymmetryContext& ctx)
{
    const std::array<std::array<int, 3>, 6> permutations{{
        {{0, 1, 2}},
        {{0, 2, 1}},
        {{1, 0, 2}},
        {{1, 2, 0}},
        {{2, 0, 1}},
        {{2, 1, 0}},
    }};
    const std::array<int, 2> signs{{-1, 1}};

    for (const auto& permutation : permutations)
    {
        for (const int sx : signs)
        {
            for (const int sy : signs)
            {
                for (const int sz : signs)
                {
                    const std::array<int, 3> sign{{sx, sy, sz}};
                    std::array<int, 9> rotation{};
                    for (int col = 0; col != 3; ++col)
                    {
                        rotation[3 * permutation[col] + col] = sign[col];
                    }

                    const Matrix3 col_cartesian_rotation(
                        rotation[0], rotation[1], rotation[2],
                        rotation[3], rotation[4], rotation[5],
                        rotation[6], rotation[7], rotation[8]);
                    const Matrix3 row_cartesian_rotation = col_cartesian_rotation.Transpose();
                    SymmetryOperation op;
                    op.rotation = ctx.lattice_available
                                      ? ctx.lattice_vectors * row_cartesian_rotation *
                                            ctx.lattice_vectors.Inverse()
                                      : row_cartesian_rotation;
                    op.translation = {0.0, 0.0, 0.0};
                    op.use_row_convention = true;
                    ctx.rspace_operations.push_back(op);
                }
            }
        }
    }
    assert(ctx.rspace_operations.size() == 48);
}

void set_mgo_primitive_structure(SymmetryContext& ctx,
                                 const std::map<atom_t, int>& atom_to_type,
                                 const std::map<atom_t, coord_t>& coord_frac)
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({0.0, 0.5, 0.5,
                    0.5, 0.0, 0.5,
                    0.5, 0.5, 0.0});
    ctx.set_crystal_structure(pbc.latvec, pbc.G, atom_to_type, coord_frac);
}

void test_single_atom_qpoint_view_reduces_full_input_to_crystal_qstars()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({0.0, 0.5, 0.5,
                    0.5, 0.0, 0.5,
                    0.5, 0.5, 0.0});

    std::vector<double> kvecs;
    for (const auto& kfrac : build_uniform_kmesh_frac({3, 3, 3}))
    {
        const auto kvec = kfrac * pbc.G;
        kvecs.push_back(kvec.x * TWO_PI);
        kvecs.push_back(kvec.y * TWO_PI);
        kvecs.push_back(kvec.z * TWO_PI);
    }
    pbc.set_kgrids_kvec(3, 3, 3, kvecs);

    SymmetryContext ctx;
    set_mgo_primitive_structure(ctx,
                                {{0, 0}},
                                {{0, {0.0, 0.0, 0.0}}});
    add_mgo_fractional_symmetry_operations(ctx);
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);

    const auto view = build_symmetry_qpoint_view(ctx, pbc, true);
    assert(view.restore_mode == SymmetryQPointRestoreMode::FULL_CRYSTAL);
    assert(view.representatives.size() == ctx.kstars.size());
    assert(view.representatives.size() == 4);
    std::size_t n_members = 0;
    double weight_sum = 0.0;
    for (const auto& q : view.representatives)
    {
        n_members += view.members.at(q).size();
        weight_sum += view.weights.at(q);
    }
    assert(n_members == pbc.klist_full.size());
    assert(std::abs(weight_sum - 1.0) < 1e-12);
}

void test_mgo_qpoint_view_reduces_time_reversal_q_list_to_crystal_qstars_for_full_input()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({0.0, 0.5, 0.5,
                    0.5, 0.0, 0.5,
                    0.5, 0.5, 0.0});

    std::vector<double> kvecs;
    for (const auto& kfrac : build_uniform_kmesh_frac({3, 3, 3}))
    {
        const auto kvec = kfrac * pbc.G;
        kvecs.push_back(kvec.x * TWO_PI);
        kvecs.push_back(kvec.y * TWO_PI);
        kvecs.push_back(kvec.z * TWO_PI);
    }
    pbc.set_kgrids_kvec(3, 3, 3, kvecs);

    std::vector<int> map_q_ks(pbc.kfrac_list.size(), -1);
    for (std::size_t ik = 0; ik != pbc.kfrac_list.size(); ++ik)
    {
        const auto minus_k =
            restrict_fractional_coordinate(Vector3_Order<double>{-pbc.kfrac_list[ik].x,
                                                                  -pbc.kfrac_list[ik].y,
                                                                  -pbc.kfrac_list[ik].z});
        int minus_ik = -1;
        for (std::size_t jk = 0; jk != pbc.kfrac_list.size(); ++jk)
        {
            if (same_fractional_kpoint(pbc.kfrac_list[jk], minus_k, 1e-5))
            {
                minus_ik = static_cast<int>(jk);
                break;
            }
        }
        assert(minus_ik >= 0);
        map_q_ks[ik] = std::min(static_cast<int>(ik), minus_ik);
    }
    pbc.set_kq_mapping(map_q_ks);
    assert(pbc.klist_coul.size() == 14);

    SymmetryContext ctx;
    set_mgo_primitive_structure(ctx,
                                {{0, 0}, {1, 1}},
                                {{0, {0.0, 0.0, 0.0}}, {1, {0.5, 0.5, 0.5}}});
    add_mgo_fractional_symmetry_operations(ctx);
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);

    const auto view = build_symmetry_qpoint_view(ctx, pbc, true);
    assert(view.restore_mode == SymmetryQPointRestoreMode::FULL_CRYSTAL);
    assert(view.representatives.size() == ctx.kstars.size());
    assert(view.representatives.size() == 4);
    std::size_t n_members = 0;
    double weight_sum = 0.0;
    for (const auto& q : view.representatives)
    {
        n_members += view.members.at(q).size();
        weight_sum += view.weights.at(q);
    }
    assert(n_members == pbc.klist_full.size());
    assert(std::abs(weight_sum - 1.0) < 1e-12);
}

void test_mgo_keeps_all_kspace_operations_for_full_qstars()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({0.0, 0.5, 0.5,
                    0.5, 0.0, 0.5,
                    0.5, 0.5, 0.0});

    std::vector<double> kvecs;
    for (const auto& kfrac : build_uniform_kmesh_frac({3, 3, 3}))
    {
        const auto kvec = kfrac * pbc.G;
        kvecs.push_back(kvec.x * TWO_PI);
        kvecs.push_back(kvec.y * TWO_PI);
        kvecs.push_back(kvec.z * TWO_PI);
    }
    pbc.set_kgrids_kvec(3, 3, 3, kvecs);

    SymmetryContext ctx;
    set_mgo_primitive_structure(ctx,
                                {{0, 0}, {1, 1}},
                                {{0, {0.0, 0.0, 0.0}}, {1, {0.5, 0.5, 0.5}}});

    SpaceGroupSymOps signed_permutation_operations;
    const std::array<std::array<int, 3>, 6> permutations{{
        {{0, 1, 2}},
        {{0, 2, 1}},
        {{1, 0, 2}},
        {{1, 2, 0}},
        {{2, 0, 1}},
        {{2, 1, 0}},
    }};
    const std::array<int, 2> signs{{-1, 1}};
    for (const auto& permutation : permutations)
    {
        for (const int sx : signs)
        {
            for (const int sy : signs)
            {
                for (const int sz : signs)
                {
                    const std::array<int, 3> sign{{sx, sy, sz}};
                    std::array<int, 9> rotation{};
                    for (int col = 0; col != 3; ++col)
                    {
                        rotation[3 * permutation[col] + col] = sign[col];
                    }
                    signed_permutation_operations.push_back(
                        make_row_symmetry_operation(rotation));
                }
            }
        }
    }

    ctx.set_rspace_operations(signed_permutation_operations);
    assert(ctx.rspace_operations.size() == 48);
    int metric_preserving_ops = 0;
    for (const auto& op : ctx.rspace_operations)
    {
        if (preserves_lattice_metric(op.rotation, ctx.lattice_vectors, 1e-6))
        {
            ++metric_preserving_ops;
        }
    }
    assert(metric_preserving_ops == 48);

    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);
    assert(ctx.kstars.size() == 4);
    const auto view = build_symmetry_qpoint_view(ctx, pbc, true);
    assert(view.restore_mode == SymmetryQPointRestoreMode::FULL_CRYSTAL);
    assert(view.representatives.size() == ctx.kstars.size());
}

void test_kstar_routes_prefer_improper_spatial_operation_over_time_reversal()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});

    std::vector<double> kvecs;
    for (const auto& kfrac : build_uniform_kmesh_frac({3, 3, 3}))
    {
        const auto kvec = kfrac * pbc.G;
        kvecs.push_back(kvec.x * TWO_PI);
        kvecs.push_back(kvec.y * TWO_PI);
        kvecs.push_back(kvec.z * TWO_PI);
    }
    pbc.set_kgrids_kvec(3, 3, 3, kvecs);

    SymmetryContext ctx;
    ctx.set_crystal_structure(pbc.latvec,
                              pbc.G,
                              {{0, 0}},
                              {{0, {0.0, 0.0, 0.0}}});
    ctx.set_rspace_operations({SpaceGroupSymOp::IDENTITY, SpaceGroupSymOp::INVERSE});
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);

    bool checked_minus_q_member = false;
    for (const auto& star : ctx.kstars)
    {
        if (same_fractional_kpoint(star.k_ibz, {0.0, 0.0, 0.0}, 1e-5))
        {
            continue;
        }
        const auto minus_rep =
            restrict_fractional_coordinate(Vector3_Order<double>{-star.k_ibz.x,
                                                                  -star.k_ibz.y,
                                                                  -star.k_ibz.z});
        for (const auto& member : star.members)
        {
            if (!same_fractional_kpoint(member.k_bz, minus_rep, 1e-5)
                || same_fractional_kpoint(member.k_bz, star.k_ibz, 1e-5))
            {
                continue;
            }
            assert(member.spatial_isym == 1);
            assert(!member.time_reversal);
            checked_minus_q_member = true;
        }
    }
    assert(checked_minus_q_member);
}

ComplexMatrix make_test_dense_operator(const int n)
{
    ComplexMatrix matrix(n, n);
    for (int row = 0; row != n; ++row)
    {
        for (int col = 0; col != n; ++col)
        {
            matrix(row, col) =
                std::complex<double>{0.1 * (row + 1) + 0.03 * (col + 1),
                                     0.07 * (row - col)};
        }
    }
    return matrix;
}

librpa_int::symmetry_atom_block_matrix_map_t dense_to_atom_blocks(
    const ComplexMatrix& matrix,
    const std::map<atom_t, size_t>& atom_nabf)
{
    librpa_int::symmetry_atom_block_matrix_map_t blocks;
    std::map<atom_t, int> offsets;
    int offset = 0;
    for (const auto& [atom, nbf] : atom_nabf)
    {
        offsets[atom] = offset;
        offset += static_cast<int>(nbf);
    }
    for (const auto& [atom_i, n_i_size] : atom_nabf)
    {
        const int n_i = static_cast<int>(n_i_size);
        for (const auto& [atom_j, n_j_size] : atom_nabf)
        {
            const int n_j = static_cast<int>(n_j_size);
            ComplexMatrix block(n_i, n_j);
            for (int i = 0; i != n_i; ++i)
            {
                for (int j = 0; j != n_j; ++j)
                {
                    block(i, j) = matrix(offsets.at(atom_i) + i,
                                         offsets.at(atom_j) + j);
                }
            }
            blocks[atom_i][atom_j] = block;
        }
    }
    return blocks;
}

ComplexMatrix atom_blocks_to_dense(
    const librpa_int::symmetry_atom_block_matrix_map_t& blocks,
    const std::map<atom_t, size_t>& atom_nabf)
{
    std::map<atom_t, int> offsets;
    int total = 0;
    for (const auto& [atom, nbf] : atom_nabf)
    {
        offsets[atom] = total;
        total += static_cast<int>(nbf);
    }
    ComplexMatrix matrix(total, total);
    for (const auto& [atom_i, row_blocks] : blocks)
    {
        for (const auto& [atom_j, block] : row_blocks)
        {
            const int row_offset = offsets.at(atom_i);
            const int col_offset = offsets.at(atom_j);
            for (int i = 0; i != block.nr; ++i)
            {
                for (int j = 0; j != block.nc; ++j)
                {
                    matrix(row_offset + i, col_offset + j) = block(i, j);
                }
            }
        }
    }
    return matrix;
}

ComplexMatrix rotate_dense_operator_from_rotation_matrix(
    const ComplexMatrix& matrix,
    const ComplexMatrix& rotation,
    const bool use_time_reversal)
{
    if (use_time_reversal)
    {
        return transpose(rotation, true) * conj(matrix) * rotation;
    }
    return transpose(rotation, false) * matrix * conj(rotation);
}

void test_mgo_dense_kspace_rotation_matches_atom_block_rotation()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({0.0, 0.5, 0.5,
                    0.5, 0.0, 0.5,
                    0.5, 0.5, 0.0});

    std::vector<double> kvecs;
    for (const auto& kfrac : build_uniform_kmesh_frac({3, 3, 3}))
    {
        const auto kvec = kfrac * pbc.G;
        kvecs.push_back(kvec.x * TWO_PI);
        kvecs.push_back(kvec.y * TWO_PI);
        kvecs.push_back(kvec.z * TWO_PI);
    }
    pbc.set_kgrids_kvec(3, 3, 3, kvecs);

    SymmetryContext ctx;
    set_mgo_primitive_structure(ctx,
                                {{0, 0}, {1, 1}},
                                {{0, {0.0, 0.0, 0.0}}, {1, {0.5, 0.5, 0.5}}});
    add_mgo_fractional_symmetry_operations(ctx);
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);
    ctx.build_rsh_rotations({-1,
                             0,
                             LIBRPA_ANGULAR_ORDER_NATURAL,
                             LIBRPA_RSH_COEFF_1_M,
                             LIBRPA_RSH_COEFF_1_M},
                            1);
    ctx.build_kstar_member_rotations(1);

    AtomicBasis basis(std::vector<std::size_t>{4, 4});
    basis.set_l_shells({{0, 1}, {0, 1}});
    const auto layouts = basis.build_species_basis_layouts(ctx.atom_to_type);
    const std::map<atom_t, size_t> atom_nabf{{0, 4}, {1, 4}};
    const auto matrix = make_test_dense_operator(8);
    const auto blocks = dense_to_atom_blocks(matrix, atom_nabf);

    for (const auto& star : ctx.kstars)
    {
        for (const auto& member : star.members)
        {
            const auto rotation = build_symmetry_kspace_rotation_matrix(
                ctx, layouts, member, atom_nabf, star.k_ibz, member.time_reversal,
                &member.k_bz);
            const auto rotated_dense = rotate_dense_operator_from_rotation_matrix(
                matrix, rotation, member.time_reversal);
            const auto rotated_blocks = rotate_symmetry_kspace_operator_blocks(
                ctx, layouts, member, blocks, atom_nabf, star.k_ibz,
                member.time_reversal, nullptr, &member.k_bz);
            assert_matrix_close(rotated_dense,
                                atom_blocks_to_dense(rotated_blocks, atom_nabf),
                                1e-10);
        }
    }
}

void test_mgo_k333_irreducible_sector_matches_single()
{
    SymmetryContext ctx;
    set_mgo_primitive_structure(ctx,
                                {{0, 0}},
                                {{0, {0.0, 0.0, 0.0}}});
    add_mgo_fractional_symmetry_operations(ctx);

    const Vector3_Order<int> period{3, 3, 3};
    const auto Rlist = construct_R_grid(period);
    const auto generated_sector =
        build_symmetry_rspace_irreducible_sector(ctx, Rlist);

    symmetry_irreducible_sector_t expected_sector;
    add_irreducible_sector_entry(expected_sector, 0, 0, {-1, -1, -1});
    add_irreducible_sector_entry(expected_sector, 0, 0, {-1, -1,  0});
    add_irreducible_sector_entry(expected_sector, 0, 0, {-1, -1,  1});
    add_irreducible_sector_entry(expected_sector, 0, 0, {-1,  0,  0});
    add_irreducible_sector_entry(expected_sector, 0, 0, { 0,  0,  0});
    assert(generated_sector == expected_sector);

    ctx.irreducible_sector = generated_sector;
    ctx.set_available();
    symmetry_rspace_sector_stars_t sector_stars;
    build_symmetry_rspace_sector_stars(ctx, period, Rlist, sector_stars);
    std::size_t restored_members = 0;
    for (const auto& pair_stars : sector_stars)
    {
        for (const auto& R_star : pair_stars.second)
        {
            restored_members += R_star.second.size();
        }
    }
    assert(restored_members == Rlist.size());
}

void test_mgo_k333_irreducible_sector_matches_both()
{
    SymmetryContext ctx;
    set_mgo_primitive_structure(ctx,
                                {{0, 0}, {1, 1}},
                                {{0, {0.0, 0.0, 0.0}}, {1, {0.5, 0.5, 0.5}}});
    add_mgo_fractional_symmetry_operations(ctx);

    const Vector3_Order<int> period{3, 3, 3};
    const auto Rlist = construct_R_grid(period);
    const auto generated_sector =
        build_symmetry_rspace_irreducible_sector(ctx, Rlist);

    symmetry_irreducible_sector_t expected_sector;
    add_irreducible_sector_entry(expected_sector, 0, 0, {-1, -1, -1});
    add_irreducible_sector_entry(expected_sector, 0, 0, {-1, -1,  0});
    add_irreducible_sector_entry(expected_sector, 0, 0, {-1, -1,  1});
    add_irreducible_sector_entry(expected_sector, 0, 0, {-1,  0,  0});
    add_irreducible_sector_entry(expected_sector, 0, 0, { 0,  0,  0});
    add_irreducible_sector_entry(expected_sector, 0, 1, {-1,  0,  0});
    add_irreducible_sector_entry(expected_sector, 0, 1, {-1,  0,  1});
    add_irreducible_sector_entry(expected_sector, 0, 1, {-1,  1,  1});
    add_irreducible_sector_entry(expected_sector, 0, 1, { 0,  0,  0});
    add_irreducible_sector_entry(expected_sector, 0, 1, { 0,  0,  1});
    add_irreducible_sector_entry(expected_sector, 0, 1, { 0,  1,  1});
    add_irreducible_sector_entry(expected_sector, 0, 1, { 1,  1,  1});
    add_irreducible_sector_entry(expected_sector, 1, 0, {-1, -1, -1});
    add_irreducible_sector_entry(expected_sector, 1, 0, {-1, -1,  0});
    add_irreducible_sector_entry(expected_sector, 1, 0, {-1, -1,  1});
    add_irreducible_sector_entry(expected_sector, 1, 0, {-1,  0,  0});
    add_irreducible_sector_entry(expected_sector, 1, 0, {-1,  0,  1});
    add_irreducible_sector_entry(expected_sector, 1, 0, { 0,  0,  0});
    add_irreducible_sector_entry(expected_sector, 1, 0, { 0,  0,  1});
    add_irreducible_sector_entry(expected_sector, 1, 1, {-1, -1, -1});
    add_irreducible_sector_entry(expected_sector, 1, 1, {-1, -1,  0});
    add_irreducible_sector_entry(expected_sector, 1, 1, {-1, -1,  1});
    add_irreducible_sector_entry(expected_sector, 1, 1, {-1,  0,  0});
    add_irreducible_sector_entry(expected_sector, 1, 1, { 0,  0,  0});
    assert(generated_sector == expected_sector);

    ctx.irreducible_sector = generated_sector;
    ctx.set_available();
    symmetry_rspace_sector_stars_t sector_stars;
    build_symmetry_rspace_sector_stars(ctx, period, Rlist, sector_stars);
    std::size_t restored_members = 0;
    for (const auto& pair_stars : sector_stars)
    {
        for (const auto& R_star : pair_stars.second)
        {
            restored_members += R_star.second.size();
        }
    }
    assert(restored_members == 2 * 2 * Rlist.size());
}

void test_bn_shrink_irreducible_sector_can_be_generated_from_symmetry()
{
    auto ctx = make_bn_shrink_symmetry_context();
    const Vector3_Order<int> period{2, 2, 2};
    const auto Rlist = construct_R_grid(period);
    const auto generated_sector =
        build_symmetry_rspace_irreducible_sector(ctx, Rlist);

    symmetry_irreducible_sector_t expected_sector;
    add_irreducible_sector_entry(expected_sector, 0, 0, {-1, -1, -1});
    add_irreducible_sector_entry(expected_sector, 0, 0, {-1, -1,  0});
    add_irreducible_sector_entry(expected_sector, 0, 0, {-1,  0,  0});
    add_irreducible_sector_entry(expected_sector, 0, 0, { 0,  0,  0});
    add_irreducible_sector_entry(expected_sector, 0, 1, {-1, -1, -1});
    add_irreducible_sector_entry(expected_sector, 0, 1, {-1, -1,  0});
    add_irreducible_sector_entry(expected_sector, 0, 1, { 0,  0,  0});
    add_irreducible_sector_entry(expected_sector, 1, 0, {-1, -1, -1});
    add_irreducible_sector_entry(expected_sector, 1, 0, {-1, -1,  0});
    add_irreducible_sector_entry(expected_sector, 1, 0, {-1,  0,  0});
    add_irreducible_sector_entry(expected_sector, 1, 0, { 0,  0,  0});
    add_irreducible_sector_entry(expected_sector, 1, 1, {-1, -1, -1});
    add_irreducible_sector_entry(expected_sector, 1, 1, {-1, -1,  0});
    add_irreducible_sector_entry(expected_sector, 1, 1, {-1,  0,  0});
    add_irreducible_sector_entry(expected_sector, 1, 1, { 0,  0,  0});
    assert(generated_sector == expected_sector);

    ctx.irreducible_sector = generated_sector;
    ctx.set_available();
    symmetry_rspace_sector_stars_t sector_stars;
    build_symmetry_rspace_sector_stars(ctx, period, Rlist, sector_stars);
    std::size_t restored_members = 0;
    for (const auto& pair_stars : sector_stars)
    {
        for (const auto& R_star : pair_stars.second)
        {
            restored_members += R_star.second.size();
        }
    }
    assert(restored_members == 2 * 2 * Rlist.size());
}

} // namespace

int main()
{
    test_symmetry_context_saves_fractional_row_operations();
    test_kspace_shell_rotations_use_direct_rotation();
    test_kstar_member_return_lattice_preserves_input_fractional_representative();
    test_species_basis_layout_keeps_shell_order();
    test_species_layouts_match_basis_dimensions();
    test_atomic_basis_builds_symmetry_species_layouts();
    test_periodic_mappings_store_full_kpoint_members();
    test_periodic_mappings_accept_kq_reduced_coulomb_grid();
    test_qpoint_view_uses_pbc_without_symmetry();
    test_qpoint_view_preserves_time_reversal_mapping_without_crystal_symmetry();
    test_qpoint_view_preserves_time_reversal_q_reduction_with_crystal_symmetry();
    test_qpoint_view_rejects_reduced_input_without_symmetry();
    test_qpoint_view_accepts_reduced_input_with_symmetry();
    test_qpoint_view_keeps_full_input_grid_without_parsed_crystal_symmetry();
    test_single_atom_qpoint_view_reduces_full_input_to_crystal_qstars();
    test_mgo_qpoint_view_reduces_time_reversal_q_list_to_crystal_qstars_for_full_input();
    test_mgo_keeps_all_kspace_operations_for_full_qstars();
    test_kstar_routes_prefer_improper_spatial_operation_over_time_reversal();
    test_mgo_dense_kspace_rotation_matches_atom_block_rotation();
    test_mgo_k333_irreducible_sector_matches_single();
    test_mgo_k333_irreducible_sector_matches_both();
    test_bn_shrink_irreducible_sector_can_be_generated_from_symmetry();
}
