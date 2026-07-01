#include "../core/symmetry_context.h"
#include "../core/pbc.h"
#include "../math/rsh.h"

#include <array>
#include <cassert>
#include <complex>
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

void test_abf_rotation_fallback_uses_basis_convention()
{
    SymmetryContext ctx;
    ctx.basis_convention = {-1,
                            0,
                            LIBRPA_ANGULAR_ORDER_NATURAL,
                            LIBRPA_RSH_COEFF_1_M,
                            LIBRPA_RSH_COEFF_1_M};
    SpeciesBasisLayout layout;
    layout.label = "X";
    layout.set({1});
    ctx.add_basis_layouts("AUX", {layout});
    ctx.lattice_vectors = Matrix3(1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0);
    ctx.lattice_available = true;

    const Matrix3 cartesian_rotation(0.0, -1.0, 0.0,
                                     1.0, 0.0, 0.0,
                                     0.0, 0.0, 1.0);
    const Matrix3 direct_rotation = cartesian_rotation.Transpose();

    const auto fallback_rotation =
        build_symmetry_rotation_matrix(ctx, "AUX", 0, {}, direct_rotation);
    const auto expected_rotation =
        real_spherical_harmonic_rotation_matrix(cartesian_rotation,
                                                1,
                                                ctx.basis_convention.order,
                                                ctx.basis_convention.coeff_m_negative,
                                                ctx.basis_convention.coeff_m_positive);
    assert_matrix_close(fallback_rotation, expected_rotation);

    const Matrix3 skew_lattice(2.0, 0.0, 0.0,
                               0.5, 1.5, 0.0,
                               0.2, 0.3, 2.0);
    const Matrix3 fractional_rotation =
        skew_lattice * cartesian_rotation.Transpose() * skew_lattice.Inverse();
    ctx.lattice_vectors = skew_lattice;

    const auto skew_fallback_rotation =
        build_symmetry_rotation_matrix(ctx, "AUX", 0, {}, fractional_rotation);
    const auto skew_expected_rotation =
        real_spherical_harmonic_rotation_matrix(cartesian_rotation,
                                                1,
                                                ctx.basis_convention.order,
                                                ctx.basis_convention.coeff_m_negative,
                                                ctx.basis_convention.coeff_m_positive);
    assert_matrix_close(skew_fallback_rotation, skew_expected_rotation);
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
    ctx.set_lattice(Matrix3(1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0),
                    Matrix3(1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0));
    ctx.basis_convention = {-1,
                            0,
                            LIBRPA_ANGULAR_ORDER_NATURAL,
                            LIBRPA_RSH_COEFF_1_M,
                            LIBRPA_RSH_COEFF_1_M};
    ctx.atom_to_type[0] = 0;
    ctx.atom_to_type[1] = 1;
    ctx.input_coord_frac[0] = {0.0, 0.0, 0.0};
    ctx.input_coord_frac[1] = {-0.25, -0.25, -0.25};

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
    star.members[0].isym = 0;
    star.members[0].k_bz = {0.0, 0.0, 0.0};
    ctx.kstars.push_back(star);

    ctx.generate_kstar_member_rotations(0);
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

    SymmetryContext ctx;
    ctx.add_basis_layouts("WFC", {layout});

    ComplexMatrix p_rotation(3, 3);
    p_rotation.zero_out();
    p_rotation(0, 0) = {2.0, 0.0};
    p_rotation(1, 1) = {3.0, 0.0};
    p_rotation(2, 2) = {4.0, 0.0};
    ComplexMatrix s_rotation(1, 1);
    s_rotation.zero_out();
    s_rotation(0, 0) = {7.0, 0.0};

    const auto rotation =
        build_symmetry_rotation_matrix(ctx, "WFC", 0, {{1, p_rotation}, {0, s_rotation}});

    ComplexMatrix expected(4, 4);
    expected.zero_out();
    expected(0, 0) = {2.0, 0.0};
    expected(1, 1) = {3.0, 0.0};
    expected(2, 2) = {4.0, 0.0};
    expected(3, 3) = {7.0, 0.0};
    assert_matrix_close(rotation, expected);
}

void test_shell_layout_key_matches_basis_dimensions()
{
    SymmetryContext ctx;
    ctx.atom_to_type = {{0, 0}, {1, 0}};

    SpeciesBasisLayout full_layout;
    full_layout.label = "full";
    full_layout.set({1, 0});

    SpeciesBasisLayout shrink_layout;
    shrink_layout.label = "shrink";
    shrink_layout.set({0});

    ctx.add_basis_layouts("AUX", {full_layout});
    ctx.add_basis_layouts("AUXSHRINK", {shrink_layout});

    assert(find_symmetry_shell_layout_key(ctx, {{0, 4}, {1, 4}}, "AUX") == "AUX");
    assert(find_symmetry_shell_layout_key(ctx, {{0, 1}, {1, 1}}, "AUX")
           == "AUXSHRINK");
}

void test_context_adds_labeled_atomic_basis_layout()
{
    AtomicBasis basis(std::vector<std::size_t>{1, 3});
    basis.label = "WFC";
    basis.set_l_shells({{0}, {1}});

    SymmetryContext ctx;
    ctx.add_basis_layouts(basis, {{0, 0}, {1, 1}});

    assert(ctx.has_shell_layout("WFC"));
    assert(ctx.ao_lmax == 1);
    assert(ctx.get_shell_layout("WFC", 0).n_ao == 1);
    assert(ctx.get_shell_layout("WFC", 1).n_ao == 3);
}

void test_full_kpoint_members_skip_full_grid()
{
    SymmetryContext ctx;
    ctx.set_available();

    SymmetryKStar star;
    star.k_ibz = {0.0, 0.0, 0.0};
    star.members.resize(2);
    star.members[0].k_bz = {0.0, 0.0, 0.0};
    star.members[1].k_bz = {0.5, 0.0, 0.0};
    ctx.kstars.push_back(star);

    const std::vector<Vector3_Order<double>> full_kpoints{
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0},
    };

    assert(build_symmetry_full_kpoint_member_list(ctx, full_kpoints).empty());
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
    ComplexMatrix shell_rotation(1, 1);
    shell_rotation(0, 0) = {2.0, 0.0};
    op.shell_rotations[0] = shell_rotation;

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
    assert_matrix_close(saved.shell_rotations.at(0), shell_rotation);

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

void set_mgo_primitive_lattice(SymmetryContext& ctx)
{
    const Matrix3 lattice(0.0, 0.5, 0.5,
                          0.5, 0.0, 0.5,
                          0.5, 0.5, 0.0);
    ctx.set_lattice(lattice, lattice.Inverse().Transpose());
}

void test_mgo_k333_irreducible_sector_matches_single()
{
    SymmetryContext ctx;
    ctx.atom_to_type = {{0, 0},};
    ctx.input_coord_frac = {{0, {0.0, 0.0, 0.0}},};
    set_mgo_primitive_lattice(ctx);
    add_mgo_fractional_symmetry_operations(ctx);

    const Vector3_Order<int> period{3, 3, 3};
    const auto Rlist = construct_R_grid(period);
    const auto generated_sector =
        build_symmetry_rspace_irreducible_sector(ctx, {}, Rlist);

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
    build_symmetry_rspace_sector_stars(ctx, {}, period, Rlist, sector_stars);
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
    ctx.atom_to_type = {{0, 0}, {1, 1}};
    ctx.input_coord_frac = {{0, {0.0, 0.0, 0.0}}, {1, {0.5, 0.5, 0.5}}};
    set_mgo_primitive_lattice(ctx);
    add_mgo_fractional_symmetry_operations(ctx);

    const Vector3_Order<int> period{3, 3, 3};
    const auto Rlist = construct_R_grid(period);
    const auto generated_sector =
        build_symmetry_rspace_irreducible_sector(ctx, {}, Rlist);

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
    build_symmetry_rspace_sector_stars(ctx, {}, period, Rlist, sector_stars);
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
        build_symmetry_rspace_irreducible_sector(ctx, {}, Rlist);

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
    build_symmetry_rspace_sector_stars(ctx, {}, period, Rlist, sector_stars);
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
    test_abf_rotation_fallback_uses_basis_convention();
    test_kspace_shell_rotations_use_direct_rotation();
    test_kstar_member_return_lattice_preserves_input_fractional_representative();
    test_species_basis_layout_keeps_shell_order();
    test_shell_layout_key_matches_basis_dimensions();
    test_context_adds_labeled_atomic_basis_layout();
    test_full_kpoint_members_skip_full_grid();
    test_mgo_k333_irreducible_sector_matches_single();
    test_mgo_k333_irreducible_sector_matches_both();
    test_bn_shrink_irreducible_sector_can_be_generated_from_symmetry();
}
