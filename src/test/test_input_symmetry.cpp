#include "../core/input_symmetry.h"
#include "../core/pbc.h"
#include "../math/rsh.h"

#include <array>
#include <cassert>
#include <complex>
#include <filesystem>
#include <fstream>
#include <vector>

#include "testutils.h"

namespace {

void assert_matrix_close(const librpa_int::ComplexMatrix& actual,
                         const librpa_int::ComplexMatrix& expected,
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
    using librpa_int::InputSymmetryContext;
    using librpa_int::Matrix3;
    using librpa_int::SpeciesBasisLayout;
    using librpa_int::build_input_symmetry_abf_rotation_matrix;
    using librpa_int::real_spherical_harmonic_rotation_matrix;

    InputSymmetryContext ctx;
    ctx.basis_convention = {-1,
                            0,
                            LIBRPA_ANGULAR_ORDER_NATURAL,
                            LIBRPA_RSH_COEFF_1_M,
                            LIBRPA_RSH_COEFF_1_M};
    SpeciesBasisLayout layout;
    layout.label = "X";
    layout.set({1});
    ctx.abf_type_layout_candidates = {{layout}};
    ctx.abf_shell_layout_available = true;
    ctx.lattice_vectors = Matrix3(1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0);
    ctx.lattice_available = true;

    const Matrix3 cartesian_rotation(0.0, -1.0, 0.0,
                                     1.0, 0.0, 0.0,
                                     0.0, 0.0, 1.0);
    const Matrix3 direct_rotation = cartesian_rotation.Transpose();

    const auto fallback_rotation =
        build_input_symmetry_abf_rotation_matrix(ctx, 0, 3, {}, direct_rotation);
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
        build_input_symmetry_abf_rotation_matrix(ctx, 0, 3, {}, fractional_rotation);
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
    using librpa_int::BasisConvention;
    using librpa_int::Matrix3;
    using librpa_int::SpaceGroupSymOp;
    using librpa_int::Vector3_Order;
    using librpa_int::build_input_symmetry_kspace_phase;
    using librpa_int::build_input_symmetry_kspace_shell_rotations;
    using librpa_int::build_input_symmetry_shell_rotations_from_direct_rotation;
    using librpa_int::real_spherical_harmonic_rotation_matrix;

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
        build_input_symmetry_shell_rotations_from_direct_rotation(
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
    const auto phase = build_input_symmetry_kspace_phase(
        k_source, k_target, atom_from, atom_to, return_lattice, basis_convention);
    assert(fequal(phase, std::complex<double>(0.0, -1.0), std::complex<double>(1e-12, 0.0)));

    const auto phased_shell_rotations =
        build_input_symmetry_kspace_shell_rotations(op,
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

void test_species_basis_layout_keeps_shell_order()
{
    using librpa_int::ComplexMatrix;
    using librpa_int::InputSymmetryContext;
    using librpa_int::SpeciesBasisLayout;
    using librpa_int::build_input_symmetry_ao_rotation_matrix;

    SpeciesBasisLayout layout;
    assert(!layout.is_shell_available());
    layout.label = "X";
    layout.set({1, 0});
    assert(layout.is_shell_available());
    assert(layout.n_ao == 4);
    assert(layout.shell_counts.at(1) == 1);
    assert(layout.shell_indices.at(1).front() == 0);
    assert(layout.shell_indices.at(0).front() == 1);

    InputSymmetryContext ctx;
    ctx.ao_type_layouts = {layout};

    ComplexMatrix p_rotation(3, 3);
    p_rotation.zero_out();
    p_rotation(0, 0) = {2.0, 0.0};
    p_rotation(1, 1) = {3.0, 0.0};
    p_rotation(2, 2) = {4.0, 0.0};
    ComplexMatrix s_rotation(1, 1);
    s_rotation.zero_out();
    s_rotation(0, 0) = {7.0, 0.0};

    const auto rotation =
        build_input_symmetry_ao_rotation_matrix(ctx, 0, {{1, p_rotation}, {0, s_rotation}});

    ComplexMatrix expected(4, 4);
    expected.zero_out();
    expected(0, 0) = {2.0, 0.0};
    expected(1, 1) = {3.0, 0.0};
    expected(2, 2) = {4.0, 0.0};
    expected(3, 3) = {7.0, 0.0};
    assert_matrix_close(rotation, expected);
}

void write_file(const std::filesystem::path& path, const std::string& text)
{
    std::ofstream ofs(path);
    assert(ofs.good());
    ofs << text;
}

void add_irreducible_sector_entry(librpa_int::input_symmetry_irreducible_sector_t& sector,
                                  const librpa_int::atom_t atom_i,
                                  const librpa_int::atom_t atom_j,
                                  const librpa_int::input_symmetry_R_t& R)
{
    sector[{atom_i, atom_j}].insert(R);
}

librpa_int::InputSymmetryOperation make_row_symmetry_operation(const std::array<int, 9>& rotation)
{
    librpa_int::InputSymmetryOperation op;
    op.rotation = librpa_int::Matrix3(rotation[0], rotation[1], rotation[2],
                                      rotation[3], rotation[4], rotation[5],
                                      rotation[6], rotation[7], rotation[8]);
    op.translation = {0.0, 0.0, 0.0};
    op.use_row_convention = true;
    return op;
}

librpa_int::InputSymmetryContext make_bn_shrink_symmetry_context()
{
    librpa_int::InputSymmetryContext ctx;
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

void test_bn_shrink_irreducible_sector_can_be_generated_from_symmetry()
{
    using librpa_int::Vector3_Order;
    using librpa_int::build_input_symmetry_rspace_irreducible_sector;
    using librpa_int::build_input_symmetry_rspace_sector_stars;
    using librpa_int::construct_R_grid;
    using librpa_int::input_symmetry_irreducible_sector_t;
    using librpa_int::input_symmetry_rspace_sector_stars_t;

    auto ctx = make_bn_shrink_symmetry_context();
    const Vector3_Order<int> period{2, 2, 2};
    const auto Rlist = construct_R_grid(period);
    const auto generated_sector =
        build_input_symmetry_rspace_irreducible_sector(ctx, {}, Rlist);

    input_symmetry_irreducible_sector_t expected_sector;
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
    ctx.available = true;
    input_symmetry_rspace_sector_stars_t sector_stars;
    build_input_symmetry_rspace_sector_stars(ctx, {}, period, Rlist, sector_stars);
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

void test_abacus_sidecars_do_not_require_symrot_r()
{
    using librpa_int::InputSymmetryContext;
    using librpa_int::InputSymmetryConvention;
    using librpa_int::load_input_symmetry_context;

    const auto dir = std::filesystem::temp_directory_path()
                     / "librpa_test_input_symmetry_no_symrot_r";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    write_file(dir / "irreducible_sector.txt",
               "atompair (0, 0), R = (0, 0, 0)\n");

    InputSymmetryContext ctx;
    assert(load_input_symmetry_context(dir.string(), InputSymmetryConvention::ABACUS, ctx));
    assert(ctx.rspace_operations.empty());
    assert(ctx.kstars.empty());

    std::filesystem::remove_all(dir);
}

void test_abacus_generated_symops_do_not_require_sidecars()
{
    using librpa_int::InputSymmetryContext;
    using librpa_int::InputSymmetryConvention;
    using librpa_int::Matrix3;
    using librpa_int::load_input_symmetry_context;

    const auto dir = std::filesystem::temp_directory_path()
                     / "librpa_test_input_symmetry_generated_symops_no_sidecars";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    InputSymmetryContext ctx = make_bn_shrink_symmetry_context();
    ctx.set_lattice(Matrix3(1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0),
                    Matrix3(1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0));

    assert(load_input_symmetry_context(dir.string(), InputSymmetryConvention::ABACUS, ctx));
    assert(ctx.available);
    assert(ctx.rspace_operations.size() == 24);
    assert(ctx.irreducible_sector.empty());

    std::filesystem::remove_all(dir);
}

void test_abacus_existing_symops_skip_k_rotation_sidecars()
{
    using librpa_int::InputSymmetryContext;
    using librpa_int::InputSymmetryConvention;
    using librpa_int::InputSymmetryOperation;
    using librpa_int::Matrix3;
    using librpa_int::load_input_symmetry_context;

    const auto dir = std::filesystem::temp_directory_path()
                     / "librpa_test_input_symmetry_existing_symops_skip_k_rotation_sidecars";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    write_file(dir / "irreducible_sector.txt",
               "atompair (0, 0), R = (0, 0, 0)\n");
    write_file(dir / "symrot_k.txt", "this would fail if parsed\n");
    write_file(dir / "symrot_abf_k.txt", "this would fail if parsed\n");

    InputSymmetryContext ctx;
    ctx.set_lattice(Matrix3(1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0),
                    Matrix3(1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0));
    InputSymmetryOperation op;
    op.rotation = Matrix3(1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0);
    op.translation = {0.0, 0.0, 0.0};
    op.use_row_convention = true;
    ctx.rspace_operations.push_back(op);

    assert(load_input_symmetry_context(dir.string(), InputSymmetryConvention::ABACUS, ctx));
    assert(ctx.rspace_operations.size() == 1);
    assert(ctx.lattice_available);
    assert(ctx.atom_to_type.empty());
    assert(ctx.kstars.empty());
    assert(ctx.abf_kstars.empty());

    std::filesystem::remove_all(dir);
}

} // namespace

int main()
{
    test_abf_rotation_fallback_uses_basis_convention();
    test_kspace_shell_rotations_use_direct_rotation();
    test_species_basis_layout_keeps_shell_order();
    test_bn_shrink_irreducible_sector_can_be_generated_from_symmetry();
    test_abacus_sidecars_do_not_require_symrot_r();
    test_abacus_generated_symops_do_not_require_sidecars();
    test_abacus_existing_symops_skip_k_rotation_sidecars();
}
