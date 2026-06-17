#include "../core/input_symmetry.h"
#include "../math/rsh.h"

#include <array>
#include <cassert>
#include <complex>
#include <filesystem>
#include <fstream>

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
    using librpa_int::InputSymmetryAOTypeLayout;
    using librpa_int::InputSymmetryContext;
    using librpa_int::Matrix3;
    using librpa_int::build_input_symmetry_abf_rotation_matrix;
    using librpa_int::real_spherical_harmonic_rotation_matrix;

    InputSymmetryContext ctx;
    ctx.basis_convention = {-1,
                            0,
                            LIBRPA_ANGULAR_ORDER_NATURAL,
                            LIBRPA_RSH_COEFF_1_M,
                            LIBRPA_RSH_COEFF_1_M};
    ctx.abf_type_layout_candidates = {
        {InputSymmetryAOTypeLayout{"X", "", {0, 1}, 3}},
    };
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

void write_file(const std::filesystem::path& path, const std::string& text)
{
    std::ofstream ofs(path);
    assert(ofs.good());
    ofs << text;
}

void test_abacus_symrot_r_rotation_is_loaded_as_row_fractional_operation()
{
    using librpa_int::InputSymmetryContext;
    using librpa_int::InputSymmetryConvention;
    using librpa_int::Vector3_Order;
    using librpa_int::apply_space_group_symmetry_operation;
    using librpa_int::load_input_symmetry_context;

    const auto dir = std::filesystem::temp_directory_path()
                     / "librpa_test_input_symmetry_symrot_r";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    write_file(dir / "irreducible_sector.txt",
               "atompair (0, 0), R = (0, 0, 0)\n");
    write_file(dir / "symrot_k.txt",
               "Number of IBZ k-points (k stars): 0\n");
    write_file(dir / "symrot_R.txt",
               "Lmax of AOs: 0\n"
               "Lmax of ABFs: 0\n"
               "1\n"
               "0 -1 0\n"
               "1 -1 0\n"
               "0 -1 1\n"
               "(0 0 0)\n"
               "(1.0,0.0)\n");

    InputSymmetryContext ctx;
    assert(load_input_symmetry_context(dir.string(), InputSymmetryConvention::ABACUS, ctx));
    assert(ctx.rspace_operations.size() == 1);

    const auto transformed =
        apply_space_group_symmetry_operation(ctx.rspace_operations[0],
                                             Vector3_Order<double>{0.25, 0.25, 0.25});
    assert(fequal(transformed.x, 0.25));
    assert(fequal(transformed.y, -0.75));
    assert(fequal(transformed.z, 0.25));

    std::filesystem::remove_all(dir);
}

} // namespace

int main()
{
    test_abf_rotation_fallback_uses_basis_convention();
    test_abacus_symrot_r_rotation_is_loaded_as_row_fractional_operation();
}
