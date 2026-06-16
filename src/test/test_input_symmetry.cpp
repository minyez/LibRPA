#include "../core/input_symmetry.h"
#include "../math/rsh.h"

#include <array>
#include <cassert>
#include <complex>

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

    const std::array<std::array<double, 3>, 3> direct_rotation{{{{0.0, -1.0, 0.0}},
                                                                {{1.0, 0.0, 0.0}},
                                                                {{0.0, 0.0, 1.0}}}};
    const Matrix3 cartesian_rotation(0.0, -1.0, 0.0,
                                     1.0, 0.0, 0.0,
                                     0.0, 0.0, 1.0);

    const auto fallback_rotation =
        build_input_symmetry_abf_rotation_matrix(ctx, 0, 3, {}, direct_rotation);
    const auto expected_rotation =
        real_spherical_harmonic_rotation_matrix(cartesian_rotation,
                                                1,
                                                ctx.basis_convention.order,
                                                ctx.basis_convention.coeff_m_negative,
                                                ctx.basis_convention.coeff_m_positive);
    assert_matrix_close(fallback_rotation, expected_rotation);
}

} // namespace

int main()
{
    test_abf_rotation_fallback_uses_basis_convention();
}
