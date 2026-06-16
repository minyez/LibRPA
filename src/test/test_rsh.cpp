#include "../math/rsh.h"
#include "../math/wigner_rotation.h"
#include "../utils/constants.h"

#include <cassert>
#include <cmath>
#include <complex>

#include "testutils.h"

namespace {

void assert_matrix_close_to_identity(const librpa_int::ComplexMatrix& matrix,
                                     const double diagonal = 1.0,
                                     const double thres = 1e-12)
{
    assert(matrix.nr == matrix.nc);
    for (int row = 0; row < matrix.nr; ++row)
    {
        for (int col = 0; col < matrix.nc; ++col)
        {
            const std::complex<double> expected((row == col) ? diagonal : 0.0, 0.0);
            assert(fequal(matrix(row, col), expected, std::complex<double>(thres, 0.0)));
        }
    }
}

void test_abs_pm_ordering()
{
    using librpa_int::rsh_m_to_index;
    using librpa_int::rsh_abs_pm_m_to_index;

    assert(rsh_abs_pm_m_to_index(0) == 0);
    assert(rsh_abs_pm_m_to_index(1) == 1);
    assert(rsh_abs_pm_m_to_index(-1) == 2);
    assert(rsh_abs_pm_m_to_index(2) == 3);
    assert(rsh_abs_pm_m_to_index(-2) == 4);

    assert(rsh_m_to_index(2, -2, LIBRPA_ANGULAR_ORDER_NATURAL) == 0);
    assert(rsh_m_to_index(2, 0, LIBRPA_ANGULAR_ORDER_NATURAL) == 2);
    assert(rsh_m_to_index(2, 2, LIBRPA_ANGULAR_ORDER_NATURAL) == 4);
    assert(rsh_m_to_index(2, -2, LIBRPA_ANGULAR_ORDER_ABS_PM) == 4);
    assert(rsh_m_to_index(2, 2, LIBRPA_ANGULAR_ORDER_ABS_PM) == 3);
}

void test_abs_pm_m1_1m_overlap_uses_natural_complex_and_abs_pm_real_indices()
{
    using librpa_int::C_IMAG;
    using librpa_int::complex_to_real_spherical_harmonic_transform;
    using librpa_int::rsh_abs_pm_m_to_index;
    using librpa_int::wigner_m_to_index;

    const auto transform =
        complex_to_real_spherical_harmonic_transform(1,
                                                     LIBRPA_ANGULAR_ORDER_ABS_PM,
                                                     LIBRPA_RSH_COEFF_M_1,
                                                     LIBRPA_RSH_COEFF_1_M);
    assert(fequal(transform(wigner_m_to_index(1, -1), rsh_abs_pm_m_to_index(-1)),
                  -C_IMAG / std::sqrt(2.0),
                  std::complex<double>(1e-14, 0.0)));
    assert(fequal(transform(wigner_m_to_index(1, 1), rsh_abs_pm_m_to_index(-1)),
                  -C_IMAG / std::sqrt(2.0),
                  std::complex<double>(1e-14, 0.0)));
}

void test_general_coefficients_and_ordering_are_independent()
{
    using librpa_int::C_IMAG;
    using librpa_int::complex_to_real_spherical_harmonic_transform;
    using librpa_int::rsh_m_to_index;
    using librpa_int::wigner_m_to_index;

    const auto transform =
        complex_to_real_spherical_harmonic_transform(1,
                                                     LIBRPA_ANGULAR_ORDER_NATURAL,
                                                     LIBRPA_RSH_COEFF_1_M,
                                                     LIBRPA_RSH_COEFF_1_M);
    assert(fequal(transform(wigner_m_to_index(1, -1),
                            rsh_m_to_index(1, -1, LIBRPA_ANGULAR_ORDER_NATURAL)),
                  C_IMAG / std::sqrt(2.0),
                  std::complex<double>(1e-14, 0.0)));
    assert(fequal(transform(wigner_m_to_index(1, 1),
                            rsh_m_to_index(1, -1, LIBRPA_ANGULAR_ORDER_NATURAL)),
                  C_IMAG / std::sqrt(2.0),
                  std::complex<double>(1e-14, 0.0)));
}

void test_complex_to_real_transform_unitary_for_abs_pm_m1_1m_convention()
{
    using librpa_int::complex_to_real_spherical_harmonic_transform;
    using librpa_int::transpose;

    const auto transform =
        complex_to_real_spherical_harmonic_transform(2,
                                                     LIBRPA_ANGULAR_ORDER_ABS_PM,
                                                     LIBRPA_RSH_COEFF_M_1,
                                                     LIBRPA_RSH_COEFF_1_M);
    const auto product = transpose(transform, true) * transform;
    assert_matrix_close_to_identity(product);
}

void test_real_spherical_inversion_parity_for_abs_pm_m1_1m_convention()
{
    using librpa_int::Matrix3;
    using librpa_int::real_spherical_harmonic_rotation_matrix;

    const Matrix3 inversion(-1.0, 0.0, 0.0,
                            0.0, -1.0, 0.0,
                            0.0, 0.0, -1.0);
    assert_matrix_close_to_identity(
        real_spherical_harmonic_rotation_matrix(inversion,
                                                1,
                                                LIBRPA_ANGULAR_ORDER_ABS_PM,
                                                LIBRPA_RSH_COEFF_M_1,
                                                LIBRPA_RSH_COEFF_1_M),
        -1.0);
    assert_matrix_close_to_identity(
        real_spherical_harmonic_rotation_matrix(inversion,
                                                2,
                                                LIBRPA_ANGULAR_ORDER_ABS_PM,
                                                LIBRPA_RSH_COEFF_M_1,
                                                LIBRPA_RSH_COEFF_1_M),
        1.0);
}

}

int main()
{
    test_abs_pm_ordering();
    test_abs_pm_m1_1m_overlap_uses_natural_complex_and_abs_pm_real_indices();
    test_general_coefficients_and_ordering_are_independent();
    test_complex_to_real_transform_unitary_for_abs_pm_m1_1m_convention();
    test_real_spherical_inversion_parity_for_abs_pm_m1_1m_convention();
}
