#include "../math/rsh.h"
#include "../math/wigner_rotation.h"
#include "../utils/constants.h"

#include <cassert>
#include <cmath>
#include <complex>

#include "librpa_enums.h"
#include "testutils.h"

namespace {

using namespace std;
using namespace librpa_int;

void assert_matrix_close_to_identity(const ComplexMatrix& matrix,
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

void test_aims_rsh()
{
    const LibrpaAngularOrder order = LIBRPA_ANGULAR_ORDER_NATURAL;
    const LibrpaRshCoeff coeff_m_nega = LIBRPA_RSH_COEFF_1_M;
    const LibrpaRshCoeff coeff_m_posi = LIBRPA_RSH_COEFF_1_M;
    const Matrix3 rot_c4z(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    const Vector3<double> euler_c4z{PI/2.0, 0.0, 0.0};
    const auto euler_from_rot = rotation_matrix_to_euler_angles_zyz(rot_c4z);
    assert(fequal(euler_from_rot.x, euler_c4z.x));
    assert(fequal(euler_from_rot.y, euler_c4z.y));
    assert(fequal(euler_from_rot.z, euler_c4z.z));

    const auto rotmat = real_spherical_harmonic_rotation_matrix(rot_c4z, 1, order, coeff_m_nega, coeff_m_posi);
    cout << rotmat << endl;
    assert(fequal(rotmat(0, 0), C_ZERO));
    assert(fequal(rotmat(0, 1), C_ZERO));
    assert(fequal(rotmat(0, 2), -C_ONE));
    assert(fequal(rotmat(1, 0), C_ZERO));
    assert(fequal(rotmat(1, 1), C_ONE));
    assert(fequal(rotmat(1, 2), C_ZERO));
    assert(fequal(rotmat(2, 0), C_ONE));
    assert(fequal(rotmat(2, 1), C_ZERO));
    assert(fequal(rotmat(2, 2), C_ZERO));
}

void test_abacus_rsh()
{
    const LibrpaAngularOrder order = LIBRPA_ANGULAR_ORDER_ABS_PM;
    const LibrpaRshCoeff coeff_m_nega = LIBRPA_RSH_COEFF_M_1;
    const LibrpaRshCoeff coeff_m_posi = LIBRPA_RSH_COEFF_1_M;

    // C4(z)
    const Matrix3 rot_c4z(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    const Vector3<double> euler_c4z{PI/2.0, 0.0, 0.0};
    const auto euler_from_rot = rotation_matrix_to_euler_angles_zyz(rot_c4z);
    assert(fequal(euler_from_rot.x, euler_c4z.x));
    assert(fequal(euler_from_rot.y, euler_c4z.y));
    assert(fequal(euler_from_rot.z, euler_c4z.z));

    const auto rotmat_from_euler = real_spherical_harmonic_rotation_matrix(euler_c4z, 1, order, coeff_m_nega, coeff_m_posi);
    assert(fequal(rotmat_from_euler(0, 0), C_ONE));
    assert(fequal(rotmat_from_euler(0, 1), C_ZERO));
    assert(fequal(rotmat_from_euler(0, 2), C_ZERO));
    assert(fequal(rotmat_from_euler(1, 0), C_ZERO));
    assert(fequal(rotmat_from_euler(1, 1), C_ZERO));
    assert(fequal(rotmat_from_euler(1, 2), -C_ONE));
    assert(fequal(rotmat_from_euler(2, 0), C_ZERO));
    assert(fequal(rotmat_from_euler(2, 1), C_ONE));
    assert(fequal(rotmat_from_euler(2, 2), C_ZERO));
    const auto rotmat = real_spherical_harmonic_rotation_matrix(rot_c4z, 1, order, coeff_m_nega, coeff_m_posi);
    assert(fequal_array(9, rotmat_from_euler.c, rotmat.c));
    cout << rotmat << endl;
}

}

int main()
{
    test_abs_pm_ordering();
    test_abs_pm_m1_1m_overlap_uses_natural_complex_and_abs_pm_real_indices();
    test_general_coefficients_and_ordering_are_independent();
    test_complex_to_real_transform_unitary_for_abs_pm_m1_1m_convention();
    test_real_spherical_inversion_parity_for_abs_pm_m1_1m_convention();
    test_aims_rsh();
    test_abacus_rsh();
}
