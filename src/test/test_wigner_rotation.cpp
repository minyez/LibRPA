#include "../math/wigner_rotation.h"
#include "../utils/constants.h"

#include <cassert>
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

void assert_matrix_close(const librpa_int::Matrix3& actual, const librpa_int::Matrix3& expected,
                         const double thres = 1e-12)
{
    assert(fequal(actual.e11, expected.e11, thres));
    assert(fequal(actual.e12, expected.e12, thres));
    assert(fequal(actual.e13, expected.e13, thres));
    assert(fequal(actual.e21, expected.e21, thres));
    assert(fequal(actual.e22, expected.e22, thres));
    assert(fequal(actual.e23, expected.e23, thres));
    assert(fequal(actual.e31, expected.e31, thres));
    assert(fequal(actual.e32, expected.e32, thres));
    assert(fequal(actual.e33, expected.e33, thres));
}

void test_magnetic_quantum_number_index()
{
    using librpa_int::wigner_m_to_index;

    assert(wigner_m_to_index(2, -2) == 0);
    assert(wigner_m_to_index(2, -1) == 1);
    assert(wigner_m_to_index(2, 0) == 2);
    assert(wigner_m_to_index(2, 1) == 3);
    assert(wigner_m_to_index(2, 2) == 4);
}

void test_wigner_small_d_identity()
{
    using librpa_int::wigner_small_d_matrix;

    assert_matrix_close_to_identity(wigner_small_d_matrix(0.0, 3));
}

void test_wigner_small_d_orthogonal()
{
    using librpa_int::transpose;
    using librpa_int::wigner_small_d_matrix;

    const auto d_matrix = wigner_small_d_matrix(0.37, 3);
    const auto product = transpose(d_matrix, false) * d_matrix;
    assert_matrix_close_to_identity(product, 1.0, 1e-11);
}

void test_wigner_D_identity()
{
    using librpa_int::Vector3;
    using librpa_int::wigner_D_matrix;

    assert_matrix_close_to_identity(wigner_D_matrix(Vector3<double>{0.0, 0.0, 0.0}, 2));
}

void test_euler_angles_to_rotation_matrix_known_beta_rotation()
{
    using librpa_int::euler_angles_zyz_to_rotation_matrix;
    using librpa_int::Matrix3;
    using librpa_int::PI;
    using librpa_int::Vector3;

    assert_matrix_close(euler_angles_zyz_to_rotation_matrix(Vector3<double>{0.0, PI / 2.0, 0.0}),
                        Matrix3(0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0));
}

void test_euler_angles_to_rotation_matrix_round_trip()
{
    using librpa_int::euler_angles_zyz_to_rotation_matrix;
    using librpa_int::rotation_matrix_to_euler_angles_zyz;
    using librpa_int::Vector3;

    const auto rotation = euler_angles_zyz_to_rotation_matrix(Vector3<double>{0.37, 1.21, 2.43});
    assert_matrix_close(
        euler_angles_zyz_to_rotation_matrix(rotation_matrix_to_euler_angles_zyz(rotation, 1e-12)),
        rotation);
}
}

int main()
{
    test_magnetic_quantum_number_index();
    test_wigner_small_d_identity();
    test_wigner_small_d_orthogonal();
    test_wigner_D_identity();
    test_euler_angles_to_rotation_matrix_known_beta_rotation();
    test_euler_angles_to_rotation_matrix_round_trip();
}
