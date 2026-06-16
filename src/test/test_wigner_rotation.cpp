#include "../math/wigner_rotation.h"

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

}

int main()
{
    test_magnetic_quantum_number_index();
    test_wigner_small_d_identity();
    test_wigner_small_d_orthogonal();
    test_wigner_D_identity();
}
