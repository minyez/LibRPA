#include "symmetry.h"

namespace librpa_int
{

Matrix3 multiply_space_group_rotation_matrices(const Matrix3& lhs,
                                               const Matrix3& rhs)
{
    return lhs * rhs;
}

Vector3_Order<double> multiply_row_vector(const Vector3_Order<double>& vec,
                                          const Matrix3& matrix)
{
    return Vector3_Order<double>(vec * matrix);
}

SpaceGroupSymOp compose_space_group_symmetry_operations(
    const SpaceGroupSymOp& lhs,
    const SpaceGroupSymOp& rhs)
{
    SpaceGroupSymOp composed;
    composed.rotation = multiply_space_group_rotation_matrices(lhs.rotation, rhs.rotation);
    composed.translation = multiply_row_vector(lhs.translation, rhs.rotation) + rhs.translation;
    return composed;
}

Vector3_Order<double> apply_space_group_symmetry_operation(
    const SpaceGroupSymOp& operation,
    const Vector3_Order<double>& coord)
{
    return multiply_row_vector(coord, operation.rotation) + operation.translation;
}

} // namespace librpa_int
