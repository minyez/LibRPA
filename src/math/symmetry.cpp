#include "symmetry.h"

namespace librpa_int
{

space_group_rotation_t multiply_space_group_rotation_matrices(
    const space_group_rotation_t& lhs,
    const space_group_rotation_t& rhs)
{
    space_group_rotation_t product{{{{0.0, 0.0, 0.0}},
                                    {{0.0, 0.0, 0.0}},
                                    {{0.0, 0.0, 0.0}}}};
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            for (int k = 0; k < 3; ++k)
            {
                product[row][col] += lhs[row][k] * rhs[k][col];
            }
        }
    }
    return product;
}

Vector3_Order<double> multiply_row_vector(const Vector3_Order<double>& vec,
                                          const space_group_rotation_t& matrix)
{
    return {
        vec.x * matrix[0][0] + vec.y * matrix[1][0] + vec.z * matrix[2][0],
        vec.x * matrix[0][1] + vec.y * matrix[1][1] + vec.z * matrix[2][1],
        vec.x * matrix[0][2] + vec.y * matrix[1][2] + vec.z * matrix[2][2],
    };
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
