/*!
 * @file symmetry.h
 * @brief Space-group symmetry operation primitives.
 */
#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "matrix3.h"
#include "vector3_order.h"

namespace librpa_int
{

/*!
 * @brief Space-group operation in fractional coordinates.
 *
 * Coordinates are treated as row vectors: r' = r * rotation + translation.
 */
struct SpaceGroupSymOp
{
    int isym = -1;
    Matrix3 rotation{0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0};
    Vector3_Order<double> translation{0.0, 0.0, 0.0};
};

/*!
 * @brief Container for a set of space-group symmetry operations.
 */
template <typename OperationType = SpaceGroupSymOp>
struct SpaceGroupSymOps
{
    using operation_type = OperationType;

    std::vector<operation_type> operations;

    void clear() { operations.clear(); }
    bool empty() const { return operations.empty(); }
    std::size_t size() const { return operations.size(); }
    void reserve(const std::size_t count) { operations.reserve(count); }

    void push_back(const operation_type& operation) { operations.push_back(operation); }
    void push_back(operation_type&& operation) { operations.push_back(std::move(operation)); }

    operation_type& at(const std::size_t index) { return operations.at(index); }
    const operation_type& at(const std::size_t index) const { return operations.at(index); }

    operation_type& operator[](const std::size_t index) { return operations[index]; }
    const operation_type& operator[](const std::size_t index) const { return operations[index]; }

    auto begin() { return operations.begin(); }
    auto begin() const { return operations.begin(); }
    auto cbegin() const { return operations.cbegin(); }
    auto end() { return operations.end(); }
    auto end() const { return operations.end(); }
    auto cend() const { return operations.cend(); }
};

Matrix3 multiply_space_group_rotation_matrices(const Matrix3& lhs,
                                               const Matrix3& rhs);

Vector3_Order<double> multiply_row_vector(const Vector3_Order<double>& vec,
                                          const Matrix3& matrix);

SpaceGroupSymOp compose_space_group_symmetry_operations(
    const SpaceGroupSymOp& lhs,
    const SpaceGroupSymOp& rhs);

Vector3_Order<double> apply_space_group_symmetry_operation(
    const SpaceGroupSymOp& operation,
    const Vector3_Order<double>& coord);

} // namespace librpa_int
