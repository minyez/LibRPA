/*!
 * @file qpoint_view.h
 * @brief Derived q-point views for response and screened-Coulomb calculations.
 */
#pragma once

#include <map>
#include <vector>

#include "../math/vector3_order.h"

namespace librpa_int
{

class PeriodicBoundaryData;
struct SymmetryContext;

enum class SymmetryQPointRestoreMode
{
    NONE,
    TIME_REVERSAL,
    FULL_CRYSTAL,
};

struct SymmetryQPointView
{
    SymmetryQPointRestoreMode restore_mode = SymmetryQPointRestoreMode::NONE;
    std::vector<Vector3_Order<double>> representatives;
    std::map<Vector3_Order<double>, std::vector<Vector3_Order<double>>> members;
    std::map<Vector3_Order<double>, double> weights;
};

SymmetryQPointView build_symmetry_qpoint_view(
    const SymmetryContext& ctx,
    const PeriodicBoundaryData& pbc,
    bool use_symmetry);

} // namespace librpa_int
