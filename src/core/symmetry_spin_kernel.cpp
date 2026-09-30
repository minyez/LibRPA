/*!
 * @file symmetry_spin_kernel.cpp
 * @brief Non-template parts of the four-block spinor kernel (Phase 3).
 */
#include "symmetry_spin_kernel.h"

namespace librpa_int
{

CollinearChannelAction classify_collinear_action(
    const std::array<std::complex<double>, 4> &U,
    bool antiunitary,
    double tol)
{
    // The conjugation of an antiunitary operation is handled by the Theta
    // remap in transform_spinor_bilinear; classification uses U only.
    (void)antiunitary;

    // R = U P_up U^dagger with P_up = diag(1, 0), U = [a b; c d] row-major:
    // R = [|a|^2, a conj(c); c conj(a), |c|^2]. The projector removes the
    // +-U double-cover sign.
    const auto &a = U[0];
    const auto &c = U[2];
    const double r00 = std::norm(a);
    const double r11 = std::norm(c);
    const double r01 = std::abs(a * std::conj(c));

    const bool is_pup = std::abs(r00 - 1.0) < tol && r11 < tol && r01 < tol;
    const bool is_pdn = r00 < tol && std::abs(r11 - 1.0) < tol && r01 < tol;
    if (is_pup)
    {
        return CollinearChannelAction::Keep;
    }
    if (is_pdn)
    {
        return CollinearChannelAction::Swap;
    }
    return CollinearChannelAction::Incompatible;
}

CollinearChannelAction classify_collinear_action_effective(
    const std::array<std::complex<double>, 4> &U,
    bool antiunitary,
    double tol)
{
    const auto action = classify_collinear_action(U, antiunitary, tol);
    if (!antiunitary || action == CollinearChannelAction::Incompatible)
    {
        return action;
    }
    // Theta remap exchanges the channels: XOR the permutation.
    return action == CollinearChannelAction::Keep ? CollinearChannelAction::Swap
                                                  : CollinearChannelAction::Keep;
}

} // namespace librpa_int
