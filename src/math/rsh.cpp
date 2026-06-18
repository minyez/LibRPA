#include "rsh.h"

#include "wigner_rotation.h"
#include "../utils/constants.h"

#include <cmath>
#include <stdexcept>

namespace librpa_int {

namespace {

double alternating_sign(const int m)
{
    return (std::abs(m) % 2 == 0) ? 1.0 : -1.0;
}

void validate_angular_momentum(const int l, const int m1, const int m2)
{
    if (l < 0)
    {
        throw std::invalid_argument("angular momentum l must be non-negative");
    }
    if (std::abs(m1) > l || std::abs(m2) > l)
    {
        throw std::invalid_argument("magnetic quantum number is outside [-l, l]");
    }
}

void validate_angular_order(const LibrpaAngularOrder angular_order)
{
    switch (angular_order)
    {
    case LIBRPA_ANGULAR_ORDER_NATURAL:
    case LIBRPA_ANGULAR_ORDER_ABS_PM:
        return;
    case LIBRPA_ANGULAR_ORDER_OPENMX:
    case LIBRPA_ANGULAR_ORDER_PYSCF:
        throw std::invalid_argument("requested angular order is not implemented for RSH rotations");
    case LIBRPA_ANGULAR_ORDER_UNSET:
        throw std::invalid_argument("angular order is unset");
    }
    throw std::invalid_argument("unknown angular order");
}

void validate_rsh_coeff(const LibrpaRshCoeff coeff)
{
    switch (coeff)
    {
    case LIBRPA_RSH_COEFF_1_M:
    case LIBRPA_RSH_COEFF_M_1:
        return;
    case LIBRPA_RSH_COEFF_UNSET:
        throw std::invalid_argument("RSH coefficient convention is unset");
    }
    throw std::invalid_argument("unknown RSH coefficient convention");
}

void validate_rsh_convention(const LibrpaAngularOrder angular_order,
                             const LibrpaRshCoeff coeff_m_negative,
                             const LibrpaRshCoeff coeff_m_positive)
{
    validate_angular_order(angular_order);
    validate_rsh_coeff(coeff_m_negative);
    validate_rsh_coeff(coeff_m_positive);
}

struct RshCoeffPair
{
    double first;
    double second;
};

RshCoeffPair rsh_coeff_pair(const LibrpaRshCoeff coeff, const int m)
{
    validate_rsh_coeff(coeff);

    switch (coeff)
    {
    case LIBRPA_RSH_COEFF_1_M:
        return {1.0, alternating_sign(m)};
    case LIBRPA_RSH_COEFF_M_1:
        return {alternating_sign(m), 1.0};
    case LIBRPA_RSH_COEFF_UNSET:
        break;
    }
    throw std::invalid_argument("unknown RSH coefficient convention");
}

void clean_nearly_integer_entries(ComplexMatrix& matrix, const double tol = 1e-10)
{
    for (int row = 0; row < matrix.nr; ++row)
    {
        for (int col = 0; col < matrix.nc; ++col)
        {
            auto& value = matrix(row, col);
            if (std::abs(value.real() - std::round(value.real())) < tol)
            {
                value.real(std::round(value.real()));
            }
            if (std::abs(value.imag() - std::round(value.imag())) < tol)
            {
                value.imag(std::round(value.imag()));
            }
            if (std::abs(value.real()) < tol)
            {
                value.real(0.0);
            }
            if (std::abs(value.imag()) < tol)
            {
                value.imag(0.0);
            }
        }
    }
}

}

int rsh_abs_pm_m_to_index(const int m)
{
    return (m > 0) ? (2 * m - 1) : (-2 * m);
}

int rsh_m_to_index(const int l,
                   const int m,
                   const LibrpaAngularOrder angular_order)
{
    validate_angular_momentum(l, m, m);
    validate_angular_order(angular_order);

    switch (angular_order)
    {
    case LIBRPA_ANGULAR_ORDER_NATURAL:
        return wigner_m_to_index(l, m);
    case LIBRPA_ANGULAR_ORDER_ABS_PM:
        return rsh_abs_pm_m_to_index(m);
    case LIBRPA_ANGULAR_ORDER_OPENMX:
    case LIBRPA_ANGULAR_ORDER_PYSCF:
    case LIBRPA_ANGULAR_ORDER_UNSET:
        break;
    }
    throw std::invalid_argument("unknown angular order");
}

std::complex<double> complex_to_real_spherical_harmonic_overlap(
    const int l,
    const int m_complex,
    const int m_real,
    const LibrpaRshCoeff coeff_m_negative,
    const LibrpaRshCoeff coeff_m_positive)
{
    validate_angular_momentum(l, m_complex, m_real);
    validate_rsh_coeff(coeff_m_negative);
    validate_rsh_coeff(coeff_m_positive);

    if (m_real == 0)
    {
        return (m_complex == 0) ? std::complex<double>(1.0, 0.0)
                                : std::complex<double>(0.0, 0.0);
    }

    const double sqrt2 = std::sqrt(2.0);
    if (m_real > 0)
    {
        const auto coeff = rsh_coeff_pair(coeff_m_positive, m_real);
        if (m_complex == m_real)
        {
            return coeff.first / sqrt2;
        }
        if (m_complex == -m_real)
        {
            return coeff.second / sqrt2;
        }
        return 0.0;
    }

    const auto coeff = rsh_coeff_pair(coeff_m_negative, m_real);
    if (m_complex == m_real)
    {
        return C_IMAG * coeff.first / sqrt2;
    }
    if (m_complex == -m_real)
    {
        return -C_IMAG * coeff.second / sqrt2;
    }
    return 0.0;
}

ComplexMatrix complex_to_real_spherical_harmonic_transform(
    const int l,
    const LibrpaAngularOrder angular_order,
    const LibrpaRshCoeff coeff_m_negative,
    const LibrpaRshCoeff coeff_m_positive)
{
    if (l < 0)
    {
        throw std::invalid_argument("angular momentum l must be non-negative");
    }
    validate_rsh_convention(angular_order, coeff_m_negative, coeff_m_positive);

    const int nm = 2 * l + 1;
    ComplexMatrix transform(nm, nm);
    for (int m_complex = -l; m_complex <= l; ++m_complex)
    {
        for (int m_real = -l; m_real <= l; ++m_real)
        {
            transform(wigner_m_to_index(l, m_complex), rsh_m_to_index(l, m_real, angular_order)) =
                complex_to_real_spherical_harmonic_overlap(l,
                                                           m_complex,
                                                           m_real,
                                                           coeff_m_negative,
                                                           coeff_m_positive);
        }
    }
    return transform;
}

ComplexMatrix real_spherical_harmonic_rotation_matrix(
    const Vector3<double>& euler_angle,
    const int l,
    const LibrpaAngularOrder angular_order,
    const LibrpaRshCoeff coeff_m_negative,
    const LibrpaRshCoeff coeff_m_positive,
    const bool improper_rotation)
{
    if (l < 0)
    {
        throw std::invalid_argument("angular momentum l must be non-negative");
    }
    validate_rsh_convention(angular_order, coeff_m_negative, coeff_m_positive);
    if (l == 0)
    {
        ComplexMatrix identity(1, 1);
        identity(0, 0) = std::complex<double>(1.0, 0.0);
        return identity;
    }

    const ComplexMatrix transform = complex_to_real_spherical_harmonic_transform(
        l, angular_order, coeff_m_negative, coeff_m_positive);
    ComplexMatrix D_mm = wigner_D_matrix(euler_angle, l);
    if (improper_rotation)
    {
        D_mm *= std::complex<double>(std::pow(-1.0, l), 0.0);
    }

    ComplexMatrix rotation = transpose(transform, true) * D_mm * transform;
    clean_nearly_integer_entries(rotation);
    return rotation;
}

ComplexMatrix real_spherical_harmonic_rotation_matrix(
    const Matrix3& cartesian_rotation,
    const int l,
    const LibrpaAngularOrder angular_order,
    const LibrpaRshCoeff coeff_m_negative,
    const LibrpaRshCoeff coeff_m_positive,
    const double threshold)
{
    if (l < 0)
    {
        throw std::invalid_argument("angular momentum l must be non-negative");
    }
    validate_rsh_convention(angular_order, coeff_m_negative, coeff_m_positive);

    const bool improper_rotation = cartesian_rotation.Det() < 0.0;
    const Matrix3 proper_cartesian =
        improper_rotation ? (cartesian_rotation * Matrix3::NEGATIVE) : cartesian_rotation;
    const auto euler_angle = rotation_matrix_to_euler_angles_zyz(proper_cartesian, threshold);
    return real_spherical_harmonic_rotation_matrix(euler_angle, l, angular_order, coeff_m_negative,
                                                   coeff_m_positive, improper_rotation);
}

}  // namespace librpa_int
