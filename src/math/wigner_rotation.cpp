#include "wigner_rotation.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "../utils/constants.h"

namespace librpa_int {

namespace {

double factorial_as_double(const int n)
{
    if (n < 0)
    {
        throw std::invalid_argument("factorial argument must be non-negative");
    }
    return std::tgamma(static_cast<double>(n) + 1.0);
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

}

int wigner_m_to_index(const int l, const int m)
{
    validate_angular_momentum(l, m, 0);
    return m + l;
}

double wigner_small_d(const double beta, const int l, const int m1, const int m2)
{
    validate_angular_momentum(l, m1, m2);

    double value = 0.0;
    for (int i = std::max(0, m2 - m1); i <= std::min(l - m1, l + m2); ++i)
    {
        const double numerator =
            std::pow(-1.0, i)
            * std::sqrt(factorial_as_double(l + m1) * factorial_as_double(l - m1)
                        * factorial_as_double(l + m2) * factorial_as_double(l - m2))
            * std::pow(std::cos(beta / 2.0), 2 * l + m2 - m1 - 2 * i)
            * std::pow(-std::sin(beta / 2.0), m1 - m2 + 2 * i);
        const double denominator =
            factorial_as_double(i) * factorial_as_double(l - m1 - i)
            * factorial_as_double(l + m2 - i) * factorial_as_double(i - m2 + m1);
        value += numerator / denominator;
    }
    return value;
}

ComplexMatrix wigner_small_d_matrix(const double beta, const int l)
{
    if (l < 0)
    {
        throw std::invalid_argument("angular momentum l must be non-negative");
    }

    const int nm = 2 * l + 1;
    ComplexMatrix d_mm(nm, nm);
    for (int m1 = -l; m1 <= l; ++m1)
    {
        for (int m2 = -l; m2 <= l; ++m2)
        {
            d_mm(wigner_m_to_index(l, m1), wigner_m_to_index(l, m2)) =
                wigner_small_d(beta, l, m1, m2);
        }
    }
    return d_mm;
}

std::complex<double> wigner_D(const Vector3<double>& euler_angle,
                              const int l,
                              const int m1,
                              const int m2)
{
    validate_angular_momentum(l, m1, m2);

    return std::exp(-C_IMAG * static_cast<double>(m1) * euler_angle.x)
           * std::exp(-C_IMAG * static_cast<double>(m2) * euler_angle.z)
           * wigner_small_d(euler_angle.y, l, m1, m2);
}

ComplexMatrix wigner_D_matrix(const Vector3<double>& euler_angle, const int l)
{
    if (l < 0)
    {
        throw std::invalid_argument("angular momentum l must be non-negative");
    }

    const int nm = 2 * l + 1;
    ComplexMatrix D_mm(nm, nm);
    for (int m1 = -l; m1 <= l; ++m1)
    {
        for (int m2 = -l; m2 <= l; ++m2)
        {
            D_mm(wigner_m_to_index(l, m1), wigner_m_to_index(l, m2)) =
                wigner_D(euler_angle, l, m1, m2);
        }
    }
    return D_mm;
}

Vector3<double> rotation_matrix_to_euler_angles_zyz(const Matrix3& rotation_matrix,
                                                    const double threshold)
{
    double alpha = 0.0;
    double beta = 0.0;
    double gamma = 0.0;

    if (std::fabs(rotation_matrix.e32) > threshold || std::fabs(rotation_matrix.e31) > threshold)
    {
        alpha = std::atan2(rotation_matrix.e32, rotation_matrix.e31);
        if (alpha < 0.0)
        {
            alpha += TWO_PI;
        }
        gamma = std::atan2(rotation_matrix.e23, -rotation_matrix.e13);
        if (gamma < 0.0)
        {
            gamma += TWO_PI;
        }
        if (std::fabs(rotation_matrix.e32) > std::fabs(rotation_matrix.e31))
        {
            beta = std::atan2(rotation_matrix.e32 / std::sin(alpha), rotation_matrix.e33);
        }
        else
        {
            beta = std::atan2(rotation_matrix.e31 / std::cos(alpha), rotation_matrix.e33);
        }
    }
    else
    {
        alpha = std::atan2(rotation_matrix.e12, rotation_matrix.e11);
        if (alpha < 0.0)
        {
            alpha += TWO_PI;
        }
        if (rotation_matrix.e33 > 0.0)
        {
            beta = 0.0;
            gamma = 0.0;
        }
        else
        {
            beta = PI;
            gamma = PI;
        }
    }
    return {alpha, beta, gamma};
}

Matrix3 euler_angles_zyz_to_rotation_matrix(const Vector3<double>& euler_angles_zyz)
{
    const double alpha = euler_angles_zyz.x;
    const double beta = euler_angles_zyz.y;
    const double gamma = euler_angles_zyz.z;
    const double ca = std::cos(alpha);
    const double sa = std::sin(alpha);
    const double cb = std::cos(beta);
    const double sb = std::sin(beta);
    const double cg = std::cos(gamma);
    const double sg = std::sin(gamma);

    return {cg * cb * ca - sg * sa, cg * cb * sa + sg * ca, -cg * sb,
            -sg * cb * ca - cg * sa, -sg * cb * sa + cg * ca, sg * sb,
            sb * ca, sb * sa, cb};
}

}
