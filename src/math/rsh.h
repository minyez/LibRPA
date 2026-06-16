/*!
 * @file rsh.h
 * @brief Real spherical harmonic ordering and convention utilities.
 */
#pragma once

#include <complex>

#include "complexmatrix.h"
#include "librpa_enums.h"
#include "matrix3.h"
#include "vector3.h"

namespace librpa_int {

//! Map magnetic quantum number m to ABS_PM real-spherical shell index: 0, 1, -1, 2, -2, ...
int rsh_abs_pm_m_to_index(int m);

//! Map magnetic quantum number m to a real-spherical shell index for a supported angular order.
int rsh_m_to_index(int l,
                   int m,
                   LibrpaAngularOrder angular_order);

//! Overlap between complex spherical harmonics Y_lm_complex and real harmonics S_lm_real.
std::complex<double> complex_to_real_spherical_harmonic_overlap(
    int l,
    int m_complex,
    int m_real,
    LibrpaRshCoeff coeff_m_negative,
    LibrpaRshCoeff coeff_m_positive);

/*!
 * @brief Build the complex-to-real spherical-harmonic transform.
 *
 * Rows are complex spherical harmonics in natural Wigner order (-l, ..., l).
 * Columns are real spherical harmonics in the requested angular order.
 */
ComplexMatrix complex_to_real_spherical_harmonic_transform(
    int l,
    LibrpaAngularOrder angular_order,
    LibrpaRshCoeff coeff_m_negative,
    LibrpaRshCoeff coeff_m_positive);

//! Build the real-spherical-harmonic rotation matrix for angular momentum l.
ComplexMatrix real_spherical_harmonic_rotation_matrix(
    const Vector3<double>& euler_angle,
    int l,
    LibrpaAngularOrder angular_order,
    LibrpaRshCoeff coeff_m_negative,
    LibrpaRshCoeff coeff_m_positive,
    bool improper_rotation = false);

//! Build the real-spherical-harmonic rotation matrix from a Cartesian rotation matrix.
ComplexMatrix real_spherical_harmonic_rotation_matrix(
    const Matrix3& cartesian_rotation,
    int l,
    LibrpaAngularOrder angular_order,
    LibrpaRshCoeff coeff_m_negative,
    LibrpaRshCoeff coeff_m_positive,
    double threshold = 1e-5);

}  // namespace librpa_int
