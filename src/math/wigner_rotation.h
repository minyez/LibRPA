/*!
 * @file wigner_rotation.h
 * @brief Wigner rotation matrices for angular-momentum shells.
 */
#pragma once

#include <complex>

#include "complexmatrix.h"
#include "matrix3.h"
#include "vector3.h"

namespace librpa_int {

//! Map magnetic quantum number m to natural Wigner matrix index: -l, ..., l.
int wigner_m_to_index(int l, int m);

//! Compute one Wigner small-d matrix element d^l_{m1,m2}(beta).
double wigner_small_d(const double beta, const int l, const int m1, const int m2);

//! Build the Wigner small-d matrix for angular momentum l.
ComplexMatrix wigner_small_d_matrix(const double beta, const int l);

//! Compute one complex Wigner D matrix element D^l_{m1,m2}(alpha,beta,gamma).
std::complex<double> wigner_D(const Vector3<double>& euler_angle,
                              const int l,
                              const int m1,
                              const int m2);

//! Build the complex Wigner D matrix for angular momentum l.
ComplexMatrix wigner_D_matrix(const Vector3<double>& euler_angle,
                              const int l);

//! Get ZYZ Euler angles from a proper rotation.
//!
//! The rotation is represented by the rotation matrix transforming the coordinate system
//! when it acts upon a vector, i.e. an active rotation. Here we use Cartesian coordinates.
//! With vector v = (e_x, e_y, e_z) (x, y, z)^T
//!
//! \hat{R} v = v' = (e_x, e_y, e_z) (x', y', z')^T = (e_x, e_y, e_z) [R (x, y, z)^T]
//!                = (e'_x, e'_y, e'_z) (x, y, z)^T = [(e_x, e_y, e_z) R] (x, y, z)^T
//!                                                 = [\hat{R} (e_x, e_y, e_z)] (x, y, z)^T
//!
//! `rotation_matrix` here is R^T.
//!
Vector3<double> rotation_matrix_to_euler_angles_zyz(const Matrix3& rotation_matrix,
                                                    const double threshold = 1e-5);

//! Get a proper Cartesian rotation matrix from ZYZ Euler angles.
Matrix3 euler_angles_zyz_to_rotation_matrix(const Vector3<double>& euler_angles_zyz);

}
