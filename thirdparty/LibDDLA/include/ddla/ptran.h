#ifndef DDLA_PTRAN_H
#define DDLA_PTRAN_H

#include <ddla/ddla_desc.h>

namespace ddla{

/**
 * @brief Out-of-place distributed matrix transpose.
 *
 * Computes d_AT = d_A^T (non-conjugate transpose) when conj == false, or
 * d_AT = d_A^H (conjugate transpose) when conj == true.  The descriptor
 * descAT must describe the transposed matrix on the same process grid as
 * descA with swapped block sizes and source process indices:
 *   descAT.m()  == descA.n()
 *   descAT.n()  == descA.m()
 *   descAT.mb() == descA.nb()
 *   descAT.nb() == descA.mb()
 *   descAT.irsrc() == descA.icsrc()
 *   descAT.icsrc() == descA.irsrc()
 */
template <typename T>
void ptran(const T* d_A, const DdlaDesc& descA,
           T* d_AT, const DdlaDesc& descAT,
           bool conj = false);

} // namespace ddla

#endif // DDLA_PTRAN_H
