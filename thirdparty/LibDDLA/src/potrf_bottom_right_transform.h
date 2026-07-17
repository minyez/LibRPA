#ifndef DDLA_POTRF_BOTTOM_RIGHT_TRANSFORM_H
#define DDLA_POTRF_BOTTOM_RIGHT_TRANSFORM_H

#include <ddla/ddla_connector.h>

namespace ddla::detail {

// Reverse-map the upper triangle of A into the lower triangle of B as
// B = J A J, where J reverses both row and column order.
template <typename T>
void reverse_upper_to_lower(
    int n, const T* d_A, int lda,
    T* d_B, int ldb, deviceStream_t stream);

// Reverse-map the lower triangle of B into the upper triangle of A. Only the
// upper triangle of A is written. The same transform is also used in the
// opposite data-flow direction when packing a lower input into upper scratch.
template <typename T>
void reverse_lower_to_upper(
    int n, const T* d_B, int ldb,
    T* d_A, int lda, deviceStream_t stream);

} // namespace ddla::detail

#endif // DDLA_POTRF_BOTTOM_RIGHT_TRANSFORM_H
