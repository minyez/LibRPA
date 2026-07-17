#ifndef DDLA_POTRF_BOTTOM_RIGHT_INTERNAL_H
#define DDLA_POTRF_BOTTOM_RIGHT_INTERNAL_H

#include <ddla/ddla_handle_t.h>

namespace ddla::detail {

// Factor one device-resident diagonal block from the bottom-right. With
// uplo='U', compute A = U * U^H; with uplo='L', compute A = L^H * L. The
// selected triangle is reverse-packed into d_work, factored with the opposite
// standard POTRF triangle, and mapped back. d_work holds at least n*n elements.
// d_info and d_work are caller-owned and may be reused across diagonal blocks.
template <typename T>
void potrf_bottom_right_block(
    const char& uplo, const int& n, T* d_A, const int& lda,
    const int& global_offset,
    T* d_work, int* d_info, int& info,
    const DdlaHandle_t& handle
);

} // namespace ddla::detail

#endif // DDLA_POTRF_BOTTOM_RIGHT_INTERNAL_H
