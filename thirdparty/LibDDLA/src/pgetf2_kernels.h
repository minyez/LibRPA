#ifndef DDLA_PGETF2_KERNELS_H
#define DDLA_PGETF2_KERNELS_H

#include <cstddef>

#include <ddla/ddla_connector.h>

namespace ddla::detail {

template <typename T>
std::size_t pgetf2_pivot_workspace_size();

// Returns a zero-based index relative to d_column. The call synchronizes
// stream once because the MPI pivot reduction consumes the host result.
template <typename T>
void pgetf2_find_local_pivot(const T* d_column, int length,
                             void* d_workspace, deviceStream_t stream,
                             double& metric, int& local_index, T& value);

template <typename T>
void pgetf2_scale_update(int length_row, int length_col,
                         const T& inverse_pivot,
                         T* d_column, const T* d_pivot_row,
                         T* d_trailing, int lld,
                         deviceStream_t stream);

} // namespace ddla::detail

#endif // DDLA_PGETF2_KERNELS_H
