#pragma once

#include <string>
#include <utility>
#include <vector>

#include "base_blacs.h"
#include "base_mpi.h"

namespace librpa_int
{

/*!
 * @brief Global-rank layout for the two-level k-point/BLACS decomposition.
 */
enum class KPointBlacsRankLayout
{
    CONTIGUOUS_BLACS,
    CONTIGUOUS_KPOINT
};

/*!
 * @brief Shape of MPI processes assigned to k-point and BLACS parallelism.
 *
 * The total process count is nprocs_kpoint * nprocs_blacs.  The first level
 * creates k-point groups.  Inside each k-point group, nprocs_blacs processes
 * form the BLACS context used for NAO/ABF basis matrices.
 * favor_square_blacs_grid controls whether the BLACS process grid should
 * prioritize a square-like grid over matching the matrix aspect ratio.
 *
 * Set either field to AUTO to let KPointBlacsParallelContext infer it from the
 * global communicator size, the number of k-points and matrix dimensions.  In
 * the fully automatic mode, the resolver prefers balanced k-point groups: if
 * there is only one k-point, all ranks are assigned to one BLACS context.
 */
struct KPointBlacsProcessShape
{
    static constexpr int AUTO = 0;

    int nprocs_kpoint;
    int nprocs_blacs;
    bool favor_square_blacs_grid;

    KPointBlacsProcessShape(int nprocs_kpoint_in = AUTO, int nprocs_blacs_in = AUTO,
                            bool favor_square_blacs_grid_in = false);

    bool auto_kpoint() const noexcept { return nprocs_kpoint == AUTO; }
    bool auto_blacs() const noexcept { return nprocs_blacs == AUTO; }
    bool has_auto() const noexcept { return auto_kpoint() || auto_blacs(); }
    int total_nprocs() const noexcept;
    std::string info() const;
};

/*!
 * @brief Resolve requested k-point/BLACS process shape against an MPI size.
 */
KPointBlacsProcessShape resolve_kpoint_blacs_process_shape(const KPointBlacsProcessShape &request,
                                                           int nprocs_global, int n_kpoints,
                                                           int matrix_nrows, int matrix_ncols);

//! Square-matrix shorthand for resolve_kpoint_blacs_process_shape.
KPointBlacsProcessShape resolve_kpoint_blacs_process_shape(const KPointBlacsProcessShape &request,
                                                           int nprocs_global, int n_kpoints,
                                                           int matrix_size);

/*!
 * @brief Choose a BLACS grid shaped for the input matrix dimensions.
 */
std::pair<int, int> choose_kpoint_blacs_grid(int nprocs_blacs, int matrix_nrows, int matrix_ncols,
                                             bool more_rows = true, bool favor_square_grid = false);

//! Square-matrix shorthand for choose_kpoint_blacs_grid.
std::pair<int, int> choose_kpoint_blacs_grid(int nprocs_blacs, int matrix_size,
                                             bool more_rows = true, bool favor_square_grid = false);

/*!
 * @brief Handler for k-point-level MPI communicators and basis-level BLACS.
 *
 * Global ranks are interpreted as a 2D process layout.  With the default
 * CONTIGUOUS_BLACS layout:
 *
 *   global_rank = kpoint_group_id * nprocs_blacs + blacs_rank
 *
 * With CONTIGUOUS_KPOINT layout:
 *
 *   global_rank = blacs_rank * nprocs_kpoint + kpoint_group_id
 *
 * This creates two useful communicators:
 * - comm_blacs_h: processes with the same kpoint_group_id; used by BLACS.
 * - comm_kpoint_h: processes with the same blacs_rank across k-point groups.
 */
class KPointBlacsParallelContext
{
private:
    bool initialized_;
    KPointBlacsProcessShape requested_process_shape_;
    KPointBlacsProcessShape process_shape_;
    int n_kpoints_;
    int matrix_nrows_;
    int matrix_ncols_;
    int kpoint_group_id_;
    int blacs_rank_;
    int blacs_nprows_;
    int blacs_npcols_;
    CTXT_LAYOUT blacs_layout_;
    KPointBlacsRankLayout rank_layout_;
    std::vector<int> kpoints_local_;

public:
    //! Wrapped global communicator. This handler does not own the communicator.
    MpiCommHandler comm_global_h;
    //! Communicator among k-point groups for the same BLACS rank.
    MpiCommHandler comm_kpoint_h;
    //! Communicator inside one k-point group, used to create the BLACS context.
    MpiCommHandler comm_blacs_h;
    //! BLACS context over comm_blacs_h.
    BlacsCtxtHandler blacs_h;

    KPointBlacsParallelContext();
    KPointBlacsParallelContext(
        const KPointBlacsProcessShape &process_shape, MPI_Comm comm_global, int n_kpoints,
        int matrix_nrows, int matrix_ncols, CTXT_LAYOUT blacs_layout = CTXT_LAYOUT::R,
        KPointBlacsRankLayout rank_layout = KPointBlacsRankLayout::CONTIGUOUS_BLACS);
    ~KPointBlacsParallelContext() { finalize(); }

    KPointBlacsParallelContext(const KPointBlacsParallelContext &) = delete;
    KPointBlacsParallelContext &operator=(const KPointBlacsParallelContext &) = delete;

    void init(const KPointBlacsProcessShape &process_shape, MPI_Comm comm_global, int n_kpoints,
              int matrix_nrows, int matrix_ncols, CTXT_LAYOUT blacs_layout = CTXT_LAYOUT::R,
              KPointBlacsRankLayout rank_layout = KPointBlacsRankLayout::CONTIGUOUS_BLACS);
    void init(const KPointBlacsProcessShape &process_shape, MPI_Comm comm_global, int n_kpoints,
              int matrix_size, CTXT_LAYOUT blacs_layout = CTXT_LAYOUT::R,
              KPointBlacsRankLayout rank_layout = KPointBlacsRankLayout::CONTIGUOUS_BLACS);
    void finalize();

    bool is_initialized() const noexcept { return initialized_; }

    const KPointBlacsProcessShape &requested_process_shape() const noexcept
    {
        return requested_process_shape_;
    }
    const KPointBlacsProcessShape &process_shape() const noexcept { return process_shape_; }
    int n_kpoints() const noexcept { return n_kpoints_; }
    int matrix_nrows() const noexcept { return matrix_nrows_; }
    int matrix_ncols() const noexcept { return matrix_ncols_; }
    int kpoint_group_id() const noexcept { return kpoint_group_id_; }
    int blacs_rank() const noexcept { return blacs_rank_; }
    int blacs_nprows() const noexcept { return blacs_nprows_; }
    int blacs_npcols() const noexcept { return blacs_npcols_; }
    KPointBlacsRankLayout rank_layout() const noexcept { return rank_layout_; }
    const std::vector<int> &kpoints_local() const noexcept { return kpoints_local_; }

    bool owns_kpoint(int ik) const;
    int kpoint_owner(int ik) const;
    ArrayDesc create_array_desc(int matrix_nrows, int matrix_ncols,
                                int mb = KPointBlacsProcessShape::AUTO,
                                int nb = KPointBlacsProcessShape::AUTO, int irsrc = 0,
                                int icsrc = 0) const;
    std::string info() const;
};

} /* end of namespace librpa_int */
