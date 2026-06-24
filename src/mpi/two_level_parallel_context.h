#pragma once

#include <string>
#include <utility>

#include "base_mpi.h"

namespace librpa_int
{

/*!
 * @brief Global-rank layout for a generic two-level decomposition.
 */
enum class TwoLevelRankLayout
{
    CONTIGUOUS_INNER,
    CONTIGUOUS_OUTER
};

/*!
 * @brief Shape of MPI processes assigned to two independent work levels.
 *
 * The total process count is nprocs_outer * nprocs_inner.  Ranks with the same
 * outer_group_id form comm_inner_h; ranks with the same inner_rank form
 * comm_outer_h.
 */
struct TwoLevelProcessShape
{
    static constexpr int AUTO = 0;

    int nprocs_outer;
    int nprocs_inner;

    TwoLevelProcessShape(int nprocs_outer_in = AUTO, int nprocs_inner_in = AUTO);

    bool auto_outer() const noexcept { return nprocs_outer == AUTO; }
    bool auto_inner() const noexcept { return nprocs_inner == AUTO; }
    bool has_auto() const noexcept { return auto_outer() || auto_inner(); }
    int total_nprocs() const noexcept;
    std::string info(const std::string &outer_name = "outer",
                     const std::string &inner_name = "inner") const;
};

TwoLevelProcessShape resolve_two_level_process_shape(const TwoLevelProcessShape &request,
                                                     int nprocs_global, int n_outer_items,
                                                     bool favor_square_inner = false);

std::pair<int, int> split_two_level_rank(int global_rank, const TwoLevelProcessShape &shape,
                                         TwoLevelRankLayout rank_layout);

int two_level_global_rank(const TwoLevelProcessShape &shape, TwoLevelRankLayout rank_layout,
                          int outer_group_id, int inner_rank);

class TwoLevelParallelContext
{
private:
    bool initialized_;
    TwoLevelProcessShape requested_process_shape_;
    TwoLevelProcessShape process_shape_;
    int outer_group_id_;
    int inner_rank_;
    TwoLevelRankLayout rank_layout_;

public:
    //! Wrapped global communicator. This handler does not own the communicator.
    MpiCommHandler comm_global_h;
    //! Communicator across outer groups for the same inner rank.
    MpiCommHandler comm_outer_h;
    //! Communicator inside one outer group.
    MpiCommHandler comm_inner_h;

    TwoLevelParallelContext();
    TwoLevelParallelContext(const TwoLevelProcessShape &process_shape, MPI_Comm comm_global,
                            TwoLevelRankLayout rank_layout = TwoLevelRankLayout::CONTIGUOUS_INNER);
    ~TwoLevelParallelContext() { finalize(); }

    TwoLevelParallelContext(const TwoLevelParallelContext &) = delete;
    TwoLevelParallelContext &operator=(const TwoLevelParallelContext &) = delete;

    void init(const TwoLevelProcessShape &process_shape, MPI_Comm comm_global,
              TwoLevelRankLayout rank_layout = TwoLevelRankLayout::CONTIGUOUS_INNER);
    void finalize();

    bool is_initialized() const noexcept { return initialized_; }
    const TwoLevelProcessShape &requested_process_shape() const noexcept
    {
        return requested_process_shape_;
    }
    const TwoLevelProcessShape &process_shape() const noexcept { return process_shape_; }
    int outer_group_id() const noexcept { return outer_group_id_; }
    int inner_rank() const noexcept { return inner_rank_; }
    TwoLevelRankLayout rank_layout() const noexcept { return rank_layout_; }
    int global_rank(int outer_group_id, int inner_rank) const;
    std::string info() const;
};

} /* end of namespace librpa_int */
