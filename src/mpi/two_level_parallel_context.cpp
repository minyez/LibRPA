#include "two_level_parallel_context.h"

#include <numeric>
#include <sstream>

#include "../utils/error.h"

namespace librpa_int
{

namespace
{

static inline void check_positive(const char *name, int value)
{
    if (value <= 0) throw LIBRPA_RUNTIME_ERROR(std::string(name) + " must be positive");
}

static inline void check_requested_two_level_process_shape(const TwoLevelProcessShape &request)
{
    if (request.nprocs_outer < 0) throw LIBRPA_RUNTIME_ERROR("nprocs_outer must be non-negative");
    if (request.nprocs_inner < 0) throw LIBRPA_RUNTIME_ERROR("nprocs_inner must be non-negative");
}

static bool is_square_number(int n)
{
    check_positive("n", n);
    int root = 1;
    while (root <= n / root && root * root < n) ++root;
    return root <= n / root && root * root == n;
}

static TwoLevelProcessShape make_two_level_process_shape(int nprocs_outer, int nprocs_inner)
{
    return {nprocs_outer, nprocs_inner};
}

static const char *two_level_rank_layout_name(TwoLevelRankLayout rank_layout)
{
    switch (rank_layout)
    {
        case TwoLevelRankLayout::CONTIGUOUS_INNER:
            return "CONTIGUOUS_INNER";
        case TwoLevelRankLayout::CONTIGUOUS_OUTER:
            return "CONTIGUOUS_OUTER";
    }
    return "UNKNOWN";
}

} /* end anonymous namespace */

TwoLevelProcessShape::TwoLevelProcessShape(int nprocs_outer_in, int nprocs_inner_in)
    : nprocs_outer(nprocs_outer_in), nprocs_inner(nprocs_inner_in)
{
    check_requested_two_level_process_shape(*this);
}

int TwoLevelProcessShape::total_nprocs() const noexcept
{
    if (has_auto()) return AUTO;
    return nprocs_outer * nprocs_inner;
}

std::string TwoLevelProcessShape::info(const std::string &outer_name,
                                       const std::string &inner_name) const
{
    std::ostringstream oss;
    oss << "TwoLevelProcessShape: " << outer_name << " ";
    if (auto_outer())
        oss << "AUTO";
    else
        oss << nprocs_outer;
    oss << " " << inner_name << " ";
    if (auto_inner())
        oss << "AUTO";
    else
        oss << nprocs_inner;
    return oss.str();
}

TwoLevelProcessShape resolve_two_level_process_shape(const TwoLevelProcessShape &request,
                                                     int nprocs_global, int n_outer_items,
                                                     bool favor_square_inner)
{
    check_requested_two_level_process_shape(request);
    check_positive("nprocs_global", nprocs_global);
    check_positive("n_outer_items", n_outer_items);

    if (!request.auto_outer() && request.nprocs_outer > n_outer_items)
    {
        throw LIBRPA_RUNTIME_ERROR(
            "requested outer process groups exceed the number of outer work items");
    }

    if (!request.has_auto())
    {
        const auto total = static_cast<long long>(request.nprocs_outer) *
                           static_cast<long long>(request.nprocs_inner);
        if (total != nprocs_global)
        {
            throw LIBRPA_RUNTIME_ERROR(
                "nprocs_outer * nprocs_inner must equal the global MPI size");
        }
        return request;
    }

    if (!request.auto_outer())
    {
        if (nprocs_global % request.nprocs_outer != 0)
        {
            throw LIBRPA_RUNTIME_ERROR(
                "global MPI size is not divisible by requested nprocs_outer");
        }
        return make_two_level_process_shape(request.nprocs_outer,
                                            nprocs_global / request.nprocs_outer);
    }

    if (!request.auto_inner())
    {
        if (nprocs_global % request.nprocs_inner != 0)
        {
            throw LIBRPA_RUNTIME_ERROR(
                "global MPI size is not divisible by requested nprocs_inner");
        }
        const int nprocs_outer = nprocs_global / request.nprocs_inner;
        if (nprocs_outer > n_outer_items)
        {
            throw LIBRPA_RUNTIME_ERROR(
                "requested nprocs_inner creates more outer groups than outer work items");
        }
        return make_two_level_process_shape(nprocs_outer, request.nprocs_inner);
    }

    if (n_outer_items >= nprocs_global)
    {
        return make_two_level_process_shape(nprocs_global, 1);
    }

    const int max_balanced_outer_groups = std::gcd(n_outer_items, nprocs_global);
    if (favor_square_inner)
    {
        for (int nprocs_outer = max_balanced_outer_groups; nprocs_outer >= 1; --nprocs_outer)
        {
            if (max_balanced_outer_groups % nprocs_outer != 0) continue;
            const int nprocs_inner = nprocs_global / nprocs_outer;
            if (is_square_number(nprocs_inner))
            {
                return make_two_level_process_shape(nprocs_outer, nprocs_inner);
            }
        }
    }

    return make_two_level_process_shape(max_balanced_outer_groups,
                                        nprocs_global / max_balanced_outer_groups);
}

std::pair<int, int> split_two_level_rank(int global_rank, const TwoLevelProcessShape &shape,
                                         TwoLevelRankLayout rank_layout)
{
    switch (rank_layout)
    {
        case TwoLevelRankLayout::CONTIGUOUS_INNER:
            return {global_rank / shape.nprocs_inner, global_rank % shape.nprocs_inner};
        case TwoLevelRankLayout::CONTIGUOUS_OUTER:
            return {global_rank % shape.nprocs_outer, global_rank / shape.nprocs_outer};
    }
    throw LIBRPA_RUNTIME_ERROR("unknown two-level rank layout");
}

int two_level_global_rank(const TwoLevelProcessShape &shape, TwoLevelRankLayout rank_layout,
                          int outer_group_id, int inner_rank)
{
    if (outer_group_id < 0 || outer_group_id >= shape.nprocs_outer)
        throw LIBRPA_RUNTIME_ERROR("outer group id out of range");
    if (inner_rank < 0 || inner_rank >= shape.nprocs_inner)
        throw LIBRPA_RUNTIME_ERROR("inner rank out of range");

    switch (rank_layout)
    {
        case TwoLevelRankLayout::CONTIGUOUS_INNER:
            return outer_group_id * shape.nprocs_inner + inner_rank;
        case TwoLevelRankLayout::CONTIGUOUS_OUTER:
            return inner_rank * shape.nprocs_outer + outer_group_id;
    }
    throw LIBRPA_RUNTIME_ERROR("unknown two-level rank layout");
}

TwoLevelParallelContext::TwoLevelParallelContext()
    : initialized_(false),
      requested_process_shape_(),
      process_shape_(),
      outer_group_id_(0),
      inner_rank_(0),
      rank_layout_(TwoLevelRankLayout::CONTIGUOUS_INNER),
      comm_global_h(),
      comm_outer_h(),
      comm_inner_h()
{
}

TwoLevelParallelContext::TwoLevelParallelContext(const TwoLevelProcessShape &process_shape,
                                                 MPI_Comm comm_global,
                                                 TwoLevelRankLayout rank_layout)
    : TwoLevelParallelContext()
{
    init(process_shape, comm_global, rank_layout);
}

void TwoLevelParallelContext::init(const TwoLevelProcessShape &process_shape,
                                   MPI_Comm comm_global, TwoLevelRankLayout rank_layout)
{
    if (initialized_) finalize();
    if (process_shape.has_auto())
        throw LIBRPA_RUNTIME_ERROR("two-level process shape must be resolved before init");

    requested_process_shape_ = process_shape;
    process_shape_ = process_shape;
    rank_layout_ = rank_layout;

    comm_global_h.reset_comm(comm_global, true);
    if (process_shape_.total_nprocs() != comm_global_h.nprocs)
        throw LIBRPA_RUNTIME_ERROR("two-level process shape does not match global MPI size");

    const auto split_rank = split_two_level_rank(comm_global_h.myid, process_shape_, rank_layout_);
    outer_group_id_ = split_rank.first;
    inner_rank_ = split_rank.second;

    MPI_Comm comm_inner = MPI_COMM_NULL;
    MPI_Comm comm_outer = MPI_COMM_NULL;
    int ierr = MPI_Comm_split(comm_global_h.comm, outer_group_id_, inner_rank_, &comm_inner);
    if (ierr != MPI_SUCCESS)
        throw LIBRPA_RUNTIME_ERROR("failed to create inner two-level communicator");
    ierr = MPI_Comm_split(comm_global_h.comm, inner_rank_, outer_group_id_, &comm_outer);
    if (ierr != MPI_SUCCESS)
    {
        MPI_Comm_free(&comm_inner);
        throw LIBRPA_RUNTIME_ERROR("failed to create outer two-level communicator");
    }

    comm_inner_h.reset_comm(comm_inner, true);
    comm_outer_h.reset_comm(comm_outer, true);
    initialized_ = true;
}

void TwoLevelParallelContext::finalize()
{
    if (!initialized_) return;

    comm_outer_h.free_comm();
    comm_inner_h.free_comm();
    comm_global_h.reset_comm();

    requested_process_shape_ = {};
    process_shape_ = {};
    outer_group_id_ = 0;
    inner_rank_ = 0;
    rank_layout_ = TwoLevelRankLayout::CONTIGUOUS_INNER;
    initialized_ = false;
}

int TwoLevelParallelContext::global_rank(int outer_group_id, int inner_rank) const
{
    if (!initialized_) throw LIBRPA_RUNTIME_ERROR("TwoLevelParallelContext not initialized");
    return two_level_global_rank(process_shape_, rank_layout_, outer_group_id, inner_rank);
}

std::string TwoLevelParallelContext::info() const
{
    std::ostringstream oss;
    oss << "TwoLevelParallelContext: "
        << "initialized " << (initialized_ ? "T" : "F") << " "
        << "process_shape [" << process_shape_.info() << "] "
        << "outer_group_id " << outer_group_id_ << " "
        << "inner_rank " << inner_rank_ << " "
        << "rank_layout " << two_level_rank_layout_name(rank_layout_);
    return oss.str();
}

} /* end of namespace librpa_int */
