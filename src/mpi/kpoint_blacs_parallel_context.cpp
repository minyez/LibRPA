#include "kpoint_blacs_parallel_context.h"

#include <algorithm>
#include <cstdlib>
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

static inline void check_requested_process_shape(const KPointBlacsProcessShape &request)
{
    if (request.nprocs_kpoint < 0) throw LIBRPA_RUNTIME_ERROR("nprocs_kpoint must be non-negative");
    if (request.nprocs_blacs < 0) throw LIBRPA_RUNTIME_ERROR("nprocs_blacs must be non-negative");
}

static int min_contiguous_block_size(int global_size, int nprocs)
{
    check_positive("global_size", global_size);
    check_positive("nprocs", nprocs);
    return (global_size + nprocs - 1) / nprocs;
}

static bool is_square_number(int n)
{
    check_positive("n", n);
    int root = 1;
    while (root <= n / root && root * root < n) ++root;
    return root <= n / root && root * root == n;
}

static KPointBlacsProcessShape make_process_shape(int nprocs_kpoint, int nprocs_blacs,
                                                  const KPointBlacsProcessShape &request)
{
    return {nprocs_kpoint, nprocs_blacs, request.favor_square_blacs_grid};
}

static std::pair<int, int> split_global_rank(int global_rank, const KPointBlacsProcessShape &shape,
                                             KPointBlacsRankLayout rank_layout)
{
    switch (rank_layout)
    {
        case KPointBlacsRankLayout::CONTIGUOUS_BLACS:
            return {global_rank / shape.nprocs_blacs, global_rank % shape.nprocs_blacs};
        case KPointBlacsRankLayout::CONTIGUOUS_KPOINT:
            return {global_rank % shape.nprocs_kpoint, global_rank / shape.nprocs_kpoint};
    }
    throw LIBRPA_RUNTIME_ERROR("unknown k-point/BLACS rank layout");
}

static const char *rank_layout_name(KPointBlacsRankLayout rank_layout)
{
    switch (rank_layout)
    {
        case KPointBlacsRankLayout::CONTIGUOUS_BLACS:
            return "CONTIGUOUS_BLACS";
        case KPointBlacsRankLayout::CONTIGUOUS_KPOINT:
            return "CONTIGUOUS_KPOINT";
    }
    return "UNKNOWN";
}

static bool sequential_kpoint_distribution(KPointDistribution kpoint_distribution)
{
    switch (kpoint_distribution)
    {
        case KPointDistribution::CYCLIC:
            return false;
        case KPointDistribution::CONTIGUOUS:
            return true;
    }
    throw LIBRPA_RUNTIME_ERROR("unknown k-point distribution");
}

static const char *kpoint_distribution_name(KPointDistribution kpoint_distribution)
{
    switch (kpoint_distribution)
    {
        case KPointDistribution::CYCLIC:
            return "CYCLIC";
        case KPointDistribution::CONTIGUOUS:
            return "CONTIGUOUS";
    }
    return "UNKNOWN";
}

} /* end anonymous namespace */

KPointBlacsProcessShape::KPointBlacsProcessShape(int nprocs_kpoint_in, int nprocs_blacs_in,
                                                 bool favor_square_blacs_grid_in)
    : nprocs_kpoint(nprocs_kpoint_in),
      nprocs_blacs(nprocs_blacs_in),
      favor_square_blacs_grid(favor_square_blacs_grid_in)
{
    check_requested_process_shape(*this);
}

int KPointBlacsProcessShape::total_nprocs() const noexcept
{
    if (has_auto()) return AUTO;
    return nprocs_kpoint * nprocs_blacs;
}

std::string KPointBlacsProcessShape::info() const
{
    std::ostringstream oss;
    oss << "KPointBlacsProcessShape: "
        << "nprocs_kpoint ";
    if (auto_kpoint())
        oss << "AUTO";
    else
        oss << nprocs_kpoint;
    oss << " nprocs_blacs ";
    if (auto_blacs())
        oss << "AUTO";
    else
        oss << nprocs_blacs;
    oss << " favor_square_blacs_grid " << (favor_square_blacs_grid ? "T" : "F");
    return oss.str();
}

KPointBlacsProcessShape resolve_kpoint_blacs_process_shape(const KPointBlacsProcessShape &request,
                                                           int nprocs_global, int n_kpoints)
{
    check_requested_process_shape(request);
    check_positive("nprocs_global", nprocs_global);
    check_positive("n_kpoints", n_kpoints);

    if (!request.auto_kpoint() && request.nprocs_kpoint > n_kpoints)
    {
        throw LIBRPA_RUNTIME_ERROR(
            "requested k-point process groups exceed the number of k-points");
    }

    if (!request.has_auto())
    {
        const auto total = static_cast<long long>(request.nprocs_kpoint) *
                           static_cast<long long>(request.nprocs_blacs);
        if (total != nprocs_global)
        {
            throw LIBRPA_RUNTIME_ERROR(
                "nprocs_kpoint * nprocs_blacs must equal the global MPI size");
        }
        return request;
    }

    if (!request.auto_kpoint())
    {
        if (nprocs_global % request.nprocs_kpoint != 0)
        {
            throw LIBRPA_RUNTIME_ERROR(
                "global MPI size is not divisible by requested nprocs_kpoint");
        }
        return make_process_shape(request.nprocs_kpoint, nprocs_global / request.nprocs_kpoint,
                                  request);
    }

    if (!request.auto_blacs())
    {
        if (nprocs_global % request.nprocs_blacs != 0)
        {
            throw LIBRPA_RUNTIME_ERROR(
                "global MPI size is not divisible by requested nprocs_blacs");
        }
        const int nprocs_kpoint = nprocs_global / request.nprocs_blacs;
        if (nprocs_kpoint > n_kpoints)
        {
            throw LIBRPA_RUNTIME_ERROR(
                "requested nprocs_blacs creates more k-point groups than k-points");
        }
        return make_process_shape(nprocs_kpoint, request.nprocs_blacs, request);
    }

    if (n_kpoints >= nprocs_global)
    {
        return make_process_shape(nprocs_global, 1, request);
    }

    // Maximize exact balanced k-point groups first; then optionally reduce the
    // group count only if that keeps k-point ownership balanced and gives a
    // square number of BLACS ranks.
    const int max_balanced_kpoint_groups = std::gcd(n_kpoints, nprocs_global);
    if (request.favor_square_blacs_grid)
    {
        for (int nprocs_kpoint = max_balanced_kpoint_groups; nprocs_kpoint >= 1; --nprocs_kpoint)
        {
            if (max_balanced_kpoint_groups % nprocs_kpoint != 0) continue;
            const int nprocs_blacs = nprocs_global / nprocs_kpoint;
            if (is_square_number(nprocs_blacs))
            {
                return make_process_shape(nprocs_kpoint, nprocs_blacs, request);
            }
        }
    }

    return make_process_shape(max_balanced_kpoint_groups,
                              nprocs_global / max_balanced_kpoint_groups, request);
}

KPointBlacsProcessShape resolve_kpoint_blacs_process_shape(const KPointBlacsProcessShape &request,
                                                           int nprocs_global, int n_kpoints,
                                                           int matrix_nrows, int matrix_ncols)
{
    check_positive("matrix_nrows", matrix_nrows);
    check_positive("matrix_ncols", matrix_ncols);
    return resolve_kpoint_blacs_process_shape(request, nprocs_global, n_kpoints);
}

KPointBlacsProcessShape resolve_kpoint_blacs_process_shape(const KPointBlacsProcessShape &request,
                                                           int nprocs_global, int n_kpoints,
                                                           int matrix_size)
{
    check_positive("matrix_size", matrix_size);
    return resolve_kpoint_blacs_process_shape(request, nprocs_global, n_kpoints);
}

std::pair<int, int> choose_kpoint_blacs_grid(int nprocs_blacs, int matrix_nrows, int matrix_ncols,
                                             bool more_rows, bool favor_square_grid)
{
    check_positive("nprocs_blacs", nprocs_blacs);
    check_positive("matrix_nrows", matrix_nrows);
    check_positive("matrix_ncols", matrix_ncols);

    int best_nprows = 1;
    int best_npcols = nprocs_blacs;
    int best_overflow =
        std::max(0, best_nprows - matrix_nrows) + std::max(0, best_npcols - matrix_ncols);
    long long best_shape_mismatch = std::llabs(static_cast<long long>(best_nprows) * matrix_ncols -
                                               static_cast<long long>(best_npcols) * matrix_nrows);
    int best_balance = std::abs(best_nprows - best_npcols);
    int best_orientation_penalty = more_rows && best_nprows < best_npcols ? 1 : 0;
    if (!more_rows && best_npcols < best_nprows) best_orientation_penalty = 1;

    for (int nprows = 1; nprows <= nprocs_blacs; ++nprows)
    {
        if (nprocs_blacs % nprows != 0) continue;
        const int npcols = nprocs_blacs / nprows;
        const int overflow =
            std::max(0, nprows - matrix_nrows) + std::max(0, npcols - matrix_ncols);
        const long long shape_mismatch = std::llabs(static_cast<long long>(nprows) * matrix_ncols -
                                                    static_cast<long long>(npcols) * matrix_nrows);
        const int balance = std::abs(nprows - npcols);
        int orientation_penalty = more_rows && nprows < npcols ? 1 : 0;
        if (!more_rows && npcols < nprows) orientation_penalty = 1;

        bool is_better;
        if (favor_square_grid)
        {
            is_better = balance < best_balance ||
                        (balance == best_balance && overflow < best_overflow) ||
                        (balance == best_balance && overflow == best_overflow &&
                         shape_mismatch < best_shape_mismatch) ||
                        (balance == best_balance && overflow == best_overflow &&
                         shape_mismatch == best_shape_mismatch &&
                         orientation_penalty < best_orientation_penalty);
        }
        else
        {
            is_better = overflow < best_overflow ||
                        (overflow == best_overflow && shape_mismatch < best_shape_mismatch) ||
                        (overflow == best_overflow && shape_mismatch == best_shape_mismatch &&
                         balance < best_balance) ||
                        (overflow == best_overflow && shape_mismatch == best_shape_mismatch &&
                         balance == best_balance && orientation_penalty < best_orientation_penalty);
        }

        if (is_better)
        {
            best_nprows = nprows;
            best_npcols = npcols;
            best_overflow = overflow;
            best_shape_mismatch = shape_mismatch;
            best_balance = balance;
            best_orientation_penalty = orientation_penalty;
        }
    }

    return {best_nprows, best_npcols};
}

std::pair<int, int> choose_kpoint_blacs_grid(int nprocs_blacs, int matrix_size, bool more_rows,
                                             bool favor_square_grid)
{
    return choose_kpoint_blacs_grid(nprocs_blacs, matrix_size, matrix_size, more_rows,
                                    favor_square_grid);
}

KPointBlacsParallelContext::KPointBlacsParallelContext()
    : initialized_(false),
      requested_process_shape_(),
      process_shape_(),
      n_kpoints_(0),
      kpoint_group_id_(0),
      blacs_rank_(0),
      blacs_nprows_(0),
      blacs_npcols_(0),
      blacs_layout_(CTXT_LAYOUT::R),
      rank_layout_(KPointBlacsRankLayout::CONTIGUOUS_BLACS),
      kpoint_distribution_(KPointDistribution::CYCLIC),
      kpoints_local_(),
      comm_global_h(),
      comm_kpoint_h(),
      comm_blacs_h(),
      blacs_h()
{
}

KPointBlacsParallelContext::KPointBlacsParallelContext(const KPointBlacsProcessShape &process_shape,
                                                       MPI_Comm comm_global, int n_kpoints,
                                                       CTXT_LAYOUT blacs_layout,
                                                       KPointBlacsRankLayout rank_layout,
                                                       KPointDistribution kpoint_distribution)
    : KPointBlacsParallelContext()
{
    init(process_shape, comm_global, n_kpoints, blacs_layout, rank_layout, kpoint_distribution);
}

void KPointBlacsParallelContext::init(const KPointBlacsProcessShape &process_shape,
                                      MPI_Comm comm_global, int n_kpoints, CTXT_LAYOUT blacs_layout,
                                      KPointBlacsRankLayout rank_layout,
                                      KPointDistribution kpoint_distribution)
{
    if (initialized_) finalize();

    requested_process_shape_ = process_shape;
    n_kpoints_ = n_kpoints;
    blacs_layout_ = blacs_layout;
    rank_layout_ = rank_layout;
    kpoint_distribution_ = kpoint_distribution;

    comm_global_h.reset_comm(comm_global, true);
    process_shape_ =
        resolve_kpoint_blacs_process_shape(process_shape, comm_global_h.nprocs, n_kpoints_);

    const auto split_rank = split_global_rank(comm_global_h.myid, process_shape_, rank_layout_);
    kpoint_group_id_ = split_rank.first;
    blacs_rank_ = split_rank.second;

    MPI_Comm comm_blacs = MPI_COMM_NULL;
    MPI_Comm comm_kpoint = MPI_COMM_NULL;
    int ierr = MPI_Comm_split(comm_global_h.comm, kpoint_group_id_, blacs_rank_, &comm_blacs);
    if (ierr != MPI_SUCCESS)
        throw LIBRPA_RUNTIME_ERROR("failed to create BLACS-level communicator");
    ierr = MPI_Comm_split(comm_global_h.comm, blacs_rank_, kpoint_group_id_, &comm_kpoint);
    if (ierr != MPI_SUCCESS)
    {
        MPI_Comm_free(&comm_blacs);
        throw LIBRPA_RUNTIME_ERROR("failed to create k-point-level communicator");
    }

    comm_blacs_h.reset_comm(comm_blacs, true);
    comm_kpoint_h.reset_comm(comm_kpoint, true);

    blacs_h.reset_comm(comm_blacs_h.comm, true);
    const auto blacs_grid =
        choose_kpoint_blacs_grid(process_shape_.nprocs_blacs, process_shape_.nprocs_blacs, true,
                                 process_shape_.favor_square_blacs_grid);
    blacs_nprows_ = blacs_grid.first;
    blacs_npcols_ = blacs_grid.second;
    blacs_h.set_grid(blacs_nprows_, blacs_npcols_, blacs_layout_);

    kpoints_local_ = dispatcher(0, n_kpoints_, kpoint_group_id_, process_shape_.nprocs_kpoint,
                                sequential_kpoint_distribution(kpoint_distribution_));

    initialized_ = true;
}

void KPointBlacsParallelContext::finalize()
{
    if (!initialized_) return;

    blacs_h.reset_comm();
    comm_kpoint_h.free_comm();
    comm_blacs_h.free_comm();
    comm_global_h.reset_comm();

    process_shape_ = {};
    n_kpoints_ = 0;
    kpoint_group_id_ = 0;
    blacs_rank_ = 0;
    blacs_nprows_ = 0;
    blacs_npcols_ = 0;
    rank_layout_ = KPointBlacsRankLayout::CONTIGUOUS_BLACS;
    kpoint_distribution_ = KPointDistribution::CYCLIC;
    kpoints_local_.clear();
    initialized_ = false;
}

bool KPointBlacsParallelContext::owns_kpoint(int ik) const
{
    return std::find(kpoints_local_.begin(), kpoints_local_.end(), ik) != kpoints_local_.end();
}

int KPointBlacsParallelContext::kpoint_owner(int ik) const
{
    if (ik < 0 || ik >= n_kpoints_) throw LIBRPA_RUNTIME_ERROR("k-point index out of range");

    for (int group_id = 0; group_id < process_shape_.nprocs_kpoint; ++group_id)
    {
        const auto kpoints = dispatcher(0, n_kpoints_, group_id, process_shape_.nprocs_kpoint,
                                        sequential_kpoint_distribution(kpoint_distribution_));
        if (std::find(kpoints.begin(), kpoints.end(), ik) != kpoints.end()) return group_id;
    }
    throw LIBRPA_RUNTIME_ERROR("failed to find k-point owner");
}

ArrayDesc KPointBlacsParallelContext::create_array_desc(int matrix_nrows, int matrix_ncols, int mb,
                                                        int nb, int irsrc, int icsrc) const
{
    if (!initialized_) throw LIBRPA_RUNTIME_ERROR("KPointBlacsParallelContext not initialized");
    check_positive("matrix_nrows", matrix_nrows);
    check_positive("matrix_ncols", matrix_ncols);
    if (mb < 0 || nb < 0) throw LIBRPA_RUNTIME_ERROR("BLACS block sizes must be non-negative");

    const int mb_eff = mb == KPointBlacsProcessShape::AUTO
                           ? min_contiguous_block_size(matrix_nrows, blacs_h.nprows)
                           : mb;
    const int nb_eff = nb == KPointBlacsProcessShape::AUTO
                           ? min_contiguous_block_size(matrix_ncols, blacs_h.npcols)
                           : nb;
    check_positive("mb", mb_eff);
    check_positive("nb", nb_eff);

    ArrayDesc desc(blacs_h);
    desc.init(matrix_nrows, matrix_ncols, mb_eff, nb_eff, irsrc, icsrc);
    return desc;
}

std::string KPointBlacsParallelContext::info() const
{
    std::ostringstream oss;
    oss << "KPointBlacsParallelContext: "
        << "initialized " << (initialized_ ? "T" : "F") << " "
        << "process_shape [" << process_shape_.info() << "] "
        << "kpoint_group_id " << kpoint_group_id_ << " "
        << "blacs_rank " << blacs_rank_ << " "
        << "n_kpoints " << n_kpoints_ << " "
        << "rank_layout " << rank_layout_name(rank_layout_) << " "
        << "kpoint_distribution " << kpoint_distribution_name(kpoint_distribution_) << " "
        << "blacs_grid (" << blacs_nprows_ << "," << blacs_npcols_ << ")";
    return oss.str();
}

} /* end of namespace librpa_int */
