#include <cassert>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../io/stl_io_helper.h"
#include "../mpi/global_mpi.h"
#include "../mpi/kpoint_blacs_parallel_context.h"
#include "testutils.h"

int expected_global_rank(librpa_int::KPointBlacsRankLayout rank_layout, int nprocs_kpoint,
                         int nprocs_blacs, int kpoint_group_id, int blacs_rank)
{
    switch (rank_layout)
    {
        case librpa_int::KPointBlacsRankLayout::CONTIGUOUS_BLACS:
            return kpoint_group_id * nprocs_blacs + blacs_rank;
        case librpa_int::KPointBlacsRankLayout::CONTIGUOUS_KPOINT:
            return blacs_rank * nprocs_kpoint + kpoint_group_id;
    }
    return -1;
}

void test_resolve_process_shape()
{
    using namespace librpa_int;

    {
        const auto resolved = resolve_kpoint_blacs_process_shape(
            {KPointBlacsProcessShape::AUTO, KPointBlacsProcessShape::AUTO, true}, 4, 3, 8, 3);
        assert(resolved.nprocs_kpoint == 1);
        assert(resolved.nprocs_blacs == 4);
        assert(resolved.favor_square_blacs_grid);
    }
    {
        const auto resolved = resolve_kpoint_blacs_process_shape(
            {KPointBlacsProcessShape::AUTO, KPointBlacsProcessShape::AUTO}, 4, 3, 8, 3);
        assert(resolved.nprocs_kpoint == 1);
        assert(resolved.nprocs_blacs == 4);
    }
    {
        const auto resolved = resolve_kpoint_blacs_process_shape(
            {KPointBlacsProcessShape::AUTO, KPointBlacsProcessShape::AUTO}, 8, 2, 1, 8);
        assert(resolved.nprocs_kpoint == 2);
        assert(resolved.nprocs_blacs == 4);
    }
    {
        const auto resolved = resolve_kpoint_blacs_process_shape(
            {KPointBlacsProcessShape::AUTO, KPointBlacsProcessShape::AUTO}, 4, 8, 8, 3);
        assert(resolved.nprocs_kpoint == 4);
        assert(resolved.nprocs_blacs == 1);
    }
    {
        const auto resolved = resolve_kpoint_blacs_process_shape(
            {KPointBlacsProcessShape::AUTO, KPointBlacsProcessShape::AUTO}, 16, 6, 8, 3);
        assert(resolved.nprocs_kpoint == 2);
        assert(resolved.nprocs_blacs == 8);
    }
    {
        const auto resolved = resolve_kpoint_blacs_process_shape(
            {KPointBlacsProcessShape::AUTO, KPointBlacsProcessShape::AUTO, true}, 16, 6, 8, 3);
        assert(resolved.nprocs_kpoint == 1);
        assert(resolved.nprocs_blacs == 16);
    }
    {
        const auto resolved =
            resolve_kpoint_blacs_process_shape({2, KPointBlacsProcessShape::AUTO}, 4, 3, 8);
        assert(resolved.nprocs_kpoint == 2);
        assert(resolved.nprocs_blacs == 2);
    }
    {
        const auto resolved =
            resolve_kpoint_blacs_process_shape({KPointBlacsProcessShape::AUTO, 2}, 4, 3, 8);
        assert(resolved.nprocs_kpoint == 2);
        assert(resolved.nprocs_blacs == 2);
    }
    {
        bool caught = false;
        try
        {
            resolve_kpoint_blacs_process_shape({3, 2}, 4, 3, 8);
        }
        catch (const std::runtime_error &)
        {
            caught = true;
        }
        assert(caught);
    }
}

void test_blacs_grid_choice()
{
    using namespace librpa_int;

    assert(choose_kpoint_blacs_grid(4, 1, 8, true) == std::make_pair(1, 4));
    assert(choose_kpoint_blacs_grid(4, 8, 1, true) == std::make_pair(4, 1));
    assert(choose_kpoint_blacs_grid(4, 1, 8, true, true) == std::make_pair(2, 2));
    assert(choose_kpoint_blacs_grid(4, 8, 1, true, true) == std::make_pair(2, 2));
    assert(choose_kpoint_blacs_grid(2, 8, true) == std::make_pair(2, 1));
    assert(choose_kpoint_blacs_grid(2, 8, false) == std::make_pair(1, 2));
    assert(choose_kpoint_blacs_grid(4, 8, true) == std::make_pair(2, 2));
}

void test_kpoint_blacs_parallel_context()
{
    using namespace librpa_int;

    const int world_rank = get_mpi_rank(MPI_COMM_WORLD);
    KPointBlacsParallelContext context({2, 2}, MPI_COMM_WORLD, 3, 8, 3);

    assert(context.is_initialized());
    assert(context.process_shape().nprocs_kpoint == 2);
    assert(context.process_shape().nprocs_blacs == 2);
    assert(context.kpoint_group_id() == world_rank / 2);
    assert(context.blacs_rank() == world_rank % 2);

    assert(context.comm_blacs_h.nprocs == 2);
    assert(context.comm_blacs_h.myid == world_rank % 2);
    assert(context.comm_kpoint_h.nprocs == 2);
    assert(context.comm_kpoint_h.myid == world_rank / 2);

    assert(context.blacs_h.nprocs == 2);
    assert(context.blacs_h.nprows == 2);
    assert(context.blacs_h.npcols == 1);

    const auto desc_default = context.create_array_desc(8, 3);
    assert(desc_default.is_initialized());
    assert(desc_default.m() == 8);
    assert(desc_default.n() == 3);
    assert(desc_default.mb() == 4);
    assert(desc_default.nb() == 3);
    assert(desc_default.is_row_consec());
    assert(desc_default.is_col_consec());

    const auto desc_custom = context.create_array_desc(8, 3, 2, 1);
    assert(desc_custom.is_initialized());
    assert(desc_custom.mb() == 2);
    assert(desc_custom.nb() == 1);

    if (context.kpoint_group_id() == 0)
    {
        const std::vector<int> ref{0, 1};
        assert(equal_vector(context.kpoints_local(), ref));
        assert(context.owns_kpoint(0));
        assert(context.owns_kpoint(1));
        assert(!context.owns_kpoint(2));
    }
    else
    {
        const std::vector<int> ref{2};
        assert(equal_vector(context.kpoints_local(), ref));
        assert(!context.owns_kpoint(0));
        assert(!context.owns_kpoint(1));
        assert(context.owns_kpoint(2));
    }

    assert(context.kpoint_owner(0) == 0);
    assert(context.kpoint_owner(1) == 0);
    assert(context.kpoint_owner(2) == 1);

    context.finalize();
    assert(!context.is_initialized());
}

void test_kpoint_contiguous_rank_layout()
{
    using namespace librpa_int;

    const int world_rank = get_mpi_rank(MPI_COMM_WORLD);
    KPointBlacsParallelContext context({2, 2}, MPI_COMM_WORLD, 4, 8, 3, CTXT_LAYOUT::R,
                                       KPointBlacsRankLayout::CONTIGUOUS_KPOINT);

    assert(context.is_initialized());
    assert(context.rank_layout() == KPointBlacsRankLayout::CONTIGUOUS_KPOINT);
    assert(context.process_shape().nprocs_kpoint == 2);
    assert(context.process_shape().nprocs_blacs == 2);
    assert(context.kpoint_group_id() == world_rank % 2);
    assert(context.blacs_rank() == world_rank / 2);

    assert(context.comm_blacs_h.nprocs == 2);
    assert(context.comm_blacs_h.myid == world_rank / 2);
    assert(context.comm_kpoint_h.nprocs == 2);
    assert(context.comm_kpoint_h.myid == world_rank % 2);

    if (context.kpoint_group_id() == 0)
    {
        const std::vector<int> ref{0, 1};
        assert(equal_vector(context.kpoints_local(), ref));
    }
    else
    {
        const std::vector<int> ref{2, 3};
        assert(equal_vector(context.kpoints_local(), ref));
    }

    context.finalize();
    assert(!context.is_initialized());
}

void test_two_level_communicators(librpa_int::KPointBlacsRankLayout rank_layout)
{
    using namespace librpa_int;

    const int world_rank = get_mpi_rank(MPI_COMM_WORLD);
    KPointBlacsParallelContext context({2, 2}, MPI_COMM_WORLD, 4, 8, 3, CTXT_LAYOUT::R,
                                       rank_layout);

    const auto &shape = context.process_shape();
    for (int i = 0; i < context.comm_global_h.nprocs; i++)
    {
        if (context.comm_global_h.myid == i)
        {
            std::cout << context.comm_global_h.myid << " " << context.comm_kpoint_h.myid << " "
                      << context.comm_blacs_h.myid << " " << context.kpoints_local() << std::endl;
        }
        context.comm_global_h.barrier();
        // same comm_blacs_h.myid should have the same kpoints_local.
    }

    std::vector<int> blacs_world_ranks(context.comm_blacs_h.nprocs);
    context.comm_blacs_h.allgather(&world_rank, 1, blacs_world_ranks.data(), 1);
    for (int iblacs = 0; iblacs != context.comm_blacs_h.nprocs; ++iblacs)
    {
        assert(blacs_world_ranks[iblacs] ==
               expected_global_rank(rank_layout, shape.nprocs_kpoint, shape.nprocs_blacs,
                                    context.kpoint_group_id(), iblacs));
    }

    const double local_blacs_value = 10.0 * context.kpoint_group_id() + context.blacs_rank() + 1.0;
    double blacs_sum = 0.0;
    context.comm_blacs_h.allreduce(&local_blacs_value, &blacs_sum, 1, MPI_SUM);
    const double expected_blacs_sum =
        shape.nprocs_blacs * (10.0 * context.kpoint_group_id() + 1.0) +
        shape.nprocs_blacs * (shape.nprocs_blacs - 1) / 2.0;
    assert(blacs_sum == expected_blacs_sum);

    std::vector<int> kpoint_world_ranks(context.comm_kpoint_h.nprocs);
    context.comm_kpoint_h.allgather(&world_rank, 1, kpoint_world_ranks.data(), 1);
    for (int ikgroup = 0; ikgroup != context.comm_kpoint_h.nprocs; ++ikgroup)
    {
        assert(kpoint_world_ranks[ikgroup] == expected_global_rank(rank_layout, shape.nprocs_kpoint,
                                                                   shape.nprocs_blacs, ikgroup,
                                                                   context.blacs_rank()));
    }

    const std::complex<double> local_kpoint_block(context.kpoint_group_id() + 1.0,
                                                  context.blacs_rank() + 0.25);
    std::complex<double> kpoint_block_sum(0.0, 0.0);
    context.comm_kpoint_h.allreduce(&local_kpoint_block, &kpoint_block_sum, 1, MPI_SUM);
    const std::complex<double> expected_kpoint_block_sum(
        shape.nprocs_kpoint * (shape.nprocs_kpoint + 1) / 2.0,
        shape.nprocs_kpoint * (context.blacs_rank() + 0.25));
    assert(kpoint_block_sum == expected_kpoint_block_sum);

    context.finalize();
    assert(!context.is_initialized());
}

void test_square_blacs_grid_preference()
{
    using namespace librpa_int;

    KPointBlacsParallelContext context(
        {KPointBlacsProcessShape::AUTO, KPointBlacsProcessShape::AUTO, true}, MPI_COMM_WORLD, 1, 1,
        8);

    assert(context.is_initialized());
    assert(context.process_shape().nprocs_kpoint == 1);
    assert(context.process_shape().nprocs_blacs == 4);
    assert(context.process_shape().favor_square_blacs_grid);
    assert(context.blacs_h.nprows == 2);
    assert(context.blacs_h.npcols == 2);

    context.finalize();
    assert(!context.is_initialized());
}

int main(int argc, char *argv[])
{
    using namespace librpa_int::global;

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    init_global_mpi(MPI_COMM_WORLD);

    if (size_global != 4) throw std::runtime_error("test imposes 4 MPI processes");

    test_resolve_process_shape();
    test_blacs_grid_choice();
    test_kpoint_blacs_parallel_context();
    test_kpoint_contiguous_rank_layout();
    test_two_level_communicators(librpa_int::KPointBlacsRankLayout::CONTIGUOUS_BLACS);
    test_two_level_communicators(librpa_int::KPointBlacsRankLayout::CONTIGUOUS_KPOINT);
    test_square_blacs_grid_preference();

    finalize_global_mpi();
    MPI_Finalize();

    return 0;
}
