#include "meanfield_mpi.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <set>
#include <unordered_map>

#include "../io/stl_io_helper.h"
#include "../math/lapack_connector.h"
#include "../math/matrix_m.h"
#include "../math/scalapack_connector.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../utils/constants.h"
#include "../utils/profiler.h"
#include "atomic_basis.h"
#include "librpa_enums.h"
#include "symmetry_context.h"
#include "utils_atomic_basis_blacs.h"
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
#include "../gpu/device_connector.h"
#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#endif

namespace librpa_int
{

static void collect_Rs(const std::vector<Vector3_Order<int>> &Rs, std::vector<int> &n_Rs_all,
                       std::vector<int> &Rs_all, int &nR_max, const MpiCommHandler &comm_h)
{
    n_Rs_all.resize(comm_h.nprocs);
    for (int pid = 0; pid < comm_h.nprocs; pid++) n_Rs_all[pid] = 0;
    const int n_Rs_this = Rs.size();
    n_Rs_all[comm_h.myid] = n_Rs_this;
    // global::ofs_myid << global::myid_global << " " << global::size_global << std::endl;
    // comm_h.barrier();
    MPI_Allreduce(MPI_IN_PLACE, n_Rs_all.data(), comm_h.nprocs, mpi_datatype<int>::value, MPI_SUM, comm_h.comm);
    nR_max = *std::max_element(n_Rs_all.cbegin(), n_Rs_all.cend());
    Rs_all.resize(3 * nR_max * comm_h.nprocs);
    for (int iR = 0; iR < n_Rs_this; iR++)
    {
        Rs_all[comm_h.myid * nR_max * 3 + iR * 3] = Rs[iR].x;
        Rs_all[comm_h.myid * nR_max * 3 + iR * 3 + 1] = Rs[iR].y;
        Rs_all[comm_h.myid * nR_max * 3 + iR * 3 + 2] = Rs[iR].z;
    }
    MPI_Allreduce(MPI_IN_PLACE, Rs_all.data(), comm_h.nprocs * nR_max * 3, mpi_datatype<int>::value, MPI_SUM, comm_h.comm);
}

static void check_same_imagtimes(const std::vector<double> &imagtimes,
                                 const MpiCommHandler &comm_h)
{
    const int n_tau = imagtimes.size();
    int n_tau_min = 0, n_tau_max = 0;
    MPI_Allreduce(&n_tau, &n_tau_min, 1, mpi_datatype<int>::value, MPI_MIN, comm_h.comm);
    MPI_Allreduce(&n_tau, &n_tau_max, 1, mpi_datatype<int>::value, MPI_MAX, comm_h.comm);
    if (n_tau_min != n_tau_max)
        throw LIBRPA_RUNTIME_ERROR("imaginary-time lists differ in k-point BLACS context");
    if (n_tau == 0) return;

    std::vector<double> imagtimes_all(n_tau * comm_h.nprocs);
    MPI_Allgather(imagtimes.data(), n_tau, mpi_datatype<double>::value,
                  imagtimes_all.data(), n_tau, mpi_datatype<double>::value, comm_h.comm);
    for (int pid = 0; pid != comm_h.nprocs; ++pid)
    {
        for (int it = 0; it != n_tau; ++it)
        {
            if (imagtimes_all[pid * n_tau + it] != imagtimes[it])
                throw LIBRPA_RUNTIME_ERROR("imaginary-time lists differ in k-point BLACS context");
        }
    }
}

static std::vector<const SymmetryKAtomRotation*> build_rotations_by_from(
    const SymmetryKStarMember &member, const std::size_t n_atoms)
{
    std::vector<const SymmetryKAtomRotation*> rotations(n_atoms, nullptr);
    for (const auto &rotation : member.atom_rotations)
    {
        if (rotation.atom_from < 0 || rotation.atom_from >= static_cast<int>(n_atoms))
        {
            throw LIBRPA_RUNTIME_ERROR("k-star atom rotation source is out of range");
        }
        rotations[static_cast<std::size_t>(rotation.atom_from)] = &rotation;
    }
    for (const auto *rotation : rotations)
    {
        if (rotation == nullptr)
        {
            throw LIBRPA_RUNTIME_ERROR("k-star atom rotations do not cover every atom");
        }
    }
    return rotations;
}

static std::unordered_map<int, std::vector<atpair_t>> build_source_pair_requests(
    const std::unordered_map<int, std::set<atpair_t>> &target_pairs,
    const SymmetryKStarMember &member,
    const std::size_t n_atoms)
{
    const auto rotations = build_rotations_by_from(member, n_atoms);
    std::unordered_map<int, std::vector<atpair_t>> requests;
    for (const auto &[pid, pairs] : target_pairs)
    {
        std::set<atpair_t> source_pairs;
        for (const auto &pair : pairs)
        {
            const auto *rot_i = rotations.at(static_cast<std::size_t>(pair.first));
            const auto *rot_j = rotations.at(static_cast<std::size_t>(pair.second));
            source_pairs.insert({static_cast<atom_t>(rot_i->atom_to),
                                 static_cast<atom_t>(rot_j->atom_to)});
        }
        requests[pid] = std::vector<atpair_t>(source_pairs.cbegin(), source_pairs.cend());
    }
    return requests;
}

static ComplexMatrix matz_to_complex_matrix(const Matz &mat)
{
    ComplexMatrix block(mat.nr(), mat.nc());
    for (int i = 0; i != mat.nr(); ++i)
    {
        for (int j = 0; j != mat.nc(); ++j)
        {
            block(i, j) = mat(i, j);
        }
    }
    return block;
}

static Matz complex_matrix_to_colmajor_matz(const ComplexMatrix &block)
{
    Matz mat(block.nr, block.nc, MAJOR::COL);
    for (int i = 0; i != block.nr; ++i)
    {
        for (int j = 0; j != block.nc; ++j)
        {
            mat(i, j) = block(i, j);
        }
    }
    return mat;
}

constexpr int wfc_gemm_block_size_opt = 128;

struct WfcGemmWorkspace
{
    ArrayDesc desc_wfc_opt;
    ArrayDesc desc_out_opt;
    Matz wfc_bra_opt;
    Matz scaled_wfc_ket_opt;
    Matz out_opt;
};

static void check_wfc_gemm_context(const ArrayDesc &desc_wfc, const ArrayDesc &desc_out,
                                   const BlacsCtxtHandler &blacs_h)
{
    if (!blacs_h.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("active BLACS context is not initialized");
    if (!desc_wfc.is_initialized() || !desc_out.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("WFC pgemm descriptors are not initialized");
    if (desc_wfc.ictxt() != blacs_h.ictxt || desc_out.ictxt() != blacs_h.ictxt)
        throw LIBRPA_RUNTIME_ERROR("WFC pgemm descriptors must use the active k-point BLACS context");
}

static WfcGemmWorkspace create_wfc_gemm_workspace(const ArrayDesc &desc_wfc,
                                                  const ArrayDesc &desc_out,
                                                  const BlacsCtxtHandler &blacs_h)
{
    check_wfc_gemm_context(desc_wfc, desc_out, blacs_h);

    WfcGemmWorkspace workspace;
    const int block_ao = get_capped_blacs_block_size(
        desc_wfc.m(), wfc_gemm_block_size_opt, blacs_h);
    const int block_state = get_capped_blacs_block_size(
        desc_wfc.n(), wfc_gemm_block_size_opt, blacs_h);
    workspace.desc_wfc_opt = ArrayDesc(blacs_h);
    workspace.desc_wfc_opt.init(desc_wfc.m(), desc_wfc.n(),
                                block_ao, block_state,
                                desc_wfc.irsrc(), desc_wfc.icsrc());
    workspace.desc_out_opt = ArrayDesc(blacs_h);
    workspace.desc_out_opt.init(desc_out.m(), desc_out.n(),
                                block_ao, block_ao,
                                desc_out.irsrc(), desc_out.icsrc());
    workspace.wfc_bra_opt.resize(workspace.desc_wfc_opt.m_loc(),
                                 workspace.desc_wfc_opt.n_loc(), MAJOR::COL);
    workspace.scaled_wfc_ket_opt.resize(workspace.desc_wfc_opt.m_loc(),
                                        workspace.desc_wfc_opt.n_loc(), MAJOR::COL);
    workspace.out_opt.resize(workspace.desc_out_opt.m_loc(),
                             workspace.desc_out_opt.n_loc(), MAJOR::COL);
    return workspace;
}

static void pgemm_wfc_scaled_wfc_h(const int n_aos, const int n_cols,
                                   const cplxdb *wfc_bra_ptr,
                                   const cplxdb *scaled_wfc_ket_ptr,
                                   const ArrayDesc &desc_wfc,
                                   WfcGemmWorkspace &workspace,
                                   Matz &out, const ArrayDesc &desc_out,
                                   const BlacsCtxtHandler &blacs_h)
{
    global::profiler.start(__FUNCTION__, LIBRPA_VERBOSE_DEBUG);
    check_wfc_gemm_context(desc_wfc, desc_out, blacs_h);
    if (n_aos != desc_wfc.m() || n_aos != desc_out.m() || n_aos != desc_out.n())
        throw LIBRPA_RUNTIME_ERROR("WFC pgemm matrix dimensions are inconsistent");
    if (n_cols < 0 || n_cols > desc_wfc.n())
        throw LIBRPA_RUNTIME_ERROR("WFC pgemm column count is inconsistent with descriptor");
    if (!workspace.desc_wfc_opt.is_initialized() || !workspace.desc_out_opt.is_initialized() ||
        workspace.desc_wfc_opt.ictxt() != blacs_h.ictxt ||
        workspace.desc_out_opt.ictxt() != blacs_h.ictxt)
        throw LIBRPA_RUNTIME_ERROR("WFC pgemm workspace is not initialized on the active k-point BLACS context");

    ScalapackConnector::pgemr2d_f(n_aos, n_cols,
                                  wfc_bra_ptr, 1, 1, desc_wfc.desc,
                                  workspace.wfc_bra_opt.ptr(), 1, 1,
                                  workspace.desc_wfc_opt.desc, blacs_h.ictxt);
    ScalapackConnector::pgemr2d_f(n_aos, n_cols,
                                  scaled_wfc_ket_ptr, 1, 1, desc_wfc.desc,
                                  workspace.scaled_wfc_ket_opt.ptr(), 1, 1,
                                  workspace.desc_wfc_opt.desc, blacs_h.ictxt);

#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    using namespace ddla;
    auto &blacs_h_nc = const_cast<BlacsCtxtHandler &>(blacs_h);
    if (blacs_h_nc.ddla_handle == nullptr)
        blacs_h_nc.init_ddla_handle();
    workspace.desc_wfc_opt.set_ddla_desc(blacs_h_nc.ddla_handle);
    workspace.desc_out_opt.set_ddla_desc(blacs_h_nc.ddla_handle);

    auto handle = blacs_h_nc.ddla_handle;
    const size_t size_ab = static_cast<size_t>(workspace.desc_wfc_opt.m_loc()) *
                           workspace.desc_wfc_opt.n_loc();
    const size_t size_c  = static_cast<size_t>(workspace.desc_out_opt.m_loc()) *
                           workspace.desc_out_opt.n_loc();
    const size_t alloc_size_ab = std::max<size_t>(size_ab, 1);
    const size_t alloc_size_c = std::max<size_t>(size_c, 1);
    cplxdb *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    DEVICE_CHECK(deviceMallocAsync((void**)&d_A, alloc_size_ab * sizeof(cplxdb), handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_B, alloc_size_ab * sizeof(cplxdb), handle->stream));
    DEVICE_CHECK(deviceMallocAsync((void**)&d_C, alloc_size_c * sizeof(cplxdb), handle->stream));
    if (size_ab > 0)
    {
        DEVICE_CHECK(deviceMemcpyAsync(d_A, workspace.wfc_bra_opt.ptr(),
                                       size_ab * sizeof(cplxdb),
                                       deviceMemcpyHostToDevice, handle->stream));
        DEVICE_CHECK(deviceMemcpyAsync(d_B, workspace.scaled_wfc_ket_opt.ptr(),
                                       size_ab * sizeof(cplxdb),
                                       deviceMemcpyHostToDevice, handle->stream));
    }
    ddla::pgemm('N', 'C', n_aos, n_aos, n_cols, C_ONE,
                d_A, workspace.desc_wfc_opt.ddla_desc(),
                d_B, workspace.desc_wfc_opt.ddla_desc(),
                C_ZERO, d_C, workspace.desc_out_opt.ddla_desc());
    if (size_c > 0)
        DEVICE_CHECK(deviceMemcpyAsync(workspace.out_opt.ptr(), d_C,
                                       size_c * sizeof(cplxdb),
                                       deviceMemcpyDeviceToHost, handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_A, handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_B, handle->stream));
    DEVICE_CHECK(deviceFreeAsync(d_C, handle->stream));
#else
    global::profiler.start("pgemm", LIBRPA_VERBOSE_DEBUG);
    ScalapackConnector::pgemm_f('N', 'C', n_aos, n_aos, n_cols, C_ONE,
                                workspace.wfc_bra_opt.ptr(), 1, 1,
                                workspace.desc_wfc_opt.desc,
                                workspace.scaled_wfc_ket_opt.ptr(), 1, 1,
                                workspace.desc_wfc_opt.desc,
                                C_ZERO, workspace.out_opt.ptr(), 1, 1,
                                workspace.desc_out_opt.desc);
    global::profiler.stop("pgemm");
#endif
    ScalapackConnector::pgemr2d_f(n_aos, n_aos,
                                  workspace.out_opt.ptr(), 1, 1,
                                  workspace.desc_out_opt.desc,
                                  out.ptr(), 1, 1, desc_out.desc,
                                  blacs_h.ictxt);
    global::profiler.stop(__FUNCTION__);
}

std::map<Vector3_Order<int>, ComplexMatrix> get_dmat_cplx_Rs_kpara(
    int ispin, int ispinor_bra, int ispinor_ket, const MeanField &mf, const std::vector<Vector3_Order<double>>& kfrac_list,
    const std::vector<Vector3_Order<int>>& Rs, const MpiCommHandler &comm_h)
{
    global::profiler.start(__FUNCTION__);
    std::map<Vector3_Order<int>, ComplexMatrix> dmat_local;

    // Collect Rs requested by each process
    std::vector<int> n_Rs_all, Rs_all;
    int nR_max;
    collect_Rs(Rs, n_Rs_all, Rs_all, nR_max, comm_h);
    // global::ofs_myid << "get_dmat_cplx_Rs_kpara nRs_all " << n_Rs_all << std::endl;

    const int n_aos = mf.get_n_aos();
    const size_t size = n_aos * n_aos;

    const auto iks_local = mf.get_iks_local();
    // global::ofs_myid << "iks_local " << iks_local << std::endl;
    const int nk_local = iks_local.size();
    // Check if there is duplicate k-point data
    int nk_local_sum;
    MPI_Allreduce(&nk_local, &nk_local_sum, 1, MPI_INT, MPI_SUM, comm_h.comm);
    if (nk_local_sum > mf.get_n_kpoints())
        throw LIBRPA_RUNTIME_ERROR("found duplicated k-point eigenvectors data");
    else if (nk_local_sum < mf.get_n_kpoints())
        throw LIBRPA_RUNTIME_ERROR("missing k-point eigenvectors data");

    Matz kmat(size, nk_local, MAJOR::COL);
    Matz transmat(nk_local, nR_max, MAJOR::COL);
    Matz rmat(size, nR_max, MAJOR::COL);

    // NOTE: Support dimension (number of basis) up to 46300 (Gamma-only) or 1930 (k 8x8x8) due to range of int of axpy.
    //       Need division into batches for larger system, particularly for memory consideration
    for (int ik_local = 0; ik_local < nk_local; ik_local++)
    {
        const auto dmat_k = mf.get_dmat_cplx(ispin, ispinor_bra, ispinor_ket, iks_local[ik_local]);
        memcpy(kmat.ptr() + size * ik_local, dmat_k.c, size * sizeof(Matz::type));
    }

    for (int pid = 0; pid < comm_h.nprocs; pid++)
    {
        const auto nR_this = n_Rs_all[pid];
        // NOTE: MPI_Reduce requires all processes use the same sendcount, but rmat is simply zero where nk_local == 0.
        const size_t count = size * nR_this;
        // global::ofs_myid << "get_dmat_cplx_Rs_kpara pid " << pid << " nR_this " << nR_this << " count " << count << " matsize " << rmat.size() << std::endl;

        if (nR_this < 1) continue;
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int iR = 0; iR < nR_this; iR++)
        {
            for (int ik = 0; ik < nk_local; ik++)
            {
                const int index = pid * nR_max * 3 + iR * 3;
                const auto &kf = kfrac_list[iks_local[ik]];
                auto ang = - (kf.x * Rs_all[index] + kf.y * Rs_all[index+1] + kf.z * Rs_all[index+2]) * TWO_PI;
                transmat(ik, iR) = cplxdb{cos(ang), sin(ang)};
            }
        }
        // global::ofs_myid << "transmat pid" << std::endl;
        // global::ofs_myid << transmat << std::endl;
        rmat = 0.0;
        if (nk_local > 0)
        {
            LapackConnector::gemm_f('N', 'N', size, nR_this, nk_local, 1.0,
                                    kmat.ptr(), size, transmat.ptr(), nk_local, 0.0, rmat.ptr(), size);
        }
        // global::ofs_myid << rmat << std::endl;
        if (comm_h.myid == pid)
        {
            MPI_Reduce(MPI_IN_PLACE, rmat.ptr(), count, mpi_datatype<Matz::type>::value, MPI_SUM, pid, comm_h.comm);
            for (int iR = 0; iR < nR_this; iR++)
            {
                const int index = pid * nR_max * 3 + iR * 3;
                ComplexMatrix m(n_aos, n_aos);
                memcpy(m.c, rmat.ptr() + size * iR, size * sizeof(Matz::type));
                Vector3_Order<int> R{Rs_all[index], Rs_all[index+1], Rs_all[index+2]};
                // global::ofs_myid << "iR " << iR << " R " << R << std::endl;
                dmat_local.emplace(R, std::move(m));
            }
        }
        else
        {
            MPI_Reduce(rmat.ptr(), rmat.ptr(), count, mpi_datatype<cplxdb>::value, MPI_SUM, pid, comm_h.comm);
        }
        comm_h.barrier(); // May try out non-blocking later
    }

    global::profiler.stop(__FUNCTION__);
    return dmat_local;
}

std::map<Vector3_Order<int>, ComplexMatrix> get_dmat_cplx_Rs_kpara(
    int ispin, const MeanField &mf, const std::vector<Vector3_Order<double>>& kfrac_list,
    const std::vector<Vector3_Order<int>>& Rs, const MpiCommHandler &comm_h)
{
    return get_dmat_cplx_Rs_kpara(ispin, 0, 0, mf, kfrac_list, Rs, comm_h);
}

std::map<double, std::map<Vector3_Order<int>, ComplexMatrix>> get_gf_cplx_imagtimes_Rs_kpara(
    int ispin, int ispinor_bra, int ispinor_ket, const MeanField &mf, const std::vector<Vector3_Order<double>> &kfrac_list, std::vector<double> imagtimes,
    const std::vector<Vector3_Order<int>> &Rs, const MpiCommHandler &comm_h)
{
    std::map<double, std::map<Vector3_Order<int>, ComplexMatrix>> gf;

    // Collect Rs requested by each process
    std::vector<int> n_Rs_all, Rs_all;
    int nR_max;
    collect_Rs(Rs, n_Rs_all, Rs_all, nR_max, comm_h);
    // global::ofs_myid << "get_gf_cplx_imagtimes_Rs_kpara nRs_all " << n_Rs_all << std::endl;

    const int n_aos = mf.get_n_aos();
    const size_t size = n_aos * n_aos;

    const auto iks_local = mf.get_iks_local();
    // global::ofs_myid << "iks_local " << iks_local << std::endl;
    // Check if there is duplicate k-point data
    const int nk_local = iks_local.size();
    int nk_local_sum;
    MPI_Allreduce(&nk_local, &nk_local_sum, 1, MPI_INT, MPI_SUM, comm_h.comm);
    if (nk_local_sum > mf.get_n_kpoints())
        throw LIBRPA_RUNTIME_ERROR("found duplicated k-point eigenvectors data");
    else if (nk_local_sum < mf.get_n_kpoints())
        throw LIBRPA_RUNTIME_ERROR("missing k-point eigenvectors data");

    Matz kmat(size, nk_local, MAJOR::COL);
    Matz transmat(nk_local, nR_max, MAJOR::COL);
    Matz rmat(size, nR_max, MAJOR::COL);

    for (auto tau: imagtimes)
    {
        std::map<Vector3_Order<int>, ComplexMatrix> gf_tau;
        // TODO: this part is the same as denstiy matrix calculation, so it may be extracted to a common function
        for (int ik_local = 0; ik_local < nk_local; ik_local++)
        {
            const auto dmat_k = mf.get_gf_cplx_imagtime(ispin, ispinor_bra, ispinor_ket, iks_local[ik_local], tau);
            memcpy(kmat.ptr() + size * ik_local, dmat_k.c, size * sizeof(Matz::type));
        }

        for (int pid = 0; pid < comm_h.nprocs; pid++)
        {
            const auto nR_this = n_Rs_all[pid];
            // NOTE: MPI_Reduce requires all processes use the same sendcount, but rmat is simply zero where nk_local == 0.
            const size_t count = size * nR_this;
            // global::ofs_myid << "get_dmat_cplx_Rs_kpara pid " << pid << " nR_this " << nR_this << " count " << count << " matsize " << rmat.size() << std::endl;

            if (nR_this < 1) continue;
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (int iR = 0; iR <nR_this; iR++)
            {
                for (int ik = 0; ik < nk_local; ik++)
                {
                    const auto R_this = Rs_all.data() + pid * nR_max * 3 + iR * 3;
                    const auto &kf = kfrac_list[iks_local[ik]];
                    auto ang = - (kf.x * R_this[0] + kf.y * R_this[1] + kf.z * R_this[2]) * TWO_PI;
                    transmat(ik, iR) = cplxdb{cos(ang), sin(ang)};
                }
            }
            rmat = 0.0;
            if (nk_local > 0)
            {
                LapackConnector::gemm_f('N', 'N', size, nR_this, nk_local, 1.0,
                                        kmat.ptr(), size, transmat.ptr(), nk_local, 0.0, rmat.ptr(), size);
            }
            // global::ofs_myid << rmat << std::endl;
            if (comm_h.myid == pid)
            {
                MPI_Reduce(MPI_IN_PLACE, rmat.ptr(), count, mpi_datatype<cplxdb>::value, MPI_SUM, pid, comm_h.comm);
                for (int iR = 0; iR < nR_this; iR++)
                {
                    const auto R_this = Rs_all.data() + pid * nR_max * 3 + iR * 3;
                    ComplexMatrix m(n_aos, n_aos);
                    memcpy(m.c, rmat.ptr() + size * iR, size * sizeof(cplxdb));
                    Vector3_Order<int> R{R_this[0], R_this[1], R_this[2]};
                    gf_tau.emplace(R, std::move(m));
                }
            }
            else
            {
                MPI_Reduce(rmat.ptr(), rmat.ptr(), count, mpi_datatype<cplxdb>::value, MPI_SUM, pid, comm_h.comm);
            }
            comm_h.barrier(); // May try out non-blocking later
        }
        gf.emplace(tau, std::move(gf_tau));
    }

    return gf;
}

std::map<double, std::map<Vector3_Order<int>, ComplexMatrix>> get_gf_cplx_imagtimes_Rs_kpara(
    int ispin, const MeanField &mf, const std::vector<Vector3_Order<double>> &kfrac_list, std::vector<double> imagtimes,
    const std::vector<Vector3_Order<int>> &Rs, const MpiCommHandler &comm_h)
{
    return get_gf_cplx_imagtimes_Rs_kpara(ispin, 0, 0, mf, kfrac_list, imagtimes, Rs, comm_h);
}

std::map<Vector3_Order<int>, Matz> get_dmat_cplx_Rs_kblacs_para(
    int ispin, int ispinor_bra, int ispinor_ket, const MeanField &mf,
    const std::vector<Vector3_Order<double>> &kfrac_list, const std::vector<Vector3_Order<int>> &Rs,
    const KPointBlacsParallelContext &kblacs_ctxt, const ArrayDesc &desc_wfc, const ArrayDesc &desc_dm)
{
    global::profiler.start(__FUNCTION__);

    if (!kblacs_ctxt.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("KPointBlacsParallelContext is not initialized");

    const int n_aos = mf.get_n_aos();
    const int n_states = mf.get_n_states();
    const int n_kpoints = mf.get_n_kpoints();
    if (static_cast<int>(kfrac_list.size()) != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point fractional coordinate list has inconsistent size");
    if (kblacs_ctxt.n_kpoints() != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point BLACS context has inconsistent number of k-points");
    if (!desc_wfc.is_initialized() || !desc_dm.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("BLACS array descriptors are not initialized");
    if (desc_wfc.m() != n_aos || desc_wfc.n() != n_states)
        throw LIBRPA_RUNTIME_ERROR("wave-function descriptor must be n_aos x n_states");
    if (desc_dm.m() != n_aos || desc_dm.n() != n_aos)
        throw LIBRPA_RUNTIME_ERROR("density-matrix descriptor must be n_aos x n_aos");
    if (desc_wfc.ictxt() != desc_dm.ictxt())
        throw LIBRPA_RUNTIME_ERROR("wave-function and density-matrix descriptors must use the same BLACS context");

    std::vector<int> n_Rs_all, Rs_all;
    int nR_max;
    collect_Rs(Rs, n_Rs_all, Rs_all, nR_max, kblacs_ctxt.comm_kpoint_h);

    std::map<Vector3_Order<int>, Matz> dmat_Rs;
    for (const auto &R: Rs)
    {
        dmat_Rs.emplace(R, Matz(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL));
    }
    if (nR_max == 0)
    {
        global::profiler.stop(__FUNCTION__);
        return dmat_Rs;
    }

    const size_t wfc_size_loc = static_cast<size_t>(desc_wfc.m_loc()) * desc_wfc.n_loc();
    const size_t dm_size_loc = static_cast<size_t>(desc_dm.m_loc()) * desc_dm.n_loc();
    if (dm_size_loc > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw LIBRPA_RUNTIME_ERROR("local density-matrix block is too large for MPI collectives");
    const int n_elem = static_cast<int>(dm_size_loc);

    std::vector<cplxdb> dummy(1, C_ZERO);
    Matz scaled_wfc_ket(desc_wfc.m_loc(), desc_wfc.n_loc(), MAJOR::COL);
    Matz dmat_k(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL);
    auto wfc_gemm_workspace = create_wfc_gemm_workspace(desc_wfc, desc_dm,
                                                        kblacs_ctxt.blacs_h);
    const auto &iks_local = kblacs_ctxt.kpoints_local();
    const int nk_local = iks_local.size();
    int nk_sum = 0;
    MPI_Allreduce(&nk_local, &nk_sum, 1, mpi_datatype<int>::value, MPI_SUM,
                  kblacs_ctxt.comm_kpoint_h.comm);
    if (nk_sum != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point BLACS context has inconsistent k-point distribution");
    Matz kmat(nk_local, n_elem, MAJOR::COL);

    const double occ_thres = 1e-4 / n_kpoints;
    const double scale_spin = 0.5 * mf.get_n_spins() * mf.get_n_spinor();
    for (int ik_local = 0; ik_local != nk_local; ++ik_local)
    {
        const int ik = iks_local[ik_local];
        const auto *wfc_bra = mf.find_wfc(ispin, ispinor_bra, ik);
        const auto *wfc_ket = mf.find_wfc(ispin, ispinor_ket, ik);
        if ((wfc_bra == nullptr || wfc_ket == nullptr) && wfc_size_loc > 0)
            throw LIBRPA_RUNTIME_ERROR("missing local wave-function block for k-point " +
                                      std::to_string(ik));
        if (wfc_bra != nullptr && static_cast<size_t>(wfc_bra->size) != wfc_size_loc)
            throw LIBRPA_RUNTIME_ERROR("wave-function bra block size is inconsistent with descriptor");
        if (wfc_ket != nullptr && static_cast<size_t>(wfc_ket->size) != wfc_size_loc)
            throw LIBRPA_RUNTIME_ERROR("wave-function ket block size is inconsistent with descriptor");

        int nocc = 0;
        std::vector<double> weights;
        weights.reserve(n_states);
        for (; nocc != n_states; ++nocc)
        {
            const double weight = mf.get_weight()[ispin](ik, nocc) * scale_spin;
            if (weight < occ_thres) break;
            weights.push_back(weight);
        }

        scaled_wfc_ket = C_ZERO;
        if (wfc_ket != nullptr && wfc_size_loc > 0)
            std::memcpy(scaled_wfc_ket.ptr(), wfc_ket->c, wfc_size_loc * sizeof(cplxdb));
        const int wfc_m_loc = desc_wfc.m_loc();
        std::vector<char> scale_wfc_col(desc_wfc.n_loc(), false);
        std::vector<double> wfc_col_scale(desc_wfc.n_loc(), 1.0);
        for (int jloc = 0; jloc != desc_wfc.n_loc(); ++jloc)
        {
            const int jglob = desc_wfc.indx_l2g_c(jloc);
            if (jglob < 0 || jglob >= nocc) continue;
            scale_wfc_col[jloc] = true;
            wfc_col_scale[jloc] = weights[jglob];
        }
#pragma omp parallel for schedule(static) if (wfc_size_loc > 4096)
        for (size_t i = 0; i < wfc_size_loc; ++i)
        {
            const int jloc = static_cast<int>(i / wfc_m_loc);
            if (scale_wfc_col[jloc])
                scaled_wfc_ket.ptr()[i] *= wfc_col_scale[jloc];
        }

        dmat_k = C_ZERO;
        if (nocc > 0)
        {
            const cplxdb *wfc_bra_ptr = wfc_bra == nullptr ? dummy.data() : wfc_bra->c;
            pgemm_wfc_scaled_wfc_h(n_aos, nocc, wfc_bra_ptr, scaled_wfc_ket.ptr(),
                                   desc_wfc, wfc_gemm_workspace, dmat_k, desc_dm,
                                   kblacs_ctxt.blacs_h);
        }
#pragma omp parallel for schedule(static) if (n_elem > 4096)
        for (int i = 0; i != n_elem; ++i)
        {
            kmat.ptr()[ik_local + static_cast<size_t>(i) * nk_local] = dmat_k.ptr()[i];
        }
    }

    for (int pid = 0; pid != kblacs_ctxt.comm_kpoint_h.nprocs; ++pid)
    {
        const int nR_this = n_Rs_all[pid];
        if (nR_this < 1) continue;

        Matz transmat(nR_this, nk_local, MAJOR::COL);
        const size_t n_trans = static_cast<size_t>(nR_this) * nk_local;
#pragma omp parallel for schedule(static) if (n_trans > 4096)
        for (size_t i = 0; i < n_trans; ++i)
        {
            const int ik_local = static_cast<int>(i / nR_this);
            const int iR = static_cast<int>(i % nR_this);
            const auto &kf = kfrac_list[iks_local[ik_local]];
            const int index = pid * nR_max * 3 + iR * 3;
            const auto ang = - (kf.x * Rs_all[index] +
                                kf.y * Rs_all[index + 1] +
                                kf.z * Rs_all[index + 2]) * TWO_PI;
            transmat.ptr()[iR + static_cast<size_t>(ik_local) * nR_this] =
                cplxdb{std::cos(ang), std::sin(ang)};
        }

        Matz rmat(nR_this, n_elem, MAJOR::COL);
        if (nk_local > 0 && n_elem > 0)
        {
            LapackConnector::gemm_f('N', 'N', nR_this, n_elem, nk_local, C_ONE,
                                    transmat.ptr(), nR_this, kmat.ptr(), nk_local, C_ZERO,
                                    rmat.ptr(), nR_this);
        }

        const size_t count = static_cast<size_t>(nR_this) * n_elem;
        if (count > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw LIBRPA_RUNTIME_ERROR("local density-matrix Fourier block is too large for MPI collectives");
        const int count_int = static_cast<int>(count);
        if (kblacs_ctxt.comm_kpoint_h.myid == pid)
        {
            kblacs_ctxt.comm_kpoint_h.reduce(MPI_IN_PLACE, rmat.ptr(), count_int, pid, MPI_SUM);
            for (int iR = 0; iR != nR_this; ++iR)
            {
                const int index = pid * nR_max * 3 + iR * 3;
                Vector3_Order<int> R{Rs_all[index], Rs_all[index + 1], Rs_all[index + 2]};
                auto &m = dmat_Rs[R];
                m.resize(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL);
                const auto *rmat_iR = rmat.ptr() + iR;
#pragma omp parallel for schedule(static) if (n_elem > 4096)
                for (int i = 0; i != n_elem; ++i)
                {
                    m.ptr()[i] = rmat_iR[static_cast<size_t>(i) * nR_this];
                }
            }
        }
        else
        {
            kblacs_ctxt.comm_kpoint_h.reduce(MPI_IN_PLACE, rmat.ptr(), count_int, pid, MPI_SUM);
        }
    }

    global::profiler.stop(__FUNCTION__);
    return dmat_Rs;
}

std::map<Vector3_Order<int>, Matz> get_dmat_cplx_Rs_kblacs_para(
    int ispin, const MeanField &mf,
    const std::vector<Vector3_Order<double>> &kfrac_list, const std::vector<Vector3_Order<int>> &Rs,
    const KPointBlacsParallelContext &kblacs_ctxt, const ArrayDesc &desc_wfc, const ArrayDesc &desc_dm)
{
    return get_dmat_cplx_Rs_kblacs_para(ispin, 0, 0, mf, kfrac_list, Rs, kblacs_ctxt, desc_wfc, desc_dm);
}

std::map<Vector3_Order<int>, Matz> get_symmetry_restored_dmat_cplx_Rs_kblacs_para(
    int ispin, int ispinor_bra, int ispinor_ket, const MeanField &mf,
    const std::vector<Vector3_Order<double>> &kfrac_list, const std::vector<Vector3_Order<int>> &Rs,
    const KPointBlacsParallelContext &kblacs_ctxt, const ArrayDesc &desc_wfc, const ArrayDesc &desc_dm,
    const SymmetryContext &symmetry_context, const PeriodicBoundaryData &pbc,
    const AtomicBasis &atbasis_wfc)
{
    if (!kblacs_ctxt.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("KPointBlacsParallelContext is not initialized");

    const int n_aos = mf.get_n_aos();
    const int n_states = mf.get_n_states();
    const int n_kpoints = mf.get_n_kpoints();
    if (static_cast<int>(kfrac_list.size()) != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point fractional coordinate list has inconsistent size");
    if (kblacs_ctxt.n_kpoints() != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point BLACS context has inconsistent number of k-points");
    if (!desc_wfc.is_initialized() || !desc_dm.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("BLACS array descriptors are not initialized");
    if (desc_wfc.m() != n_aos || desc_wfc.n() != n_states)
        throw LIBRPA_RUNTIME_ERROR("wave-function descriptor must be n_aos x n_states");
    if (desc_dm.m() != n_aos || desc_dm.n() != n_aos)
        throw LIBRPA_RUNTIME_ERROR("density-matrix descriptor must be n_aos x n_aos");
    if (desc_wfc.ictxt() != desc_dm.ictxt())
        throw LIBRPA_RUNTIME_ERROR("wave-function and density-matrix descriptors must use the same BLACS context");

    const auto atom_nw = atbasis_wfc.get_atom_nb_map();
    const auto wfc_layouts = atbasis_wfc.has_l_shells()
        ? atbasis_wfc.build_species_basis_layouts(symmetry_context.atom_to_type)
        : std::vector<SpeciesBasisLayout>{};
    if (!can_restore_symmetry_kstar_meanfield(
            symmetry_context, wfc_layouts, mf, kfrac_list, atom_nw))
    {
        auto direct = get_dmat_cplx_Rs_kblacs_para(
            ispin, ispinor_bra, ispinor_ket, mf, kfrac_list, Rs,
            kblacs_ctxt, desc_wfc, desc_dm);
        return direct;
    }

    std::vector<int> n_Rs_all, Rs_all;
    int nR_max;
    collect_Rs(Rs, n_Rs_all, Rs_all, nR_max, kblacs_ctxt.comm_kpoint_h);

    std::map<Vector3_Order<int>, Matz> dmat_Rs;
    for (const auto &R : Rs)
    {
        dmat_Rs.emplace(R, Matz(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL));
    }
    if (nR_max == 0)
    {
        return dmat_Rs;
    }

    const auto target_pair_distribution =
        get_balanced_ap_distribution_for_consec_descriptor(
            atbasis_wfc, atbasis_wfc, desc_dm, true);
    IndexScheduler target_sched;
    target_sched.init(target_pair_distribution, atbasis_wfc, atbasis_wfc, desc_dm, false);

    std::set<atpair_t> target_pairs_local_set(target_sched.atpairs.cbegin(),
                                              target_sched.atpairs.cend());
    std::map<atpair_t, std::size_t> target_pair_offsets;
    std::size_t block_size_total = 0;
    for (const auto &pair : target_sched.atpairs)
    {
        target_pair_offsets[pair] = block_size_total;
        block_size_total += atbasis_wfc.get_atom_nb(pair.first)
                          * atbasis_wfc.get_atom_nb(pair.second);
    }

    const size_t wfc_size_loc = static_cast<size_t>(desc_wfc.m_loc()) * desc_wfc.n_loc();
    const size_t dm_size_loc = static_cast<size_t>(desc_dm.m_loc()) * desc_dm.n_loc();
    if (dm_size_loc > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw LIBRPA_RUNTIME_ERROR("local density-matrix block is too large for MPI collectives");

    std::vector<cplxdb> dummy(1, C_ZERO);
    Matz scaled_wfc_ket(desc_wfc.m_loc(), desc_wfc.n_loc(), MAJOR::COL);
    Matz dmat_k(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL);
    auto wfc_gemm_workspace = create_wfc_gemm_workspace(desc_wfc, desc_dm,
                                                        kblacs_ctxt.blacs_h);
    const auto &iks_local = kblacs_ctxt.kpoints_local();
    const int nk_local = iks_local.size();
    int nk_sum = 0;
    MPI_Allreduce(&nk_local, &nk_sum, 1, mpi_datatype<int>::value, MPI_SUM,
                  kblacs_ctxt.comm_kpoint_h.comm);
    if (nk_sum != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point BLACS context has inconsistent k-point distribution");

    const double occ_thres = 1e-4 / n_kpoints;
    const double scale_spin = 0.5 * mf.get_n_spins() * mf.get_n_spinor();
    const auto member_kfrac_targets =
        build_symmetry_kstar_member_kfrac_targets(symmetry_context, pbc);

    for (int pid = 0; pid != kblacs_ctxt.comm_kpoint_h.nprocs; ++pid)
    {
        const int nR_this = n_Rs_all[pid];
        if (nR_this < 1) continue;

        const std::size_t count =
            static_cast<std::size_t>(nR_this) * block_size_total;
        if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw LIBRPA_RUNTIME_ERROR(
                "local symmetry-restored density-matrix block is too large for MPI collectives");
        }
        std::vector<cplxdb> buffer(count, C_ZERO);

        for (int ik_local = 0; ik_local != nk_local; ++ik_local)
        {
            const int ik = iks_local[ik_local];
            const auto *wfc_bra = mf.find_wfc(ispin, ispinor_bra, ik);
            const auto *wfc_ket = mf.find_wfc(ispin, ispinor_ket, ik);
            if ((wfc_bra == nullptr || wfc_ket == nullptr) && wfc_size_loc > 0)
                throw LIBRPA_RUNTIME_ERROR("missing local wave-function block for k-point " +
                                          std::to_string(ik));
            if (wfc_bra != nullptr && static_cast<size_t>(wfc_bra->size) != wfc_size_loc)
                throw LIBRPA_RUNTIME_ERROR("wave-function bra block size is inconsistent with descriptor");
            if (wfc_ket != nullptr && static_cast<size_t>(wfc_ket->size) != wfc_size_loc)
                throw LIBRPA_RUNTIME_ERROR("wave-function ket block size is inconsistent with descriptor");

            int nocc = 0;
            std::vector<double> weights;
            weights.reserve(n_states);
            for (; nocc != n_states; ++nocc)
            {
                const double weight = mf.get_weight()[ispin](ik, nocc) * scale_spin;
                if (weight < occ_thres) break;
                weights.push_back(weight);
            }

            scaled_wfc_ket = C_ZERO;
            if (wfc_ket != nullptr && wfc_size_loc > 0)
                std::memcpy(scaled_wfc_ket.ptr(), wfc_ket->c,
                            wfc_size_loc * sizeof(cplxdb));
            const int wfc_m_loc = desc_wfc.m_loc();
            std::vector<char> scale_wfc_col(desc_wfc.n_loc(), false);
            std::vector<double> wfc_col_scale(desc_wfc.n_loc(), 1.0);
            for (int jloc = 0; jloc != desc_wfc.n_loc(); ++jloc)
            {
                const int jglob = desc_wfc.indx_l2g_c(jloc);
                if (jglob < 0 || jglob >= nocc) continue;
                scale_wfc_col[jloc] = true;
                wfc_col_scale[jloc] = weights[jglob];
            }
#pragma omp parallel for schedule(static) if (wfc_size_loc > 4096)
            for (size_t i = 0; i < wfc_size_loc; ++i)
            {
                const int jloc = static_cast<int>(i / wfc_m_loc);
                if (scale_wfc_col[jloc])
                    scaled_wfc_ket.ptr()[i] *= wfc_col_scale[jloc];
            }

            dmat_k = C_ZERO;
            if (nocc > 0)
            {
                const cplxdb *wfc_bra_ptr = wfc_bra == nullptr ? dummy.data() : wfc_bra->c;
                pgemm_wfc_scaled_wfc_h(n_aos, nocc, wfc_bra_ptr, scaled_wfc_ket.ptr(),
                                       desc_wfc, wfc_gemm_workspace, dmat_k, desc_dm,
                                       kblacs_ctxt.blacs_h);
            }

            const auto &k_ibz = kfrac_list[static_cast<std::size_t>(ik)];
            const auto &star = find_symmetry_kstar_for_ibz_kpoint(symmetry_context, k_ibz);
            const auto star_factor =
                1.0 / static_cast<double>(star.members.size());

            for (std::size_t imember = 0; imember != star.members.size(); ++imember)
            {
                const auto &member = star.members[imember];
                const Vector3_Order<double> *k_bz_target = nullptr;
                if (!member_kfrac_targets.empty()
                    && static_cast<std::size_t>(ik) < member_kfrac_targets.size()
                    && imember < member_kfrac_targets[static_cast<std::size_t>(ik)].size())
                {
                    k_bz_target = &member_kfrac_targets[static_cast<std::size_t>(ik)][imember];
                }
                const auto &k_for_phase = k_bz_target == nullptr ? member.k_bz : *k_bz_target;
                const auto source_pair_requests = build_source_pair_requests(
                    target_pair_distribution, member, atbasis_wfc.n_atoms);
                const auto source_pair_mats = get_ap_map_from_blacs_dist(
                    dmat_k, source_pair_requests, atbasis_wfc, atbasis_wfc, desc_dm);

                symmetry_atom_block_matrix_map_t source_blocks;
                for (const auto &[pair, mat] : source_pair_mats)
                {
                    source_blocks[pair.first][pair.second] = matz_to_complex_matrix(mat);
                }

                auto rotated_blocks = rotate_symmetry_kspace_operator_blocks(
                    symmetry_context, wfc_layouts, member, source_blocks, atom_nw,
                    k_ibz, member.time_reversal, &target_pairs_local_set, k_bz_target);

                for (int iR = 0; iR != nR_this; ++iR)
                {
                    const int index = pid * nR_max * 3 + iR * 3;
                    const Vector3_Order<int> R{Rs_all[index],
                                               Rs_all[index + 1],
                                               Rs_all[index + 2]};
                    const std::size_t R_offset =
                        static_cast<std::size_t>(iR) * block_size_total;
                    for (const auto &atom_i_blocks : rotated_blocks)
                    {
                        const auto atom_i = atom_i_blocks.first;
                        for (const auto &atom_j_block : atom_i_blocks.second)
                        {
                            const atpair_t pair{atom_i, atom_j_block.first};
                            const auto offset_iter = target_pair_offsets.find(pair);
                            if (offset_iter == target_pair_offsets.end()) continue;
                            const auto angle = -(k_for_phase * R) * TWO_PI;
                            const cplxdb phase{std::cos(angle), std::sin(angle)};
                            const cplxdb factor = star_factor * phase;
                            const auto &block = atom_j_block.second;
                            const auto n_J = atbasis_wfc.get_atom_nb(pair.second);
                            const std::size_t block_offset = R_offset + offset_iter->second;
                            for (int i = 0; i != block.nr; ++i)
                            {
                                for (int j = 0; j != block.nc; ++j)
                                {
                                    buffer[block_offset + static_cast<std::size_t>(i) * n_J + j] +=
                                        factor * block(i, j);
                                }
                            }
                        }
                    }
                }
            }
        }

        if (kblacs_ctxt.comm_kpoint_h.myid == pid)
        {
            kblacs_ctxt.comm_kpoint_h.reduce(
                MPI_IN_PLACE, buffer.data(), static_cast<int>(count), pid, MPI_SUM);
            for (int iR = 0; iR != nR_this; ++iR)
            {
                const int index = pid * nR_max * 3 + iR * 3;
                const Vector3_Order<int> R{Rs_all[index],
                                           Rs_all[index + 1],
                                           Rs_all[index + 2]};
                ap_p_map<Matz> dmat_ap;
                const std::size_t R_offset =
                    static_cast<std::size_t>(iR) * block_size_total;
                for (const auto &pair : target_sched.atpairs)
                {
                    const auto n_I = atbasis_wfc.get_atom_nb(pair.first);
                    const auto n_J = atbasis_wfc.get_atom_nb(pair.second);
                    Matz block(n_I, n_J, MAJOR::COL);
                    const std::size_t block_offset =
                        R_offset + target_pair_offsets.at(pair);
                    for (std::size_t i = 0; i != n_I; ++i)
                    {
                        for (std::size_t j = 0; j != n_J; ++j)
                        {
                            block(i, j) = buffer[block_offset + i * n_J + j];
                        }
                    }
                    dmat_ap[pair] = std::move(block);
                }
                auto &mat = dmat_Rs.at(R);
                mat = C_ZERO;
                fill_local_mat_from_ap_dist_scheduler(
                    mat, dmat_ap, target_sched, atbasis_wfc, atbasis_wfc, desc_dm);
            }
        }
        else
        {
            kblacs_ctxt.comm_kpoint_h.reduce(
                buffer.data(), buffer.data(), static_cast<int>(count), pid, MPI_SUM);
        }
    }

    return dmat_Rs;
}

std::map<double, std::map<Vector3_Order<int>, Matz>> get_gf_cplx_imagtimes_Rs_kblacs_para(
    int ispin, int ispinor_bra, int ispinor_ket, const MeanField &mf,
    const std::vector<Vector3_Order<double>> &kfrac_list, std::vector<double> imagtimes,
    const std::vector<Vector3_Order<int>> &Rs,
    const KPointBlacsParallelContext &kblacs_ctxt, const ArrayDesc &desc_wfc, const ArrayDesc &desc_dm)
{
    global::profiler.start(__FUNCTION__);

    if (!kblacs_ctxt.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("KPointBlacsParallelContext is not initialized");

    const int n_aos = mf.get_n_aos();
    const int n_states = mf.get_n_states();
    const int n_kpoints = mf.get_n_kpoints();
    if (static_cast<int>(kfrac_list.size()) != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point fractional coordinate list has inconsistent size");
    if (kblacs_ctxt.n_kpoints() != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point BLACS context has inconsistent number of k-points");
    if (!desc_wfc.is_initialized() || !desc_dm.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("BLACS array descriptors are not initialized");
    if (desc_wfc.m() != n_aos || desc_wfc.n() != n_states)
        throw LIBRPA_RUNTIME_ERROR("wave-function descriptor must be n_aos x n_states");
    if (desc_dm.m() != n_aos || desc_dm.n() != n_aos)
        throw LIBRPA_RUNTIME_ERROR("Green's-function descriptor must be n_aos x n_aos");
    if (desc_wfc.ictxt() != desc_dm.ictxt())
        throw LIBRPA_RUNTIME_ERROR("wave-function and Green's-function descriptors must use the same BLACS context");

    check_same_imagtimes(imagtimes, kblacs_ctxt.comm_global_h);

    std::vector<int> n_Rs_all, Rs_all;
    int nR_max;
    collect_Rs(Rs, n_Rs_all, Rs_all, nR_max, kblacs_ctxt.comm_kpoint_h);

    std::map<double, std::map<Vector3_Order<int>, Matz>> gf;
    if (imagtimes.empty())
    {
        global::profiler.stop(__FUNCTION__);
        return gf;
    }
    if (nR_max == 0)
    {
        for (const auto tau: imagtimes) gf.emplace(tau, std::map<Vector3_Order<int>, Matz>{});
        global::profiler.stop(__FUNCTION__);
        return gf;
    }

    const size_t wfc_size_loc = static_cast<size_t>(desc_wfc.m_loc()) * desc_wfc.n_loc();
    const size_t gf_size_loc = static_cast<size_t>(desc_dm.m_loc()) * desc_dm.n_loc();
    if (gf_size_loc > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw LIBRPA_RUNTIME_ERROR("local Green's-function block is too large for MPI collectives");
    const int n_elem = static_cast<int>(gf_size_loc);

    const auto &iks_local = kblacs_ctxt.kpoints_local();
    const int nk_local = iks_local.size();
    int nk_sum = 0;
    MPI_Allreduce(&nk_local, &nk_sum, 1, mpi_datatype<int>::value, MPI_SUM,
                  kblacs_ctxt.comm_kpoint_h.comm);
    if (nk_sum != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point BLACS context has inconsistent k-point distribution");

    std::vector<cplxdb> dummy(1, C_ZERO);
    Matz scaled_wfc_ket(desc_wfc.m_loc(), desc_wfc.n_loc(), MAJOR::COL);
    Matz gf_k(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL);
    auto wfc_gemm_workspace = create_wfc_gemm_workspace(desc_wfc, desc_dm,
                                                        kblacs_ctxt.blacs_h);
    Matz kmat(nk_local, n_elem, MAJOR::COL);

    const double scale_spin = 0.5 * mf.get_n_spins() * mf.get_n_spinor();
    for (const auto tau: imagtimes)
    {
        std::map<Vector3_Order<int>, Matz> gf_tau;
        for (const auto &R: Rs)
        {
            gf_tau.emplace(R, Matz(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL));
        }

        for (int ik_local = 0; ik_local != nk_local; ++ik_local)
        {
            const int ik = iks_local[ik_local];
            const auto *wfc_bra = mf.find_wfc(ispin, ispinor_bra, ik);
            const auto *wfc_ket = mf.find_wfc(ispin, ispinor_ket, ik);
            if ((wfc_bra == nullptr || wfc_ket == nullptr) && wfc_size_loc > 0)
                throw LIBRPA_RUNTIME_ERROR("missing local wave-function block for k-point " +
                                          std::to_string(ik));
            if (wfc_bra != nullptr && static_cast<size_t>(wfc_bra->size) != wfc_size_loc)
                throw LIBRPA_RUNTIME_ERROR("wave-function bra block size is inconsistent with descriptor");
            if (wfc_ket != nullptr && static_cast<size_t>(wfc_ket->size) != wfc_size_loc)
                throw LIBRPA_RUNTIME_ERROR("wave-function ket block size is inconsistent with descriptor");

            std::vector<double> scales(n_states);
#pragma omp parallel for schedule(static) if (n_states > 64)
            for (int ib = 0; ib != n_states; ++ib)
            {
                const double wg_occ = mf.get_weight()[ispin](ik, ib) * scale_spin;
                double wg_empty = 1.0 / n_kpoints - wg_occ;
                if (wg_empty < 0.0) wg_empty = 0.0;
                const double prefac = tau > 0 ? wg_empty : wg_occ;
                double scale = -tau * (mf.get_eigenvals()[ispin](ik, ib) - mf.get_efermi());
                if (scale > 0.0) scale = 0.0;
                scales[ib] = std::exp(scale) * prefac;
            }

            scaled_wfc_ket = C_ZERO;
            if (wfc_ket != nullptr && wfc_size_loc > 0)
                std::memcpy(scaled_wfc_ket.ptr(), wfc_ket->c, wfc_size_loc * sizeof(cplxdb));
            const int wfc_m_loc = desc_wfc.m_loc();
            std::vector<double> wfc_col_scale(desc_wfc.n_loc(), 0.0);
            for (int jloc = 0; jloc != desc_wfc.n_loc(); ++jloc)
            {
                const int jglob = desc_wfc.indx_l2g_c(jloc);
                if (jglob < 0 || jglob >= n_states) continue;
                wfc_col_scale[jloc] = scales[jglob];
            }
#pragma omp parallel for schedule(static) if (wfc_size_loc > 4096)
            for (size_t i = 0; i < wfc_size_loc; ++i)
            {
                const int jloc = static_cast<int>(i / wfc_m_loc);
                scaled_wfc_ket.ptr()[i] *= wfc_col_scale[jloc];
            }

            gf_k = C_ZERO;
            const cplxdb *wfc_bra_ptr = wfc_bra == nullptr ? dummy.data() : wfc_bra->c;
            pgemm_wfc_scaled_wfc_h(n_aos, n_states, wfc_bra_ptr, scaled_wfc_ket.ptr(),
                                   desc_wfc, wfc_gemm_workspace, gf_k, desc_dm,
                                   kblacs_ctxt.blacs_h);
#pragma omp parallel for schedule(static) if (n_elem > 4096)
            for (int i = 0; i != n_elem; ++i)
            {
                kmat.ptr()[ik_local + static_cast<size_t>(i) * nk_local] = gf_k.ptr()[i];
            }
        }

        for (int pid = 0; pid != kblacs_ctxt.comm_kpoint_h.nprocs; ++pid)
        {
            const int nR_this = n_Rs_all[pid];
            if (nR_this < 1) continue;

            Matz transmat(nR_this, nk_local, MAJOR::COL);
            const double tau_sign = tau > 0 ? 1.0 : -1.0;
            const size_t n_trans = static_cast<size_t>(nR_this) * nk_local;
#pragma omp parallel for schedule(static) if (n_trans > 4096)
            for (size_t i = 0; i < n_trans; ++i)
            {
                const int ik_local = static_cast<int>(i / nR_this);
                const int iR = static_cast<int>(i % nR_this);
                const auto &kf = kfrac_list[iks_local[ik_local]];
                const int index = pid * nR_max * 3 + iR * 3;
                const auto ang = - (kf.x * Rs_all[index] +
                                    kf.y * Rs_all[index + 1] +
                                    kf.z * Rs_all[index + 2]) * TWO_PI;
                transmat.ptr()[iR + static_cast<size_t>(ik_local) * nR_this] =
                    tau_sign * cplxdb{std::cos(ang), std::sin(ang)};
            }

            Matz rmat(nR_this, n_elem, MAJOR::COL);
            if (nk_local > 0 && n_elem > 0)
            {
                LapackConnector::gemm_f('N', 'N', nR_this, n_elem, nk_local, C_ONE,
                                        transmat.ptr(), nR_this, kmat.ptr(), nk_local, C_ZERO,
                                        rmat.ptr(), nR_this);
            }

            const size_t count = static_cast<size_t>(nR_this) * n_elem;
            if (count > static_cast<size_t>(std::numeric_limits<int>::max()))
                throw LIBRPA_RUNTIME_ERROR("local Green's-function Fourier block is too large for MPI collectives");
            const int count_int = static_cast<int>(count);
            if (kblacs_ctxt.comm_kpoint_h.myid == pid)
            {
                kblacs_ctxt.comm_kpoint_h.reduce(MPI_IN_PLACE, rmat.ptr(), count_int, pid, MPI_SUM);
                for (int iR = 0; iR != nR_this; ++iR)
                {
                    const int index = pid * nR_max * 3 + iR * 3;
                    Vector3_Order<int> R{Rs_all[index], Rs_all[index + 1], Rs_all[index + 2]};
                    auto &m = gf_tau[R];
                    m.resize(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL);
                    const auto *rmat_iR = rmat.ptr() + iR;
#pragma omp parallel for schedule(static) if (n_elem > 4096)
                    for (int i = 0; i != n_elem; ++i)
                    {
                        m.ptr()[i] = rmat_iR[static_cast<size_t>(i) * nR_this];
                    }
                }
            }
            else
            {
                kblacs_ctxt.comm_kpoint_h.reduce(MPI_IN_PLACE, rmat.ptr(), count_int, pid, MPI_SUM);
            }
        }
        gf.emplace(tau, std::move(gf_tau));
    }

    global::profiler.stop(__FUNCTION__);
    return gf;
}

std::map<double, std::map<Vector3_Order<int>, Matz>>
get_symmetry_restored_gf_cplx_imagtimes_Rs_kblacs_para(
    int ispin, int ispinor_bra, int ispinor_ket, const MeanField &mf,
    const std::vector<Vector3_Order<double>> &kfrac_list, std::vector<double> imagtimes,
    const std::vector<Vector3_Order<int>> &Rs,
    const KPointBlacsParallelContext &kblacs_ctxt, const ArrayDesc &desc_wfc, const ArrayDesc &desc_dm,
    const SymmetryContext &symmetry_context, const PeriodicBoundaryData &pbc,
    const AtomicBasis &atbasis_wfc)
{
    if (!kblacs_ctxt.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("KPointBlacsParallelContext is not initialized");

    const int n_aos = mf.get_n_aos();
    const int n_states = mf.get_n_states();
    const int n_kpoints = mf.get_n_kpoints();
    if (static_cast<int>(kfrac_list.size()) != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point fractional coordinate list has inconsistent size");
    if (kblacs_ctxt.n_kpoints() != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point BLACS context has inconsistent number of k-points");
    if (!desc_wfc.is_initialized() || !desc_dm.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("BLACS array descriptors are not initialized");
    if (desc_wfc.m() != n_aos || desc_wfc.n() != n_states)
        throw LIBRPA_RUNTIME_ERROR("wave-function descriptor must be n_aos x n_states");
    if (desc_dm.m() != n_aos || desc_dm.n() != n_aos)
        throw LIBRPA_RUNTIME_ERROR("Green's-function descriptor must be n_aos x n_aos");
    if (desc_wfc.ictxt() != desc_dm.ictxt())
        throw LIBRPA_RUNTIME_ERROR("wave-function and Green's-function descriptors must use the same BLACS context");

    check_same_imagtimes(imagtimes, kblacs_ctxt.comm_global_h);

    const auto atom_nw = atbasis_wfc.get_atom_nb_map();
    const auto wfc_layouts = atbasis_wfc.has_l_shells()
        ? atbasis_wfc.build_species_basis_layouts(symmetry_context.atom_to_type)
        : std::vector<SpeciesBasisLayout>{};
    if (!can_restore_symmetry_kstar_meanfield(
            symmetry_context, wfc_layouts, mf, kfrac_list, atom_nw))
    {
        return get_gf_cplx_imagtimes_Rs_kblacs_para(
            ispin, ispinor_bra, ispinor_ket, mf, kfrac_list, imagtimes, Rs,
            kblacs_ctxt, desc_wfc, desc_dm);
    }

    std::vector<int> n_Rs_all, Rs_all;
    int nR_max;
    collect_Rs(Rs, n_Rs_all, Rs_all, nR_max, kblacs_ctxt.comm_kpoint_h);

    std::map<double, std::map<Vector3_Order<int>, Matz>> gf;
    for (const auto tau : imagtimes)
    {
        auto &gf_tau = gf[tau];
        for (const auto &R : Rs)
        {
            gf_tau.emplace(R, Matz(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL));
        }
    }
    if (imagtimes.empty() || nR_max == 0)
    {
        return gf;
    }

    const auto target_pair_distribution =
        get_balanced_ap_distribution_for_consec_descriptor(
            atbasis_wfc, atbasis_wfc, desc_dm, true);
    IndexScheduler target_sched;
    target_sched.init(target_pair_distribution, atbasis_wfc, atbasis_wfc, desc_dm, false);

    std::set<atpair_t> target_pairs_local_set(target_sched.atpairs.cbegin(),
                                              target_sched.atpairs.cend());
    std::map<atpair_t, std::size_t> target_pair_offsets;
    std::size_t block_size_total = 0;
    for (const auto &pair : target_sched.atpairs)
    {
        target_pair_offsets[pair] = block_size_total;
        block_size_total += atbasis_wfc.get_atom_nb(pair.first)
                          * atbasis_wfc.get_atom_nb(pair.second);
    }

    const size_t wfc_size_loc = static_cast<size_t>(desc_wfc.m_loc()) * desc_wfc.n_loc();
    if (block_size_total > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw LIBRPA_RUNTIME_ERROR("local Green's-function atom-pair block is too large for MPI collectives");

    std::vector<cplxdb> dummy(1, C_ZERO);
    Matz scaled_wfc_ket(desc_wfc.m_loc(), desc_wfc.n_loc(), MAJOR::COL);
    Matz gf_k(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL);
    auto wfc_gemm_workspace = create_wfc_gemm_workspace(desc_wfc, desc_dm,
                                                        kblacs_ctxt.blacs_h);
    const auto &iks_local = kblacs_ctxt.kpoints_local();
    const int nk_local = iks_local.size();
    int nk_sum = 0;
    MPI_Allreduce(&nk_local, &nk_sum, 1, mpi_datatype<int>::value, MPI_SUM,
                  kblacs_ctxt.comm_kpoint_h.comm);
    if (nk_sum != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point BLACS context has inconsistent k-point distribution");

    const double scale_spin = 0.5 * mf.get_n_spins() * mf.get_n_spinor();
    const double full_k_count =
        static_cast<double>(symmetry_context.count_kstar_members());
    if (full_k_count <= 0.0)
        throw LIBRPA_RUNTIME_ERROR("k-star restore found zero full-k members");
    const auto member_kfrac_targets =
        build_symmetry_kstar_member_kfrac_targets(symmetry_context, pbc);

    for (const auto tau : imagtimes)
    {
        const double tau_sign = tau > 0.0 ? 1.0 : -1.0;
        for (int pid = 0; pid != kblacs_ctxt.comm_kpoint_h.nprocs; ++pid)
        {
            const int nR_this = n_Rs_all[pid];
            if (nR_this < 1) continue;

            const std::size_t count =
                static_cast<std::size_t>(nR_this) * block_size_total;
            if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            {
                throw LIBRPA_RUNTIME_ERROR(
                    "local symmetry-restored Green's-function block is too large for MPI collectives");
            }
            std::vector<cplxdb> buffer(count, C_ZERO);

            for (int ik_local = 0; ik_local != nk_local; ++ik_local)
            {
                const int ik = iks_local[ik_local];
                const auto *wfc_bra = mf.find_wfc(ispin, ispinor_bra, ik);
                const auto *wfc_ket = mf.find_wfc(ispin, ispinor_ket, ik);
                if ((wfc_bra == nullptr || wfc_ket == nullptr) && wfc_size_loc > 0)
                    throw LIBRPA_RUNTIME_ERROR("missing local wave-function block for k-point " +
                                              std::to_string(ik));
                if (wfc_bra != nullptr && static_cast<size_t>(wfc_bra->size) != wfc_size_loc)
                    throw LIBRPA_RUNTIME_ERROR("wave-function bra block size is inconsistent with descriptor");
                if (wfc_ket != nullptr && static_cast<size_t>(wfc_ket->size) != wfc_size_loc)
                    throw LIBRPA_RUNTIME_ERROR("wave-function ket block size is inconsistent with descriptor");

                const auto &k_ibz = kfrac_list[static_cast<std::size_t>(ik)];
                const auto &star = find_symmetry_kstar_for_ibz_kpoint(symmetry_context, k_ibz);
                const auto star_factor =
                    1.0 / static_cast<double>(star.members.size());
                const auto kpoint_weight =
                    static_cast<double>(star.members.size()) / full_k_count;

                std::vector<double> scales(n_states);
#pragma omp parallel for schedule(static) if (n_states > 64)
                for (int ib = 0; ib != n_states; ++ib)
                {
                    const double wg_occ = mf.get_weight()[ispin](ik, ib) * scale_spin;
                    const double prefac = tau > 0.0
                        ? std::max(0.0, kpoint_weight - wg_occ)
                        : wg_occ;
                    double scale = -tau * (mf.get_eigenvals()[ispin](ik, ib) - mf.get_efermi());
                    if (scale > 0.0) scale = 0.0;
                    scales[ib] = std::exp(scale) * prefac;
                }

                scaled_wfc_ket = C_ZERO;
                if (wfc_ket != nullptr && wfc_size_loc > 0)
                    std::memcpy(scaled_wfc_ket.ptr(), wfc_ket->c,
                                wfc_size_loc * sizeof(cplxdb));
                const int wfc_m_loc = desc_wfc.m_loc();
                std::vector<double> wfc_col_scale(desc_wfc.n_loc(), 0.0);
                for (int jloc = 0; jloc != desc_wfc.n_loc(); ++jloc)
                {
                    const int jglob = desc_wfc.indx_l2g_c(jloc);
                    if (jglob < 0 || jglob >= n_states) continue;
                    wfc_col_scale[jloc] = scales[jglob];
                }
#pragma omp parallel for schedule(static) if (wfc_size_loc > 4096)
                for (size_t i = 0; i < wfc_size_loc; ++i)
                {
                    const int jloc = static_cast<int>(i / wfc_m_loc);
                    scaled_wfc_ket.ptr()[i] *= wfc_col_scale[jloc];
                }

                gf_k = C_ZERO;
                const cplxdb *wfc_bra_ptr = wfc_bra == nullptr ? dummy.data() : wfc_bra->c;
                pgemm_wfc_scaled_wfc_h(n_aos, n_states, wfc_bra_ptr, scaled_wfc_ket.ptr(),
                                       desc_wfc, wfc_gemm_workspace, gf_k, desc_dm,
                                       kblacs_ctxt.blacs_h);

                for (std::size_t imember = 0; imember != star.members.size(); ++imember)
                {
                    const auto &member = star.members[imember];
                    const Vector3_Order<double> *k_bz_target = nullptr;
                    if (!member_kfrac_targets.empty()
                        && static_cast<std::size_t>(ik) < member_kfrac_targets.size()
                        && imember < member_kfrac_targets[static_cast<std::size_t>(ik)].size())
                    {
                        k_bz_target = &member_kfrac_targets[static_cast<std::size_t>(ik)][imember];
                    }
                    const auto &k_for_phase = k_bz_target == nullptr ? member.k_bz : *k_bz_target;
                    const auto source_pair_requests = build_source_pair_requests(
                        target_pair_distribution, member, atbasis_wfc.n_atoms);
                    const auto source_pair_mats = get_ap_map_from_blacs_dist(
                        gf_k, source_pair_requests, atbasis_wfc, atbasis_wfc, desc_dm);

                    symmetry_atom_block_matrix_map_t source_blocks;
                    for (const auto &[pair, mat] : source_pair_mats)
                    {
                        source_blocks[pair.first][pair.second] = matz_to_complex_matrix(mat);
                    }

                    auto rotated_blocks = rotate_symmetry_kspace_operator_blocks(
                        symmetry_context, wfc_layouts, member, source_blocks, atom_nw,
                        k_ibz, member.time_reversal, &target_pairs_local_set, k_bz_target);

                    for (int iR = 0; iR != nR_this; ++iR)
                    {
                        const int index = pid * nR_max * 3 + iR * 3;
                        const Vector3_Order<int> R{Rs_all[index],
                                                   Rs_all[index + 1],
                                                   Rs_all[index + 2]};
                        const auto angle = -(k_for_phase * R) * TWO_PI;
                        const cplxdb phase{std::cos(angle), std::sin(angle)};
                        const cplxdb factor = star_factor * tau_sign * phase;
                        const std::size_t R_offset =
                            static_cast<std::size_t>(iR) * block_size_total;
                        for (const auto &atom_i_blocks : rotated_blocks)
                        {
                            const auto atom_i = atom_i_blocks.first;
                            for (const auto &atom_j_block : atom_i_blocks.second)
                            {
                                const atpair_t pair{atom_i, atom_j_block.first};
                                const auto offset_iter = target_pair_offsets.find(pair);
                                if (offset_iter == target_pair_offsets.end()) continue;
                                const auto &block = atom_j_block.second;
                                const auto n_J = atbasis_wfc.get_atom_nb(pair.second);
                                const std::size_t block_offset = R_offset + offset_iter->second;
                                for (int i = 0; i != block.nr; ++i)
                                {
                                    for (int j = 0; j != block.nc; ++j)
                                    {
                                        buffer[block_offset + static_cast<std::size_t>(i) * n_J + j] +=
                                            factor * block(i, j);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (kblacs_ctxt.comm_kpoint_h.myid == pid)
            {
                kblacs_ctxt.comm_kpoint_h.reduce(
                    MPI_IN_PLACE, buffer.data(), static_cast<int>(count), pid, MPI_SUM);
                for (int iR = 0; iR != nR_this; ++iR)
                {
                    const int index = pid * nR_max * 3 + iR * 3;
                    const Vector3_Order<int> R{Rs_all[index],
                                               Rs_all[index + 1],
                                               Rs_all[index + 2]};
                    ap_p_map<Matz> gf_ap;
                    const std::size_t R_offset =
                        static_cast<std::size_t>(iR) * block_size_total;
                    for (const auto &pair : target_sched.atpairs)
                    {
                        const auto n_I = atbasis_wfc.get_atom_nb(pair.first);
                        const auto n_J = atbasis_wfc.get_atom_nb(pair.second);
                        Matz block(n_I, n_J, MAJOR::COL);
                        const std::size_t block_offset =
                            R_offset + target_pair_offsets.at(pair);
                        for (std::size_t i = 0; i != n_I; ++i)
                        {
                            for (std::size_t j = 0; j != n_J; ++j)
                            {
                                block(i, j) = buffer[block_offset + i * n_J + j];
                            }
                        }
                        gf_ap[pair] = std::move(block);
                    }
                    auto &mat = gf.at(tau).at(R);
                    mat = C_ZERO;
                    fill_local_mat_from_ap_dist_scheduler(
                        mat, gf_ap, target_sched, atbasis_wfc, atbasis_wfc, desc_dm);
                }
            }
            else
            {
                kblacs_ctxt.comm_kpoint_h.reduce(
                    buffer.data(), buffer.data(), static_cast<int>(count), pid, MPI_SUM);
            }
        }
    }

    return gf;
}

std::map<double, std::map<Vector3_Order<int>, Matz>> get_gf_cplx_imagtimes_Rs_kblacs_para(
    int ispin, const MeanField &mf,
    const std::vector<Vector3_Order<double>> &kfrac_list, std::vector<double> imagtimes,
    const std::vector<Vector3_Order<int>> &Rs,
    const KPointBlacsParallelContext &kblacs_ctxt, const ArrayDesc &desc_wfc, const ArrayDesc &desc_dm)
{
    return get_gf_cplx_imagtimes_Rs_kblacs_para(ispin, 0, 0, mf, kfrac_list, imagtimes, Rs, kblacs_ctxt, desc_wfc, desc_dm);
}

}
