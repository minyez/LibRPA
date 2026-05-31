#include "meanfield_mpi.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>

#include "../io/stl_io_helper.h"
#include "../math/lapack_connector.h"
#include "../math/matrix_m.h"
#include "../math/scalapack_connector.h"
#include "../utils/constants.h"
#include "../utils/profiler.h"

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
        for (int jloc = 0; jloc != desc_wfc.n_loc(); ++jloc)
        {
            const int jglob = desc_wfc.indx_l2g_c(jloc);
            if (jglob < 0 || jglob >= nocc) continue;
            for (int iloc = 0; iloc != desc_wfc.m_loc(); ++iloc)
            {
                scaled_wfc_ket(iloc, jloc) *= weights[jglob];
            }
        }

        dmat_k = C_ZERO;
        if (nocc > 0)
        {
            const cplxdb *wfc_bra_ptr = wfc_bra == nullptr ? dummy.data() : wfc_bra->c;
            ScalapackConnector::pgemm_f('N', 'C', n_aos, n_aos, nocc, C_ONE,
                                        wfc_bra_ptr, 1, 1, desc_wfc.desc,
                                        scaled_wfc_ket.ptr(), 1, 1, desc_wfc.desc, C_ZERO,
                                        dmat_k.ptr(), 1, 1, desc_dm.desc);
        }

        for (int i = 0; i != n_elem; ++i)
        {
            kmat(ik_local, i) = dmat_k.ptr()[i];
        }
    }

    for (int pid = 0; pid != kblacs_ctxt.comm_kpoint_h.nprocs; ++pid)
    {
        const int nR_this = n_Rs_all[pid];
        if (nR_this < 1) continue;

        Matz transmat(nR_this, nk_local, MAJOR::COL);
        for (int ik_local = 0; ik_local != nk_local; ++ik_local)
        {
            const auto &kf = kfrac_list[iks_local[ik_local]];
            for (int iR = 0; iR != nR_this; ++iR)
            {
                const int index = pid * nR_max * 3 + iR * 3;
                const auto ang = - (kf.x * Rs_all[index] +
                                    kf.y * Rs_all[index + 1] +
                                    kf.z * Rs_all[index + 2]) * TWO_PI;
                transmat(iR, ik_local) = cplxdb{std::cos(ang), std::sin(ang)};
            }
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
            MPI_Reduce(MPI_IN_PLACE, rmat.ptr(), count_int, mpi_datatype<cplxdb>::value,
                       MPI_SUM, pid, kblacs_ctxt.comm_kpoint_h.comm);
            for (int iR = 0; iR != nR_this; ++iR)
            {
                const int index = pid * nR_max * 3 + iR * 3;
                Vector3_Order<int> R{Rs_all[index], Rs_all[index + 1], Rs_all[index + 2]};
                auto &m = dmat_Rs[R];
                m.resize(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL);
                for (int i = 0; i != n_elem; ++i)
                {
                    m.ptr()[i] = rmat(iR, i);
                }
            }
        }
        else
        {
            MPI_Reduce(rmat.ptr(), rmat.ptr(), count_int, mpi_datatype<cplxdb>::value,
                       MPI_SUM, pid, kblacs_ctxt.comm_kpoint_h.comm);
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
            for (int jloc = 0; jloc != desc_wfc.n_loc(); ++jloc)
            {
                const int jglob = desc_wfc.indx_l2g_c(jloc);
                for (int iloc = 0; iloc != desc_wfc.m_loc(); ++iloc)
                {
                    scaled_wfc_ket(iloc, jloc) *= scales[jglob];
                }
            }

            gf_k = C_ZERO;
            const cplxdb *wfc_bra_ptr = wfc_bra == nullptr ? dummy.data() : wfc_bra->c;
            ScalapackConnector::pgemm_f('N', 'C', n_aos, n_aos, n_states, C_ONE,
                                        wfc_bra_ptr, 1, 1, desc_wfc.desc,
                                        scaled_wfc_ket.ptr(), 1, 1, desc_wfc.desc, C_ZERO,
                                        gf_k.ptr(), 1, 1, desc_dm.desc);
            for (int i = 0; i != n_elem; ++i)
            {
                kmat(ik_local, i) = gf_k.ptr()[i];
            }
        }

        for (int pid = 0; pid != kblacs_ctxt.comm_kpoint_h.nprocs; ++pid)
        {
            const int nR_this = n_Rs_all[pid];
            if (nR_this < 1) continue;

            Matz transmat(nR_this, nk_local, MAJOR::COL);
            const double tau_sign = tau > 0 ? 1.0 : -1.0;
            for (int ik_local = 0; ik_local != nk_local; ++ik_local)
            {
                const auto &kf = kfrac_list[iks_local[ik_local]];
                for (int iR = 0; iR != nR_this; ++iR)
                {
                    const int index = pid * nR_max * 3 + iR * 3;
                    const auto ang = - (kf.x * Rs_all[index] +
                                        kf.y * Rs_all[index + 1] +
                                        kf.z * Rs_all[index + 2]) * TWO_PI;
                    transmat(iR, ik_local) = tau_sign * cplxdb{std::cos(ang), std::sin(ang)};
                }
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
                MPI_Reduce(MPI_IN_PLACE, rmat.ptr(), count_int, mpi_datatype<cplxdb>::value,
                           MPI_SUM, pid, kblacs_ctxt.comm_kpoint_h.comm);
                for (int iR = 0; iR != nR_this; ++iR)
                {
                    const int index = pid * nR_max * 3 + iR * 3;
                    Vector3_Order<int> R{Rs_all[index], Rs_all[index + 1], Rs_all[index + 2]};
                    auto &m = gf_tau[R];
                    m.resize(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL);
                    for (int i = 0; i != n_elem; ++i)
                    {
                        m.ptr()[i] = rmat(iR, i);
                    }
                }
            }
            else
            {
                MPI_Reduce(rmat.ptr(), rmat.ptr(), count_int, mpi_datatype<cplxdb>::value,
                           MPI_SUM, pid, kblacs_ctxt.comm_kpoint_h.comm);
            }
        }
        gf.emplace(tau, std::move(gf_tau));
    }

    global::profiler.stop(__FUNCTION__);
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
