#include "meanfield.h"

#include "../math/matrix_m.h"
#include "../mpi/base_mpi.h"
#include "../mpi/kpoint_blacs_parallel_context.h"

namespace librpa_int
{

// ==================================================================
// Density matrix and green's function calculation, MPI parallel (over k-points) version
// User should ensure that wave function at a certain k-point exists only on one MPI process

std::map<Vector3_Order<int>, ComplexMatrix> get_dmat_cplx_Rs_kpara(
    int ispin, int ispinor_bra, int ispinor_ket, const MeanField &mf,
    const std::vector<Vector3_Order<double>> &kfrac_list, const std::vector<Vector3_Order<int>> &Rs,
    const MpiCommHandler &comm_h);

std::map<Vector3_Order<int>, ComplexMatrix> get_dmat_cplx_Rs_kpara(
    int ispin, const MeanField &mf,
    const std::vector<Vector3_Order<double>> &kfrac_list, const std::vector<Vector3_Order<int>> &Rs,
    const MpiCommHandler &comm_h);

std::map<double, std::map<Vector3_Order<int>, ComplexMatrix>> get_gf_cplx_imagtimes_Rs_kpara(
    int ispin, int ispinor_bra, int ispinor_ket, const MeanField &mf,
    const std::vector<Vector3_Order<double>> &kfrac_list, std::vector<double> imagtimes,
    const std::vector<Vector3_Order<int>> &Rs, const MpiCommHandler &comm_h);

std::map<double, std::map<Vector3_Order<int>, ComplexMatrix>> get_gf_cplx_imagtimes_Rs_kpara(
    int ispin, const MeanField &mf,
    const std::vector<Vector3_Order<double>> &kfrac_list, std::vector<double> imagtimes,
    const std::vector<Vector3_Order<int>> &Rs, const MpiCommHandler &comm_h);

// ==================================================================
// Density matrix and green's function calculation, two-level parallel (over k-points and BLACS) version.
// User should ensure that wave function are distributed among processes in kblacs_ctxt,
// and there is no duplicated k-point data.

// The output complex matrices are column-major Matz, which is different from conventions of functions above.
// This is mainly to account for usage of column major in BLACS functions, and can help to perform 2D->IJ transformation later
// using IndexScheduler.

// Rs: real-space vectors requested on this process. Processes in the same comm_kpoint may request
//     different R lists while sharing the same local BLACS matrix region.
// desc_wfc: descriptor of eigenvectors source of the MeanField object
// desc_gf: descriptor of the resulting distributed density matrix
std::map<Vector3_Order<int>, Matz> get_dmat_cplx_Rs_kblacs_para(
    int ispin, int ispinor_bra, int ispinor_ket, const MeanField &mf,
    const std::vector<Vector3_Order<double>> &kfrac_list, const std::vector<Vector3_Order<int>> &Rs,
    const KPointBlacsParallelContext &kblacs_ctxt, const ArrayDesc &desc_wfc, const ArrayDesc &desc_dm);

std::map<Vector3_Order<int>, Matz> get_dmat_cplx_Rs_kblacs_para(
    int ispin, const MeanField &mf,
    const std::vector<Vector3_Order<double>> &kfrac_list, const std::vector<Vector3_Order<int>> &Rs,
    const KPointBlacsParallelContext &kblacs_ctxt, const ArrayDesc &desc_wfc, const ArrayDesc &desc_dm);

}  // namespace librpa_int
