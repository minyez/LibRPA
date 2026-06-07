#include "pbc.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "../utils/constants.h"
#include "../utils/error.h"

namespace librpa_int {

static const double a = 1e5;  // 10000 nm
static const double inva = 1.0 / a;

// Initialized as a huge box to emulate isolated case
PeriodicBoundaryData::PeriodicBoundaryData():
    lattice_reset_(false),
    kgrid_no_symmetry_(true),
    latvec(a, 0, 0, 0, a, 0, 0, 0, a),
    G(inva, 0, 0, 0, inva, 0, 0, 0, inva),
    period(1, 1, 1),
    period_array({1, 1, 1}),
    Rlist({{0, 0, 0}}),
    klist({{0, 0, 0}}),
    kfrac_list({{0, 0, 0}}),
    klist_full({{0, 0, 0}}),
    kfrac_list_full({{0, 0, 0}})
{
    this->latvec_array[0] = {this->latvec.e11,this->latvec.e12,this->latvec.e13};
    this->latvec_array[1] = {this->latvec.e21,this->latvec.e22,this->latvec.e23};
    this->latvec_array[2] = {this->latvec.e31,this->latvec.e32,this->latvec.e33};
    this->set_period(1, 1, 1);

    // Gamma-only
    this->set_ibz_mapping({0}, {0});
}

void PeriodicBoundaryData::set_latvec(const std::vector<double> &latt_mat)
{
    if (latt_mat.size() != 9)
        throw std::runtime_error("Invalid lattice vectors input");

    this->latvec.e11 = latt_mat[0];
    this->latvec.e12 = latt_mat[1];
    this->latvec.e13 = latt_mat[2];
    this->latvec.e21 = latt_mat[3];
    this->latvec.e22 = latt_mat[4];
    this->latvec.e23 = latt_mat[5];
    this->latvec.e31 = latt_mat[6];
    this->latvec.e32 = latt_mat[7];
    this->latvec.e33 = latt_mat[8];

    this->latvec_array[0] = {this->latvec.e11,this->latvec.e12,this->latvec.e13};
    this->latvec_array[1] = {this->latvec.e21,this->latvec.e22,this->latvec.e23};
    this->latvec_array[2] = {this->latvec.e31,this->latvec.e32,this->latvec.e33};

    this->G = this->latvec.Inverse().Transpose();

    lattice_reset_ = true;
}

void PeriodicBoundaryData::set_latvec_and_G(const std::vector<double> &latt_mat,
                                            const std::vector<double> &recp_mat)
{
    this->set_latvec(latt_mat);

    if (recp_mat.size() != 9)
        throw std::runtime_error("Invalid reciprocal lattice vectors input");

    this->G.e11 = recp_mat[0];
    this->G.e12 = recp_mat[1];
    this->G.e13 = recp_mat[2];
    this->G.e21 = recp_mat[3];
    this->G.e22 = recp_mat[4];
    this->G.e23 = recp_mat[5];
    this->G.e31 = recp_mat[6];
    this->G.e32 = recp_mat[7];
    this->G.e33 = recp_mat[8];
    this->G /= TWO_PI;
}

void PeriodicBoundaryData::set_period(int nk1, int nk2, int nk3)
{
    if (nk1 <= 0 || nk2 <= 0 || nk3 <= 0)
        throw LIBRPA_RUNTIME_ERROR("Invalid BvK period input");

    this->period = {nk1, nk2, nk3};
    this->period_array = {nk1, nk2, nk3};
    this->Rlist = construct_R_grid(this->period);
}

void PeriodicBoundaryData::set_kgrids_kvec(int nk1, int nk2, int nk3, const std::vector<double> &kvecs)
{
    this->set_period(nk1, nk2, nk3);

    klist.clear();
    kfrac_list.clear();
    klist_full.clear();
    kfrac_list_full.clear();
    int n_k_points = nk1 * nk2 * nk3;
    for (int ik = 0; ik < n_k_points; ik++)
    {
        double kx = kvecs[ik * 3];
        double ky = kvecs[ik * 3 + 1];
        double kz = kvecs[ik * 3 + 2];
        Vector3_Order<double> kvec{kx, ky, kz};
        kvec /= TWO_PI;
        klist.emplace_back(kvec);
        auto kfrac = latvec * kvec;
        // clean up -0.000
        if (std::abs(kfrac.x) < 1e-8) kfrac.x = 0.0e0;
        if (std::abs(kfrac.y) < 1e-8) kfrac.y = 0.0e0;
        if (std::abs(kfrac.z) < 1e-8) kfrac.z = 0.0e0;
        kfrac_list.emplace_back(kfrac);
    }
    klist_full = klist;
    kfrac_list_full = kfrac_list;

    // Initialize the irreducible k points: same as the full ones
    std::vector<int> irk_point_id_mapping_in(n_k_points);
    std::iota(irk_point_id_mapping_in.begin(), irk_point_id_mapping_in.end(), 0);
    this->set_ibz_mapping(irk_point_id_mapping_in);
}

void PeriodicBoundaryData::set_irreducible_kgrids_kvec(
    int nk1, int nk2, int nk3,
    const std::vector<double> &kvecs_ibz,
    const std::vector<std::vector<Vector3_Order<double>>> &full_kstars)
{
    this->period = {nk1, nk2, nk3};
    this->period_array = {nk1, nk2, nk3};
    this->Rlist = construct_R_grid(this->period);

    const int n_ibz = static_cast<int>(kvecs_ibz.size() / 3);
    if (kvecs_ibz.size() != static_cast<std::size_t>(3 * n_ibz))
    {
        throw LIBRPA_RUNTIME_ERROR("invalid irreducible k-point vector buffer size");
    }
    if (full_kstars.size() != static_cast<std::size_t>(n_ibz))
    {
        throw LIBRPA_RUNTIME_ERROR("ABACUS full-BZ k-star count does not match IBZ k-point count");
    }

    klist.clear();
    kfrac_list.clear();
    klist_full.clear();
    kfrac_list_full.clear();
    klist_ibz.clear();
    kweight_ibz.clear();
    map_ibzk_weight.clear();
    map_irk_ks.clear();
    irk_point_id_mapping.clear();
    isymops.clear();

    const auto append_k = [this](const Vector3_Order<double> &kvec,
                                 std::vector<Vector3_Order<double>> &ks,
                                 std::vector<Vector3_Order<double>> &kfracs) {
        ks.emplace_back(kvec);
        auto kfrac = this->latvec * kvec;
        if (std::abs(kfrac.x) < 1e-8) kfrac.x = 0.0e0;
        if (std::abs(kfrac.y) < 1e-8) kfrac.y = 0.0e0;
        if (std::abs(kfrac.z) < 1e-8) kfrac.z = 0.0e0;
        kfracs.emplace_back(kfrac);
    };

    for (int ik = 0; ik != n_ibz; ++ik)
    {
        Vector3_Order<double> kvec{kvecs_ibz[ik * 3],
                                   kvecs_ibz[ik * 3 + 1],
                                   kvecs_ibz[ik * 3 + 2]};
        kvec /= TWO_PI;
        append_k(kvec, klist, kfrac_list);
    }

    klist_ibz = klist;
    const double full_count = static_cast<double>(this->get_n_cells_bvk());
    int n_full_members = 0;
    for (int ik_ibz = 0; ik_ibz != n_ibz; ++ik_ibz)
    {
        const auto &k_ibz = klist_ibz[ik_ibz];
        auto &members = map_irk_ks[k_ibz];
        for (const auto &k_member : full_kstars[ik_ibz])
        {
            if (std::find(members.begin(), members.end(), k_member) != members.end())
            {
                continue;
            }
            members.emplace_back(k_member);
            append_k(k_member, klist_full, kfrac_list_full);
            ++n_full_members;
        }
        if (members.empty())
        {
            members.emplace_back(k_ibz);
            append_k(k_ibz, klist_full, kfrac_list_full);
            ++n_full_members;
        }

        const double weight = static_cast<double>(members.size()) / full_count;
        map_ibzk_weight[k_ibz] = weight;
        kweight_ibz.emplace_back(weight);
        irk_point_id_mapping.emplace_back(ik_ibz);
    }

    if (n_full_members != this->get_n_cells_bvk())
    {
        throw LIBRPA_RUNTIME_ERROR(
            "ABACUS symmetry k-star members do not cover the full BZ grid: "
            + std::to_string(n_full_members) + " != " + std::to_string(this->get_n_cells_bvk()));
    }

    kgrid_no_symmetry_ = klist_ibz.size() == klist_full.size();
}

void PeriodicBoundaryData::set_ibz_mapping(const std::vector<int> &irk_point_id_mapping_in,
                                           const std::vector<int> &isymops_in)
{
    const size_t &n_k_points = this->klist.size();
    const double step = 1.0 / n_k_points;

    if (irk_point_id_mapping_in.size() != n_k_points)
    {
        throw LIBRPA_RUNTIME_ERROR("irk_point_id_mapping size not equal to n_k_points: " + std::to_string(n_k_points));
    }
    // TODO: implement using symmetry operation mapping
    this->isymops = isymops_in;

    irk_point_id_mapping = irk_point_id_mapping_in;

    klist_ibz.clear();
    klist_full = klist;
    kfrac_list_full = kfrac_list;
    map_ibzk_weight.clear();
    map_irk_ks.clear();
    for (size_t ik = 0; ik < n_k_points; ik++)
    {
        int ik_ibz = irk_point_id_mapping[ik];
        const auto &k = klist[ik];
        const auto &k_ibz = klist[ik_ibz];
        auto it = map_ibzk_weight.find(k_ibz);
        if (it == map_ibzk_weight.end())
        {
            klist_ibz.emplace_back(k_ibz);
            map_ibzk_weight.emplace(k_ibz, step);
        }
        else
        {
            it->second += step;
        }
        map_irk_ks[k_ibz].emplace_back(k);
    }

    kweight_ibz.clear();
    for (const auto &k_ibz: klist_ibz)
    {
        kweight_ibz.emplace_back(map_ibzk_weight.at(k_ibz));
    }

    kgrid_no_symmetry_ = klist_ibz.size() == klist.size();
}

int PeriodicBoundaryData::get_R_index(const Vector3_Order<int> &R) const
{
    return librpa_int::get_R_index(this->Rlist, R);
}

static int get_k_index_(const std::vector<Vector3_Order<double>> &klist, const Vector3_Order<double> &k)
{
    auto it = std::find(klist.cbegin(), klist.cend(), k);
    if (it != klist.cend()) return distance(klist.cbegin(), it);
    return -1;
}

int PeriodicBoundaryData::get_k_index_full(const Vector3_Order<double> &k) const
{
    return get_k_index_(this->klist_full, k);
}

int PeriodicBoundaryData::get_k_index_ibz(const Vector3_Order<double> &k) const
{
    return get_k_index_(this->klist_ibz, k);
}


std::vector<Vector3_Order<int>> construct_R_grid(const Vector3_Order<int> &period)
{
    // cout<<" begin to construct_R_grid"<<endl;
    std::vector<Vector3_Order<int>> R_grid;
    R_grid.clear();

    for (int x = -(period.x) / 2; x <= (period.x - 1) / 2; ++x)
        for (int y = -(period.y) / 2; y <= (period.y - 1) / 2; ++y)
            for (int z = -(period.z) / 2; z <= (period.z - 1) / 2; ++z)
                R_grid.push_back({x, y, z});

    return R_grid;
}

int get_R_index(const std::vector<Vector3_Order<int>> &Rlist, const Vector3_Order<int> &R)
{
    auto itr = std::find(Rlist.cbegin(), Rlist.cend(), R);
    if ( itr != Rlist.cend()) return distance(Rlist.cbegin(), itr);
    return -1;
}

bool is_gamma_point(const Vector3_Order<double> &kpt, double thres)
{
    return -thres < kpt.x && kpt.x < thres
        && -thres < kpt.y && kpt.y < thres
        && -thres < kpt.z && kpt.z < thres;
}

bool is_gamma_point(const Vector3_Order<int> &kpt_int)
{
    return kpt_int.x == 0 && kpt_int.y == 0 && kpt_int.z == 0;
}

Vector3_Order<int> find_nearest_bvk_cell(const Vector3<double> &coord_frac_I,
                                         const Vector3<double> &coord_frac_J,
                                         const Vector3_Order<int> &bvk_direct,
                                         const Vector3_Order<int> &period, const Matrix3 &latvec)
{
    auto distsq = std::numeric_limits<double>::max();
    Vector3<int> R_IJ;
    Vector3_Order<int> R_bvk;
    const auto &R = bvk_direct;
    const auto diff_IJ0 = coord_frac_I - coord_frac_J;
    for (int i = -1; i < 2; i++)
    {
        R_IJ.x = i * period.x + R.x;
        for (int j = -1; j < 2; j++)
        {
            R_IJ.y = j * period.y + R.y;
            for (int k = -1; k < 2; k++)
            {
                R_IJ.z = k * period.z + R.z;
                const auto diff = (diff_IJ0 - Vector3<double>(R_IJ.x, R_IJ.y, R_IJ.z)) * latvec;
                const auto norm2 = diff.norm2();
                if (norm2 < distsq)
                {
                    distsq = norm2;
                    R_bvk = R_IJ;
                }
            }
        }
    }
    return R_bvk;
}

std::vector<Vector3_Order<int>> find_nearest_bvk_cells(const Vector3<double> &coord_frac_I,
                                                       const Vector3<double> &coord_frac_J,
                                                       const Vector3_Order<int> &bvk_direct,
                                                       const Vector3_Order<int> &period,
                                                       const Matrix3 &latvec)
{
    auto distsq = std::numeric_limits<double>::max();
    std::vector<Vector3_Order<int>> R_bvks;
    Vector3<int> R_IJ;
    const auto &R = bvk_direct;
    const auto diff_IJ0 = coord_frac_I - coord_frac_J;
    for (int i = -1; i < 2; i++)
    {
        R_IJ.x = i * period.x + R.x;
        for (int j = -1; j < 2; j++)
        {
            R_IJ.y = j * period.y + R.y;
            for (int k = -1; k < 2; k++)
            {
                R_IJ.z = k * period.z + R.z;
                const auto diff = (diff_IJ0 - Vector3<double>(R_IJ.x, R_IJ.y, R_IJ.z)) * latvec;
                const auto norm2 = diff.norm2();
                const auto tol = 1.0e-10 * std::max(1.0, std::max(std::abs(norm2), std::abs(distsq)));
                if (norm2 + tol < distsq)
                {
                    distsq = norm2;
                    R_bvks.clear();
                    R_bvks.emplace_back(R_IJ);
                }
                else if (std::abs(norm2 - distsq) <= tol)
                {
                    R_bvks.emplace_back(R_IJ);
                }
            }
        }
    }
    return R_bvks;
}

// int kv_nmp[3] = {1, 1, 1};
// Vector3<double> *kvec_c;
// std::vector<Vector3_Order<double>> klist;
// std::vector<Vector3_Order<double>> klist_ibz;
// std::vector<Vector3_Order<double>> kfrac_list;
// std::vector<int> irk_point_id_mapping;
// std::map<Vector3_Order<double>, std::vector<Vector3_Order<double>>> map_irk_ks;
// Matrix3 latvec;
// std::array<std::array<double, 3>, 3> lat_array;
// Matrix3 G;

}
