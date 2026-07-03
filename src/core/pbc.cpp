#include "pbc.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "../utils/constants.h"
#include "../utils/error.h"
#include "../io/global_io.h"

namespace librpa_int {

static const double A = 1e5;  // 10000 nm
static const double INVA = 1.0 / A;
static const double KPT_MATCH_TOL = 1e-5;
static const double KWEIGHT_MATCH_TOL = 1e-6;
static const double KWEIGHT_SUM_TOL = 1e-12;

static Vector3_Order<double> normalize_fractional_kpoint(const Vector3_Order<double> &kfrac)
{
    return restrict_fractional_coordinate(kfrac, KPT_MATCH_TOL);
}

static Vector3_Order<double> kvec_from_fractional(const Matrix3 &G,
                                                  const Vector3_Order<double> &kfrac)
{
    return kfrac * G;
}

std::vector<Vector3_Order<double>> build_uniform_kmesh_frac(const Vector3_Order<int> &period)
{
    std::vector<Vector3_Order<double>> kpoints;
    kpoints.reserve(static_cast<std::size_t>(period.x * period.y * period.z));
    for (int i = 0; i != period.x; ++i)
        for (int j = 0; j != period.y; ++j)
            for (int k = 0; k != period.z; ++k)
                kpoints.push_back({static_cast<double>(i) / period.x,
                                   static_cast<double>(j) / period.y,
                                   static_cast<double>(k) / period.z});
    return kpoints;
}

static int find_matching_kpoint(const std::vector<Vector3_Order<double>> &kpoints,
                                const Vector3_Order<double> &target)
{
    int matched_index = -1;
    for (std::size_t ik = 0; ik != kpoints.size(); ++ik)
    {
        if (!nearly_integer_vector(kpoints[ik] - target, KPT_MATCH_TOL))
        {
            continue;
        }
        if (matched_index >= 0)
        {
            throw LIBRPA_RUNTIME_ERROR("Ambiguous k-point match in BvK grid");
        }
        matched_index = static_cast<int>(ik);
    }
    return matched_index;
}

// Initialized as a huge box to emulate isolated case
PeriodicBoundaryData::PeriodicBoundaryData():
    lattice_reset_(false),
    kgrid_no_symmetry_(true),
    latvec(A, 0, 0, 0, A, 0, 0, 0, A),
    G(INVA, 0, 0, 0, INVA, 0, 0, 0, INVA),
    period(1, 1, 1),
    period_array({1, 1, 1}),
    Rlist({{0, 0, 0}}),
    klist({{0, 0, 0}}),
    kfrac_list({{0, 0, 0}}),
    weight_k({1.0}),
    klist_full({{0, 0, 0}}),
    kfrac_list_full({{0, 0, 0}}),
    k_to_kfull({0}),
    kfull_to_k({0}),
    kfull_to_k_relation({KFullToKRelation::DIRECT})
{
    this->latvec_array[0] = {this->latvec.e11,this->latvec.e12,this->latvec.e13};
    this->latvec_array[1] = {this->latvec.e21,this->latvec.e22,this->latvec.e23};
    this->latvec_array[2] = {this->latvec.e31,this->latvec.e32,this->latvec.e33};
    this->set_period(1, 1, 1);

    // Gamma-only
    this->set_kq_mapping({0}, {0});
}

void PeriodicBoundaryData::reset_k_info()
{
    klist.clear();
    kfrac_list.clear();
    weight_k.clear();
    klist_full.clear();
    kfrac_list_full.clear();
    k_to_kfull.clear();
    kfull_to_k.clear();
    kfull_to_k_relation.clear();
    kgrid_uses_time_reversal = false;
    klist_coul.clear();
    weight_q.clear();
    map_q_weight.clear();
    irk_point_id_mapping.clear();
    isymops.clear();
    map_irk_ks.clear();
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
    if (this->period == Vector3_Order<int>{nk1, nk2, nk3})
    {
        return;
    }

    this->period = {nk1, nk2, nk3};
    this->period_array = {nk1, nk2, nk3};
    this->Rlist = construct_R_grid(this->period);
    reset_k_info();
}

void PeriodicBoundaryData::set_kgrids_kvec(int nk1, int nk2, int nk3,
                                           const std::vector<double> &kvecs,
                                           const std::vector<double> &kweights)
{
    this->set_period(nk1, nk2, nk3);
    reset_k_info();

    const int n_k_points = static_cast<int>(kvecs.size() / 3);
    if (kvecs.size() != static_cast<std::size_t>(3 * n_k_points))
        throw LIBRPA_RUNTIME_ERROR("invalid k-point vector buffer size");
    if (n_k_points <= 0)
        throw LIBRPA_RUNTIME_ERROR("empty k-point vector buffer");
    if (!kweights.empty() && kweights.size() != static_cast<std::size_t>(n_k_points))
        throw LIBRPA_RUNTIME_ERROR("k-point weight count does not match loaded k-point count");
    int n_shifted_kpoints = 0;
    for (int ik = 0; ik < n_k_points; ik++)
    {
        double kx = kvecs[ik * 3];
        double ky = kvecs[ik * 3 + 1];
        double kz = kvecs[ik * 3 + 2];
        Vector3_Order<double> kvec{kx, ky, kz};
        kvec /= TWO_PI;
        const auto kfrac = normalize_fractional_kpoint(latvec * kvec);
        const auto kvec_normalized = kvec_from_fractional(G, kfrac);
        if ((kvec_normalized - kvec).norm() > KPT_MATCH_TOL)
        {
            ++n_shifted_kpoints;
        }
        kfrac_list.emplace_back(kfrac);
        klist.emplace_back(kvec_normalized);
    }
    if (n_shifted_kpoints > 0 && global::ofs_myid.is_open())
    {
        global::ofs_myid << "Shifted " << n_shifted_kpoints
                         << " loaded k-points into the [0,1) fractional convention" << std::endl;
    }

    k_to_kfull.assign(static_cast<std::size_t>(n_k_points), -1);
    const auto canonical_kfrac_list_full = build_uniform_kmesh_frac(this->period);
    kfull_to_k.assign(canonical_kfrac_list_full.size(), -1);
    kfull_to_k_relation.assign(canonical_kfrac_list_full.size(), KFullToKRelation::NONE);
    kgrid_uses_time_reversal = false;

    if (n_k_points == this->get_n_cells_bvk())
    {
        std::vector<int> canonical_seen(canonical_kfrac_list_full.size(), -1);
        for (int ik = 0; ik != n_k_points; ++ik)
        {
            const int ifull = find_matching_kpoint(canonical_kfrac_list_full, kfrac_list[ik]);
            if (ifull < 0)
            {
                throw LIBRPA_RUNTIME_ERROR("Loaded k-point is not on the BvK full grid");
            }
            if (canonical_seen[static_cast<std::size_t>(ifull)] >= 0)
            {
                throw LIBRPA_RUNTIME_ERROR("Duplicate loaded k-points map to the same BvK full-grid point");
            }
            canonical_seen[static_cast<std::size_t>(ifull)] = ik;
            k_to_kfull[static_cast<std::size_t>(ik)] = ik;
            kfull_to_k[static_cast<std::size_t>(ik)] = ik;
            kfull_to_k_relation[static_cast<std::size_t>(ik)] = KFullToKRelation::DIRECT;
        }
        if (std::find(canonical_seen.cbegin(), canonical_seen.cend(), -1)
            != canonical_seen.cend())
        {
            throw LIBRPA_RUNTIME_ERROR("Loaded full k-point list does not cover the BvK full grid");
        }
        klist_full = klist;
        kfrac_list_full = kfrac_list;
    }
    else
    {
        kfrac_list_full = canonical_kfrac_list_full;
        klist_full.clear();
        klist_full.reserve(kfrac_list_full.size());
        for (const auto &kfrac : kfrac_list_full)
        {
            klist_full.push_back(kvec_from_fractional(G, kfrac));
        }

        for (int ik = 0; ik != n_k_points; ++ik)
        {
            const int ifull = find_matching_kpoint(kfrac_list_full, kfrac_list[ik]);
            if (ifull < 0)
            {
                throw LIBRPA_RUNTIME_ERROR("Loaded k-point is not on the BvK full grid");
            }
            if (kfull_to_k[static_cast<std::size_t>(ifull)] >= 0)
            {
                throw LIBRPA_RUNTIME_ERROR("Duplicate loaded k-points map to the same BvK full-grid point");
            }
            k_to_kfull[static_cast<std::size_t>(ik)] = ifull;
            kfull_to_k[static_cast<std::size_t>(ifull)] = ik;
            kfull_to_k_relation[static_cast<std::size_t>(ifull)] = KFullToKRelation::DIRECT;
            kfrac_list_full[static_cast<std::size_t>(ifull)] = kfrac_list[ik];
            klist_full[static_cast<std::size_t>(ifull)] = klist[ik];
        }

        bool has_missing_full_kpoint = false;
        bool all_missing_full_kpoints_have_time_reversal = true;
        for (std::size_t ifull = 0; ifull != kfrac_list_full.size(); ++ifull)
        {
            if (kfull_to_k[ifull] >= 0)
            {
                continue;
            }
            has_missing_full_kpoint = true;
            const auto kfrac_minus = normalize_fractional_kpoint(-kfrac_list_full[ifull]);
            const int ik = find_matching_kpoint(kfrac_list, kfrac_minus);
            if (ik >= 0)
            {
                kfull_to_k[ifull] = ik;
                kfull_to_k_relation[ifull] = KFullToKRelation::TIME_REVERSAL;
            }
            else
            {
                all_missing_full_kpoints_have_time_reversal = false;
            }
        }
        kgrid_uses_time_reversal =
            has_missing_full_kpoint && all_missing_full_kpoints_have_time_reversal;
    }

    std::vector<double> computed_kweights;
    if (static_cast<int>(klist.size()) == this->get_n_cells_bvk()
        || kgrid_uses_time_reversal)
    {
        computed_kweights.assign(static_cast<std::size_t>(n_k_points), 0.0);
        const double step = 1.0 / this->get_n_cells_bvk();
        for (const int ik : kfull_to_k)
        {
            if (ik < 0)
            {
                throw LIBRPA_RUNTIME_ERROR("Cannot compute k-point weights from incomplete full-grid mapping");
            }
            computed_kweights[static_cast<std::size_t>(ik)] += step;
        }
    }
    if (kweights.empty())
    {
        weight_k = std::move(computed_kweights);
    }
    else
    {
        weight_k = kweights;
        double weight_sum = 0.0;
        for (const double weight : weight_k)
        {
            if (!std::isfinite(weight) || weight < 0.0)
            {
                throw LIBRPA_RUNTIME_ERROR("invalid k-point weight");
            }
            weight_sum += weight;
        }
        if (weight_sum <= KWEIGHT_SUM_TOL)
        {
            throw LIBRPA_RUNTIME_ERROR("k-point weights sum to zero");
        }
        for (double &weight : weight_k)
        {
            weight /= weight_sum;
        }
        for (std::size_t ik = 0; ik != computed_kweights.size(); ++ik)
        {
            if (std::abs(weight_k[ik] - computed_kweights[ik]) > KWEIGHT_MATCH_TOL)
            {
                throw LIBRPA_RUNTIME_ERROR("parsed k-point weights do not match the BvK mapping");
            }
        }
    }

    // Initialize the irreducible k points: same as the full ones
    std::vector<int> irk_point_id_mapping_in(n_k_points);
    std::iota(irk_point_id_mapping_in.begin(), irk_point_id_mapping_in.end(), 0);
    this->set_kq_mapping(irk_point_id_mapping_in);
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
    weight_k.clear();
    klist_full.clear();
    kfrac_list_full.clear();
    klist_coul.clear();
    weight_q.clear();
    map_q_weight.clear();
    map_irk_ks.clear();
    k_to_kfull.clear();
    kfull_to_k.clear();
    kfull_to_k_relation.clear();
    kgrid_uses_time_reversal = false;
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

    klist_coul = klist;
    const double full_count = static_cast<double>(this->get_n_cells_bvk());
    int n_full_members = 0;
    for (int ik_ibz = 0; ik_ibz != n_ibz; ++ik_ibz)
    {
        const auto &k_ibz = klist_coul[ik_ibz];
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
        weight_k.emplace_back(weight);
        map_q_weight[k_ibz] = weight;
        weight_q.emplace_back(weight);
        irk_point_id_mapping.emplace_back(ik_ibz);
    }

    if (n_full_members != this->get_n_cells_bvk())
    {
        throw LIBRPA_RUNTIME_ERROR(
            "ABACUS symmetry k-star members do not cover the full BZ grid: "
            + std::to_string(n_full_members) + " != " + std::to_string(this->get_n_cells_bvk()));
    }
    k_to_kfull.assign(klist.size(), -1);
    kfull_to_k.assign(klist_full.size(), -1);
    kfull_to_k_relation.assign(klist_full.size(), KFullToKRelation::NONE);
    for (std::size_t ik = 0; ik != kfrac_list.size(); ++ik)
    {
        const int ifull = find_matching_kpoint(kfrac_list_full, kfrac_list[ik]);
        if (ifull >= 0)
        {
            k_to_kfull[ik] = ifull;
            kfull_to_k[static_cast<std::size_t>(ifull)] = static_cast<int>(ik);
            kfull_to_k_relation[static_cast<std::size_t>(ifull)] = KFullToKRelation::DIRECT;
        }
    }

    kgrid_no_symmetry_ = klist_coul.size() == klist_full.size();
}

void PeriodicBoundaryData::set_kq_mapping(const std::vector<int> &map_q_ks,
                                          const std::vector<int> &isymops_in)
{
    const size_t &n_k_points = this->klist.size();
    const bool have_kweights = !weight_k.empty();

    if (map_q_ks.size() != n_k_points)
    {
        throw LIBRPA_RUNTIME_ERROR("k-to-q mapping size does not match n_k_points: "
                                  + std::to_string(n_k_points));
    }
    if (have_kweights && weight_k.size() != n_k_points)
    {
        throw LIBRPA_RUNTIME_ERROR("k-point weight count does not match n_k_points");
    }
    // TODO: implement using symmetry operation mapping
    this->isymops = isymops_in;

    irk_point_id_mapping = map_q_ks;

    klist_coul.clear();
    map_q_weight.clear();
    map_irk_ks.clear();
    for (size_t ik = 0; ik < n_k_points; ik++)
    {
        const int iq = irk_point_id_mapping[ik];
        if (iq < 0 || iq >= static_cast<int>(n_k_points))
        {
            throw LIBRPA_RUNTIME_ERROR("k-to-q mapping index out of k-point range");
        }
        const auto &k = klist[ik];
        const auto &q = klist[iq];
        if (map_irk_ks.find(q) == map_irk_ks.end())
        {
            klist_coul.emplace_back(q);
        }
        if (have_kweights)
        {
            const double weight = weight_k[ik];
            if (!std::isfinite(weight) || weight < 0.0)
            {
                throw LIBRPA_RUNTIME_ERROR("invalid k-point weight");
            }
            map_q_weight[q] += weight;
        }
        map_irk_ks[q].emplace_back(k);
    }

    weight_q.clear();
    if (have_kweights)
    {
        for (const auto &q: klist_coul)
        {
            weight_q.emplace_back(map_q_weight.at(q));
        }
    }

    kgrid_no_symmetry_ =
        static_cast<int>(klist.size()) == this->get_n_cells_bvk()
        && klist_coul.size() == klist.size();
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
    return get_k_index_(this->klist_coul, k);
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
// std::vector<Vector3_Order<double>> klist_coul;
// std::vector<Vector3_Order<double>> kfrac_list;
// std::vector<int> irk_point_id_mapping;
// std::map<Vector3_Order<double>, std::vector<Vector3_Order<double>>> map_irk_ks;
// Matrix3 latvec;
// std::array<std::array<double, 3>, 3> lat_array;
// Matrix3 G;

}
