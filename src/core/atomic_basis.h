/*!
 * @file atomic_basis.h
 * @brief Utilities for handling atomic basis functions
 */
#pragma once
#include <algorithm>
#include <cassert>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "atom.h"
#include "librpa_enums.h"

namespace librpa_int {

typedef int locid_t;
typedef std::size_t gloid_t;
typedef std::pair<locid_t, locid_t> locid_ap_t;
typedef std::pair<gloid_t, gloid_t> gloid_ap_t;

/*! @brief Basis ordering, real-spherical-harmonic, and Bloch-sum convention.
 *
 * The Bloch-sum phase convention is parameterized as
 * \f[
 * \phi^{\mathbf{K}}_{U\mu}(r) =
 *     \frac{1}{\sqrt{N_k}} \sum_{\mathbf{R}}
 *     \exp[i\,\mathrm{bloch\_phase}\,\mathbf{k}\cdot
 *          (\mathbf{R} + \mathrm{bloch\_ratom}\,\mathbf{s}_U)]
 *     \phi^R_{U\mu}(r)
 * \f]
 * where
 * \f[
 * \phi^R_{U\mu}(r) =
 *     \phi_{\mu}(r - \mathbf{s}_U - \mathbf{R}).
 * \f]
 */
struct BasisConvention
{
    //! Bloch-sum phase sign; unset means not specified.
    int bloch_phase = LIBRPA_UNSET;
    //! Coefficient of atom position in the Bloch-sum phase; unset means not specified.
    int bloch_ratom = LIBRPA_UNSET;
    LibrpaAngularOrder order = LIBRPA_ANGULAR_ORDER_UNSET;
    LibrpaRshCoeff coeff_m_negative = LIBRPA_RSH_COEFF_UNSET;
    LibrpaRshCoeff coeff_m_positive = LIBRPA_RSH_COEFF_UNSET;
};

inline bool is_basis_rsh_convention_set(const BasisConvention &bconv)
{
    return bconv.order != LIBRPA_ANGULAR_ORDER_UNSET
           && bconv.coeff_m_negative != LIBRPA_RSH_COEFF_UNSET
           && bconv.coeff_m_positive != LIBRPA_RSH_COEFF_UNSET;
}

inline bool is_basis_bloch_convention_set(const BasisConvention &bconv)
{
    return bconv.bloch_phase != LIBRPA_UNSET
           && bconv.bloch_ratom != LIBRPA_UNSET;
}

inline bool is_basis_convention_set(const BasisConvention &bconv)
{
    return is_basis_bloch_convention_set(bconv)
           && is_basis_rsh_convention_set(bconv);
}

/*!
 * @brief Basis shell layout of one atomic species.
 *
 * l_shells[ishell] is the angular momentum of radial shell ishell in basis order.
 * shell_indices[l] stores the radial-shell indices with angular momentum l.
 * This does not describe a universal basis order. Current rotation assembly
 * groups by shell_indices and uses l_shells to place each radial-shell block.
 */
struct SpeciesBasisLayout
{
    std::string label;
    std::vector<int> l_shells;
    std::map<int, std::vector<int>> shell_indices;
    std::vector<int> shell_offsets;
    std::map<int, int> shell_counts;
    int n_shell = 0;
    int n_ao = 0;
    int max_l = 0;

    SpeciesBasisLayout()
        : label(), l_shells(), shell_indices(), shell_offsets(), shell_counts(), n_shell(0), n_ao(0), max_l(0) {};
    SpeciesBasisLayout(const std::string& label_in, const std::vector<int>& l_shells_in): label(label_in), l_shells(l_shells_in)
    {
        compute_map();
    }

    void set(const std::vector<int> &l_shells_in)
    {
        l_shells = l_shells_in;
        compute_map();
    }

    bool is_shell_available() const noexcept { return n_shell > 0; }

private:
    void compute_map()
    {
        shell_indices.clear();
        shell_counts.clear();
        shell_offsets.clear();

        n_shell = l_shells.size();
        shell_offsets.reserve(n_shell);
        n_ao = 0;
        max_l = 0;

        for (int ishell = 0; ishell < n_shell; ++ishell)
        {
            const int l = l_shells[ishell];
            if (l < 0)
            {
                throw std::invalid_argument("SpeciesBasisLayout cannot contain negative l");
            }
            max_l = std::max(l, max_l);
            shell_indices[l].push_back(static_cast<int>(ishell));
            ++shell_counts[l];
            shell_offsets.push_back(n_ao);
            n_ao += 2 * l + 1;
        }
        // populate missing l channels, just for convenience
        for (int l = 0; l <= max_l; l++)
        {
            if (shell_indices.find(l) == shell_indices.cend())
            {
                shell_indices[l] = {};
                shell_counts[l] = 0;
            }
        }
    }
};

/*! @class
 * @brief Object to handle atomic basis
 */
class AtomicBasis
{
private:
    bool initialized_;
public:
    std::string label;
private:
    std::vector<std::size_t> nbs_;
    std::vector<std::size_t> part_range_;
    std::vector<int> glo2iat_;
    std::vector<std::size_t> glo2loc_;
    std::vector<std::vector<int>> l_shells_;
    int max_l_;
    // std::vector<int> irad_;
    // std::vector<int> l_;
    // std::vector<int> m_;

    void initialize();
public:
    //! Total number of atoms
    std::size_t n_atoms;
    //! Total number of basis functions
    std::size_t nb_total;

    // Constructors
    AtomicBasis(): initialized_(false),
                   nbs_(), part_range_(), glo2iat_(), glo2loc_(), l_shells_(), max_l_(-1),
                   n_atoms(0), nb_total(0) {};
    AtomicBasis(const std::vector<std::size_t>& nbs);
    AtomicBasis(const std::vector<int>& atom_species,
                const std::map<int, std::size_t>& map_species_nb);
    AtomicBasis(const std::map<std::size_t, std::size_t>& iatom_nbs);

    //! Set number of basis functions for each atom
    void set(const std::vector<std::size_t>& nbs);
    void set(const std::vector<int>& atom_species,
             const std::map<int, std::size_t>& map_species_nb);
    void set(const std::map<std::size_t, std::size_t>& iatom_nbs);
    void set_l_shells(const std::vector<std::vector<int>>& l_shells);

    //! Get the global index of a certain basis function of an atom
    inline std::size_t get_global_index(const int& i_atom, const std::size_t& i_loc_b) const noexcept
    {
        return part_range_[i_atom] + i_loc_b;
    }

    //! Get the size of submatrix corresponding to atom pair i and j
    std::size_t get_pair_matrix_size(const int& i_atom, const int& j_atom) const noexcept
    {
        return nbs_[i_atom] * nbs_[j_atom];
    };

    //! Get the global indices of all basis functions on an atom
    std::vector<std::size_t> get_global_indices(const int& i_atom) const;

    //! Get the index of atom on which a certain basis function is located
    inline int get_i_atom(const std::size_t& i_glo_b) const noexcept
    {
        assert (i_glo_b < nb_total && i_glo_b >= 0);
        return glo2iat_[i_glo_b];
    }

    //! Get the local indices of a basis function from its global index
    inline void get_local_index(const std::size_t& i_glo_b, int& i_atom, int& i_loc_b) const
    {
        i_atom = get_i_atom(i_glo_b);
        i_loc_b = as_int(glo2loc_[i_glo_b]);
    }
    inline void get_local_index(const std::size_t& i_glo_b, size_t& i_atom, size_t& i_loc_b) const
    {
        i_atom = as_size(get_i_atom(i_glo_b));
        i_loc_b = glo2loc_[i_glo_b];
    }
    inline int get_local_index(const std::size_t& i_glo_b, const int& i_atom) const noexcept
    {
        // return i_glo_b - part_range_[i_atom];
        return as_int(glo2loc_[i_glo_b]);
    }

    inline std::pair<int, int> get_local_index(const std::size_t& i_glo_b) const noexcept
    {
        // int i_atom, i_loc_b;
        // this->get_local_index(i_glo_b, i_atom, i_loc_b);
        return {glo2iat_[i_glo_b], as_int(glo2loc_[i_glo_b])};
    }

    inline std::size_t get_atom_nb(int i_atom) const { return nbs_.at(as_size(i_atom)); }
    inline std::size_t operator[](int i_atom) const { return nbs_.at(as_size(i_atom)); }
    inline std::vector<std::size_t> get_atom_nbs() const noexcept { return nbs_; }
    inline bool has_l_shells() const noexcept { return l_shells_.size() == n_atoms && n_atoms > 0; }
    inline const std::vector<std::vector<int>>& get_l_shells() const noexcept { return l_shells_; }
    inline const std::vector<int>& get_l_shells(int i_atom) const noexcept { return l_shells_.at(i_atom); }
    inline int get_max_l() const noexcept { return max_l_; }
    inline const std::vector<std::size_t>& get_part_range() const noexcept { return part_range_; }
    inline bool initialized() const noexcept { return initialized_; }
};

bool same_species_basis_layout(const SpeciesBasisLayout &lhs,
                               const SpeciesBasisLayout &rhs);

SpeciesBasisLayout species_basis_layout_from_atom(const AtomicBasis &basis,
                                                  atom_t atom,
                                                  int atom_type);

void condense_species_basis_layouts(const AtomicBasis &basis,
                                    const std::map<atom_t, int> &atom_to_type,
                                    std::map<int, SpeciesBasisLayout> &layouts);

bool type_layouts_have_shells(const std::map<int, SpeciesBasisLayout> &layouts);

inline bool operator==(const AtomicBasis &ab1, const AtomicBasis &ab2)
{
    if (ab1.n_atoms != ab2.n_atoms) return false;
    if (ab1.nb_total != ab2.nb_total) return false;
    const int n_atoms = ab1.n_atoms;
    for (int ia = 0; ia < n_atoms; ia++)
    {
        if (ab1.get_atom_nb(ia) != ab2.get_atom_nb(ia)) return false;
    }
    return true;
}

inline bool operator!=(const AtomicBasis &ab1, const AtomicBasis &ab2)
{
    return !(ab1 == ab2);
}

/*!
 * @brief Get the size of matrix block corresponding to an atom pair.
 *
 * @param  [in]  atbasis_r  AtomicBasis object for row
 * @param  [in]  at_r       Index of atom for row
 * @param  [in]  atbasis_c  AtomicBasis object for column
 * @param  [in]  at_c       Index of atom for column
 *
 * @retval       size       Number of elements in the atom-pair matrix block.
 */
inline std::size_t get_pair_matrix_size(const AtomicBasis& atbasis_r, const int& at_r,
                                        const AtomicBasis& atbasis_c, const int& at_c)
{
    return atbasis_r.get_atom_nb(at_r) * atbasis_c.get_atom_nb(at_c);
}

/*!
 * @brief Get the size of matrix block corresponding to an atom pair.
 *
 * @param  [in]  atbasis    AtomicBasis object for both row and column
 * @param  [in]  at_r       Index of atom for row
 * @param  [in]  at_c       Index of atom for column
 *
 * @retval       size       Number of elements in the atom-pair matrix block.
 */
inline std::size_t get_pair_matrix_size(const AtomicBasis& atbasis, const int& at_r, const int& at_c)
{
    return get_pair_matrix_size(atbasis, at_r, atbasis, at_c);
}

/*!
 * @brief Get the 2D indices for matrix elements in the atom-pair blocks
 *        corresponding to requested atom pairs.
 *
 * @param  [in]  atbasis_r  AtomicBasis object for row
 * @param  [in]  atbasis_c  AtomicBasis object for column
 * @param  [in]  IJs        Atomic pairs to request
 * @param  [in]  row_fast   Flag to set the row basis index faster
 * @param  [in]  sort_fast  Flag to sort with the faster dimension.
 *                          The indices are continuous within one atom-pair block.
 *                          When sort_fast is true, the indices will be sorted and
 *                          hence the continuity may break.
 *
 * @retval       indices    2D indices of elements in the request atom-pair blocks
 */
std::vector<std::pair<size_t, size_t>> get_2d_mat_indices_atpair(const AtomicBasis &atbasis_r,
                                                                 const AtomicBasis &atbasis_c,
                                                                 const std::vector<atpair_t> &IJs,
                                                                 const bool row_fast,
                                                                 const bool sort_fast = false);

/*!
 * @brief Get the 1D indices for matrix elements in the atom-pair blocks
 *        corresponding to requested atom pairs.
 *
 * @param  [in]  atbasis_r  AtomicBasis object for row
 * @param  [in]  atbasis_c  AtomicBasis object for column
 * @param  [in]  IJs        Atomic pairs to request
 * @param  [in]  row_fast   Flag to set the row basis index faster
 * @param  [in]  row_major  Flag to compute 1D indices in row major (C-style)
 * @param  [in]  sort_fast  Flag to sort with the faster dimension.
 *                          The indices are continuous within one atom-pair block.
 *                          When sort_fast is true, the indices will be sorted and
 *                          hence the continuity may break.
 *
 * @retval       indices    1D indices of elements
 */
std::vector<size_t> get_1d_mat_indices_atpair(const AtomicBasis &atbasis_r,
                                              const AtomicBasis &atbasis_c,
                                              const std::vector<atpair_t> &IJs,
                                              const bool row_fast,
                                              const bool row_major,
                                              const bool sort_fast = false);

// extern AtomicBasis atomic_basis_wfc;
// extern AtomicBasis atomic_basis_abf;

} // namespace librpa_int
