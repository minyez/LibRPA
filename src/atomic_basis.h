/*!
 * @file atomic_basis.h
 * @brief Utilities for handling atomic basis functions
 */
#pragma once
#include <map>
#include <utility>
#include <vector>

namespace LIBRPA {

/*! @enum
 *
 *  Control for phase and order convention for basis functions in the same shell
 */
enum class AngularConvention
{
    AIMS = 0,
    ABACUS = 1,
    OPENMX = 2,
    PYSCF = 3,
};

/*! @class
 * @brief Object to handle atomic basis
 */
class AtomicBasis
{
private:
    std::vector<std::size_t> nbs_;
    std::vector<std::size_t> part_range;
    void initialize();
public:
    //! Total number of atoms
    std::size_t n_atoms;
    //! Total number of basis functions
    std::size_t nb_total;

    // Constructors
    AtomicBasis(): nbs_(), part_range(), n_atoms(0), nb_total(0) {};
    AtomicBasis(const std::vector<std::size_t>& nbs);
    AtomicBasis(const std::vector<int>& atoms,
                const std::map<int, std::size_t>& map_atom_nb);
    AtomicBasis(const std::map<std::size_t, std::size_t>& iatom_nbs);

    //! Set number of basis functions for each atom
    void set(const std::vector<std::size_t>& nbs);
    void set(const std::map<std::size_t, std::size_t>& iatom_nbs);

    //! Get the global index of a certain basis function of an atom
    std::size_t get_global_index(const int& i_atom, const std::size_t& i_loc_b) const;

    //! Get the index of atom on which a certain basis function is located
    int get_i_atom(const std::size_t& i_glo_b) const;

    //! Get the local indices of a basis function from its global index
    void get_local_index(const std::size_t& i_glo_b, int& i_atom, int& i_loc_b) const;
    int get_local_index(const std::size_t& i_glo_b, const int& i_atom) const;
    std::pair<int, int> get_local_index(const std::size_t& i_glo_b) const;

    std::size_t get_atom_nb(const int& i_atom) const { return nbs_[i_atom]; }
    std::vector<std::size_t> get_atom_nbs() const { return nbs_; }
    const std::vector<std::size_t>& get_part_range() const { return part_range; }
};

extern AtomicBasis atomic_basis_wfc;
extern AtomicBasis atomic_basis_abf;

} // namespace LIBRPA
