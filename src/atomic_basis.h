/*
 * @file basis.h
 */
#pragma once
#include <map>
#include <utility>
#include <vector>

#include "atoms.h"

namespace LIBRPA {

class AtomicBasis
{
private:
    std::vector<std::size_t> nbs_;
    std::vector<std::size_t> part_range;
    void initialize();
public:
    atom_t n_atoms;
    std::size_t nb_total;
    AtomicBasis(): nbs_(), part_range(), n_atoms(0), nb_total(0) {};
    AtomicBasis(const std::vector<std::size_t>& nbs);
    AtomicBasis(const std::vector<atom_t>& atoms,
                const std::map<atom_t, std::size_t>& map_atom_nb);
    AtomicBasis(const std::map<atom_t, std::size_t>& iatom_nbs);
    void set(const std::vector<std::size_t>& nbs);
    void set(const std::map<atom_t, std::size_t>& iatom_nbs);
    int get_global_index(const atom_t& i_atom, const int& i_loc_b) const;
    atom_t get_i_atom(const int& i_glo_b) const;
    void get_local_index(const int& i_glo_b, atom_t& i_atom, int& i_loc_b) const;
    int get_local_index(const int& i_glo_b, const atom_t& i_atom) const;
    std::pair<atom_t, int> get_local_index(const int& i_glo_b) const;
    std::size_t get_atom_nb(const atom_t& i_atom) const { return nbs_[i_atom]; }
    std::vector<std::size_t> get_atom_nbs() const { return nbs_; }
    const std::vector<std::size_t>& get_part_range() const { return part_range; }
};

extern AtomicBasis atomic_basis_wfc;
extern AtomicBasis atomic_basis_abf;

} // namespace LIBRPA
