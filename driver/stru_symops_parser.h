#pragma once

#include <string>
#include <vector>

namespace driver
{

//! Parsed optional symmetry tail of a stru_out file.
//!
//! Layout after the atom section:
//!   [legacy k-point section]            (skipped, see reader_structure.cpp history)
//!   n_symops (row|col)                  header of the spatial operation block
//!   (9 int rot + 3 double trans) x n_symops
//!   spin_symmetry grey source           optional header of the spin-operation block
//!   (anti [8 doubles U_s]) x n_symops   per-operation antiunitary flag and, only
//!                                       when source == 1 (ExplicitSpinSpace), the
//!                                       row-major 2x2 SU(2) matrix as (re,im) pairs
//!
//! source: 0 = Identity, 1 = ExplicitSpinSpace, 2 = DerivedFromSpatialSOC,
//! matching librpa_set_symmetry_spin_operations.
struct StruSymopsTail
{
    int n_symops = 0;
    int row_conv = 0;
    std::vector<int> rotmats;
    std::vector<double> trans;

    bool has_spin_block = false;
    int grey_group = 0;
    int spin_source = 0;
    std::vector<int> antiunitary;
    std::vector<double> spin_u;   //!< 8 doubles per op, only when spin_source == 1
};

//! Parse the symmetry tail from whitespace-separated tokens (everything after the
//! atom section). `n_kpoints` is the loaded SCF k-point count used to disambiguate
//! the legacy k-point section (pass 0 when unknown). Returns false when the tail is
//! empty or contains only the legacy k-point section (no symmetry operations).
//! Throws std::runtime_error on malformed input.
bool parse_stru_symops_tail(const std::vector<std::string> &tokens,
                            const std::string &file_path,
                            int n_kpoints,
                            StruSymopsTail &tail);

} // namespace driver
