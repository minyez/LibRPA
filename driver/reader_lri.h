#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "../src/core/ri.h"

size_t read_Cs(const std::string &dir_path, double threshold,
               const std::vector<librpa_int::atpair_t> &local_atpair);

size_t read_Cs_evenly_distribute(const std::string &dir_path, double threshold, int myid,
                                 int nprocs, const std::string keyword = "Cs_data");

void get_natom_ncell_from_first_Cs_file(int &n_atom, int &n_cell,
                                        const std::string &dir_path);

// Fallback method to read basis dimensions from Cs files, for early dataset versions
void read_basis_from_Cs(const std::string &dir_path);
