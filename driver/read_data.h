//! Functions to parse files generated by FHI-aims
/*
 */
#ifndef READ_DATA_H
#define READ_DATA_H

#include <string>
#include <vector>

#include "matrix.h"
#include "meanfield.h"
#include "ri.h"
#include "vector3_order.h"

using std::string;

/*!
 * @brief Read occupation numbers and eigenvalues of SCF calculation
 */
void read_scf_occ_eigenvalues(const string &file_path, MeanField &mf);

/*!
 * @brief Read exchange-correlation potential
 *
 * @param[in] file_path   data file path
 * @return    status code, 0 for succesful read, 1 with error
 */
int read_vxc(const string &file_path, std::vector<matrix> &vxc);

int read_eigenvector(const string &dir_path, MeanField &mf);

size_t read_Cs(const string &dir_path, double threshold, const vector<atpair_t> &local_atpair,
               bool binary = false);

size_t read_Cs_evenly_distribute(const string &dir_path, double threshold, int myid, int nprocs,
                                 bool binary = false);

size_t read_Vq_full(const string &dir_path, const string &vq_fprefix, bool is_cut_coulomb);

size_t read_Vq_row(const string &dir_path, const string &vq_fprefix, double threshold,
                   const vector<atpair_t> &local_atpair, bool is_cut_coulomb);

void read_stru(const int& n_kpoints, const std::string &file_path);

void read_dielec_func(const string &file_path, std::vector<double> &omegas,
                      std::vector<double> &dielec_func_imagfreq);

void erase_Cs_from_local_atp(atpair_R_mat_t &Cs, vector<atpair_t> &local_atpair);

void get_natom_ncell_from_first_Cs_file(int &n_atom, int &n_cell, const string &dir_path, bool binary = false);

std::vector<Vector3_Order<double>> read_band_kpath_info(int &n_basis, int &n_states, int &n_spin);

MeanField read_meanfield_band(int n_basis, int n_states, int n_spin, int n_kpoints_band);

std::vector<matrix> read_vxc_band(int n_states, int n_spin, int n_kpoints_band);

#endif // !READ_DATA_H
