//! Functions to parse files generated by FHI-aims
/*
 */
#ifndef READ_AIMS_H
#define READ_AIMS_H

#include <string>
#include <map>
#include "meanfield.h"
#include "ri.h"
#include "vector3_order.h"
using std::string;
using std::map;

void READ_AIMS_BAND(const string &file_path, MeanField &mf);
void READ_AIMS_EIGENVECTOR(const string &dir_path, MeanField &mf);
void handle_KS_file(const string &file_path, MeanField &mf);
size_t READ_AIMS_Cs(const string &dir_path, double threshold);
size_t READ_AIMS_Vq(const string &dir_path, const string &vq_fprefix, double threshold, atpair_k_cplx_mat_t &coulomb);
size_t handle_Cs_file(const std::string &file_path, double threshold);
void handle_Vq_file(const std::string &file_path, double threshold, map<Vector3_Order<double>, ComplexMatrix> &Vq_full);


void read_aims(MeanField &mf);

#endif // !READ_AIMS_H
