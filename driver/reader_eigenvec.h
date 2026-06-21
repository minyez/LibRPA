#pragma once

#include <string>
#include <vector>

#include "../src/core/meanfield.h"

int read_eigenvector(const std::string &dir_path);
int read_eigenvector(const std::string &dir_path, librpa_int::MeanField &mf, bool use_spinor_wfc,
                     const std::vector<int> *iks_selected = nullptr);
