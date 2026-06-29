#pragma once

#include <string>
#include <vector>

#include "../src/core/meanfield.h"

enum class LegacyTextWfcOrder
{
    BasisSpinorBandSpin,
    SpinBasisBand
};

int read_eigenvector(const std::string &dir_path);
int read_eigenvector(const std::string &dir_path, librpa_int::MeanField &mf, bool use_spinor_wfc,
                     const std::vector<int> *iks_selected = nullptr);
int read_eigenvector(const std::string &dir_path, librpa_int::MeanField &mf, bool use_spinor_wfc,
                     const std::vector<int> &source_to_target_ik,
                     const std::vector<int> *source_iks_selected,
                     LegacyTextWfcOrder text_order = LegacyTextWfcOrder::BasisSpinorBandSpin);
