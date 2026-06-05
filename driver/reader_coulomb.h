#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "../src/core/ri.h"

bool check_coulomb_file_binary(const std::string &file_path);

int detect_coulomb_reader_version(const std::string &dir_path,
                                  const std::string &vq_fprefix);

size_t read_Vq_full(const std::string &dir_path, const std::string &vq_fprefix,
                    bool is_cut_coulomb, int reader_version = 0);

size_t read_Vq_row(const std::string &dir_path, const std::string &vq_fprefix,
                   double threshold, const std::vector<librpa_int::atpair_t> &local_atpair,
                   bool is_cut_coulomb, int reader_version = 0);
