#pragma once
#include <string>

#include "input_parser.h"

//! Class to represent the input file
/*!
  Use the member function load to get the parameter parser.
 */
class InputFile
{
private:
    std::string filename;
    std::string orig_content;

public:
    InputFile() {};
    InputParser load(const std::string& fn, bool error_if_fail_open = false);
    std::string get_filename() { return filename; };
    std::string get_orig_content() { return orig_content; };
};

void parse_inputfile_to_params(const std::string& fn);
