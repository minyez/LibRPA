#pragma once
#include <string>

struct DriverParams
{
    std::string input_dir;

    void print();
};

extern DriverParams driver_params;
