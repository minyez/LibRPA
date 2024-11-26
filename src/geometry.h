#pragma once
#include <array>

#include "atoms.h"

typedef std::array<double, 3> coord_t;
extern std::map<atom_t, coord_t> coord;
extern std::map<atom_t, coord_t> coord_frac;
