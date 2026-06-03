#ifndef TRANSPORT_BLOCK_H
#define TRANSPORT_BLOCK_H

#include "ddla_desc.h"
#include <complex>

namespace ddla{

template <typename T>
void transport_block(
    const char& sData, const char& trans,
    const int& m, const int& n,
    const T* d_A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    T* d_block_A
);

}

#endif // TRANSPORT_BLOCK_H