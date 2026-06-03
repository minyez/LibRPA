# LibDDLA — Distributed Device Linear Algebra Library

LibDDLA is a C++ template library for **distributed dense linear algebra on GPU
devices**. It supports both NVIDIA CUDA and AMD HIP/ROCm backends through a
unified programming interface.

It provides **ScaLAPACK-style APIs** with 2D block-cyclic data distribution
over an MPI process grid, and uses **NCCL/RCCL** for inter-process GPU
communication.

---

## Features

- Multi-GPU distributed computing via MPI + NCCL/RCCL
- Unified CUDA/HIP backend — write once, run on NVIDIA or AMD GPUs
- Template-based — supports `float`, `double`, `std::complex<float>`,
  `std::complex<double>`
- ScaLAPACK-compatible naming and semantics — easy migration from CPU to GPU

---

## Supported Routines

| Function   | Description |
|------------|-------------|
| `pgetrf`   | LU factorization with partial (row) pivoting |
| `pgetrs`   | Triangular solve using LU factors |
| `pgesv`    | Linear-system solver (driver: LU + solve) |
| `ptrtrs`   | Distributed triangular solve |
| `pgemm`    | Matrix multiplication: C = alpha*op(A)*op(B) + beta*C |
| `pgeadd`   | Matrix addition: C = alpha*op(A) + beta*op(B) |
| `plapiv`   | Apply row-pivot permutation |
| `pswap`    | Swap rows or columns between distributed matrices |
| `ppotrf`   | Cholesky factorization for Hermitian positive-definite matrices |
| `ppotrs`   | Triangular solve using Cholesky factor |
| `pposv`    | Positive-definite system solver (driver: Cholesky + solve) |

---

## Quick Example

```cpp
#include <ddla/ddla.h>
#include <complex>

using namespace ddla;
using T = std::complex<double>;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    DdlaHandle_t handle;
    ddla_init(handle);
    ddla_set(handle, MPI_COMM_WORLD, 'R');

    int      n = 4096, nrhs = 64;
    DdlaDesc descA(handle), descB(handle);
    descA.init_square_blk(n, n, 0, 0);
    descB.init_square_blk(n, nrhs, 0, 0);

    T *d_A, *d_B;
    deviceMalloc(&d_A, descA.m_loc() * descA.n_loc() * sizeof(T));
    deviceMalloc(&d_B, descB.m_loc() * descB.n_loc() * sizeof(T));

    // ... fill A and B ...

    pgesv(n, nrhs, d_A, descA, d_B, descB);

    deviceFree(d_A);
    deviceFree(d_B);
    ddla_destroy(handle);
    MPI_Finalize();
    return 0;
}
```

---

## Build Requirements

| Dependency | Version |
|------------|---------|
| C++ compiler | C++17 or later |
| MPI | OpenMPI / MPICH |
| CUDA Toolkit | >= 11.0 (CUDA) |
| ROCm | >= 5.0 (HIP) |
| NCCL / RCCL | recent stable release |
| CMake | >= 3.18 |

---

## Building

```bash
mkdir build && cd build

# CUDA backend
cmake .. -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80"

# HIP/ROCm backend
cmake .. -DENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx90a"

make -j
```

---

## Architecture

```
LibDDLA/
├── include/ddla/           # Public headers
│   ├── ddla.h              # Main API: all function declarations
│   ├── ddla_desc.h         # DdlaDesc: distributed matrix descriptor
│   ├── ddla_stream.h       # DdlaStream: device streams, handles, NCCL comms
│   ├── ddla_connector.h    # CUDA/HIP type aliases and macros
│   ├── ddla_comm.h         # Communication primitives (bcast, send/recv)
│   ├── ddla_handle_t.h     # Handle type and init / destroy helpers
│   ├── transport_block.h   # Block extraction with transpose support
│   ├── gemm.h, trsm.h,     # BLAS wrapper functions (type-overloaded)
│   │   scal.h, axpy.h,
│   │   swap.h, geru.h,
│   │   iamax.h, geam.h,
│   │   herk.h, gemmBatched.h
│   └── potrf.h             # GPU-solver Potrf wrapper
├── src/                    # Implementation files
│   ├── pgetrf.cpp          # LU factorization
│   ├── pgetrs.cpp          # LU solve
│   ├── pgesv.cpp           # LU driver
│   ├── ptrtrs.cpp          # Triangular solve
│   ├── pgemm.cpp           # Matrix multiplication
│   ├── pgeadd.cpp          # Matrix addition
│   ├── plapiv.cpp          # Pivot application
│   ├── pswap.cpp           # Row/column swap
│   ├── ppotrf.cpp          # Cholesky factorization
│   ├── ppotrs.cpp          # Cholesky solve
│   ├── pposv.cpp           # Cholesky driver
│   ├── pgetf2.cpp          # Unblocked LU panel (inner kernel)
│   ├── pgetf2_panel.cpp    # Alternative panel factorization
│   ├── transport_block.cpp # Block extraction utilities
│   ├── ddla_stream.cpp     # DdlaStream init / cleanup
│   ├── ddla_handle_t.cpp   # Handle init / set / destroy
│   └── ddla_desc.cpp       # DdlaDesc construction and indexing
├── tests/                  # Integration tests
├── cmake/                  # CMake helper modules
└── CMakeLists.txt          # Top-level build
```

---

## Data Distribution

Matrices are distributed over a 2D process grid (`nprows` × `npcols`) in
**2D block-cyclic** fashion, identical to ScaLAPACK. Each process holds a
contiguous local submatrix of dimensions `m_loc × n_loc` stored in GPU device
memory.

The `DdlaDesc` descriptor tracks:
- Global dimensions: `m`, `n`
- Block sizes: `mb`, `nb`
- Process-grid coordinates: `myprow`, `mypcol`, `nprows`, `npcols`
- Source process for row/col distribution: `irsrc`, `icsrc`
- Local leading dimension: `lld`

Index mapping helpers (`indxg2p`, `indxg2l`, `indxl2g`, `num_loc`) are
provided in `ddla_desc.h`.

---

## Communication

Inter-process GPU data movement uses **NCCL** (NVIDIA) or **RCCL** (AMD):

- **Row communicator** (`nccl_row_comm`): broadcast / reduce along process rows
- **Column communicator** (`nccl_col_comm`): broadcast / reduce along process columns

An optional CPU-tunnel fallback (`ENABLE_GPU_CPU_TUNNEL`) routes data through
host memory when NCCL is unavailable, using MPI for inter-node communication.

---

## License

See [LICENSE](./LICENSE).
