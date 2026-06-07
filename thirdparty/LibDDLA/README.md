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


## Mathematical Foundations

Below are the block-wise derivations for each supported routine.  All matrices are assumed to be distributed in 2D block-cyclic fashion; the formulas describe the **local algebraic operations** that the library performs panel-by-panel.

### `pgetrf` — LU Factorization with Partial Pivoting

Given a square (or tall) matrix $A$, we seek a permutation matrix $P$, a unit lower-triangular matrix $L$, and an upper-triangular matrix $U$ such that

$$
PA=LU
$$
Partition the current trailing submatrix at step k as
$$
A = \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}
$$


where $A_{11}$ is the $nb \times nb$ diagonal panel.  The factorization proceeds in three stages:

1. **Panel factorization** (`pgetf2`):  
   $$
   A_{11} = L_{11} U_{11}
   $$
   
   (with partial row pivoting stored in `ipiv`).
   
2. **Trailing-submatrix update**:
   $$U_{12} = L_{11}^{-1} A_{12}$$
   $$L_{21} = A_{21} U_{11}^{-1}$$

3. **Schur-complement update**:
   $$A_{22} \leftarrow A_{22} - L_{21} U_{12}$$

The updated $A_{22}$ becomes the new trailing matrix for the next step, i.e. $A_{22} = L_{22} U_{22}$ is factored recursively.

---

### `pgetrs` — Solve Using LU Factors

With $PA = LU$ already computed, solve $AX = B$:

$$AX = B \quad\Longrightarrow\quad PAX = PB$$

Substitute $PA = LU$:

$$LU X = PB$$

Introduce the intermediate $Y = UX$:

1. **Forward substitution** (unit lower triangular):
   $$LY = PB$$

2. **Backward substitution** (upper triangular):
   $$UX = Y$$

In the distributed implementation `pgetrs` first applies the pivot permutation `plapiv`, then calls `ptrtrs` twice (first with $L$, then with $U$).

---

### `pgesv` — Linear System Solver (LU Driver)

This is the driver routine that composes the two steps above:

$$AX = B \xrightarrow{\text{pgetrf}} PA = LU \xrightarrow{\text{plapiv}} PB \xrightarrow{\text{ptrtrs}(L)} Y \xrightarrow{\text{ptrtrs}(U)} X$$

---

### `ptrtrs` — Distributed Triangular Solve

Solve $TX = B$ where $T$ is triangular.  For a lower-triangular $T$ partitioned block-wise:

$$\begin{pmatrix} T_{11} & 0 \\ T_{21} & T_{22} \end{pmatrix} \begin{pmatrix} X_1 \\ X_2 \end{pmatrix} = \begin{pmatrix} B_1 \\ B_2 \end{pmatrix}$$

The solution is obtained block-by-block:

1. $$T_{11} X_1 = B_1 \quad\Longrightarrow\quad X_1 = T_{11}^{-1} B_1$$
2. $$X_2 = T_{22}^{-1}(B_2 - T_{21} X_1)$$

For an upper-triangular $T$ the sweep direction is reversed:

1. $$T_{22} X_2 = B_2 \quad\Longrightarrow\quad X_2 = T_{22}^{-1} B_2$$
2. $$X_1 = T_{11}^{-1}(B_1 - T_{12} X_2)$$

---

### `pgemm` — Distributed Matrix Multiplication

General matrix-matrix product:

$$C = \alpha \cdot \text{op}(A) \cdot \text{op}(B) + \beta \cdot C$$

LibDDLA implements the **SUMMA** algorithm.  The local block $C_{ij}$ is accumulated over panels of width $nb$:

$$C_{ij} = \beta C_{ij} + \sum_{k} \alpha \cdot A_{ik}^{\text{op}} \cdot B_{kj}^{\text{op}}$$

where $A_{ik}^{\text{op}}$ denotes the properly transposed block.  At each step $k$ the required panel of $A$ is broadcast along the process row, the panel of $B$ along the process column, and a local `gemm` updates $C_{ij}$.

---

### `pgeadd` — Distributed Matrix Addition

Element-wise addition with optional transposition:

$$C = \alpha \cdot \text{op}(A) + \beta \cdot \text{op}(B)$$

When $op(A)=A^{T}$ (or $A^{\dagger}$) the routine communicates the transposed local blocks between the symmetric process pairs $(r,c)\leftrightarrow(c,r)$ before calling the local `geam` kernel.

---

### `plapiv` — Apply Row Pivot Permutation

Given a pivot vector `ipiv` produced by `pgetrf`, construct the permutation matrix $P$ and apply it to a matrix $A$:

$$A \leftarrow PA$$

For each row $i$ the routine looks up the target row $j = \text{ipiv}[i] - 1$ and performs a distributed row swap.  When the two rows reside on different processes the swap uses a temporary buffer and point-to-point communication.

---

### `pswap` — Swap Rows or Columns

Distributed swap of two vectors (rows or columns):

$$\text{swap}(X, Y) : \quad X \leftrightarrow Y$$

If the two vectors are stored on different processes the data is exchanged via `cclSend`/`cclRecv`; otherwise a local `swap` BLAS call is used.

---

### `ppotrf` — Cholesky Factorization

For a Hermitian positive-definite matrix $A$ (lower-triangular storage, $A = A^{\mathsf{H}}$), the Cholesky factorization is

$$
A = LL^{\dagger}$$
$$


Partition $A$ block-wise:

$$A = \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix} = \begin{pmatrix} LL^{\mathsf{H}} & B \\ B^{\mathsf{H}} & C \end{pmatrix}$$

Because $A$ is Hermitian, $A_{12} = A_{21}^{\mathsf{H}} \equiv B$ and $A_{22} \equiv C$.

Introduce the block factors:

$$\begin{pmatrix} L_{11} & 0 \\ L_{21} & L_{22} \end{pmatrix} \begin{pmatrix} L_{11}^{\mathsf{H}} & L_{21}^{\mathsf{H}} \\ 0 & L_{22}^{\mathsf{H}} \end{pmatrix} = \begin{pmatrix} L_{11}L_{11}^{\mathsf{H}} & L_{11}L_{21}^{\mathsf{H}} \\ L_{21}L_{11}^{\mathsf{H}} & L_{21}L_{21}^{\mathsf{H}}+L_{22}L_{22}^{\mathsf{H}} \end{pmatrix}$$

Equating blocks gives the three update formulas used in the right-looking algorithm:

1. **Diagonal panel**:
   $$A_{11} = L_{11} L_{11}^{\mathsf{H}} \quad\Longrightarrow\quad L_{11} = \text{potrf}(A_{11})$$

2. **Sub-diagonal panel**:
   $$A_{21} = L_{21} L_{11}^{\mathsf{H}} \quad\Longrightarrow\quad L_{21} = A_{21} \, (L_{11}^{\mathsf{H}})^{-1}$$

3. **Schur-complement update**:
   $$A_{22} \leftarrow A_{22} - L_{21} L_{21}^{\mathsf{H}}$$

The updated $A_{22}$ is again Hermitian positive-definite, so the process repeats recursively.

---

### `ppotrs` — Solve Using Cholesky Factors

With $A = LL^{\mathsf{H}}$ already computed, solve $AX = B$:

$$LL^{\mathsf{H}} X = B$$

Introduce $Y = L^{\mathsf{H}} X$:

1. **Forward substitution** (lower triangular):
   $$LY = B$$
2. **Backward substitution** (upper triangular, conjugate transpose):
   $$L^{\mathsf{H}} X = Y$$

Both steps are performed by `ptrtrs` with appropriate `uplo` / `trans` arguments.

---

### `pposv` — Positive-Definite System Solver (Cholesky Driver)

Driver that composes Cholesky factorization and triangular solve:

$$AX = B \xrightarrow{\text{ppotrf}} A = LL^{\mathsf{H}} \xrightarrow{\text{ptrtrs}(L)} Y \xrightarrow{\text{ptrtrs}(L^{\mathsf{H}})} X$$

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
cmake .. -DDDLA_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80"

# HIP/ROCm backend
cmake .. -DDDLA_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx90a"

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

An optional CPU-tunnel fallback (`DDLA_USE_GPU_CPU_TUNNEL`) routes data through
host memory when NCCL is unavailable, using MPI for inter-node communication.


## Experimental Routines

### Block LU with Partial Pivoting (Recursive Panel Factorization)

This section documents the block-wise derivation for a recursive panel LU factorization with partial row pivoting, where the diagonal panel is factored by a single-process GPU solver (cuSOLVER / rocSOLVER).

Given a matrix $M$ partitioned into $2 \times 2$ blocks:

$$M = \begin{pmatrix} A & B \\ C & D \end{pmatrix}$$

We seek permutations $P_1, P_2$ and block triangular factors $L, U$ such that $PM = LU$.  The derivation proceeds recursively.

**Step 1 — Pivot and factor the first panel.**

Apply a row permutation $P_1$ to the first block-row and factor the panel:

$$\begin{pmatrix} P_1^{-1} & 0 \\ 0 & I \end{pmatrix} \begin{pmatrix} A & B \\ C & D \end{pmatrix} = \begin{pmatrix} P_1^{-1} & 0 \\ 0 & I \end{pmatrix} \begin{pmatrix} P_1 A & P_1 B \\ C & D \end{pmatrix} = \begin{pmatrix} P_1^{-1} & 0 \\ 0 & I \end{pmatrix} \begin{pmatrix} L_1 U_1 & P_1 B \\ C & D \end{pmatrix}$$

**Step 2 — Introduce the first block column of $L$ and first block row of $U$.**

Rewrite the permuted matrix as a product of a unit block-lower triangular matrix and a block-upper triangular matrix:

$$\begin{pmatrix} P_1^{-1} & 0 \\ 0 & I \end{pmatrix} \begin{pmatrix} L_1 & 0 \\ C U_1^{-1} & I \end{pmatrix} \begin{pmatrix} U_1 & L_1^{-1} P_1 B \\ 0 & D - C U_1^{-1} L_1^{-1} P_1 B \end{pmatrix}$$

Here:
- $L_1$ is unit lower triangular (from the panel LU).
- $U_1$ is upper triangular (from the panel LU).
- The Schur complement of the trailing block is $S = D - C U_1^{-1} L_1^{-1} P_1 B = D - (C U_1^{-1})(L_1^{-1} P_1 B)$.

**Step 3 — Recurse on the Schur complement.**

Factor the trailing block recursively with its own pivot $P_2$:

$$S = P_2^{-1} L_2 U_2$$

Absorb $P_2$ into the global permutation and write the final factorization:

$$\begin{pmatrix} P_1^{-1} & 0 \\ 0 & P_2^{-1} \end{pmatrix} \begin{pmatrix} L_1 & 0 \\ P_2 C U_1^{-1} & L_2 \end{pmatrix} \begin{pmatrix} U_1 & L_1^{-1} P_1 B \\ 0 & U_2 \end{pmatrix}$$

**Summary of the right-looking block algorithm (one step).**

For a distributed matrix with block size $nb$, a single step updates the trailing matrix as follows:

1. **Panel LU** (local, single GPU via cuSOLVER/rocSOLVER `getrf`):  
   $$P_1 A = L_1 U_1$$

2. **Apply pivot to the right panel** (distributed row swaps):  
   $$B \leftarrow P_1 B$$

3. **Compute the block row of $U$** (triangular solve, `trsm`):  
   $$U_{12} = L_1^{-1} B$$

4. **Compute the block column of $L$** (triangular solve, `trsm`):  
   $$L_{21} = C \, U_1^{-1}$$

5. **Schur-complement update** (matrix multiply, `gemm`):  
   $$D \leftarrow D - L_{21} U_{12}$$

The updated $D$ becomes the new trailing matrix for the next recursive step.

---
---

## License

See [LICENSE](./LICENSE).
