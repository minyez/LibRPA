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
| `pgetrf_nopiv` | LU factorization without pivoting (multi-process) |
| `pgetrs_nopiv` | Triangular solve using LU factors without pivoting |
| `pgesv_nopiv`  | Linear-system solver without pivoting (driver) |
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

### No-Pivot LU Variants (`pgetrf_nopiv`, `pgetrs_nopiv`, `pgesv_nopiv`)

For matrices that are known to be diagonally dominant or otherwise structurally
safe without pivoting, LibDDLA provides no-pivot counterparts to the LU family.
The algorithms are identical to the pivoted versions except that the row
permutation $P$ is omitted, which removes the distributed pivot search, the
row swaps, and the pivot bookkeeping.

#### `pgetrf_nopiv` — LU Factorization without Pivoting

Given a square matrix $A$, we seek a unit lower-triangular matrix $L$ and an
upper-triangular matrix $U$ such that

$$A = LU$$

Partition the current trailing submatrix at step $k$ as

$$A = \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}$$

where $A_{11}$ is the $nb \times nb$ diagonal panel.  The right-looking block
algorithm proceeds in four stages:

1. **Diagonal panel factorization** (local single-GPU `getrf_nopiv`):
   $$A_{11} = L_{11} U_{11}$$

2. **Broadcast the factored panel** to the owning process row and column via
   NCCL/RCCL row/column communicators, so that every process holding $A_{12}$
   or $A_{21}$ receives the updated $A_{11} = L_{11} U_{11}$.

3. **Trailing-submatrix update** (triangular solve, `trsm`):
   $$U_{12} = L_{11}^{-1} A_{12}$$
   $$L_{21} = A_{21} U_{11}^{-1}$$

4. **Schur-complement update** (matrix multiply, `gemm`):
   $$A_{22} \leftarrow A_{22} - L_{21} U_{12}$$

The updated $A_{22}$ becomes the new trailing matrix for the next step, i.e.
$A_{22} = L_{22} U_{22}$ is factored recursively.  Because no pivoting is
performed, the panel factorization can use a large block size (up to the size
that fits on a single GPU), which reduces the number of communication rounds
and yields higher throughput than the pivoted variants for numerically stable
inputs.

#### `pgetrs_nopiv` — Solve Using LU Factors without Pivoting

With $A = LU$ already computed, solve $AX = B$:

Substitute $A = LU$:

$$LU X = B$$

Introduce the intermediate $Y = UX$:

1. **Forward substitution** (unit lower triangular):
   $$LY = B$$

2. **Backward substitution** (upper triangular):
   $$UX = Y$$

In the distributed implementation `pgetrs_nopiv` calls `ptrtrs` twice: first
with the lower-triangular factor $L$ (unit diagonal, no transpose), then with
the upper-triangular factor $U$ (no transpose).  No pivot permutation is
applied.

#### `pgesv_nopiv` — Linear System Solver without Pivoting (Driver)

This is the no-pivot driver routine that composes the two steps above:

$$AX = B \xrightarrow{\text{pgetrf_nopiv}} A = LU
       \xrightarrow{\text{ptrtrs}(L)} Y
       \xrightarrow{\text{ptrtrs}(U)} X$$

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

> **Note:** `install_scripts/install_cuda.sh` currently sets `CMAKE_CUDA_ARCHITECTURES=80` by default (A100 / SM80). If you run on V100 (SM70) or other GPUs, make sure the architecture matches the target device, e.g.:
>
> ```bash
> cmake .. -DDDLA_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="70"
> ```
>
> Mismatched architecture (e.g. building SM80 and running on V100) can cause kernel launch failures or silent numerical errors such as `log-determinant mismatch` in the tests.

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

$$\begin{pmatrix} P_1^{-1} & 0 \\ 0 & I \end{pmatrix} \begin{pmatrix} L_1 & 0 \\ C U_1^{-1} & I \end{pmatrix}\begin{pmatrix} I & 0 \\ 0 & P_2^{-1} \end{pmatrix} \begin{pmatrix} I & 0 \\ 0 & L_2\end{pmatrix}\begin{pmatrix} U_1 & L_1^{-1} P_1 B \\ 0 & U_2 \end{pmatrix}$$

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

## Benchmarks

The following table shows single-GPU `zgetrf_nopiv` timings on an NVIDIA V100
(SM70) using `v100g32fat`.  The test matrix is a random complex matrix with a
diagonal shift to keep it non-singular.  Timings include device synchronization.

| n     | LibDDLA `getrf_nopiv` (s) | MAGMA `zgetrf_nopiv_gpu` (s) | cuSOLVER `zgetrf` (s) | diff(L/R) | diff(M/R) | diff(L/M) |
|------:|--------------------------:|-----------------------------:|----------------------:|----------:|----------:|----------:|
| 100   | 7.18e-03                  | 2.42e-01                     | 1.68e-03              | 1.42e-14  | 0.00e+00  | 1.42e-14  |
| 500   | 1.88e-03                  | 4.89e-03                     | 5.93e-03              | 5.68e-14  | 0.00e+00  | 5.68e-14  |
| 1000  | 4.39e-03                  | 1.31e-02                     | 8.77e-03              | 0.00e+00  | 1.14e-13  | 1.14e-13  |
| 5000  | 9.60e-02                  | 4.24e-01                     | 9.92e-02              | 4.55e-13  | 0.00e+00  | 4.55e-13  |
| 10000 | 6.02e-01                  | 2.32e+00                     | 5.12e-01              | 1.82e-12  | 1.82e-12  | 0.00e+00  |

Notes:
- LibDDLA and cuSOLVER stay on the GPU for the whole factorization.
- MAGMA's `magma_zgetrf_nopiv_gpu` is a **hybrid** routine: it copies each
  panel to the CPU, factors it with OpenBLAS, and copies it back.  This
  explains the higher run time, especially for small matrices where the
  CPU-panel overhead dominates.
- The `log|det|` values agree to floating-point round-off in all cases.

### Multi-Process No-Pivot LU Benchmarks

The following tables summarize 4-MPI-rank / 4-GPU runs on a single HPC3 node,
using complex double precision (`zgetrf_nopiv` / `zgesv_nopiv`) and block size
$nb = 128$.

#### Factorization comparison: `pgetrf` vs `pgetrf_bpiv` vs `pgetrf_nopiv`

| n     | `pgetrf` (s) | `pgetrf_bpiv` (s) | `pgetrf_nopiv` (s) | ln_det                 | `|ln_det_nopiv - ln_det_pgetrf|` | `|ln_det_nopiv - ln_det_bpiv|` |
|------:|-------------:|------------------:|-------------------:|:-----------------------|---------------------------------:|-------------------------------:|
| 500   | 6.44e-02     | 7.02e-03          | 3.23e-03           | 346.270152+i-0.745968  | 0.00e+00                         | 0.00e+00                       |
| 1000  | 9.82e-02     | 6.27e-03          | 5.76e-03           | 692.104913+i-1.882746  | 2.22e-16                         | 2.22e-16                       |
| 5000  | 5.39e-01     | 5.49e-02          | 5.15e-02           | 3456.164522+i-11.786721| 9.09e-13                         | 0.00e+00                       |
| 10000 | 1.16e+00     | 1.88e-01          | 1.80e-01           | 6910.034013+i-24.275085| 9.13e-13                         | 9.13e-13                       |
| 15000 | 1.96e+00     | 4.86e-01          | 4.80e-01           | 10363.707884+i-36.828523| 1.82e-12                        | 0.00e+00                       |
| 20000 | 3.05e+00     | 1.03e+00          | 1.02e+00           | 13817.180801+i-49.299152| 1.82e-12                        | 3.65e-12                       |

`pgetrf_nopiv` matches the `log|det|` values of both pivoting variants to
floating-point round-off.  Because the block LU algorithm does not perform any
row interchanges, the panel factorization uses the local single-GPU
`getrf_nopiv` kernel with a large block size, and the dominant cost is the
Schur-complement `gemm` update.

#### Driver solve accuracy: `pgesv_nopiv` solving $AX = I$

The test generates a random diagonally-shifted matrix $A$, solves $AX = I$ with
`pgesv_nopiv`, multiplies $A \cdot X$ locally with `pgemm`, and checks the
maximum entrywise deviation from the identity matrix.  The global max error is
obtained via `MPI_Reduce(MPI_MAX, ...)`.

| n     | `pgesv_nopiv` (s) | verification `pgemm` (s) | global max error |
|------:|------------------:|-------------------------:|:----------------:|
| 500   | 3.19e-02          | 6.30e-04                 | 2.11e-15         |
| 1000  | 1.03e-02          | 1.65e-03                 | 3.55e-15         |
| 5000  | 1.33e-01          | 5.98e-02                 | 3.77e-15         |
| 10000 | 5.95e-01          | 3.56e-01                 | 4.22e-15         |

The global max error stays at the level of machine epsilon for complex double
precision, confirming that the no-pivot LU factors can reconstruct the inverse
with full numerical accuracy for this class of test matrices.

## License

See [LICENSE](./LICENSE).
