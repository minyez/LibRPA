#ifndef DDLA_H
#define DDLA_H

#include "ddla_desc.h"
#include <complex>

namespace ddla{

/**
 * @brief Distributed triangular solve: B := op(A)^{-1} * B    (side='L')
 *                                     B := B * op(A)^{-1}    (side='R').
 *
 * Solves a triangular system on a distributed GPU matrix using block-cyclic
 * data distribution and NCCL/RCCL communication.  Corresponds to the
 * ScaLAPACK PZTRTRS / PDTRTRS routine.
 *
 * @tparam T  Scalar type (float, double, complex<float>, complex<double>).
 * @param side   'L' (left) or 'R' (right) -- which side of op(A) multiplies B.
 * @param uplo   'U' or 'L' -- which triangle of A is stored.
 * @param trans  'N' (no transpose), 'T' (transpose), 'C' (conjugate-transpose).
 * @param diag   'U' (unit diagonal) or 'N' (non-unit diagonal).
 * @param m      Number of rows of B.
 * @param n      Number of columns of B.
 * @param d_A    Device pointer to distributed triangular matrix A.
 * @param array_descA  DdlaDesc for A (must be square, mb == nb).
 * @param d_B    Device pointer to RHS / solution B.
 * @param array_descB  DdlaDesc for B.
 */
template<typename T>
void ptrtrs(
    const char& side, const char& uplo, const char& trans, const char& diag,
    const int& m, const int& n,
    T* d_A, const DdlaDesc& array_descA,
    T* d_B, const DdlaDesc& array_descB
);

/**
 * @brief Apply row-pivot permutation to a distributed matrix: A := P * A.
 *
 * Implements the column-cyclic forward row pivoting applied after LU
 * factorization.  Only direc='F', rowcol='R', pivroc='C' is supported.
 *
 * @tparam T   Scalar type.
 * @param direc   'F' -- forward pivoting direction.
 * @param rowcol  'R' -- pivot rows.
 * @param pivroc  'C' -- column-cyclic pivot distribution.
 * @param m       Number of rows to pivot.
 * @param n       Number of columns in A.
 * @param d_A     Device pointer to distributed matrix A.
 * @param array_descA   DdlaDesc for A.
 * @param ipiv    Host array of pivot indices (1-based, length >= m).
 * @param array_descIP  DdlaDesc for pivot vector (same row distribution as A).
 * @param iwork   Workspace (unused, pass nullptr).
 */
template <typename T>
void plapiv(
    const char& direc, const char& rowcol, const char& pivroc,
    const int& m, const int& n,
    T* d_A,const DdlaDesc& array_descA,
    const int* ipiv, const DdlaDesc& array_descIP,
    int* iwork
);

/**
 * @brief Swap two rows or two columns between distributed matrices.
 *
 * Exchanges the segment of length N starting at (ia,ja) in A with the
 * segment starting at (ib,jb) in B.  inca=1 swaps columns, inca=m swaps rows.
 * Communication occurs only when the source and target rows/columns reside on
 * different processes.
 *
 * @tparam T    Scalar type.
 * @param N     Length of the segment to swap.
 * @param A     Device pointer to distributed matrix A.
 * @param ia    Starting global row index in A (1-based).
 * @param ja    Starting global column index in A (1-based).
 * @param array_descA  DdlaDesc for A.
 * @param inca  1 (swap columns) or m_A (swap rows).
 * @param B     Device pointer to distributed matrix B.
 * @param ib    Starting global row index in B (1-based).
 * @param jb    Starting global column index in B (1-based).
 * @param array_descB  DdlaDesc for B.
 * @param incb  Increment for B (must match inca when inca == 1).
 */
template <typename T>
void pswap(
    const int& N, 
    T* A, int ia, int ja, const DdlaDesc& array_descA, const int& inca,
    T* B, int ib, int jb, const DdlaDesc& array_descB, const int& incb
);

/**
 * @brief Internal unblocked panel LU factorization for distributed matrices.
 *
 * Factors the panel starting at global column n_s with width nb_real.  This
 * is the inner kernel called by pgetrf to factor each diagonal block.
 * Outputs pivot indices into ipiv (1-based).
 *
 * @tparam T   Scalar type.
 * @param m        Total rows of A.
 * @param nb_real  Actual width of this panel (<= nb).
 * @param d_A      Device pointer to matrix A (input/output).
 * @param n_s      Global starting column index of the panel.
 * @param array_descA  DdlaDesc for A.
 * @param ipiv     Host pivot array (output, 1-based).
 * @param info     0 on success, >0 if singular.
 */
template <typename T>
void pgetf2(
    const int& m, const int& nb_real,
    T* d_A, const int& n_s, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
);

/**
 * @brief Alternative panel LU factorization (rank-revealing variant).
 *
 * Uses a slightly different communication pattern than pgetf2 for
 * pivot selection within a panel.
 *
 * @tparam T   Scalar type.
 * @param m        Total rows of A.
 * @param nb_real  Actual panel width.
 * @param d_A      Device pointer to matrix A.
 * @param n_start  Global starting column of the panel.
 * @param array_descA  DdlaDesc for A.
 * @param ipiv     Host pivot array (output).
 * @param info     0 on success.
 */
template <typename T>
void pgetf2_panel(
    const int& m, const int& nb_real,
    T* d_A, const int& n_start, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
);

/**
 * @brief Distributed LU factorization with partial (row) pivoting:
 *        A = P * L * U.
 *
 * Factors a distributed m-by-n matrix panel by panel:
 *   1. Factor the diagonal block (pgetf2).
 *   2. Broadcast L and U factors along process rows / columns.
 *   3. Solve for the U panel (trsm).
 *   4. Update the trailing submatrix (gemm: C -= L*U).
 *
 * Requires square blocks (mb == nb).  Corresponds to ScaLAPACK
 * PZGETRF / PDGETRF.
 *
 * @tparam T   Scalar type.
 * @param m        Number of rows of A.
 * @param n        Number of columns of A.
 * @param d_A      Device pointer to matrix A (input/output -- L+U factors).
 * @param array_descA  DdlaDesc for A (mb == nb required).
 * @param ipiv     Host pivot array (output, 1-based, length >= m_loc).
 * @param info     0 on success, >0 if singular.
 */
template <typename T>
void pgetrf(
    const int& m, const int& n,
    T* d_A, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
);

/**
 * @brief Distributed LU solve: solve A * X = B using the factors from pgetrf.
 *
 * Steps:  apply row pivots (plapiv), forward solve L*Y=B (ptrtrs), backward
 * solve U*X=Y (ptrtrs).  Currently only trans='N' (non-transposed) is supported.
 *
 * @tparam T   Scalar type.
 * @param trans   'N' -- no transpose (only 'N' supported).
 * @param n       Order of matrix A.
 * @param nrhs    Number of right-hand sides.
 * @param d_A     Device pointer to LU factors (from pgetrf).
 * @param array_descA  DdlaDesc for A.
 * @param ipiv    Host pivot array from pgetrf.
 * @param d_B     Device pointer to RHS / solution B (input/output).
 * @param array_descB  DdlaDesc for B.
 */
template <typename T>
void pgetrs(
    const char& trans, const int& n, const int& nrhs,
    T* d_A, const DdlaDesc& array_descA,
    const int* ipiv, // host
    T* d_B, const DdlaDesc& array_descB
);

/**
 * @brief Distributed linear-system solver (driver): solve A * X = B.
 *
 * Convenience wrapper: pgetrf (LU) + pgetrs (solve).  Corresponds to
 * ScaLAPACK PZGESV / PDGESV.
 *
 * @tparam T   Scalar type.
 * @param n       Order of square matrix A.
 * @param nrhs    Number of right-hand sides.
 * @param d_A     Device pointer to A (input: coefficient; output: LU factors).
 * @param array_descA  DdlaDesc for A.
 * @param d_B     Device pointer to RHS / solution B (input/output).
 * @param array_descB  DdlaDesc for B.
 * @throws std::runtime_error if LU factorization fails (info != 0).
 */
template <typename T>
void pgesv(
    const int& n, const int& nrhs,
    T* d_A, const DdlaDesc& array_descA,
    T* d_B, const DdlaDesc& array_descB
);

/**
 * @brief Distributed matrix-matrix multiplication:
 *        C := alpha * op(A) * op(B) + beta * C.
 *
 * Uses a 2D block-cyclic data distribution and NCCL/RCCL broadcast of
 * panel columns of A and panel rows of B (AB-path).  Supports all standard
 * transpose options:
 *   - 'N': op(X) = X
 *   - 'T': op(X) = X^T
 *   - 'C': op(X) = X^H  (conjugate-transpose)
 *
 * When transa or transb is not 'N', the process grid must be square
 * (nprows == npcols).
 *
 * @tparam T    Scalar type.
 * @param transa   Operation applied to A ('N','T','C').
 * @param transb   Operation applied to B ('N','T','C').
 * @param m        Rows of op(A) and C.
 * @param n        Cols of op(B) and C.
 * @param k        Cols of op(A) / rows of op(B).
 * @param alpha    Scalar multiplier for A*B.
 * @param d_A      Device pointer to distributed A.
 * @param array_descA  DdlaDesc for A.
 * @param d_B      Device pointer to distributed B.
 * @param array_descB  DdlaDesc for B.
 * @param beta     Scalar multiplier for C.
 * @param d_C      Device pointer to distributed C (input/output).
 * @param array_descC  DdlaDesc for C.
 */
template <typename T>
void pgemm(
    const char& transa, const char& transb,
    const int& m, const int& n, const int& k,
    const T& alpha,
    const T* d_A, const DdlaDesc& array_descA,
    const T* d_B, const DdlaDesc& array_descB,
    const T& beta,
    T* d_C, const DdlaDesc& array_descC
);

/**
 * @brief Distributed matrix addition: C := alpha * op(A) + beta * op(B).
 *
 * Element-wise addition of two distributed matrices with optional transpose
 * operations.  Communication between processes is required when the data
 * distribution of op(A) differs from that of op(B) (e.g. one is transposed
 * and the other is not).
 *
 * @tparam T    Scalar type.
 * @param transa   Operation for A ('N','T','C').
 * @param transb   Operation for B ('N','T','C').
 * @param m        Rows of C.
 * @param n        Cols of C.
 * @param alpha    Scalar multiplier for op(A).
 * @param d_A      Device pointer to distributed A.
 * @param array_descA  DdlaDesc for A.
 * @param beta     Scalar multiplier for op(B).
 * @param d_B      Device pointer to distributed B.
 * @param array_descB  DdlaDesc for B.
 * @param d_C      Device pointer to result C (output).
 * @param array_descC  DdlaDesc for C.
 */
template <typename T>
void pgeadd(
    const char& transa, const char& transb,
    const int& m, const int& n,
    const T& alpha,
    const T* d_A, const DdlaDesc& array_descA,
    const T& beta,
    const T* d_B, const DdlaDesc& array_descB,
    T* d_C, const DdlaDesc& array_descC
);

/**
 * @brief Distributed Cholesky factorization: A = L * L^H  (uplo='L').
 *
 * Factors a Hermitian positive-definite distributed matrix using GPU solver
 * libraries (cusolverDn / hipsolver).  Algorithm: factor diagonal block
 * (potrf), broadcast factor, solve off-diagonal (trsm), update trailing
 * submatrix via batched gemm or herk.
 *
 * @note Only uplo='L' (lower) is supported.  Only complex<float> and
 *       complex<double> are instantiated.
 *
 * @tparam T   Scalar type (complex<float> or complex<double>).
 * @param uplo     'L' -- lower triangle of A is stored and factored.
 * @param n        Order of A.
 * @param A        Device pointer to A (input: Hermitian pos-def; output: L).
 * @param ia       Global starting row (1-based).
 * @param ja       Global starting col (1-based).
 * @param array_descA  DdlaDesc for A (mb == nb required).
 * @param info     0 on success, >0 if not positive-definite.
 * @param is_head  Internal flag for multi-head Cholesky (default false).
 * @param location Internal row/col rearrangement index (default -1).
 * @return true if the last diagonal element needed a sign correction,
 *         false otherwise.
 */
template<typename T>
bool ppotrf(
    const char& uplo, const int& n,
    T* A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    int& info, // host pointer
    bool is_head = false, int location = -1
);

/**
 * @brief Distributed solve using Cholesky factorization: A * X = B.
 *
 * Solves a Hermitian positive-definite system using the factor L from
 * ppotrf.  Two triangular solves:  forward (L*Y=B), backward (L^H*X=Y).
 * Only side='L' and trans='N' are supported.
 *
 * @tparam T   Scalar type.
 * @param side     'L' (left) -- solve A*X = B.
 * @param uplo     'L' (lower) -- factor is lower triangular.
 * @param trans    'N' (no transpose).
 * @param n        Order of A.
 * @param nrhs     Number of right-hand sides.
 * @param d_A      Device pointer to Cholesky factor L (from ppotrf).
 * @param array_descA  DdlaDesc for A.
 * @param d_B      Device pointer to RHS / solution B (input/output).
 * @param array_descB  DdlaDesc for B.
 * @param is_nega  Diagonal sign-correction flag (from ppotrf return).
 * @param location Internal parameter (must be -1).
 */
template <typename T>
void ppotrs(
    const char& side, const char& uplo, const char& trans,
    const int& n, const int& nrhs,
    T* d_A, const DdlaDesc& array_descA,
    T* d_B, const DdlaDesc& array_descB,
    bool is_nega = false, int location = -1
);

/**
 * @brief Distributed solver for Hermitian positive-definite systems
 *        (driver): solve A * X = B via Cholesky factorization.
 *
 * Convenience wrapper:  ppotrf + ppotrs.  Corresponds to ScaLAPACK PZPOSV.
 *
 * @tparam T   Scalar type.
 * @param side     'L' -- solve A*X = B.
 * @param uplo     'L' -- lower triangle of A is stored.
 * @param trans    'N' -- no transpose.
 * @param n        Order of A.
 * @param nrhs     Number of right-hand sides.
 * @param d_A      Device pointer to A (input: pos-def; output: Cholesky L).
 * @param ia       Global starting row of A (1-based).
 * @param ja       Global starting col of A (1-based).
 * @param array_descA  DdlaDesc for A.
 * @param d_B      Device pointer to RHS / solution B (input/output).
 * @param ib       Global starting row of B (1-based).
 * @param jb       Global starting col of B (1-based).
 * @param array_descB  DdlaDesc for B.
 * @param info     Output: 0 on success, >0 if not positive-definite.
 * @param is_head  Forwarded to ppotrf.
 * @param location Forwarded to ppotrf.
 */
template <typename T>
void pposv(
    const char& side, const char& uplo, const char& trans,
    const int & n, const int& nrhs,
    T* d_A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    T* d_B, const int& ib, const int& jb, const DdlaDesc& array_descB,
    int& info, // host pointer
    bool is_head = false, int location = -1
);


} // namespace ddla

#endif // DDLA_H
