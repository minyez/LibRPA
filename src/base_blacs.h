#pragma once

#include <string>

#include "parallel_mpi.h"

namespace LIBRPA
{

enum class CTXT_LAYOUT {R, C};
enum class CTXT_SCOPE {R, C, A};

void CTXT_barrier(int ictxt, CTXT_SCOPE scope = CTXT_SCOPE::A);

class BLACS_CTXT_handler
{
private:
    MPI_COMM_handler mpi_comm_h;
    char layout_ch;
    bool initialized_;
    bool pgrid_set_;
    bool comm_set_;
public:
    int ictxt;
    CTXT_LAYOUT layout;
    int myid;
    int nprocs;
    int nprows;
    int npcols;
    int mypcol;
    int myprow;
    BLACS_CTXT_handler() { comm_set_ = pgrid_set_ = initialized_ = false; }
    BLACS_CTXT_handler(MPI_Comm comm_in): mpi_comm_h(comm_in) { comm_set_ = true; pgrid_set_ = initialized_ = false; }
    ~BLACS_CTXT_handler() {};

    void init();
    void reset_comm(MPI_Comm comm_in);
    void set_grid(const int &nprows_in, const int &npcols_in, CTXT_LAYOUT layout_in = CTXT_LAYOUT::R);
    void set_square_grid(bool more_rows = true, CTXT_LAYOUT layout_in = CTXT_LAYOUT::R);
    void set_horizontal_grid();
    void set_vertical_grid();
    std::string info() const;
    int get_pnum(int prow, int pcol) const;
    void get_pcoord(int pid, int &prow, int &pcol) const;
    void barrier(CTXT_SCOPE scope = CTXT_SCOPE::A) const;
    //! call gridexit to reset process grid
    void exit();
    bool initialized() const { return initialized_; }
};

class Array_Desc
{
private:
    // BLACS parameters obtained upon construction
    int ictxt_;
    int nprocs_;
    int myid_;
    int nprows_;
    int myprow_;
    int npcols_;
    int mypcol_;
    void set_blacs_params_(int ictxt, int nprocs, int myid, int nprows, int myprow, int npcols, int mypcol);
    int set_desc_(const int &m, const int &n, const int &mb, const int &nb,
                  const int &irsrc, const int &icsrc);

    // Array dimensions
    int m_;
    int n_;
    int mb_;
    int nb_;
    int irsrc_;
    int icsrc_;
    int lld_;
    int m_local_;
    int n_local_;

    //! flag to indicate that the current process should contain no data of local matrix, but for scalapack routines, it will generate a dummy matrix of size 1, nrows = ncols = 1
    bool empty_local_mat_ = false;

    //! flag for initialization
    bool initialized_ = false;

public:
    int desc[9];
    Array_Desc(const BLACS_CTXT_handler &blacs_ctxt_h);
    Array_Desc(const int &ictxt);
    //! initialize the array descriptor
    int init(const int &m, const int &n,
             const int &mb, const int &nb,
             const int &irsrc, const int &icsrc);
    //! initialize the array descriptor such that each process has exactly one block
    int init_1b1p(const int &m, const int &n,
                  const int &irsrc, const int &icsrc);
    int init_square_blk(const int &m, const int &n,
                        const int &irsrc, const int &icsrc);
    int indx_g2l_r(int gindx) const;
    int indx_g2l_c(int gindx) const;
    int indx_l2g_r(int lindx) const;
    int indx_l2g_c(int lindx) const;
    const int& ictxt() const { return ictxt_; }
    const int& m() const { return m_; }
    const int& n() const { return n_; }
    const int& mb() const { return mb_; }
    const int& nb() const { return nb_; }
    const int& lld() const { return lld_; }
    const int& irsrc() const { return irsrc_; }
    const int& icsrc() const { return icsrc_; }
    const int& m_loc() const { return m_local_; }
    const int& n_loc() const { return n_local_; }
    const int& myprow() const { return myprow_; }
    const int& mypcol() const { return mypcol_; }
    const int& nprows() const { return nprows_; }
    const int& npcols() const { return npcols_; }
    std::string info() const;
    std::string info_desc() const;
    bool is_src() const;
    void barrier(CTXT_SCOPE scope = CTXT_SCOPE::A);
};

} /* end of namespace LIBRPA */
