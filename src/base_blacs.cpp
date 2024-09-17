#include "base_blacs.h"

#include "interface/blacs_scalapack.h"
#include "scalapack_connector.h"
#include "utils_io.h"


namespace LIBRPA
{

void CTXT_barrier(int ictxt, CTXT_SCOPE scope)
{
    char scope_ch;
    switch (scope)
    {
        case (CTXT_SCOPE::R): scope_ch = 'R';
        case (CTXT_SCOPE::C): scope_ch = 'C';
        case (CTXT_SCOPE::A): scope_ch = 'A';
    }
    Cblacs_barrier(ictxt, &scope_ch);
}


void BLACS_CTXT_handler::init()
{
    this->mpi_comm_h.init();
    this->ictxt = Csys2blacs_handle(this->mpi_comm_h.comm);
    Cblacs_pinfo(&this->myid, &this->nprocs);
    this->initialized_ = true;
}

void BLACS_CTXT_handler::reset_comm(MPI_Comm comm_in)
{
    this->mpi_comm_h.reset_comm(comm_in);
    this->comm_set_ = true;
    this->pgrid_set_ = false;
    this->initialized_ = false;
}

void BLACS_CTXT_handler::set_grid(const int &nprows_in, const int &npcols_in,
                                  CTXT_LAYOUT layout_in)
{
    // if the grid has been set, exit it first
    if (pgrid_set_) exit();
    if (nprocs != nprows_in * npcols_in)
        throw std::invalid_argument("nprocs != nprows * npcols");
    layout = layout_in;
    if (layout == CTXT_LAYOUT::C)
        layout_ch = 'C';
    else
        layout_ch = 'R';
    Cblacs_gridinit(&ictxt, &layout_ch, nprows_in, npcols_in);
    Cblacs_gridinfo(ictxt, &nprows, &npcols, &myprow, &mypcol);
    pgrid_set_ = true;
}

void BLACS_CTXT_handler::set_square_grid(bool more_rows, CTXT_LAYOUT layout_in)
{
    int nroc;
    layout = layout_in;
    for (nroc = int(sqrt(double(nprocs))); nroc >= 2; --nroc)
    {
        if ((nprocs) % nroc == 0) break;
    }
    more_rows ? nprows = nprocs / (npcols = nroc)
              : npcols = nprocs / (nprows = nroc);
    set_grid(nprows, npcols, layout_in);
}

void BLACS_CTXT_handler::set_horizontal_grid()
{
    set_grid(1, nprocs, CTXT_LAYOUT::R);
}

void BLACS_CTXT_handler::set_vertical_grid()
{
    set_grid(nprocs, 1, CTXT_LAYOUT::C);
}

void BLACS_CTXT_handler::exit()
{
    if (pgrid_set_)
    {
        Cblacs_gridexit(ictxt);
        // recollect the system context
        ictxt = Csys2blacs_handle(mpi_comm_h.comm);
        pgrid_set_ = false;
    }
}

std::string BLACS_CTXT_handler::info() const
{
    std::string info;
    info = std::string("BLACS_CTXT_handler: ")
         + "ICTXT " + std::to_string(ictxt) + " "
         + "PSIZE " + std::to_string(nprocs) + " "
         + "PID " + std::to_string(myid) + " "
         + "PGRID (" + std::to_string(nprows) + "," + std::to_string(npcols) + ") "
         + "PCOOD (" + std::to_string(myprow) + "," + std::to_string(mypcol) +")";
    return info;
}

int BLACS_CTXT_handler::get_pnum(int prow, int pcol) const
{
    return Cblacs_pnum(ictxt, prow, pcol);
}

void BLACS_CTXT_handler::get_pcoord(int pid, int &prow, int &pcol) const
{
    Cblacs_pcoord(ictxt, pid, &prow, &pcol);
}

void BLACS_CTXT_handler::barrier(CTXT_SCOPE scope) const
{
    CTXT_barrier(ictxt, scope);
}

void Array_Desc::set_blacs_params_(int ictxt, int nprocs, int myid, int nprows,
                                  int myprow, int npcols, int mypcol)
{
    assert(myid < nprocs && myprow < nprows && mypcol < npcols);
    ictxt_ = ictxt;
    nprocs_ = nprocs;
    myid_ = myid;
    nprows_ = nprows;
    myprow_ = myprow;
    npcols_ = npcols;
    mypcol_ = mypcol;
}

int Array_Desc::set_desc_(const int &m, const int &n, const int &mb, const int &nb,
                          const int &irsrc, const int &icsrc)
{
    int info = 0;
    m_local_ = ScalapackConnector::numroc(m, mb, myprow_, irsrc, nprows_);
    // leading dimension
    lld_ = std::max(m_local_, 1);
    n_local_ = ScalapackConnector::numroc(n, nb, mypcol_, icsrc, npcols_);
    if (m_local_ < 1 || n_local_ < 1)
    {
        empty_local_mat_ = true;
    }

    ScalapackConnector::descinit(this->desc, m, n, mb, nb, irsrc, icsrc, ictxt_, lld_, info);
    if (info)
    {
        LIBRPA::utils::lib_printf(
            "ERROR DESCINIT! PROC %d (%d,%d) PARAMS: DESC %d %d %d %d %d %d %d %d\n",
            myid_, myprow_, mypcol_, m, n, mb, nb, irsrc, icsrc, ictxt_, m_local_);
    }
    // else
    //     LIBRPA::utils::lib_printf("SUCCE DESCINIT! PROC %d (%d,%d) PARAMS: DESC %d %d %d %d %d %d %d %d\n", myid_, myprow_, mypcol_, m, n, mb, nb, irsrc, icsrc, ictxt_, m_local_);
    m_ = desc[2];
    n_ = desc[3];
    mb_ = desc[4];
    nb_ = desc[5];
    irsrc_ = desc[6];
    icsrc_ = desc[7];
    lld_ = desc[8];
    initialized_ = true;
    return info;
}

Array_Desc::Array_Desc(const BLACS_CTXT_handler &blacs_h)
    : ictxt_(0), nprocs_(0), myid_(0),
      nprows_(0), myprow_(0), npcols_(0), mypcol_(0),
      m_(0), n_(0), mb_(0), nb_(0), irsrc_(0), icsrc_(0),
      lld_(0), m_local_(0), n_local_(0),
      empty_local_mat_(false), initialized_(false)
{
    if (!blacs_h.initialized())
        throw std::logic_error("BLACS context is not initialized before creating ArrayDesc");
    set_blacs_params_(blacs_h.ictxt, blacs_h.nprocs, blacs_h.myid,
                      blacs_h.nprows, blacs_h.myprow, blacs_h.npcols,
                      blacs_h.mypcol);
}

Array_Desc::Array_Desc(const int &ictxt)
    : ictxt_(0), nprocs_(0), myid_(0),
      nprows_(0), myprow_(0), npcols_(0), mypcol_(0),
      m_(0), n_(0), mb_(0), nb_(0), irsrc_(0), icsrc_(0),
      lld_(0), m_local_(0), n_local_(0),
      empty_local_mat_(false), initialized_(false)
{
    int nprocs, myid, nprows, npcols, myprow, mypcol;
    Cblacs_gridinfo(ictxt, &nprows, &npcols, &myprow, &mypcol);
    myid = Cblacs_pnum(ictxt, myprow, mypcol);
    nprocs = nprows * npcols;
    set_blacs_params_(ictxt, nprocs, myid,
                      nprows, myprow, npcols,
                      mypcol);
}

int Array_Desc::init(const int &m, const int &n, const int &mb, const int &nb,
                    const int &irsrc, const int &icsrc)
{
    return set_desc_(m, n, mb, nb, irsrc, icsrc);
}

int Array_Desc::init_1b1p(const int &m, const int &n,
                          const int &irsrc, const int &icsrc)
{
    int mb = 1, nb = 1;
    mb = std::ceil(double(m)/nprows_);
    nb = std::ceil(double(n)/npcols_);
    return set_desc_(m, n, mb, nb, irsrc, icsrc);
}

int Array_Desc::init_square_blk(const int &m, const int &n,
                                    const int &irsrc, const int &icsrc)
{
    int mb = 1, nb = 1, minblk = 1;
    mb = std::ceil(double(m)/nprows_);
    nb = std::ceil(double(n)/npcols_);
    minblk = std::min(mb, nb);
    return set_desc_(m, n, minblk, minblk, irsrc, icsrc);
}

std::string Array_Desc::info() const
{
    std::string info;
    info = std::string("ArrayDesc: ")
         + "ICTXT " + std::to_string(ictxt_) + " "
         + "ID " + std::to_string(myid_) + " "
         + "PCOOR (" + std::to_string(myprow_) + "," + std::to_string(mypcol_) + ") "
         + "GSIZE (" + std::to_string(m_) + "," + std::to_string(n_) + ") "
         + "LSIZE (" + std::to_string(m_local_) + "," + std::to_string(n_local_) + ") "
         + "DUMMY? " + std::string(empty_local_mat_? "T" : "F");
    return info;
}

std::string Array_Desc::info_desc() const
{
    char s[100];
    sprintf(s, "DESC %d %d %d %d %d %d %d %d %d",
            desc[0], desc[1], desc[2],
            desc[3], desc[4], desc[5],
            desc[6], desc[7], desc[8]);
    return std::string(s);
}

bool Array_Desc::is_src() const { return myprow_ == irsrc_ && mypcol_ == icsrc_; }

void Array_Desc::barrier(CTXT_SCOPE scope)
{
    CTXT_barrier(ictxt_, scope);
}

int Array_Desc::indx_g2l_r(int gindx) const
{
    return myprow_ != ScalapackConnector::indxg2p(gindx, mb_, myprow_, irsrc_, nprows_) || gindx >= m_
               ? -1
               : ScalapackConnector::indxg2l(gindx, mb_, myprow_, irsrc_, nprows_);
	// int inproc = int((gindx % (mb_*nprows_)) / mb_);
	// if(myprow_==inproc)
	// {
	// 	return int(gindx / (mb_*nprows_))*mb_ + gindx % mb_;
	// }
	// else
	// {
	// 	return -1;
	// }
}

int Array_Desc::indx_g2l_c(int gindx) const
{
    return mypcol_ != ScalapackConnector::indxg2p(gindx, nb_, mypcol_, icsrc_, npcols_) || gindx >= n_
               ? -1
               : ScalapackConnector::indxg2l(gindx, nb_, mypcol_, icsrc_, npcols_);
	// int inproc = int((gindx % (nb_*npcols_)) / mb_);
	// if(mypcol_==inproc)
	// {
	// 	return int(gindx / (nb_*npcols_))*nb_ + gindx % nb_;
	// }
	// else
	// {
	// 	return -1;
	// }
}

int Array_Desc::indx_l2g_r(int lindx) const
{
    return ScalapackConnector::indxl2g(lindx, mb_, myprow_, irsrc_, nprows_);
	// int iblock, gIndex;
	// iblock = lindx / mb_;
	// gIndex = (iblock*nprows_ + myprow_)* mb_ + lindx % mb_;
	// return gIndex;
}

int Array_Desc::indx_l2g_c(int lindx) const
{
    return ScalapackConnector::indxl2g(lindx, nb_, mypcol_, icsrc_, npcols_);
	// int iblock, gIndex;
	// iblock = lindx / nb_;
	// gIndex = (iblock*npcols_ + mypcol_)* nb_ + lindx % nb_;
	// return gIndex;
}

} /* end of namespace LIBRPA */
