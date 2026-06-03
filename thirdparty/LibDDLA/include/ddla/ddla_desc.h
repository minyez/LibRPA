#ifndef DDLA_DESC_H
#define DDLA_DESC_H

#include "ddla_handle_t.h"

namespace ddla{

inline int indxg2p(const int &indxglob, const int &nb,
                              const int &isrcproc, const int &nprocs)
{
    return (isrcproc + indxglob / nb) % nprocs;
}

inline int indxg2l(const int &indxglob, const int &nb, const int &nprocs)
{
    return nb * (indxglob / (nb * nprocs)) + indxglob % nb;
}
inline int indxl2g(const int &indxloc, const int &nb, const int &iproc, const int &isrcproc, const int &nprocs)
{
    return nprocs * nb * (indxloc / nb) + indxloc % nb +
            ((nprocs + iproc - isrcproc) % nprocs) * nb;
}

inline int num_loc(const int& n, const int& nb, const int& iproc, const int& srcproc, const int& nprocs)
{
    int count = n / (nb * nprocs) * nb;
    int rest = n % (nb * nprocs);
    if (rest > nb * ((iproc + nprocs - srcproc) % nprocs))
    {
        if (rest - nb * ((iproc + nprocs - srcproc) % nprocs) >= nb)
            count += nb;
        else
            count += rest - nb * ((iproc + nprocs - srcproc) % nprocs);
    }
    return count;
}
// class DdlaStream;

class DdlaDesc{
private:
    int m_;
    int n_;
    int mb_;
    int nb_;
    int irsrc_;
    int icsrc_;
    int m_local_;
    int n_local_;
    int lld_; // leading dimension of local matrix
    int nprows_;
    int npcols_;
    int myprow_;
    int mypcol_;
    DdlaHandle_t ddla_handle_ = nullptr;    
    bool is_initialized_ = false;
public:
    DdlaDesc(const DdlaHandle_t& ddla_handle);
    DdlaDesc(){};
    void set_ddla_handle(const DdlaHandle_t& ddla_handle);
    void init_square_blk(const int &m, const int &n, const int &irsrc, const int &icsrc);
    void init(const int &m, const int &n, const int &mb, const int &nb, const int &irsrc, const int &icsrc);
    int indx_g2l_r(int gindx) const;
    int indx_g2l_c(int gindx) const{
        if(this->mypcol_ != indxg2p(gindx, this->nb_, this->icsrc_, this->npcols_) || gindx >= this->n_)
            return -1;
        return indxg2l(gindx, this->nb_, this->npcols_);
    }
    int indx_l2g_r(int lindx) const{
        return indxl2g(lindx, this->mb_, this->myprow_, this->irsrc_, this->nprows_);
    }
    int indx_l2g_c(int lindx) const{
        return indxl2g(lindx, this->nb_, this->mypcol_, this->icsrc_, this->npcols_);
    }
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
    const bool& is_initialized() const { return is_initialized_; }

    const DdlaHandle_t& ddla_handle() const{ return ddla_handle_; }
    
};

} // end of namespace DDLA


#endif // DDLA_DESC_H