#include "hartree.h"
#include "constants.h"
#include "envs_blacs.h"
#include "envs_io.h"
#include "envs_mpi.h"
#include "geometry.h"
#include "lapack_connector.h"
#include "libri_utils.h"
#include "matrix_m_parallel_utils.h"
#include "params.h"
#include "pbc.h"
#include "profiler.h"
#include "stl_io_helper.h"
#include "utils_blacs.h"
#include "vector3_order.h"
#ifdef LIBRPA_USE_LIBRI
#include <RI/physics/Exx.h>
#include <RI/ri/Cell_Nearest.h>
#else
#include "libri_stub.h"
#endif
#include "utils_io.h"

namespace LIBRPA
{

Hartree::Hartree(const MeanField &mf, const vector<Vector3_Order<double>> &kfrac_list,
         const Vector3_Order<int> &period)
    : mf_(mf), kfrac_list_(kfrac_list), period_(period)
{
    is_rspace_build_ = false;
    is_kspace_built_ = false;
};

ComplexMatrix Hartree::get_dmat_cplx_R_global(const int &ispin, const int &isoc1, const int &isoc2,
                                          const Vector3_Order<int> &R)
{
    const auto nspins = this->mf_.get_n_spins();
    auto dmat_cplx = this->mf_.get_dmat_cplx_R(ispin, isoc1, isoc2, this->kfrac_list_, R);
    // renormalize to single spin channel
    if (!Params::use_soc) dmat_cplx *= 0.5 * nspins;

    return dmat_cplx;
}

ComplexMatrix Hartree::extract_dmat_cplx_R_IJblock(const ComplexMatrix &dmat_cplx, const atom_t &I,
                                               const atom_t &J)
{
    const auto I_num = atom_nw.at(I);
    const auto J_num = atom_nw.at(J);
    ComplexMatrix dmat_cplx_IJR(I_num, J_num);
    for (size_t i = 0; i != I_num; i++)
    {
        size_t i_glo = atom_iw_loc2glo(I, i);
        for (size_t j = 0; j != J_num; j++)
        {
            size_t j_glo = atom_iw_loc2glo(J, j);
            dmat_cplx_IJR(i, j) = dmat_cplx(i_glo, j_glo);
        }
    }
    return dmat_cplx_IJR;
}

void Hartree::build_dmat_R(const Vector3_Order<int> &R)
{
    const auto nspins = this->mf_.get_n_spins();
    const auto nsoc = this->mf_.get_n_soc();

    for (int is = 0; is != nspins; is++)
    {
        for (int isoc1 = 0; isoc1 != nsoc; isoc1++)
        {
            for (int isoc2 = 0; isoc2 != nsoc; isoc2++)
            {
                auto dmat_cplx = this->get_dmat_cplx_R_global(is, isoc1, isoc2, R);
                for (int I = 0; I != natom; I++)
                {
                    for (int J = 0; J != natom; J++)
                    {
                        const auto dmat_cplx_IJR =
                            this->extract_dmat_cplx_R_IJblock(dmat_cplx, I, J);
                        this->warn_dmat_IJR_nonzero_imag(dmat_cplx_IJR, is, I, J, R);
                        this->dmat[is][isoc1][isoc2][I][J][R] = std::make_shared<matrix>();
                        *(this->dmat[is][isoc1][isoc2][I][J][R]) = dmat_cplx_IJR.real();
                    }
                }
            }
        }
    }
}

void Hartree::build_dmat_R(const atom_t &I, const atom_t &J, const Vector3_Order<int> &R)
{
    const auto nspins = this->mf_.get_n_spins();
    const auto nsoc = this->mf_.get_n_soc();

    for (int is = 0; is != nspins; is++)
    {
        for (int isoc1 = 0; isoc1 != nsoc; isoc1++)
        {
            for (int isoc2 = 0; isoc2 != nsoc; isoc2++)
            {
                auto dmat_cplx = this->get_dmat_cplx_R_global(is, isoc1, isoc2, R);
                const auto dmat_cplx_IJR = this->extract_dmat_cplx_R_IJblock(dmat_cplx, I, J);
                this->warn_dmat_IJR_nonzero_imag(dmat_cplx_IJR, is, I, J, R);
                this->dmat[is][isoc1][isoc2][I][J][R] = std::make_shared<matrix>();
                *(this->dmat[is][isoc1][isoc2][I][J][R]) = dmat_cplx_IJR.real();
            }
        }
    }
}

void Hartree::warn_dmat_IJR_nonzero_imag(const ComplexMatrix &dmat_cplx, const int &ispin,
                                     const atom_t &I, const atom_t &J, const Vector3_Order<int> R)
{
    if (dmat_cplx.get_max_abs_imag() > 1e-2)
        utils::lib_printf(
            "Warning: complex-valued density matrix, spin %d IJR %zu %zu (%d, %d, %d)\n", ispin, I,
            J, R.x, R.y, R.z);
}


void Hartree::build(const Cs_LRI &Cs, const vector<Vector3_Order<int>> &Rlist,
                const atpair_R_mat_t &coul_mat)
{
    using LIBRPA::envs::mpi_comm_global;
    using LIBRPA::envs::mpi_comm_global_h;
    using LIBRPA::atomic_basis_abf;
    using LIBRPA::atomic_basis_wfc;
    assert(parallel_routing == ParallelRouting::LIBRI);

    if (this->is_rspace_build_)
    {
        return;
    }

    const auto &n_spins = this->mf_.get_n_spins();
    const auto &n_soc = this->mf_.get_n_soc();

#ifdef LIBRPA_USE_LIBRI
    if (mpi_comm_global_h.is_root())
    {
        utils::lib_printf("Computing Hartree orbital energy using LibRI\n");
    }
    mpi_comm_global_h.barrier();

    
    map<int, std::array<double, 3>> atoms_pos;
    for (int i = 0; i != atom_mu.size(); i++)
        atoms_pos.insert(pair<int, std::array<double, 3>>{i, {0, 0, 0}});

    std::array<double, 3> xa{latvec.e11, latvec.e12, latvec.e13};
    std::array<double, 3> ya{latvec.e21, latvec.e22, latvec.e23};
    std::array<double, 3> za{latvec.e31, latvec.e32, latvec.e33};
    std::array<std::array<double, 3>, 3> lat_array{xa, ya, za};
    // 修改 period_array 的初始化方式
    #ifdef LIBRPA_USE_LIBRI
    std::array<int, 3> period_array{period.x, period.y, period.z}; // 使用参数中的 period 而非成员变量
    #else
    std::array<int, 3> period_array{period_.x, period_.y, period_.z};
    #endif
    

    // Initialize Cs libRI container on each process
    // Note: we use different treatment in different routings
    //     R-tau routing:
    //         Each process has a full Cs copy.
    //         Thus in each process we only pass a few to LibRI container.
    //     atom-pair routing:
    //         Cs is already distributed across all processes.
    //         Pass the all Cs to libRI container.

    Profiler::start("build_real_space_Hartree_1", "Prepare C libRI object");
    envs::ofs_myid << "Number of Cs keys: " << get_num_keys(Cs.data_libri) << "\n";
    // print_keys(envs::ofs_myid, Cs.data_libri);

     
    Profiler::stop("build_real_space_Hartree_1");
    envs::ofs_myid << "Finished setup Cs for Hartree\n";
    std::flush(envs::ofs_myid);

    // initialize Coulomb matrix
    Profiler::start("build_real_space_Hartree_2", "Prepare V libRI object");
    std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<double>>> V_libri;
    Profiler::start("build_real_space_Hartree_2_1");
    if (LIBRPA::parallel_routing == LIBRPA::ParallelRouting::R_TAU)
    {
        // Full Coulomb case, have to re-distribute
        for (auto IJR : dispatch_vector_prod(get_atom_pair(coul_mat), Rlist, mpi_comm_global_h.myid,
                                             mpi_comm_global_h.nprocs, true, true))
        {
            const auto I = IJR.first.first;
            const auto J = IJR.first.second;
            const auto R = IJR.second;
            const auto &VIJR = coul_mat.at(I).at(J).at(R);
            std::array<int, 3> Ra{R.x, R.y, R.z};
            std::valarray<double> VIJR_va;
            // utils::lib_printf("Checking  R1: (%d,%d,%d)\n", 
            //                      R.x, R.y, R.z);
            VIJR_va = std::valarray<double>(VIJR->c, VIJR->size);
            auto pv = std::make_shared<std::valarray<double>>();
            *pv = VIJR_va;
            V_libri[I][{J, Ra}] = RI::Tensor<double>({size_t(VIJR->nr), size_t(VIJR->nc)}, pv);
        }
    }
    else
    {
        for (const auto &I_JRV : coul_mat)
        {
            const auto I = I_JRV.first;
            for (const auto &J_RV : I_JRV.second)
            {
                const auto J = J_RV.first;
                for (const auto &R_V : J_RV.second)
                {
                    const auto &R = R_V.first;
                    const auto &V = R_V.second;
                    std::array<int, 3> Ra{R.x, R.y, R.z};
                    std::valarray<double> VIJR_va;
                    // utils::lib_printf("Checking  R1: (%d,%d,%d)\n", 
                    //              R.x, R.y, R.z);
                    VIJR_va = std::valarray<double>(V->c, V->size);
                    auto pv = std::make_shared<std::valarray<double>>();
                    *pv = VIJR_va;
                    V_libri[I][{J, Ra}] = RI::Tensor<double>({size_t(V->nr), size_t(V->nc)}, pv);
                }
            }
        }
    }
    Profiler::cease("build_real_space_Hartree_2_1");
    envs::ofs_myid << "Number of V keys: " << get_num_keys(V_libri) << "\n";
    Profiler::start("build_real_space_Hartree_2_2");
    
     

    Profiler::cease("build_real_space_Hartree_2_2");
    Profiler::cease("build_real_space_Hartree_2");
    utils::lib_printf("Task %4d: V setup for Hartree\n", mpi_comm_global_h.myid);
    // cout << V_libri << endl;



    
    // initialize density matrix
    vector<atpair_t> atpair_dmat;
    for (int I = 0; I < atom_nw.size(); I++)
        for (int J = 0; J < atom_nw.size(); J++) atpair_dmat.push_back({I, J});
    const auto dmat_IJRs_local = dispatch_vector_prod(atpair_dmat, Rlist, mpi_comm_global_h.myid,
                                                      mpi_comm_global_h.nprocs, true, true);
    
    for (auto isp = 0; isp != n_spins; isp++) {
        for (auto is1 = 0; is1 != n_soc; is1++) {
            for (auto is2 = 0; is2 != n_soc; is2++) {
                Profiler::start("build_real_space_Hartree_3", "Prepare DM libRI object");
                std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<double>>> dmat_libri;
                
                for (const auto &R : Rlist) {
                    std::array<int, 3> Ra{R.x, R.y, R.z};
                    // utils::lib_printf("Checking  R2: (%d,%d,%d)\n", 
                    //              R.x, R.y, R.z);
                    const auto dmat_cplx = this->get_dmat_cplx_R_global(isp, is1, is2, R);
                    for (const auto &IJR : dmat_IJRs_local)
                    {
                        if (IJR.second == R)
                        {
                            const auto &I = IJR.first.first;
                            const auto &J = IJR.first.second;
                            const auto dmat_IJR =
                                this->extract_dmat_cplx_R_IJblock(dmat_cplx, I, J);
                            this->warn_dmat_IJR_nonzero_imag(dmat_IJR, isp, I, J, R);
                            std::valarray<double> dmat_va;
                            
                            dmat_va = std::valarray<double>(dmat_IJR.real().c, dmat_IJR.size);
                            auto pdmat = std::make_shared<std::valarray<double>>();
                            *pdmat = dmat_va;
                            dmat_libri[I][{J, Ra}] = RI::Tensor<double>(
                                {size_t(dmat_IJR.nr), size_t(dmat_IJR.nc)}, pdmat);
                        }
                    }
                }
                
                for (const auto &R : Rlist) {
                    std::array<int, 3> Ra{R.x, R.y, R.z};
                    // utils::lib_printf("Checking  R: (%d,%d,%d)\n", 
                    //              R.x, R.y, R.z);
                    // utils::lib_printf("Checking  Ra: (%d,%d,%d)\n", 
                    //              Ra[0], Ra[1], Ra[2]); 
                    
                    const int n_atom = Cs.data_libri.size();
                    // std::cout << "Number of atom: " << n_atom << std::endl;
                    // utils::lib_printf("Step 1\n");
                    // Step 1: 初始化并计算S1[I](μ)
                    std::map<int, std::vector<double>> S1;
                    for (const auto&I_JR : Cs.data_libri) {
                        const auto& I = I_JR.first;
                        const int n_mu = Cs.data_libri.at(I).begin()->second.shape[0];
                        S1[I].resize(n_mu, 0.0);
                    }
                    
                    
                    // utils::lib_printf("Step 1.1\n");

                    // for (int I = 0; I < n_atom; I++)
                    // {
                    //     for (int J = 0; J < n_atom; J++)
                    //     {
                    for (const auto&I_JR : Cs.data_libri) {
                        const auto& I = I_JR.first;
                        // utils::lib_printf("Cs.data_libri_I=%d\n", I);
                        for (const auto& JR_Cs : I_JR.second) {
                            const auto& J = JR_Cs.first.first;
                            // utils::lib_printf("Cs.data_libri_J=%d\n", J);
                            const auto& Rc = JR_Cs.first.second;
                            // utils::lib_printf("Checking Ra: (%d,%d,%d)\n", 
                            //     Rc[0], Rc[1], Rc[2]);
                            if(Rc[0]==R.x && Rc[1]==R.y && Rc[2]==R.z) 
                            {
                                // utils::lib_printf("Checking Ra: (%d,%d,%d) vs current R: (%d,%d,%d)\n", 
                                //     Rc[0], Rc[1], Rc[2], R.x, R.y, R.z);
                                // std::cout << "I: " << I << std::endl;
                                // std::cout << "J: " << J << std::endl;
                                const auto& Cs_block = JR_Cs.second;
                                // 添加维度检查
                                // utils::lib_printf("Step 1.2\n");
                                int max_mu = atomic_basis_abf.get_atom_nb(I);
                                // utils::lib_printf("max_mu: %d\n", max_mu);
                                // utils::lib_printf("Step 1.3\n");
                                int max_i = atomic_basis_wfc.get_atom_nb(I);
                                // utils::lib_printf("max_i: %d\n", max_i);
                                // utils::lib_printf("Step 1.4\n");
                                int max_j = atomic_basis_wfc.get_atom_nb(J);
                                // utils::lib_printf("max_j: %d\n", max_j);
                                // utils::lib_printf("Step 1.5\n");
                                
                                
                                // if(dmat_libri.at(I).count({J, R}) == 0) continue;
                                
                                for(int k = 0; k < max_mu ; k++){
                                    // utils::lib_printf("Step 1.6\n");
                                    for(int i = 0; i < max_i ; i++){
                                        // utils::lib_printf("Step 1.7\n");
                                        for(int j = 0; j < max_j ; j++){
                                            // utils::lib_printf("Step 1.8\n");
                                            const auto& D_block = dmat_libri.at(I).at({J, Rc});  
                                            // utils::lib_printf("Step 1.8.1\n");
                                            // const auto& Cs_block = Cs.data_libri.at(I).at({J, Ra});     // 
                                            // utils::lib_printf("Step 1.8.2\n");                        
                                            const double cval = Cs_block(k, i, j);
                                            // utils::lib_printf("Step 1.8.3\n");
                                            const double dval = D_block(i, j);
                                            // utils::lib_printf("Step 1.9\n");
                                            S1[I][k] += cval * dval * 2.0 ;
                                            
                                            // utils::lib_printf("Si[%d][%d] = %e,cval * dval = %e  \n",
                                            //     I, k, S1[I][k], cval * dval);
                                            
                                            // utils::lib_printf("Step 1.10\n");
                                        }
                                    }
                                    // utils::lib_printf("Si[%d][%d] = %e \n",
                                    //             I, k, S1[I][k]);
                                }
                            }
                                
                                
                            
                        
                        
                        }
                    }
                    
                        
                        
                
                            
                    
                    // utils::lib_printf("Step 2\n");
                    // Step 2: 计算S2[P](ν)
                    std::map<int, std::vector<double>> S2;
                    for (const auto& P_IV : V_libri) {
                        const auto& P = P_IV.first;
                        const int n_nu = V_libri.at(P).begin()->second.shape[0];
                        // utils::lib_printf("n_nu: %d\n", n_nu);
                        // int max_nu = atomic_basis_abf.get_atom_nb(P);
                        // utils::lib_printf("max_nu: %d\n", max_nu);
                        S2[P].resize(n_nu, 0.0);
                    }
                    // for (const auto& P_IV : dmat_libri) {
                    //     const auto& P = P_IV.first;
                    //     utils::lib_printf("dmat_libri_P: %d\n", P);
                    //     for (const auto& IV_V : P_IV.second) {
                    //         const auto& I = IV_V.first.first;
                    //         utils::lib_printf("dmat_libri_I: %d\n", I);
                            
                    //     }
                    // }
                    for (const auto& P_IV : V_libri) {
                        const auto& P = P_IV.first;    
                        // utils::lib_printf("V_libri_P: %d\n", P);
                        for (const auto& IV_V : P_IV.second) {
                            const auto& I = IV_V.first.first;
                            // utils::lib_printf("V_libri_I: %d\n", I);
                            const auto& Rc = IV_V.first.second;
                            
                            int max_nu = atomic_basis_abf.get_atom_nb(P);
                            int max_mu = atomic_basis_abf.get_atom_nb(I);
                            // utils::lib_printf("Step 2.1\n");
                            if(Rc[0]==R.x && Rc[1]==R.y && Rc[2]==R.z){
                                // utils::lib_printf("Checking Rb: (%d,%d,%d)\n", 
                                // Rc[0], Rc[1], Rc[2]);
                                // utils::lib_printf("Step 2.1.1\n");
                                const auto& V_block = IV_V.second;        
                                // utils::lib_printf("Step 2.1.2\n");
                                for(int nu = 0; nu < max_nu ; nu++){
                                    for(int mu = 0; mu < max_mu ; mu++)
                                    {
                                        //printf mu,nu
                                        // utils::lib_printf("mu,nu", mu, nu); 
                                        const auto vval = V_block(nu, mu);
                                        // utils::lib_printf("Step 2.1.3\n");
                                        const auto s1val = S1[I][mu];
                                        // utils::lib_printf("Step 2.2\n");
                                        S2[P][nu] += vval * s1val;
                                    }
                                }
                                
                                // // 输出非零值
                                // if(std::abs(vval) > 1e-10 || std::abs(s1val) > 1e-10) {
                                //     utils::lib_printf("V[%d][%d] = %e, S1[%d][%d] = %e (product = %e)\n",
                                //         nu, mu, vval, I, mu, s1val, vval*s1val);
                                // }                                  
                            }
                        }
                    }
                    // utils::lib_printf("Step 3\n");
                    // Step 3: 计算并存储Hartree项
                    std::map<int, std::map<int, Matd>> S3;
                    for (const auto& P_QC : Cs.data_libri) {
                        const auto& P = P_QC.first;
                        for (const auto& QC_Cs : P_QC.second) {
                            const auto& Q = QC_Cs.first.first;
                            // const auto& Ra = QC_Cs.first.second;
                            const auto& Cs_block = QC_Cs.second;
                            
                            S3[P][Q] = Matd(Cs_block.shape[1], Cs_block.shape[2]); // S3[P][Q](p,q) = Cs[P][{Q,Ra}](ν,p,q) * S2[P](ν)
                        }
                    }
                    // utils::lib_printf("Step 3.1\n");
                    for (const auto&I_JR : Cs.data_libri) {
                        const auto& P = I_JR.first;
                        for (const auto& JR_Cs : I_JR.second) {
                            const auto& Q = JR_Cs.first.first;
                            const auto& Rc = JR_Cs.first.second;
                            // utils::lib_printf("Checking Rc: (%d,%d,%d) vs current R: (%d,%d,%d)\n", 
                            //     Rc[0], Rc[1], Rc[2], R.x, R.y, R.z);
                            if(Rc[0]==R.x && Rc[1]==R.y && Rc[2]==R.z) 
                            {
                                // utils::lib_printf("Checking Rc: (%d,%d,%d) vs current R: (%d,%d,%d)\n", 
                                //     Rc[0], Rc[1], Rc[2], R.x, R.y, R.z);
                            
                                // for (auto& P_S3 : S3) {
                                //     const auto& P = P_S3.first;
                                //     for (auto& Q_S3 : P_S3.second) {
                                //         const auto& Q = Q_S3.first;
                                //         auto& S3_block = Q_S3.second; // S3[P][Q](p,q)
                                
                                int max_nu = atomic_basis_abf.get_atom_nb(P);
                                int max_nu_2 = atomic_basis_abf.get_atom_nb(Q);
                                int max_p = atomic_basis_wfc.get_atom_nb(P);
                                int max_q = atomic_basis_wfc.get_atom_nb(Q);
                                
                                
                                const auto key = std::make_pair(Q, Rc);
                                const auto key_2 = std::make_pair(P, Rc);
                                if (Cs.data_libri.at(P).count(key) == 0) continue;

                                // utils::lib_printf("Step 3.2.1\n");
                                const auto& Cs_block = Cs.data_libri.at(P).at(key);
                                const auto& Cs_block_2 = Cs.data_libri.at(Q).at(key_2);
                                // auto& S3_block = S3.at(P).at(Q);
                                // utils::lib_printf("Step 3.2.2\n");
                                for(int p = 0; p < max_p; p++){
                                    for(int q = 0; q < max_q; q++){
                                        for(int nu = 0; nu < max_nu; nu++){
                                            const auto Cs_val_1 = Cs_block(nu, p, q) ;
                                            const auto S2_val_1 = S2[P][nu];
                                            
                                            S3[P][Q](p,q) += Cs_val_1 * S2_val_1  ;
                                        }
                                        for(int nu_2 = 0; nu_2 < max_nu_2; nu_2++){
                                            const auto Cs_val_2 = Cs_block_2(nu_2, q, p);
                                            const auto S2_val_2 = S2[Q][nu_2];
                                            
                                            S3[P][Q](p,q) += Cs_val_2 * S2_val_2  ;
                                        }

                                        // utils::lib_printf("S3[%d][%d](%d,%d) = %e\n",
                                        //     P, Q, p, q, S3[P][Q](p,q));
                                    }
                                }
                                this->hartree[isp][is1][is2][R][P][Q] = -2.0 * S3[P][Q];
                                // utils::lib_printf("print:Hartree[%d][%d][%d](%d,%d,%d)[%d][%d] = %e\n",
                                // isp, is1, is2, R.x, R.y, R.z, P, Q, this->hartree[isp][is1][is2][R][P][Q](0,0));
                            }
                            // utils::lib_printf("Step 3.2.3\n");
                            // 添加R变量到存储路径
                            
                            

                            // utils::lib_printf("Step 3.2\n");
                             
                        
                        }
                    }
                }
                
                utils::lib_printf("Step4\n");

                // 移动日志输出到有效作用域
                envs::ofs_myid << "Number of Dmat keys: " << get_num_keys(dmat_libri) << "\n";
                // print_keys(envs::ofs_myid, dmat_libri);
                
                Profiler::stop("build_real_space_Hartree_3");
                utils::lib_printf("Task %4d: DM setup for Hartree\n", mpi_comm_global_h.myid);
                

                utils::lib_printf("Task %4d: cal_Hs elapsed time: %f\n", mpi_comm_global_h.myid,
                                    Profiler::get_wall_time_last("build_real_space_Hartree_4"));
            
            }
                

            
        }
    }
    


#else
    if (mpi_comm_global_h.is_root())
    {
        utils::lib_printf(
            "Error: trying build Hartree orbital energy with LibRI, but the program is not compiled "
            "against LibRI\n");
    }
    throw std::logic_error("compilation");
    mpi_comm_global_h.barrier();
#endif

    is_rspace_build_ = true;
}



void Hartree::build_KS(const std::vector<std::vector<std::vector<ComplexMatrix>>> &wfc_target,
                   const std::vector<Vector3_Order<double>> &kfrac_target)
{
    using LIBRPA::envs::blacs_ctxt_global_h;
    using LIBRPA::envs::mpi_comm_global_h;
    using RI::Communicate_Tensors_Map_Judge::comm_map2_first;

    assert(this->is_rspace_build_);
    // Reset k-space matrices built from last call
    if (this->is_kspace_built_)
    {
        utils::lib_printf("Warning: reset Hartree k-space matrices\n");
        this->reset_kspace();
    }

    const auto &n_aos = this->mf_.get_n_aos();
    const auto &n_spins = this->mf_.get_n_spins();
    const auto &n_bands = this->mf_.get_n_bands();
    const auto &n_soc = this->mf_.get_n_soc();

    // prepare scalapack array descriptors
    Array_Desc desc_nao_nao(blacs_ctxt_global_h);
    Array_Desc desc_nband_nao(blacs_ctxt_global_h);
    Array_Desc desc_nband_nband(blacs_ctxt_global_h);
    Array_Desc desc_nband_nband_fb(blacs_ctxt_global_h);

    desc_nao_nao.init_1b1p(n_aos, n_aos, 0, 0);
    desc_nband_nao.init_1b1p(n_bands, n_aos, 0, 0);
    desc_nband_nband.init_1b1p(n_bands, n_bands, 0, 0);
    desc_nband_nband_fb.init(n_bands, n_bands, n_bands, n_bands, 0, 0);

    // local 2D-block submatrices
    auto HHartree_nao_nao = init_local_mat<complex<double>>(desc_nao_nao, MAJOR::COL);
    auto temp_nband_nao = init_local_mat<complex<double>>(desc_nband_nao, MAJOR::COL);
    auto HHartree_nband_nband = init_local_mat<complex<double>>(desc_nband_nband, MAJOR::COL);
    auto HHartree_nband_nband_fb = init_local_mat<complex<double>>(desc_nband_nband_fb, MAJOR::COL);

    const auto set_IJ_naonao = LIBRPA::utils::get_necessary_IJ_from_block_2D(
        atomic_basis_wfc, atomic_basis_wfc, desc_nao_nao);
    const auto Iset_Jset = convert_IJset_to_Iset_Jset(set_IJ_naonao);
    // char fn[80];
    // sprintf(fn, "hartree.mtx");
    for (int isp = 0; isp < n_spins; isp++)
    {
        for (int isoc1 = 0; isoc1 < n_soc; isoc1++)
        {
            for (int isoc2 = 0; isoc2 < n_soc; isoc2++)
            {
                // collect necessary data
                Profiler::start("build_real_space_Hartree_5", "Collect HHartree IJ from world");
                map<Vector3_Order<int>, map<atom_t, map<atom_t, Matz>>> Hartree_is;
                
                if (this->hartree.count(isp) && this->hartree.at(isp).count(isoc1)
                        && this->hartree.at(isp).at(isoc1).count(isoc2))
                    {
                        for (const auto &R_IJ_Hartree : this->hartree.at(isp).at(isoc1).at(isoc2))
                        {
                            const auto R = R_IJ_Hartree.first;
                            for (const auto &I_J_Hartree : R_IJ_Hartree.second)
                            {
                                const auto I = I_J_Hartree.first;
                                for (const auto &J_Hartree : I_J_Hartree.second)
                                {
                                    const auto J = J_Hartree.first;
                                    Hartree_is[R][I][J] = J_Hartree.second.to_complex();
                                }
                            }
                        }
                    }

                std::map<int, std::map<std::pair<int, std::array<int, 3>>,
                                       RI::Tensor<std::complex<double>>>>
                    Hartree_I_JR_local;
                for (const auto &R_IJ_Hartree : Hartree_is)
                {
                    const auto R = R_IJ_Hartree.first;
                    for (const auto &I_J_Hartree : R_IJ_Hartree.second)
                    {
                        const auto I = I_J_Hartree.first;
                        const auto &n_I = atomic_basis_wfc.get_atom_nb(I);
                        for (const auto &J_Hartree : I_J_Hartree.second)
                        {
                            const auto J = J_Hartree.first;
                            const auto &n_J = atomic_basis_wfc.get_atom_nb(J);
                            const std::array<int, 3> Ra{R.x, R.y, R.z};
                            Hartree_I_JR_local[I][{J, Ra}] =
                                RI::Tensor<std::complex<double>>({n_I, n_J}, J_Hartree.second.sptr());
                        }
                    }
                }
                // Collect the IJ pair of Hs with all R for Fourier transform
                auto Hartree_I_JR = comm_map2_first(mpi_comm_global_h.comm, Hartree_I_JR_local,
                                                Iset_Jset.first, Iset_Jset.second);
                Hartree_I_JR_local.clear();

                // Convert each <I,<J, R>> pair to the nearest neighbour to speed up later
                // Fourier transform while keep the accuracy in further band interpolation.
                // Reuse the cleared-up Hartree_I_JR_local object
                if (coord_frac.size() > 0)
                {
                    for (auto &I_HartreeJR : Hartree_I_JR)
                    {
                        const auto &I = I_HartreeJR.first;
                        for (auto &JR_Hartree : I_HartreeJR.second)
                        {
                            const auto &J = JR_Hartree.first.first;
                            const auto &R = JR_Hartree.first.second;

                            auto distsq = std::numeric_limits<double>::max();
                            Vector3<int> R_IJ;
                            std::array<int, 3> R_bvk;
                            for (int i = -1; i < 2; i++)
                            {
                                R_IJ.x = i * this->period_.x + R[0];
                                for (int j = -1; j < 2; j++)
                                {
                                    R_IJ.y = j * this->period_.y + R[1];
                                    for (int k = -1; k < 2; k++)
                                    {
                                        R_IJ.z = k * this->period_.z + R[2];
                                        const auto diff =
                                            (Vector3<double>(coord_frac[I][0], coord_frac[I][1],
                                                             coord_frac[I][2]) -
                                             Vector3<double>(coord_frac[J][0], coord_frac[J][1],
                                                             coord_frac[J][2]) -
                                             Vector3<double>(R_IJ.x, R_IJ.y, R_IJ.z)) *
                                            latvec;
                                        const auto norm2 = diff.norm2();
                                        if (norm2 < distsq)
                                        {
                                            distsq = norm2;
                                            R_bvk[0] = R_IJ.x;
                                            R_bvk[1] = R_IJ.y;
                                            R_bvk[2] = R_IJ.z;
                                        }
                                    }
                                }
                            }
                            Hartree_I_JR_local[I][{J, R_bvk}] = std::move(JR_Hartree.second);
                        }
                    }
                }
                else
                {
                    Hartree_I_JR_local = std::move(Hartree_I_JR);
                }

                Hartree_I_JR.clear();
                Profiler::stop("build_real_space_Hartree_5");

                utils::lib_printf("Task %4d: tensor communicate elapsed time: %f\n",
                                  mpi_comm_global_h.myid,
                                  Profiler::get_wall_time_last("build_real_space_Hartree_5"));
                // cout << I_JallR_Hs << endl;

                for (int ik = 0; ik < kfrac_target.size(); ik++)
                {
                    HHartree_nao_nao.zero_out();
                    Profiler::start("build_real_space_Hartree_6", "HHartree IJ -> 2D block");
                    const auto &kfrac = kfrac_target[ik];
                    const std::function<complex<double>(const int &,
                                                        const std::pair<int, std::array<int, 3>> &)>
                        fourier =
                            [kfrac](const int &I, const std::pair<int, std::array<int, 3>> &J_Ra)
                    {
                        const auto &Ra = J_Ra.second;
                        Vector3<double> R_IJ(Ra[0], Ra[1], Ra[2]);
                        const auto ang = (kfrac * R_IJ) * TWO_PI;
                        return complex<double>{std::cos(ang), std::sin(ang)};
                    };
                    collect_block_from_IJ_storage_tensor_transform(
                        HHartree_nao_nao, desc_nao_nao, atomic_basis_wfc, atomic_basis_wfc, fourier,
                        Hartree_I_JR_local);
                    Profiler::stop("build_real_space_Hartree_6");
                    // utils::lib_printf("%s\n", str(HHartree_nao_nao).c_str());
                    if (this->Hartree_is_ik_nao.count(isp) == 0 ||
                        this->Hartree_is_ik_nao[isp].count(ik) == 0)
                    {
                        this->Hartree_is_ik_nao[isp][ik] =
                            init_local_mat<complex<double>>(desc_nao_nao, MAJOR::COL);
                    }
                    this->Hartree_is_ik_nao[isp][ik] += HHartree_nao_nao.copy();
                    
                    const auto &wfc_isp1_k = wfc_target[isp][isoc1][ik];
                    const auto &wfc_isp2_k = wfc_target[isp][isoc2][ik];
                    blacs_ctxt_global_h.barrier();
                    const auto wfc1_block =
                        get_local_mat(wfc_isp1_k.c, MAJOR::ROW, desc_nband_nao, MAJOR::COL).conj();
                    const auto wfc2_block =
                        get_local_mat(wfc_isp2_k.c, MAJOR::ROW, desc_nband_nao, MAJOR::COL).conj();
                    // utils::lib_printf("%s\n", str(wfc_block).c_str());
                    // utils::lib_printf("%s\n", desc_nao_nao.info_desc().c_str());
                    // utils::lib_printf("%s\n", desc_nband_nao.info_desc().c_str());
                    Profiler::start("build_real_space_Hartree_7", "Rotate HHartree ij -> KS");
                    ScalapackConnector::pgemm_f('N', 'N', n_bands, n_aos, n_aos, 1.0,
                                                wfc1_block.ptr(), 1, 1, desc_nband_nao.desc,
                                                HHartree_nao_nao.ptr(), 1, 1, desc_nao_nao.desc, 0.0,
                                                temp_nband_nao.ptr(), 1, 1, desc_nband_nao.desc);
                    ScalapackConnector::pgemm_f(
                        'N', 'C', n_bands, n_bands, n_aos, -1.0, temp_nband_nao.ptr(), 1, 1,
                        desc_nband_nao.desc, wfc2_block.ptr(), 1, 1, desc_nband_nao.desc, 0.0,
                        HHartree_nband_nband.ptr(), 1, 1, desc_nband_nband.desc);
                    Profiler::stop("build_real_space_Hartree_7");

                    // collect to master
                    Profiler::start("build_real_space_Hartree_8", "Collect EHartree to root process");
                    ScalapackConnector::pgemr2d_f(n_bands, n_bands, HHartree_nband_nband.ptr(), 1, 1,
                                                  desc_nband_nband.desc, HHartree_nband_nband_fb.ptr(),
                                                  1, 1, desc_nband_nband_fb.desc,
                                                  desc_nband_nband_fb.ictxt());
                    if (this->Hartree_is_ik_KS.count(isp) == 0 ||
                        this->Hartree_is_ik_KS[isp].count(ik) == 0)
                    {
                        this->Hartree_is_ik_KS[isp][ik] =
                            init_local_mat<complex<double>>(desc_nband_nband_fb, MAJOR::COL);
                    }
                    this->Hartree_is_ik_KS[isp][ik] += HHartree_nband_nband_fb.copy();
                    // for (int ib = 0; ib != n_bands; ib++){
                    //     for (int jb = 0; jb != n_bands; jb++){
                    //         printf("Hartree_is_ik_KS[%d][%d] (%d,%d) = %e\n", isp, ik, ib, jb,
                    //                this->Hartree_is_ik_KS[isp][ik](ib, jb).real());
                    //         printf("Hartree_is_ik_KS[%d][%d] (%d,%d) = %e\n", isp, ik, ib, jb,
                    //                this->Hartree_is_ik_KS[isp][ik](ib, jb).imag());
                    //     }
                    // }        
                    // print_matrix_mm_file(HHartree_nband_nband_fb,Params::output_dir + "/" + fn, 1e-15);
                    // cout << "HHartree_nband_nband_fb isp " << isp  << " ik " << ik << endl <<
                    // HHartree_nband_nband_fb;
                    if (blacs_ctxt_global_h.myid == 0)
                    {
                        for (int ib = 0; ib != n_bands; ib++)
                            this->EHartree[isp][ik][ib] += HHartree_nband_nband_fb(ib, ib).real();
                    }
                    Profiler::stop("build_real_space_Hartree_8");
                }
            }
        }
    }
}
void Hartree::build_KS_kgrid0()
{
    this->build_KS(this->mf_.get_eigenvectors0(), this->kfrac_list_);
}
void Hartree::build_KS_kgrid() { this->build_KS(this->mf_.get_eigenvectors(), this->kfrac_list_); }

void Hartree::build_KS_band(const std::vector<std::vector<std::vector<ComplexMatrix>>> &wfc_band,
                        const std::vector<Vector3_Order<double>> &kfrac_band)
{
    this->build_KS(wfc_band, kfrac_band);
}

void Hartree::reset_rspace()
{
    this->hartree.clear();
    this->is_rspace_build_ = false;
}

void Hartree::reset_kspace()
{
    this->Hartree_is_ik_KS.clear();
    this->EHartree.clear();
    this->is_kspace_built_ = false;
}

}  // namespace LIBRPA
