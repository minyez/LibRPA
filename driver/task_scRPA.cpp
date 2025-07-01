#include "task_scRPA.h"
#include "task_qsgw.h"
#include "app_rpa.h"
// 标准库头文件
#include <cmath>
#include <fstream>   // 用于文件存在检查
#include <iomanip>   // 用于格式化
#include <iostream>  // 用于输入输出操作
#include <map>       // 用于std::map容器
#include <sstream>
#include <string>  // 用于std::string类
#include <vector>
// 自定义头文件

#include "Hamiltonian.h"  // 哈密顿量相关
#include "analycont.h"    // 分析延拓相关
#include "chi0.h"         // 响应函数相关
#include "constants.h"    // 常量定义
#include "convert_csc.h"
#include "coulmat.h"  // 库仑矩阵相关
#include "driver_params.h"
#include "driver_utils.h"
#include "envs_io.h"
#include "envs_mpi.h"
#include "epsilon.h"
#include "hartree.h"                  
#include "exx.h"                      // Exact exchange相关
#include "fermi_energy_occupation.h"  // 费米能和占据数计算相关
#include "gw.h"                       // GW计算相关
#include "matrix.h"
#include "meanfield.h"   // MeanField类相关
#include "params.h"      // 参数设置相关
#include "pbc.h"         // 周期性边界条件相关
#include "profiler.h"    // 性能分析工具
#include "qpe_solver.h"  // 准粒子方程求解器
#include "read_data.h"
#include "ri.h"
#include "utils_timefreq.h"
#include "write_aims.h"


void task_scRPA(std::map<Vector3_Order<double>, ComplexMatrix> &sinvS)
{
    using LIBRPA::envs::mpi_comm_global_h;
    using LIBRPA::envs::blacs_ctxt_global_h;
    using LIBRPA::envs::ofs_myid;
    using LIBRPA::utils::lib_printf;
 

    Profiler::start("scRPA", "scRPA quasi-particle calculation");

    Vector3_Order<int> period {kv_nmp[0], kv_nmp[1], kv_nmp[2]};
    auto Rlist = construct_R_grid(period);

    vector<Vector3_Order<double>> qlist;
    for (auto q_weight: irk_weight)
    {
        qlist.push_back(q_weight.first);
    }

    

    
    const auto n_spins = meanfield.get_n_spins();
    const auto n_bands = meanfield.get_n_bands();
    const auto n_kpoints = meanfield.get_n_kpoints();
    const auto n_aos = meanfield.get_n_aos();
    const auto n_soc = meanfield.get_n_soc();


    using cplxdb = std::complex<double>;
    // 初始化
    Profiler::start("read_vxc_HKS");
    std::map<int, std::map<int, Matz>> hf_nao;  
    std::map<int, std::map<int, Matz>> vxc;  
    std::map<int, std::map<int, Matz>> hf;
    std::map<int, std::map<int, Matz>> vxc0;
    std::map<int, std::map<int, Matz>> vxc1;
    std::map<int, std::map<int, Matz>> exx0;
    std::map<int, std::map<int, Matz>> H_KS; // H_KS矩阵
    std::map<int, std::map<int, Matz>> H_KS0;
    std::map<int, std::map<int, Matz>> H_KS1;//用于混合迭代
    std::map<int, std::map<int, Matz>> Hartree_0;
    std::map<int, std::map<int, Matz>> Hartree_i;
    std::map<int, std::map<int, Matz>> Hartree_i_delta;
    std::map<int,std::map<int, std::map<int, std::vector<cplxdb>>>> Omega_total; 
    std::map<int,std::map<int,std::map<int, std::map<int, std::map<int, cplxdb>>>>> Sigma_iskwnn; 
    bool all_files_processed_successfully = true;
    const std::string final_banner(90, '-');

        // 自旋和 k 点的循环，读取初始数据
    for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin) {
        for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt) {
            std::map<std::string, Matz> arrays;
            std::string key_hf, key_vxc;

            // 使用 ostringstream 构建文件名
            std::ostringstream oss_hf, oss_vxc;
            oss_hf << "hf_exchange_spin_0" << (ispin + 1) << "_kpt_" << std::setw(6) << std::setfill('0') << (ikpt + 1) << ".csc";
            oss_vxc << "xc_matr_spin_" << (ispin + 1) << "_kpt_" << std::setw(6) << std::setfill('0') << (ikpt + 1) << ".csc";

            std::string hfFilePath = oss_hf.str();
            std::string vxcFilePath = oss_vxc.str();
            
            Matz wfc1(n_bands, n_aos * n_soc, MAJOR::COL);
            for (int ib1 = 0; ib1 < n_bands; ++ib1)
            {
                for (int isoc = 0; isoc < n_soc; isoc++)
                {
                    for (int iao = 0; iao < n_aos; iao++)
                    {
                        int ib2 = iao * n_soc + isoc;
                        wfc1(ib1, ib2) = meanfield.get_eigenvectors()[ispin][isoc][ikpt](ib1, iao);
                        meanfield.get_eigenvectors0()[ispin][isoc][ikpt](ib1, iao) = wfc1(ib1, ib2);
                    }
                }
            }

       
            hf_nao[ispin][ikpt] = Matz(n_aos, n_aos, MAJOR::COL);  
            vxc0[ispin][ikpt] = Matz(n_aos, n_aos, MAJOR::COL);     
            // 初始化 hf 和 vxc 矩阵为零矩阵
            for (int i = 0; i < n_aos; ++i) {
                for (int j = 0; j < n_aos; ++j) {
                    hf_nao[ispin][ikpt](i, j) = 0.0;
                    vxc0[ispin][ikpt](i, j) = 0.0;
                }
            }
          
            

            bool hf_file_found = false;
            bool vxc_file_found = false;

            // 读取 hf 文件
            std::ifstream hf_file(hfFilePath.c_str());
            if (hf_file.good()) {
                if (!convert_csc(hfFilePath, arrays, key_hf)) {
                    all_files_processed_successfully = false;
                    std::cerr << "Failed to process file: " << hfFilePath << std::endl;
                } 
                else 
                {
                    hf_nao[ispin][ikpt] = arrays[key_hf];
                    hf_file_found = true;
                }
            } 
            else 
            {
                std::cerr << "HF file not found: " << hfFilePath << std::endl;
            }

            // 读取 vxc 文件
            std::ifstream vxc_file(vxcFilePath.c_str());
            if (vxc_file.good()) 
            {
                if (!convert_csc(vxcFilePath, arrays, key_vxc)) 
                {
                    all_files_processed_successfully = false;
                    std::cerr << "Failed to process file: " << vxcFilePath << std::endl;
                } 
                else 
                {
                    vxc0[ispin][ikpt] = arrays[key_vxc]; 
                    vxc_file_found = true;
                }
            } 
            else 
            {
                std::cerr << "VXC file not found: " << vxcFilePath << std::endl;
            }

            // 如果两个文件都不存在，报错并跳过该 k 点
            if (!hf_file_found && !vxc_file_found) 
            {
                all_files_processed_successfully = false;
                std::cerr << "Both HF and VXC files not found for spin " << ispin + 1 << ", k-point " << ikpt + 1 << std::endl;
                continue;
            }


            // 生成 H_KS 和 H_KS0 矩阵
            hf[ispin][ikpt] = Matz(n_aos, n_aos, MAJOR::COL);     
            hf[ispin][ikpt] = conj(wfc1) * hf_nao[ispin][ikpt] * transpose(wfc1);//row hf,KS basis
            
            // 将 hf 和 vxc 在 KS 基下相加，生成最终的 vxc 矩阵
            
            vxc[ispin][ikpt] = vxc0[ispin][ikpt] + hf[ispin][ikpt];
            vxc0[ispin][ikpt] = vxc[ispin][ikpt];

            H_KS[ispin][ikpt] = Matz(n_bands, n_bands, MAJOR::COL);
            H_KS0[ispin][ikpt] = Matz(n_bands, n_bands, MAJOR::COL);
            for (int i_band = 0; i_band < n_bands; ++i_band) 
            {
                H_KS[ispin][ikpt](i_band, i_band) = meanfield.get_eigenvals()[ispin](ikpt, i_band);
                H_KS0[ispin][ikpt](i_band, i_band) = meanfield.get_eigenvals()[ispin](ikpt, i_band);
            }
         
        }
   
    }
    


    Profiler::stop("read_vxc_HKS");
    mpi_comm_global_h.barrier();
    std::flush(ofs_myid);
    
    // 在迭代开始前计算初始 HOMO, LUMO 和费米能级
    double efermi = meanfield.get_efermi();
    double homo = -1e6;
    double lumo = 1e6;

    for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin) 
    {
        for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt) 
        {
            int homo_level = -1;
            for (int ib = 0; ib < meanfield.get_n_bands(); ++ib) 
            {
                double weight = meanfield.get_weight()[ispin](ikpt, ib);
                double energy = meanfield.get_eigenvals()[ispin](ikpt, ib);

                if (weight >= 1.0 / (meanfield.get_n_spins() * meanfield.get_n_kpoints())) 
                {
                    homo_level = ib;
                }
            }

            if (homo_level != -1) 
            {
                homo = std::max(homo, meanfield.get_eigenvals()[ispin](ikpt, homo_level));
                lumo = std::min(lumo, meanfield.get_eigenvals()[ispin](ikpt, homo_level + 1));
            }
        }
    }

    // 保存初始状态数据
    homo_values.push_back(homo * HA2EV);  // 初始 HOMO 值
    lumo_values.push_back(lumo * HA2EV);  // 初始 LUMO 值
    efermi_values.push_back(efermi * HA2EV);  // 初始费米能级
    iteration_numbers.push_back(0);  // 初始迭代次数为 0

    std::cout << "Initial HOMO = " << homo * HA2EV << " eV, "
              << "LUMO = " << lumo * HA2EV << " eV, "
              << "Fermi Energy = " << efermi * HA2EV << " eV\n";
    plot_homo_lumo_vs_iterations();
    
    //计算初始体系总电子数/初始总占据数
    double total_electrons = meanfield.get_total_weight();
    printf("%5s\n","Total_electrons");
    printf("%5f\n",total_electrons);
    

    // 设置收敛条件
    double eigenvalue_tolerance = 1e-4; // 设置一个适当的小值，作为本征值收敛的判断标准
    int max_iterations =200;           // 最大迭代次数
    int iteration = 0;
    const double temperature = 0.0001;
    bool converged = false;
    int frequency = n_bands + 1; 
    std::vector<std::pair<int, int>> significant_positions;
    // 定义存储前一轮的本征值以检查收敛性
    std::vector<matrix> previous_eigenvalues(n_spins);
    mpi_comm_global_h.barrier();
    if (mpi_comm_global_h.is_root()) 
    {
        std::ofstream file("homo_lumo_vs_iterations.dat", std::ios::trunc);
        file.close();
    }
    // 初始化完毕，开始循环
    while (!converged && iteration < max_iterations) 
    {
        iteration++;

        // 更新前一轮的本征值
        if (mpi_comm_global_h.is_root()) 
        {
            for (int i_spin = 0; i_spin < n_spins; i_spin++)
            {
                previous_eigenvalues[i_spin] = meanfield.get_eigenvals()[i_spin];
            }
        }
        mpi_comm_global_h.barrier();

        // std::complex<double> corr;
        // std::vector<std::complex<double>> corr_irk(n_irk_points);
        std::complex<double> rpa_corr;
        std::vector<std::complex<double>> rpa_corr_irk_contrib(n_irk_points);

        // LIBRPA::app::get_rpa_correlation_energy_(corr, corr_irk, sinvS, driver_params.input_dir,
        //                                         Params::use_shrink_abfs);
        const int Rt_num = Rlist.size() * Params::nfreq;

        tot_atpair = generate_atom_pair_from_nat(natom, false);

        set_parallel_routing(Params::parallel_routing, tot_atpair.size(), Rt_num,
                            LIBRPA::parallel_routing);

        // Build time-frequency objects
        // // Prepare time-frequency grids
        auto tfg =
            LIBRPA::utils::generate_timefreq_grids(Params::nfreq, Params::tfgrids_type, meanfield);

        Chi0 chi0(meanfield, klist, tfg);
        chi0.gf_R_threshold = Params::gf_R_threshold;
        chi0.set_input_dir(driver_params.input_dir);
        Profiler::start("chi0_build", "Build response function chi0");
        chi0.build(Cs_data, Rlist, period, local_atpair, qlist, sinvS);
        Profiler::stop("chi0_build");
        std::flush(ofs_myid);
        mpi_comm_global_h.barrier();

        if (Params::debug)
        {  // debug, check chi0
            char fn[80];
            for (const auto &chi0q : chi0.get_chi0_q())
            {
                const int ifreq = chi0.tfg.get_freq_index(chi0q.first);
                for (const auto &q_IJchi0 : chi0q.second)
                {
                    const int iq = std::distance(
                        klist.begin(), std::find(klist.begin(), klist.end(), q_IJchi0.first));
                    for (const auto &I_Jchi0 : q_IJchi0.second)
                    {
                        const auto &I = I_Jchi0.first;
                        for (const auto &J_chi0 : I_Jchi0.second)
                        {
                            const auto &J = J_chi0.first;
                            sprintf(fn, "chi0fq_ifreq_%d_iq_%d_I_%d_J_%d_id_%d.mtx", ifreq, iq, I,
                                    J, mpi_comm_global_h.myid);
                            print_complex_matrix_mm(J_chi0.second, Params::output_dir + "/" + fn,
                                                    1e-15);
                        }
                    }
                }
            }
        }

        // NOTE: Cs is cleaned up.
        // This means that the behavior will be undefined if this function is
        // called again
        
        // LIBRPA::utils::release_free_mem();

        mpi_comm_global_h.barrier();
        Profiler::start("EcRPA", "Compute RPA correlation Energy");
        CorrEnergy corr;
        if (Params::use_scalapack_ecrpa &&
            (LIBRPA::parallel_routing == LIBRPA::ParallelRouting::ATOM_PAIR ||
            LIBRPA::parallel_routing == LIBRPA::ParallelRouting::LIBRI))
        {
            if (meanfield.get_n_kpoints() == 1)
            {
                corr = compute_RPA_correlation_blacs_2d_gamma_only(chi0, Vq);
            }
            else
            {
                corr = compute_RPA_correlation_blacs_2d(chi0, Vq);
            }
        }
        else
            corr = compute_RPA_correlation(chi0, Vq);

        rpa_corr = corr.value;

        // std::cout << qlist << "\n";
        for (const auto &irk_corr : corr.qcontrib)
        {
            auto ite = std::find(qlist.cbegin(), qlist.cend(), irk_corr.first);
            // cout << irk_corr.first << " " << irk_corr.second << "\n";
            const auto iq = std::distance(qlist.cbegin(), ite);
            // cout << iq << " " << irk_corr.second << "\n";
            rpa_corr_irk_contrib[iq] = irk_corr.second;
        }

        Profiler::stop("EcRPA");
        if (mpi_comm_global_h.is_root())
        {
            lib_printf("RPA correlation energy (Hartree)\n");
            lib_printf("| Weighted contribution from each k:\n");

            for (int i_irk = 0; i_irk < n_irk_points; i_irk++)
            {
                cout << "| " << irk_points[i_irk] << ": " << rpa_corr_irk_contrib[i_irk] << endl;
            }
            lib_printf("| Total EcRPA: %18.9f\n", rpa_corr.real());
            if (std::abs(rpa_corr.imag()) > 1.e-3)
                lib_printf("Warning: considerable imaginary part of EcRPA = %f\n", rpa_corr.imag());
        }
        

        // 读取库伦相互作用
        Profiler::start("read_vq_cut", "Load truncated Coulomb");
        if (LIBRPA::parallel_routing == LIBRPA::ParallelRouting::R_TAU)
        {
            read_Vq_full(driver_params.input_dir, "coulomb_cut_", true);
        }
        else
        {
            // NOTE: local_atpair already set in the main.cpp.
            //       It can consists of distributed atom pairs of only upper half.
            //       Setup of local_atpair may be better to extracted as some util function,
            //       instead of in the main driver.
            read_Vq_row(driver_params.input_dir, "coulomb_cut_", Params::vq_threshold, local_atpair, true);
        }
        Profiler::stop("read_vq_cut");

        
        // 读取和处理介电函数
        std::vector<double> epsmac_LF_imagfreq_re;
        if (Params::replace_w_head)
        {
            std::vector<double> omegas_dielect;
            std::vector<double> dielect_func;
            read_dielec_func(driver_params.input_dir + "dielecfunc_out", omegas_dielect, dielect_func);
    
            epsmac_LF_imagfreq_re = interpolate_dielec_func(
                    Params::option_dielect_func, omegas_dielect, dielect_func,
                    chi0.tfg.get_freq_nodes());
            if (Params::debug)
            {
                if (mpi_comm_global_h.is_root())
                {
                    lib_printf("Dielectric function parsed:\n");
                    for (int i = 0; i < chi0.tfg.get_freq_nodes().size(); i++)
                        lib_printf("%d %f %f\n", i+1, chi0.tfg.get_freq_nodes()[i], epsmac_LF_imagfreq_re[i]);
                }
                mpi_comm_global_h.barrier();
            }
        }

        //构建V^{Hartree}矩阵
        Profiler::start("qsgw_hartree", "Build Hartree potential");
        auto Hartree = LIBRPA::Hartree(meanfield, kfrac_list, period);
        {
            Profiler::start("ft_vq_cut", "Fourier transform truncated Coulomb");
            const auto VR1 = FT_Vq(Vq_cut, meanfield.get_n_kpoints(), Rlist, true);
            Profiler::stop("ft_vq_cut"); 
            Profiler::start("qsgw_hartree_real_work");
            Hartree.build(Cs_data, Rlist, VR1); 
            // // 新增调试输出
            // for (int isp = 0; isp < meanfield.get_n_spins(); ++isp) {
            //     for (int is1 = 0; is1 < meanfield.get_n_soc(); ++is1) {
            //         for (int is2 = 0; is2 < meanfield.get_n_soc(); ++is2) {
            //             // 遍历R空间
            //             for (const auto& R_entry : Hartree.hartree[isp][is1][is2]) {
            //                 Vector3_Order<int> R = R_entry.first;
            //                 // 只检查R=0的情况
            //                 // if (R.x == 0 && R.y == 0 && R.z == 0) {
            //                     for (const auto& P_entry : R_entry.second) {
            //                         atom_t P = P_entry.first;
            //                         for (const auto& Q_entry : P_entry.second) {
            //                             atom_t Q = Q_entry.first;
            //                             const Matd& hartree_mat = Q_entry.second;
                                        
            //                             // 输出矩阵基本信息
            //                             std::cout << "Hartree[" << isp << "][" << is1 << "][" << is2 << "]" 
            //                                     << "[R=(" << R.x << "," << R.y << "," << R.z << ")]"
            //                                     << "[P=" << P << "][Q=" << Q << "] Matrix:" << std::endl;
                                        
            //                             // 输出矩阵前3x3部分
            //                             for (int i = 0; i < 20 && i < hartree_mat.nr(); ++i) {
            //                                 for (int j = 0; j < 20 && j < hartree_mat.nc(); ++j) {
            //                                     std::cout << std::setw(12) << hartree_mat(i,j) << " ";
            //                                 }
            //                                 std::cout << std::endl;
            //                             }
            //                         }
            //                     }
            //                 // }
            //             }
            //         }
            //     }
            // }
            Hartree.build_KS_kgrid0();//rotate  
            Profiler::stop("qsgw_hartree_real_work");
        
        }

        Profiler::stop("qsgw_hartree");
        std::flush(ofs_myid);
        mpi_comm_global_h.barrier();

        std::complex<double> hartree_energy;
        hartree_energy = 0.0;
        if (mpi_comm_global_h.is_root())
        {
            for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin) {
                for(int isoc1 = 0; isoc1 < meanfield.get_n_soc(); ++isoc1)
                {
                    for(int isoc2 = 0; isoc2 < meanfield.get_n_soc(); ++isoc2)
                    {
                        for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt) {    
                            Hartree_i[ispin][ikpt] = Matz(n_bands, n_bands, MAJOR::COL);
                            Hartree_0[ispin][ikpt] = Matz(n_bands, n_bands, MAJOR::COL);
                            Hartree_i_delta[ispin][ikpt] = Matz(n_bands, n_bands, MAJOR::COL);
                            for (int i = 0; i < n_bands; ++i) {                                
                                const auto &hartree0_k_ks_value = Hartree.EHartree[ispin][ikpt][i];
                                printf("%16.6f ", hartree0_k_ks_value ); 
                                for (int j = 0; j < n_bands;++j) {

                                    const auto &hartree_k_ks_value = Hartree.Hartree_is_ik_KS[ispin][ikpt](i, j);
                                    
                                    Hartree_i[ispin][ikpt](i, j) = hartree_k_ks_value;
                                    
                                    if(iteration==1){
                                        Hartree_0[ispin][ikpt](i, j) =  Hartree.Hartree_is_ik_KS[ispin][ikpt](i,j);
                                    }
                                    else{
                                        Hartree_i_delta[ispin][ikpt](i, j) = Hartree_i[ispin][ikpt](i, j) - Hartree_0[ispin][ikpt](i,j);
                                        
                                    }
                                    hartree_energy += - 0.5 * Hartree.Hartree_is_ik_nao[ispin][ikpt](i, j) * meanfield.get_dmat_cplx(ispin, isoc1, isoc2, ikpt)(j, i);
                                }
                                printf("\n");
                            }
                            printf("\n");
                          
                        }
                    }
                }
            }
            lib_printf(" Hartree_energy (Hartree)\n");
            lib_printf("| Hartree_energy: %18.9f\n", hartree_energy.real());
        }
        // 构建V^{exx}矩阵,得到Hexx_nband_nband: exx.exx_is_ik_KS

        Profiler::start("scRPA_exx", "Build exchange self-energy");
        auto exx = LIBRPA::Exx(meanfield, kfrac_list, period);
        {
            Profiler::start("ft_vq_cut", "Fourier transform truncated Coulomb");
            const auto VR = FT_Vq(Vq_cut, meanfield.get_n_kpoints(), Rlist, true);
            Profiler::stop("ft_vq_cut");

            Profiler::start("g0w0_exx_real_work");
            if (Params::use_soc)
                exx.build<std::complex<double>>(Cs_data, Rlist, VR);
            else
                exx.build<double>(Cs_data, Rlist, VR);
            exx.build_KS_kgrid0();  // rotate
            Profiler::stop("g0w0_exx_real_work");
        
        }
        Profiler::stop("scRPA_exx");
        std::flush(ofs_myid);

        mpi_comm_global_h.barrier();

        if (mpi_comm_global_h.is_root())
        {
            std::complex<double> exx_energy;
            exx_energy = 0.0;
            // ComplexMatrix exx_energy_cplx(meanfield.get_n_aos(), meanfield.get_n_aos(), MAJOR::COL);
            // exx_energy
            for (int ispin = 0; ispin < meanfield.get_n_spins(); ispin++)
            {
                for(int isoc1 = 0; isoc1 < n_soc; isoc1++)
                {
                    for(int isoc2 = 0; isoc2 < n_soc; isoc2++)
                    {
                        for(int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ikpt++)
                        {
                            for(int ib1 = 0; ib1 < meanfield.get_n_bands(); ib1++)
                            {
                                for(int ib2 = 0; ib2 < meanfield.get_n_bands(); ib2++)
                                {
                                    exx_energy += -0.5 * exx.exx_is_ik_nao[ispin][ikpt](ib1, ib2) * meanfield.get_dmat_cplx(ispin, isoc1, isoc2, ikpt)(ib2, ib1);
                                }
                            }
                        }
                    }
                }
                
            }
            lib_printf("Exchange self-energy (Hartree)\n");
            lib_printf("| Exx_energy: %18.9f\n", exx_energy.real());
            lib_printf("| XC_energy: %18.9f\n", rpa_corr.real()+exx_energy.real());
            lib_printf("| Total_energy: %18.9f\n", hartree_energy.real()+rpa_corr.real()+exx_energy.real());

        }
        
        
      
        
        
        
        // Build screened interaction
        Profiler::start("scRPA_wc", "Build screened interaction");
        vector<std::complex<double>> epsmac_LF_imagfreq(epsmac_LF_imagfreq_re.cbegin(),
                                                        epsmac_LF_imagfreq_re.cend());
        map<double,
            atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old>
            Wc_freq_q;
        if (Params::use_scalapack_gw_wc)
        {
            Wc_freq_q = compute_Wc_freq_q_blacs(chi0, Vq, Vq_cut, epsmac_LF_imagfreq);
        }
        else
        {
            Wc_freq_q = compute_Wc_freq_q(chi0, Vq, Vq_cut, epsmac_LF_imagfreq);
        }
        Profiler::stop("scRPA_wc");


        LIBRPA::G0W0 s_g0w0(meanfield, kfrac_list, chi0.tfg, period);
        Profiler::start("g0w0_sigc_IJ", "Build correlation self-energy");
        if (Params::use_soc)
            s_g0w0.build_spacetime<std::complex<double>>(Cs_data, Wc_freq_q, Rlist, qlist, sinvS);
        else
            s_g0w0.build_spacetime<double>(Cs_data, Wc_freq_q, Rlist, qlist, sinvS);
        Profiler::stop("g0w0_sigc_IJ");
        std::flush(ofs_myid);
        Profiler::start("g0w0_sigc_rotate_KS", "Rotate self-energy, IJ -> ij -> KS");
        s_g0w0.build_sigc_matrix_KS_kgrid0();//rotate
        Profiler::stop("g0w0_sigc_rotate_KS");

        // 构建哈密顿量矩阵并对角化，旋转基底，并存储本征值，本征矢量
        // 第一步：构建关联势矩阵
        std::map<int, std::map<int, Matz>> Vc_all;

        // 构建虚频点列表
        std::vector<cplxdb> imagfreqs;
        for (const auto &freq : chi0.tfg.get_freq_nodes()) 
        {
            imagfreqs.push_back(cplxdb{0.0, freq});
        }

        std::map<int, std::map<int, std::map<int, double>>> e_qp_all;
        std::map<int, std::map<int, std::map<int, cplxdb>>> sigc_all;
        
        if (all_files_processed_successfully)
        {
            Profiler::start("scRPA_solve_qpe", "Solve quasi-particle equation");

            if (mpi_comm_global_h.is_root()) 
            {
                std::cout << "Solving quasi-particle equation\n";
            }

            if (mpi_comm_global_h.is_root()) {
                // 遍历自旋、k点和能带状态
                for (int i_spin = 0; i_spin < n_spins; i_spin++) {
                    for (int i_kpoint = 0; i_kpoint < n_kpoints; i_kpoint++) {
                        std::vector<std::vector<std::vector<cplxdb>>> sigcmat(
                            n_bands, std::vector<std::vector<cplxdb>>(n_bands, std::vector<cplxdb>(n_bands + 1))
                        );                  
                        const auto &sigc_sk = s_g0w0.sigc_is_ik_f_KS[i_spin][i_kpoint];

                        for (int i_state_row = 0; i_state_row < n_bands; i_state_row++) {   
                            for (int i_state_col = 0; i_state_col < meanfield.get_n_bands(); i_state_col++) {
                                std::vector<cplxdb> sigc_mn;
                           
                                for (const auto &freq : chi0.tfg.get_freq_nodes()) {
                                    sigc_mn.push_back(sigc_sk.at(freq)(i_state_row, i_state_col));
                            
                                }    
                                
                                LIBRPA::AnalyContPade pade(Params::n_params_anacon, imagfreqs, sigc_mn);

                                auto energy0 = meanfield.get_eigenvals()[i_spin](i_kpoint, i_state_row); 
                                efermi = meanfield.get_efermi();
                                // 计算得到的值
                                auto result = pade.get(energy0 - efermi);
                                auto result1 = pade.get(0.0);
                                // 存储值到 sigcmat
                                sigcmat[i_state_row][i_state_col][i_state_row] = result;
                                sigcmat[i_state_row][i_state_col][n_bands] = result1;
                                

                            }
                        }
 
                        const auto& freq = chi0.tfg.get_freq_nodes();
                        // printf("%zu\n ",freq.size());
                        const auto& f_weight= chi0.tfg.get_freq_weights();

             
                        auto G0_matrix= build_G0(meanfield,freq,i_spin,i_kpoint,n_bands);
                    
                        Vc_all[i_spin][i_kpoint] = calculate_scRPA_exchange_correlation(meanfield,freq,f_weight,sigc_sk,sigcmat,G0_matrix,i_spin,i_kpoint,n_bands,temperature);
                        

                        // printf("%77s\n", final_banner.c_str());
                        // printf("Vc_all.real:\n");
                        // for (int i = 0; i < meanfield.get_n_bands(); i++) {
                        //     for (int j = 0; j < meanfield.get_n_bands(); j++) {
                        //         const auto &Vc_all_0 = Vc_all[i_spin][i_kpoint](i, j) ;
                        //         printf("%20.16f ", Vc_all_0.real()); 
                        //     }
                        //     printf("\n"); // 换行
                        // }
                        // printf("Vc_all.imag:\n");
                        // for (int i = 0; i < meanfield.get_n_bands(); i++) {
                        //     for (int j = 0; j < meanfield.get_n_bands(); j++) {
                        //         const auto &Vc_all_0 = Vc_all[i_spin][i_kpoint](i, j) ;
                        //         printf("%20.16f ", Vc_all_0.imag()); 
                        //     }
                        //     printf("\n"); // 换行
                        // }
                    }

          
                }
                Profiler::stop("scRPA_solve_qpe");
               
                
                auto H0_GW_all = construct_H0_GW(meanfield, H_KS0, vxc0, exx.exx_is_ik_KS, Vc_all, n_spins, n_kpoints, n_bands);
                
            
                //混合
                // if(iteration > 1){
                //     for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin) {
                //         for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt) {
                //             H0_GW_all[ispin][ikpt] = 0.2 * H0_GW_all[ispin][ikpt] + 0.8 * H_KS[ispin][ikpt];
                //         }
                //     }
                // }
                // 第三步：对 Hamiltonian 进行对角化并存储本征值
                diagonalize_and_store(meanfield, H0_GW_all, n_spins, n_kpoints, n_bands);
  

                // 计算全局费米能和占据数
                const auto &Efermi0 = meanfield.get_efermi() ;
                printf("%5s\n","efermi0");
                printf("%5f\n",Efermi0);
                // 计算费米能级
                
                double efermi = calculate_fermi_energy(meanfield, temperature, total_electrons);
                printf("%5s\n","efermi0");
                printf("%5f\n",efermi);

                update_fermi_energy_and_occupations(meanfield, temperature, efermi);
                efermi_values.push_back(efermi * HA2EV);  
                // 比较本轮和前一轮的本征值判断是否收敛
                converged = true;
                for (int ispin = 0; ispin < n_spins; ++ispin) {
                    const auto &current_eigenvals = meanfield.get_eigenvals()[ispin];
                    const auto max_diff = (current_eigenvals - previous_eigenvalues[ispin]).absmax();
                    if (max_diff > eigenvalue_tolerance) {
                        converged = false;
                        break;
                    }
                }
                std::cout << "Converged after " << iteration << " iterations.\n";
                // const std::string final_banner(90, '-');
                lib_printf("Final Quasi-Particle Energy after scRPA Iterations [unit: eV]\n\n");
                const auto &Efermi = meanfield.get_efermi() ;
                printf("%5s\n","efermi");
                printf("%5f\n",Efermi);
                for (int i_spin = 0; i_spin < meanfield.get_n_spins(); i_spin++)
                {
                    for (int i_kpoint = 0; i_kpoint < meanfield.get_n_kpoints(); i_kpoint++)
                    {
                        const auto &k = kfrac_list[i_kpoint];
                        printf("spin %2d, k-point %4d: (%.5f, %.5f, %.5f) \n",
                                i_spin + 1, i_kpoint + 1, k.x, k.y, k.z);
                        printf("%77s\n", final_banner.c_str());
                        printf("%5s %16s %16s %16s \n", "State", "e_mf", "v_xc", "v_exx");
                        printf("%77s\n", final_banner.c_str());
                        for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
                        {
                            const auto &eks_state = meanfield.get_eigenvals()[i_spin](i_kpoint, i_state) * HA2EV;
                            const auto &exx_state = exx.Eexx[i_spin][i_kpoint][i_state] * HA2EV;
                            // const auto &exx_state2 = exx.exx_is_ik_KS[i_spin][i_kpoint](i_state, i_state)* HA2EV;
                            const auto &vxc_state = vxc0[i_spin][i_kpoint](i_state, i_state) * HA2EV;
                            // const auto &resigc = sigc_all[i_spin][i_kpoint][i_state].real() * HA2EV;
                            // const auto &imsigc = sigc_all[i_spin][i_kpoint][i_state].imag() * HA2EV;
                            // const auto &eqp = e_qp_all[i_spin][i_kpoint][i_state] * HA2EV;
                            printf("%5d %20.15f %16.5f %16.5f  \n",
                                i_state + 1, eks_state, vxc_state.real(), exx_state);
                        }
                        printf("\n");
                    }
                }

                



                // 计算 HOMO 和 LUMO
                homo = -1e6;  // 
                lumo = 1e6;   // 
                for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin) {
                    for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt) {
                        int homo_level = -1;
                        for (int ib = 0; ib < meanfield.get_n_bands(); ++ib) {
                            double weight = meanfield.get_weight()[ispin](ikpt, ib);
                            double energy = meanfield.get_eigenvals()[ispin](ikpt, ib);

                            // 
                            if (weight >= 1.0 / (meanfield.get_n_spins() * meanfield.get_n_kpoints())) {
                                homo_level = ib;
                            }
                        }

                        // 
                        if (homo_level != -1) {
                            // 
                            homo = std::max(homo, meanfield.get_eigenvals()[ispin](ikpt, homo_level));
                            // 
                            lumo = std::min(lumo, meanfield.get_eigenvals()[ispin](ikpt, homo_level + 1));
                        }
                    }
                }

                // 
                homo_values.push_back(homo * HA2EV);  // 
                lumo_values.push_back(lumo * HA2EV);  // 
                iteration_numbers.push_back(iteration);

                // 输出当前 HOMO 和 LUMO 值
                std::cout << "Iteration " << iteration
                << ": HOMO = " << homo * HA2EV << " eV, "
                << "LUMO = " << lumo * HA2EV << " eV, "
                << "Efermi = " << efermi * HA2EV << " eV\n";
                
                
                // 保存 HOMO、LUMO 和费米能级数据
                {
                    std::ofstream file("homo_lumo_vs_iterations.dat", std::ios::app); // 使用 std::ios::app 以追加模式打开文件
                    file << iteration << " "
                        << homo_values[iteration] << " "
                        << lumo_values[iteration] << " "
                        << efermi_values[iteration] << std::endl;
                }
                // 比较本轮和前一轮的本征值判断是否收敛
                converged = true;
                for (int ispin = 0; ispin < n_spins; ++ispin) {
                    const auto &current_eigenvals = meanfield.get_eigenvals()[ispin];
                    const auto max_diff = (current_eigenvals - previous_eigenvalues[ispin]).absmax();
                    if (max_diff > eigenvalue_tolerance) {
                        converged = false;
                        break;
                    }
                }
            
                std::cout << "Converged after " << iteration << " iterations.\n";
            }   
        }
        mpi_comm_global_h.barrier(); 
        
        mpi_comm_global_h.broadcast(converged, 0);
        mpi_comm_global_h.barrier(); 
        meanfield.broadcast(mpi_comm_global_h, 0);
        mpi_comm_global_h.barrier();
        
        if (converged || iteration == max_iterations) {
            if (mpi_comm_global_h.is_root()) {
                std::cout << " iterations: " << iteration;
            }
            break; 
        }

        mpi_comm_global_h.barrier();
        

        
        
    }

    Profiler::stop("scRPA");
}

