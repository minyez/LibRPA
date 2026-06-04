#include "read_data.h"
#include "reader_lri.h"
#include "reader_coulomb.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../include/librpa.hpp"

#include "driver.h"
#include "../src/mpi/global_mpi.h"
#include "../src/math/matrix.h"
#include "../src/utils/constants.h"
#include "../src/api/instance_manager.h"
#include "../src/io/fs.h"
#include "../src/io/global_io.h"
#include "../src/io/stl_io_helper.h"
#include "../src/utils/error.h"
#include "../src/utils/profiler.h"
#include "../src/utils/utils_mem.h"


using std::ifstream;
using std::string;

struct QpStateRange
{
    int low;
    int high;
};

static QpStateRange automatic_qp_state_range(const librpa_int::MeanField &mf)
{
    const int n_states_mf = mf.get_n_states();
    if (n_states_mf <= 0)
    {
        throw std::runtime_error("Cannot resolve QP state range from an empty meanfield object");
    }

    const double gap = mf.get_band_gap();
    const double efermi = mf.get_efermi();
    const double e_low = efermi - 0.5 * gap - 0.5;
    const double e_high = efermi + 0.5 * gap + 0.5;
    const int i_state_low = mf.get_max_state_below_energy(e_low) + 1;
    const int i_state_high = mf.get_min_state_above_energy(e_high);

    if (i_state_high <= i_state_low)
    {
        return {0, n_states_mf};
    }
    return {i_state_low, i_state_high};
}

static void normalize_qp_state_range_from_kgrid_mf(const librpa_int::MeanField &mf)
{
    auto &params = driver::driver_params;
    const int n_states_mf = mf.get_n_states();
    if (n_states_mf <= 0)
    {
        throw std::runtime_error("Cannot resolve QP state range from an empty meanfield object");
    }

    const bool automatic_low = params.i_state_low < 0;
    const bool automatic_high = params.i_state_high < 0;
    const bool use_automatic_default_high =
        params.i_state_high == driver::DriverParams::default_i_state_high &&
        (automatic_low || automatic_high);
    const QpStateRange automatic_range =
        (automatic_low || automatic_high || use_automatic_default_high)
            ? automatic_qp_state_range(mf)
            : QpStateRange{0, n_states_mf};

    if (automatic_low)
    {
        params.i_state_low = automatic_range.low;
    }
    else if (params.i_state_low > n_states_mf)
    {
        std::stringstream ss;
        ss << "i_state_low = " << params.i_state_low
           << " exceeds the maximum number of states (" << n_states_mf << ")";
        throw std::runtime_error(ss.str());
    }

    if (automatic_high || use_automatic_default_high)
    {
        params.i_state_high = automatic_range.high;
    }
    else if (params.i_state_high > n_states_mf)
    {
        params.i_state_high = n_states_mf;
    }

    if (params.i_state_high <= params.i_state_low)
    {
        std::stringstream ss;
        ss << "Empty QP state range: i_state_low = " << params.i_state_low
           << ", i_state_high = " << params.i_state_high
           << ". The high state index is exclusive.";
        throw std::runtime_error(ss.str());
    }
}

void read_scf_occ_eigenvalues(const string &file_path)
{
    using std::to_string;
    using driver::n_spins;
    using driver::n_kpoints;
    using driver::n_states;
    using driver::n_basis_wfc;
    using driver::n_basis_ao;
    using driver::n_spinor;
    using driver::iks_eigvec_this;
    using librpa_int::global::myid_global;
    using librpa_int::global::size_global;

    // cout << "Begin to read aims-band_out" << endl;
    ifstream infile;
    infile.open(file_path);
    if (!infile.good())
    {
        throw std::logic_error("Failed to open " + file_path);
    }

    string ks, ss, a, ws, es, d;
    double efermi;
    infile >> n_kpoints;
    infile >> n_spins;
    infile >> n_states;
    infile >> n_basis_wfc;
    infile >> efermi;

    const bool use_spinor_wfc = driver::driver_params.use_spinor_wfc;

    if (use_spinor_wfc)
    {
        assert(n_spins == 1);
        assert(n_basis_wfc % 2 == 0 && "Error: nbasis is not even when SOC!");
        n_spinor = 2;
        n_basis_ao = n_basis_wfc / 2;
    }
    else
    {
        n_spinor = 1;
        n_basis_ao = n_basis_wfc;
    }

    driver::h.set_scf_dimension(n_spins, n_kpoints, n_states, n_basis_ao, n_spinor);
    auto pds = librpa_int::api::get_dataset_instance(driver::h);
    const auto &kbctxt = pds->scfk_blacs_ctxt;

    iks_eigvec_this.clear();
    if (driver::get_bool(driver::opts.use_kpara_scf_eigvec))
    {
        // reusing the internal distribution
        if (kbctxt.comm_blacs_h.myid == 0)
            iks_eigvec_this = kbctxt.kpoints_local();
    }
    else
    {
        for (int ik = 0; ik < driver::n_kpoints; ik++)
            iks_eigvec_this.emplace_back(ik);
    }

    driver::n_ibz_kpoints = n_kpoints;

    // Load the file data
    auto eskb = new double [n_spins * n_kpoints * n_states];
    auto wskb = new double [n_spins * n_kpoints * n_states];

    const int n_kb = n_kpoints * n_states;

    int iline = 6;

    // cout<<"|eskb: "<<endl;
    for (int ik = 0; ik != n_kpoints; ik++)
    {
        for (int is = 0; is != n_spins; is++)
        {
            infile >> ks >> ss;
            if (!infile.good())
            {
                throw std::logic_error("Error in reading k- and spin- index: line " +
                                       to_string(iline) + ", file: " + file_path);
            }
            iline++;
            // cout<<ik<<is<<endl;
            int k_index = stoi(ks) - 1;
            // int s_index = stoi(ss) - 1;
            for (int i = 0; i != n_states; i++)
            {
                // iband weight energy(Ha) energy(eV)
                infile >> a >> ws >> es >> d;
                if (!infile.good())
                {
                    throw std::logic_error("Error in reading band energy and occupation: line " +
                                           to_string(iline) + ", file: " + file_path);
                }
                iline++;
                wskb[is * n_kb + k_index * n_states + i] = stod(ws); // different with abacus!
                eskb[is * n_kb + k_index * n_states + i] = stod(es);
                //cout<<" i_band: "<<i<<"    eskb: "<<eskb[is](k_index, i)<<endl;
            }
        }
    }
    // for (int is = 0; is != n_spins; is++)
    //     print_matrix("eskb_mat",eskb[is]);

    driver::h.set_wg_ekb_efermi(n_spins, n_kpoints, n_states, wskb, eskb, efermi);

    // free buffer
    delete[] eskb;
    delete[] wskb;

    normalize_qp_state_range_from_kgrid_mf(pds->mf);
}

int read_vxc(const string &file_path, std::vector<matrix> &vxc)
{
    ifstream infile;
    infile.open(file_path);
    double ha, ev;
    int n_spins, n_kpoints, n_states;
    // int retcode;

    // dimension information
    infile >> n_kpoints;
    infile >> n_spins;
    infile >> n_states;
    if (!infile.good())
    {
        return 1;
    }

    vxc.clear();
    vxc.resize(n_spins);
    for (int is = 0; is != n_spins; is++)
    {
        vxc[is].create(n_kpoints, n_states);
    }

    for (int ik = 0; ik != n_kpoints; ik++)
    {
        for (int is = 0; is != n_spins; is++)
        {
            for (int i = 0; i != n_states; i++)
            {
                infile >> ha >> ev;
                if (!infile.good())
                {
                    return 2;
                }
                vxc[is](ik, i) = ha;
            }
        }
    }
    return 0;
}

static int handle_KS_file(const string &file_path)
{
    using driver::iks_eigvec_this;

    int ret = 0;
    // cout<<file_path<<endl;
    ifstream infile;
    // cout << "Reading eigenvector from file " << file_path << endl;
    infile.open(file_path);
    if (!infile.good()) return 1;

    string rvalue, ivalue, kstr;

    const auto nspin = driver::n_spins;
    const auto nsoc = driver::n_spinor;
    const auto nband = driver::n_states;
    const auto nao = driver::n_basis_ao;
    const auto nbao = nband * nao;
    const auto nbs = nband * nsoc;
    const auto n = nsoc * nbao;
    const bool use_spinor_wfc = driver::driver_params.use_spinor_wfc;

    std::vector<double> re(nspin * n);
    std::vector<double> im(nspin * n);

    while (infile.peek() != EOF)
    {
        infile >> kstr;
        int ik = stoi(kstr) - 1;
        // cout<<"     ik: "<<ik<<endl;
        if (infile.peek() == EOF) break;
        // for aims !!!
        bool skip_this_ik = false;
        if (driver::get_bool(driver::opts.use_kpara_scf_eigvec))
        {
            const auto it = std::find(iks_eigvec_this.cbegin(), iks_eigvec_this.cend(), ik);
            // this k does not belong to this process
            skip_this_ik = (it == iks_eigvec_this.cend());
        }
        for (int iw = 0; iw != nao; iw++)
        {
            for (int isoc = 0; isoc != nsoc; isoc++)
            {
                for (int ib = 0; ib != nband; ib++)
                {
                    for (int is = 0; is != nspin; is++)
                    {
                        // cout<<iw<<ib<<is<<ik;
                        infile >> rvalue >> ivalue;
                        if (infile.bad())
                        {
                            ret = 1;
                            break;
                        }
                        // cout<<rvalue<<ivalue<<endl;
                        if (skip_this_ik) continue;
                        if (use_spinor_wfc)
                        {
                            // re[is * n + isoc * nbao + iw * nband + ib] = stod(rvalue);
                            // im[is * n + isoc * nbao + iw * nband + ib] = stod(ivalue);
                            // re[is * n + iw * nbs + isoc * nband + ib] = stod(rvalue);
                            // im[is * n + iw * nbs + isoc * nband + ib] = stod(ivalue);
                            re[is * n + isoc * nbao + ib * nao + iw] = stod(rvalue);
                            im[is * n + isoc * nbao + ib * nao + iw] = stod(ivalue);
                        }
                        else
                        {
                            re[is * n + isoc * nbao + ib * nao + iw] = stod(rvalue);
                            im[is * n + isoc * nbao + ib * nao + iw] = stod(ivalue);
                        }
                    }
                }
            }
        }
        if (skip_this_ik) continue;
        for (int is = 0; is != nspin; is++)
        {
            if (use_spinor_wfc)
            {
                assert(is == 0);
                driver::h.set_wfc_spinor(ik, driver::n_states, driver::n_basis_ao,
                                         re.data(), im.data(),
                                         re.data() + nbao, im.data() + nbao);
            }
            else
                driver::h.set_wfc(is, ik, driver::n_states, driver::n_basis_ao, re.data() + is * n, im.data() + is * n);
        }
        // if (ik==0) librpa_int::global::ofs_myid << re << std::endl;
        // for abacus
        // for (int ib = 0; ib != NBANDS; ib++)
        //     for (int iw = 0; iw != NLOCAL; iw++)
        //         for (int is = 0; is != NSPIN; is++)
        //         {
        //             // cout<<iw<<ib<<is<<ik;
        //             infile >> rvalue >> ivalue;
        //             // cout<<rvalue<<ivalue<<endl;
        //             wfc_k.at(stoi(ik) - 1)(ib, iw) = complex<double>(stod(rvalue), stod(ivalue));
        //         }
    }
    return ret;
}

int read_eigenvector(const string &dir_path)
{
    // return code
    int ret = 0;
    int files_read = 0;

    struct dirent *ptr;
    DIR *dir;
    dir = opendir(dir_path.c_str());
    std::vector<string> files;
    while ((ptr = readdir(dir)) != NULL)
    {
        string fm(ptr->d_name);
        // cout << fm << " find:" << fm.find("KS_eigenvector") << "\n";
        if (fm.find("KS_eigenvector") == 0)
        {
            ret = handle_KS_file(dir_path + fm);
            if (ret != 0)
            {
                break;
            }
            files_read++;
        }
    }
    closedir(dir);
    dir = NULL;

    if (files_read == 0)
    {
        ret = -1;
    }

    // auto tmp_wfc=mf.get_eigenvectors();
    //  for(int is=0;is!=mf.get_n_spins();is++)
    //      print_complex_matrix("wfc ",tmp_wfc.at(is).at(0));
    //  cout << "Finish read KS_eignvector! " << endl;
    return ret;
}

void read_ri(const string &dir_path, librpa::ParallelRouting &routing)
{
    using driver::n_atoms;
    using driver::n_kpoints;
    using driver::local_atpair;
    using librpa_int::generate_atom_pair_from_nat;
    using librpa_int::decide_auto_routing;
    using librpa_int::dispatch_upper_triangular_tasks;
    using namespace librpa_int::global;

    mpi_comm_global_h.barrier();
    lib_printf_root("Loading RI file from directory: %s\n", dir_path.c_str());

    const auto tot_atpair = generate_atom_pair_from_nat(n_atoms, false);
    const auto tot_atpair_ordered = generate_atom_pair_from_nat(n_atoms, true);

    if (routing == LIBRPA_ROUTING_AUTO)
    {
        routing = decide_auto_routing(n_atoms, driver::opts.nfreq * n_kpoints);
    }

    auto pds = librpa_int::api::get_dataset_instance(driver::h.get_c_handler());
    const auto &Cs_data = pds->cs_data;
    const auto &blacs_h = pds->blacs_h;

    local_atpair.clear();

    // HACK: local_atpair should be set in the same mechanism as inside the dataset object,
    //       which is implemented in initialize_ds_atpairs_local in dataset_helper.cpp.
    //       It consists of distributed atom pairs of only upper half, since repsonse function matrix is Hermitian.
    if(routing == LIBRPA_ROUTING_ATOMPAIR)
    {
        lib_printf_root("Triangular dispatching of atom pairs\n");
        auto tri_local_atpair = librpa_int::dispatch_upper_triangular_tasks(
            n_atoms, blacs_h.myid, blacs_h.nprows, blacs_h.npcols,
            blacs_h.myprow, blacs_h.mypcol);
        for (const auto &p: tri_local_atpair)
            local_atpair.push_back(p);
        profiler.start("driver_read_Cs");
        read_Cs(dir_path, driver::driver_params.cs_threshold, local_atpair,
                driver::driver_params.prefix_lri_coeff,
                driver::driver_params.version_lri_reader);
        profiler.stop("driver_read_Cs");

        mpi_comm_global_h.barrier();
        profiler.start("driver_read_Vq");
        read_Vq_row(dir_path, driver::driver_params.prefix_coul_full,
                    driver::opts.vq_threshold, local_atpair, false,
                    driver::driver_params.version_coul_reader);
        profiler.stop("driver_read_Vq");
    }
    else if(routing == LIBRPA_ROUTING_LIBRI)
    {
        lib_printf_root("Evenly distributed Cs and V for LibRI\n");
        profiler.start("driver_read_Cs");
        read_Cs_evenly_distribute(dir_path, driver::driver_params.cs_threshold,
                                  mpi_comm_global_h.myid, mpi_comm_global_h.nprocs,
                                  driver::driver_params.prefix_lri_coeff,
                                  driver::driver_params.version_lri_reader);
        profiler.stop("driver_read_Cs");
        // Vq distributed using the same strategy
        // There should be no duplicate for V

        mpi_comm_global_h.barrier();
        profiler.start("driver_read_Vq");
        auto trangular_loc_atpair = librpa_int::dispatch_upper_triangular_tasks(
            n_atoms, blacs_h.myid, blacs_h.nprows, blacs_h.npcols,
            blacs_h.myprow, blacs_h.mypcol);
        for(auto &iap:trangular_loc_atpair)
            local_atpair.push_back(iap);
        read_Vq_row(dir_path, driver::driver_params.prefix_coul_full,
                    driver::opts.vq_threshold, local_atpair, false,
                    driver::driver_params.version_coul_reader);
        profiler.stop("driver_read_Vq");
    }
    else
    {
        lib_printf_root("Complete copy of Cs and V on each process\n");
        local_atpair = generate_atom_pair_from_nat(n_atoms, false);
        profiler.start("driver_read_Cs");
        read_Cs(dir_path, driver::driver_params.cs_threshold, local_atpair,
                driver::driver_params.prefix_lri_coeff,
                driver::driver_params.version_lri_reader);
        profiler.stop("driver_read_Cs");

        mpi_comm_global_h.barrier();
        profiler.start("driver_read_Vq");
        read_Vq_full(dir_path, driver::driver_params.prefix_coul_full, false,
                     driver::driver_params.version_coul_reader);
        profiler.stop("driver_read_Vq");
    }

    mpi_comm_global_h.barrier();
    lib_printf_coll("| Process %5d: coulomb_mat read. Wall/CPU time [min]: %12.4f %12.4f\n",
                    mpi_comm_global_h.myid,
                    profiler.get_wall_time_last("driver_read_Vq") / 60.0,
                    profiler.get_cpu_time_last("driver_read_Vq") / 60.0);
    mpi_comm_global_h.barrier();
    lib_printf_coll("| Process %5d: Cs with %14zu non-zero keys from local atpair size %7zu. "
                    "Data memory: %10.2f MB. Wall/CPU time [min]: %12.4f %12.4f\n",
                    mpi_comm_global_h.myid, Cs_data.n_keys(), local_atpair.size(),
                    Cs_data.n_data_bytes() * 8.0e-6,
                    profiler.get_wall_time_last("driver_read_Cs") / 60.0,
                    profiler.get_cpu_time_last("driver_read_Cs") / 60.0);
    mpi_comm_global_h.barrier();
}

void read_velocity(const string &file_path, MeanField &mf)
{
    using librpa_int::global::mpi_comm_global_h;
    using librpa_int::ANG2BOHR;
    using librpa_int::HA2EV;

    ifstream infile;
    infile.open(file_path);
    string alpha, kk, ss, single_re, single_im;
    int n_kpoints, n_spins, n_bands, n_aos;
    infile >> n_kpoints;
    infile >> n_spins;
    infile >> n_bands;
    infile >> n_aos;

    auto &velocity = mf.get_velocity();
    for (int is = 0; is != n_spins; is++)
    {
        for (int ik = 0; ik != n_kpoints; ik++)
        {
            for (int ia = 0; ia != 3; ia++)
            {
                infile >> alpha >> kk >> ss;
                int k_index = stoi(kk) - 1;
                int a_index = stoi(alpha) - 1;
                int s_index = stoi(ss) - 1;
                assert(k_index == ik);
                assert(a_index == ia);
                assert(s_index == is);
                for (int i = 0; i != n_bands; i++)
                {
                    for (int j = 0; j != n_bands; j++)
                    {
                        infile >> single_re >> single_im;
                        velocity.at(is).at(ik).at(ia)(i, j) =
                            ANG2BOHR * std::complex<double>(stod(single_re), stod(single_im)) / HA2EV;
                    }
                }
            }
        }
    }
    if (mpi_comm_global_h.is_root())
        std::cout << "* Success: read velocity from pyatb_librpa_df(ABACUS)." << std::endl;
}

void read_velocity_aims(MeanField &mf, const string &file_path)
{
    using std::complex;
    using std::vector;
    using std::cerr;
    using std::endl;
    using librpa_int::global::mpi_comm_global_h;

    int nk = mf.get_n_kpoints();
    int n_spins = mf.get_n_spins();
    int nbands = mf.get_n_bands();
    auto &velocity = mf.get_velocity();

    for (int ik = 0; ik < nk; ik++)
    {
        std::stringstream ss;
        ss << file_path << "mommat_ks_kpt_" << std::setfill('0') << std::setw(6) << ik + 1
           << ".dat";

        std::ifstream infile(ss.str(), std::ios::binary);
        if (!infile.is_open())
        {
            std::cerr << "Failed to open file: " << ss.str() << std::endl;
            continue;
        }

        int i_k_point, n_state_min, n_state_max, ld, n_spin_in, n_pol_dir;
        infile.read(reinterpret_cast<char *>(&i_k_point), sizeof(int));
        infile.read(reinterpret_cast<char *>(&n_state_min), sizeof(int));
        infile.read(reinterpret_cast<char *>(&n_state_max), sizeof(int));
        infile.read(reinterpret_cast<char *>(&ld), sizeof(int));
        infile.read(reinterpret_cast<char *>(&n_spin_in), sizeof(int));
        infile.read(reinterpret_cast<char *>(&n_pol_dir), sizeof(int));

        int n_pairs = ld * n_spin_in * n_pol_dir;
        std::vector<std::complex<double>> mommat(n_pairs);
        infile.read(reinterpret_cast<char *>(mommat.data()),
                    n_pairs * sizeof(std::complex<double>));
        infile.close();

        int iline = 0;
        for (int ipol = 0; ipol < n_pol_dir; ipol++)
        {
            for (int is = 0; is < n_spins; is++)
            {
                for (int im = 0; im < nbands; im++)
                {
                    for (int in = im; in < nbands; in++)
                    {
                        velocity.at(is).at(ik).at(ipol)(in, im) = mommat[iline];
                        velocity.at(is).at(ik).at(ipol)(im, in) = std::conj(mommat[iline]);
                        iline++;
                    }
                }
            }
        }
    }

    if (mpi_comm_global_h.is_root())
        std::cout << "* Success: read moment from mommat_ks_kpt_*.dat (FHI-aims)." << std::endl;
}

/* void read_velocity_aims(MeanField &mf, const string &file_path)
{
    int nk = mf.get_n_kpoints();
    int n_spins = mf.get_n_spins();
    int nbands = mf.get_n_bands();
    auto &velocity = mf.get_velocity();

    ifstream infile;
    for (int ik = 0; ik != nk; ik++)
    {
        vector<complex<double>> px(n_spins * nbands * nbands), py(n_spins * nbands * nbands),
            pz(n_spins * nbands * nbands);
        // Load momentum matrix
        std::stringstream ss;
        ss << file_path << "mommat_ks_kpt_" << std::setfill('0') << std::setw(6) << ik + 1
           << ".dat";
        infile.open(ss.str());
        if (!infile.is_open())
        {
            cerr << "Failed to open file: " << ss.str() << endl;
            continue;
        }
        std::string px_re, px_im, py_re, py_im, pz_re, pz_im;
        int line = 0;
        while (infile.peek() != EOF)
        {
            infile >> px_re >> px_im >> py_re >> py_im >> pz_re >> pz_im;
            px[line] = complex<double>(stod(px_re), stod(px_im));
            py[line] = complex<double>(stod(py_re), stod(py_im));
            pz[line] = complex<double>(stod(pz_re), stod(pz_im));
            line++;
        }
        infile.close();
        int iline = 0;
        for (int im = 0; im != nbands; im++)
        {
            for (int in = im; in != nbands; in++)
            {
                for (int is = 0; is != n_spins; is++)
                {
                    velocity.at(is).at(ik).at(0)(in, im) = px[iline];
                    velocity.at(is).at(ik).at(1)(in, im) = py[iline];
                    velocity.at(is).at(ik).at(2)(in, im) = pz[iline];
                    velocity.at(is).at(ik).at(0)(im, in) = conj(px[iline]);
                    velocity.at(is).at(ik).at(1)(im, in) = conj(py[iline]);
                    velocity.at(is).at(ik).at(2)(im, in) = conj(pz[iline]);
                    iline++;
                }
            }
        }
    }
    if (LIBRPA::envs::mpi_comm_global_h.is_root())
        std::cout << "* Success: read moment from mommat_ks_kpt_*.dat(FHI-aims)." << std::endl;
} */

void read_dielec_func(const string &file_path, std::vector<double> &omegas,
                      std::vector<double> &dielec_func_imagfreq)
{
    std::ifstream ifs;
    double omega, re, im;
    ifs.open(file_path);

    if (!ifs.good())
    {
        throw std::logic_error("Failed to open " + file_path);
    }

    while (ifs >> omega >> re >> im)
    {
        omegas.push_back(omega);
        dielec_func_imagfreq.push_back(re);
    }
    ifs.close();
}



void erase_Cs_from_local_atp(atpair_R_mat_t &Cs, std::vector<atpair_t> &local_atpair)
{
    using namespace std;
    using namespace librpa_int;
    //erase no need Cs

    set<size_t> loc_atp_index;
    for (auto &lap : local_atpair)
    {
        loc_atp_index.insert(lap.first);
        loc_atp_index.insert(lap.second);
    }
    std::vector<atom_t> Cs_first;
    for (const auto &Ip: Cs)
        Cs_first.push_back(Ip.first);
    for (const auto &I: Cs_first)
    {
        if (!loc_atp_index.count(I)) Cs.erase(I);
    }
    // for(auto &Ip:Cs)
    //     if(!loc_atp_index.count(Ip.first))
    //     {
    //         Cs.erase(Ip.first);
    //     }
    release_free_mem();
    global::lib_printf("| process %d, size of Cs after erase: %lu\n", librpa_int::global::mpi_comm_global_h.myid, Cs.size());
}

void read_stru(const std::string &file_path)
{
    using namespace librpa_int;
    global::lib_printf_root("Reading structure file: %s\n", file_path.c_str());

    ifstream infile;
    infile.open(file_path);
    if (!infile.good())
        throw LIBRPA_RUNTIME_ERROR("Fail to open structure file " + file_path);
    string x, y, z, tmp;

    std::vector<double> lat_mat(9);
    std::vector<double> G_mat(9);

    for (int i = 0; i < 3; i++)
    {
        infile >> x >> y >> z;
        lat_mat[i * 3] = stod(x);
        lat_mat[i * 3 + 1] = stod(y);
        lat_mat[i * 3 + 2] = stod(z);
    }

    for (int i = 0; i < 3; i++)
    {
        infile >> x >> y >> z;
        G_mat[i * 3] = stod(x);
        G_mat[i * 3 + 1] = stod(y);
        G_mat[i * 3 + 2] = stod(z);
    }

    driver::h.set_latvec_and_G(lat_mat.data(), G_mat.data());

    // Read coordinates of atoms
    infile >> driver::n_atoms;
    const auto n_atoms = driver::n_atoms;
    driver::atom_types.resize(n_atoms);
    std::vector<double> coords(n_atoms * 3);
    int type;
    for (size_t iat = 0; iat < n_atoms; iat++)
    {
        for (int i = 0; i < 3; i++) infile >> coords[3 * iat + i];
        infile >> type;
        driver::atom_types[iat] = type - 1;
    }
    // Parsed after lattice is set, so that the fractional coordinates are calculated
    driver::h.set_atoms(driver::atom_types, coords);

    // // Internal check
    // const auto ds = api::get_dataset_instance(driver::h.get_c_handler());
    // const auto &pbc = ds->pbc;
    // Matrix3 latG = pbc.latvec * pbc.G.Transpose();
    // cout << " lat * G^T" << endl;
    // latG.print(5);
}

void read_bz_sampling(const std::string &file_path)
{
    using namespace librpa_int;

    global::lib_printf_root("Reading Brillouin zone sampling file: %s\n", file_path.c_str());

    ifstream infile;
    infile.open(file_path);
    if (!infile.good())
        throw LIBRPA_RUNTIME_ERROR("Fail to open BZ sampling file " + file_path);

    string x, y, z, tmp;

    int nk[3];
    for (int i = 0; i < 3; i++)
    {
        infile >> nk[i];
    }
    int nk_full, nk_ibz;
    infile >> nk_full >> nk_ibz;
    assert(nk_full == nk[0] * nk[1] * nk[2]);

    std::vector<double> kvecs(3 * nk_full);
    std::vector<int> map_ibzk(nk_full, -1);

    // kvec_c = new Vector3<double>[n_kpoints];
    for (int i = 0; i != nk_full; i++)
    {
        // id weight kfrac[3] kcart[3] ik_ibz map_ibz
        infile >> x >> y;
        infile >> x >> y >> z;
        // kcart[3]
        infile >> kvecs[3 * i] >> kvecs[3 * i + 1] >> kvecs[3 * i + 2];
        // ik_ibz map_ibz
        infile >> tmp >> map_ibzk[i];
        map_ibzk[i] -= 1;
        // Save a copy in the driver for information printing
        Vector3_Order<double> kvec(kvecs[3 * i], kvecs[3 * i + 1], kvecs[3 * i + 2]);
        auto it = std::find(driver::ibz_kpoints.cbegin(), driver::ibz_kpoints.cend(), kvec);
        if (it == driver::ibz_kpoints.cend()) driver::ibz_kpoints.emplace_back(kvec);
    }
    infile.close();

    driver::n_ibz_kpoints = driver::ibz_kpoints.size();

    driver::h.set_kgrids_kvec(nk[0], nk[1], nk[2], kvecs.data());
    driver::h.set_ibz_mapping(map_ibzk);
}

void read_bz_sampling_from_stru(const std::string &file_path)
{
    using namespace librpa_int;

    global::lib_printf_root("Fallback reading Brillouin zone sampling from stru file: %s\n", file_path.c_str());

    ifstream infile;
    string x;
    infile.open(file_path);
    if (!infile.good())
        throw LIBRPA_RUNTIME_ERROR("Fail to open structure file " + file_path);

    // Skip direct and reciprocal lattice vectors
    for (int i = 0; i < 6; i++)
    {
        infile >> x >> x >> x;
    }
    // Skip atom coordinates
    int n_atoms;
    infile >> n_atoms;
    for (int iat = 0; iat < n_atoms; iat++)
    {
        for (int i = 0; i < 4; i++) infile >> x;
    }

    // Begin k-grid information
    int nk[3];
    for (int i = 0; i < 3; i++)
    {
        infile >> x;
        nk[i] = stoi(x);
    }
    const int nk_full = nk[0] * nk[1] * nk[2];
    std::vector<double> kvecs(3 * nk_full);
    std::vector<int> map_ibzk(nk_full, -1);

    for (int i = 0; i != 3 * nk_full; i++)
    {
        infile >> x;
        kvecs[i] = stod(x);
    }

    for (int i = 0; i != nk_full; i++)
    {
        infile >> map_ibzk[i];
        map_ibzk[i] -= 1;
        Vector3_Order<double> kvec(kvecs[3 * i], kvecs[3 * i + 1], kvecs[3 * i + 2]);
        auto it = std::find(driver::ibz_kpoints.cbegin(), driver::ibz_kpoints.cend(), kvec);
        if (it == driver::ibz_kpoints.cend()) driver::ibz_kpoints.emplace_back(kvec);
    }
    infile.close();

    driver::n_ibz_kpoints = driver::ibz_kpoints.size();

    driver::h.set_kgrids_kvec(nk[0], nk[1], nk[2], kvecs.data());
    driver::h.set_ibz_mapping(map_ibzk);
}

void read_basis(const std::string &file_path)
{
    using namespace librpa_int;

    global::lib_printf_root("Reading basis information file: %s\n", file_path.c_str());
    ifstream infile;
    infile.open(file_path);

    int n_atoms = driver::atom_types.size();
    if (static_cast<size_t>(n_atoms) != driver::n_atoms)
        throw LIBRPA_RUNTIME_ERROR("Number of atoms not consistent with the geometry file!");
    std::map<int, size_t> map_at_wfc;
    std::map<int, size_t> map_at_aux;
    std::vector<size_t> nbs_wfc(n_atoms);
    std::vector<size_t> nbs_aux(n_atoms);

    int ntypes, type;
    size_t n_wfc, n_aux;
    string kind_str;

    infile >> ntypes;
    // total basis, not used here
    infile >> n_wfc >> n_aux >> kind_str;

    for (int itype = 0; itype < ntypes; itype++)
    {
        infile >> type >> n_wfc >> n_aux;
        type--;
        map_at_wfc[type] = n_wfc;
        map_at_aux[type] = n_aux;
    }
    for (int iat = 0; iat < n_atoms; iat++)
    {
        auto type = driver::atom_types[iat];
        nbs_wfc[iat] = map_at_wfc.at(type);
        nbs_aux[iat] = map_at_aux.at(type);
    }

    // std::cout << "nbs_wfc " << nbs_wfc << std::endl;
    // std::cout << "nbs_aux " << nbs_aux << std::endl;

    driver::h.set_ao_basis_wfc(nbs_wfc);
    driver::h.set_ao_basis_aux(nbs_aux);

    infile.close();
}

void read_band_kpath_info(const string &file_path)
{
    using driver::n_spins;
    using driver::n_basis_wfc;
    using driver::n_basis_ao;
    using driver::n_states;
    using driver::n_kpoints_band;
    using driver::kfrac_band;

    int n_basis_band, n_states_band, n_spin_band;

    ifstream infile;
    infile.open(file_path);
    if (!infile.good())
    {
        throw std::logic_error("Failed to open " + file_path);
    }

    string x, y, z;

    // Read dimensions in the first row
    infile >> x;
    n_basis_band = stoi(x);
    if (n_basis_band != n_basis_wfc)
        throw LIBRPA_RUNTIME_ERROR("band & SCF #basis inconsistent");
    infile >> x;
    n_states_band = stoi(x);
    if (n_states_band != n_states)
        throw LIBRPA_RUNTIME_ERROR("band & SCF #state inconsistent");
    infile >> x;
    n_spin_band = stoi(x);
    if (n_spin_band != n_spins)
        throw LIBRPA_RUNTIME_ERROR("band & SCF #spin inconsistent");
    infile >> x;
    n_kpoints_band = stoi(x);

    kfrac_band.clear();
    std::vector<double> vector_kfrac_band(n_kpoints_band * 3); // For API parsing
    for (int i = 0; i < n_kpoints_band; i++)
    {
        infile >> x >> y >> z;
        Vector3_Order<double> kfrac{stod(x), stod(y), stod(z)};
        kfrac_band.emplace_back(kfrac);
        vector_kfrac_band[3*i] = kfrac.x;
        vector_kfrac_band[3*i+1] = kfrac.y;
        vector_kfrac_band[3*i+2] = kfrac.z;
    }

    infile.close();

    driver::h.set_band_kvec(n_kpoints_band, vector_kfrac_band.data());
}

void read_band_meanfield_data(const string &dir_path)
{
    using namespace librpa_int;
    using namespace librpa_int::global;
    using std::endl;
    using driver::n_spins;
    using driver::n_kpoints_band;
    using driver::n_states;
    using driver::n_basis_wfc;
    using driver::n_basis_ao;
    using driver::iks_band_eigvec_this;

    if (driver::n_kpoints_band == 0)
        throw LIBRPA_RUNTIME_ERROR("Number of band k-points not set, run read_band_kpath_info first");

    iks_band_eigvec_this.clear();

    if (driver::get_bool(driver::opts.use_kpara_scf_eigvec))
    {
        for (int ik = 0; ik < driver::n_kpoints_band; ik++)
        {
            if (ik % size_global == myid_global) iks_band_eigvec_this.emplace_back(ik);
        }
    }
    else
    {
        for (int ik = 0; ik < driver::n_kpoints_band; ik++)
            iks_band_eigvec_this.emplace_back(ik);
    }

    std::vector<double> eskb(n_spins * n_kpoints_band * n_states);
    std::vector<double> wskb(n_spins * n_kpoints_band * n_states);

    const int n_kb = n_kpoints_band * n_states;
    std::string s1, s2, s3, s4, s5;

    // Load occupation weights and eigenvalues
    for (int ik = 0; ik < n_kpoints_band; ik++)
    {
        std::stringstream ss;
        ss << dir_path << "band_KS_eigenvalue_k_" << std::setfill('0') << std::setw(5) << ik + 1
           << ".txt";
        ofs_myid << "Loading band eigenvalues from " << ss.str() << std::endl;
        ifstream infile;
        infile.open(ss.str());
        for (int i_spin = 0; i_spin < n_spins; i_spin++)
        {
            for (int i_state = 0; i_state < n_states; i_state++)
            {
                infile >> s1 >> s2 >> s3 >> s4 >> s5;
                const int index = i_spin * n_kb + ik * n_states + i_state;
                wskb[index] = stod(s3);
                eskb[index] = stod(s4);
            }
        }
        infile.close();
    }
    driver::h.set_band_occ_eigval(n_spins, n_kpoints_band, n_states, wskb.data(), eskb.data());

    // Load eigenvectors
    for (int ik = 0; ik < n_kpoints_band; ik++)
    {
        bool skip_this_ik = false;
        if (driver::get_bool(driver::opts.use_kpara_scf_eigvec))
        {
            const auto it =
                std::find(iks_band_eigvec_this.cbegin(), iks_band_eigvec_this.cend(), ik);
            skip_this_ik = (it == iks_band_eigvec_this.cend());
        }
        if (skip_this_ik) continue;

        std::stringstream ss;
        ss << dir_path << "band_KS_eigenvector_k_" << std::setfill('0') << std::setw(5) << ik + 1 << ".txt";

        ifstream infile;
        infile.open(ss.str(), std::ios::in | std::ios::binary);
        if (!infile.good())
            throw LIBRPA_RUNTIME_ERROR("Fail to open band eigenvector file " + ss.str());
        else
            ofs_myid << "Loading band eigenvector file " + ss.str() << endl;

        std::vector<std::complex<double>> wfc(n_states * n_basis_wfc);
        // for (int i_spin = 0; i_spin < n_spins; i_spin++)
        // {
        //     const size_t nbytes = n_basis_wfc * n_states * sizeof(std::complex<double>);
        //     infile.read((char *) wfc.data(), nbytes);
        //     // TODO: adapt to SOC case
        //     driver::h.set_wfc_band_packed(i_spin, ik, n_states, n_basis_wfc, wfc.data());
        // }

        // TODO: decide which basis to use
        size_t total_complex_comp = static_cast<size_t>(n_states) * static_cast<size_t>(n_basis_ao);  // for one component
        size_t total_complex_spin = static_cast<size_t>(n_states) * static_cast<size_t>(n_basis_wfc);
        size_t total_complex = total_complex_spin * n_spins;
        size_t bytes_doubles = total_complex * 2 * sizeof(double);

        std::vector<std::complex<double>> vecs(total_complex);
        infile.read(reinterpret_cast<char *>(vecs.data()), bytes_doubles);
        if (!infile || infile.gcount() != static_cast<ptrdiff_t>(bytes_doubles))
        {
            throw LIBRPA_RUNTIME_ERROR("Error: failed to read " + ss.str());
        }

        const bool use_spinor_wfc = driver::driver_params.use_spinor_wfc;
        int n_soc = use_spinor_wfc ? 2 : 1;
        assert (n_soc * n_spins <= 2);
        for (int i_spin = 0; i_spin < n_spins; ++i_spin)
        {
            std::vector<std::complex<double>> vecs_sp(total_complex_spin);
            for (int ib = 0; ib < n_states; ++ib)
            {
                for (int iw = 0; iw < n_basis_ao; ++iw)
                {
                    for (int i_soc = 0; i_soc < n_soc; ++i_soc)
                    {
                        const size_t index_dst = i_soc * total_complex_comp + ib * n_basis_ao + iw;
                        size_t index_src;
                        if (use_spinor_wfc)
                        {
                            // NOTE: i_spin should be 0 for spinor-form wavefunction
                            assert(i_spin < 1);
                            index_src = ib * n_basis_ao * n_soc + iw * n_soc + i_soc;
                        }
                        else
                        {
                            index_src = i_spin * n_basis_ao * n_states + ib * n_basis_ao + iw;
                        }
                        vecs_sp[index_dst] = vecs[index_src];
                    }
                }
            }
            if (use_spinor_wfc)
            {
                driver::h.set_wfc_band_spinor_packed(ik, n_states, n_basis_ao, vecs_sp.data(),
                                                     vecs_sp.data() + total_complex_comp);
            }
            else
            {
                driver::h.set_wfc_band_packed(i_spin, ik, n_states, n_basis_ao, vecs_sp.data());
            }
        }

        infile.close();
    }
}

std::vector<matrix> read_vxc_band(const string &dir_path, int n_states, int n_spin,
                                  int n_kpoints_band)
{
    std::vector<matrix> vxc_band(n_spin);
    for (int i_spin = 0; i_spin < n_spin; i_spin++)
    {
        vxc_band[i_spin].create(n_kpoints_band, n_states);
    }
    std::string s1, s2, s3;

    for (int ik = 0; ik < n_kpoints_band; ik++)
    {
        // Load occupation weights and eigenvalues
        std::stringstream ss;
        ss << dir_path << "band_vxc_k_" << std::setfill('0') << std::setw(5) << ik + 1 << ".txt";
        ifstream infile;
        infile.open(ss.str());
        ss.clear();

        for (int i_spin = 0; i_spin < n_spin; i_spin++)
        {
            for (int i_state = 0; i_state < n_states; i_state++)
            {
                infile >> s1 >> s2 >> s3;
                vxc_band[i_spin](ik, i_state) = stod(s3);
            }
        }

        infile.close();
    }
    return vxc_band;
}

void read_elsi_csc(const string &file_path, bool save_row_major, std::vector<double> &mat,
                   int &n_basis, bool &is_real)
{
    ifstream infile;
    infile.open(file_path, std::ios::binary);
    if (!infile.good())
    {
        throw std::logic_error("Failed to open " + file_path);
    }

    // Read the whole buffer
    infile.seekg(0, std::ios::end);
    std::streampos size = infile.tellg();
    infile.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    infile.read(buffer.data(), size);
    infile.close();

    int64_t header[16];
    std::memcpy(header, buffer.data(), 128);

    n_basis = header[3];
    int64_t nnz = header[5];
    // cout << n_basis << " " << nnz << endl;

    int64_t *col_ptr_raw = reinterpret_cast<int64_t *>(buffer.data() + 128);
    std::vector<int> col_ptr;
    col_ptr.assign(col_ptr_raw, col_ptr_raw + n_basis);
    // Trailing column index to mark the end. +1 for index starting from 1 in ELSI CSC
    col_ptr.push_back(nnz + 1);

    int32_t *row_idx_raw = reinterpret_cast<int32_t *>(buffer.data() + 128 + n_basis * 8);

    char *nnz_val_raw = buffer.data() + 128 + n_basis * 8 + nnz * 4;
    double *nnz_val_double = reinterpret_cast<double *>(nnz_val_raw);

    if (header[2] == 0)
    {
        // Real valued
        is_real = true;
        mat.resize(n_basis * n_basis);
    }
    else
    {
        // Complex valued
        is_real = false;
        mat.resize(2 * n_basis * n_basis);
    }

    for (auto col = 0; col < n_basis; ++col)
    {
        for (auto idx = col_ptr[col]; idx < col_ptr[col + 1]; ++idx)
        {
            int row = row_idx_raw[idx - 1] - 1;
            int index = save_row_major ? row * n_basis + col : col * n_basis + row;
            // cout << idx - 1 << " " << col << " " << row << " " << index << endl;
            if (is_real)
            {
                mat[index] = nnz_val_double[idx - 1];
            }
            else
            {
                mat[2 * index] = nnz_val_double[2 * idx - 2];
                mat[2 * index + 1] = nnz_val_double[2 * idx - 1];
            }
        }
    }
}

static int handle_sinvS_file(const std::string &file_path,
                             std::map<Vector3_Order<double>, ComplexMatrix> &sinvS, bool binary)
{
    ifstream infile;
    int n_irk_points_local;
    // TODO: variables that needs to be adapted into pbc object
    std::map<Vector3_Order<double>, double> irk_weight;
    int n_irk_points;

    auto pds = librpa_int::api::get_dataset_instance(driver::h.get_c_handler());
    auto &pbc = pds->pbc;

    if (binary)
    {
        infile.open(file_path, std::ios::in | std::ios::binary);
        infile.read((char *) &n_irk_points, sizeof(int));
        infile.read((char *)&n_irk_points_local, sizeof(int));
    }
    else
    {
        infile.open(file_path);
        infile >> n_irk_points;
    }

    if (!infile.good()) return 1;

    const int nk_ibz = pbc.klist_ibz.size();

    if (binary)
    {
        int nbasbas_s, nbasbas, brow, erow, bcol, ecol, iq;
        double q_weight;

        for (int i_irk = 0; i_irk < n_irk_points_local; i_irk++)
        {
            infile.read((char *)&nbasbas_s, sizeof(int));
            infile.read((char *)&nbasbas, sizeof(int));
            infile.read((char *)&brow, sizeof(int));
            infile.read((char *)&erow, sizeof(int));
            infile.read((char *)&bcol, sizeof(int));
            infile.read((char *)&ecol, sizeof(int));
            infile.read((char *)&iq, sizeof(int));
            infile.read((char *)&q_weight, sizeof(double));

            brow--;
            erow--;
            bcol--;
            ecol--;
            iq--;
            if ((erow - brow < 0) || (ecol - bcol < 0) || iq < 0 || iq >= nk_ibz) return 4;
            const auto qvec = pbc.klist_ibz[iq];

            if (!sinvS.count(qvec))
            {
                sinvS[qvec].create(nbasbas_s, nbasbas);
            }

            const int nrow = erow - brow + 1;
            const int ncol = ecol - bcol + 1;
            const size_t n = nrow * ncol;
            std::vector<std::complex<double>> tmp(n);
            infile.read((char *)tmp.data(), 2 * n * sizeof(double));
            for (int i = 0; i < nrow; i++)
            {
                for (int j = 0; j < ncol; j++)
                {
                    const auto i_mu = i + brow;
                    const auto i_nu = j + bcol;
                    sinvS[qvec](i_mu, i_nu) = tmp[i * ncol + j];  // for abacus
                }
            }
        }
    }
    else
    {
        string nbasbas_s, nbasbas, begin_row, end_row, begin_col, end_col, q1, q2, q3, vq_r, vq_i,
            q_num, q_weight;
        while (infile.peek() != EOF)
        {
            // row is mu_s, col is mu
            infile >> nbasbas_s >> nbasbas >> begin_row >> end_row >> begin_col >> end_col;
            if (infile.peek() == EOF) break;
            if (!infile.good()) return 2;

            infile >> q_num >> q_weight;
            if (!infile.good()) return 3;
            int mu = stoi(nbasbas_s);
            int nu = stoi(nbasbas);
            int brow = stoi(begin_row) - 1;
            int erow = stoi(end_row) - 1;
            int bcol = stoi(begin_col) - 1;
            int ecol = stoi(end_col) - 1;
            int iq = stoi(q_num) - 1;

            // skip empty coulumb_file
            if ((erow - brow < 0) || (ecol - bcol < 0) || iq < 0 || iq >= nk_ibz) return 4;
            const auto qvec = pbc.klist_ibz[iq];
            if (!sinvS.count(qvec))
            {
                sinvS[qvec].create(mu, nu);
            }
            for (int i_mu = brow; i_mu <= erow; i_mu++)
            {
                for (int i_nu = bcol; i_nu <= ecol; i_nu++)
                {
                    infile >> vq_r >> vq_i;
                    // Vq_full[qvec](i_nu, i_mu) = complex<double>(stod(vq_r), stod(vq_i)); // for
                    // FHI-aims
                    sinvS[qvec](i_mu, i_nu) =
                        std::complex<double>(stod(vq_r), stod(vq_i));  // for abacus
                }
            }
        }
    }

    return 0;
}

void read_ri_shrink(const string &dir_path)
{
    using std::cout;
    using std::endl;
    using librpa_int::global::profiler;
    using librpa_int::global::mpi_comm_global_h;
    using librpa_int::global::myid_global;
    using librpa_int::global::lib_printf;
    using driver::driver_params;

    std::map<Vector3_Order<double>, ComplexMatrix> sinvS;

    auto pds = librpa_int::api::get_dataset_instance(driver::h.get_c_handler());
    const auto &abf = pds->basis_aux;
    const auto &abf_shrink = pds->basis_aux_shrink;

    if (mpi_comm_global_h.is_root())
    {
        std::cout << "iatom & large Nabfs: " << std::endl;
        int I = 0;
        for (auto &mu : abf.get_atom_nbs())
        {
            // use i and x
            std::cout << I << "," << mu << std::endl;
            ++I;
        }
    }

    profiler.start("read_Cs_shrink");
    // backup large atom_mu
    // atom_mu_l = atom_mu;  // TODO: replace with the actual shrinked ABFs
    read_Cs_evenly_distribute(driver_params.input_dir, driver_params.cs_threshold,
                              mpi_comm_global_h.myid, mpi_comm_global_h.nprocs,
                              driver_params.prefix_lri_coeff_shrink,
                              driver_params.version_lri_reader);
    profiler.stop("read_Cs_shrink");

    profiler.start("read_shrink_sinvS_fold", "Load shrink transformation");
    // change atom_mu: number of {Mu,mu} in the later calculations
    read_shrink_sinvS(driver_params.input_dir, "shrink_sinvS_", sinvS);

    if (mpi_comm_global_h.is_root())
    {
        std::cout << "iatom & small Nabfs: " << std::endl;
        int I = 0;
        for (auto &mu : abf_shrink.get_atom_nbs())
        {
            // use i and x
            std::cout << I << "," << mu << std::endl;
            ++I;
        }
    }
    profiler.stop("read_shrink_sinvS_fold");
}

size_t read_shrink_sinvS(const string &dir_path, const string &vq_fprefix,
                         std::map<Vector3_Order<double>, ComplexMatrix> &sinvS)
{
    using std::cout;
    using std::endl;
    using librpa_int::global::profiler;
    using librpa_int::global::myid_global;
    using librpa_int::global::lib_printf;

    size_t vq_save = 0;
    size_t vq_discard = 0;
    struct dirent *ptr;
    DIR *dir;
    dir = opendir(dir_path.c_str());
    std::vector<std::string> files;

    bool binary;
    bool binary_checked = false;

    profiler.start("handle_sinvS_file");
    while ((ptr = readdir(dir)) != NULL)
    {
        string fm(ptr->d_name);
        if (fm.find(vq_fprefix) == 0)
        {
            string file_path = dir_path + fm;
            if (!binary_checked)
            {
                binary = check_coulomb_file_binary(file_path);
                binary_checked = true;
                if (myid_global == 0)
                {
                    if (binary)
                    {
                        cout << "sinvS: Unformatted binary V files detected" << endl;
                    }
                    else
                    {
                        cout << "sinvS: ASCII format V files detected" << endl;
                    }
                }
            }
            int retcode = handle_sinvS_file(file_path, sinvS, binary);
            if (retcode != 0)
            {
                lib_printf("Error encountered when reading %s, return code %d",
                           fm.c_str(), retcode);
            }
        }
    }
    profiler.stop("handle_sinvS_file");

    closedir(dir);
    dir = NULL;
    return vq_discard;
}
