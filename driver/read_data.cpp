#include "read_data.h"
#include <librpa_enums.h>

#include "reader_basis.h"
#include "reader_lri.h"
#include "reader_coulomb.h"
#include "reader_structure.h"

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
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

#include <librpa.hpp>

#include "driver.h"
#include "../src/mpi/global_mpi.h"
#include "../src/core/input_symmetry.h"
#include "../src/core/pbc.h"
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

namespace
{

constexpr double kInputSymmetryKpointMatchTol = 1e-5;
constexpr double kBzSamplingWeightSumTol = 1e-6;
constexpr std::int32_t READER_SHRINK_SINVS_V1_MARKER = -30241621;
constexpr std::int32_t EIGENVECTOR_V1_MARKER = -12345678;

bool any_symmetry_speedup_enabled()
{
    return driver::get_bool(driver::opts.use_symmetry_exx)
           || driver::get_bool(driver::opts.use_symmetry_gw)
           || driver::get_bool(driver::opts.use_symmetry_rpa);
}

bool nearly_same_kpoint(const librpa_int::Vector3_Order<double> &lhs,
                       const librpa_int::Vector3_Order<double> &rhs,
                       const double tol = kInputSymmetryKpointMatchTol)
{
    return std::abs(lhs.x - rhs.x) <= tol
           && std::abs(lhs.y - rhs.y) <= tol
           && std::abs(lhs.z - rhs.z) <= tol;
}

} // namespace

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

void read_scf_occ_eigenvalues(const string &file_path, MeanField &mf, bool use_spinor_wfc)
{
    ifstream infile;
    infile.open(file_path);
    if (!infile.good())
    {
        throw std::logic_error("Failed to open " + file_path);
    }

    string ks, ss, a, ws, es, d;
    int n_kpoints, n_spins, n_states, n_basis_wfc;
    double efermi;
    infile >> n_kpoints;
    infile >> n_spins;
    infile >> n_states;
    infile >> n_basis_wfc;
    infile >> efermi;

    int n_spinor = 1;
    int n_basis_ao = n_basis_wfc;
    if (use_spinor_wfc)
    {
        assert(n_spins == 1);
        assert(n_basis_wfc % 2 == 0 && "Error: nbasis is not even when SOC!");
        n_spinor = 2;
        n_basis_ao = n_basis_wfc / 2;
    }

    mf = MeanField();
    mf.set(n_spins, n_kpoints, n_states, n_basis_ao, n_spinor);
    mf.get_efermi() = efermi;

    auto &eskb = mf.get_eigenvals();
    auto &wskb = mf.get_weight();

    int iline = 6;
    for (int ik = 0; ik != n_kpoints; ik++)
    {
        for (int is = 0; is != n_spins; is++)
        {
            infile >> ks >> ss;
            if (!infile.good())
            {
                throw std::logic_error("Error in reading k- and spin- index: line " +
                                       std::to_string(iline) + ", file: " + file_path);
            }
            iline++;
            int k_index = stoi(ks) - 1;
            for (int i = 0; i != n_states; i++)
            {
                infile >> a >> ws >> es >> d;
                if (!infile.good())
                {
                    throw std::logic_error("Error in reading band energy and occupation: line " +
                                           std::to_string(iline) + ", file: " + file_path);
                }
                iline++;
                wskb[is](k_index, i) = stod(ws) / n_kpoints;
                eskb[is](k_index, i) = stod(es);
            }
        }
    }
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

static bool check_KS_file_binary(const string &file_path)
{
    ifstream infile(file_path, std::ios::in | std::ios::binary);
    if (!infile.good()) return false;
    std::int32_t marker = 0;
    infile.read(reinterpret_cast<char *>(&marker), sizeof(std::int32_t));
    return infile.good() && marker == EIGENVECTOR_V1_MARKER;
}

static int handle_KS_file(const string &file_path, bool binary)
{
    using driver::iks_eigvec_this;

    int ret = 0;
    ifstream infile;
    if (binary)
    {
        infile.open(file_path, std::ios::in | std::ios::binary);
    }
    else
    {
        infile.open(file_path);
    }
    if (!infile.good()) return 1;

    const auto nspin = driver::n_spins;
    const auto nsoc = driver::n_spinor;
    const auto nband = driver::n_states;
    const auto nao = driver::n_basis_ao;
    const auto nbao = nband * nao;
    const auto n = nsoc * nbao;
    const bool use_spinor_wfc = driver::driver_params.use_spinor_wfc;

    std::vector<double> re(nspin * n);
    std::vector<double> im(nspin * n);

    if (binary)
    {
        std::int32_t marker = 0;
        std::int32_t n_kpoints_file = 0;
        std::int32_t n_spins_file = 0;
        std::int32_t n_spinor_file = 0;
        std::int32_t n_states_file = 0;
        std::int32_t n_aos_file = 0;
        infile.read(reinterpret_cast<char *>(&marker), sizeof(std::int32_t));
        infile.read(reinterpret_cast<char *>(&n_kpoints_file), sizeof(std::int32_t));
        infile.read(reinterpret_cast<char *>(&n_spins_file), sizeof(std::int32_t));
        infile.read(reinterpret_cast<char *>(&n_spinor_file), sizeof(std::int32_t));
        infile.read(reinterpret_cast<char *>(&n_states_file), sizeof(std::int32_t));
        infile.read(reinterpret_cast<char *>(&n_aos_file), sizeof(std::int32_t));
        if (!infile.good() ||
            marker != EIGENVECTOR_V1_MARKER ||
            n_spins_file != nspin ||
            n_spinor_file != nsoc ||
            n_states_file != nband ||
            n_aos_file != nao)
        {
            return 1;
        }

        const std::size_t kpoint_data_bytes =
            static_cast<std::size_t>(nspin) * nsoc * nband * nao *
            sizeof(std::complex<double>);

        for (std::int32_t ik_read = 0; ik_read != n_kpoints_file; ++ik_read)
        {
            std::int32_t ik_file = 0;
            infile.read(reinterpret_cast<char *>(&ik_file), sizeof(std::int32_t));
            if (!infile.good()) { ret = 1; break; }
            const int ik = ik_file - 1;

            bool skip_this_ik = false;
            if (driver::get_bool(driver::opts.use_kpara_scf_eigvec))
            {
                const auto it = std::find(iks_eigvec_this.cbegin(), iks_eigvec_this.cend(), ik);
                skip_this_ik = (it == iks_eigvec_this.cend());
            }

            if (skip_this_ik)
            {
                infile.seekg(static_cast<std::streamoff>(kpoint_data_bytes), std::ios::cur);
                if (!infile.good()) { ret = 1; break; }
                continue;
            }

            std::fill(re.begin(), re.end(), 0.0);
            std::fill(im.begin(), im.end(), 0.0);

            for (int iw = 0; iw != nao; ++iw)
            {
                for (int isoc = 0; isoc != nsoc; ++isoc)
                {
                    for (int ib = 0; ib != nband; ++ib)
                    {
                        for (int is = 0; is != nspin; ++is)
                        {
                            double rv = 0.0;
                            double iv = 0.0;
                            infile.read(reinterpret_cast<char *>(&rv), sizeof(double));
                            infile.read(reinterpret_cast<char *>(&iv), sizeof(double));
                            if (infile.bad()) { ret = 1; break; }
                            re[is * n + isoc * nbao + ib * nao + iw] = rv;
                            im[is * n + isoc * nbao + ib * nao + iw] = iv;
                        }
                    }
                }
            }
            if (ret != 0) break;

            for (int is = 0; is != nspin; ++is)
            {
                if (use_spinor_wfc)
                {
                    assert(is == 0);
                    driver::h.set_wfc_spinor(ik, driver::n_states, driver::n_basis_ao,
                                             re.data(), im.data(),
                                             re.data() + nbao, im.data() + nbao);
                }
                else
                {
                    driver::h.set_wfc(is, ik, driver::n_states, driver::n_basis_ao,
                                      re.data() + is * n, im.data() + is * n);
                }
            }
        }
    }
    else
    {
        string rvalue, ivalue, kstr;
        while (infile.peek() != EOF)
        {
            infile >> kstr;
            int ik = stoi(kstr) - 1;
            if (infile.peek() == EOF) break;
            bool skip_this_ik = false;
            if (driver::get_bool(driver::opts.use_kpara_scf_eigvec))
            {
                const auto it = std::find(iks_eigvec_this.cbegin(), iks_eigvec_this.cend(), ik);
                skip_this_ik = (it == iks_eigvec_this.cend());
            }
            for (int iw = 0; iw != nao; ++iw)
            {
                for (int isoc = 0; isoc != nsoc; ++isoc)
                {
                    for (int ib = 0; ib != nband; ++ib)
                    {
                        for (int is = 0; is != nspin; ++is)
                        {
                            infile >> rvalue >> ivalue;
                            if (infile.bad()) { ret = 1; break; }
                            if (skip_this_ik) continue;
                            re[is * n + isoc * nbao + ib * nao + iw] = stod(rvalue);
                            im[is * n + isoc * nbao + ib * nao + iw] = stod(ivalue);
                        }
                    }
                }
            }
            if (skip_this_ik) continue;
            for (int is = 0; is != nspin; ++is)
            {
                if (use_spinor_wfc)
                {
                    assert(is == 0);
                    driver::h.set_wfc_spinor(ik, driver::n_states, driver::n_basis_ao,
                                             re.data(), im.data(),
                                             re.data() + nbao, im.data() + nbao);
                }
                else
                {
                    driver::h.set_wfc(is, ik, driver::n_states, driver::n_basis_ao,
                                      re.data() + is * n, im.data() + is * n);
                }
            }
        }
    }
    return ret;
}

static int handle_KS_file(const string &file_path, MeanField &mf, bool use_spinor_wfc,
                          const std::vector<int> *iks_selected)
{
    int ret = 0;
    ifstream infile;
    infile.open(file_path);
    if (!infile.good()) return 1;

    string rvalue, ivalue, kstr;
    const int nspin = mf.get_n_spins();
    const int nsoc = mf.get_n_spinor();
    const int nband = mf.get_n_states();
    const int nao = mf.get_n_aos();
    const int nbao = nband * nao;
    const int n = nsoc * nbao;

    std::vector<double> re(nspin * n);
    std::vector<double> im(nspin * n);

    while (infile.peek() != EOF)
    {
        infile >> kstr;
        if (!infile.good()) break;
        const int ik = stoi(kstr) - 1;
        if (infile.peek() == EOF) break;

        bool skip_this_ik = false;
        if (iks_selected)
        {
            const auto it = std::find(iks_selected->cbegin(), iks_selected->cend(), ik);
            skip_this_ik = (it == iks_selected->cend());
        }

        std::fill(re.begin(), re.end(), 0.0);
        std::fill(im.begin(), im.end(), 0.0);
        for (int iw = 0; iw != nao; iw++)
        {
            for (int isoc = 0; isoc != nsoc; isoc++)
            {
                for (int ib = 0; ib != nband; ib++)
                {
                    for (int is = 0; is != nspin; is++)
                    {
                        infile >> rvalue >> ivalue;
                        if (infile.bad())
                        {
                            ret = 1;
                            break;
                        }
                        if (skip_this_ik) continue;
                        re[is * n + isoc * nbao + ib * nao + iw] = stod(rvalue);
                        im[is * n + isoc * nbao + ib * nao + iw] = stod(ivalue);
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
                auto &wfcs_up = mf.get_eigenvectors()[0][0][ik];
                auto &wfcs_dn = mf.get_eigenvectors()[0][1][ik];
                wfcs_up.create(nband, nao);
                wfcs_dn.create(nband, nao);
                for (int i = 0; i < nbao; ++i)
                {
                    wfcs_up.c[i] = std::complex<double>(re[i], im[i]);
                    wfcs_dn.c[i] = std::complex<double>(re[nbao + i], im[nbao + i]);
                }
            }
            else
            {
                auto &wfc = mf.get_eigenvectors()[is][0][ik];
                wfc.create(nband, nao);
                for (int i = 0; i < nbao; ++i)
                    wfc.c[i] = std::complex<double>(re[is * n + i], im[is * n + i]);
            }
        }
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
        if (fm.find(driver::driver_params.prefix_eigvecs_scf) == 0)
        {
            const string file_path = dir_path + fm;
            const bool binary = check_KS_file_binary(file_path);
            ret = handle_KS_file(file_path, binary);
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

int read_eigenvector(const string &dir_path, MeanField &mf, bool use_spinor_wfc,
                     const std::vector<int> *iks_selected)
{
    int ret = 0;
    int files_read = 0;

    DIR *dir = opendir(dir_path.c_str());
    if (dir == nullptr) return -1;

    struct dirent *ptr;
    while ((ptr = readdir(dir)) != NULL)
    {
        string fm(ptr->d_name);
        if (fm.find(driver::driver_params.prefix_eigvecs_scf) == 0)
        {
            ret = handle_KS_file(dir_path + fm, mf, use_spinor_wfc, iks_selected);
            if (ret != 0)
            {
                break;
            }
            files_read++;
        }
    }
    closedir(dir);

    if (files_read == 0)
    {
        ret = -1;
    }
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
    const bool use_shrink_abfs = driver::get_bool(driver::opts.use_shrink_abfs);

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

        if (use_shrink_abfs)
            read_ri_shrink(dir_path);

        mpi_comm_global_h.barrier();
        profiler.start("driver_read_Vq");
        read_Vq_row(dir_path, driver::driver_params.prefix_coul_full,
                    driver::opts.vq_threshold, local_atpair, false,
                    driver::driver_params.version_coul_reader,
                    use_shrink_abfs);
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
        if (use_shrink_abfs)
            read_ri_shrink(dir_path);
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
                    driver::driver_params.version_coul_reader,
                    use_shrink_abfs);
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

        if (use_shrink_abfs)
            read_ri_shrink(dir_path);

        mpi_comm_global_h.barrier();
        profiler.start("driver_read_Vq");
        read_Vq_full(dir_path, driver::driver_params.prefix_coul_full, false,
                     driver::driver_params.version_coul_reader,
                     use_shrink_abfs);
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

void read_velocity(const string &file_path, const MeanField &mf, velocity_matrix_t &velocity)
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
    if (!infile.good())
        throw std::logic_error("Failed to read velocity dimensions from " + file_path);
    if (n_kpoints != mf.get_n_kpoints() || n_spins != mf.get_n_spins() ||
        n_bands != mf.get_n_states())
    {
        std::stringstream ss;
        ss << "velocity_matrix dimensions are inconsistent with meanfield: velocity=("
           << n_spins << "," << n_kpoints << "," << n_bands << "), meanfield=("
           << mf.get_n_spins() << "," << mf.get_n_kpoints() << "," << mf.get_n_states() << ")";
        throw std::logic_error(ss.str());
    }

    initialize_velocity_matrix(velocity, n_spins, n_kpoints, n_bands);
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

void read_velocity_aims(const MeanField &mf, const string &file_path,
                        velocity_matrix_t &velocity)
{
    using std::complex;
    using std::vector;
    using std::cerr;
    using std::endl;
    using librpa_int::global::mpi_comm_global_h;

    int nk = mf.get_n_kpoints();
    int n_spins = mf.get_n_spins();
    int nbands = mf.get_n_bands();
    initialize_velocity_matrix(velocity, n_spins, nk, nbands);

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

static std::vector<Vector3_Order<double>> read_headwing_k_path_info(
    const string &file_path, int &n_basis, int &n_states, int &n_spin)
{
    ifstream infile;
    infile.open(file_path);
    if (!infile.good())
    {
        throw std::logic_error("Failed to open " + file_path);
    }

    int n_kpoints;
    infile >> n_basis >> n_states >> n_spin >> n_kpoints;
    if (!infile.good())
    {
        throw std::logic_error("Failed to read headwing k_path_info header from " + file_path);
    }

    std::vector<Vector3_Order<double>> kfrac_list;
    kfrac_list.reserve(n_kpoints);
    string x, y, z;
    for (int ik = 0; ik < n_kpoints; ++ik)
    {
        infile >> x >> y >> z;
        if (!infile.good())
        {
            throw std::logic_error("Failed to read headwing k point from " + file_path);
        }
        kfrac_list.push_back({stod(x), stod(y), stod(z)});
    }
    return kfrac_list;
}

void read_headwing_input(const string &dir_path, bool need_wing)
{
    using namespace librpa_int;
    using namespace librpa_int::global;

    auto pds = librpa_int::api::get_dataset_instance(driver::h.get_c_handler());
    auto &mf = pds->mf;
    auto &velocity_matrix = pds->velocity_matrix;
    velocity_matrix.clear();
    struct MfRestore
    {
        MeanField &mf;
        MeanField original;
        bool active = false;

        explicit MfRestore(MeanField &mf_in) : mf(mf_in) {}
        void capture()
        {
            original = mf;
            active = true;
        }
        ~MfRestore()
        {
            if (active)
            {
                mf = std::move(original);
            }
        }
    } restore_mf(mf);

    const bool use_spinor_wfc = driver::driver_params.use_spinor_wfc;
    const string pyatb_dir = path_as_directory(dir_path) + "pyatb_librpa_df/";
    const string pyatb_velocity = pyatb_dir + "velocity_matrix";

    std::vector<Vector3_Order<double>> kfrac_headwing;
    int n_basis = 0;
    int n_states = 0;
    int n_spin = 0;

    std::vector<double> freq_weights;
    driver::h.get_imaginary_frequency_grids(driver::opts, pds->omegas_imagfreq, freq_weights);
    const auto &freqs = pds->tfg.get_freq_nodes();

    if (path_exists(pyatb_velocity.c_str()))
    {
        // Temporarily load the PyATB mean-field data into Dataset::mf for the
        // head/wing construction, then restore the SCF mean field before the
        // downstream GW path continues. The velocity/momentum matrix remains
        // separate from the mean-field data.
        if (mpi_comm_global_h.is_root())
        {
            std::cout << "Reading head/wing input from " << pyatb_dir << std::endl;
        }
        restore_mf.capture();
        read_scf_occ_eigenvalues(pyatb_dir + "band_out", mf, use_spinor_wfc);
        const int ret_eigenvec =
            read_eigenvector(pyatb_dir, mf, use_spinor_wfc, nullptr);
        if (ret_eigenvec != 0)
        {
            throw std::runtime_error("Failed to read pyatb head/wing eigenvectors from " + pyatb_dir);
        }
        read_velocity(pyatb_velocity, mf, velocity_matrix);
        kfrac_headwing = read_headwing_k_path_info(pyatb_dir + "k_path_info",
                                                   n_basis, n_states, n_spin);
        if (use_spinor_wfc)
        {
            if (n_basis % 2 != 0)
                throw std::runtime_error("Head/wing spinor basis size is not even");
            n_basis /= 2;
        }
        if (n_basis != mf.get_n_aos() || n_states != mf.get_n_states() ||
            n_spin != mf.get_n_spins())
        {
            throw std::runtime_error("Head/wing k_path_info dimensions are inconsistent with band_out");
        }
    }
    else
    {
        // ABACUS/FHI-aims package outputs use the SCF k grid for head/wing.
        // Only the velocity/momentum matrix needs an extra reader here.
        kfrac_headwing = pds->pbc.kfrac_list;
        n_basis = mf.get_n_aos();
        n_states = mf.get_n_states();
        n_spin = mf.get_n_spins();

        const string file_abacus = path_as_directory(dir_path) + "velocity_matrix";
        const string file_aims = path_as_directory(dir_path) + "mommat_ks_kpt_000001.dat";
        if (path_exists(file_abacus.c_str()))
        {
            read_velocity(file_abacus, mf, velocity_matrix);
        }
        else if (path_exists(file_aims.c_str()))
        {
            read_velocity_aims(mf, path_as_directory(dir_path), velocity_matrix);
        }
        else
        {
            throw std::runtime_error("Cannot find moment files for head/wing calculation");
        }
    }

    if (static_cast<int>(kfrac_headwing.size()) != mf.get_n_kpoints())
    {
        throw std::runtime_error("Head/wing k-point count is inconsistent with meanfield");
    }
    if (static_cast<int>(pds->pbc.kfrac_list.size()) != mf.get_n_kpoints())
    {
        throw std::runtime_error("SCF k-point list is inconsistent with meanfield");
    }
    for (int ik = 0; ik != mf.get_n_kpoints(); ++ik)
    {
        if (!nearly_same_kpoint(kfrac_headwing[ik], pds->pbc.kfrac_list[ik]))
        {
            std::ostringstream oss;
            oss << "Head/wing k-point " << ik
                << " is inconsistent with the SCF meanfield k grid";
            throw std::runtime_error(oss.str());
        }
    }

    const auto &headwing_basis_aux =
        driver::get_bool(driver::opts.use_shrink_abfs) ? pds->basis_aux_shrink : pds->basis_aux;
    if (!headwing_basis_aux.initialized())
        throw std::runtime_error("Head/wing auxiliary basis is not initialized");
    pds->p_headwing = std::make_unique<diele_func>(
        mf, velocity_matrix, pds->pbc.kfrac_list, pds->basis_wfc,
        headwing_basis_aux, freqs,
        n_basis, n_states, n_spin, headwing_basis_aux.nb_total, pds->pbc, pds->comm_h, pds->blacs_h);
    pds->p_headwing->use_2d_dielectric = driver::get_bool(driver::opts.use_2d_dielectric);
    pds->p_headwing->use_soc = mf.get_n_spinor() > 1;
    pds->p_headwing->debug = librpa_int::global::should_output(LIBRPA_VERBOSE_DEBUG);
    pds->p_headwing->init(driver::opts.sqrt_coulomb_threshold, pds->vq);
    pds->p_headwing->cal_head();
    pds->epsmacs_imagfreq = pds->p_headwing->get_head_vec();
    pds->omegas_imagfreq = freqs;
    pds->p_headwing->test_head();
    if (need_wing)
    {
        const auto &headwing_cs =
            driver::get_bool(driver::opts.use_shrink_abfs) ? pds->cs_data_shrink : pds->cs_data;
        pds->p_headwing->cal_wing(headwing_cs, driver::opts.sqrt_coulomb_threshold, pds->vq);
        if (librpa_int::global::should_output(LIBRPA_VERBOSE_DEBUG))
            pds->p_headwing->test_wing();
    }
}

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
    reader_structure(file_path);
}

void read_bz_sampling(const std::string &file_path)
{
    using namespace librpa_int;

    global::lib_printf_root("Reading Brillouin zone sampling file: %s\n", file_path.c_str());

    ifstream infile;
    infile.open(file_path);
    if (!infile.good())
        throw LIBRPA_RUNTIME_ERROR("Fail to open BZ sampling file " + file_path);

    int nk[3];
    for (int i = 0; i < 3; i++)
    {
        infile >> nk[i];
    }
    if (!infile.good() || nk[0] <= 0 || nk[1] <= 0 || nk[2] <= 0)
    {
        throw LIBRPA_RUNTIME_ERROR("Invalid BZ sampling k-grid in " + file_path);
    }
    const int nk_full = nk[0] * nk[1] * nk[2];

    int n_kpoints_scf, nk_ibz;
    infile >> n_kpoints_scf >> nk_ibz;
    if (!infile.good())
    {
        throw LIBRPA_RUNTIME_ERROR("Fail to read BZ sampling k-point counts from " + file_path);
    }
    if (n_kpoints_scf <= 0 || nk_ibz <= 0)
        throw LIBRPA_RUNTIME_ERROR("BZ sampling k-point counts must be positive");
    if (n_kpoints_scf > nk_full)
        throw LIBRPA_RUNTIME_ERROR("SCF k-point count exceeds the full BZ grid size");
    if (nk_ibz > n_kpoints_scf)
        throw LIBRPA_RUNTIME_ERROR("Coulomb IBZ k-point count exceeds the SCF k-point count");
    if (driver::n_kpoints > 0 && n_kpoints_scf != driver::n_kpoints)
    {
        throw LIBRPA_RUNTIME_ERROR(
            "BZ sampling SCF k-point count does not match band_out: "
            + std::to_string(n_kpoints_scf) + " != " + std::to_string(driver::n_kpoints));
    }

    std::vector<double> kvecs(3 * n_kpoints_scf);
    std::vector<double> kweights(n_kpoints_scf);
    std::vector<int> map_ibzk(n_kpoints_scf, -1);
    std::vector<int> ibz_label_to_rep(nk_ibz, -1);
    std::vector<int> ibz_representatives;
    double weight_sum = 0.0;

    for (int i = 0; i != n_kpoints_scf; i++)
    {
        int ik_read, ik_ibz, ik_rep;
        double kfrac_x, kfrac_y, kfrac_z;
        infile >> ik_read >> kweights[i];
        infile >> kfrac_x >> kfrac_y >> kfrac_z;
        infile >> kvecs[3 * i] >> kvecs[3 * i + 1] >> kvecs[3 * i + 2];
        infile >> ik_ibz >> ik_rep;
        if (!infile.good())
        {
            throw LIBRPA_RUNTIME_ERROR(
                "Fail to read BZ sampling k-point row " + std::to_string(i + 1)
                + " from " + file_path);
        }
        if (ik_read != i + 1)
        {
            throw LIBRPA_RUNTIME_ERROR("BZ sampling k-point index does not match row order");
        }
        if (!std::isfinite(kweights[i]) || kweights[i] < 0.0
            || !std::isfinite(kfrac_x) || !std::isfinite(kfrac_y) || !std::isfinite(kfrac_z)
            || !std::isfinite(kvecs[3 * i])
            || !std::isfinite(kvecs[3 * i + 1])
            || !std::isfinite(kvecs[3 * i + 2]))
        {
            throw LIBRPA_RUNTIME_ERROR("BZ sampling k-point row contains an invalid number");
        }
        if (ik_ibz <= 0 || ik_ibz > nk_ibz)
        {
            throw LIBRPA_RUNTIME_ERROR("BZ sampling IBZ index out of range");
        }
        if (ik_rep <= 0 || ik_rep > n_kpoints_scf)
        {
            throw LIBRPA_RUNTIME_ERROR("BZ sampling representative k-point index out of range");
        }
        map_ibzk[i] = ik_rep - 1;
        auto &label_rep = ibz_label_to_rep[static_cast<std::size_t>(ik_ibz - 1)];
        if (label_rep < 0)
        {
            label_rep = map_ibzk[i];
        }
        else if (label_rep != map_ibzk[i])
        {
            throw LIBRPA_RUNTIME_ERROR(
                "BZ sampling irreducible Coulomb k-point label maps to multiple representatives");
        }
        if (std::find(ibz_representatives.cbegin(), ibz_representatives.cend(), map_ibzk[i])
            == ibz_representatives.cend())
        {
            ibz_representatives.emplace_back(map_ibzk[i]);
        }
        weight_sum += kweights[i];
    }
    infile.close();

    if (std::abs(weight_sum - 1.0) > kBzSamplingWeightSumTol)
    {
        throw LIBRPA_RUNTIME_ERROR(
            "BZ sampling SCF k-point weights do not sum to 1: "
            + std::to_string(weight_sum));
    }
    if (ibz_representatives.size() != static_cast<std::size_t>(nk_ibz))
    {
        throw LIBRPA_RUNTIME_ERROR(
            "BZ sampling representative count does not match Coulomb IBZ count");
    }
    if (std::find(ibz_label_to_rep.cbegin(), ibz_label_to_rep.cend(), -1)
        != ibz_label_to_rep.cend())
    {
        throw LIBRPA_RUNTIME_ERROR(
            "BZ sampling does not contain every irreducible Coulomb k-point label");
    }

    auto pds = api::get_dataset_instance(driver::h);
    auto &pbc = pds->pbc;
    if (n_kpoints_scf < nk_full)
    {
        if (!any_symmetry_speedup_enabled())
        {
            throw LIBRPA_RUNTIME_ERROR(
                "BZ sampling contains a symmetry-reduced SCF k-point list; "
                "switch on use_symmetry_exx, use_symmetry_gw, or use_symmetry_rpa to use this input");
        }
        auto &symmetry_context = pds->symmetry_context;
        if (symmetry_context.rspace_operations.empty())
        {
            throw LIBRPA_RUNTIME_ERROR(
                "BZ sampling contains a symmetry-reduced SCF k-point list, "
                "but no symmetry operations were loaded from stru_out");
        }
    }

    pbc.set_kgrids_kvec(nk[0], nk[1], nk[2], kvecs);
    pbc.set_ibz_mapping(map_ibzk, {}, kweights);
    driver::ibz_kpoints = pbc.klist_coul;
    driver::n_ibz_kpoints = static_cast<int>(driver::ibz_kpoints.size());
}

void read_basis(const std::string &file_path)
{
    reader_basis(file_path);
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

    const int nk_ibz = pbc.klist_coul.size();

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
            const auto qvec = pbc.klist_coul[iq];

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
            const auto qvec = pbc.klist_coul[iq];
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

static bool sinvS_file_has_v1_marker(const std::string &file_path)
{
    ifstream infile(file_path, std::ios::in | std::ios::binary);
    if (!infile.good())
    {
        throw LIBRPA_RUNTIME_ERROR("Failed to open " + file_path);
    }
    std::int32_t marker = 0;
    infile.read(reinterpret_cast<char *>(&marker), sizeof(marker));
    return infile.good() && marker == READER_SHRINK_SINVS_V1_MARKER;
}

static std::streamoff checked_streamoff_from_i64(const std::int64_t value,
                                                const std::string &context)
{
    if (value < 0 ||
        static_cast<unsigned long long>(value) >
            static_cast<unsigned long long>(std::numeric_limits<std::streamoff>::max()))
    {
        throw LIBRPA_RUNTIME_ERROR(context + ": invalid file offset");
    }
    return static_cast<std::streamoff>(value);
}

static std::size_t checked_sinvS_payload_bytes(const std::int32_t nrow,
                                               const std::int32_t ncol,
                                               const std::string &file_path)
{
    if (nrow <= 0 || ncol <= 0)
    {
        throw LIBRPA_RUNTIME_ERROR(file_path + ": invalid shrink_sinvS v1 block dimensions");
    }
    const auto max_count =
        std::numeric_limits<std::size_t>::max() / sizeof(std::complex<double>);
    const auto nrow_size = static_cast<std::size_t>(nrow);
    const auto ncol_size = static_cast<std::size_t>(ncol);
    if (nrow_size > max_count / ncol_size)
    {
        throw LIBRPA_RUNTIME_ERROR(file_path + ": shrink_sinvS v1 block is too large");
    }
    return nrow_size * ncol_size * sizeof(std::complex<double>);
}

static int handle_sinvS_v1_file(const std::string &file_path,
                                std::map<Vector3_Order<double>, ComplexMatrix> &sinvS)
{
    struct Record
    {
        std::int32_t iq = 0;
        std::int32_t nrow_total = 0;
        std::int32_t ncol_total = 0;
        std::int32_t begin_row = 0;
        std::int32_t end_row = 0;
        std::int32_t begin_col = 0;
        std::int32_t end_col = 0;
        double weight = 0.0;
        std::int64_t offset = 0;
    };

    auto pds = librpa_int::api::get_dataset_instance(driver::h.get_c_handler());
    auto &pbc = pds->pbc;
    const int nk_ibz = pbc.klist_coul.size();

    ifstream infile(file_path, std::ios::in | std::ios::binary);
    if (!infile.good())
    {
        return 1;
    }

    std::int32_t marker = 0;
    std::int32_t nrecords_i32 = 0;
    infile.read(reinterpret_cast<char *>(&marker), sizeof(marker));
    infile.read(reinterpret_cast<char *>(&nrecords_i32), sizeof(nrecords_i32));
    if (!infile.good() || marker != READER_SHRINK_SINVS_V1_MARKER || nrecords_i32 < 0)
    {
        return 2;
    }

    infile.seekg(0, std::ios::end);
    const auto end_pos = infile.tellg();
    if (end_pos == std::streampos(-1))
    {
        return 3;
    }
    const auto file_size = static_cast<std::streamoff>(end_pos);
    infile.seekg(2 * static_cast<std::streamoff>(sizeof(std::int32_t)), std::ios::beg);

    std::vector<Record> records(static_cast<std::size_t>(nrecords_i32));
    for (auto &record: records)
    {
        infile.read(reinterpret_cast<char *>(&record.iq), sizeof(record.iq));
        infile.read(reinterpret_cast<char *>(&record.nrow_total), sizeof(record.nrow_total));
        infile.read(reinterpret_cast<char *>(&record.ncol_total), sizeof(record.ncol_total));
        infile.read(reinterpret_cast<char *>(&record.begin_row), sizeof(record.begin_row));
        infile.read(reinterpret_cast<char *>(&record.end_row), sizeof(record.end_row));
        infile.read(reinterpret_cast<char *>(&record.begin_col), sizeof(record.begin_col));
        infile.read(reinterpret_cast<char *>(&record.end_col), sizeof(record.end_col));
        infile.read(reinterpret_cast<char *>(&record.weight), sizeof(record.weight));
        infile.read(reinterpret_cast<char *>(&record.offset), sizeof(record.offset));
        if (!infile.good())
        {
            return 4;
        }
    }

    for (const auto &record: records)
    {
        const int iq = record.iq - 1;
        if (iq < 0 || iq >= nk_ibz)
        {
            return 5;
        }
        if (record.begin_row < 1 || record.begin_col < 1 ||
            record.end_row < record.begin_row || record.end_col < record.begin_col ||
            record.end_row > record.nrow_total || record.end_col > record.ncol_total)
        {
            return 6;
        }
        const auto nrow_block = record.end_row - record.begin_row + 1;
        const auto ncol_block = record.end_col - record.begin_col + 1;
        const auto bytes = checked_sinvS_payload_bytes(nrow_block, ncol_block, file_path);
        const auto offset = checked_streamoff_from_i64(record.offset, file_path);
        if (file_size - offset < static_cast<std::streamoff>(bytes))
        {
            return 7;
        }

        std::vector<std::complex<double>> tmp(
            static_cast<std::size_t>(nrow_block) * static_cast<std::size_t>(ncol_block));
        infile.seekg(offset, std::ios::beg);
        infile.read(reinterpret_cast<char *>(tmp.data()), static_cast<std::streamsize>(bytes));
        if (!infile.good())
        {
            return 8;
        }

        const auto qvec = pbc.klist_coul[iq];
        if (!sinvS.count(qvec))
        {
            sinvS[qvec].create(record.nrow_total, record.ncol_total);
        }
        if (sinvS[qvec].nr != record.nrow_total || sinvS[qvec].nc != record.ncol_total)
        {
            return 9;
        }

        for (int i = 0; i != nrow_block; ++i)
        {
            for (int j = 0; j != ncol_block; ++j)
            {
                sinvS[qvec](record.begin_row - 1 + i, record.begin_col - 1 + j) =
                    tmp[static_cast<std::size_t>(i) * static_cast<std::size_t>(ncol_block) +
                        static_cast<std::size_t>(j)];
            }
        }
    }

    return 0;
}

void read_ri_shrink(const string &dir_path)
{
    using librpa_int::global::mpi_comm_global_h;
    using librpa_int::global::profiler;
    using driver::driver_params;

    auto pds = librpa_int::api::get_dataset_instance(driver::h.get_c_handler());
    const auto &abf = pds->basis_aux;

    if (mpi_comm_global_h.is_root())
    {
        std::cout << "iatom & large Nabfs: " << std::endl;
        int I = 0;
        for (auto &mu : abf.get_atom_nbs())
        {
            std::cout << I << "," << mu << std::endl;
            ++I;
        }
    }

    const auto shrink_basis_path =
        librpa_int::join_path(driver_params.input_dir, driver_params.fn_basis_aux_shrink);
    const auto legacy_shrink_basis_path =
        librpa_int::join_path(driver_params.input_dir, "basis_out_shrink");
    const auto legacy_backup_basis_path =
        librpa_int::join_path(driver_params.input_dir, "basis_out.shrink_backup");
    if (librpa_int::path_exists(shrink_basis_path.c_str()))
    {
        reader_basis_aux_shrink(shrink_basis_path);
    }
    else if (librpa_int::path_exists(legacy_shrink_basis_path.c_str()))
    {
        reader_basis_aux_shrink(legacy_shrink_basis_path);
    }
    else if (librpa_int::path_exists(legacy_backup_basis_path.c_str()))
    {
        reader_basis_aux_shrink(legacy_backup_basis_path);
    }
    else
    {
        pds->basis_aux_shrink.set(read_aux_basis_from_Cs(
            driver_params.input_dir, driver_params.prefix_lri_coeff_shrink));
    }
    pds->desc_abf_shrink.reset_handler(pds->blacs_h);
    pds->desc_abf_shrink.init_1b1p(pds->basis_aux_shrink.nb_total,
                                   pds->basis_aux_shrink.nb_total, 0, 0);

    profiler.start("read_Cs_shrink");
    read_Cs_evenly_distribute(driver_params.input_dir, driver_params.cs_threshold,
                              mpi_comm_global_h.myid, mpi_comm_global_h.nprocs,
                              driver_params.prefix_lri_coeff_shrink,
                              driver_params.version_lri_reader);
    profiler.stop("read_Cs_shrink");

    profiler.start("read_shrink_sinvS_fold", "Load shrink transformation");
    pds->sinvS.clear();
    read_shrink_sinvS(driver_params.input_dir, driver_params.prefix_shrink_sinvS, pds->sinvS);

    if (!pds->sinvS.empty())
    {
        const auto &first_sinvS = pds->sinvS.begin()->second;
        if (static_cast<size_t>(first_sinvS.nr) != pds->basis_aux_shrink.nb_total ||
            static_cast<size_t>(first_sinvS.nc) != pds->basis_aux.nb_total)
        {
            throw std::runtime_error("shrink_sinvS dimensions are inconsistent with auxiliary bases");
        }
    }

    if (mpi_comm_global_h.is_root())
    {
        std::cout << "iatom & small Nabfs: " << std::endl;
        int I = 0;
        for (auto &mu : pds->basis_aux_shrink.get_atom_nbs())
        {
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

    size_t vq_discard = 0;
    auto files = librpa_int::discover_files_with_prefix(dir_path, vq_fprefix);
    if (files.empty())
    {
        throw LIBRPA_RUNTIME_ERROR("No shrink_sinvS files found with prefix " + vq_fprefix +
                                  " under: " + dir_path);
    }

    profiler.start("handle_sinvS_file");
    for (const auto &file_path: files)
    {
        int retcode = 0;
        if (sinvS_file_has_v1_marker(file_path))
        {
            if (myid_global == 0)
            {
                cout << "sinvS: reader v1 binary files detected" << endl;
            }
            retcode = handle_sinvS_v1_file(file_path, sinvS);
        }
        else
        {
            const bool binary = check_coulomb_file_binary(file_path);
            if (myid_global == 0)
            {
                cout << "sinvS: " << (binary ? "Unformatted binary" : "ASCII")
                     << " legacy files detected" << endl;
            }
            retcode = handle_sinvS_file(file_path, sinvS, binary);
        }
        if (retcode != 0)
        {
            lib_printf(LIBRPA_VERBOSE_CRITICAL, "Error encountered when reading %s, return code %d",
                       file_path.c_str(), retcode);
            throw LIBRPA_RUNTIME_ERROR("Failed to read shrink_sinvS file " + file_path +
                                      ", return code " + std::to_string(retcode));
        }
    }
    profiler.stop("handle_sinvS_file");
    return vq_discard;
}
