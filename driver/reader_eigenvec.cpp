#include "reader_eigenvec.h"

#include <dirent.h>

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdint>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/io/fs.h"
#include "../src/io/global_io.h"
#include "driver.h"

namespace
{

constexpr std::int32_t READER_EIGENVEC_V1_MARKER = -12345679;
constexpr std::int32_t EIGENVEC_V1_KIND_COMPLEX_DOUBLE = 28;

static_assert(sizeof(std::complex<double>) == 2 * sizeof(double),
              "KS eigenvector v1 expects std::complex<double> as two doubles");

struct KpointBlock
{
    std::int32_t ik;
    std::int64_t offset;
};

struct WfcShape
{
    int nspin;
    int nsoc;
    int nband;
    int nao;
    int nkpoints;
    bool use_spinor;
};

int check_KS_file_version(const std::string &file_path)
{
    std::ifstream infile(file_path, std::ios::in | std::ios::binary);
    if (!infile.good()) return false;
    std::int32_t marker = 0;
    infile.read(reinterpret_cast<char *>(&marker), sizeof(marker));
    // Legacy text file
    if (!infile.good() || marker > 0) return 0;
    // Version 1 binary
    if (marker == READER_EIGENVEC_V1_MARKER) return 1;
    throw std::runtime_error("Unsupported KS eigenvector file marker: " + std::to_string(marker) +
                             " in " + file_path);
}

template <typename T>
bool read_one(std::ifstream &infile, T &value)
{
    infile.read(reinterpret_cast<char *>(&value), sizeof(value));
    return infile.good();
}

std::size_t wfc_index(const WfcShape &shape, const int is, const int isoc, const int ib,
                      const int iw)
{
    const auto nbao = static_cast<std::size_t>(shape.nband) * shape.nao;
    const auto n = static_cast<std::size_t>(shape.nsoc) * nbao;
    return static_cast<std::size_t>(is) * n + static_cast<std::size_t>(isoc) * nbao +
           static_cast<std::size_t>(ib) * shape.nao + static_cast<std::size_t>(iw);
}

bool selected_ik(const std::vector<int> *iks_selected, const int ik)
{
    return iks_selected == nullptr ||
           std::find(iks_selected->cbegin(), iks_selected->cend(), ik) != iks_selected->cend();
}

bool selected_driver_ik(const int ik)
{
    if (!driver::get_bool(driver::opts.use_kpara_scf_eigvec)) return true;
    return std::find(driver::iks_eigvec_this.cbegin(), driver::iks_eigvec_this.cend(), ik) !=
           driver::iks_eigvec_this.cend();
}

std::vector<std::string> eigenvector_files(const std::string &dir_path)
{
    std::vector<std::string> files;
    DIR *dir = opendir(dir_path.c_str());
    if (dir == nullptr) return files;

    while (auto *ptr = readdir(dir))
    {
        const std::string name(ptr->d_name);
        if (name.find(driver::driver_params.prefix_eigvecs_scf) == 0)
        {
            files.emplace_back(dir_path + name);
        }
    }
    closedir(dir);
    std::sort(files.begin(), files.end());
    return files;
}

template <typename ShouldReadIk, typename StoreIk>
int read_legacy_text_file(const std::string &file_path, const WfcShape &shape,
                          ShouldReadIk should_read_ik, StoreIk store_ik,
                          const LegacyTextWfcOrder text_order =
                              LegacyTextWfcOrder::BasisSpinorBandSpin)
{
    std::ifstream infile(file_path);
    if (!infile.good()) return 1;

    const auto n = static_cast<std::size_t>(shape.nspin) * shape.nsoc * shape.nband * shape.nao;
    std::vector<double> re(n);
    std::vector<double> im(n);

    std::string rvalue, ivalue, kstr;
    while (infile >> kstr)
    {
        const int ik = std::stoi(kstr) - 1;
        const bool keep_ik = should_read_ik(ik);
        std::fill(re.begin(), re.end(), 0.0);
        std::fill(im.begin(), im.end(), 0.0);

        if (text_order == LegacyTextWfcOrder::BasisSpinorBandSpin)
        {
            for (int iw = 0; iw != shape.nao; ++iw)
            {
                for (int isoc = 0; isoc != shape.nsoc; ++isoc)
                {
                    for (int ib = 0; ib != shape.nband; ++ib)
                    {
                        for (int is = 0; is != shape.nspin; ++is)
                        {
                            if (!(infile >> rvalue >> ivalue)) return 1;
                            if (!keep_ik) continue;
                            const auto dst = wfc_index(shape, is, isoc, ib, iw);
                            re[dst] = std::stod(rvalue);
                            im[dst] = std::stod(ivalue);
                        }
                    }
                }
            }
        }
        else
        {
            for (int is = 0; is != shape.nspin; ++is)
            {
                for (int iwfc = 0; iwfc != shape.nao * shape.nsoc; ++iwfc)
                {
                    const int isoc = iwfc % shape.nsoc;
                    const int iw = iwfc / shape.nsoc;
                    for (int ib = 0; ib != shape.nband; ++ib)
                    {
                        if (!(infile >> rvalue >> ivalue)) return 1;
                        if (!keep_ik) continue;
                        const auto dst = wfc_index(shape, is, isoc, ib, iw);
                        re[dst] = std::stod(rvalue);
                        im[dst] = std::stod(ivalue);
                    }
                }
            }
        }
        if (keep_ik) store_ik(ik, re, im);
    }
    return 0;
}

template <typename ShouldReadIk, typename StoreIk>
int read_binary_v1_file(const std::string &file_path, const WfcShape &shape,
                        ShouldReadIk should_read_ik, StoreIk store_ik)
{
    std::ifstream infile(file_path, std::ios::in | std::ios::binary);
    if (!infile.good()) return 1;

    std::int32_t marker = 0;
    std::int32_t kind_raw = 0;
    std::int32_t nkpoints_file = 0;
    std::int32_t nspins_file = 0;
    std::int32_t nstates_file = 0;
    std::int32_t nbasis_wfc_file = 0;
    if (!read_one(infile, marker) || !read_one(infile, kind_raw) ||
        !read_one(infile, nkpoints_file) || !read_one(infile, nspins_file) ||
        !read_one(infile, nstates_file) || !read_one(infile, nbasis_wfc_file))
    {
        return 1;
    }

    const int nbasis_wfc = shape.nao * shape.nsoc;
    if (marker != READER_EIGENVEC_V1_MARKER || kind_raw != EIGENVEC_V1_KIND_COMPLEX_DOUBLE ||
        nkpoints_file < 0 || nspins_file != shape.nspin || nstates_file != shape.nband ||
        nbasis_wfc_file != nbasis_wfc)
    {
        return 1;
    }

    std::vector<KpointBlock> blocks(static_cast<std::size_t>(nkpoints_file));
    for (auto &block : blocks)
    {
        if (!read_one(infile, block.ik) || !read_one(infile, block.offset))
        {
            return 1;
        }
        if (block.offset < 0) return 1;
    }

    const auto n = static_cast<std::size_t>(shape.nspin) * shape.nsoc * shape.nband * shape.nao;
    std::vector<std::complex<double>> block_data(n);

    for (const auto &block : blocks)
    {
        const int ik = block.ik - 1;
        if (ik < 0 || ik >= shape.nkpoints) return 1;
        if (!should_read_ik(ik)) continue;

        infile.clear();
        infile.seekg(static_cast<std::streamoff>(block.offset), std::ios::beg);
        if (!infile.good()) return 1;

        infile.read(reinterpret_cast<char *>(block_data.data()),
                    static_cast<std::streamsize>(block_data.size() * sizeof(std::complex<double>)));
        if (!infile.good()) return 1;
        store_ik(ik, block_data);
    }
    return 0;
}

void set_driver_wfc_packed(const WfcShape &shape, const int ik,
                           const std::vector<std::complex<double>> &block_data)
{
    const int nbao = shape.nband * shape.nao;
    for (int is = 0; is != shape.nspin; ++is)
    {
        const auto *spin_block =
            block_data.data() + static_cast<std::size_t>(is) * shape.nsoc * nbao;
        if (shape.use_spinor)
        {
            assert(is == 0);
            driver::h.set_wfc_spinor_packed(ik, shape.nband, shape.nao, spin_block,
                                            spin_block + nbao);
        }
        else
        {
            driver::h.set_wfc_packed(is, ik, shape.nband, shape.nao, spin_block);
        }
    }
}

void set_driver_wfc(const WfcShape &shape, const int ik, const std::vector<double> &re,
                    const std::vector<double> &im)
{
    const int nbao = shape.nband * shape.nao;
    const int n = shape.nsoc * nbao;
    for (int is = 0; is != shape.nspin; ++is)
    {
        if (shape.use_spinor)
        {
            assert(is == 0);
            driver::h.set_wfc_spinor(ik, shape.nband, shape.nao, re.data(), im.data(),
                                     re.data() + nbao, im.data() + nbao);
        }
        else
        {
            driver::h.set_wfc(is, ik, shape.nband, shape.nao,
                              re.data() + static_cast<std::size_t>(is) * n,
                              im.data() + static_cast<std::size_t>(is) * n);
        }
    }
}

void set_meanfield_wfc_packed(librpa_int::MeanField &mf, const WfcShape &shape, const int ik,
                              const std::vector<std::complex<double>> &block_data)
{
    const int nbao = shape.nband * shape.nao;
    for (int is = 0; is != shape.nspin; ++is)
    {
        const auto *spin_block =
            block_data.data() + static_cast<std::size_t>(is) * shape.nsoc * nbao;
        if (shape.use_spinor)
        {
            assert(is == 0);
            auto &wfcs_up = mf.get_eigenvectors()[0][0][ik];
            auto &wfcs_dn = mf.get_eigenvectors()[0][1][ik];
            wfcs_up.create(shape.nband, shape.nao);
            wfcs_dn.create(shape.nband, shape.nao);
            std::copy(spin_block, spin_block + nbao, wfcs_up.c);
            std::copy(spin_block + nbao, spin_block + 2 * nbao, wfcs_dn.c);
        }
        else
        {
            auto &wfc = mf.get_eigenvectors()[is][0][ik];
            wfc.create(shape.nband, shape.nao);
            std::copy(spin_block, spin_block + nbao, wfc.c);
        }
    }
}

void set_meanfield_wfc(librpa_int::MeanField &mf, const WfcShape &shape, const int ik,
                       const std::vector<double> &re, const std::vector<double> &im)
{
    const int nbao = shape.nband * shape.nao;
    const int n = shape.nsoc * nbao;
    for (int is = 0; is != shape.nspin; ++is)
    {
        if (shape.use_spinor)
        {
            assert(is == 0);
            auto &wfcs_up = mf.get_eigenvectors()[0][0][ik];
            auto &wfcs_dn = mf.get_eigenvectors()[0][1][ik];
            wfcs_up.create(shape.nband, shape.nao);
            wfcs_dn.create(shape.nband, shape.nao);
            for (int i = 0; i != nbao; ++i)
            {
                wfcs_up.c[i] = std::complex<double>(re[i], im[i]);
                wfcs_dn.c[i] = std::complex<double>(re[nbao + i], im[nbao + i]);
            }
        }
        else
        {
            auto &wfc = mf.get_eigenvectors()[is][0][ik];
            wfc.create(shape.nband, shape.nao);
            for (int i = 0; i != nbao; ++i)
            {
                const auto src = static_cast<std::size_t>(is) * n + i;
                wfc.c[i] = std::complex<double>(re[src], im[src]);
            }
        }
    }
}

}  // namespace

int read_eigenvector(const std::string &dir_path)
{
    const WfcShape shape{driver::n_spins,   driver::n_spinor,
                         driver::n_states,  driver::n_basis_ao,
                         driver::n_kpoints, driver::driver_params.use_spinor_wfc};

    int files_read = 0;
    int version_first = -1;
    for (const auto &file_path : eigenvector_files(dir_path))
    {
        librpa_int::require_readable_file(file_path);
        const int version = check_KS_file_version(file_path);
        if (version_first < 0)
        {
            version_first = version;
            librpa_int::global::lib_printf_root("KS eigenvector reader: %s\n",
                                                version == 1 ? "binary v1" : "legacy text");
        }
        else
        {
            if (version != version_first)
                throw std::runtime_error(
                    "Versions across KS eigenvector files are inconsistent! "
                    "Version of first file " + std::to_string(version_first) +
                    " vs " + std::to_string(version) + " of " + file_path);
        }
        int ret = 1;
        switch (version)
        {
            case 0:
                ret = read_legacy_text_file(
                    file_path, shape, selected_driver_ik,
                    [&](const int ik, const std::vector<double> &re, const std::vector<double> &im)
                    { set_driver_wfc(shape, ik, re, im); });
                break;
            case 1:
                ret = read_binary_v1_file(
                    file_path, shape, selected_driver_ik,
                    [&](const int ik, const std::vector<std::complex<double>> &block_data)
                    { set_driver_wfc_packed(shape, ik, block_data); });
                break;
        }
        if (ret != 0) return ret;
        ++files_read;
    }
    return files_read == 0 ? -1 : 0;
}

int read_eigenvector(const std::string &dir_path, librpa_int::MeanField &mf, bool use_spinor_wfc,
                     const std::vector<int> *iks_selected)
{
    const WfcShape shape{mf.get_n_spins(), mf.get_n_spinor(),  mf.get_n_states(),
                         mf.get_n_aos(),   mf.get_n_kpoints(), use_spinor_wfc};

    int files_read = 0;
    bool printed_reader_version = false;
    for (const auto &file_path : eigenvector_files(dir_path))
    {
        librpa_int::require_readable_file(file_path);
        const int version = check_KS_file_version(file_path);
        if (!printed_reader_version)
        {
            librpa_int::global::lib_printf_root("KS eigenvector reader: %s\n",
                                                version == 1 ? "binary v1" : "legacy text");
            printed_reader_version = true;
        }
        const auto should_read_ik = [&](const int ik) { return selected_ik(iks_selected, ik); };
        const auto store_ik =
            [&](const int ik, const std::vector<double> &re, const std::vector<double> &im)
        { set_meanfield_wfc(mf, shape, ik, re, im); };
        const auto ret =
            version == 1
                ? read_binary_v1_file(
                      file_path, shape, should_read_ik,
                      [&](const int ik, const std::vector<std::complex<double>> &block_data)
                      { set_meanfield_wfc_packed(mf, shape, ik, block_data); })
                : read_legacy_text_file(file_path, shape, should_read_ik, store_ik);
        if (ret != 0) return ret;
        ++files_read;
    }
    return files_read == 0 ? -1 : 0;
}

int read_eigenvector(const std::string &dir_path, librpa_int::MeanField &mf, bool use_spinor_wfc,
                     const std::vector<int> &source_to_target_ik,
                     const std::vector<int> *source_iks_selected,
                     const LegacyTextWfcOrder text_order)
{
    const int n_target_kpoints = mf.get_n_kpoints();
    const int n_source_kpoints = static_cast<int>(source_to_target_ik.size());
    if (n_source_kpoints <= 0)
    {
        return 1;
    }
    for (const int ik_target : source_to_target_ik)
    {
        if (ik_target < -1 || ik_target >= n_target_kpoints)
        {
            return 1;
        }
    }

    const WfcShape shape{mf.get_n_spins(), mf.get_n_spinor(),  mf.get_n_states(),
                         mf.get_n_aos(),   n_source_kpoints,   use_spinor_wfc};

    std::vector<char> target_hits(static_cast<std::size_t>(n_target_kpoints), 0);
    const auto source_selected = [&](const int ik_source)
    {
        if (ik_source < 0 || ik_source >= n_source_kpoints) return false;
        if (source_to_target_ik[ik_source] < 0) return false;
        return selected_ik(source_iks_selected, ik_source);
    };
    const auto target_from_source = [&](const int ik_source)
    {
        return source_to_target_ik.at(static_cast<std::size_t>(ik_source));
    };

    int files_read = 0;
    bool printed_reader_version = false;
    for (const auto &file_path : eigenvector_files(dir_path))
    {
        const int version = check_KS_file_version(file_path);
        if (!printed_reader_version)
        {
            librpa_int::global::lib_printf_root("KS eigenvector reader: %s\n",
                                                version == 1 ? "binary v1" : "legacy text");
            printed_reader_version = true;
        }

        const auto store_text =
            [&](const int ik_source, const std::vector<double> &re,
                const std::vector<double> &im)
        {
            const int ik_target = target_from_source(ik_source);
            set_meanfield_wfc(mf, shape, ik_target, re, im);
            target_hits.at(static_cast<std::size_t>(ik_target)) = 1;
        };
        const auto store_packed =
            [&](const int ik_source, const std::vector<std::complex<double>> &block_data)
        {
            const int ik_target = target_from_source(ik_source);
            set_meanfield_wfc_packed(mf, shape, ik_target, block_data);
            target_hits.at(static_cast<std::size_t>(ik_target)) = 1;
        };

        const auto ret =
            version == 1
                ? read_binary_v1_file(file_path, shape, source_selected, store_packed)
                : read_legacy_text_file(file_path, shape, source_selected, store_text,
                                        text_order);
        if (ret != 0) return ret;
        ++files_read;
    }

    if (files_read == 0) return -1;
    if (source_iks_selected == nullptr)
    {
        for (const auto hit : target_hits)
        {
            if (!hit) return 1;
        }
    }
    return 0;
}
