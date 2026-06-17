// API headers
#include "librpa_enums.h"
#include "librpa_input.h"

// Standard headers
#include <complex>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

// Internal headers
#include "../io/global_io.h"
#include "../io/stl_io_helper.h"
#include "../math/matrix.h"
#include "../math/vector3_order.h"
#include "../utils/error.h"
#include "../utils/profiler.h"
#include "instance_manager.h"

// External headers and stubs
#ifdef LIBRPA_USE_LIBRI
#include <initializer_list>
#include <RI/global/Tensor.h>
#else
#include "../utils/libri_stub.h"
#endif

namespace
{

void mark_band_data_set(const librpa_int::dataset_ptr_t &pds)
{
    pds->is_band_data_set = true;
    pds->is_band_calc_done = false;
}

std::vector<std::vector<int>> parse_l_shells(const int natoms,
                                             const int *nshells,
                                             const int *l_shells)
{
    if (natoms < 0)
    {
        throw LIBRPA_RUNTIME_ERROR("Number of atoms for l-shell metadata must be non-negative");
    }
    if (natoms > 0 && nshells == nullptr)
    {
        throw LIBRPA_RUNTIME_ERROR("Missing per-atom shell counts for l-shell metadata");
    }

    std::vector<std::vector<int>> parsed(static_cast<std::size_t>(natoms));
    std::size_t offset = 0;
    for (int iat = 0; iat < natoms; ++iat)
    {
        if (nshells[iat] < 0)
        {
            throw LIBRPA_RUNTIME_ERROR("Number of l-shells per atom must be non-negative");
        }
        if (nshells[iat] > 0 && l_shells == nullptr)
        {
            throw LIBRPA_RUNTIME_ERROR("Missing l-shell metadata");
        }

        auto &atom_l_shells = parsed[static_cast<std::size_t>(iat)];
        atom_l_shells.reserve(static_cast<std::size_t>(nshells[iat]));
        for (int ishell = 0; ishell < nshells[iat]; ++ishell)
        {
            atom_l_shells.push_back(l_shells[offset]);
            ++offset;
        }
    }
    return parsed;
}

} // namespace

void librpa_set_scf_dimension(LibrpaHandler* h, int nspins, int nkpts, int nstates, int nbasis, int nspinor)
{
    using std::endl;
    using namespace librpa_int;
    using api::get_dataset_instance;
    using global::mpi_comm_global_h;
    using global::lib_printf;
    using global::profiler;
    using global::ofs_myid;

    const std::string tname = "api_set_scf_dimension";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);

    auto &meanfield = pds->mf;
    // Local dimensions, currently not used
    int st_ib = 0;
    int nb_local = nstates;
    int st_iao = 0;
    int nao_local = nbasis;
    meanfield.set(nspins, nkpts, nstates, nbasis, nspinor,
                  st_ib, nb_local, st_iao, nao_local);
    pds->comm_h.barrier();
    if (pds->comm_h.is_root())
    {
        lib_printf("Mean-field dimensions set:\n");
        lib_printf("| number of spins           : %d\n", meanfield.get_n_spins());
        lib_printf("| number of k-points        : %d\n", meanfield.get_n_kpoints());
        lib_printf("| number of bands           : %d\n", meanfield.get_n_bands());
        lib_printf("| number of NAOs            : %d\n", meanfield.get_n_aos());
        lib_printf("| number of spin components : %d\n", meanfield.get_n_spinor());
    }

    // Initialize global BLACS shape for eigenvectors
    // TODO: it would be more flexible to allow the user to force a shape.
    //       Currently the API does not support it, so we may have to adapt librpa_set_scf_dimension,
    //       or add a new API function.
    KPointBlacsProcessShape scfk_blacs_shape;
    scfk_blacs_shape.favor_square_blacs_grid = true;
    auto &kbctxt = pds->scfk_blacs_ctxt;
    kbctxt.init(scfk_blacs_shape, pds->comm_h.comm, nkpts);
    const auto &process_shape = kbctxt.process_shape();
    if (pds->comm_h.is_root())
    {
        lib_printf("Internal two-level (SCF k-points/matrix block) parallelization for eigenvectors set:\n");
        lib_printf("| processes per k-point      : %d\n", process_shape.nprocs_blacs);
        lib_printf("| processes per matrix block : %d\n", process_shape.nprocs_kpoint);
    }
    ofs_myid << "Local k-point indices: " << kbctxt.kpoints_local() << endl;
    pds->comm_h.barrier();
    // As the two-level communicator handlers are settled down,
    // the wave function array descriptors are fixed.
    ofs_myid << "k-point communicator : " << kbctxt.comm_kpoint_h.str() << endl;
    ofs_myid << "BLACS communicator   : " << kbctxt.comm_blacs_h.str() << endl;

    pds->desc_wfc_kb = kbctxt.create_array_desc(nbasis, nstates);
    pds->desc_wfc_kb_full = kbctxt.create_array_desc(nbasis, nstates, nbasis, nstates);
    ofs_myid << "desc_wfc_kb      : " << pds->desc_wfc_kb.info_desc() << endl;
    ofs_myid << "desc_wfc_kb_full : " << pds->desc_wfc_kb_full.info_desc() << endl;

    profiler.stop(tname);
}

void librpa_set_wg_ekb_efermi(LibrpaHandler* h, int nspins, int nkpts, int nstates,
                              const double* wg, const double* ekb, double efermi)
{
    using librpa_int::global::profiler;
    using librpa_int::global::lib_printf;

    const std::string tname = "api_set_wg_ekb_efermi";
    profiler.start(tname);
    auto pds = librpa_int::api::get_dataset_instance(h);
    auto &meanfield = pds->mf;

    meanfield.get_efermi() = efermi;
    auto& eskb = meanfield.get_eigenvals();
    auto& swg = meanfield.get_weight();
    int length_kb = nkpts * nstates;
    for (int is = 0; is != nspins; is++)
    {
        memcpy(eskb[is].c, ekb + length_kb * is, length_kb * sizeof(double));
        memcpy(swg[is].c, wg + length_kb * is, length_kb * sizeof(double));
        // wg[is](k_index, i) = stod(ws) / n_kpoints; // different with abacus!
        swg[is] *= (1.0 / nkpts);
    }

    double emin, emax;
    pds->mf.get_E_min_max(emin, emax);

    pds->comm_h.barrier();
    if (pds->comm_h.is_root())
    {
        lib_printf("Mean-field eigenvalues and occupation numbers set:\n");
        lib_printf("| Minimal transition energy (Ha): %18.10f\n", emin);
        lib_printf("| Maximal transition energy (Ha): %18.10f\n", emax);
        lib_printf("| Fermi level               (Ha): %18.10f\n", efermi);
    }
    pds->comm_h.barrier();

    profiler.stop(tname);
}

void librpa_set_wfc(LibrpaHandler* h, int ispin, int ik, int nstates_local, int nbasis_local,
                    const double* wfc_real, const double* wfc_imag)
{
    using librpa_int::global::profiler;

    const std::string tname = "api_set_wfc";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    auto &meanfield = pds->mf;

    auto& wfc = meanfield.get_eigenvectors()[ispin][0][ik];
    wfc.create(nstates_local, nbasis_local);
    const size_t n = meanfield.get_n_bands() * meanfield.get_n_aos();
    for (size_t i = 0; i < n; i++)
    {
        wfc.c[i] = std::complex<double>(wfc_real[i], wfc_imag[i]);
    }
    // std::cout << "Maxabs: " << wfc.get_max_abs() << std::endl;
    librpa_int::global::ofs_myid
        << "Wave-function set : ispin = " << ispin << " ik = " << ik
        << " nstates_local = " << nstates_local << " nbasis_local = " << nbasis_local
        << std::endl;

    profiler.stop(tname);
}

void librpa_set_wfc_spinor(LibrpaHandler* h, int ik, int nstates_local, int nbasis_local,
                           const double* wfc_up_real, const double* wfc_up_imag,
                           const double* wfc_dn_real, const double* wfc_dn_imag)
{
    using librpa_int::global::profiler;

    const std::string tname = "api_set_wfc_spinor";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    auto &meanfield = pds->mf;

    if (meanfield.get_n_spinor() != 2)
        throw LIBRPA_RUNTIME_ERROR("spinor wfc set requires n_spinor initialized to 2");

    auto& wfcs_up = meanfield.get_eigenvectors()[0][0][ik];
    auto& wfcs_dn = meanfield.get_eigenvectors()[0][1][ik];
    wfcs_up.create(nstates_local, nbasis_local);
    wfcs_dn.create(nstates_local, nbasis_local);
    const size_t n = meanfield.get_n_bands() * meanfield.get_n_aos();
    for (size_t i = 0; i < n; i++)
    {
        wfcs_up.c[i] = std::complex<double>(wfc_up_real[i], wfc_up_imag[i]);
        wfcs_dn.c[i] = std::complex<double>(wfc_dn_real[i], wfc_dn_imag[i]);
    }
    // std::cout << "Maxabs: " << wfc.get_max_abs() << std::endl;
    librpa_int::global::ofs_myid
        << "Spinor wave-function set : ik = " << ik
        << " nstates_local = " << nstates_local << " nbasis_local = " << nbasis_local
        << std::endl;

    profiler.stop(tname);
}

void librpa_set_wfc_packed(LibrpaHandler* h, int ispin, int ik, int nstates_local, int nbasis_local,
                           const double* wfc_ri)
{
    using librpa_int::global::profiler;

    const std::string tname = "api_set_wfc_packed";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    auto &meanfield = pds->mf;

    auto& wfc = meanfield.get_eigenvectors()[ispin][0][ik];
    wfc.create(nstates_local, nbasis_local);
    const size_t n = meanfield.get_n_bands() * meanfield.get_n_aos();
    for (size_t i = 0; i < n; i++)
    {
        wfc.c[i] = std::complex<double>(wfc_ri[2*i], wfc_ri[2*i+1]);
    }
    // std::cout << "Maxabs: " << wfc.get_max_abs() << std::endl;
    librpa_int::global::ofs_myid
        << "Wave-function set : ispin = " << ispin << " ik = " << ik
        << " nstates_local = " << nstates_local << " nbasis_local = " << nbasis_local
        << std::endl;

    profiler.stop(tname);
}

void librpa_set_wfc_spinor_packed(LibrpaHandler* h, int ik, int nstates_local, int nbasis_local,
                                  const double* wfc_up_ri, const double* wfc_dn_ri)
{
    using librpa_int::global::profiler;

    const std::string tname = "api_set_wfc_spinor_packed";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    auto &meanfield = pds->mf;

    if (meanfield.get_n_spinor() != 2)
        throw LIBRPA_RUNTIME_ERROR("spinor wfc set requires n_spinor initialized to 2");

    auto& wfcs_up = meanfield.get_eigenvectors()[0][0][ik];
    auto& wfcs_dn = meanfield.get_eigenvectors()[0][1][ik];
    wfcs_up.create(nstates_local, nbasis_local);
    wfcs_dn.create(nstates_local, nbasis_local);
    const size_t n = meanfield.get_n_bands() * meanfield.get_n_aos();
    for (size_t i = 0; i < n; i++)
    {
        wfcs_up.c[i] = std::complex<double>(wfc_up_ri[2*i], wfc_up_ri[2*i+1]);
        wfcs_dn.c[i] = std::complex<double>(wfc_dn_ri[2*i], wfc_dn_ri[2*i+1]);
    }
    // std::cout << "Maxabs: " << wfc.get_max_abs() << std::endl;
    librpa_int::global::ofs_myid
        << "Spinor wave-function set : ik = " << ik
        << " nstates_local = " << nstates_local << " nbasis_local = " << nbasis_local
        << std::endl;

    profiler.stop(tname);
}

static void set_ao_basis(librpa_int::AtomicBasis& ab, const int natoms, const size_t* nbs,
                         const int* nshells, const int* l_shells,
                         const std::map<librpa_int::atom_t, int>& types,
                         std::map<int, librpa_int::SpeciesBasisLayout>& layouts)
{
    std::vector<size_t> v_nbs(natoms);
    for (int i = 0; i < natoms; i++) v_nbs[i] = librpa_int::as_size(nbs[i]);

    ab.set(v_nbs);
    if (nshells != nullptr || l_shells != nullptr)
    {
        ab.set_l_shells(parse_l_shells(natoms, nshells, l_shells));
    }
    condense_species_basis_layouts(ab, types, layouts);
}

void librpa_set_ao_basis_wfc(LibrpaHandler* h,
                             const int natoms,
                             const size_t *nbs_wfc,
                             const int *nshells,
                             const int *l_shells)
{
    using librpa_int::global::lib_printf;
    using librpa_int::global::profiler;

    const std::string tname = "api_set_ao_basis_wfc";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    set_ao_basis(pds->basis_wfc, natoms, nbs_wfc, nshells, l_shells, pds->atoms.types, pds->basis_wfc_layouts);
    pds->desc_wfc.reset_handler(pds->blacs_h);
    const auto n = pds->basis_wfc.nb_total;
    pds->desc_wfc.init_1b1p(n, n, 0, 0);

    pds->comm_h.barrier();
    if (pds->comm_h.is_root())
    {
        lib_printf("Wave-function basis functions set:\n");
        lib_printf("| total number of basis: %lu\n", pds->basis_wfc.nb_total);
    }
    pds->comm_h.barrier();

    profiler.stop(tname);
}

void librpa_set_ao_basis_aux(LibrpaHandler* h,
                             int natoms,
                             const size_t *nbs_aux,
                             const int *nshells,
                             const int *l_shells)
{
    using librpa_int::global::lib_printf;
    using librpa_int::global::profiler;
    using librpa_int::global::ofs_myid;

    const std::string tname = "api_set_ao_basis_aux";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    set_ao_basis(pds->basis_aux, natoms, nbs_aux, nshells, l_shells, pds->atoms.types, pds->basis_aux_layouts);

    // After auxiliary basis is set, we can initialize the global (continous) array descriptor for N_abf size basis.
    pds->desc_abf.reset_handler(pds->blacs_h);
    const auto n = pds->basis_aux.nb_total;
    pds->desc_abf.init_1b1p(n, n, 0, 0);

    pds->comm_h.barrier();
    if (pds->comm_h.is_root())
    {
        lib_printf("Auxiliary basis functions set:\n");
        lib_printf("| total number of basis: %lu\n", pds->basis_aux.nb_total);
    }
    pds->comm_h.barrier();

    profiler.stop(tname);
}

void librpa_set_ao_basis_aux_shrink(LibrpaHandler* h,
                                    int natoms,
                                    const size_t *nbs_aux_shrink,
                                    const int *nshells,
                                    const int *l_shells)
{
    using librpa_int::global::lib_printf;
    using librpa_int::global::profiler;

    const std::string tname = "api_set_ao_basis_aux_shrink";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    set_ao_basis(pds->basis_aux_shrink, natoms, nbs_aux_shrink, nshells, l_shells, pds->atoms.types, pds->basis_aux_shrink_layouts);

    pds->desc_abf_shrink.reset_handler(pds->blacs_h);
    const auto n = pds->basis_aux_shrink.nb_total;
    pds->desc_abf_shrink.init_1b1p(n, n, 0, 0);

    pds->comm_h.barrier();
    if (pds->comm_h.is_root())
    {
        lib_printf("Shrink auxiliary basis functions set:\n");
        lib_printf("| total number of basis: %lu\n", pds->basis_aux_shrink.nb_total);
    }
    pds->comm_h.barrier();

    profiler.stop(tname);
}

namespace
{

bool is_valid_basis_order(const LibrpaAngularOrder order)
{
    switch (order)
    {
    case LIBRPA_ANGULAR_ORDER_NATURAL:
    case LIBRPA_ANGULAR_ORDER_ABS_PM:
    case LIBRPA_ANGULAR_ORDER_OPENMX:
    case LIBRPA_ANGULAR_ORDER_PYSCF:
        return true;
    case LIBRPA_ANGULAR_ORDER_UNSET:
        return false;
    }
    return false;
}

bool is_valid_rsh_coeff(const LibrpaRshCoeff coeff)
{
    switch (coeff)
    {
    case LIBRPA_RSH_COEFF_1_M:
    case LIBRPA_RSH_COEFF_M_1:
        return true;
    case LIBRPA_RSH_COEFF_UNSET:
        return false;
    }
    return false;
}

bool is_valid_bloch_phase(const int bloch_phase)
{
    return bloch_phase == 1 || bloch_phase == -1;
}

bool is_valid_bloch_ratom(const int bloch_ratom)
{
    return bloch_ratom == 1 || bloch_ratom == 0 || bloch_ratom == -1;
}

} // namespace

void librpa_set_basis_convention(LibrpaHandler* h, int bloch_phase, int bloch_ratom,
                                 LibrpaAngularOrder order,
                                 LibrpaRshCoeff nega_m,
                                 LibrpaRshCoeff posi_m)
{
    using librpa_int::global::lib_printf;
    using librpa_int::global::profiler;

    const std::string tname = "api_set_basis_convention";
    profiler.start(tname);

    if (!is_valid_bloch_phase(bloch_phase))
    {
        throw LIBRPA_RUNTIME_ERROR("Basis Bloch phase must be either 1 or -1");
    }
    if (!is_valid_bloch_ratom(bloch_ratom))
    {
        throw LIBRPA_RUNTIME_ERROR("Basis Bloch atom-position coefficient must be -1, 0, or 1");
    }
    if (!is_valid_basis_order(order))
    {
        throw LIBRPA_RUNTIME_ERROR("Invalid basis angular ordering convention");
    }
    if (!is_valid_rsh_coeff(nega_m) || !is_valid_rsh_coeff(posi_m))
    {
        throw LIBRPA_RUNTIME_ERROR("Invalid real-spherical-harmonic coefficient convention");
    }

    auto pds = librpa_int::api::get_dataset_instance(h);
    pds->basis_convention = {bloch_phase, bloch_ratom, order, nega_m, posi_m};

    pds->comm_h.barrier();
    if (pds->comm_h.is_root())
    {
        lib_printf("Basis convention set:\n");
        lib_printf("| Bloch phase      : %d\n", pds->basis_convention.bloch_phase);
        lib_printf("| Bloch r_atom     : %d\n", pds->basis_convention.bloch_ratom);
        lib_printf("| angular order    : %d\n", pds->basis_convention.order);
        lib_printf("| RSH coeff m < 0  : %d\n", pds->basis_convention.coeff_m_negative);
        lib_printf("| RSH coeff m > 0  : %d\n", pds->basis_convention.coeff_m_positive);
    }
    pds->comm_h.barrier();

    profiler.stop(tname);
}

void librpa_set_latvec_and_G(LibrpaHandler* h, const double lat_mat[9], const double G_mat[9])
{
    using std::cout;
    using std::endl;
    using librpa_int::global::profiler;

    const std::string tname = "api_set_latvec_and_G";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    auto &pbc = pds->pbc;

    std::vector<double> latt(lat_mat, lat_mat + 9);
    std::vector<double> recp(G_mat, G_mat + 9);

    pbc.set_latvec_and_G(latt, recp);

    pds->comm_h.barrier();
    if (pds->comm_h.is_root())
    {
        cout << "Lattice vectors (Bohr): latt" << endl;
        pbc.latvec.print(16);
        cout << "Reciprocal lattice vectors (2PI Bohr^-1): G" << endl;
        pbc.G.print(16);
        const auto iden_test = pbc.latvec * pbc.G.Transpose();
        cout << "Consistency check: latt * G.T" << endl;
        iden_test.print(16);
    }
    pds->comm_h.barrier();

    // Set fractional coordinates if Cartesian coordinates have been parsed.
    auto &atoms = pds->atoms;
    if (atoms.size() > 0)
    {
        atoms.set({}, {}, pbc.latvec);
    }
    profiler.stop(tname);
}

void librpa_set_atoms(LibrpaHandler* h, int natoms, const int *types, const double *posi_cart)
{
    using std::cout;
    using std::endl;
    using librpa_int::coord_t;
    using librpa_int::global::lib_printf;
    using librpa_int::global::profiler;

    const std::string tname = "api_set_atoms";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    auto &pbc = pds->pbc;
    auto &atoms = pds->atoms;

    std::vector<int> v_types(types, types + natoms);
    std::vector<coord_t> v_coords(natoms);
    for (int i = 0; i < natoms; i++)
    {
        v_coords[i] = coord_t{posi_cart[3 * i], posi_cart[3 * i + 1], posi_cart[3 * i + 2]};
    }

    // Set the fractional coordinates as well if the lattice has been set manually
    if (pbc.is_latt_set())
    {
        atoms.set(v_types, v_coords, pbc.latvec);
        const auto &coords = atoms.coords;
        const auto &coords_frac = atoms.coords_frac;
        pds->comm_h.barrier();
        if (pds->comm_h.is_root())
        {
            cout << "Atom positions read (Cartesian in Bohr | fractional):" << endl;
            for (int i = 0; i != natoms; i++)
            {
                const auto i_at = librpa_int::as_atom(i);
                const auto &c = coords.at(i_at);
                const auto &cf = coords_frac.at(i_at);
                lib_printf("ia %4d: %12.7f %12.7f %12.7f | %12.7f %12.7f %12.7f\n", i + 1, c.x, c.y,
                        c.z, cf.x, cf.y, cf.z);
            }
        }
        pds->comm_h.barrier();
    }
    else
    {
        atoms.set(v_types, v_coords);
        const auto &coords = atoms.coords;
        pds->comm_h.barrier();
        if (pds->comm_h.is_root())
        {
            cout << "Atom positions read (Cartesian in Bohr, fractional not set due to uninitialized lattice):" << endl;
            for (int i = 0; i != natoms; i++)
            {
                const auto i_at = librpa_int::as_atom(i);
                const auto &c = coords.at(i_at);
                lib_printf("ia %4d: %12.7f %12.7f %12.7f\n", i + 1, c.x, c.y, c.z);
            }
        }
        pds->comm_h.barrier();
    }

    if (pds->basis_wfc.initialized())
        condense_species_basis_layouts(pds->basis_wfc, pds->atoms.types, pds->basis_wfc_layouts);
    if (pds->basis_aux.initialized())
        condense_species_basis_layouts(pds->basis_aux, pds->atoms.types, pds->basis_aux_layouts);
    if (pds->basis_aux_shrink.initialized())
        condense_species_basis_layouts(pds->basis_aux_shrink, pds->atoms.types, pds->basis_aux_shrink_layouts);

    profiler.stop(tname);
}

void librpa_set_kgrids_kvec(LibrpaHandler* h, int nk1, int nk2, int nk3, const double* kvecs)
{
    using librpa_int::global::lib_printf;
    using std::cout;
    using std::endl;
    using librpa_int::global::profiler;

    const std::string tname = "api_set_kgrids_kvec";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    auto &pbc = pds->pbc;

    const int nkpts = nk1 * nk2 * nk3;
    std::vector<double> v_kvecs(kvecs, kvecs + 3 * nkpts);

    pbc.set_kgrids_kvec(nk1, nk2, nk3, v_kvecs);

    pds->comm_h.barrier();
    if (pds->comm_h.is_root())
    {
        librpa_int::global::lib_printf("kgrids: %3d %3d %3d\n", pbc.period.x, pbc.period.y, pbc.period.z);
        cout << "k-points read (Cartesian in 2Pi Bohr^-1 | fractional):" << endl;

        const auto &klist = pbc.klist;
        const auto &kfrac_list = pbc.kfrac_list;
        for (int ik = 0; ik != nkpts; ik++)
        {
            lib_printf("ik %4d: %12.7f %12.7f %12.7f | %12.7f %12.7f %12.7f\n",
                    ik+1, klist[ik].x, klist[ik].y, klist[ik].z,
                    kfrac_list[ik].x, kfrac_list[ik].y, kfrac_list[ik].z);
        }

        const auto &Rlist = pbc.Rlist;
        cout << "R-points to compute:" << endl;
        for (int iR = 0; iR != nkpts; iR++)
        {
            lib_printf("%4d: %3d %3d %3d\n", iR+1, Rlist[iR].x, Rlist[iR].y, Rlist[iR].z);
        }
        cout << endl;
    }
    pds->comm_h.barrier();

    profiler.stop(tname);
}

void librpa_set_ibz_mapping(LibrpaHandler* h, int nkpts, const int* map_ibzk)
{
    using std::cout;
    using std::endl;
    using namespace librpa_int;  // for STL io
    using namespace librpa_int::global;
    using librpa_int::global::profiler;

    const std::string tname = "api_set_ibz_mapping";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    auto &pbc = pds->pbc;
    assert(nkpts == pbc.get_n_cells_bvk());

    std::vector<int> map(map_ibzk, map_ibzk + nkpts);
    pbc.set_ibz_mapping(map, {});
    // ofs_myid << map << std::endl;
    pds->comm_h.barrier();
    if (pds->comm_h.is_root())
    {
        const int nkpt = pbc.irk_point_id_mapping.size();
        cout << "Irreducible Brillouin mapping:" << endl;
        lib_printf("%4s: %12s %12s %12s    %4s\n",
                   "ik", "k_1", "k_2", "k_3", "ibzk");
        for (int ik = 0; ik < nkpt; ik++)
        {
            int ibzk = pbc.irk_point_id_mapping[ik];
            if (ibzk == ik)
            {
                lib_printf("%4d: %12.7f %12.7f %12.7f\n",
                           ik + 1, pbc.kfrac_list[ik].x, pbc.kfrac_list[ik].y, pbc.kfrac_list[ik].z);
            }
            else
            {
                lib_printf("%4d: %12.7f %12.7f %12.7f -> %4d\n",
                           ik + 1, pbc.kfrac_list[ik].x, pbc.kfrac_list[ik].y, pbc.kfrac_list[ik].z, ibzk + 1);
            }
        }
    }
    pds->comm_h.barrier();

    profiler.stop(tname);
}

void librpa_set_lri_coeff(LibrpaHandler* h, LibrpaParallelRouting routing, int I, int J,
                          int nbasis_i, int nbasis_j, int naux_mu, const int R[3],
                          const double* Cs_in)
{
    using std::endl;
    using namespace librpa_int;
    using librpa_int::matrix;
    using librpa_int::Vector3_Order;
    using librpa_int::as_size;
    using librpa_int::global::ofs_myid;
    using librpa_int::global::profiler;

    const std::string tname = "api_set_lri_coeff";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    auto &cs_data = pds->cs_data;
    if (!pds->basis_wfc.initialized())
        throw LIBRPA_RUNTIME_ERROR("wave function basis not set, call (librpa_)set_ao_basis_wfc first");
    if (!pds->basis_aux.initialized())
        throw LIBRPA_RUNTIME_ERROR("auxiliary basis not set, call (librpa_)set_ao_basis_aux first");

    const size_t cs_size = nbasis_i * nbasis_j * naux_mu;
    const size_t n_ij = nbasis_i * nbasis_j;

    // ofs_myid << "Parsing I J " << I << " " << J << " "
    //          << "ds->basis_aux " << pds->basis_aux.nbs_ << " "
    //          << "ds->basis_wfc " << pds->basis_wfc.nbs_ << endl;
    // ofs_myid << "ds->basis_aux[I] == naux_mu ? "
    //          << pds->basis_aux[I] << " " << as_size(naux_mu) << " "
    //          << std::boolalpha << (pds->basis_aux[I] == as_size(naux_mu))
    //          << endl;
    // ofs_myid << "ds->basis_wfc[I] == nbasis_i ? "
    //          << pds->basis_wfc[I] << " " << as_size(nbasis_i) << " "
    //          << std::boolalpha << (pds->basis_wfc[I] == as_size(nbasis_i))
    //          << endl;
    // ofs_myid << "ds->basis_wfc[J] == nbasis_j ? "
    //          << pds->basis_wfc[J] << " " << as_size(nbasis_j) << " "
    //          << std::boolalpha << (pds->basis_wfc[J] == as_size(nbasis_j))
    //          << endl;

    assert(pds->basis_aux[I] == as_size(naux_mu));
    assert(pds->basis_wfc[I] == as_size(nbasis_i));
    assert(pds->basis_wfc[J] == as_size(nbasis_j));

    if (routing == LIBRPA_ROUTING_LIBRI)
    {
        const std::array<int, 3> Ra{R[0], R[1], R[2]};
        // RI tensor uses ABF as slowest index, so we need transpose the input data.
        auto data = std::make_shared<std::valarray<double>>(cs_size);
        // Unless n_cols >> n_rows, cache-friendly reading (by row in C/C++) is more efficient
        for (size_t i_row = 0; i_row < n_ij; i_row++)
        {
            for (size_t i_col = 0; i_col != as_size(naux_mu); i_col++)
            {
                (*data)[i_col * n_ij + i_row] = Cs_in[i_row * naux_mu + i_col];
            }
        }
        const std::initializer_list<std::size_t> shape{as_size(naux_mu), as_size(nbasis_i),
                                                       as_size(nbasis_j)};
        cs_data.data_libri[I][{J, Ra}] = RI::Tensor<double>(shape, data);
        cs_data.use_libri = true;
    }
    else
    {
        librpa_int::Vector3_Order<int> box(R[0], R[1], R[2]);
        std::shared_ptr<matrix> cs_ptr = std::make_shared<matrix>();
        cs_ptr->create(nbasis_i * nbasis_j, naux_mu);
        memcpy((*cs_ptr).c, Cs_in, sizeof(double) * cs_size);
        cs_data.data_IJR[I][J][box] = cs_ptr;
        cs_data.use_libri = false;
    }

    profiler.stop(tname);
}

static void _set_aux_coulomb_k_atom_pair(const librpa_int::Vector3_Order<double> &qvec,
                                         librpa_int::atom_t I, librpa_int::atom_t J, size_t naux_mu, size_t naux_nu,
                                         const double* Vq_real_in, const double* Vq_imag_in,
                                         librpa_int::atpair_k_cplx_mat_t &coulomb_mat,
                                         double vq_threshold)
{
    using librpa_int::ComplexMatrix;

    std::shared_ptr<ComplexMatrix> vq_ptr = std::make_shared<ComplexMatrix>();
    vq_ptr->create(naux_mu, naux_nu);

    // Copy real data
    for (size_t i_mu = 0; i_mu < naux_mu; i_mu++)
    {
        for (size_t i_nu = 0; i_nu < naux_nu; i_nu++)
        {
            const auto id = i_nu + i_mu * naux_nu;
            (*vq_ptr)(i_mu, i_nu) = librpa_int::cplxdb(Vq_real_in[id], Vq_imag_in[id]);
        }
    }

    if ((*vq_ptr).real().absmax() >= vq_threshold)
    {
        coulomb_mat[I][J][qvec] = vq_ptr;
    }
}

static void _set_aux_coulomb_k_atom_pair_packed(
    const librpa_int::Vector3_Order<double> &qvec, librpa_int::atom_t I, librpa_int::atom_t J,
    size_t naux_mu, size_t naux_nu, const double* Vq_ri_in,
    librpa_int::atpair_k_cplx_mat_t &coulomb_mat, double vq_threshold)
{
    using librpa_int::ComplexMatrix;

    std::shared_ptr<ComplexMatrix> vq_ptr = std::make_shared<ComplexMatrix>();
    vq_ptr->create(naux_mu, naux_nu);

    for (size_t i_mu = 0; i_mu < naux_mu; i_mu++)
    {
        for (size_t i_nu = 0; i_nu < naux_nu; i_nu++)
        {
            const auto id = i_nu + i_mu * naux_nu;
            (*vq_ptr)(i_mu, i_nu) =
                librpa_int::cplxdb(Vq_ri_in[2 * id], Vq_ri_in[2 * id + 1]);
        }
    }

    if ((*vq_ptr).real().absmax() >= vq_threshold)
    {
        coulomb_mat[I][J][qvec] = vq_ptr;
    }
}

void librpa_set_aux_bare_coulomb_k_atom_pair(LibrpaHandler* h, int ik, int I, int J, int naux_mu,
                                             int naux_nu, const double* Vq_real_in,
                                             const double* Vq_imag_in, double vq_threshold)
{
    using librpa_int::global::profiler;

    const std::string tname = "api_set_aux_bare_coulomb_k_atom_pair";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    const auto &qvec = pds->pbc.klist[ik];
    _set_aux_coulomb_k_atom_pair(qvec, I, J, naux_mu, naux_nu, Vq_real_in, Vq_imag_in, pds->vq, vq_threshold);

    profiler.stop(tname);
}

void librpa_set_aux_bare_coulomb_k_atom_pair_packed(LibrpaHandler* h, int ik, int I, int J,
                                                    int naux_mu, int naux_nu,
                                                    const double* Vq_ri_in,
                                                    double vq_threshold)
{
    using librpa_int::global::profiler;

    const std::string tname = "api_set_aux_bare_coulomb_k_atom_pair_packed";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    const auto &qvec = pds->pbc.klist[ik];
    _set_aux_coulomb_k_atom_pair_packed(qvec, I, J, naux_mu, naux_nu, Vq_ri_in,
                                        pds->vq, vq_threshold);

    profiler.stop(tname);
}

void librpa_set_aux_cut_coulomb_k_atom_pair(LibrpaHandler* h, int ik, int I, int J, int naux_mu,
                                            int naux_nu, const double* Vq_real_in,
                                            const double* Vq_imag_in, double vq_threshold)
{
    using librpa_int::global::profiler;

    const std::string tname = "api_set_aux_cut_coulomb_k_atom_pair";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    const auto &qvec = pds->pbc.klist[ik];
    _set_aux_coulomb_k_atom_pair(qvec, I, J, naux_mu, naux_nu, Vq_real_in, Vq_imag_in, pds->vq_cut, vq_threshold);

    profiler.stop(tname);
}

void librpa_set_aux_cut_coulomb_k_atom_pair_packed(LibrpaHandler* h, int ik, int I, int J,
                                                   int naux_mu, int naux_nu,
                                                   const double* Vq_ri_in,
                                                   double vq_threshold)
{
    using librpa_int::global::profiler;

    const std::string tname = "api_set_aux_cut_coulomb_k_atom_pair_packed";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    const auto &qvec = pds->pbc.klist[ik];
    _set_aux_coulomb_k_atom_pair_packed(qvec, I, J, naux_mu, naux_nu, Vq_ri_in,
                                        pds->vq_cut, vq_threshold);

    profiler.stop(tname);
}

static void _set_aux_coulomb_k_2D_block(
    const librpa_int::Vector3_Order<double>& qvec, int mu_begin, int mu_end,
    int nu_begin, int nu_end, const double* Vq_real_in, const double* Vq_imag_in,
    std::map<librpa_int::Vector3_Order<double>, librpa_int::Matz>& vq_block)
{
    using namespace librpa_int;
    using std::shared_ptr;
    using std::make_shared;

    int n_mu_loc = mu_end - mu_begin;
    int n_nu_loc = nu_end - nu_begin;
    if (n_mu_loc < 1 || n_nu_loc < 1) return;

    Matz mat(n_mu_loc, n_nu_loc, MAJOR::ROW);

    size_t ii = 0;
    for (int i_mu = 0; i_mu < n_mu_loc; i_mu++)
    {
        for (int i_nu = 0; i_nu < n_nu_loc; i_nu++)
        {
            mat(i_mu, i_nu) = complex<double>(Vq_real_in[ii], Vq_imag_in[ii]);
            ii += 1;
        }
    }
    mat.swap_to_col_major();
    vq_block[qvec] = std::move(mat);
}

static void _set_aux_coulomb_k_2D_block_packed(
    const librpa_int::Vector3_Order<double>& qvec, int mu_begin, int mu_end,
    int nu_begin, int nu_end, const double* Vq_ri_in,
    std::map<librpa_int::Vector3_Order<double>, librpa_int::Matz>& vq_block)
{
    using namespace librpa_int;
    using std::shared_ptr;
    using std::make_shared;

    int n_mu_loc = mu_end - mu_begin;
    int n_nu_loc = nu_end - nu_begin;
    if (n_mu_loc < 1 || n_nu_loc < 1) return;

    Matz mat(n_mu_loc, n_nu_loc, MAJOR::ROW);

    size_t ii = 0;
    for (int i_mu = 0; i_mu < n_mu_loc; i_mu++)
    {
        for (int i_nu = 0; i_nu < n_nu_loc; i_nu++)
        {
            mat(i_mu, i_nu) = complex<double>(Vq_ri_in[2 * ii], Vq_ri_in[2 * ii + 1]);
            ii += 1;
        }
    }
    mat.swap_to_col_major();
    vq_block[qvec] = std::move(mat);
}

static void _parse_vq_dims(int &lbrow, int &ubrow, int &lbcol, int &ubcol,
                           const int lbrow_in, const int ubrow_in, const int lbcol_in, const int ubcol_in)
{
    if (lbrow < 0 || ubrow < 0 || lbcol < 0 || ubcol < 0)
    {
        lbrow = lbrow_in;
        ubrow = ubrow_in;
        lbcol = lbcol_in;
        ubcol = ubcol_in;
        return;
    }

    if (lbrow != lbrow_in || ubrow != ubrow_in || lbcol != lbcol_in || ubcol != ubcol_in)
    {
        throw LIBRPA_RUNTIME_ERROR("{lb,ub}{row,col} input is inconsistent from previous value");
    }
}

void librpa_set_aux_bare_coulomb_k_2d_block(LibrpaHandler* h, int ik, int mu_begin, int mu_end,
                                            int nu_begin, int nu_end, const double* Vq_real_in,
                                            const double* Vq_imag_in)
{
    using librpa_int::global::profiler;

    const std::string tname = "api_set_aux_bare_coulomb_k_2d_block";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    const auto &qvec = pds->pbc.klist[ik];
    _parse_vq_dims(pds->vq_lbrow, pds->vq_ubrow, pds->vq_lbcol, pds->vq_ubcol,
                   mu_begin, mu_end, nu_begin, nu_end);
    _set_aux_coulomb_k_2D_block(qvec, mu_begin, mu_end, nu_begin, nu_end, Vq_real_in,
                                Vq_imag_in, pds->vq_block_loc);

    profiler.stop(tname);
}

void librpa_set_aux_bare_coulomb_k_2d_block_packed(LibrpaHandler* h, int ik, int mu_begin,
                                                   int mu_end, int nu_begin, int nu_end,
                                                   const double* Vq_ri_in)
{
    using librpa_int::global::profiler;

    const std::string tname = "api_set_aux_bare_coulomb_k_2d_block_packed";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    const auto &qvec = pds->pbc.klist[ik];
    _parse_vq_dims(pds->vq_lbrow, pds->vq_ubrow, pds->vq_lbcol, pds->vq_ubcol,
                   mu_begin, mu_end, nu_begin, nu_end);
    _set_aux_coulomb_k_2D_block_packed(qvec, mu_begin, mu_end, nu_begin, nu_end,
                                       Vq_ri_in, pds->vq_block_loc);

    profiler.stop(tname);
}

void librpa_set_aux_cut_coulomb_k_2d_block(LibrpaHandler* h, int ik, int mu_begin, int mu_end,
                                           int nu_begin, int nu_end, const double* Vq_real_in,
                                           const double* Vq_imag_in)
{
    using librpa_int::global::profiler;

    const std::string tname = "api_set_aux_cut_coulomb_k_2d_block";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    const auto &qvec = pds->pbc.klist[ik];
    _parse_vq_dims(pds->vq_lbrow, pds->vq_ubrow, pds->vq_lbcol, pds->vq_ubcol,
                   mu_begin, mu_end, nu_begin, nu_end);
    _set_aux_coulomb_k_2D_block(qvec, mu_begin, mu_end, nu_begin, nu_end, Vq_real_in,
                                Vq_imag_in, pds->vq_cut_block_loc);

    profiler.stop(tname);
}

void librpa_set_aux_cut_coulomb_k_2d_block_packed(LibrpaHandler* h, int ik, int mu_begin,
                                                  int mu_end, int nu_begin, int nu_end,
                                                  const double* Vq_ri_in)
{
    using librpa_int::global::profiler;

    const std::string tname = "api_set_aux_cut_coulomb_k_2d_block_packed";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    const auto &qvec = pds->pbc.klist[ik];
    _parse_vq_dims(pds->vq_lbrow, pds->vq_ubrow, pds->vq_lbcol, pds->vq_ubcol,
                   mu_begin, mu_end, nu_begin, nu_end);
    _set_aux_coulomb_k_2D_block_packed(qvec, mu_begin, mu_end, nu_begin, nu_end,
                                       Vq_ri_in, pds->vq_cut_block_loc);

    profiler.stop(tname);
}

void librpa_set_dielect_func_imagfreq(LibrpaHandler* h, int nfreq, const double *omegas_imag, const double *dielect_func)
{
    using namespace librpa_int;
    auto pds = librpa_int::api::get_dataset_instance(h);
    pds->omegas_imagfreq = std::vector<double>(omegas_imag, omegas_imag + nfreq);
    pds->epsmacs_imagfreq = std::vector<double>(dielect_func, dielect_func + nfreq);
}

static std::size_t _velocity_matrix_flat_index(const int ispin, const int ik,
                                               const int alpha, const int i_state,
                                               const int j_state, const int n_kpts,
                                               const int n_states)
{
    return (((static_cast<std::size_t>(ispin) * n_kpts + ik) * 3 + alpha) * n_states
            + i_state) * n_states + j_state;
}

void librpa_set_velocity_matrix(LibrpaHandler* h, int n_spins, int n_kpts,
                                int n_states, const double* velocity_real,
                                const double* velocity_imag)
{
    using namespace librpa_int;
    using librpa_int::global::profiler;

    const std::string tname = "api_set_velocity_matrix";
    profiler.start(tname);
    if (n_spins <= 0 || n_kpts <= 0 || n_states <= 0)
        throw LIBRPA_RUNTIME_ERROR("velocity matrix dimensions must be positive");
    if (velocity_real == nullptr || velocity_imag == nullptr)
        throw LIBRPA_RUNTIME_ERROR("velocity matrix real/imag arrays must not be null");

    auto pds = librpa_int::api::get_dataset_instance(h);
    initialize_velocity_matrix(pds->velocity_matrix, n_spins, n_kpts, n_states);
    for (int ispin = 0; ispin != n_spins; ++ispin)
    {
        for (int ik = 0; ik != n_kpts; ++ik)
        {
            for (int alpha = 0; alpha != 3; ++alpha)
            {
                auto &vmat = pds->velocity_matrix[ispin][ik][alpha];
                for (int i = 0; i != n_states; ++i)
                {
                    for (int j = 0; j != n_states; ++j)
                    {
                        const auto idx =
                            _velocity_matrix_flat_index(ispin, ik, alpha, i, j,
                                                        n_kpts, n_states);
                        vmat(i, j) = std::complex<double>(velocity_real[idx],
                                                          velocity_imag[idx]);
                    }
                }
            }
        }
    }
    pds->p_headwing.reset();

    profiler.stop(tname);
}

void librpa_set_velocity_matrix_packed(LibrpaHandler* h, int n_spins, int n_kpts,
                                       int n_states, const double* velocity_ri)
{
    using namespace librpa_int;
    using librpa_int::global::profiler;

    const std::string tname = "api_set_velocity_matrix_packed";
    profiler.start(tname);
    if (n_spins <= 0 || n_kpts <= 0 || n_states <= 0)
        throw LIBRPA_RUNTIME_ERROR("velocity matrix dimensions must be positive");
    if (velocity_ri == nullptr)
        throw LIBRPA_RUNTIME_ERROR("velocity matrix packed array must not be null");

    auto pds = librpa_int::api::get_dataset_instance(h);
    initialize_velocity_matrix(pds->velocity_matrix, n_spins, n_kpts, n_states);
    for (int ispin = 0; ispin != n_spins; ++ispin)
    {
        for (int ik = 0; ik != n_kpts; ++ik)
        {
            for (int alpha = 0; alpha != 3; ++alpha)
            {
                auto &vmat = pds->velocity_matrix[ispin][ik][alpha];
                for (int i = 0; i != n_states; ++i)
                {
                    for (int j = 0; j != n_states; ++j)
                    {
                        const auto idx =
                            _velocity_matrix_flat_index(ispin, ik, alpha, i, j,
                                                        n_kpts, n_states);
                        vmat(i, j) = std::complex<double>(velocity_ri[2 * idx],
                                                          velocity_ri[2 * idx + 1]);
                    }
                }
            }
        }
    }
    pds->p_headwing.reset();

    profiler.stop(tname);
}

void librpa_set_band_kvec(LibrpaHandler* h, int n_kpts_band, const double* kfrac_list_band)
{
    using librpa_int::global::lib_printf;
    using librpa_int::Vector3_Order;
    using std::cout;
    using std::endl;
    using librpa_int::global::profiler;

    const std::string tname = "api_set_band_kvec";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    mark_band_data_set(pds);
    pds->kfrac_band_list.clear();
    const auto &kf = kfrac_list_band;
    for (int ik = 0; ik < n_kpts_band; ik++)
    {
        const int i = ik * 3;
        Vector3_Order<double> kfrac{kf[i], kf[i+1], kf[i+2]};
        pds->kfrac_band_list.emplace_back(kfrac);
    }

    profiler.stop(tname);
}

void librpa_set_band_occ_eigval(LibrpaHandler* h, int n_spins, int n_kpts_band, int n_states,
                                const double* occ, const double* eig)
{
    using librpa_int::global::profiler;

    const std::string tname = "api_set_band_occ_eigval";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    const int n_basis = pds->mf.get_n_aos();
    const int n_spinor = pds->mf.get_n_spinor();
    const double efermi = pds->mf.get_efermi();
    mark_band_data_set(pds);

    auto &mfb = pds->mf_band;
    mfb.set(n_spins, n_kpts_band, n_states, n_basis, n_spinor);
    // Band-path QPE uses the same reference Fermi level as the SCF k-grid.
    mfb.get_efermi() = efermi;
    auto& eskb = mfb.get_eigenvals();
    auto& swg = mfb.get_weight();
    int length_kb = n_kpts_band * n_states;
    for (int is = 0; is != n_spins; is++)
    {
        memcpy(eskb[is].c, eig + length_kb * is, length_kb * sizeof(double));
        memcpy(swg[is].c, occ + length_kb * is, length_kb * sizeof(double));
        // wg[is](k_index, i) = stod(ws) / n_kpoints; // different with abacus!
        swg[is] *= (1.0 / n_kpts_band);
    }

    profiler.stop(tname);
}

void librpa_set_wfc_band(LibrpaHandler* h, int ispin, int ik_band, int nstates_local,
                         int nbasis_local, const double* wfc_real, const double* wfc_imag)
{
    using librpa_int::global::profiler;

    const std::string tname = "api_set_wfc_band";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    mark_band_data_set(pds);
    auto &mfb = pds->mf_band;

    auto& wfc = mfb.get_eigenvectors()[ispin][0][ik_band];
    wfc.create(nstates_local, nbasis_local);
    const size_t n = mfb.get_n_bands() * mfb.get_n_aos();
    for (size_t i = 0; i < n; i++)
    {
        wfc.c[i] = std::complex<double>(wfc_real[i], wfc_imag[i]);
    }
    // std::cout << "Maxabs: " << wfc.get_max_abs() << std::endl;
    librpa_int::global::ofs_myid
        << "Wave-function (band) set : ispin = " << ispin << " ik = " << ik_band
        << " nstates_local = " << nstates_local << " nbasis_local = " << nbasis_local
        << std::endl;

    profiler.stop(tname);
}

void librpa_set_wfc_band_spinor(LibrpaHandler* h, int ik_band, int nstates_local, int nbasis_local,
                                const double* wfc_up_real, const double* wfc_up_imag,
                                const double* wfc_dn_real, const double* wfc_dn_imag)
{
    using librpa_int::global::profiler;

    const std::string tname = "api_set_wfc_band";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    mark_band_data_set(pds);
    auto &mfb = pds->mf_band;

    auto& wfcs_up = mfb.get_eigenvectors()[0][0][ik_band];
    auto& wfcs_dn = mfb.get_eigenvectors()[0][1][ik_band];
    wfcs_up.create(nstates_local, nbasis_local);
    wfcs_dn.create(nstates_local, nbasis_local);
    const size_t n = mfb.get_n_bands() * mfb.get_n_aos();
    for (size_t i = 0; i < n; i++)
    {
        wfcs_up.c[i] = std::complex<double>(wfc_up_real[i], wfc_up_imag[i]);
        wfcs_dn.c[i] = std::complex<double>(wfc_dn_real[i], wfc_dn_imag[i]);
    }
    // std::cout << "Maxabs: " << wfc.get_max_abs() << std::endl;
    librpa_int::global::ofs_myid
        << "Spinor wave-function (band) set : ik = " << ik_band
        << " nstates_local = " << nstates_local << " nbasis_local = " << nbasis_local
        << std::endl;

    profiler.stop(tname);
}

void librpa_set_wfc_band_packed(LibrpaHandler* h, int ispin, int ik_band, int nstates_local,
                                int nbasis_local, const double* wfc_ri)
{
    using librpa_int::global::profiler;

    const std::string tname = "api_set_wfc_band_packed";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    mark_band_data_set(pds);
    auto &mfb = pds->mf_band;

    auto& wfc = mfb.get_eigenvectors()[ispin][0][ik_band];
    wfc.create(nstates_local, nbasis_local);
    const size_t n = mfb.get_n_bands() * mfb.get_n_aos();
    for (size_t i = 0; i < n; i++)
    {
        wfc.c[i] = std::complex<double>(wfc_ri[2*i], wfc_ri[2*i+1]);
    }
    // std::cout << "Maxabs: " << wfc.get_max_abs() << std::endl;
    librpa_int::global::ofs_myid
        << "Wave-function (band) set : ispin = " << ispin << " ik = " << ik_band
        << " nstates_local = " << nstates_local << " nbasis_local = " << nbasis_local
        << std::endl;

    profiler.stop(tname);
}

void librpa_set_wfc_band_spinor_packed(LibrpaHandler* h, int ik_band, int nstates_local,
                                       int nbasis_local, const double* wfc_up_ri,
                                       const double* wfc_dn_ri)
{
    using librpa_int::global::profiler;

    const std::string tname = "api_set_wfc_band_packed";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    mark_band_data_set(pds);
    auto &mfb = pds->mf_band;

    auto& wfcs_up = mfb.get_eigenvectors()[0][0][ik_band];
    auto& wfcs_dn = mfb.get_eigenvectors()[0][1][ik_band];
    wfcs_up.create(nstates_local, nbasis_local);
    wfcs_dn.create(nstates_local, nbasis_local);
    const size_t n = mfb.get_n_bands() * mfb.get_n_aos();
    for (size_t i = 0; i < n; i++)
    {
        wfcs_up.c[i] = std::complex<double>(wfc_up_ri[2*i], wfc_up_ri[2*i+1]);
        wfcs_dn.c[i] = std::complex<double>(wfc_dn_ri[2*i], wfc_dn_ri[2*i+1]);
    }
    // std::cout << "Maxabs: " << wfc.get_max_abs() << std::endl;
    librpa_int::global::ofs_myid
        << "Spinor wave-function (band) set : ik = " << ik_band
        << " nstates_local = " << nstates_local << " nbasis_local = " << nbasis_local
        << std::endl;

    profiler.stop(tname);
}

void librpa_reset_band_data(LibrpaHandler* h)
{
    using librpa_int::global::profiler;

    const std::string tname = "api_reset_band_data";
    profiler.start(tname);

    auto pds = librpa_int::api::get_dataset_instance(h);
    pds->is_band_data_set = false;
    pds->is_band_calc_done = false;
    pds->kfrac_band_list = {};
    pds->mf_band = librpa_int::MeanField();

    profiler.stop(tname);
}
