This QSGW testcase is derived from the official `g0w0_aims_Si_libri`
silicon testcase. It keeps the small official FHI-aims geometry/basis and adds
the full-grid exchange-correlation matrix files required by the QSGW driver.

The FHI-aims input contract is intentionally stricter than the original G0W0
case:

```
symmetry_reduced_k_grid .false.
periodic_gw_full_matrix .true.
output self_energy_matrix 1 6 .true.
periodic_gw_self_energy_restart write
periodic_gw_optimize_kgrid_symmetry none
output librpa binary develop
print_input_librpa .true.
output gw_regular_kgrid
```

The `1 6` band range is specific to this small Si benchmark. For another
dataset, the FHI-aims `output self_energy_matrix` range must cover the same
band space read by LibRPA QSGW; the current QSGW task checks that each
`xc_matr` file is `n_bands x n_bands`.

Do not add `fold_C` to this packed benchmark unless the dataset is regenerated
and revalidated with the matching LibRPA reader contract. The reference data
here comes from the full-grid no-fold path, which produced the sane iter-1
QSGW/H0/Vc scale used for the reference.

The packed dataset uses the binary-v1 Coulomb filenames consumed by
`librpa/librpa.in`:

```
version_coul_reader = 1
prefix_coul_full = coulomb_full_iq
prefix_coul_cut  = coulomb_cut_iq
```

The regression run must be self-contained in `librpa/librpa.in`, without
external QSGW environment knobs.  The active QSGW controls in this case are:

```
max_iter = 1
qsgw_dump_iter1 = t
qsgw_mixer = linear
qsgw_mixing_beta = 0.25
qsgw_mixing_history = 12
qsgw_linear_mixing_steps = 3
```

`max_iter` is intentionally one for this packed CI-sized testcase; longer
self-consistency scans are benchmark artifacts, not part of this small
regression case.  `qsgw_dump_iter1` writes the `librpa.d/qsgw_dump/*.mm`
matrices validated by `testsuite.xml`.

If the source FHI-aims run only writes legacy Coulomb files, convert them before
refreshing this benchmark:

```
cd dataset
script=$LIBRPA_ROOT/utilities/convert_legacy_coulomb_mat.py
$script -i coulomb_mat -o coulomb_full_iq
$script -i coulomb_cut -o coulomb_cut_iq
rm -f coulomb_mat_*.txt coulomb_cut_*.txt
```
