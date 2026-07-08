This QSGW testcase is derived from the committed
`g0w0_band_abacus_MnO2_nspin2_shrink_wing_libri` ABACUS benchmark dataset.
It exercises the ABACUS/PyATB QSGW input path with shrinked ABFs, head/wing,
spin polarization, `vxc_out` fallback, and fixed-basis iter-1 dump matrices.

The reference files are not generated from this merged implementation. They
come from the pre-merge QSGW implementation run:

```
/home/bhj/ai-runs/librpa-qsgw-pr-gate-20260708/mno2_dump_compare_iter1_20260708_180417/old
```

The merged implementation was checked against that old run before committing
this case; the selected iter-1 matrices matched exactly in the old-vs-current
artifact comparison.

The input intentionally uses the same ABACUS reader contract as the G0W0 MnO2
benchmark dataset:

```
version_coul_reader = 1
version_lri_reader = 1
prefix_lri_coeff = v1_Cs_data_
prefix_lri_coeff_shrink = v1_Cs_shrinked_data_
prefix_shrink_sinvS = v1_shrink_sinvS_
prefix_coul_full = v1_coulomb_full_iq_
prefix_coul_cut = v1_coulomb_cut_iq_
```

Do not add `fold_C` here. This is an ABACUS/PyATB packed dataset, not the
FHI-aims full-grid output path.
