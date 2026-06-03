The original dataset from FHI-aims has been post-processed through

```
cd dataset
script=$LIBRPA_ROOT/utilities/convert_legacy_coulomb_mat.py
$script -i coulomb_mat -o coulomb_full_iq
$script -i coulomb_cut -o coulomb_cut_iq
rm -f coulomb_mat_*.txt coulomb_cut_*.txt
```
