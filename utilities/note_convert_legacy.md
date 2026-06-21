# Converting legacy data

Run these commands from the repository root unless noted otherwise.

## LRI coefficients (`Cs_data`)

Build the converter if needed:

```bash
make -C utilities convert_legacy_Cs.exe
```

Convert one legacy `Cs_data` file to reader v1:

```bash
./utilities/convert_legacy_Cs.exe Cs_data_1.txt v1_Cs_data_1.dat --overwrite
```

The input may be legacy text or unformatted binary. Auto-detection is the
default; pass `--input-format text` or `--input-format binary` only when needed.

## Coulomb matrix

Convert legacy `coulomb_mat_*.txt` files in a dataset directory:

```bash
python3 utilities/convert_legacy_coulomb_mat.py /path/to/dataset
```

The default output prefix is `coulomb_full_iq`, one binary file per q-point.
The converter reads atom auxiliary sizes from `basis_out`/`stru_out` or
`Cs_data*` when possible; otherwise pass `--atom-naux`.

For large datasets, use the MPI converter:

```bash
mpirun -np 4 utilities/convert_legacy_coulomb_mat_mpi.exe /path/to/dataset
```

The MPI converter targets reader v1, streams complex output, and skips lower
input entries. Use `--restart` to resume from its checkpoint file.

## ABACUS `shrink_sinvS`

Convert one legacy text file:

```bash
g++ -std=c++17 -O2 -o convert_legacy_sinvS.exe convert_legacy_sinvS.cpp
./utilities/convert_legacy_sinvS.exe shrink_sinvS_0.txt v1_shrink_sinvS_0.txt --overwrite
```

## KS eigenvectors

Use `convert_legacy_KS_eigenvector.py` to convert legacy text `KS_eigenvector*`
files to binary v1 files for parallel k-point reading.

```bash
./utilities/convert_legacy_KS_eigenvector.py /path/to/dataset --in-place
```

`--in-place` replaces each legacy file with its v1 binary counterpart and keeps
the original text as `legacy_<old-name>`, which the reader ignores because it no
longer starts with `KS_eigenvector`.

For spinor wave functions, pass:

```bash
./utilities/convert_legacy_KS_eigenvector.py /path/to/dataset --n-spinor 2 --in-place
```

The converter reads dimensions from `band_out` by default. If `band_out` is not
available, provide `--n-spins`, `--n-states`, and either `--n-basis-wfc` or
`--n-aos`.

Do not leave legacy text files and v1 binary files under the same
`KS_eigenvector` prefix in one input directory; the reader requires one format
per prefix.

Current v1 output is `kind=28` only: packed `complex<double>` with basis fastest,
then state, spinor, and spin.
