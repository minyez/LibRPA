#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert LibRPA text KS_eigenvector files to the binary format.

The binary format is the one consumed by driver/read_data.cpp::read_eigenvector
after the binary-reader patch.  Layout per file:

    int32  marker           = -12345678 (EIGENVECTOR_V1_MARKER)
    int32  n_kpoints_file
    int32  n_spins
    int32  n_spinor
    int32  n_states
    int32  n_aos
    for each k-point in the file:
        int32  ik_file      (1-based index as in the text file)
        for iw in 0..n_aos-1:
            for isoc in 0..n_spinor-1:
                for ib in 0..n_states-1:
                    for is in 0..n_spins-1:
                        double real
                        double imag

The order of the complex coefficients in the binary file is identical to the
order in which they appear in the text file.
"""

import argparse
import os
import struct
import sys
import tempfile
from pathlib import Path

EIGENVECTOR_V1_MARKER = -12345678


def parse_band_out(path: Path):
    """Read n_kpoints, n_spins, n_states, n_aos from the band_out header."""
    with path.open("r") as f:
        tokens = f.read().split()
    if len(tokens) < 5:
        raise ValueError(f"{path} does not contain a valid band_out header")
    n_kpoints = int(tokens[0])
    n_spins = int(tokens[1])
    n_states = int(tokens[2])
    n_aos = int(tokens[3])
    return n_kpoints, n_spins, n_states, n_aos


def token_stream(file_obj):
    """Yield whitespace-separated tokens from a text file line by line."""
    for line in file_obj:
        for token in line.split():
            yield token


def convert_file(
    in_path: Path,
    out_path: Path,
    n_spins: int,
    n_spinor: int,
    n_states: int,
    n_aos: int,
):
    """Convert a single text KS_eigenvector file to binary."""
    pairs_per_k = n_spins * n_spinor * n_states * n_aos
    values_per_k = 2 * pairs_per_k

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r") as fin, out_path.open("wb") as fout:
        # Reserve space for the header; n_kpoints_file will be patched later.
        header_pos = fout.tell()
        fout.write(struct.pack("i", EIGENVECTOR_V1_MARKER))
        fout.write(
            struct.pack(
                "iiiii",
                0,  # n_kpoints_file placeholder
                n_spins,
                n_spinor,
                n_states,
                n_aos,
            )
        )

        tokens = token_stream(fin)
        n_kpoints_file = 0
        while True:
            try:
                ktok = next(tokens)
            except StopIteration:
                break
            ik = int(ktok)
            n_kpoints_file += 1

            fout.write(struct.pack("i", ik))
            for _ in range(pairs_per_k):
                try:
                    r = float(next(tokens))
                    i = float(next(tokens))
                except StopIteration:
                    raise ValueError(
                        f"{in_path}: k-point {ik} is truncated: expected "
                        f"{pairs_per_k} (real, imag) pairs"
                    )
                fout.write(struct.pack("dd", r, i))

        # Patch n_kpoints_file in the header.
        fout.seek(header_pos + 4)
        fout.write(struct.pack("i", n_kpoints_file))

    return n_kpoints_file


def main():
    parser = argparse.ArgumentParser(
        description="Convert LibRPA text KS_eigenvector files to binary format."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the text KS_eigenvector* files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for the binary output files.  Defaults to input_dir.  "
            "Use --in-place to overwrite the original text files instead."
        ),
    )
    parser.add_argument(
        "--prefix",
        default="KS_eigenvector",
        help="Filename prefix used to locate input files (default: KS_eigenvector).",
    )
    parser.add_argument(
        "--band-out",
        type=Path,
        default=None,
        help=(
            "Path to band_out file from which n_kpoints, n_spins, n_states and "
            "n_aos are read.  Defaults to <input_dir>/band_out."
        ),
    )
    parser.add_argument(
        "--n-spins",
        type=int,
        default=None,
        help="Override number of spins (default: read from band_out).",
    )
    parser.add_argument(
        "--n-spinor",
        type=int,
        default=None,
        help=(
            "Override number of spinor components.  "
            "Default is 1; use 2 for spinor/non-collinear calculations."
        ),
    )
    parser.add_argument(
        "--n-states",
        type=int,
        default=None,
        help="Override number of states/bands (default: read from band_out).",
    )
    parser.add_argument(
        "--n-aos",
        type=int,
        default=None,
        help="Override number of atomic-orbital basis functions (default: read from band_out).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "Replace each text file by its binary counterpart, keeping the "
            "original filename.  The text file is moved to <filename>.txt.bak."
        ),
    )
    parser.add_argument(
        "--suffix",
        default=".bin",
        help="Suffix for binary output files when not using --in-place (default: .bin).",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: input directory does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    band_out_path = args.band_out or args.input_dir / "band_out"

    # Resolve dimensions
    if args.n_spins is not None and args.n_states is not None and args.n_aos is not None:
        n_spins = args.n_spins
        n_states = args.n_states
        n_aos = args.n_aos
    elif band_out_path.is_file():
        _, n_spins_bo, n_states_bo, n_aos_bo = parse_band_out(band_out_path)
        n_spins = args.n_spins if args.n_spins is not None else n_spins_bo
        n_states = args.n_states if args.n_states is not None else n_states_bo
        n_aos = args.n_aos if args.n_aos is not None else n_aos_bo
    else:
        print(
            "Error: cannot determine system dimensions.  Provide --band-out or "
            "all of --n-spins, --n-states, --n-aos.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_spinor = args.n_spinor if args.n_spinor is not None else 1

    if n_spins <= 0 or n_spinor <= 0 or n_states <= 0 or n_aos <= 0:
        print("Error: all dimensions must be positive.", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir if args.output_dir is not None else args.input_dir

    input_files = sorted(
        p for p in args.input_dir.iterdir()
        if p.is_file() and p.name.startswith(args.prefix)
    )
    if not input_files:
        print(
            f"No files matching prefix '{args.prefix}' found in {args.input_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    for in_path in input_files:
        if args.in_place:
            tmp_out = in_path.with_suffix(".tmp.bin")
            final_out = in_path
            backup_path = in_path.with_suffix(in_path.suffix + ".txt.bak")
        else:
            tmp_out = output_dir / (in_path.stem + args.suffix)
            final_out = tmp_out
            backup_path = None

        print(f"Converting {in_path.name} -> {final_out.name} ...")
        n_kpts = convert_file(in_path, tmp_out, n_spins, n_spinor, n_states, n_aos)
        print(f"  wrote {n_kpts} k-point(s), dims=({n_spins},{n_spinor},{n_states},{n_aos})")

        if args.in_place:
            in_path.rename(backup_path)
            tmp_out.rename(final_out)
            print(f"  backed up text file to {backup_path.name}")


if __name__ == "__main__":
    main()
