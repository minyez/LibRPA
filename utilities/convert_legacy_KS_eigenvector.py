#!/usr/bin/env python3
"""Convert legacy text KS_eigenvector files to binary v1 kind=28."""

import argparse
import os
import shutil
import struct
import sys
import tempfile
from array import array
from pathlib import Path

KS_EIGENVECTOR_V1_MARKER = -12345679
KS_EIGENVECTOR_V1_KIND = 28


def parse_band_out(path: Path):
    with path.open("r") as f:
        tokens = f.read().split()
    if len(tokens) < 4:
        raise ValueError(f"{path} does not contain a valid band_out header")
    return int(tokens[1]), int(tokens[2]), int(tokens[3])


def token_stream(file_obj):
    for line in file_obj:
        yield from line.split()


def next_token(tokens, path: Path, ik: int):
    try:
        return next(tokens)
    except StopIteration as exc:
        raise ValueError(f"{path}: k-point {ik} is truncated") from exc


def convert_file(in_path: Path, out_path: Path, dims):
    n_spins, n_spinor, n_states, n_basis_wfc = dims
    n_basis_ao = n_basis_wfc // n_spinor
    pairs_per_k = n_spins * n_spinor * n_states * n_basis_ao
    records = []

    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_tmp = tempfile.NamedTemporaryFile(
        "w+b", prefix="._ksvec_data_", dir=out_path.parent, delete=False
    )
    data_tmp_path = Path(data_tmp.name)

    try:
        with in_path.open("r") as fin, data_tmp:
            tokens = token_stream(fin)
            while True:
                try:
                    ik = int(next(tokens))
                except StopIteration:
                    break

                records.append((ik, data_tmp.tell()))
                values = array("d", [0.0]) * (2 * pairs_per_k)

                for iw in range(n_basis_ao):
                    for isoc in range(n_spinor):
                        for ib in range(n_states):
                            for ispin in range(n_spins):
                                re = float(next_token(tokens, in_path, ik))
                                im = float(next_token(tokens, in_path, ik))
                                dst = (((ispin * n_spinor + isoc) * n_states + ib)
                                       * n_basis_ao + iw) * 2
                                values[dst] = re
                                values[dst + 1] = im

                data_tmp.write(values.tobytes())

        table_offset = 6 * 4
        data_offset = table_offset + len(records) * (4 + 8)
        tmp_out = Path(
            tempfile.NamedTemporaryFile(
                "wb", prefix="._ksvec_v1_", dir=out_path.parent, delete=False
            ).name
        )
        try:
            with tmp_out.open("wb") as fout, data_tmp_path.open("rb") as fdata:
                fout.write(
                    struct.pack(
                        "=6i",
                        KS_EIGENVECTOR_V1_MARKER,
                        KS_EIGENVECTOR_V1_KIND,
                        len(records),
                        n_spins,
                        n_states,
                        n_basis_wfc,
                    )
                )
                for ik, offset in records:
                    fout.write(struct.pack("=iq", ik, data_offset + offset))
                shutil.copyfileobj(fdata, fout, 1024 * 1024)
            os.replace(tmp_out, out_path)
        finally:
            if tmp_out.exists():
                tmp_out.unlink()
    finally:
        if data_tmp_path.exists():
            data_tmp_path.unlink()

    return len(records)


def collect_input_files(input_dir: Path, prefix: str, suffix: str):
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file()
        and p.name.startswith(prefix)
        and not p.name.endswith(suffix)
        and not p.name.endswith(".v1tmp")
        and not p.name.startswith("legacy_")
    )


def resolve_dims(args):
    n_spinor = args.n_spinor
    band_out = args.band_out or args.input_dir / "band_out"
    if args.n_spins is None or args.n_states is None or (
        args.n_basis_wfc is None and args.n_aos is None
    ):
        if not band_out.is_file():
            raise ValueError(
                "cannot determine dimensions: provide --band-out or "
                "--n-spins, --n-states and --n-basis-wfc"
            )
        bo_spins, bo_states, bo_basis_wfc = parse_band_out(band_out)
    else:
        bo_spins = bo_states = bo_basis_wfc = None

    n_spins = args.n_spins if args.n_spins is not None else bo_spins
    n_states = args.n_states if args.n_states is not None else bo_states
    if args.n_aos is not None and args.n_basis_wfc is not None:
        raise ValueError("use only one of --n-aos or --n-basis-wfc")
    n_basis_wfc = (
        args.n_aos * n_spinor
        if args.n_aos is not None
        else args.n_basis_wfc if args.n_basis_wfc is not None
        else bo_basis_wfc
    )

    if min(n_spins, n_spinor, n_states, n_basis_wfc) <= 0:
        raise ValueError("all dimensions must be positive")
    if n_basis_wfc % n_spinor != 0:
        raise ValueError("n_basis_wfc must be divisible by n_spinor")
    return n_spins, n_spinor, n_states, n_basis_wfc


def main():
    parser = argparse.ArgumentParser(
        description="Convert legacy text KS_eigenvector files to binary v1 kind=28."
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    parser.add_argument("--prefix", default="KS_eigenvector")
    parser.add_argument("--suffix", default=".bin")
    parser.add_argument("--band-out", type=Path, default=None)
    parser.add_argument("--n-spins", type=int, default=None)
    parser.add_argument("--n-spinor", type=int, default=1)
    parser.add_argument("--n-states", type=int, default=None)
    parser.add_argument("--n-basis-wfc", type=int, default=None)
    parser.add_argument("--n-aos", type=int, default=None,
                        help="Legacy alias: AO basis count; multiplied by --n-spinor.")
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()

    try:
        if not args.input_dir.is_dir():
            raise ValueError(f"input directory does not exist: {args.input_dir}")

        dims = resolve_dims(args)
        output_dir = args.output_dir or args.input_dir
        files = collect_input_files(args.input_dir, args.prefix, args.suffix)
        if not files:
            raise ValueError(f"no files matching prefix '{args.prefix}' found")

        for in_path in files:
            if args.in_place:
                out_path = in_path
                tmp_out = in_path.with_name(in_path.name + ".v1tmp")
            else:
                out_path = output_dir / (in_path.stem + args.suffix)
                tmp_out = out_path

            n_kpts = convert_file(in_path, tmp_out, dims)
            if args.in_place:
                backup = in_path.with_name("legacy_" + in_path.name)
                os.replace(in_path, backup)
                os.replace(tmp_out, out_path)
                print(f"{in_path.name}: wrote {n_kpts} k-point(s), backup {backup.name}")
            else:
                print(f"{in_path.name}: wrote {n_kpts} k-point(s) -> {out_path.name}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
