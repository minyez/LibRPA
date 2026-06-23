#!/usr/bin/env python3
"""Convert legacy LibRPA structure metadata to current split files."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from typing import Iterator, List, Optional, Sequence, Tuple


CS_V1_MARKER = -10267453
SYMMETRY_CONVENTIONS = {"row", "col"}
BASIS_CONVENTIONS = {"aims", "abacus", "fallback"}


class ConversionError(RuntimeError):
    pass


@dataclass
class Symops:
    convention: str
    ops: List[Tuple[List[int], List[float]]]


@dataclass
class LegacyBz:
    nk: Tuple[int, int, int]
    kcart: List[Tuple[float, float, float]]
    full_to_q: List[int]


@dataclass
class Stru:
    lattice: List[List[float]]
    reciprocal: List[List[float]]
    atom_coords: List[Tuple[float, float, float]]
    atom_types: List[int]
    legacy_bz: Optional[LegacyBz]
    symops: Optional[Symops]


def parse_int(token: str, label: str) -> int:
    try:
        return int(token)
    except ValueError as exc:
        raise ConversionError(f"expected integer for {label}, got {token!r}") from exc


def parse_float(token: str, label: str) -> float:
    try:
        return float(token.replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise ConversionError(f"expected float for {label}, got {token!r}") from exc


def require(tokens: Sequence[str], pos: int, count: int, label: str) -> None:
    if pos + count > len(tokens):
        raise ConversionError(f"unexpected end of stru_out while reading {label}")


def is_symop_at(tokens: Sequence[str], pos: int) -> bool:
    return pos + 1 < len(tokens) and tokens[pos + 1].lower() in SYMMETRY_CONVENTIONS


def is_tail_boundary(tokens: Sequence[str], pos: int) -> bool:
    return pos == len(tokens) or is_symop_at(tokens, pos)


def parse_symops(tokens: Sequence[str], pos: int) -> Tuple[Symops, int]:
    require(tokens, pos, 2, "symmetry operation header")
    n_symops = parse_int(tokens[pos], "number of symmetry operations")
    convention = tokens[pos + 1].lower()
    if n_symops < 0 or convention not in SYMMETRY_CONVENTIONS:
        raise ConversionError("invalid symmetry operation header")
    pos += 2

    ops = []
    for _ in range(n_symops):
        require(tokens, pos, 12, "symmetry operation")
        rot = [parse_int(tokens[pos + i], "symmetry rotation") for i in range(9)]
        trans = [parse_float(tokens[pos + 9 + i], "symmetry translation") for i in range(3)]
        ops.append((rot, trans))
        pos += 12
    return Symops(convention, ops), pos


def try_legacy_bz_layout(
    tokens: Sequence[str],
    rows_start: int,
    nrows: int,
    nk_full: int,
) -> Optional[Tuple[List[int], int, int]]:
    rows_end = rows_start + 3 * nrows
    if rows_end > len(tokens):
        return None
    if nrows == nk_full and is_tail_boundary(tokens, rows_end):
        return list(range(1, nk_full + 1)), rows_end, nrows

    map_end = rows_end + nk_full
    if map_end > len(tokens) or not is_tail_boundary(tokens, map_end):
        return None
    mapping = [parse_int(tokens[rows_end + i], "legacy k-point mapping") for i in range(nk_full)]
    if any(rep < 1 or rep > nrows for rep in mapping):
        return None
    if nrows < nk_full and set(mapping) != set(range(1, nrows + 1)):
        raise ConversionError("legacy reduced k-point mapping misses a listed k-point")
    return mapping, map_end, nrows


def parse_legacy_bz(tokens: Sequence[str], pos: int) -> Tuple[LegacyBz, int]:
    require(tokens, pos, 3, "legacy k-grid")
    nk = tuple(parse_int(tokens[pos + i], "legacy k-grid") for i in range(3))
    if any(n <= 0 for n in nk):
        raise ConversionError(f"invalid legacy k-grid {nk}")
    nk_full = nk[0] * nk[1] * nk[2]
    rows_start = pos + 3

    layout = try_legacy_bz_layout(tokens, rows_start, nk_full, nk_full)
    if layout is None:
        for nrows in range(1, nk_full):
            layout = try_legacy_bz_layout(tokens, rows_start, nrows, nk_full)
            if layout is not None:
                break
    if layout is None:
        raise ConversionError("failed to locate legacy k-point rows in stru_out")

    mapping, end, nrows = layout
    kcart = []
    for i in range(nrows):
        row = rows_start + 3 * i
        kcart.append(
            (
                parse_float(tokens[row], "legacy k-point x"),
                parse_float(tokens[row + 1], "legacy k-point y"),
                parse_float(tokens[row + 2], "legacy k-point z"),
            )
        )
    return LegacyBz(nk, kcart, mapping), end


def parse_stru(path: Path) -> Stru:
    tokens = path.read_text(encoding="utf-8").split()
    pos = 0

    lattice = []
    reciprocal = []
    for label, target in (("lattice", lattice), ("reciprocal lattice", reciprocal)):
        for _ in range(3):
            require(tokens, pos, 3, label)
            target.append([parse_float(tokens[pos + i], label) for i in range(3)])
            pos += 3

    require(tokens, pos, 1, "number of atoms")
    n_atoms = parse_int(tokens[pos], "number of atoms")
    if n_atoms < 0:
        raise ConversionError("number of atoms must be non-negative")
    pos += 1

    atom_coords = []
    atom_types = []
    for _ in range(n_atoms):
        require(tokens, pos, 4, "atom row")
        atom_coords.append(tuple(parse_float(tokens[pos + i], "atom coordinate") for i in range(3)))
        atom_type = parse_int(tokens[pos + 3], "atom type")
        if atom_type <= 0:
            raise ConversionError("atom types must be 1-based positive integers")
        atom_types.append(atom_type)
        pos += 4

    legacy_bz = None
    symops = None
    if pos < len(tokens):
        if is_symop_at(tokens, pos):
            symops, pos = parse_symops(tokens, pos)
        else:
            legacy_bz, pos = parse_legacy_bz(tokens, pos)
            if pos < len(tokens):
                symops, pos = parse_symops(tokens, pos)
    if pos != len(tokens):
        raise ConversionError("unexpected trailing data in stru_out")

    return Stru(lattice, reciprocal, atom_coords, atom_types, legacy_bz, symops)


def format_stru(stru: Stru) -> str:
    lines = []
    for rows in (stru.lattice, stru.reciprocal):
        for row in rows:
            lines.append(" ".join(f"{value:24.16E}" for value in row) + "\n")
    lines.append(f"{len(stru.atom_types):8d}\n")
    for coords, atom_type in zip(stru.atom_coords, stru.atom_types):
        lines.append(" ".join(f"{value:24.16E}" for value in coords) + f" {atom_type:5d}\n")
    if stru.symops is not None:
        lines.append(f"{len(stru.symops.ops):7d}  {stru.symops.convention}\n")
        for rot, trans in stru.symops.ops:
            lines.append(
                " ".join(f"{value:3d}" for value in rot)
                + " "
                + " ".join(f"{value:12.5E}" for value in trans)
                + "\n"
            )
    return "".join(lines)


def backup(path: Path) -> Path:
    for i in range(1000):
        suffix = ".legacy_backup" if i == 0 else f".legacy_backup.{i}"
        target = path.with_name(path.name + suffix)
        if not target.exists():
            shutil.copy2(path, target)
            return target
    raise ConversionError(f"too many backup files for {path}")


def write_if_changed(path: Path, text: str) -> bool:
    old = path.read_text(encoding="utf-8") if path.exists() else None
    if old == text:
        return False
    if path.exists():
        backup(path)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return True


def k_fraction(index: int, nk: Tuple[int, int, int]) -> Tuple[float, float, float]:
    nk1, nk2, nk3 = nk
    i1 = index // (nk2 * nk3)
    rem = index % (nk2 * nk3)
    i2 = rem // nk3
    i3 = rem % nk3
    return i1 / nk1, i2 / nk2, i3 / nk3


def format_bz_sampling(bz: LegacyBz) -> str:
    nk_full = bz.nk[0] * bz.nk[1] * bz.nk[2]
    n_scf = len(bz.kcart)
    lines = [f"{bz.nk[0]:4d} {bz.nk[1]:4d} {bz.nk[2]:4d}\n"]

    if n_scf == nk_full:
        reps = []
        for rep in bz.full_to_q:
            if rep not in reps:
                reps.append(rep)
        rep_to_ibz = {rep: i + 1 for i, rep in enumerate(reps)}
        lines.append(f"{n_scf:7d} {len(reps):7d}\n")
        for index, (cart, rep) in enumerate(zip(bz.kcart, bz.full_to_q), 1):
            frac = k_fraction(index - 1, bz.nk)
            lines.append(format_bz_row(index, 1.0 / nk_full, frac, cart, rep_to_ibz[rep], rep))
    else:
        counts = Counter(bz.full_to_q)
        first_full = {}
        for full_index, rep in enumerate(bz.full_to_q):
            first_full.setdefault(rep, full_index)
        lines.append(f"{n_scf:7d} {n_scf:7d}\n")
        for index, cart in enumerate(bz.kcart, 1):
            frac = k_fraction(first_full[index], bz.nk)
            lines.append(format_bz_row(index, counts[index] / nk_full, frac, cart, index, index))

    return "".join(lines)


def format_bz_row(
    index: int,
    weight: float,
    frac: Tuple[float, float, float],
    cart: Tuple[float, float, float],
    ibz: int,
    rep: int,
) -> str:
    return (
        f"{index:7d} {weight:20.12E} "
        f"{frac[0]:20.12E} {frac[1]:20.12E} {frac[2]:20.12E} "
        f"{cart[0]:20.12E} {cart[1]:20.12E} {cart[2]:20.12E} "
        f"{ibz:7d} {rep:7d}\n"
    )


def trim_bz_tail(path: Path) -> bool:
    if not path.exists():
        return False
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    nonempty = [i for i, line in enumerate(lines) if line.split()]
    if len(nonempty) < 2:
        raise ConversionError(f"{path}: invalid bz_sampling_out")
    fields = lines[nonempty[1]].split()
    if len(fields) < 2:
        raise ConversionError(f"{path}: invalid bz_sampling_out count line")
    n_scf = parse_int(fields[0], "BZ sampling SCF k-point count")
    keep = 2 + n_scf
    if len(nonempty) <= keep:
        return False
    cut_line = nonempty[keep - 1] + 1
    if all(not line.strip() for line in lines[cut_line:]):
        return False
    text = "".join(lines[:cut_line])
    if not text.endswith("\n"):
        text += "\n"
    return write_if_changed(path, text)


def token_stream(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield from line.split()


def next_token(tokens: Iterator[str], path: Path, label: str) -> str:
    try:
        return next(tokens)
    except StopIteration as exc:
        raise ConversionError(f"{path}: unexpected EOF while reading {label}") from exc


def put_size(sizes: dict[int, int], atom: int, value: int, path: Path, label: str) -> None:
    if atom < 0:
        raise ConversionError(f"{path}: invalid atom index")
    old = sizes.get(atom)
    if old is not None and old != value:
        raise ConversionError(f"{path}: inconsistent {label} size for atom {atom + 1}")
    sizes[atom] = value


def read_text_cs_basis(path: Path) -> Tuple[int, dict[int, int], dict[int, int]]:
    tokens = token_stream(path)
    natom = parse_int(next_token(tokens, path, "Cs atom count"), "Cs atom count")
    ncell = parse_int(next_token(tokens, path, "Cs cell count"), "Cs cell count")
    if natom <= 0 or ncell < 0:
        raise ConversionError(f"{path}: invalid Cs header")

    wfc: dict[int, int] = {}
    aux: dict[int, int] = {}
    while len(wfc) < natom or len(aux) < natom:
        try:
            ia1 = parse_int(next(tokens), "Cs atom index") - 1
        except StopIteration:
            break
        ia2 = parse_int(next_token(tokens, path, "Cs atom index"), "Cs atom index") - 1
        for _ in range(3):
            next_token(tokens, path, "Cs cell index")
        n_i = parse_int(next_token(tokens, path, "Cs block n_i"), "Cs block n_i")
        n_j = parse_int(next_token(tokens, path, "Cs block n_j"), "Cs block n_j")
        n_mu = parse_int(next_token(tokens, path, "Cs block n_mu"), "Cs block n_mu")
        if min(n_i, n_j, n_mu) <= 0 or ia1 >= natom or ia2 >= natom:
            raise ConversionError(f"{path}: invalid Cs block")
        put_size(wfc, ia1, n_i, path, "wave-function basis")
        put_size(wfc, ia2, n_j, path, "wave-function basis")
        put_size(aux, ia1, n_mu, path, "auxiliary basis")
        if len(wfc) == natom and len(aux) == natom:
            break
        for _ in range(n_i * n_j * n_mu):
            next_token(tokens, path, "Cs payload")
    return natom, wfc, aux


def looks_binary(path: Path) -> bool:
    with path.open("rb") as handle:
        head = handle.read(64)
    if len(head) >= 4 and struct.unpack("=i", head[:4])[0] == CS_V1_MARKER:
        return True
    return any(byte < 32 and byte not in (9, 10, 13) for byte in head)


def read_binary_cs_basis(path: Path) -> Tuple[int, dict[int, int], dict[int, int]]:
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) != 12:
            raise ConversionError(f"{path}: truncated binary Cs header")
        natom, ncell, nblocks = struct.unpack("=3i", header)
        if natom == CS_V1_MARKER:
            raise ConversionError(f"{path}: reader-v1 Cs cannot be used to infer basis sizes")
        if natom <= 0 or ncell < 0 or nblocks < 0:
            raise ConversionError(f"{path}: invalid binary Cs header")

        wfc: dict[int, int] = {}
        aux: dict[int, int] = {}
        for _ in range(nblocks):
            data = handle.read(32)
            if len(data) != 32:
                raise ConversionError(f"{path}: truncated binary Cs block header")
            ia1, ia2, _r1, _r2, _r3, n_i, n_j, n_mu = struct.unpack("=8i", data)
            ia1 -= 1
            ia2 -= 1
            if min(n_i, n_j, n_mu) <= 0 or ia1 >= natom or ia2 >= natom:
                raise ConversionError(f"{path}: invalid binary Cs block")
            put_size(wfc, ia1, n_i, path, "wave-function basis")
            put_size(wfc, ia2, n_j, path, "wave-function basis")
            put_size(aux, ia1, n_mu, path, "auxiliary basis")
            if len(wfc) == natom and len(aux) == natom:
                break
            handle.seek(n_i * n_j * n_mu * 8, os.SEEK_CUR)
    return natom, wfc, aux


def discover_cs_files(input_dir: Path, prefix: str) -> List[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.name.startswith(prefix) and not path.name.startswith("legacy_")
    )


def infer_basis_from_cs(input_dir: Path, prefix: str, n_atoms: int) -> Tuple[List[int], List[int]]:
    files = discover_cs_files(input_dir, prefix)
    if not files:
        raise ConversionError(f"no Cs files with prefix {prefix!r} found under {input_dir}")

    wfc: dict[int, int] = {}
    aux: dict[int, int] = {}
    for path in files:
        natom, one_wfc, one_aux = (
            read_binary_cs_basis(path) if looks_binary(path) else read_text_cs_basis(path)
        )
        if natom != n_atoms:
            raise ConversionError(f"{path}: Cs atom count {natom} != stru_out atom count {n_atoms}")
        for atom, value in one_wfc.items():
            put_size(wfc, atom, value, path, "wave-function basis")
        for atom, value in one_aux.items():
            put_size(aux, atom, value, path, "auxiliary basis")
        if len(wfc) == n_atoms and len(aux) == n_atoms:
            break

    if len(wfc) != n_atoms or len(aux) != n_atoms:
        raise ConversionError("could not infer all atom basis sizes from Cs files")
    return [wfc[i] for i in range(n_atoms)], [aux[i] for i in range(n_atoms)]


def basis_text(atom_types: Sequence[int], atom_sizes: Sequence[int], convention: str) -> str:
    ntypes = max(atom_types)
    type_sizes = [0] * ntypes
    for atom_type, size in zip(atom_types, atom_sizes):
        old = type_sizes[atom_type - 1]
        if old not in (0, size):
            raise ConversionError(f"basis size differs between atoms of type {atom_type}")
        type_sizes[atom_type - 1] = size

    lines = [f"{ntypes:10d} {sum(atom_sizes):10d}    {convention}\n"]
    for atom_type, size in enumerate(type_sizes, 1):
        lines.append(f"{atom_type:10d} {size:10d}\n")
    return "".join(lines)


def maybe_write_basis_files(args: argparse.Namespace, input_dir: Path, stru: Stru) -> List[str]:
    basis_path = input_dir / args.basis_name
    wfc_path = input_dir / args.basis_wfc_name
    aux_path = input_dir / args.basis_aux_name
    if basis_path.exists():
        return [f"kept {basis_path.name}; split basis files skipped"]
    if wfc_path.exists() and aux_path.exists():
        return [f"kept existing {wfc_path.name} and {aux_path.name}"]

    wfc, aux = infer_basis_from_cs(input_dir, args.cs_prefix, len(stru.atom_types))
    changed = []
    if write_if_changed(wfc_path, basis_text(stru.atom_types, wfc, args.basis_convention)):
        changed.append(wfc_path.name)
    if write_if_changed(aux_path, basis_text(stru.atom_types, aux, args.basis_convention)):
        changed.append(aux_path.name)
    return [f"wrote {', '.join(changed)}"] if changed else ["basis files already current"]


def convert(args: argparse.Namespace) -> List[str]:
    input_dir = args.input_dir.resolve()
    stru_path = input_dir / args.stru_name
    if not stru_path.exists():
        raise ConversionError(f"{stru_path} does not exist")

    actions = []
    stru = parse_stru(stru_path)
    if stru.legacy_bz is not None:
        if write_if_changed(stru_path, format_stru(stru)):
            actions.append(f"rewrote {args.stru_name}")
    else:
        actions.append(f"{args.stru_name} already has no legacy BZ tail")

    bz_path = input_dir / args.bz_name
    if bz_path.exists():
        if trim_bz_tail(bz_path):
            actions.append(f"trimmed {args.bz_name}")
        else:
            actions.append(f"{args.bz_name} already current")
    elif stru.legacy_bz is not None:
        write_if_changed(bz_path, format_bz_sampling(stru.legacy_bz))
        actions.append(f"wrote {args.bz_name}")
    else:
        actions.append(f"skipped {args.bz_name}; no legacy BZ data found")

    actions.extend(maybe_write_basis_files(args, input_dir, stru))
    return actions


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "stru_out").write_text(
            """1 0 0
0 1 0
0 0 1
1 0 0
0 1 0
0 0 1
2
0 0 0 1
0.5 0 0 2
2 1 1
0 0 0
0.5 0 0
1
1
1 row
1 0 0 0 1 0 0 0 1 0 0 0
""",
            encoding="utf-8",
        )
        (root / "Cs_data_1.txt").write_text(
            "2 1\n"
            "1 1 0 0 0 2 2 3\n" + " ".join(["0"] * 12) + "\n"
            "2 2 0 0 0 4 4 5\n" + " ".join(["0"] * 80) + "\n",
            encoding="utf-8",
        )

        args = build_parser().parse_args([str(root)])
        actions = convert(args)
        assert "rewrote stru_out" in actions
        assert "wrote bz_sampling_out" in actions
        assert (root / "stru_out.legacy_backup").exists()
        assert "2 1 1" not in (root / "stru_out").read_text(encoding="utf-8")
        assert len((root / "bz_sampling_out").read_text(encoding="utf-8").splitlines()) == 4
        assert (root / "basis_wfc_out").read_text(encoding="utf-8").split()[1] == "6"
        assert (root / "basis_aux_out").read_text(encoding="utf-8").split()[1] == "8"

        with (root / "bz_sampling_out").open("a", encoding="utf-8") as handle:
            handle.write("      1       1   1.000000000000E+00\n")
        assert trim_bz_tail(root / "bz_sampling_out")
        assert len((root / "bz_sampling_out").read_text(encoding="utf-8").splitlines()) == 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert legacy stru_out metadata into current stru_out, bz_sampling_out, and split basis files."
    )
    parser.add_argument("input_dir", nargs="?", type=Path, default=Path("."))
    parser.add_argument("--stru-name", default="stru_out")
    parser.add_argument("--bz-name", default="bz_sampling_out")
    parser.add_argument("--basis-name", default="basis_out")
    parser.add_argument("--basis-wfc-name", default="basis_wfc_out")
    parser.add_argument("--basis-aux-name", default="basis_aux_out")
    parser.add_argument("--cs-prefix", default="Cs_data")
    parser.add_argument("--basis-convention",
                        default="fallback", choices=BASIS_CONVENTIONS)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.self_test:
            self_test()
            print("self-test passed")
            return 0
        for action in convert(args):
            print(action)
        return 0
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
