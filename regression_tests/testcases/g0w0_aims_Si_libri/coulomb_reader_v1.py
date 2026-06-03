#!/usr/bin/env python3
"""After extracting dataset.tar.gz, you can move this script and try to read
Coulomb matrix data files, e.g. coulomb_full_iq_18.dat.
"""
import argparse
import struct
from pathlib import Path

import numpy as np


V2_MARKER = 20129432
REAL_FLAG = 0
COMPLEX_FLAG = 1


def detect_endian(path: Path) -> str:
    with path.open("rb") as f:
        raw = f.read(4)
    for endian in ("<", ">"):
        (marker,) = struct.unpack(endian + "i", raw)
        if marker == V2_MARKER:
            return endian
    raise ValueError(f"{path}: not a Coulomb v1 file")


def atom_pair_index(i: int, j: int, natoms: int) -> int:
    if i > j:
        raise ValueError("expects i <= j")
    return i * natoms - i * (i - 1) // 2 + (j - i)


class CoulombV1:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.endian = detect_endian(self.path)

        with self.path.open("rb") as f:
            header = f.read(5 * 4)
            self.marker, self.iq, self.naux, self.value_flag, self.natoms = struct.unpack(
                self.endian + "iiiii", header
            )
            self.atom_naux = list(
                struct.unpack(self.endian + f"{self.natoms}i", f.read(4 * self.natoms))
            )

        if sum(self.atom_naux) != self.naux:
            raise ValueError("bad v2 file: per-atom Naux does not sum to total Naux")

        self.atom_offsets = np.cumsum([0] + self.atom_naux)
        self.header_bytes = 5 * 4 + 4 * self.natoms
        self.dtype = np.dtype(self.endian + ("c16" if self.value_flag == COMPLEX_FLAG else "f8"))

        self.block_offsets = {}
        offset = 0
        for i in range(self.natoms):
            for j in range(i, self.natoms):
                self.block_offsets[(i, j)] = offset
                offset += self.atom_naux[i] * self.atom_naux[j]

        expected_size = self.header_bytes + offset * self.dtype.itemsize
        if self.path.stat().st_size != expected_size:
            raise ValueError(f"bad v2 file size: expected {expected_size}")

        self.payload = np.memmap(
            self.path,
            mode="r",
            dtype=self.dtype,
            offset=self.header_bytes,
            shape=(offset,),
        )

    def block(self, i: int, j: int) -> np.ndarray:
        """Return atom-pair block using 0-based atom indices.

        For i > j, returns the Hermitian-conjugated counterpart.
        """
        if i > j:
            return self.block(j, i).T.conj()

        nrow = self.atom_naux[i]
        ncol = self.atom_naux[j]
        start = self.block_offsets[(i, j)]
        stop = start + nrow * ncol
        return np.asarray(self.payload[start:stop]).reshape(nrow, ncol)

    def full_matrix(self) -> np.ndarray:
        dtype = np.complex128 if self.value_flag == COMPLEX_FLAG else np.float64
        mat = np.empty((self.naux, self.naux), dtype=dtype)

        for i in range(self.natoms):
            r0, r1 = self.atom_offsets[i], self.atom_offsets[i + 1]
            for j in range(i, self.natoms):
                c0, c1 = self.atom_offsets[j], self.atom_offsets[j + 1]
                blk = self.block(i, j)
                mat[r0:r1, c0:c1] = blk
                if i != j:
                    mat[c0:c1, r0:r1] = blk.T.conj()

        return mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--block", nargs=2, type=int, metavar=("I", "J"))
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    vq = CoulombV1(args.file)
    print(f"iq={vq.iq}, Naux={vq.naux}, Natoms={vq.natoms}, atom_naux={vq.atom_naux}")

    if args.block:
        i, j = args.block
        print(vq.block(i, j))
    if args.full:
        mat = vq.full_matrix()
        print(mat)
        print("shape:", mat.shape)


if __name__ == "__main__":
    main()
