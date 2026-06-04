#!/usr/bin/env python3
"""Convert LibRPA Coulomb matrix files to one binary file per q-point.

The input files are the legacy ``<input_prefix>_*.txt`` files consumed by
``driver/read_data.cpp``.  Both supported legacy layouts are accepted:

* ASCII text: one ``n_irk_points`` header, then rectangular matrix blocks.
* Binary: ``n_irk_points`` and ``n_irk_points_local`` int32 headers, then
  rectangular matrix blocks with complex<double> payloads.

The output file ``<output_prefix>_<iq>.dat`` is binary.
Reader version 1 stores deterministic atom-pair blocks:

* int32 marker for target reader version
* int32 q-point index, 1-based
* int32 number of auxiliary basis functions, Naux
* int32 value type flag: 0 for real double data, 1 for complex double data
* int32 number of atoms, Natoms
* int32 number of stored atom-pair blocks, Nblocks
* Natoms int32 values with per-atom auxiliary basis sizes
* Nblocks records of int32 atom-pair block index and int64 byte offset
* dense row-major atom-pair blocks for (0,0), (0,1), ..., (0,Natoms-1),
  (1,1), ..., with full dense diagonal blocks

When legacy inputs are known to be full or upper-triangular, ``--skip-lower``
can be used to ignore lower-triangular entries instead of reading them into the
new output.

For large streaming conversions split across multiple legacy input files,
``--workers`` can process those input files concurrently while sharing the
per-q output files through positional writes.

When possible, the script also writes ``bz_sampling_out`` from the legacy
``stru_out`` k-grid tail and the q-point weights in the Coulomb block headers.
This is skipped if ``bz_sampling_out`` already exists or ``stru_out`` is absent.
"""

from __future__ import annotations

import argparse
import array
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import struct
import sys
import threading
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple


REAL_FLAG = 0
COMPLEX_FLAG = 1
Q_WEIGHT_TOL = 1e-10

# Version markers. Must match those in driver/read_data.cpp
COULOMB_V1_MARKER = 20129433
COULOMB_V1_HEADER_BASE_SIZE = 6 * 4
COULOMB_V1_BLOCK_RECORD_SIZE = 4 + 8


class ConversionError(RuntimeError):
    """Raised when an input Coulomb file is malformed or incomplete."""


class TokenStream:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = path.open("r", encoding="utf-8")
        self._tokens = self._iter_tokens()

    def close(self) -> None:
        self._handle.close()

    def _iter_tokens(self) -> Iterator[Tuple[str, int]]:
        for line_no, line in enumerate(self._handle, 1):
            for token in line.split():
                yield token, line_no

    def next_optional(self) -> Optional[Tuple[str, int]]:
        return next(self._tokens, None)

    def next_token(self, what: str) -> Tuple[str, int]:
        token = self.next_optional()
        if token is None:
            raise ConversionError(f"{self.path}: unexpected EOF while reading {what}")
        return token

    def next_int(self, what: str) -> int:
        token, line_no = self.next_token(what)
        try:
            return int(token)
        except ValueError as exc:
            raise ConversionError(
                f"{self.path}:{line_no}: expected integer for {what}, got {token!r}"
            ) from exc

    def next_float(self, what: str) -> float:
        token, line_no = self.next_token(what)
        try:
            return float(token.replace("D", "E").replace("d", "e"))
        except ValueError as exc:
            raise ConversionError(
                f"{self.path}:{line_no}: expected float for {what}, got {token!r}"
            ) from exc


class CoulombMatrix:
    def __init__(self, iq: int, naux: int, check_complete: bool) -> None:
        if iq <= 0:
            raise ConversionError(f"invalid q-point index {iq}; expected a 1-based index")
        if naux <= 0:
            raise ConversionError(f"invalid Naux {naux} for q-point {iq}")

        self.iq = iq
        self.naux = naux
        self._ncell = naux * naux
        self.values = array.array("d", [0.0]) * (2 * self._ncell)
        self.present = bytearray(self._ncell)
        self.check_complete = check_complete

    def set_row_segment(self, row: int, col_begin: int, row_values: array.array) -> None:
        nvalues = len(row_values)
        if nvalues % 2 != 0:
            raise ConversionError("internal error: complex row segment has odd double count")
        ncol = nvalues // 2
        if row < 0 or row >= self.naux or col_begin < 0 or col_begin + ncol > self.naux:
            raise ConversionError(
                f"iq={self.iq}: block row/column range is outside a {self.naux}x{self.naux} matrix"
            )

        value_begin = 2 * (row * self.naux + col_begin)
        self.values[value_begin:value_begin + nvalues] = row_values

        mask_begin = row * self.naux + col_begin
        self.present[mask_begin:mask_begin + ncol] = b"\x01" * ncol

    def missing_count(self) -> int:
        if not self.check_complete:
            return 0
        missing = 0
        for row in range(self.naux):
            for col in range(row, self.naux):
                if not self.has_upper_value(row, col):
                    missing += 1
        return missing

    def first_missing(self) -> Optional[Tuple[int, int]]:
        if not self.check_complete:
            return None
        for row in range(self.naux):
            for col in range(row, self.naux):
                if not self.has_upper_value(row, col):
                    return row + 1, col + 1
        return None

    def has_upper_value(self, row: int, col: int) -> bool:
        if self.present[row * self.naux + col]:
            return True
        return row != col and bool(self.present[col * self.naux + row])

    def upper_value(self, row: int, col: int) -> Tuple[float, float]:
        upper_index = row * self.naux + col
        if self.present[upper_index]:
            value_begin = 2 * upper_index
            return self.values[value_begin], self.values[value_begin + 1]

        if row != col:
            lower_index = col * self.naux + row
            if self.present[lower_index]:
                value_begin = 2 * lower_index
                return self.values[value_begin], -self.values[value_begin + 1]

        return 0.0, 0.0

    def max_abs_imag(self) -> float:
        max_imag = 0.0
        for row in range(self.naux):
            for col in range(row, self.naux):
                _real, imag = self.upper_value(row, col)
                if abs(imag) > max_imag:
                    max_imag = abs(imag)
        return max_imag

    def output_kind(self, real_tol: float, force_complex: bool) -> Tuple[int, str]:
        if force_complex or self.max_abs_imag() > real_tol:
            return COMPLEX_FLAG, "complex"
        return REAL_FLAG, "real"

    def write(
        self, target_reader_version: int,
        output_path: Path,
        endian: str,
        real_tol: float,
        force_complex: bool,
        overwrite: bool,
        atom_layout: Optional["AtomLayout"] = None,
    ) -> Tuple[str, int]:
        if target_reader_version != 1:
            raise ValueError("Unknown target reader version")
        if atom_layout is None:
            raise ConversionError("reader v1 output requires per-atom auxiliary sizes")
        return self._write_1(
            output_path, endian, real_tol, force_complex, overwrite, atom_layout
        )

    def _write_1(
        self,
        output_path: Path,
        endian: str,
        real_tol: float,
        force_complex: bool,
        overwrite: bool,
        atom_layout: "AtomLayout",
    ) -> Tuple[str, int]:
        atom_layout.validate_naux(self.naux, output_path)
        if output_path.exists() and not overwrite:
            raise ConversionError(
                f"{output_path} already exists. Please clean up old output files manually "
                "before rerunning."
            )

        flag, kind = self.output_kind(real_tol, force_complex)
        tmp_path = output_path.with_name(output_path.name + ".tmp")
        try:
            with tmp_path.open("wb") as handle:
                value_bytes = 16 if flag == COMPLEX_FLAG else 8
                handle.write(
                    struct.pack(
                        endian + "iiiiii",
                        COULOMB_V1_MARKER,
                        self.iq,
                        self.naux,
                        flag,
                        atom_layout.natoms,
                        atom_layout.pair_count(),
                    )
                )
                handle.write(struct.pack(endian + f"{atom_layout.natoms}i", *atom_layout.atom_naux))
                handle.write(atom_layout.block_table_bytes(endian, value_bytes))
                self.write_atom_pair_blocks(handle, endian, flag, atom_layout)
            os.replace(tmp_path, output_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return kind, output_path.stat().st_size

    def write_atom_pair_blocks(
        self,
        handle,
        endian: str,
        flag: int,
        atom_layout: "AtomLayout",
    ) -> None:
        for i_atom in range(atom_layout.natoms):
            row_begin = atom_layout.offsets[i_atom]
            for j_atom in range(i_atom, atom_layout.natoms):
                col_begin = atom_layout.offsets[j_atom]
                nrow = atom_layout.atom_naux[i_atom]
                ncol = atom_layout.atom_naux[j_atom]
                for local_row in range(nrow):
                    row = row_begin + local_row
                    row_values = array.array("d")
                    for local_col in range(ncol):
                        col = col_begin + local_col
                        if row <= col:
                            real, imag = self.upper_value(row, col)
                        else:
                            real, imag = self.upper_value(col, row)
                            imag = -imag
                        row_values.append(real)
                        if flag == COMPLEX_FLAG:
                            row_values.append(imag)
                    write_doubles(handle, row_values, endian)


def endian_prefix(name: str) -> str:
    if name == "native":
        return "<" if sys.byteorder == "little" else ">"
    if name == "little":
        return "<"
    if name == "big":
        return ">"
    raise ValueError(f"unsupported endian setting {name!r}")


def need_byteswap(endian: str) -> bool:
    return (endian == "<" and sys.byteorder != "little") or (
        endian == ">" and sys.byteorder != "big"
    )


def write_doubles(handle, values: array.array, endian: str) -> None:
    if need_byteswap(endian):
        swapped = array.array("d", values)
        swapped.byteswap()
        handle.write(swapped.tobytes())
    else:
        handle.write(values.tobytes())


def doubles_to_bytes(values: array.array, endian: str) -> bytes:
    if need_byteswap(endian):
        swapped = array.array("d", values)
        swapped.byteswap()
        return swapped.tobytes()
    return values.tobytes()


def read_exact(handle, nbytes: int, path: Path, what: str) -> bytes:
    data = handle.read(nbytes)
    if len(data) != nbytes:
        raise ConversionError(f"{path}: truncated file while reading {what}")
    return data


def validate_block(
    path: Path,
    naux: int,
    row_begin: int,
    row_end: int,
    col_begin: int,
    col_end: int,
    iq: int,
) -> None:
    if naux <= 0:
        raise ConversionError(f"{path}: invalid Naux {naux}")
    if iq <= 0:
        raise ConversionError(f"{path}: invalid q-point index {iq}")
    if row_begin <= 0 or row_end < row_begin or row_end > naux:
        raise ConversionError(
            f"{path}: invalid row range {row_begin}:{row_end} for Naux={naux}"
        )
    if col_begin <= 0 or col_end < col_begin or col_end > naux:
        raise ConversionError(
            f"{path}: invalid column range {col_begin}:{col_end} for Naux={naux}"
        )


def atom_pair_index(i_atom: int, j_atom: int, natoms: int) -> int:
    if i_atom > j_atom:
        raise ValueError("atom-pair index expects i_atom <= j_atom")
    return i_atom * natoms - i_atom * (i_atom - 1) // 2 + (j_atom - i_atom)


class AtomLayout:
    def __init__(self, atom_naux: Sequence[int]) -> None:
        if not atom_naux:
            raise ConversionError("reader v1 output requires at least one atom auxiliary size")
        self.atom_naux = [int(nb) for nb in atom_naux]
        if any(nb <= 0 for nb in self.atom_naux):
            raise ConversionError(f"invalid per-atom auxiliary sizes: {self.atom_naux}")
        self.natoms = len(self.atom_naux)
        self.offsets = [0]
        for nb in self.atom_naux:
            self.offsets.append(self.offsets[-1] + nb)
        self.naux = self.offsets[-1]
        self.pair_offsets = [0] * (self.natoms * (self.natoms + 1) // 2 + 1)
        payload_offset = 0
        for i_atom in range(self.natoms):
            for j_atom in range(i_atom, self.natoms):
                index = atom_pair_index(i_atom, j_atom, self.natoms)
                self.pair_offsets[index] = payload_offset
                payload_offset += self.atom_naux[i_atom] * self.atom_naux[j_atom]
        self.pair_offsets[-1] = payload_offset

    def pair_count(self) -> int:
        return self.natoms * (self.natoms + 1) // 2

    def validate_naux(self, naux: int, path: Path) -> None:
        if self.naux != naux:
            raise ConversionError(
                f"{path}: reader v1 atom auxiliary sizes sum to {self.naux}, "
                f"but Coulomb block has Naux={naux}"
            )

    def atom_for_aux(self, index: int) -> Tuple[int, int]:
        if index < 0 or index >= self.naux:
            raise ConversionError(
                f"global auxiliary index {index} is outside [0, {self.naux})"
            )
        atom = bisect_right(self.offsets, index) - 1
        return atom, index - self.offsets[atom]

    def pair_payload_offset(self, i_atom: int, j_atom: int) -> int:
        return self.pair_offsets[atom_pair_index(i_atom, j_atom, self.natoms)]

    def pair_value_offset(self, i_atom: int, j_atom: int, i_local: int, j_local: int) -> int:
        return (
            self.pair_payload_offset(i_atom, j_atom)
            + i_local * self.atom_naux[j_atom]
            + j_local
        )

    def payload_value_count(self) -> int:
        return self.pair_offsets[-1]

    def payload_byte_offset(self) -> int:
        return (
            COULOMB_V1_HEADER_BASE_SIZE
            + 4 * self.natoms
            + self.pair_count() * COULOMB_V1_BLOCK_RECORD_SIZE
        )

    def block_byte_offset(self, pair_index: int, value_bytes: int) -> int:
        return self.payload_byte_offset() + self.pair_offsets[pair_index] * value_bytes

    def iter_pair_indices(self) -> Iterator[int]:
        return iter(range(self.pair_count()))

    def block_table_bytes(self, endian: str, value_bytes: int) -> bytes:
        table = bytearray()
        for pair_index in self.iter_pair_indices():
            table.extend(
                struct.pack(
                    endian + "iq",
                    pair_index,
                    self.block_byte_offset(pair_index, value_bytes),
                )
            )
        return bytes(table)


class StreamingCoulombOutput:
    def __init__(
        self,
        output_dir: Path,
        output_prefix: str,
        iq: int,
        naux: int,
        endian: str,
        target_reader_version: int,
        atom_layout: Optional[AtomLayout] = None,
    ) -> None:
        if target_reader_version != 1:
            raise ValueError("Unknown target reader version")
        self.iq = iq
        self.naux = naux
        self.endian = endian
        self.target_reader_version = target_reader_version
        self.atom_layout = atom_layout
        if atom_layout is None:
            raise ConversionError("reader v1 streaming output requires per-atom auxiliary sizes")
        atom_layout.validate_naux(naux, output_dir / f"{output_prefix}_{iq}.dat")
        self.path = output_dir / f"{output_prefix}_{iq}.dat"
        self.tmp_path = self.path.with_name(self.path.name + ".tmp")
        if self.path.exists() or self.tmp_path.exists():
            raise ConversionError(
                f"{self.path} or its temporary file already exists. Please clean up "
                "old output files manually before rerunning."
            )
        self.fd = os.open(self.tmp_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
        header = struct.pack(
            endian + "iiiiii",
            COULOMB_V1_MARKER,
            iq,
            naux,
            COMPLEX_FLAG,
            atom_layout.natoms,
            atom_layout.pair_count(),
        )
        os.write(self.fd, header)
        os.write(
            self.fd,
            struct.pack(endian + f"{atom_layout.natoms}i", *atom_layout.atom_naux),
        )
        os.write(self.fd, atom_layout.block_table_bytes(endian, 16))
        os.ftruncate(
            self.fd,
            atom_layout.payload_byte_offset()
            + atom_layout.payload_value_count() * 16,
        )
        self.closed = False

    def _offset(self, row: int, col: int) -> int:
        assert self.atom_layout is not None
        i_atom, i_local = self.atom_layout.atom_for_aux(row)
        j_atom, j_local = self.atom_layout.atom_for_aux(col)
        if i_atom <= j_atom:
            return self._offset_by_local(i_atom, j_atom, i_local, j_local)
        return self._offset_by_local(j_atom, i_atom, j_local, i_local)

    def _offset_by_local(
        self,
        i_atom: int,
        j_atom: int,
        i_local: int,
        j_local: int,
    ) -> int:
        assert self.atom_layout is not None
        return (
            self.atom_layout.payload_byte_offset()
            + self.atom_layout.pair_value_offset(i_atom, j_atom, i_local, j_local) * 16
        )

    def write_pair(self, row: int, col: int, real: float, imag: float) -> None:
        assert self.atom_layout is not None
        i_atom, i_local = self.atom_layout.atom_for_aux(row)
        j_atom, j_local = self.atom_layout.atom_for_aux(col)
        if i_atom < j_atom:
            payload = struct.pack(self.endian + "dd", real, imag)
            os.pwrite(
                self.fd,
                payload,
                self._offset_by_local(i_atom, j_atom, i_local, j_local),
            )
        elif i_atom > j_atom:
            payload = struct.pack(self.endian + "dd", real, -imag)
            os.pwrite(
                self.fd,
                payload,
                self._offset_by_local(j_atom, i_atom, j_local, i_local),
            )
        else:
            payload = struct.pack(self.endian + "dd", real, imag)
            os.pwrite(
                self.fd,
                payload,
                self._offset_by_local(i_atom, i_atom, i_local, j_local),
            )
            if i_local != j_local:
                payload_conj = struct.pack(self.endian + "dd", real, -imag)
                os.pwrite(
                    self.fd,
                    payload_conj,
                    self._offset_by_local(i_atom, i_atom, j_local, i_local),
                )

    def write_row_segment(
        self,
        row: int,
        col_begin: int,
        row_values: array.array,
        write_lower_part: bool,
    ) -> None:
        assert self.atom_layout is not None
        ncol = len(row_values) // 2
        row_atom, row_local = self.atom_layout.atom_for_aux(row)
        col = col_begin
        while col < col_begin + ncol:
            col_atom, col_local = self.atom_layout.atom_for_aux(col)
            span_end = min(
                col_begin + ncol,
                self.atom_layout.offsets[col_atom + 1],
            )
            span_pairs_begin = 2 * (col - col_begin)
            span_pairs_end = 2 * (span_end - col_begin)

            if row_atom < col_atom:
                os.pwrite(
                    self.fd,
                    doubles_to_bytes(row_values[span_pairs_begin:span_pairs_end], self.endian),
                    self._offset_by_local(row_atom, col_atom, row_local, col_local),
                )
            elif row_atom == col_atom:
                exact_col = col
                exact_pairs_begin = span_pairs_begin
                if not write_lower_part and exact_col < row:
                    skip_cols = min(row - exact_col, span_end - exact_col)
                    exact_col += skip_cols
                    exact_pairs_begin += 2 * skip_cols
                if exact_col < span_end:
                    exact_local = exact_col - self.atom_layout.offsets[col_atom]
                    os.pwrite(
                        self.fd,
                        doubles_to_bytes(row_values[exact_pairs_begin:span_pairs_end], self.endian),
                        self._offset_by_local(row_atom, row_atom, row_local, exact_local),
                    )
                    for col2 in range(exact_col, span_end):
                        col2_local = col2 - self.atom_layout.offsets[col_atom]
                        if col2_local == row_local:
                            continue
                        pair_begin = 2 * (col2 - col_begin)
                        real = row_values[pair_begin]
                        imag = row_values[pair_begin + 1]
                        os.pwrite(
                            self.fd,
                            struct.pack(self.endian + "dd", real, -imag),
                            self._offset_by_local(row_atom, row_atom, col2_local, row_local),
                        )
            elif write_lower_part:
                for col2 in range(col, span_end):
                    pair_begin = 2 * (col2 - col_begin)
                    self.write_pair(
                        row,
                        col2,
                        row_values[pair_begin],
                        row_values[pair_begin + 1],
                    )
            col = span_end

    def write_lower_transpose_tile(
        self,
        row_begin: int,
        col_begin: int,
        ncol: int,
        tile_values: array.array,
        tile_nrow: int,
    ) -> None:
        for local_row in range(tile_nrow):
            row = row_begin + local_row
            for local_col in range(ncol):
                col = col_begin + local_col
                pair_begin = 2 * (local_row * ncol + local_col)
                self.write_pair(
                    row,
                    col,
                    tile_values[pair_begin],
                    tile_values[pair_begin + 1],
                )

    def close(self, commit: bool) -> Tuple[int, int, str, int, Path]:
        if self.closed:
            return self.iq, self.naux, "complex", self.path.stat().st_size, self.path
        os.close(self.fd)
        self.closed = True
        if commit:
            os.replace(self.tmp_path, self.path)
            return self.iq, self.naux, "complex", self.path.stat().st_size, self.path
        if self.tmp_path.exists():
            self.tmp_path.unlink()
        return self.iq, self.naux, "complex", 0, self.path


def streaming_output_for(
    outputs: Dict[int, StreamingCoulombOutput],
    output_dir: Path,
    output_prefix: str,
    iq: int,
    naux: int,
    endian: str,
    target_reader_version: int,
    path: Path,
    atom_layout: Optional[AtomLayout],
) -> StreamingCoulombOutput:
    output = outputs.get(iq)
    if output is None:
        output = StreamingCoulombOutput(
            output_dir, output_prefix, iq, naux, endian, target_reader_version, atom_layout
        )
        outputs[iq] = output
    elif output.naux != naux:
        raise ConversionError(
            f"{path}: q-point {iq} appears with inconsistent Naux values "
            f"({output.naux} and {naux})"
    )
    return output


def streaming_output_for_locked(
    outputs: Dict[int, StreamingCoulombOutput],
    output_dir: Path,
    output_prefix: str,
    iq: int,
    naux: int,
    endian: str,
    target_reader_version: int,
    path: Path,
    atom_layout: Optional[AtomLayout],
    lock: Optional[threading.Lock],
) -> StreamingCoulombOutput:
    if lock is None:
        return streaming_output_for(
            outputs, output_dir, output_prefix, iq, naux, endian,
            target_reader_version, path, atom_layout
        )
    with lock:
        return streaming_output_for(
            outputs, output_dir, output_prefix, iq, naux, endian,
            target_reader_version, path, atom_layout
        )


def stream_buffer_nrows(ncol: int, stream_buffer_mb: float) -> int:
    if stream_buffer_mb <= 0.0:
        return 1
    bytes_per_row = ncol * 16
    buffer_bytes = int(stream_buffer_mb * 1024 * 1024)
    return max(1, buffer_bytes // bytes_per_row)


def matrix_for(
    matrices: Dict[int, CoulombMatrix],
    iq: int,
    naux: int,
    check_complete: bool,
    path: Path,
) -> CoulombMatrix:
    matrix = matrices.get(iq)
    if matrix is None:
        matrix = CoulombMatrix(iq, naux, check_complete)
        matrices[iq] = matrix
    elif matrix.naux != naux:
        raise ConversionError(
            f"{path}: q-point {iq} appears with inconsistent Naux values "
            f"({matrix.naux} and {naux})"
        )
    return matrix


def record_q_weight(q_weights: Dict[int, float], iq: int, q_weight: float, path: Path) -> None:
    old_weight = q_weights.get(iq)
    if old_weight is None:
        q_weights[iq] = q_weight
        return

    tolerance = Q_WEIGHT_TOL * max(1.0, abs(old_weight), abs(q_weight))
    if abs(old_weight - q_weight) > tolerance:
        raise ConversionError(
            f"{path}: q-point {iq} appears with inconsistent weights "
            f"({old_weight:.16e} and {q_weight:.16e})"
        )


def record_q_weight_locked(
    q_weights: Dict[int, float],
    iq: int,
    q_weight: float,
    path: Path,
    lock: Optional[threading.Lock],
) -> None:
    if lock is None:
        record_q_weight(q_weights, iq, q_weight, path)
        return
    with lock:
        record_q_weight(q_weights, iq, q_weight, path)


def binary_layout_matches(path: Path, endian: str) -> bool:
    file_size = path.stat().st_size
    if file_size < 8:
        return False

    try:
        with path.open("rb") as handle:
            header = handle.read(8)
            if len(header) != 8:
                return False
            n_irk_points, n_irk_points_local = struct.unpack(endian + "ii", header)
            if n_irk_points <= 0 or n_irk_points_local < 0:
                return False
            if n_irk_points_local > n_irk_points:
                return False

            pos = 8
            for _ in range(n_irk_points_local):
                block_header = handle.read(32)
                if len(block_header) != 32:
                    return False
                naux, row_begin, row_end, col_begin, col_end, iq = struct.unpack(
                    endian + "iiiiii", block_header[:24]
                )
                if naux <= 0 or iq <= 0:
                    return False
                if row_begin <= 0 or row_end < row_begin or row_end > naux:
                    return False
                if col_begin <= 0 or col_end < col_begin or col_end > naux:
                    return False

                nrow = row_end - row_begin + 1
                ncol = col_end - col_begin + 1
                payload_size = nrow * ncol * 16
                pos += 32 + payload_size
                if pos > file_size:
                    return False
                handle.seek(payload_size, os.SEEK_CUR)
            return pos == file_size
    except OSError:
        return False


def detect_format(path: Path, endian: str) -> str:
    if binary_layout_matches(path, endian):
        return "binary"
    return "text"


def parse_binary_file(
    path: Path,
    matrices: Dict[int, CoulombMatrix],
    q_weights: Dict[int, float],
    check_complete: bool,
    endian: str,
    skip_lower: bool,
) -> Tuple[int, int]:
    with path.open("rb") as handle:
        n_irk_points, n_irk_points_local = struct.unpack(
            endian + "ii", read_exact(handle, 8, path, "binary Coulomb header")
        )

        for _ in range(n_irk_points_local):
            block_header = read_exact(handle, 32, path, "binary Coulomb block header")
            naux, row_begin, row_end, col_begin, col_end, iq = struct.unpack(
                endian + "iiiiii", block_header[:24]
            )
            (q_weight,) = struct.unpack(endian + "d", block_header[24:32])
            validate_block(path, naux, row_begin, row_end, col_begin, col_end, iq)
            record_q_weight(q_weights, iq, q_weight, path)

            nrow = row_end - row_begin + 1
            ncol = col_end - col_begin + 1
            if skip_lower and row_begin > col_end:
                handle.seek(nrow * ncol * 16, os.SEEK_CUR)
                continue

            payload = read_exact(
                handle,
                nrow * ncol * 16,
                path,
                f"q-point {iq} Coulomb block payload",
            )
            block_values = array.array("d")
            block_values.frombytes(payload)
            if need_byteswap(endian):
                block_values.byteswap()

            matrix = matrix_for(matrices, iq, naux, check_complete, path)
            for local_row in range(nrow):
                src_begin = 2 * local_row * ncol
                src_end = src_begin + 2 * ncol
                row = row_begin - 1 + local_row
                col_begin0 = col_begin - 1
                if skip_lower:
                    upper_begin = max(row, col_begin0)
                    if upper_begin > col_end - 1:
                        continue
                    lower_cols = upper_begin - col_begin0
                    src_begin += 2 * lower_cols
                    col_begin0 = upper_begin
                matrix.set_row_segment(
                    row,
                    col_begin0,
                    block_values[src_begin:src_end],
                )

    return n_irk_points, n_irk_points_local


def parse_text_file(
    path: Path,
    matrices: Dict[int, CoulombMatrix],
    q_weights: Dict[int, float],
    check_complete: bool,
    skip_lower: bool,
) -> Tuple[int, int]:
    stream = TokenStream(path)
    try:
        n_irk_points = stream.next_int("number of irreducible q-points")
        nblocks = 0
        while True:
            first = stream.next_optional()
            if first is None:
                break

            naux_token, line_no = first
            try:
                naux = int(naux_token)
            except ValueError as exc:
                raise ConversionError(
                    f"{path}:{line_no}: expected Naux integer, got {naux_token!r}"
                ) from exc

            row_begin = stream.next_int("row_start")
            row_end = stream.next_int("row_end")
            col_begin = stream.next_int("col_start")
            col_end = stream.next_int("col_end")
            iq = stream.next_int("q-point index")
            q_weight = stream.next_float("q-point weight")
            validate_block(path, naux, row_begin, row_end, col_begin, col_end, iq)
            record_q_weight(q_weights, iq, q_weight, path)

            ncol = col_end - col_begin + 1
            if skip_lower and row_begin > col_end:
                nrow = row_end - row_begin + 1
                for _ in range(nrow * ncol):
                    stream.next_token("ignored lower Coulomb matrix real part")
                    stream.next_token("ignored lower Coulomb matrix imaginary part")
                nblocks += 1
                continue

            matrix = matrix_for(matrices, iq, naux, check_complete, path)
            for row in range(row_begin - 1, row_end):
                row_values = array.array("d")
                row_col_begin = col_begin - 1
                for col in range(col_begin - 1, col_end):
                    if skip_lower and col < row:
                        stream.next_token("ignored lower Coulomb matrix real part")
                        stream.next_token("ignored lower Coulomb matrix imaginary part")
                        continue
                    if not row_values:
                        row_col_begin = col
                    row_values.append(stream.next_float("Coulomb matrix real part"))
                    row_values.append(stream.next_float("Coulomb matrix imaginary part"))
                if row_values:
                    matrix.set_row_segment(row, row_col_begin, row_values)
            nblocks += 1
    finally:
        stream.close()

    return n_irk_points, nblocks


def parse_binary_file_streaming(
    path: Path,
    outputs: Dict[int, StreamingCoulombOutput],
    output_dir: Path,
    output_prefix: str,
    q_weights: Dict[int, float],
    endian: str,
    target_reader_version: int,
    atom_layout: Optional[AtomLayout],
    stream_buffer_mb: float,
    skip_lower: bool,
    q_weight_lock: Optional[threading.Lock] = None,
    output_lock: Optional[threading.Lock] = None,
) -> Tuple[int, int]:
    with path.open("rb") as handle:
        n_irk_points, n_irk_points_local = struct.unpack(
            endian + "ii", read_exact(handle, 8, path, "binary Coulomb header")
        )

        for _ in range(n_irk_points_local):
            block_header = read_exact(handle, 32, path, "binary Coulomb block header")
            naux, row_begin, row_end, col_begin, col_end, iq = struct.unpack(
                endian + "iiiiii", block_header[:24]
            )
            (q_weight,) = struct.unpack(endian + "d", block_header[24:32])
            validate_block(path, naux, row_begin, row_end, col_begin, col_end, iq)
            record_q_weight_locked(q_weights, iq, q_weight, path, q_weight_lock)

            nrow = row_end - row_begin + 1
            ncol = col_end - col_begin + 1
            row_begin0 = row_begin - 1
            col_begin0 = col_begin - 1
            if skip_lower and row_begin > col_end:
                handle.seek(nrow * ncol * 16, os.SEEK_CUR)
                continue

            output = streaming_output_for_locked(
                outputs, output_dir, output_prefix, iq, naux, endian,
                target_reader_version, path, atom_layout, output_lock
            )
            if row_begin > col_end:
                rows_per_tile = stream_buffer_nrows(ncol, stream_buffer_mb)
                rows_left = nrow
                local_row = 0
                while rows_left > 0:
                    tile_nrow = min(rows_per_tile, rows_left)
                    payload = read_exact(
                        handle,
                        tile_nrow * ncol * 16,
                        path,
                        f"q-point {iq} lower-triangle tile payload",
                    )
                    tile_values = array.array("d")
                    tile_values.frombytes(payload)
                    if need_byteswap(endian):
                        tile_values.byteswap()
                    output.write_lower_transpose_tile(
                        row_begin0 + local_row,
                        col_begin0,
                        ncol,
                        tile_values,
                        tile_nrow,
                    )
                    local_row += tile_nrow
                    rows_left -= tile_nrow
            else:
                write_lower_part = row_begin > col_begin and not skip_lower
                for local_row in range(nrow):
                    payload = read_exact(
                        handle,
                        ncol * 16,
                        path,
                        f"q-point {iq} Coulomb row payload",
                    )
                    row_values = array.array("d")
                    row_values.frombytes(payload)
                    if need_byteswap(endian):
                        row_values.byteswap()
                    output.write_row_segment(
                        row_begin0 + local_row,
                        col_begin0,
                        row_values,
                        write_lower_part,
                    )

    return n_irk_points, n_irk_points_local


def parse_text_file_streaming(
    path: Path,
    outputs: Dict[int, StreamingCoulombOutput],
    output_dir: Path,
    output_prefix: str,
    q_weights: Dict[int, float],
    endian: str,
    target_reader_version: int,
    atom_layout: Optional[AtomLayout],
    stream_buffer_mb: float,
    skip_lower: bool,
    q_weight_lock: Optional[threading.Lock] = None,
    output_lock: Optional[threading.Lock] = None,
) -> Tuple[int, int]:
    stream = TokenStream(path)
    try:
        n_irk_points = stream.next_int("number of irreducible q-points")
        nblocks = 0
        while True:
            first = stream.next_optional()
            if first is None:
                break

            naux_token, line_no = first
            try:
                naux = int(naux_token)
            except ValueError as exc:
                raise ConversionError(
                    f"{path}:{line_no}: expected Naux integer, got {naux_token!r}"
                ) from exc

            row_begin = stream.next_int("row_start")
            row_end = stream.next_int("row_end")
            col_begin = stream.next_int("col_start")
            col_end = stream.next_int("col_end")
            iq = stream.next_int("q-point index")
            q_weight = stream.next_float("q-point weight")
            validate_block(path, naux, row_begin, row_end, col_begin, col_end, iq)
            record_q_weight_locked(q_weights, iq, q_weight, path, q_weight_lock)

            ncol = col_end - col_begin + 1
            row_begin0 = row_begin - 1
            col_begin0 = col_begin - 1
            if skip_lower and row_begin > col_end:
                nrow = row_end - row_begin + 1
                for _ in range(nrow * ncol):
                    stream.next_token("ignored lower Coulomb matrix real part")
                    stream.next_token("ignored lower Coulomb matrix imaginary part")
                nblocks += 1
                continue

            output = streaming_output_for_locked(
                outputs, output_dir, output_prefix, iq, naux, endian,
                target_reader_version, path, atom_layout, output_lock
            )
            if row_begin > col_end:
                nrow = row_end - row_begin + 1
                rows_per_tile = stream_buffer_nrows(ncol, stream_buffer_mb)
                rows_left = nrow
                local_row = 0
                while rows_left > 0:
                    tile_nrow = min(rows_per_tile, rows_left)
                    tile_values = array.array("d")
                    for _ in range(tile_nrow * ncol):
                        tile_values.append(stream.next_float("Coulomb matrix real part"))
                        tile_values.append(stream.next_float("Coulomb matrix imaginary part"))
                    output.write_lower_transpose_tile(
                        row_begin0 + local_row,
                        col_begin0,
                        ncol,
                        tile_values,
                        tile_nrow,
                    )
                    local_row += tile_nrow
                    rows_left -= tile_nrow
            else:
                write_lower_part = row_begin > col_begin and not skip_lower
                for row in range(row_begin0, row_end):
                    row_values = array.array("d")
                    row_col_begin = col_begin0
                    for col in range(col_begin0, col_begin0 + ncol):
                        if skip_lower and col < row:
                            stream.next_token("ignored lower Coulomb matrix real part")
                            stream.next_token("ignored lower Coulomb matrix imaginary part")
                            continue
                        if not row_values:
                            row_col_begin = col
                        row_values.append(stream.next_float("Coulomb matrix real part"))
                        row_values.append(stream.next_float("Coulomb matrix imaginary part"))
                    if row_values:
                        output.write_row_segment(
                            row, row_col_begin, row_values, write_lower_part
                        )
            nblocks += 1
    finally:
        stream.close()

    return n_irk_points, nblocks


def natural_key(path: Path) -> List[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def discover_input_files(input_dir: Path, input_prefix: str, output_prefix: str) -> List[Path]:
    files = []
    for path in input_dir.glob(f"{input_prefix}_*.txt"):
        if path.name.startswith(f"{output_prefix}_"):
            continue
        if not path.is_file():
            continue
        files.append(path)
    return sorted(files, key=natural_key)


def discover_existing_output_files(output_dir: Path, output_prefix: str) -> List[Path]:
    return sorted(
        (path for path in output_dir.glob(f"{output_prefix}_*.dat") if path.is_file()),
        key=natural_key,
    )


def check_no_existing_output_files(output_dir: Path, output_prefix: str) -> None:
    existing_outputs = discover_existing_output_files(output_dir, output_prefix)
    if not existing_outputs:
        return

    preview_count = min(5, len(existing_outputs))
    preview = "\n".join(f"  {path}" for path in existing_outputs[:preview_count])
    more = "" if len(existing_outputs) == preview_count else (
        f"\n  ... and {len(existing_outputs) - preview_count} more"
    )
    raise ConversionError(
        f"found existing output files matching {output_prefix}_*.dat in {output_dir}.\n"
        "Please clean them up manually before rerunning.\n"
        f"{preview}{more}"
    )


def check_complete_matrices(matrices: Dict[int, CoulombMatrix]) -> None:
    for iq, matrix in sorted(matrices.items()):
        missing = matrix.missing_count()
        if missing == 0:
            continue
        first_missing = matrix.first_missing()
        if first_missing is None:
            where = ""
        else:
            where = f"; first missing entry is row {first_missing[0]}, col {first_missing[1]}"
        raise ConversionError(
            f"q-point {iq} is incomplete: {missing} of "
            f"{matrix.naux * (matrix.naux + 1) // 2} upper-triangle matrix "
            f"entries were not provided{where}"
        )


def parse_legacy_stru_for_bz(stru_path: Path) -> Optional[Tuple[Tuple[int, int, int], List[Tuple[float, float, float]], List[int]]]:
    stream = TokenStream(stru_path)
    try:
        for _ in range(18):
            stream.next_float("lattice or reciprocal lattice vector")

        n_atoms = stream.next_int("number of atoms")
        for _ in range(n_atoms):
            stream.next_float("atom x coordinate")
            stream.next_float("atom y coordinate")
            stream.next_float("atom z coordinate")
            stream.next_int("atom type")

        first_nk = stream.next_optional()
        if first_nk is None:
            return None

        token, line_no = first_nk
        try:
            nk1 = int(token)
        except ValueError as exc:
            raise ConversionError(
                f"{stru_path}:{line_no}: expected k-grid dimension, got {token!r}"
            ) from exc
        nk2 = stream.next_int("second k-grid dimension")
        nk3 = stream.next_int("third k-grid dimension")
        if nk1 <= 0 or nk2 <= 0 or nk3 <= 0:
            raise ConversionError(f"{stru_path}: invalid k-grid dimensions {nk1} {nk2} {nk3}")

        nk_full = nk1 * nk2 * nk3
        kcart = []
        for _ in range(nk_full):
            kcart.append(
                (
                    stream.next_float("k-point Cartesian x"),
                    stream.next_float("k-point Cartesian y"),
                    stream.next_float("k-point Cartesian z"),
                )
            )

        map_representatives = []
        for _ in range(nk_full):
            representative = stream.next_int("irreducible k-point representative")
            if representative <= 0 or representative > nk_full:
                raise ConversionError(
                    f"{stru_path}: invalid irreducible k-point representative {representative}"
                )
            map_representatives.append(representative)
    finally:
        stream.close()

    return (nk1, nk2, nk3), kcart, map_representatives


def k_fraction(index: int, nk: Tuple[int, int, int]) -> Tuple[float, float, float]:
    nk1, nk2, nk3 = nk
    i1 = index // (nk2 * nk3)
    rest = index % (nk2 * nk3)
    i2 = rest // nk3
    i3 = rest % nk3
    return i1 / nk1, i2 / nk2, i3 / nk3


def representative_order(map_representatives: Sequence[int]) -> List[int]:
    seen = set()
    order = []
    for representative in map_representatives:
        if representative not in seen:
            seen.add(representative)
            order.append(representative)
    return order


def representative_weight(
    representative: int,
    ibz_index: int,
    q_weights: Dict[int, float],
    multiplicity: int,
    nk_full: int,
) -> float:
    if representative in q_weights:
        return q_weights[representative]
    if ibz_index in q_weights:
        return q_weights[ibz_index]
    return multiplicity / nk_full


def write_bz_sampling(
    output_path: Path,
    nk: Tuple[int, int, int],
    kcart: Sequence[Tuple[float, float, float]],
    map_representatives: Sequence[int],
    q_weights: Dict[int, float],
) -> None:
    nk_full = len(kcart)
    reps = representative_order(map_representatives)
    rep_to_ibz = {representative: index + 1 for index, representative in enumerate(reps)}
    rep_counts = {representative: 0 for representative in reps}
    for representative in map_representatives:
        rep_counts[representative] += 1

    rep_weights = {
        representative: representative_weight(
            representative,
            rep_to_ibz[representative],
            q_weights,
            rep_counts[representative],
            nk_full,
        )
        for representative in reps
    }

    tmp_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(f"{nk[0]:4d} {nk[1]:4d} {nk[2]:4d}\n")
            handle.write(f"{nk_full:7d} {len(reps):7d}\n")
            for index, (cart, representative) in enumerate(zip(kcart, map_representatives), 1):
                frac = k_fraction(index - 1, nk)
                ibz_index = rep_to_ibz[representative]
                full_weight = rep_weights[representative] / rep_counts[representative]
                handle.write(
                    f"{index:7d} {full_weight:20.12E} "
                    f"{frac[0]:20.12E} {frac[1]:20.12E} {frac[2]:20.12E} "
                    f"{cart[0]:20.12E} {cart[1]:20.12E} {cart[2]:20.12E} "
                    f"{ibz_index:7d} {representative:7d}\n"
                )

            for representative in reps:
                handle.write(
                    f"{rep_to_ibz[representative]:7d} {representative:7d} "
                    f"{rep_weights[representative]:20.12E}\n"
                )
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def maybe_generate_bz_sampling(
    input_dir: Path,
    output_dir: Path,
    bz_output_name: str,
    q_weights: Dict[int, float],
    quiet: bool,
) -> Optional[Path]:
    source_bz_path = input_dir / bz_output_name
    output_bz_path = output_dir / bz_output_name
    if source_bz_path.exists():
        if not quiet:
            print(f"Skipped BZ sampling generation: {source_bz_path} already exists.")
        return None
    if output_bz_path.exists():
        if not quiet:
            print(f"Skipped BZ sampling generation: {output_bz_path} already exists.")
        return None

    stru_path = input_dir / "stru_out"
    if not stru_path.exists():
        if not quiet:
            print(f"Skipped BZ sampling generation: {stru_path} is missing.")
        return None

    parsed = parse_legacy_stru_for_bz(stru_path)
    if parsed is None:
        if not quiet:
            print(f"Skipped BZ sampling generation: {stru_path} has no legacy k-grid tail.")
        return None

    nk, kcart, map_representatives = parsed
    write_bz_sampling(output_bz_path, nk, kcart, map_representatives, q_weights)
    if not quiet:
        print(f"Generated BZ sampling file: {output_bz_path}")
    return output_bz_path


def parse_atom_naux_argument(text: str) -> List[int]:
    parts = [part for part in re.split(r"[\s,]+", text.strip()) if part]
    if not parts:
        raise ConversionError("--atom-naux was provided but no sizes were found")
    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise ConversionError(f"invalid --atom-naux value {text!r}") from exc
    if any(value <= 0 for value in values):
        raise ConversionError(f"--atom-naux sizes must be positive: {values}")
    return values


def read_atom_types_from_stru(stru_path: Path) -> List[int]:
    stream = TokenStream(stru_path)
    try:
        for _ in range(18):
            stream.next_float("lattice or reciprocal lattice vector")

        n_atoms = stream.next_int("number of atoms")
        atom_types = []
        for _ in range(n_atoms):
            stream.next_float("atom x coordinate")
            stream.next_float("atom y coordinate")
            stream.next_float("atom z coordinate")
            atom_type = stream.next_int("atom type")
            if atom_type <= 0:
                raise ConversionError(f"{stru_path}: invalid atom type {atom_type}")
            atom_types.append(atom_type - 1)
    finally:
        stream.close()
    return atom_types


def read_atom_naux_from_basis_and_stru(basis_path: Path, stru_path: Path) -> List[int]:
    atom_types = read_atom_types_from_stru(stru_path)
    stream = TokenStream(basis_path)
    try:
        ntypes = stream.next_int("number of atom types")
        stream.next_int("total number of wavefunction basis functions")
        stream.next_int("total number of auxiliary basis functions")
        stream.next_token("basis kind")
        type_naux: Dict[int, int] = {}
        for _ in range(ntypes):
            atom_type = stream.next_int("atom type") - 1
            stream.next_int("number of wavefunction basis functions for atom type")
            naux = stream.next_int("number of auxiliary basis functions for atom type")
            if atom_type < 0 or naux <= 0:
                raise ConversionError(f"{basis_path}: invalid atom type basis entry")
            type_naux[atom_type] = naux
    finally:
        stream.close()

    try:
        return [type_naux[atom_type] for atom_type in atom_types]
    except KeyError as exc:
        raise ConversionError(
            f"{basis_path}: missing basis entry for atom type {exc.args[0] + 1}"
        ) from exc


def cs_binary_layout_matches(path: Path, endian: str) -> bool:
    file_size = path.stat().st_size
    if file_size < 12:
        return False
    try:
        with path.open("rb") as handle:
            header = handle.read(12)
            if len(header) != 12:
                return False
            natom, ncell, nblocks = struct.unpack(endian + "iii", header)
            if natom <= 0 or ncell < 0 or nblocks < 0:
                return False
            pos = 12
            for _ in range(nblocks):
                block_header = handle.read(32)
                if len(block_header) != 32:
                    return False
                dims = struct.unpack(endian + "iiiiiiii", block_header)
                if dims[0] <= 0 or dims[0] > natom or dims[1] <= 0 or dims[1] > natom:
                    return False
                if dims[5] <= 0 or dims[6] <= 0 or dims[7] <= 0:
                    return False
                payload_size = dims[5] * dims[6] * dims[7] * 8
                pos += 32 + payload_size
                if pos > file_size:
                    return False
                handle.seek(payload_size, os.SEEK_CUR)
            return pos == file_size
    except OSError:
        return False


def parse_cs_text_atom_naux(path: Path, atom_naux: Dict[int, int]) -> int:
    stream = TokenStream(path)
    try:
        natom = stream.next_int("number of atoms in Cs file")
        stream.next_int("number of cells in Cs file")
        while True:
            first = stream.next_optional()
            if first is None:
                break
            ia1_token, line_no = first
            try:
                ia1 = int(ia1_token) - 1
            except ValueError as exc:
                raise ConversionError(
                    f"{path}:{line_no}: expected atom index, got {ia1_token!r}"
                ) from exc
            stream.next_int("second atom index")
            stream.next_int("cell index 1")
            stream.next_int("cell index 2")
            stream.next_int("cell index 3")
            n_i = stream.next_int("number of wavefunction basis functions on atom I")
            n_j = stream.next_int("number of wavefunction basis functions on atom J")
            n_mu = stream.next_int("number of auxiliary basis functions on atom I")
            if ia1 < 0 or ia1 >= natom or n_i <= 0 or n_j <= 0 or n_mu <= 0:
                raise ConversionError(f"{path}: invalid Cs block header")
            atom_naux[ia1] = n_mu
            for _ in range(n_i * n_j * n_mu):
                stream.next_float("RI coefficient")
    finally:
        stream.close()
    return natom


def parse_cs_binary_atom_naux(path: Path, endian: str, atom_naux: Dict[int, int]) -> int:
    with path.open("rb") as handle:
        natom, _ncell, nblocks = struct.unpack(
            endian + "iii", read_exact(handle, 12, path, "binary Cs header")
        )
        for _ in range(nblocks):
            dims = struct.unpack(
                endian + "iiiiiiii",
                read_exact(handle, 32, path, "binary Cs block header"),
            )
            ia1 = dims[0] - 1
            if ia1 < 0 or ia1 >= natom or dims[5] <= 0 or dims[6] <= 0 or dims[7] <= 0:
                raise ConversionError(f"{path}: invalid binary Cs block header")
            atom_naux[ia1] = dims[7]
            handle.seek(dims[5] * dims[6] * dims[7] * 8, os.SEEK_CUR)
    return natom


def read_atom_naux_from_cs(input_dir: Path, cs_prefix: str, endian: str) -> List[int]:
    files = sorted(
        (path for path in input_dir.glob(f"{cs_prefix}*") if path.is_file()),
        key=natural_key,
    )
    if not files:
        raise ConversionError(
            "reader v1 output requires per-atom auxiliary sizes, but neither basis/structure "
            f"metadata nor files matching {cs_prefix}* were found"
        )

    atom_naux_map: Dict[int, int] = {}
    natom_seen: Optional[int] = None
    for path in files:
        if cs_binary_layout_matches(path, endian):
            natom = parse_cs_binary_atom_naux(path, endian, atom_naux_map)
        else:
            natom = parse_cs_text_atom_naux(path, atom_naux_map)
        if natom_seen is None:
            natom_seen = natom
        elif natom_seen != natom:
            raise ConversionError(
                f"{path}: inconsistent atom count in Cs files ({natom_seen} and {natom})"
            )

    assert natom_seen is not None
    missing = [iat + 1 for iat in range(natom_seen) if iat not in atom_naux_map]
    if missing:
        raise ConversionError(
            "could not infer auxiliary sizes for atom(s) "
            + ", ".join(str(iat) for iat in missing)
            + " from Cs files"
        )
    return [atom_naux_map[iat] for iat in range(natom_seen)]


def resolve_atom_layout(args: argparse.Namespace, input_dir: Path, endian: str) -> AtomLayout:
    if args.atom_naux:
        return AtomLayout(parse_atom_naux_argument(args.atom_naux))

    basis_path = input_dir / args.basis_name
    stru_path = input_dir / args.stru_name
    if basis_path.exists() and stru_path.exists():
        return AtomLayout(read_atom_naux_from_basis_and_stru(basis_path, stru_path))

    return AtomLayout(read_atom_naux_from_cs(input_dir, args.ri_prefix, endian))


def convert(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    check_no_existing_output_files(output_dir, args.output_prefix)

    endian = endian_prefix(args.endian)
    atom_layout = resolve_atom_layout(args, input_dir, endian)
    input_files = discover_input_files(input_dir, args.input_prefix, args.output_prefix)
    if not input_files:
        raise ConversionError(
            f"no input files matching {args.input_prefix}_*.txt found in {input_dir}"
        )

    if args.stream_complex:
        if args.check_complete:
            raise ConversionError("--stream-complex cannot be combined with --check-complete")
        q_weights, total_blocks, format_counts, outputs = convert_streaming_complex(
            args, input_files, input_dir, output_dir, endian, atom_layout
        )
        if not args.no_bz_sampling:
            maybe_generate_bz_sampling(
                input_dir, output_dir, args.bz_output_name, q_weights, args.quiet
            )
        print_conversion_summary(args, input_files, total_blocks, format_counts, outputs)
        return

    matrices: Dict[int, CoulombMatrix] = {}
    q_weights: Dict[int, float] = {}
    check_complete = args.check_complete
    total_blocks = 0
    format_counts = {"text": 0, "binary": 0}

    for path in input_files:
        input_format = detect_format(path, endian)
        format_counts[input_format] += 1
        if input_format == "binary":
            _nirk, nblocks = parse_binary_file(
                path, matrices, q_weights, check_complete, endian, args.skip_lower
            )
        else:
            _nirk, nblocks = parse_text_file(
                path, matrices, q_weights, check_complete, args.skip_lower
            )
        total_blocks += nblocks

    if check_complete:
        check_complete_matrices(matrices)

    outputs = []
    for iq, matrix in sorted(matrices.items()):
        output_path = output_dir / f"{args.output_prefix}_{iq}.dat"
        kind, nbytes = matrix.write(
            args.target_reader_version,
            output_path,
            endian,
            args.real_tol,
            args.force_complex,
            False,
            atom_layout,
        )
        outputs.append((iq, matrix.naux, kind, nbytes, output_path))

    if not args.no_bz_sampling:
        maybe_generate_bz_sampling(
            input_dir, output_dir, args.bz_output_name, q_weights, args.quiet
        )

    if args.quiet:
        return

    print_conversion_summary(args, input_files, total_blocks, format_counts, outputs)


def convert_streaming_complex(
    args: argparse.Namespace,
    input_files: Sequence[Path],
    input_dir: Path,
    output_dir: Path,
    endian: str,
    atom_layout: Optional[AtomLayout],
) -> Tuple[Dict[int, float], int, Dict[str, int], List[Tuple[int, int, str, int, Path]]]:
    del input_dir
    outputs_streaming: Dict[int, StreamingCoulombOutput] = {}
    q_weights: Dict[int, float] = {}
    total_blocks = 0
    format_counts = {"text": 0, "binary": 0}
    committed = False
    try:
        if args.workers <= 1 or len(input_files) == 1:
            for path in input_files:
                input_format, nblocks = convert_streaming_complex_file(
                    path,
                    outputs_streaming,
                    output_dir,
                    args.output_prefix,
                    q_weights,
                    endian,
                    args.target_reader_version,
                    atom_layout,
                    args.stream_buffer_mb,
                    args.skip_lower,
                    None,
                    None,
                )
                format_counts[input_format] += 1
                total_blocks += nblocks
        else:
            q_weight_lock = threading.Lock()
            output_lock = threading.Lock()
            first_error: Optional[BaseException] = None
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(
                        convert_streaming_complex_file,
                        path,
                        outputs_streaming,
                        output_dir,
                        args.output_prefix,
                        q_weights,
                        endian,
                        args.target_reader_version,
                        atom_layout,
                        args.stream_buffer_mb,
                        args.skip_lower,
                        q_weight_lock,
                        output_lock,
                    )
                    for path in input_files
                ]
                for future in as_completed(futures):
                    try:
                        input_format, nblocks = future.result()
                    except BaseException as exc:
                        if first_error is None:
                            first_error = exc
                    else:
                        format_counts[input_format] += 1
                        total_blocks += nblocks
            if first_error is not None:
                raise first_error

        outputs = [
            output.close(commit=True)
            for _iq, output in sorted(outputs_streaming.items())
        ]
        committed = True
        return q_weights, total_blocks, format_counts, outputs
    finally:
        if not committed:
            for output in outputs_streaming.values():
                output.close(commit=False)


def convert_streaming_complex_file(
    path: Path,
    outputs_streaming: Dict[int, StreamingCoulombOutput],
    output_dir: Path,
    output_prefix: str,
    q_weights: Dict[int, float],
    endian: str,
    target_reader_version: int,
    atom_layout: Optional[AtomLayout],
    stream_buffer_mb: float,
    skip_lower: bool,
    q_weight_lock: Optional[threading.Lock],
    output_lock: Optional[threading.Lock],
) -> Tuple[str, int]:
    input_format = detect_format(path, endian)
    if input_format == "binary":
        _nirk, nblocks = parse_binary_file_streaming(
            path,
            outputs_streaming,
            output_dir,
            output_prefix,
            q_weights,
            endian,
            target_reader_version,
            atom_layout,
            stream_buffer_mb,
            skip_lower,
            q_weight_lock,
            output_lock,
        )
    else:
        _nirk, nblocks = parse_text_file_streaming(
            path,
            outputs_streaming,
            output_dir,
            output_prefix,
            q_weights,
            endian,
            target_reader_version,
            atom_layout,
            stream_buffer_mb,
            skip_lower,
            q_weight_lock,
            output_lock,
        )
    return input_format, nblocks


def print_conversion_summary(
    args: argparse.Namespace,
    input_files: Sequence[Path],
    total_blocks: int,
    format_counts: Dict[str, int],
    outputs: Sequence[Tuple[int, int, str, int, Path]],
) -> None:
    if args.quiet:
        return
    formats = ", ".join(
        f"{count} {name}" for name, count in sorted(format_counts.items()) if count
    )
    print(
        f"Converted {len(input_files)} input file(s) ({formats}), {total_blocks} block(s), "
        f"to {len(outputs)} per-q binary file(s)."
    )
    for iq, naux, kind, nbytes, output_path in outputs:
        print(f"  iq={iq:>4}  Naux={naux:>6}  {kind:<7}  {nbytes:>12} bytes  {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert LibRPA coulomb_mat_*.txt files to binary "
            "<output_prefix>_<iq>.dat files."
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=Path("."),
        help="directory containing <input_prefix>_*.txt files (default: current directory)",
    )
    parser.add_argument(
        "-O",
        "--output-dir",
        type=Path,
        help="directory for generated binary files (default: input directory)",
    )
    parser.add_argument(
        "-i", "--input-prefix",
        default="coulomb_mat",
        help="input filename prefix before _*.txt (default: coulomb_mat)",
    )
    parser.add_argument(
        "-o", "--output-prefix",
        default="coulomb_full_iq",
        help="output filename prefix before _<iq>.dat (default: coulomb_full_iq)",
    )
    parser.add_argument(
        "--endian",
        choices=("native", "little", "big"),
        default="native",
        help="endianness for legacy binary input and new binary output (default: native)",
    )
    parser.add_argument(
        "-T", "--target-reader-version",
        choices=(1,),
        default=1, type=int,
        help="Targeting version of Coulomb reader of driver (default: 1)",
    )
    parser.add_argument(
        "--atom-naux",
        help=(
            "comma- or whitespace-separated per-atom auxiliary sizes for reader v1 output; "
            "used when basis/structure metadata is unavailable"
        ),
    )
    parser.add_argument(
        "--basis-name",
        default="basis_out",
        help="basis metadata filename used for reader v1 output (default: basis_out)",
    )
    parser.add_argument(
        "--stru-name",
        default="stru_out",
        help="structure metadata filename used for reader v1 output (default: stru_out)",
    )
    parser.add_argument(
        "--ri-prefix",
        default="Cs_data",
        help="RI coefficient filename prefix used to infer reader v1 atom basis if needed (default: Cs_data)",
    )
    parser.add_argument(
        "--real-tol",
        type=float,
        default=1e-13,
        help=(
            "write a real-valued output if all imaginary parts are <= this absolute "
            "tolerance (default: 1e-13)"
        ),
    )
    parser.add_argument(
        "--force-complex",
        action="store_true",
        help="always write complex-valued output, even when the imaginary part is zero",
    )
    parser.add_argument(
        "--stream-complex",
        action="store_true",
        help=(
            "stream directly to complex output files using positional writes; "
            "recommended for very large Naux and incompatible with --check-complete"
        ),
    )
    parser.add_argument(
        "--stream-buffer-mb",
        type=float,
        default=512.0,
        help=(
            "buffer size for transposing fully lower-triangular blocks in "
            "--stream-complex mode (default: 512)"
        ),
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        help=(
            "number of legacy input files to process concurrently in "
            "--stream-complex mode (default: 1)"
        ),
    )
    parser.add_argument(
        "--skip-lower",
        action="store_true",
        help=(
            "ignore strictly lower-triangular input entries; use only when the "
            "legacy matrices are known to contain explicit upper-triangle data"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "accepted for compatibility; existing <output_prefix>_*.dat files "
            "still abort and must be cleaned manually"
        ),
    )
    parser.add_argument(
        "--check-complete",
        action="store_true",
        help="verify that every upper-triangle matrix entry was supplied",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the conversion summary",
    )
    parser.add_argument(
        "--no-bz-sampling",
        action="store_true",
        help="do not try to generate bz_sampling_out from legacy stru_out",
    )
    parser.add_argument(
        "--bz-output-name",
        default="bz_sampling_out",
        help="BZ sampling output filename (default: bz_sampling_out)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.real_tol < 0.0:
        parser.error("--real-tol must be non-negative")
    if args.stream_buffer_mb < 0.0:
        parser.error("--stream-buffer-mb must be non-negative")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    try:
        convert(args)
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
