"""Read LibRPA binary GW spectral-function files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
import tempfile

import numpy as np


_INT = np.dtype("=i4")
_FLOAT = np.dtype("=f8")
_HEADER = struct.Struct("=iiiidi")


@dataclass(frozen=True)
class SpectralFunctionBinary:
    path: Path
    n_spin: int
    n_kpoints: int
    sf_state_low: int
    sf_state_high: int
    efermi: float
    n_real_omegas: int
    offsets: dict[str, int]

    @property
    def n_states(self) -> int:
        return self.sf_state_high - self.sf_state_low

    @classmethod
    def from_file(cls, path: str | Path) -> "SpectralFunctionBinary":
        path = Path(path)
        with path.open("rb") as handle:
            header = handle.read(_HEADER.size)
        if len(header) != _HEADER.size:
            raise ValueError(f"{path} is too short to contain a spectral header")

        n_spin, n_kpoints, sf_state_low, sf_state_high, efermi, n_real_omegas = _HEADER.unpack(header)
        n_states = sf_state_high - sf_state_low
        if min(n_spin, n_kpoints, n_states, n_real_omegas) <= 0:
            raise ValueError(f"{path} has invalid spectral dimensions")

        n_state_values = n_spin * n_kpoints * n_states
        n_omega_values = n_state_values * n_real_omegas
        offset = _HEADER.size
        offsets: dict[str, int] = {}
        for name, n_values in (
            ("eks", n_state_values),
            ("vxc", n_state_values),
            ("vexx", n_state_values),
            ("real_omegas", n_real_omegas),
            ("specfunc", n_omega_values),
            ("resigc", n_omega_values),
            ("imsigc", n_omega_values),
        ):
            offsets[name] = offset
            offset += n_values * _FLOAT.itemsize

        actual_size = path.stat().st_size
        if actual_size != offset:
            raise ValueError(f"{path} has {actual_size} bytes, expected {offset}")

        return cls(path, n_spin, n_kpoints, sf_state_low, sf_state_high, efermi, n_real_omegas, offsets)

    def _map(self, name: str, shape: tuple[int, ...]) -> np.memmap:
        return np.memmap(self.path, dtype=_FLOAT, mode="r", offset=self.offsets[name], shape=shape, order="C")

    @property
    def eks(self) -> np.memmap:
        return self._map("eks", (self.n_spin, self.n_kpoints, self.n_states))

    @property
    def vxc(self) -> np.memmap:
        return self._map("vxc", (self.n_spin, self.n_kpoints, self.n_states))

    @property
    def vexx(self) -> np.memmap:
        return self._map("vexx", (self.n_spin, self.n_kpoints, self.n_states))

    @property
    def real_omegas(self) -> np.memmap:
        return self._map("real_omegas", (self.n_real_omegas,))

    @property
    def specfunc(self) -> np.memmap:
        return self._map("specfunc", (self.n_spin, self.n_kpoints, self.n_states, self.n_real_omegas))

    @property
    def resigc(self) -> np.memmap:
        return self._map("resigc", (self.n_spin, self.n_kpoints, self.n_states, self.n_real_omegas))

    @property
    def imsigc(self) -> np.memmap:
        return self._map("imsigc", (self.n_spin, self.n_kpoints, self.n_states, self.n_real_omegas))

    def state(self, ispin: int, ik: int, istate: int, *, one_based: bool = False) -> dict[str, object]:
        """Return data for one absolute state index.

        Indices are zero-based by default. Set one_based=True for human-facing
        spin/k/state labels; the returned ispin/ik/istate remain zero-based.
        """
        if one_based:
            ispin -= 1
            ik -= 1
            istate -= 1

        if not 0 <= ispin < self.n_spin:
            raise IndexError(f"ispin={ispin} outside [0, {self.n_spin})")
        if not 0 <= ik < self.n_kpoints:
            raise IndexError(f"ik={ik} outside [0, {self.n_kpoints})")
        if not self.sf_state_low <= istate < self.sf_state_high:
            raise IndexError(f"istate={istate} outside [{self.sf_state_low}, {self.sf_state_high})")

        state_rel = istate - self.sf_state_low
        return {
            "ispin": ispin,
            "ik": ik,
            "istate": istate,
            "efermi": self.efermi,
            "real_omegas": np.asarray(self.real_omegas),
            "eks": float(self.eks[ispin, ik, state_rel]),
            "vxc": float(self.vxc[ispin, ik, state_rel]),
            "vexx": float(self.vexx[ispin, ik, state_rel]),
            "specfunc": np.asarray(self.specfunc[ispin, ik, state_rel, :]),
            "resigc": np.asarray(self.resigc[ispin, ik, state_rel, :]),
            "imsigc": np.asarray(self.imsigc[ispin, ik, state_rel, :]),
        }


def read_spectral_function(
    path: str | Path,
    ispin: int | None = None,
    ik: int | None = None,
    istate: int | None = None,
    *,
    one_based: bool = False,
) -> SpectralFunctionBinary | dict[str, object]:
    """Read a LibRPA spectral binary.

    Without indices, returns a SpectralFunctionBinary object whose array
    attributes are read-only memmaps. With ispin, ik, and istate, returns only
    that state's metadata and omega-dependent arrays.
    """
    data = SpectralFunctionBinary.from_file(path)
    if ispin is None and ik is None and istate is None:
        return data
    if ispin is None or ik is None or istate is None:
        raise ValueError("selective reading requires ispin, ik, and istate")
    return data.state(ispin, ik, istate, one_based=one_based)


def _self_check() -> None:
    n_spin, n_kpoints, sf_low, sf_high, efermi, n_omegas = 1, 2, 3, 5, 1.25, 4
    n_states = sf_high - sf_low
    n_state_values = n_spin * n_kpoints * n_states
    n_omega_values = n_state_values * n_omegas

    eks = np.arange(n_state_values, dtype=_FLOAT) + 10.0
    vxc = np.arange(n_state_values, dtype=_FLOAT) + 20.0
    vexx = np.arange(n_state_values, dtype=_FLOAT) + 30.0
    omegas = np.linspace(-1.0, 1.0, n_omegas, dtype=_FLOAT)
    spec = np.arange(n_omega_values, dtype=_FLOAT).reshape(n_spin, n_kpoints, n_states, n_omegas) + 100.0
    res = spec + 100.0
    ims = spec + 200.0

    with tempfile.NamedTemporaryFile() as handle:
        handle.write(_HEADER.pack(n_spin, n_kpoints, sf_low, sf_high, efermi, n_omegas))
        for array in (eks, vxc, vexx, omegas, spec.ravel(order="C"), res.ravel(order="C"), ims.ravel(order="C")):
            array.tofile(handle)
        handle.flush()

        selected = read_spectral_function(handle.name, ispin=0, ik=1, istate=4)
        assert selected["eks"] == 13.0
        assert np.allclose(selected["real_omegas"], omegas)
        assert np.allclose(selected["specfunc"], spec[0, 1, 1, :])

        full = read_spectral_function(handle.name)
        assert isinstance(full, SpectralFunctionBinary)
        assert full.specfunc.shape == (1, 2, 2, 4)


if __name__ == "__main__":
    _self_check()
    print("self-check passed")
