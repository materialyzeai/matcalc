from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest
from ase.calculators.emt import EMT
from numpy.testing import assert_allclose

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


def test_run_pes_calc_lammps_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """When SIMULATION_BACKEND is not ASE, run_pes_calc forwards to run_lammps."""
    from matcalc import backend

    sentinel = object()
    fake_run_lammps = MagicMock(return_value=sentinel)

    monkeypatch.setattr(backend, "SIMULATION_BACKEND", "LAMMPS")
    monkeypatch.setattr(backend, "run_lammps", fake_run_lammps)

    result = backend.run_pes_calc("arg", key="val")
    assert result is sentinel
    fake_run_lammps.assert_called_once_with("arg", key="val")


def test_run_pes_calc_ase_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default ASE backend path forwards to run_ase."""
    from matcalc import backend

    sentinel = object()
    fake_run_ase = MagicMock(return_value=sentinel)

    monkeypatch.setattr(backend, "SIMULATION_BACKEND", "ASE")
    monkeypatch.setattr(backend, "run_ase", fake_run_ase)

    result = backend.run_pes_calc("arg", key="val")
    assert result is sentinel
    fake_run_ase.assert_called_once_with("arg", key="val")


@pytest.mark.ase_only(reason="batched forces apply to the ASE backend only")
@pytest.mark.parametrize("batch_size", [2, 3, 16])
def test_calc_forces_batched_matches_serial(
    Si: Structure, Li2O: Structure, matpes_calculator: PESCalculator, batch_size: int
) -> None:
    """Batched MatGL forces match the one-at-a-time path, for structures of different sizes and elements."""
    _assert_batched_matches_serial(matpes_calculator, _mixed_structures(Si, Li2O), batch_size)


@pytest.mark.ase_only(reason="batched forces apply to the ASE backend only")
@pytest.mark.parametrize(
    ("package", "model"),
    [("mace", "MACE-MPtrj-0a-small"), ("tensorpotential", "GRACE-OMat24-ft-MPA-1L-small")],
)
def test_calc_forces_batched_providers(Si: Structure, Li2O: Structure, package: str, model: str) -> None:
    """MACE and GRACE batched forces match the one-at-a-time path (skipped unless the provider is installed)."""
    pytest.importorskip(package)
    from matcalc import load_fp
    from matcalc.backend import supports_batched_forces

    calculator = load_fp(model)
    assert supports_batched_forces(calculator)
    structures = _mixed_structures(Si, Li2O)
    for batch_size in (2, 16):
        _assert_batched_matches_serial(calculator, structures, batch_size)


def _mixed_structures(Si: Structure, Li2O: Structure) -> list[Structure]:
    """Perturbed supercells of different sizes and elements, so that batches mix structures."""
    pairs = ((Si, 2), (Li2O, 1), (Si, 1), (Li2O, 2), (Si, 2))
    return [(base * scale).perturb(0.05, seed=i) for i, (base, scale) in enumerate(pairs)]


def _assert_batched_matches_serial(calculator: Calculator, structures: list[Structure], batch_size: int) -> None:
    from matcalc.backend import calc_forces

    serial = calc_forces(structures, calculator)
    batched = calc_forces(structures, calculator, batch_size=batch_size)
    assert [f.shape for f in batched] == [(len(s), 3) for s in structures]
    for f_serial, f_batched in zip(serial, batched, strict=True):
        assert_allclose(f_batched, f_serial, atol=1e-5)


def test_calc_forces_serial_fallback(monkeypatch: pytest.MonkeyPatch, Si: Structure) -> None:
    """Calculators without batch support go through run_pes_calc one structure at a time."""
    from matcalc import backend

    fake_run = MagicMock(side_effect=lambda s, _: MagicMock(forces=np.zeros((len(s), 3))))
    monkeypatch.setattr(backend, "SIMULATION_BACKEND", "ASE")
    monkeypatch.setattr(backend, "run_pes_calc", fake_run)

    forces = backend.calc_forces([Si, Si * 2], EMT(), batch_size=8)
    assert [f.shape for f in forces] == [(2, 3), (16, 3)]
    assert fake_run.call_count == 2


def test_calc_forces_invalid_batch_size(Si: Structure) -> None:
    from matcalc.backend import calc_forces

    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        calc_forces([Si], EMT(), batch_size=0)


def test_supports_batched_forces(matpes_calculator: PESCalculator) -> None:
    from matcalc.backend import supports_batched_forces

    assert supports_batched_forces(matpes_calculator)
    assert not supports_batched_forces(EMT())
