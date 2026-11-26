from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from ase.calculators.emt import EMT
from numpy.testing import assert_allclose

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from ase import Atoms
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


@pytest.fixture
def registry() -> Generator[None, None, None]:
    """Restore the batch adapter registry after a test that registers adapters."""
    from matcalc.backend import _batch

    saved, loaded = list(_batch._REGISTRY), _batch._entry_points_loaded  # noqa: SLF001
    yield
    _batch._REGISTRY[:] = saved  # noqa: SLF001
    _batch._entry_points_loaded = loaded  # noqa: SLF001


@pytest.fixture
def ase_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the ASE backend, where batch adapters apply."""
    from matcalc import backend

    monkeypatch.setattr(backend, "SIMULATION_BACKEND", "ASE")


class _RecordingAdapter:
    """Test adapter: energy = number of atoms, forces = 1, and records the batch sizes it was called with."""

    implemented_properties = ("energy", "forces")

    def __init__(self) -> None:
        self.batches: list[int] = []

    def calculate(self, atoms: Sequence[Atoms], properties: Sequence[str]) -> list[dict[str, Any]]:
        self.batches.append(len(atoms))
        results = [{"energy": float(len(a)), "forces": np.ones((len(a), 3))} for a in atoms]
        return [{k: v for k, v in r.items() if k in properties} for r in results]


def _fake_serial(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace run_pes_calc with a fake that returns zeros, to detect the serial path."""
    from matcalc import backend

    def fake(structure: Structure, _calculator: Calculator) -> MagicMock:
        n = len(structure)
        return MagicMock(potential_energy=0.0, forces=np.zeros((n, 3)), stress=np.zeros((3, 3)))

    mock = MagicMock(side_effect=fake)
    monkeypatch.setattr(backend, "run_pes_calc", mock)
    return mock


@pytest.mark.usefixtures("registry", "ase_backend")
def test_registered_adapter_batches(Si: Structure) -> None:
    """A registered adapter receives the structures in batches of batch_size and its results are returned in order."""
    from matcalc.backend import BatchAdapter, calc_batch, register_batch_adapter

    adapter = _RecordingAdapter()
    assert isinstance(adapter, BatchAdapter)
    register_batch_adapter(EMT, lambda _calc: adapter)
    structures = [Si * n for n in (1, 2, 1, 2, 1)]
    results = calc_batch(structures, EMT(), properties=("energy", "forces"), batch_size=2)
    assert adapter.batches == [2, 2, 1]
    assert [r["energy"] for r in results] == [float(len(s)) for s in structures]
    assert all(r["forces"].shape == (len(s), 3) for r, s in zip(results, structures, strict=True))


@pytest.mark.usefixtures("registry", "ase_backend")
def test_register_by_path_and_precedence(monkeypatch: pytest.MonkeyPatch, Si: Structure) -> None:
    """Paths resolve only for imported modules, later registrations win, and a factory can decline."""
    from matcalc.backend import calc_batch, get_batch_adapter, register_batch_adapter

    first, second = _RecordingAdapter(), _RecordingAdapter()
    register_batch_adapter("not_an_installed_package.calculators:Calc", lambda _calc: pytest.fail("never resolved"))
    register_batch_adapter("ase.calculators.emt:EMT", lambda _calc: first)
    register_batch_adapter(EMT, lambda _calc: second)
    assert get_batch_adapter(EMT()) is second

    register_batch_adapter(EMT, lambda _calc: None)  # declines: falls through to the next registration
    assert get_batch_adapter(EMT()) is second

    serial = _fake_serial(monkeypatch)
    calc_batch([Si, Si], EMT(), properties=("forces",), batch_size=1)  # batch_size 1 never uses an adapter
    assert serial.call_count == 2
    assert second.batches == []


@pytest.mark.usefixtures("registry", "ase_backend")
def test_adapter_missing_property_uses_serial(monkeypatch: pytest.MonkeyPatch, Si: Structure) -> None:
    """If the adapter does not implement a requested property, every structure goes through run_pes_calc."""
    from matcalc.backend import calc_batch, register_batch_adapter

    adapter = _RecordingAdapter()
    register_batch_adapter(EMT, lambda _calc: adapter)
    serial = _fake_serial(monkeypatch)
    results = calc_batch([Si, Si * 2], EMT(), properties=("energy", "stress"), batch_size=4)
    assert serial.call_count == 2
    assert adapter.batches == []
    assert [sorted(r) for r in results] == [["energy", "stress"], ["energy", "stress"]]


@pytest.mark.usefixtures("registry", "ase_backend")
def test_adapter_wrong_result_count(Si: Structure) -> None:
    from matcalc.backend import calc_batch, register_batch_adapter

    class Broken(_RecordingAdapter):
        def calculate(self, atoms: Sequence[Atoms], properties: Sequence[str]) -> list[dict[str, Any]]:
            return super().calculate(atoms, properties)[:-1]

    register_batch_adapter(EMT, lambda _calc: Broken())
    with pytest.raises(RuntimeError, match="returned 1 results for 2 structures"):
        calc_batch([Si, Si], EMT(), properties=("forces",), batch_size=2)


@pytest.mark.usefixtures("ase_backend")
def test_native_calculate_batch(Si: Structure) -> None:
    """A calculator with its own calculate_batch method is used without registration."""
    from matcalc.backend import calc_batch, get_batch_adapter

    class BatchEMT(EMT):
        batch_properties = ("forces",)

        def __init__(self) -> None:
            super().__init__()
            self.batches: list[int] = []

        def calculate_batch(self, atoms: Sequence[Atoms], properties: Sequence[str]) -> list[dict[str, Any]]:
            self.batches.append(len(atoms))
            return [{p: np.full((len(a), 3), 2.0) for p in properties} for a in atoms]

    calc = BatchEMT()
    adapter = get_batch_adapter(calc)
    assert adapter is not None
    assert adapter.implemented_properties == ("forces",)
    results = calc_batch([Si, Si, Si], calc, properties=("forces",), batch_size=2)
    assert calc.batches == [2, 1]
    assert all((r["forces"] == 2.0).all() for r in results)


@pytest.mark.usefixtures("registry")
def test_entry_point_plugins(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Entry points are loaded once, on the first lookup; a failing plugin is logged and skipped."""
    from matcalc.backend import _batch, get_batch_adapter, register_batch_adapter

    adapter = _RecordingAdapter()

    def register() -> None:
        register_batch_adapter(EMT, lambda _calc: adapter)

    def broken() -> None:
        raise ImportError("missing dependency")

    plugins = [MagicMock(load=MagicMock(return_value=register)), MagicMock(load=MagicMock(return_value=broken))]
    plugins[1].name = "broken"
    fake_entry_points = MagicMock(return_value=plugins)
    monkeypatch.setattr(_batch, "entry_points", fake_entry_points)
    monkeypatch.setattr(_batch, "_entry_points_loaded", False)

    with caplog.at_level(logging.WARNING, logger="matcalc.backend._batch"):
        assert get_batch_adapter(EMT()) is adapter
    assert "Failed to load matcalc batch adapter plugin 'broken'" in caplog.text
    get_batch_adapter(EMT())
    fake_entry_points.assert_called_once_with(group="matcalc.batch_adapters")


@pytest.mark.ase_only(reason="batch adapters apply to the ASE backend only")
@pytest.mark.parametrize("batch_size", [2, 3, 16])
def test_calc_batch_matgl_matches_serial(
    Si: Structure, Li2O: Structure, matpes_calculator: PESCalculator, batch_size: int
) -> None:
    """Batched MatGL energies, forces and stresses match the one-at-a-time path, for mixed sizes and elements."""
    _assert_batched_matches_serial(matpes_calculator, _mixed_structures(Si, Li2O), batch_size)


@pytest.mark.ase_only(reason="batch adapters apply to the ASE backend only")
@pytest.mark.parametrize(
    ("package", "model"),
    [("mace", "MACE-MPtrj-0a-small"), ("tensorpotential", "GRACE-OMat24-ft-MPA-1L-small")],
)
def test_calc_batch_providers_match_serial(Si: Structure, Li2O: Structure, package: str, model: str) -> None:
    """MACE and GRACE batched results match the one-at-a-time path (skipped unless the provider is installed)."""
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
    from matcalc.backend import calc_batch, calc_forces

    serial = calc_batch(structures, calculator)
    batched = calc_batch(structures, calculator, batch_size=batch_size)
    for s, r_serial, r_batched in zip(structures, serial, batched, strict=True):
        assert r_batched["forces"].shape == (len(s), 3)
        assert r_batched["stress"].shape == (3, 3)
        assert r_batched["energy"] == pytest.approx(r_serial["energy"], rel=1e-6, abs=1e-4)
        assert_allclose(r_batched["forces"], r_serial["forces"], atol=1e-5)
        assert_allclose(r_batched["stress"], r_serial["stress"], atol=1e-5)
    # Forces alone skip the stress autograd path, which moves float32 forces by ~1e-6 eV/A (seen with MACE).
    forces = calc_forces(structures, calculator, batch_size=batch_size)
    for f, r in zip(forces, batched, strict=True):
        assert_allclose(f, r["forces"], atol=1e-5)


def test_calc_forces_serial_fallback(monkeypatch: pytest.MonkeyPatch, Si: Structure) -> None:
    """Calculators without a batch adapter go through run_pes_calc one structure at a time."""
    from matcalc import backend

    monkeypatch.setattr(backend, "SIMULATION_BACKEND", "ASE")
    serial = _fake_serial(monkeypatch)
    forces = backend.calc_forces([Si, Si * 2], EMT(), batch_size=8)
    assert [f.shape for f in forces] == [(2, 3), (16, 3)]
    assert serial.call_count == 2


def test_calc_batch_invalid_arguments(Si: Structure) -> None:
    from matcalc.backend import calc_batch, calc_forces

    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        calc_forces([Si], EMT(), batch_size=0)
    with pytest.raises(ValueError, match="Unknown properties"):
        calc_batch([Si], EMT(), properties=("energy", "magmoms"))


def test_supports_batched_forces(matpes_calculator: PESCalculator) -> None:
    from matcalc.backend import supports_batched_forces

    assert supports_batched_forces(matpes_calculator)
    assert not supports_batched_forces(EMT())
