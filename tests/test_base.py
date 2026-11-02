"""Tests for shared PropCalc helpers."""

from __future__ import annotations

import inspect
import os
from typing import TYPE_CHECKING, Any

import pytest
from joblib import cpu_count, parallel_config
from joblib.parallel import LokyBackend

from matcalc import (
    AdsorptionCalc,
    DefectCalc,
    ElasticityCalc,
    EnergeticsCalc,
    GBCalc,
    InterfaceCalc,
    LAMMPSMDCalc,
    MDCalc,
    PhononCalc,
    SurfaceCalc,
)
from matcalc._base import PropCalc, _worker_thread_cap
from matcalc._relaxation import RelaxCalc

if TYPE_CHECKING:
    from collections.abc import Callable

    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure

# Keys that the calculators also pass explicitly; repeating them used to raise a duplicate-keyword TypeError.
OVERRIDES = {"fmax": 0.123, "optimizer": "FIRE", "relax_cell": False}


class _DummyCalc(PropCalc):
    def __init__(self, calculator: PESCalculator, relax_calc_kwargs: dict | None = None) -> None:
        self.calculator = calculator
        self.relax_calc_kwargs = relax_calc_kwargs

    def calc(self, structure: Any) -> dict[str, Any]:
        return super().calc(structure)


def test_relaxer_precedence(matpes_calculator: PESCalculator) -> None:
    relaxer = _DummyCalc(matpes_calculator, OVERRIDES)._relaxer(fmax=0.1, relax_cell=True, max_steps=7)  # noqa: SLF001
    assert relaxer.calculator is matpes_calculator
    assert relaxer.fmax == pytest.approx(0.123)
    assert relaxer.optimizer == "FIRE"
    assert not relaxer.relax_cell
    assert relaxer.max_steps == 7

    relaxer = _DummyCalc(matpes_calculator)._relaxer(fmax=0.1, relax_cell=True)  # noqa: SLF001
    assert relaxer.fmax == pytest.approx(0.1)
    assert relaxer.relax_cell


@pytest.mark.parametrize(
    ("prev", "expected"),
    [
        (None, {"x": "eV"}),
        ({"_units": {"a": "A", "x": "meV"}, "a": 1.0}, {"a": "A", "x": "eV"}),
        ({"a": "A"}, {"a": "A", "x": "eV"}),
        # A result dict with no upstream ``_units``: its entries are values, not units.
        ({"final_structure": object(), "structure": object(), "foo": 1.0}, {"x": "eV"}),
        ({"_units": None, "a": 1.0}, {"x": "eV"}),
    ],
)
def test_merge_units(prev: dict | None, expected: dict) -> None:
    assert PropCalc._merge_units(prev, {"x": "eV"}) == expected  # noqa: SLF001


def test_units_exclude_result_values(Li2O: Structure, matpes_calculator: PESCalculator) -> None:
    result = RelaxCalc(matpes_calculator, relax_atoms=False, relax_cell=False).calc({"structure": Li2O, "foo": 1.0})
    assert all(isinstance(unit, str) for unit in result["_units"].values())
    assert {"final_structure", "structure", "foo"}.isdisjoint(result["_units"])


class _Captured(Exception):  # noqa: N818
    """Raised by the patched RelaxCalc.__init__ to stop the calculation once its arguments are known."""


CASES: dict[str, Callable[[Any, Structure], Any]] = {
    "surface": lambda c, s: SurfaceCalc(c, relax_calc_kwargs=OVERRIDES).calc_slabs(s),
    "gb": lambda c, s: GBCalc(c, relax_calc_kwargs=OVERRIDES).calc({"bulk": s, "grain_boundary": s}),
    "interface": lambda c, s: InterfaceCalc(c, relax_calc_kwargs=OVERRIDES).calc(
        {"interface": s, "film_bulk": s, "substrate_bulk": s, "film_slab": s, "substrate_slab": s}
    ),
    "adsorption_bulk": lambda c, s: AdsorptionCalc(c, relax_calc_kwargs=OVERRIDES).calc_bulk(s),
    "adsorption_slab": lambda c, s: AdsorptionCalc(c, relax_calc_kwargs=OVERRIDES).calc_slab(s),
    "defect": lambda c, s: DefectCalc(c, relax_calc_kwargs=OVERRIDES).calc_defects(s),
    "energetics": lambda c, s: EnergeticsCalc(c, relax_structure=False, relax_calc_kwargs=OVERRIDES).calc(s),
    "elasticity": lambda c, s: ElasticityCalc(
        c, relax_structure=False, relax_deformed_structures=True, relax_calc_kwargs=OVERRIDES
    ).calc(s),
    "phonon": lambda c, s: PhononCalc(c, relax_calc_kwargs=OVERRIDES).calc(s),
    "md": lambda c, s: MDCalc(c, relax_calc_kwargs=OVERRIDES).calc(s),
    "lammps": lambda _, s: LAMMPSMDCalc("TensorNet-MatPES-PBE-2025.2", relax_calc_kwargs=OVERRIDES).calc(s),
}


@pytest.mark.parametrize("case", CASES)
def test_relax_calc_kwargs_override(
    Si: Structure, matpes_calculator: PESCalculator, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """relax_calc_kwargs may repeat arguments the calculator sets itself, and they take precedence."""
    captured: dict[str, Any] = {}

    def init(self: RelaxCalc, calculator: Any, **kwargs: Any) -> None:  # noqa: ARG001
        captured.update(kwargs)
        raise _Captured

    monkeypatch.setattr(RelaxCalc, "__init__", init)
    with pytest.raises(_Captured):
        CASES[case](matpes_calculator, Si)
    assert {key: captured[key] for key in OVERRIDES} == OVERRIDES


class _PrelaxCalc(_DummyCalc):
    relax_structure = True


def test_check_and_prelax_raises_on_unrelaxed_cell(Li2O: Structure, matpes_calculator: PESCalculator) -> None:
    """A cell relaxation stopped with ~zero forces but a large stress must not be passed downstream."""
    expanded = Li2O.copy()
    expanded.scale_lattice(1.1 * Li2O.volume)
    calc = _PrelaxCalc(matpes_calculator, {"max_steps": 3, "relax_cell": True, "fmax": 0.1})
    with pytest.raises(RuntimeError, match=r"Pre-relaxation did not converge.*GPa"):
        calc._check_and_prelax(expanded, {})  # noqa: SLF001

    calc = _PrelaxCalc(matpes_calculator, {"max_steps": 500, "relax_cell": True, "fmax": 0.1})
    result, structure = calc._check_and_prelax(expanded, {})  # noqa: SLF001
    assert result["is_converged"] is True
    assert structure.volume < expanded.volume


class _EnvProbe(PropCalc):
    """Reports the worker's thread-limit env; fails on the string "bad"."""

    def calc(self, structure: Any) -> dict[str, Any]:
        if structure == "bad":
            raise ValueError("bad structure")
        return {"omp": os.environ.get("OMP_NUM_THREADS"), "pid": os.getpid()}


@pytest.fixture
def no_thread_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for var in LokyBackend.MAX_NUM_THREADS_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("matcalc._base.cpu_count", lambda: 8)
    return monkeypatch


@pytest.mark.parametrize(
    ("env", "n_jobs", "kwargs", "expected"),
    [
        ({}, 4, {}, None),  # joblib already sets cpu_count // n_workers
        ({"OMP_NUM_THREADS": "8"}, 4, {}, 2),
        ({"OMP_NUM_THREADS": "8"}, -1, {}, 1),
        ({"OMP_NUM_THREADS": "8"}, -5, {}, 2),  # 8 + 1 - 5 = 4 workers
        ({"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "1"}, 2, {}, 1),  # never raise an inherited limit
        ({"OMP_NUM_THREADS": "2"}, 4, {}, None),  # inherited limit already fits
        ({"OMP_NUM_THREADS": "8"}, 1, {}, None),
        ({"OMP_NUM_THREADS": "8"}, None, {}, None),
        ({"OMP_NUM_THREADS": "8"}, 4, {"backend": "threading"}, None),
        ({"OMP_NUM_THREADS": "8"}, 4, {"prefer": "threads"}, None),
    ],
)
def test_worker_thread_cap(
    no_thread_env: pytest.MonkeyPatch, env: dict[str, str], n_jobs: int | None, kwargs: dict, expected: int | None
) -> None:
    for var, val in env.items():
        no_thread_env.setenv(var, val)
    assert _worker_thread_cap(n_jobs, kwargs) == expected


def test_worker_thread_cap_respects_parallel_config(no_thread_env: pytest.MonkeyPatch) -> None:
    no_thread_env.setenv("OMP_NUM_THREADS", "8")
    with parallel_config(n_jobs=4):
        assert _worker_thread_cap(None, {}) == 2
    with parallel_config(backend="loky", inner_max_num_threads=3):
        assert _worker_thread_cap(4, {}) is None
    with parallel_config(backend="threading"):
        assert _worker_thread_cap(4, {}) is None


def test_calc_many_caps_inherited_worker_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """An inherited OMP_NUM_THREADS must not be copied into every loky worker."""
    monkeypatch.setenv("OMP_NUM_THREADS", str(cpu_count()))
    results = _EnvProbe().calc_many(["ok", "bad", "ok", "ok"], n_jobs=2, allow_errors=True)
    assert inspect.isgenerator(results)
    results = list(results)
    assert results[1] is None
    assert {r["omp"] for r in results if r is not None} == {str(max(1, cpu_count() // 2))}
    assert os.getpid() not in {r["pid"] for r in results if r is not None}

    with pytest.raises(ValueError, match="bad structure"):
        list(_EnvProbe().calc_many(["ok", "bad"], n_jobs=2))
