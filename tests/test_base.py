"""Tests for shared PropCalc helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from matcalc import (
    AdsorptionCalc,
    ElasticityCalc,
    EnergeticsCalc,
    GBCalc,
    InterfaceCalc,
    LAMMPSMDCalc,
    MDCalc,
    PhononCalc,
    SurfaceCalc,
)
from matcalc._base import PropCalc
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


class _Captured(Exception):  # noqa: N818
    """Raised by the patched RelaxCalc.__init__ to stop the calculation once its arguments are known."""


CASES: dict[str, Callable[[Any, Structure], Any]] = {
    "surface": lambda c, s: SurfaceCalc(c, relax_calc_kwargs=OVERRIDES).calc_slabs(s),
    "gb": lambda c, s: GBCalc(c, relax_calc_kwargs=OVERRIDES).calc({"bulk": s, "grain_boundary": s}),
    "interface": lambda c, s: InterfaceCalc(c, relax_calc_kwargs=OVERRIDES).calc(
        {"interface": s, "film_bulk": s, "substrate_bulk": s}
    ),
    "adsorption_bulk": lambda c, s: AdsorptionCalc(c, relax_calc_kwargs=OVERRIDES).calc_bulk(s),
    "adsorption_slab": lambda c, s: AdsorptionCalc(c, relax_calc_kwargs=OVERRIDES).calc_slab(s),
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
