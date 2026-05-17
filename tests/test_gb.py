"""Tests for the GBCalc class."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from matcalc import GBCalc

if TYPE_CHECKING:
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


def test_gb_calc(Si: Structure, matpes_calculator: PESCalculator) -> None:
    """Test GBCalc.calc_gb on the Si Sigma3 [111] 60° twin boundary."""
    gb_calc = GBCalc(
        calculator=matpes_calculator,
        relax_bulk=True,
        relax_gb=True,
        fmax=0.1,
        max_steps=100,
    )

    results = gb_calc.calc_gb(
        Si,
        sigma=3,
        rotation_axis=(1, 1, 1),
        gb_plane=(1, 1, 2),
        rotation_angle=60.0,
        expand_times=1,
    )

    assert "final_bulk" in results
    assert "final_grain_boundary" in results
    assert "gb_relax_energy" in results
    assert "grain_boundary_energy" in results
    assert "bulk_energy_per_atom" in results

    assert isinstance(results["bulk_energy_per_atom"], float)
    assert isinstance(results["gb_relax_energy"], float)
    assert isinstance(results["grain_boundary_energy"], float)
    assert math.isfinite(results["grain_boundary_energy"])
    assert results["grain_boundary_energy"] >= 0


def test_gb_calc_from_dict(Si: Structure, matpes_calculator: PESCalculator) -> None:
    """Test GBCalc.calc with explicit grain_boundary/bulk dict input."""
    gb_calc = GBCalc(
        calculator=matpes_calculator,
        relax_bulk=True,
        relax_gb=True,
        fmax=0.1,
        max_steps=100,
    )

    results = gb_calc.calc({"grain_boundary": Si, "bulk": Si})

    assert "final_bulk" in results
    assert "final_grain_boundary" in results
    assert "gb_relax_energy" in results
    assert "grain_boundary_energy" in results
    assert "bulk_energy_per_atom" in results

    assert isinstance(results["bulk_energy_per_atom"], float)
    assert isinstance(results["gb_relax_energy"], float)
    assert isinstance(results["grain_boundary_energy"], float)
    assert math.isfinite(results["grain_boundary_energy"])


def test_gb_calc_invalid_input(Si: Structure, matpes_calculator: PESCalculator) -> None:
    """Verify ValueError for non-dict or incomplete dict inputs to calc."""
    gb_calc = GBCalc(calculator=matpes_calculator)
    match = "For grain boundary calculations, structure must be a dict in one of the following formats:"
    with pytest.raises(ValueError, match=match):
        gb_calc.calc(Si)
    with pytest.raises(ValueError, match=match):
        gb_calc.calc({"grain_boundary": Si})


def test_gb_calc_invalid_rotation_angle(Si: Structure, matpes_calculator: PESCalculator) -> None:
    """Verify ValueError when rotation_angle has no match for the given sigma."""
    gb_calc = GBCalc(calculator=matpes_calculator, relax_bulk=False, relax_gb=False)
    with pytest.raises(ValueError, match="No matching rotation angle"):
        gb_calc.calc_gb(
            Si,
            sigma=3,
            rotation_axis=(1, 1, 1),
            gb_plane=(1, 1, 2),
            rotation_angle=45.0,
        )
