"""Tests for the GBCalc class."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
from pymatgen.core import Lattice, Structure

from matcalc import GBCalc

if TYPE_CHECKING:
    from matgl.ext.ase import PESCalculator


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


def test_gb_calc_trigonal(matpes_calculator: PESCalculator) -> None:
    """Rotation angles of a trigonal (hexagonal-lattice) crystal use the hexagonal CSL, not the tetragonal one.

    The Sigma7 [001] twist of the hexagonal lattice is at 21.79°, which the tetragonal formulas do not produce.
    """
    tis2 = Structure.from_spacegroup(
        "P-3m1", Lattice.hexagonal(3.4, 5.7), ["Ti", "S"], [[0, 0, 0], [1 / 3, 2 / 3, 0.25]]
    )
    gb_calc = GBCalc(calculator=matpes_calculator, relax_bulk=False, relax_gb=False)
    results = gb_calc.calc_gb(
        tis2, sigma=7, rotation_axis=(0, 0, 1), gb_plane=(0, 0, 1), rotation_angle=21.79, expand_times=1
    )
    assert results["grain_boundary"].rotation_angle == pytest.approx(21.786789, abs=1e-4)
    assert math.isfinite(results["grain_boundary_energy"])


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


def test_gb_calc_bulk_relaxation(Si: Structure, matpes_calculator: PESCalculator) -> None:
    """The bulk goes through PropCalc._check_and_prelax: an unconverged bulk relaxation raises, and with
    relax_bulk=False the reference is a single point on the input bulk."""
    rattled = Si.copy().perturb(0.2, seed=0)
    with pytest.raises(RuntimeError, match="Pre-relaxation did not converge"):
        GBCalc(calculator=matpes_calculator, max_steps=1, relax_gb=False).calc({"grain_boundary": Si, "bulk": rattled})

    results = GBCalc(calculator=matpes_calculator, relax_bulk=False, relax_gb=False).calc(
        {"grain_boundary": Si, "bulk": Si}
    )
    assert results["final_bulk"] == Si
    assert results["bulk_energy_per_atom"] == pytest.approx(
        matpes_calculator.get_potential_energy(Si.to_ase_atoms()) / len(Si)
    )
