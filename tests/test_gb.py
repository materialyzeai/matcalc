"""Tests for the GBCalc class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from matcalc import GBCalc

if TYPE_CHECKING:
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


def test_GB_calc_basic(Si: Structure, m3gnet_calculator: PESCalculator) -> None:
    """
    Test the basic workflow:
      1) calc_gbs on a known structure
      2) Check the final results
    """
    gb_calc = GBCalc(
        calculator=m3gnet_calculator,
        relax_bulk=True,
        relax_gb=True,
        fmax=0.1,
        max_steps=100,
    )

    results = gb_calc.calc_gb(
        Si,
        sigma=1,
        rotation_axis=(1, 1, 1),
        gb_plane=(1, 1, 1),
        rotation_angle=0.0,
        expand_times=1,
    )

    assert "final_bulk" in results
    assert "final_grain_boundary" in results
    assert "gb_relax_energy" in results
    assert "grain_boundary_energy" in results
    assert "bulk_energy_per_atom" in results

    assert results["bulk_energy_per_atom"] == pytest.approx(ABC, rel=1e-1)
    assert results["gb_relax_energy"] == pytest.approx(ABC, rel=1e-1)
    assert results["grain_boundary_energy"] == pytest.approx(ABC, rel=1e-1)

    results = gb_calc.calc({"grain_boundary": Si, "bulk": Si})

    assert "final_bulk" in results
    assert "final_grain_boundary" in results
    assert "gb_relax_energy" in results
    assert "grain_boundary_energy" in results
    assert "bulk_energy_per_atom" in results

    assert results["bulk_energy_per_atom"] == pytest.approx(ABC, rel=1e-1)
    assert results["gb_relax_energy"] == pytest.approx(ABC, rel=1e-1)
    assert results["grain_boundary_energy"] == pytest.approx(ABC, rel=1e-1)


def test_surface_calc_invalid_input(Si: Structure, m3gnet_calculator: PESCalculator) -> None:
    """
    If the user passes a non-dict to calc, it should raise ValueError.
    """
    surf_calc = GBCalc(calculator=m3gnet_calculator)
    with pytest.raises(
        ValueError,
        match="For grain boundary calculations, structure must be a dict in one of the following formats:",
    ):
        surf_calc.calc(Si)
    with pytest.raises(
        ValueError, match="For grain boundary calculations, structure must be a dict in one of the following formats:"
    ):
        surf_calc.calc({"grain_boundary": Si})
