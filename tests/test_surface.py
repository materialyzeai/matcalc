"""Tests for the SurfaceCalc class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from matcalc import SurfaceCalc

if TYPE_CHECKING:
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


def test_surface_calc_basic(Si: Structure, matpes_calculator: PESCalculator) -> None:
    """
    Test the basic workflow:
      1) calc_slabs on a known structure
      2) calc on the resulting slabs
      3) Check the final results
    """
    surf_calc = SurfaceCalc(
        calculator=matpes_calculator,
        relax_bulk=True,
        relax_slab=True,
        fmax=0.1,
        max_steps=100,
    )

    results = list(
        surf_calc.calc_slabs(
            Si,
            miller_index=(1, 1, 1),
            min_slab_size=10.0,
            min_vacuum_size=10.0,
        )
    )
    assert len(results) == 2, "Expected two slabs for Silicon (111)."

    slab_res = results[0]

    assert "final_bulk" in slab_res
    assert "final_slab" in slab_res

    assert slab_res["bulk_energy_per_atom"] == pytest.approx(-5.419038772583008, rel=1e-1)
    assert slab_res["slab_energy"] == pytest.approx(-42.81388473510742, rel=1e-1)
    assert slab_res["surface_energy"] == pytest.approx(0.04857103179280209, rel=1e-1)

    slab_res = results[1]

    assert slab_res["bulk_energy_per_atom"] == pytest.approx(-5.419038772583008, rel=1e-1)
    assert slab_res["slab_energy"] == pytest.approx(-39.40110397338867, rel=1e-1)
    assert slab_res["surface_energy"] == pytest.approx(0.15327125170767797, rel=1e-1)


def test_surface_calc_invalid_input(Si: Structure, matpes_calculator: PESCalculator) -> None:
    """
    If the user passes a non-dict to calc, it should raise ValueError.
    """
    surf_calc = SurfaceCalc(calculator=matpes_calculator)
    with pytest.raises(
        ValueError,
        match="For surface calculations, structure must be a dict in one of the following formats:",
    ):
        surf_calc.calc(Si)
    with pytest.raises(
        ValueError, match="For surface calculations, structure must be a dict in one of the following formats:"
    ):
        surf_calc.calc({"slab": Si})
