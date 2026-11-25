"""Tests for EOSCalc class"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pytest
from ase.filters import ExpCellFilter

from matcalc import EOSCalc, RelaxCalc

if TYPE_CHECKING:
    from ase import Atoms
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


def test_eos_calc(
    Li2O: Structure,
    LiFePO4: Structure,
    matpes_calculator: PESCalculator,
) -> None:
    """Tests for EOSCalc class"""
    # Note that the fmax is probably too high. This is for testing purposes only.
    eos_calc = EOSCalc(matpes_calculator, fmax=0.1, relax_calc_kwargs={"cell_filter": ExpCellFilter})
    result = eos_calc.calc(Li2O)

    assert result["bulk_modulus"] == pytest.approx(84.08164565338282, rel=1e-1)
    assert result["_units"]["bulk_modulus"] == "GPa"
    assert result["_units"]["eos.volumes"] == "A^3"
    assert result["_units"]["eos.energies"] == "eV"
    assert result["_units"]["energy"] == "eV"
    assert {*result["eos"]} == {"volumes", "energies"}
    assert result["eos"]["volumes"] == pytest.approx(
        [
            18.370519873441342,
            19.622638380261037,
            20.93039763863155,
            22.295007230108368,
            23.71767673624691,
            25.199615738602667,
            26.742033818731066,
            28.34614055818755,
            30.01314553852758,
            31.74425834130665,
            33.54068854808016,
        ],
        rel=1e-1,
    )
    assert result["eos"]["energies"] == pytest.approx(
        [
            -13.49211311340332,
            -13.829581260681152,
            -14.058572769165039,
            -14.203819274902344,
            -14.280638694763184,
            -14.301274299621582,
            -14.275708198547363,
            -14.212100982666016,
            -14.11718463897705,
            -13.99584674835205,
            -13.851357460021973,
        ],
        rel=1e-2,
    )
    # Unrelaxed LiFePO4 stays well conditioned within +-5% strain; at the default +-10%, the shape relaxations of the
    # most expanded points fall into different local minima depending on the optimizer path.
    eos_calc = EOSCalc(matpes_calculator, relax_structure=False, max_abs_strain=0.05)
    results = list(eos_calc.calc_many([Li2O, LiFePO4]))
    assert len(results) == 2
    assert results[1]["bulk_modulus"] == pytest.approx(68.09, rel=1e-2)
    assert results[1]["r2_score"] > 0.999


def test_eos_calc_atoms(
    Si_atoms: Atoms,
    matpes_calculator: PESCalculator,
) -> None:
    """Tests for EOSCalc class"""
    # Note that the fmax is probably too high. This is for testing purposes only.
    eos_calc = EOSCalc(matpes_calculator, fmax=0.1, relax_structure=False)
    result = eos_calc.calc(Si_atoms)

    assert result["bulk_modulus"] == pytest.approx(76.57813407426684, rel=1e-1)


def test_eos_calc_keeps_user_cell_filter_kwargs(
    Si: Structure,
    matpes_calculator: PESCalculator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User ``cell_filter_kwargs`` (e.g. a mask) must survive the constant-volume setting of the EOS points."""
    captured: list[dict[str, Any]] = []
    orig_init = RelaxCalc.__init__

    def spy_init(self: RelaxCalc, calculator: Any, **kwargs: Any) -> None:
        captured.append(kwargs)
        orig_init(self, calculator, **kwargs)

    monkeypatch.setattr(RelaxCalc, "__init__", spy_init)
    mask = [1, 1, 0, 0, 0, 1]
    result = EOSCalc(
        matpes_calculator,
        n_points=5,
        relax_structure=False,
        relax_calc_kwargs={"cell_filter_kwargs": {"mask": mask}},
    ).calc(Si)

    assert result["bulk_modulus"] > 0
    (eos_kwargs,) = captured
    assert eos_kwargs["relax_cell"] is True
    assert eos_kwargs["cell_filter_kwargs"] == {"mask": mask, "constant_volume": True}


def test_eos_calc_fit_failure_keeps_ev(
    Si: Structure,
    matpes_calculator: PESCalculator,
) -> None:
    """A failed Birch-Murnaghan fit warns and still returns the energy-volume data."""
    expanded = Si.copy()
    expanded.apply_strain(0.15)
    eos_calc = EOSCalc(matpes_calculator, relax_structure=False, allow_shape_change=False)
    with pytest.warns(RuntimeWarning, match="Birch-Murnaghan EOS fit failed"):
        result = eos_calc.calc(expanded)

    assert math.isnan(result["bulk_modulus"])
    assert math.isnan(result["r2_score"])
    assert len(result["eos"]["volumes"]) == 11
    assert len(result["eos"]["energies"]) == 11
    assert result["eos"]["volumes"] == sorted(result["eos"]["volumes"])
    assert result["eos"]["volumes"][5] == pytest.approx(expanded.volume)
    assert result["eos"]["energies"] == sorted(result["eos"]["energies"])


def test_eos_calc_eos_model(
    Si: Structure,
    matpes_calculator: PESCalculator,
) -> None:
    """The fitted equation of state is selected with ``eos``."""
    result_bm = EOSCalc(matpes_calculator, relax_structure=False).calc(Si)

    assert result_bm["_units"]["bulk_modulus"] == "GPa"

    for eos in ("vinet", "murnaghan", "pourier_tarantola"):
        result = EOSCalc(matpes_calculator, relax_structure=False, eos=eos).calc(Si)
        assert math.isfinite(result["bulk_modulus"])
        assert result["bulk_modulus"] > 0
        assert result["bulk_modulus"] == pytest.approx(result_bm["bulk_modulus"], rel=1e-1)
        assert result["r2_score"] > 0.99
        assert result["_units"]["bulk_modulus"] == "GPa"


@pytest.mark.parametrize(("eos", "eos_name"), [("vinet", "Vinet"), ("pourier_tarantola", "Pourier-Tarantola")])
def test_eos_calc_eos_model_fit_failure(
    eos: str,
    eos_name: str,
    Si: Structure,
    matpes_calculator: PESCalculator,
) -> None:
    """A failed fit of another EOS model warns with the model name and still returns the energy-volume data."""
    expanded = Si.copy()
    expanded.apply_strain(0.15)
    eos_calc = EOSCalc(matpes_calculator, relax_structure=False, allow_shape_change=False, eos=eos)
    with pytest.warns(RuntimeWarning, match=f"{eos_name} EOS fit failed"):
        result = eos_calc.calc(expanded)

    assert math.isnan(result["bulk_modulus"])
    assert math.isnan(result["r2_score"])
    assert len(result["eos"]["volumes"]) == 11
    assert len(result["eos"]["energies"]) == 11


def test_eos_calc_unknown_eos(matpes_calculator: PESCalculator) -> None:
    """An unknown EOS model name is rejected at construction."""
    with pytest.raises(ValueError, match=r"Unknown eos 'not_an_eos'\. Allowed values: .*'birch_murnaghan'.*'vinet'"):
        EOSCalc(matpes_calculator, eos="not_an_eos")
