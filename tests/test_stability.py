from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

import pytest
from monty.serialization import loadfn

from matcalc._relaxation import RelaxCalc
from matcalc._stability import ELEMENTAL_REFS_DIR, EnergeticsCalc, _infer_elemental_refs

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ase import Atoms
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


def test_energetics_calc(
    Li2O: Structure,
    matpes_calculator: PESCalculator,
) -> None:
    result = EnergeticsCalc(matpes_calculator).calc(Li2O)
    for key in (
        "final_structure",
        "formation_energy_per_atom",
        "cohesive_energy_per_atom",
    ):
        assert key in result, f"{key=} not in result"
    assert result["formation_energy_per_atom"] == pytest.approx(-1.7850605646769206, rel=1e-2)
    assert result["cohesive_energy_per_atom"] == pytest.approx(-4.053125177202149, rel=1e-2)
    assert result["_units"]["formation_energy_per_atom"] == "eV/atom"
    assert result["_units"]["cohesive_energy_per_atom"] == "eV/atom"
    assert result["_units"]["energy"] == "eV"

    # Note that the value differs from MP primarily because of the correction. A correction of -0.70 eV is applied to
    # each O atom, which accounts almost entirely for the difference between this predicted formation energy and the
    # MP calculated value of -2.0 eV.

    result = EnergeticsCalc(matpes_calculator, use_gs_reference=True).calc(Li2O)
    for key in (
        "final_structure",
        "formation_energy_per_atom",
        "cohesive_energy_per_atom",
    ):
        assert key in result, f"{key=} not in result"
    assert result["formation_energy_per_atom"] == pytest.approx(-1.8454608797021486, rel=1e-2)
    assert result["cohesive_energy_per_atom"] == pytest.approx(-4.053125177202149, rel=1e-2)


def test_energetics_calc_element_atoms(
    Si_atoms: Atoms,
    matpes_calculator: PESCalculator,
) -> None:
    result = EnergeticsCalc(matpes_calculator).calc(Si_atoms)
    for key in (
        "final_structure",
        "formation_energy_per_atom",
        "cohesive_energy_per_atom",
    ):
        assert key in result, f"{key=} not in result"
    # This is an element. The formation energy should be close to 0.
    assert result["formation_energy_per_atom"] == pytest.approx(0, abs=1e-2)


def test_energetics_cohesive_missing_ref(
    Li2O: Structure,
    matpes_calculator: PESCalculator,
) -> None:
    """When the elemental refs lack ``energy_atomic``, cohesive energy is returned as None."""
    elemental_refs = {
        "Li": {"energy_per_atom": -4.5},
        "O": {"energy_per_atom": -0.5},
    }
    result = EnergeticsCalc(
        matpes_calculator,
        elemental_refs=elemental_refs,
        use_gs_reference=True,
        relax_structure=False,
    ).calc(Li2O)

    assert result["cohesive_energy_per_atom"] is None
    # With use_gs_reference and fixed refs the formation energy is deterministic from
    # the PES energy; it must be a finite number (not None/inf/nan).
    assert math.isfinite(result["formation_energy_per_atom"])
    assert result["_units"]["cohesive_energy_per_atom"] == "eV/atom"


def test_energetics_all_references_yield_none(
    Si_atoms: Atoms,
    matpes_calculator: PESCalculator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every reference relaxation returns None, so the None-filter drops them all.

    ``get_gs_energy`` relaxes each elemental reference via ``RelaxCalc.calc_many`` and
    keeps only results where ``r is not None``. When every reference relaxation fails
    (one ``None`` per reference, as ``allow_errors=True`` mode does), the filter empties
    the list and the RuntimeError fires. This is the production "all reference
    relaxations failed" case, not a degenerate empty reference list.
    """
    # A real Si reference structure: the failure comes from calc_many yielding None,
    # not from an empty structure list.
    elemental_refs = {"Si": {"structure": Si_atoms}}

    def fake_calc_many(self: object, structures: list, **kwargs: object) -> Iterator[None]:  # noqa: ARG001
        # Mirror allow_errors=True mode: one None per input reference structure.
        for _ in structures:
            yield None

    monkeypatch.setattr("matcalc._relaxation.RelaxCalc.calc_many", fake_calc_many)

    with pytest.raises(RuntimeError, match="All elemental reference relaxations failed"):
        EnergeticsCalc(
            matpes_calculator,
            elemental_refs=elemental_refs,
            use_gs_reference=False,
            relax_structure=False,
        ).calc(Si_atoms)


def test_energetics_empty_reference_structure_list(
    Si_atoms: Atoms,
    matpes_calculator: PESCalculator,
) -> None:
    """An empty reference structure list yields no energies, so calc raises RuntimeError.

    The degenerate counterpart: ``calc_many([])`` produces nothing, which the same
    ``if not energies_per_atom`` guard catches.
    """
    elemental_refs: dict[str, dict[str, list]] = {"Si": {"structure": []}}
    with pytest.raises(RuntimeError, match="All elemental reference relaxations failed"):
        EnergeticsCalc(
            matpes_calculator,
            elemental_refs=elemental_refs,
            use_gs_reference=False,
            relax_structure=False,
        ).calc(Si_atoms)


def test_energetics_reference_relaxations_cached(
    Li2O: Structure,
    matpes_calculator: PESCalculator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Elemental references are relaxed once per calculator, not on every ``calc`` call."""
    n_ref_relaxations = 0
    orig_calc_many = RelaxCalc.calc_many

    def counting_calc_many(self: RelaxCalc, structures: list, *args: Any, **kwargs: Any) -> Any:
        nonlocal n_ref_relaxations
        n_ref_relaxations += len(structures)
        return orig_calc_many(self, structures, *args, **kwargs)

    monkeypatch.setattr(RelaxCalc, "calc_many", counting_calc_many)

    energetics_calc = EnergeticsCalc(matpes_calculator, relax_structure=False)
    first = energetics_calc.calc(Li2O)
    assert n_ref_relaxations == 2  # one MatPES reference structure each for Li and O
    second = energetics_calc.calc(Li2O)
    assert n_ref_relaxations == 2
    assert second["formation_energy_per_atom"] == pytest.approx(first["formation_energy_per_atom"], abs=1e-6)


def test_energetics_missing_element_ref(Li2O: Structure, matpes_calculator: PESCalculator) -> None:
    """A composition element absent from elemental_refs raises a KeyError naming it."""
    with pytest.raises(KeyError, match="No elemental reference for 'O'"):
        EnergeticsCalc(
            matpes_calculator,
            elemental_refs={"Li": {"energy_per_atom": -1.9}},
            use_gs_reference=True,
            relax_structure=False,
        ).calc(Li2O)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("TensorNet-MatPES-r2SCAN-2025.2", "MatPES-r2SCAN"),
        ("r2scan", "MatPES-r2SCAN"),
        ("MACE-OMat24-ft-MatPES-r2SCAN-0", "MatPES-r2SCAN"),
        ("TensorNet-PES-MatPES-r2SCAN-2025.2", "MatPES-r2SCAN"),  # Raw MatGL name (load_universal fallback).
        ("TensorNet-MatPES-PBE-2025.2", "MatPES-PBE"),
        ("pbe", "MatPES-PBE"),
        ("M3GNet-MatPES-PBE-2025.2", "MatPES-PBE"),
    ],
)
def test_infer_elemental_refs_from_name(model: str, expected: str) -> None:
    """The builtin reference set follows the training functional encoded in the model name."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _infer_elemental_refs(model) == expected


@pytest.mark.parametrize("model", ["PET-MAD-PBEsol-1.0.2-S", "MACE-MPtrj-0b3-medium", "UMA-S-1.2"])
def test_infer_elemental_refs_unknown_functional_warns(model: str) -> None:
    """Names without an exact PBE/r2SCAN token (incl. PBEsol) fall back to MatPES-PBE with a warning."""
    with pytest.warns(UserWarning, match="Cannot infer the DFT functional"):
        assert _infer_elemental_refs(model) == "MatPES-PBE"


@pytest.mark.parametrize(
    ("model", "expected"),
    [("TensorNet-MatPES-r2SCAN-2025.2", "MatPES-r2SCAN"), ("TensorNet-MatPES-PBE-2025.2", "MatPES-PBE")],
)
def test_energetics_infers_refs_from_model_name(
    model: str,
    expected: str,
    matpes_calculator: PESCalculator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EnergeticsCalc(<model name>) loads the elemental references matching the model's functional."""
    # Only the name matters for the inference; avoid loading a second model.
    monkeypatch.setattr("matcalc._base.PESCalculator.load_universal", lambda _name: matpes_calculator)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        energetics_calc = EnergeticsCalc(model)
    assert energetics_calc.elemental_refs == loadfn(ELEMENTAL_REFS_DIR / f"{expected}-Element-Refs.json.gz")


def test_energetics_calculator_instance_warns(matpes_calculator: PESCalculator) -> None:
    """A Calculator instance gives no functional to infer: MatPES-PBE refs with a UserWarning."""
    with pytest.warns(UserWarning, match="Cannot infer the DFT functional"):
        energetics_calc = EnergeticsCalc(matpes_calculator)
    assert energetics_calc.elemental_refs == loadfn(ELEMENTAL_REFS_DIR / "MatPES-PBE-Element-Refs.json.gz")


@pytest.mark.parametrize("refs", ["MatPES-r2SCAN", "MP-PBE"])
def test_energetics_explicit_refs_respected(refs: str, matpes_calculator: PESCalculator) -> None:
    """An explicit elemental_refs value is used as given, without inference or warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        energetics_calc = EnergeticsCalc(matpes_calculator, elemental_refs=refs)  # type: ignore[arg-type]
    assert energetics_calc.elemental_refs == loadfn(ELEMENTAL_REFS_DIR / f"{refs}-Element-Refs.json.gz")
