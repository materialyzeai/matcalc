from __future__ import annotations

import re
import shutil
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from matcalc import RelaxCalc
from matcalc.utils import (
    MODEL_ALIASES,
    MODEL_REGISTRY,
    UNIVERSAL_CALCULATORS,
    PESCalculator,
    _resolve_canonical,
)

DIR = Path(__file__).parent.absolute()

if TYPE_CHECKING:
    from pymatgen.core import Structure

UNIVERSAL_CALCULATORS = [calc for calc in UNIVERSAL_CALCULATORS if "DGL" not in calc.name and "CHGNet" not in calc.name]


def _map_calculators_to_packages(calculators: list) -> dict[str, str]:
    provider_package_map = {
        "matgl": "matgl",
        "mace_mp": "mace",
        "sevennet": "sevenn",
        "grace": "tensorpotential",
        "orb": "orb_models",
        "mattersim": "mattersim",
        "fairchem": "fairchem",
        "upet": "upet",
        "deepmd": "deepmd",
    }
    return {calc.name: provider_package_map[MODEL_REGISTRY[calc.name]["provider"]] for calc in calculators}


UNIVERSAL_TO_PACKAGE = _map_calculators_to_packages(UNIVERSAL_CALCULATORS)


@pytest.mark.parametrize(
    ("expected_unit", "expected_weight"),
    [
        ("eV/A3", 0.006241509125883258),
        ("GPa", 1.0),
    ],
)
@pytest.mark.skipif(not find_spec("maml"), reason="maml is not installed")
def test_pescalculator_load_mtp(expected_unit: str, expected_weight: float) -> None:
    calc = PESCalculator.load_mtp(
        filename=DIR / "pes" / "MTP-Cu-2020.1-PES" / "fitted.mtp",
        elements=["Cu"],
    )
    assert isinstance(calc, Calculator)
    assert PESCalculator(
        potential=calc.potential,
        stress_unit=expected_unit,
    ).stress_weight == pytest.approx(expected_weight)
    with pytest.raises(ValueError, match=re.escape("Unsupported stress_unit: Pa. Must be 'GPa' or 'eV/A3'.")):
        PESCalculator(potential=calc.potential, stress_unit="Pa")


@pytest.mark.skipif(not find_spec("maml"), reason="maml is not installed")
def test_pescalculator_load_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    # gap.xml references its soapparam files relative to the repository root.
    monkeypatch.chdir(DIR.parent)
    calc = PESCalculator.load_gap(filename=DIR / "pes" / "GAP-NiMo-2020.3-PES" / "gap.xml")
    assert isinstance(calc, Calculator)


@pytest.mark.skipif(not find_spec("maml"), reason="maml is not installed")
def test_pescalculator_load_nnp() -> None:
    calc = PESCalculator.load_nnp(
        input_filename=DIR / "pes" / "NNP-Cu-2020.1-PES" / "input.nn",
        scaling_filename=DIR / "pes" / "NNP-Cu-2020.1-PES" / "scaling.data",
        weights_filenames=[DIR / "pes" / "NNP-Cu-2020.1-PES" / "weights.029.data"],
    )
    assert isinstance(calc, Calculator)


@pytest.mark.skipif(not find_spec("maml"), reason="maml is not installed")
@pytest.mark.skipif(not shutil.which("lmp"), reason="lammps (lmp) executable is not on PATH")
def test_pescalculator_load_snap() -> None:
    for name in ("SNAP", "qSNAP"):
        calc = PESCalculator.load_snap(
            param_file=DIR / "pes" / f"{name}-Cu-2020.1-PES" / "SNAPotential.snapparam",
            coeff_file=DIR / "pes" / f"{name}-Cu-2020.1-PES" / "SNAPotential.snapcoeff",
        )
        assert isinstance(calc, Calculator)


@pytest.mark.skipif(not find_spec("pyace"), reason="pyace is not installed")
def test_pescalculator_load_ace() -> None:
    calc = PESCalculator.load_ace(basis_set=DIR / "pes" / "ACE-Cu-2021.5.15-PES" / "Cu-III.yaml")
    assert isinstance(calc, Calculator)


@pytest.mark.skipif(not find_spec("nequip"), reason="nequip is not installed")
def test_pescalculator_load_nequip() -> None:
    calc = PESCalculator.load_nequip(model_path=DIR / "pes" / "NequIP-am-Al2O3-2023.9-PES" / "default.pth")
    assert isinstance(calc, Calculator)


@pytest.mark.skipif(not find_spec("deepmd"), reason="deepmd is not installed")
def test_pescalculator_load_deepmd() -> None:
    calc = PESCalculator.load_deepmd(
        model_path=(DIR / "pes" / "DPA3-LAM-2025.3.14-PES" / "2025-03-14-dpa3-openlam.pth")
    )
    assert isinstance(calc, Calculator)


@pytest.mark.parametrize("name", (c.name for c in UNIVERSAL_CALCULATORS))
def test_pescalculator_load_universal(Li2O: Structure, name: str) -> None:
    if name not in UNIVERSAL_TO_PACKAGE:
        pytest.fail(f"No package mapping found for {name}. Please add it to UNIVERSAL_TO_PACKAGE.")

    pkg_name = UNIVERSAL_TO_PACKAGE[name]

    if not find_spec(pkg_name):
        pytest.skip(f"{pkg_name} is not installed. Skipping {name} test.")

    calc = PESCalculator.load_universal(name)
    assert isinstance(calc, Calculator)
    same_calc = PESCalculator.load_universal(calc)  # test ASE Calculator classes are returned as-is
    assert calc is same_calc
    # We run a basic relaxation calc to make sure that all calculators work.
    relaxed_calc = RelaxCalc(calc, relax_cell=False, relax_atoms=False)
    result = relaxed_calc.calc(Li2O)
    assert isinstance(result, dict)

    name = "whatever"
    with pytest.raises(ValueError, match=f"Unrecognized {name=}"):
        PESCalculator.load_universal(name)


@pytest.mark.skip(reason="Skipping test_pescalculator_calculate because problems in upstream maml for now.")
def test_pescalculator_calculate() -> None:
    calc = PESCalculator.load_snap(
        param_file=DIR / "pes" / "SNAP-Cu-2020.1-PES" / "SNAPotential.snapparam",
        coeff_file=DIR / "pes" / "SNAP-Cu-2020.1-PES" / "SNAPotential.snapcoeff",
    )

    atoms = Atoms(
        "Cu4",
        scaled_positions=[(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)],
        cell=[3.57743067] * 3,
        pbc=True,
    )

    atoms.set_calculator(calc)

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stresses = atoms.get_stress()

    assert isinstance(energy, float)
    assert list(forces.shape) == [4, 3]
    assert list(stresses.shape) == [6]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("m3gnet", "M3GNet-MatPES-PBE-2025.2"),
        ("M3GNet-MatPES-PBE-2025.2", "M3GNet-MatPES-PBE-2025.2"),
        ("M3GNet-MatPES-r2SCAN-2025.2", "M3GNet-MatPES-r2SCAN-2025.2"),
        ("m3gnet-pes-matpes-pbe-2025.2", "M3GNet-MatPES-PBE-2025.2"),
        ("m3gnet-pes-matpes-r2scan-2025.2", "M3GNet-MatPES-r2SCAN-2025.2"),
        ("M3GNet-MatPES-PBE-2025.1", None),
        ("m3gnet-pes-matpes-pbe-2025.1", None),
    ],
)
def test_resolve_m3gnet(name: str, expected: str | None) -> None:
    assert _resolve_canonical(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("PET-MAD-PBEsol-1.0.2-S", "PET-MAD-PBEsol-1.0.2-S"),
        ("pet-mad-pbesol-1.0.2-s", "PET-MAD-PBEsol-1.0.2-S"),
        ("PET-MAD-1.0.0", None),
        ("PETMAD-1.0.0", None),
    ],
)
def test_resolve_pet_mad(name: str, expected: str | None) -> None:
    assert _resolve_canonical(name) == expected


def test_aliases() -> None:
    # Every alias must resolve to a canonical name in the registry.
    for v in MODEL_ALIASES.values():
        assert v in MODEL_REGISTRY, f"Alias target {v!r} is not a canonical model name."


def test_resolve_canonical_variants() -> None:
    """``_resolve_canonical`` recovers canonical names via direct, alias, and case-folded lookups."""
    from matcalc.utils import MODEL_ALIASES, MODEL_REGISTRY, _resolve_canonical

    canonical = next(iter(MODEL_REGISTRY))
    # 1. Direct canonical hit preserves the original casing.
    assert _resolve_canonical(canonical) == canonical
    # 2. Alias resolution returns the canonical target.
    alias_key = next(iter(MODEL_ALIASES))
    assert _resolve_canonical(alias_key) == MODEL_ALIASES[alias_key]
    # 2b. Mixed-case alias: the alias lookup lower-cases its input, so a mixed-case
    #     spelling of a real alias must resolve to the same canonical target.
    mixed_case_alias = "TensorNet"
    assert mixed_case_alias.lower() in MODEL_ALIASES
    assert _resolve_canonical(mixed_case_alias) == MODEL_ALIASES[mixed_case_alias.lower()]
    assert MODEL_ALIASES[mixed_case_alias.lower()] in MODEL_REGISTRY
    # 3. Case-folded canonical lookup recovers a mixed-case typo. Use an explicit
    #     literal (not ``next(iter(...)).lower()``) so a future registry reordering
    #     cannot silently route this onto the alias branch, and guard that the literal
    #     is not itself an alias key.
    case_folded_name = "tensornet-matpes-pbe-2025.2"
    assert case_folded_name not in MODEL_ALIASES
    assert _resolve_canonical(case_folded_name) == "TensorNet-MatPES-PBE-2025.2"
    # Unknown names resolve to None.
    assert _resolve_canonical("not-a-real-model-xyz") is None


def test_install_hint() -> None:
    """``_install_hint`` builds a friendly "pip install matcalc[<extra>]" message."""
    from matcalc.utils import _PROVIDER_INSTALL_EXTRA, _install_hint

    # MACE is reached through the 'mace_mp' provider key (mapping 'mace_mp' -> 'mace');
    # 'mace' itself is NOT a key, so this exercises the real mapping, not the fallback.
    msg = _install_hint("mace_mp")
    assert "pip install" in msg
    assert "matcalc[mace]" in msg

    # Unmapped provider: the .get fallback inserts the provider name itself.
    unmapped = "totally-unmapped-xyz"
    assert unmapped not in _PROVIDER_INSTALL_EXTRA
    hint = _install_hint(unmapped)
    assert "pip install" in hint
    assert "matcalc[totally-unmapped-xyz]" in hint
    assert "totally-unmapped-xyz" in hint


@pytest.mark.parametrize(
    ("old_name", "canonical"),
    [
        ("MACE-MP-0-small", "MACE-MPtrj-0a-small"),
        ("MACE-MP-0-medium", "MACE-MPtrj-0a-medium"),
        ("MACE-MP-0-large", "MACE-MPtrj-0a-large"),
        ("MACE-MP-0b-small", "MACE-MPtrj-0b-small"),
        ("MACE-MP-0b-medium", "MACE-MPtrj-0b-medium"),
        ("MACE-MP-0b2-small", "MACE-MPtrj-0b2-small"),
        ("MACE-MP-0b2-medium", "MACE-MPtrj-0b2-medium"),
        ("MACE-MP-0b2-large", "MACE-MPtrj-0b2-large"),
        ("MACE-MP-0b3-medium", "MACE-MPtrj-0b3-medium"),
        ("MACE-OMAT-0-small", "MACE-OMat24-0-small"),
        ("MACE-OMAT-0-medium", "MACE-OMat24-0-medium"),
        ("GRACE-1L-OAM", "GRACE-OAM-1L-small"),
        ("GRACE-2L-OAM", "GRACE-OAM-2L-small"),
        ("GRACE-2L-OMAT", "GRACE-OMat24-2L-small"),
        ("GRACE-2L-MPtrj", "GRACE-MPtrj-2L-r6"),
        ("ORB-v2", "ORB-MPA-v2"),
        ("ORB-d3-v2", "ORB-MPA-v2-D3"),
        ("ORB-d3-sm-v2", "ORB-MPA-v2-D3-sm"),
        ("ORB-d3-xs-v2", "ORB-MPA-v2-D3-xs"),
        ("SevenNet-0", "SevenNet-MPtrj-0"),
        ("SevenNet-l3i5", "SevenNet-MPtrj-l3i5"),
        ("SevenNet-MF-OMPA", "SevenNet-OAM-MF"),
        ("SevenNet-OMAT", "SevenNet-OMat24"),
    ],
)
def test_renamed_models(old_name: str, canonical: str) -> None:
    assert canonical in MODEL_REGISTRY
    assert old_name not in MODEL_REGISTRY
    for name in (old_name, old_name.lower(), old_name.upper(), canonical, canonical.lower()):
        assert _resolve_canonical(name) == canonical


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("grace", "GRACE-SMAX-OMat24-2L-medium"),
        ("tensorpotential", "GRACE-SMAX-OMat24-2L-medium"),
        ("orb", "ORB-OMat24-v3-conservative-inf"),
        ("sevennet", "SevenNet-Omni"),
        ("petmad", "PET-MAD-PBEsol-1.0.2-S"),
    ],
)
def test_short_aliases(alias: str, canonical: str) -> None:
    assert _resolve_canonical(alias) == canonical
