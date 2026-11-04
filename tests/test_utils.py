from __future__ import annotations

import re
import shutil
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from matcalc import RelaxCalc, backend
from matcalc.utils import (
    _REMOVED_MODELS,
    MODEL_ALIASES,
    MODEL_REGISTRY,
    UNIVERSAL_CALCULATORS,
    PESCalculator,
    _resolve_canonical,
)

DIR = Path(__file__).parent.absolute()

if TYPE_CHECKING:
    from pymatgen.core import Structure

UNIVERSAL_CALCULATORS = [calc for calc in UNIVERSAL_CALCULATORS if "CHGNet" not in calc.name]


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
    if backend.is_lammps_backend() and (pkg_name != "matgl" or name.startswith(("M3GNet", "QET"))):
        # MatGL exports TensorNet to LAMMPS; M3GNet only with use_smooth=True, and QET's charge heads not at all.
        pytest.skip(f"Not supported by the LAMMPS backend: {name} has no LAMMPS representation.")

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


@pytest.mark.skipif(not find_spec("maml"), reason="maml is not installed")
def test_pescalculator_stress_convention(monkeypatch: pytest.MonkeyPatch) -> None:
    """maml returns the LAMMPS pressure tensor (GPa, xx yy zz xy xz yz); ASE wants -p in Voigt order."""
    import maml.apps.pes
    import numpy as np

    from matcalc.units import eVA3ToGPa

    lammps_pressure = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # xx yy zz xy xz yz

    class FakeEFS:  # stands in for maml's LAMMPS-backed EnergyForceStress
        def __init__(self, ff_settings: object) -> None:
            pass

        def calculate(self, structures: list) -> list:
            return [(-1.0, np.zeros((len(structures[0]), 3)), lammps_pressure)]

    monkeypatch.setattr(maml.apps.pes, "EnergyForceStress", FakeEFS)
    atoms = Atoms("Cu", positions=[(0, 0, 0)], cell=[2.5] * 3, pbc=True)
    expected = -np.array([1.0, 2.0, 3.0, 6.0, 5.0, 4.0])  # xx yy zz yz xz xy
    for unit, factor in (("GPa", 1.0), ("eV/A3", 1 / eVA3ToGPa)):
        atoms.calc = PESCalculator(potential=None, stress_unit=unit)  # type: ignore[arg-type]
        assert atoms.get_stress() == pytest.approx(expected * factor)
        # Full tensor is symmetric with xy (index 4 in LAMMPS order) in the [0, 1] slot.
        assert atoms.get_stress(voigt=False)[0, 1] == pytest.approx(-4.0 * factor)
    # Default must be ASE's eV/A^3: RelaxCalc cell filters and ElasticityCalc assume it.
    assert PESCalculator(potential=None).stress_weight == pytest.approx(1 / eVA3ToGPa)  # type: ignore[arg-type]


def test_load_universal_path(tmp_path: Path) -> None:
    """A ``Path`` (not just a ``str``) to a saved MatGL model directory must load, not be returned as-is."""
    import matgl

    model_dir = tmp_path / "finetuned"
    matgl.load_model("TensorNet-PES-MatPES-PBE-2025.2").save(model_dir)
    assert isinstance(PESCalculator.load_universal(model_dir), Calculator)
    with pytest.raises(ValueError, match="Unrecognized"):
        PESCalculator.load_universal(tmp_path / "missing")


def test_import_does_not_load_matgl() -> None:
    """``import matcalc`` must not import MatGL or query the HF Hub for its model list."""
    code = "import sys, matcalc; assert 'matgl' not in sys.modules, 'matgl imported eagerly'"
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603


def test_lazy_matgl_models(monkeypatch: pytest.MonkeyPatch) -> None:
    import matgl

    from matcalc.utils import _LazyMatGLModels

    calls = []

    def fake_list() -> list[str]:
        calls.append(1)
        return ["Foo-PES-Bar", "Foo-Bandgap"]

    monkeypatch.setattr(matgl, "get_available_pretrained_models", fake_list)
    models = _LazyMatGLModels()
    assert not calls
    assert "Foo-PES-Bar" in models
    assert "Foo-Bandgap" not in models
    assert set(models) == {"Foo-PES-Bar"}
    assert len(calls) == 1


def test_aliases() -> None:
    # Every alias must resolve to a canonical name in the registry.
    for alias, target in MODEL_ALIASES.items():
        assert target in MODEL_REGISTRY, f"Alias target {target!r} is not a canonical model name."
        # Lookups lower-case the input, so a non-lower-case alias key would be unreachable.
        assert alias == alias.lower(), f"Alias key {alias!r} must be lower case."
        assert _resolve_canonical(alias.upper()) == target
    # An alias must not shadow a different canonical name's case-folded spelling.
    for canonical in MODEL_REGISTRY:
        assert _resolve_canonical(canonical.lower()) == canonical
    # Removed names must not silently resolve to something else.
    for removed in _REMOVED_MODELS:
        assert _resolve_canonical(removed) is None


def test_fp_loader_workflow_matches_registry() -> None:
    """``.github/workflows/test-fp-loaders.yml`` must only smoke-test canonical names of its own provider."""
    import json

    from ruamel.yaml import YAML

    from matcalc.utils import _PROVIDER_INSTALL_EXTRA

    workflow = YAML(typ="safe").load(DIR.parent / ".github" / "workflows" / "test-fp-loaders.yml")
    extra_to_provider = {extra: provider for provider, extra in _PROVIDER_INSTALL_EXTRA.items()}
    for entry in workflow["jobs"]["test-fp-loaders"]["strategy"]["matrix"]["include"]:
        provider = extra_to_provider[entry["extras"]]
        for name in json.loads(entry["models"]):
            assert name in MODEL_REGISTRY, f"{entry['name']}: {name!r} is not a canonical model name."
            assert MODEL_REGISTRY[name]["provider"] == provider, f"{name!r} is not a {provider!r} model."


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
        ("MACE-MatPES-PBE-0", "MACE-OMat24-ft-MatPES-PBE-0"),
        ("MACE-MatPES-r2SCAN-0", "MACE-OMat24-ft-MatPES-r2SCAN-0"),
        ("GRACE-1L-OAM", "GRACE-OMat24-ft-MPA-1L-small"),
        ("GRACE-2L-OAM", "GRACE-OMat24-ft-MPA-2L-small"),
        ("GRACE-2L-OMAT", "GRACE-OMat24-2L-small"),
        ("GRACE-2L-MPtrj", "GRACE-MPtrj-2L-r6"),
        ("ORB-v2", "ORB-MPtrj-Alex-v2"),
        ("ORB-d3-v2", "ORB-MPtrj-Alex-v2-D3"),
        ("ORB-d3-sm-v2", "ORB-MPtrj-Alex-v2-D3-sm"),
        ("ORB-d3-xs-v2", "ORB-MPtrj-Alex-v2-D3-xs"),
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


@pytest.mark.parametrize("name", ["M3GNet-MatPES-PBE-2025.1", "PETMAD-1.0.0", "DPA3-LAM-2025.3.14"])
def test_removed_models(name: str) -> None:
    with pytest.raises(ValueError, match="was removed"):
        PESCalculator.load_universal(name)


# Every canonical name and alias in the v0.6.0 release.
V060_MODEL_NAMES = (
    "TensorNet-MatPES-PBE-2025.2",
    "TensorNet-MatPES-r2SCAN-2025.2",
    "M3GNet-MatPES-PBE-2025.1",
    "CHGNet-MatPES-PBE-1M-2026.9",
    "CHGNet-MatPES-r2SCAN-1M-2026.9",
    "CHGNet-MatPES-PBE-2025.2.10",
    "CHGNet-MatPES-r2SCAN-2025.2.10",
    "MACE-MP-0-small",
    "MACE-MP-0-medium",
    "MACE-MP-0-large",
    "MACE-MP-0b-small",
    "MACE-MP-0b-medium",
    "MACE-MP-0b2-small",
    "MACE-MP-0b2-medium",
    "MACE-MP-0b2-large",
    "MACE-MP-0b3-medium",
    "MACE-MPA-0-medium",
    "MACE-OMAT-0-small",
    "MACE-OMAT-0-medium",
    "MACE-MatPES-PBE-0",
    "MACE-MatPES-r2SCAN-0",
    "SevenNet-0",
    "SevenNet-l3i5",
    "SevenNet-MF-OMPA",
    "SevenNet-OMAT",
    "GRACE-1L-OAM",
    "GRACE-2L-OAM",
    "GRACE-2L-OMAT",
    "GRACE-2L-MPtrj",
    "ORB-v2",
    "ORB-d3-v2",
    "ORB-d3-sm-v2",
    "ORB-d3-xs-v2",
    "MatterSim-v1.0.0-1M",
    "MatterSim-v1.0.0-5M",
    "UMA-S-1.2.1",
    "UMA-S-1.2",
    "UMA-S-1.1",
    "UMA-M-1.1",
    "PETMAD-1.0.0",
    "DPA3-LAM-2025.3.14",
    "tensornet",
    "m3gnet",
    "chgnet",
    "pbe",
    "r2scan",
    "mace",
    "sevennet",
    "grace",
    "tensorpotential",
    "orb",
    "mattersim",
    "fairchem",
    "uma",
    "petmad",
    "deepmd",
    "tensornet-pes-matpes-pbe-2025.2",
    "tensornet-pes-matpes-r2scan-2025.2",
    "m3gnet-pes-matpes-pbe-2025.1",
    "chgnet-pes-matpes-pbe-1m-2026.9",
    "chgnet-pes-matpes-r2scan-1m-2026.9",
    "chgnet-pes-matpes-pbe-2025.2.10",
    "chgnet-pes-matpes-r2scan-2025.2.10",
    "chgnet-matpes-pbe-2025.2.10-2.7m-pes",
)


def test_released_model_names() -> None:
    """Every v0.6.0 name resolves or is reported as removed."""
    unresolved = [
        name for name in V060_MODEL_NAMES if _resolve_canonical(name) is None and name.lower() not in _REMOVED_MODELS
    ]
    assert not unresolved


def test_load_universal_matgl_model_dir(tmp_path: Path) -> None:
    import matgl

    model_dir = tmp_path / "finetuned"
    matgl.load_model("TensorNet-PES-MatPES-PBE-2025.2").save(model_dir)
    assert isinstance(PESCalculator.load_universal(str(model_dir)), Calculator)
    with pytest.raises(ValueError, match="Unrecognized"):
        PESCalculator.load_universal(str(tmp_path))
