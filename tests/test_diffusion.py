"""Tests for DiffusionCalc."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from ase import units
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.constants import Boltzmann, elementary_charge

from matcalc import DiffusionCalc

if TYPE_CHECKING:
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


@pytest.mark.parametrize("as_atoms", [False, True])
@pytest.mark.parametrize(
    ("temperature", "timestep", "loginterval"),
    [(600, 1, 10), (1200, 0.5, 10), (600, 1, 20)],
)
def test_existing_trajectory(
    Li2O: Structure,
    matpes_calculator: PESCalculator,
    as_atoms: bool,
    temperature: float,
    timestep: float,
    loginterval: int,
) -> None:
    trajectory = []
    mobile_indices = [i for i, site in enumerate(Li2O) if site.specie.symbol == "Li"]
    for i in range(16):
        frame = Li2O.copy()
        frame.translate_sites(mobile_indices, [np.sqrt(0.01 * i), 0, 0], frac_coords=False)
        trajectory.append(AseAtomsAdaptor.get_atoms(frame) if as_atoms else frame)

    calc = DiffusionCalc(
        matpes_calculator,
        specie="Li",
        temperature=temperature,
        timestep=timestep,
        loginterval=loginterval,
        analyzer_kwargs={"smoothed": False},
    )
    upstream = {
        "final_structure": Li2O,
        "trajectory": trajectory,
        "potential_energy": -10.0,
        "_units": {"potential_energy": "eV"},
    }
    result = calc.calc(upstream)

    # MSD rises by 0.01 Å² per saved frame; D = slope / 6, with Å²/fs -> cm²/s.
    diffusivity = 0.01 / (timestep * loginterval * 60)
    number_density = len(mobile_indices) / (Li2O.volume * 1e-30)
    conductivity = number_density * elementary_charge**2 * (diffusivity * 1e-4) / (Boltzmann * temperature)
    assert result["diffusivity"] == pytest.approx(diffusivity)
    assert result["conductivity"] == pytest.approx(conductivity * 10)
    assert result["trajectory"] is trajectory
    assert result["final_structure"] is Li2O
    assert result["potential_energy"] == -10.0
    assert result["_units"] == {"potential_energy": "eV", "diffusivity": "cm^2/s", "conductivity": "mS/cm"}


def _hopping_trajectory(structure: Structure, specie: str) -> list[Structure]:
    """Frames in which every ``specie`` ion moves so that its MSD rises by 0.01 A^2 per frame."""
    mobile_indices = [i for i, site in enumerate(structure) if site.specie.symbol == specie]
    trajectory = []
    for i in range(16):
        frame = structure.copy()
        frame.translate_sites(mobile_indices, [np.sqrt(0.01 * i), 0, 0], frac_coords=False)
        trajectory.append(frame)
    return trajectory


def _nernst_einstein(structure: Structure, specie: str, charge: float, diffusivity: float) -> float:
    """Conductivity in mS/cm for a diffusivity in cm^2/s at 600 K."""
    number_density = sum(site.specie.symbol == specie for site in structure) / (structure.volume * 1e-30)
    return 10 * number_density * (charge * elementary_charge) ** 2 * (diffusivity * 1e-4) / (Boltzmann * 600)


@pytest.mark.parametrize(
    ("substitution", "oxi_states", "specie", "charge", "expected_charge"),
    [
        # pymatgen's DiffusionAnalyzer would use 10 (4d10) for Ag, 10 for Cu and 4 (2p4) for O.
        ({"Li": "Ag"}, None, "Ag", None, 1),
        ({"Li": "Cu"}, None, "Cu", None, 1),
        ({}, None, "O", None, -2),
        # Oxidation states of a decorated structure take precedence over the guess.
        ({}, {"Li": 1, "O": -1}, "O", None, -1),
        ({}, None, "Li", 2, 2),
    ],
)
def test_conductivity_charge(
    Li2O: Structure,
    matpes_calculator: PESCalculator,
    substitution: dict[str, str],
    oxi_states: dict[str, int] | None,
    specie: str,
    charge: float | None,
    expected_charge: float,
) -> None:
    structure = Li2O.copy()
    structure.remove_oxidation_states()
    structure.replace_species(substitution)
    if oxi_states:
        structure.add_oxidation_state_by_element(oxi_states)
    calc = DiffusionCalc(
        matpes_calculator,
        specie=specie,
        charge=charge,
        temperature=600,
        timestep=1,
        loginterval=10,
        analyzer_kwargs={"smoothed": False},
    )
    result = calc.calc({"final_structure": structure, "trajectory": _hopping_trajectory(structure, specie)})

    diffusivity = 0.01 / (10 * 60)
    assert result["diffusivity"] == pytest.approx(diffusivity)
    assert result["conductivity"] == pytest.approx(_nernst_einstein(structure, specie, expected_charge, diffusivity))


def test_conductivity_charge_fallback(Li2O: Structure, matpes_calculator: PESCalculator) -> None:
    """Without a charge-balanced guess (Li7O4), the common oxidation state is used with a warning."""
    structure = Li2O.to_conventional()
    structure.remove_oxidation_states()
    structure.remove_sites([next(i for i, site in enumerate(structure) if site.specie.symbol == "Li")])
    trajectory = _hopping_trajectory(structure, "Li")
    kwargs = {"temperature": 600, "timestep": 1, "loginterval": 10, "analyzer_kwargs": {"smoothed": False}}
    calc = DiffusionCalc(matpes_calculator, specie="Li", **kwargs)
    with pytest.warns(UserWarning, match="common oxidation state 1 of Li"):
        result = calc.calc({"final_structure": structure, "trajectory": trajectory})
    assert result["conductivity"] == pytest.approx(_nernst_einstein(structure, "Li", 1, result["diffusivity"]))

    calc = DiffusionCalc(matpes_calculator, specie="He", **kwargs)
    with pytest.raises(ValueError, match="charge of He"):
        calc.calc({"final_structure": structure, "trajectory": trajectory})


def test_diffusion_defaults(matpes_calculator: PESCalculator) -> None:
    calc = DiffusionCalc(matpes_calculator, specie="Li", temperature=1000, timestep=1.0, loginterval=10)
    md_calc = calc._build_md_calc()  # noqa: SLF001
    assert md_calc.ensemble == "nvt_nose_hoover"
    assert md_calc.steps == 100_000
    assert md_calc.equilibration_frames == 0
    assert md_calc.taut == pytest.approx(100 * units.fs)
    assert md_calc.seed is None
    assert md_calc.relax_structure
    assert md_calc.set_com_stationary
    assert not md_calc.set_zero_rotation
    assert md_calc.relax_calc_kwargs == {"relax_cell": False}


@pytest.mark.parametrize(
    ("relax_atoms_only", "relax_calc_kwargs", "relax_cell"),
    [(True, None, False), (False, None, True), (True, {"relax_cell": True, "fmax": 0.05}, True)],
)
def test_diffusion_relax_atoms_only(
    Li2O: Structure,
    matpes_calculator: PESCalculator,
    relax_atoms_only: bool,
    relax_calc_kwargs: dict | None,
    relax_cell: bool,
) -> None:
    calc = DiffusionCalc(
        matpes_calculator,
        specie="Li",
        temperature=1000,
        timestep=1.0,
        loginterval=10,
        relax_atoms_only=relax_atoms_only,
        relax_calc_kwargs=relax_calc_kwargs,
    )
    assert calc._build_md_calc().relax_calc_kwargs["relax_cell"] is relax_cell  # noqa: SLF001

    strained = Li2O.copy()
    strained.scale_lattice(Li2O.volume * 1.1)
    md_calc = calc._build_md_calc()  # noqa: SLF001
    md_calc.steps = 10
    result = md_calc.calc(strained)
    assert (abs(result["final_structure"].volume - strained.volume) > 1e-3) is relax_cell


def test_diffusion_md(Li2O: Structure, matpes_calculator: PESCalculator) -> None:
    calc = DiffusionCalc(
        matpes_calculator,
        specie="Li",
        temperature=600.5,
        timestep=1.0,
        loginterval=10,
        steps=200,
        equilibration_frames=5,
        seed=42,
        relax_structure=False,
        analyzer_kwargs={"smoothed": False},
    )
    result = calc.calc({"final_structure": Li2O, "label": "short MD", "_units": {"volume": "Å^3"}})

    # Reference: the same protocol run directly with ASE.
    atoms = AseAtomsAdaptor.get_atoms(Li2O)
    atoms.calc = matpes_calculator
    MaxwellBoltzmannDistribution(atoms, temperature_K=600.5, rng=np.random.RandomState(42))
    Stationary(atoms)
    md = NoseHooverChainNVT(atoms, timestep=units.fs, temperature_K=600.5, tdamp=100 * units.fs)
    reference = []
    md.attach(lambda: reference.append(atoms.copy()), interval=10)
    md.run(200)

    assert len(result["trajectory"]) == 16
    for actual, expected in zip(result["trajectory"], reference[5:], strict=True):
        np.testing.assert_allclose(actual.positions, expected.positions, atol=1e-5)
        np.testing.assert_allclose(actual.get_momenta(), expected.get_momenta(), atol=1e-4)
    np.testing.assert_allclose(result["final_structure"].cart_coords, reference[-1].positions, atol=1e-5)
    assert result["label"] == "short MD"
    assert result["_units"]["volume"] == "Å^3"
    assert result["_units"]["total_energy"] == "eV"
    assert np.isfinite(result["total_energy"])
    assert result["diffusivity"] >= 0
    assert result["conductivity"] >= 0


@pytest.mark.parametrize("frames", [None, 7])
def test_diffusion_md_protocol(matpes_calculator: PESCalculator, frames: int | None) -> None:
    """The MD settings of DiffusionCalc reach MDCalc, in MDCalc's units."""
    kwargs = {} if frames is None else {"frames": frames}
    calc = DiffusionCalc(
        matpes_calculator,
        specie="Li",
        temperature=600.5,
        timestep=2.0,
        loginterval=10,
        taut=150,
        ensemble="npt",
        seed=27,
        set_com_stationary=False,
        set_zero_rotation=False,
        **kwargs,
    )
    with pytest.warns(UserWarning, match="fixed lattice"):
        md_calc = calc._build_md_calc()  # noqa: SLF001

    assert md_calc.ensemble == "npt"
    assert md_calc.temperature == pytest.approx(600.5)
    assert md_calc.timestep == pytest.approx(2.0)
    assert md_calc.taut == pytest.approx(150 * units.fs)
    assert md_calc.seed == 27
    assert md_calc.loginterval == 10
    assert md_calc.frames == (frames if frames is not None else md_calc.steps)
    assert md_calc.equilibration_frames == 0
    assert md_calc.steps == 100_000
    assert md_calc.relax_structure
    assert not md_calc.set_com_stationary
    assert not md_calc.set_zero_rotation


def test_diffusion_fixed_cell(Li2O: Structure, matpes_calculator: PESCalculator) -> None:
    """Each run simulates its own structure at fixed cell, so the MD keeps that volume."""
    calc = DiffusionCalc(
        matpes_calculator,
        specie="Li",
        temperature=600.5,
        timestep=1.0,
        loginterval=10,
        steps=200,
        equilibration_frames=5,
        relax_structure=False,
        analyzer_kwargs={"smoothed": False},
    )
    expanded = Li2O.copy()
    expanded.scale_lattice(Li2O.volume * 1.2)
    for structure in (Li2O, expanded):
        result = calc.calc(structure)
        assert len(result["trajectory"]) == 16
        assert np.isfinite(result["diffusivity"])
        assert np.isfinite(result["conductivity"])
        assert np.isfinite(result["total_energy"])
        assert result["final_structure"].volume == pytest.approx(structure.volume)
        np.testing.assert_allclose(result["final_structure"].cart_coords, result["trajectory"][-1].positions)


@pytest.mark.parametrize("ensemble", ["npt", "npt_nose_hoover", "NPT_berendsen"])
def test_diffusion_npt_warning(matpes_calculator: PESCalculator, ensemble: str) -> None:
    calc = DiffusionCalc(
        matpes_calculator,
        specie="Li",
        temperature=1000,
        timestep=1.0,
        loginterval=10,
        ensemble=ensemble,  # type: ignore[arg-type]
    )
    with pytest.warns(UserWarning, match="fixed lattice"):
        calc._build_md_calc()  # noqa: SLF001
