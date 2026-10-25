"""Tests for DiffusionCalc."""

from __future__ import annotations

from shutil import which
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
    from pathlib import Path

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
def test_diffusion_lammps_protocol(Li2O: Structure, tmp_path: Path, frames: int | None) -> None:
    kwargs = {} if frames is None else {"frames": frames}
    calc = DiffusionCalc(
        "TensorNet",
        backend="lammps",
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
    md_calc = calc._build_md_calc()  # noqa: SLF001
    md_calc.write_inputs(Li2O, directory=tmp_path)
    script = (tmp_path / "in.md").read_text()
    commands = {
        line.split()[0]: line.split()[1:] for line in script.splitlines() if line.strip() and not line.startswith("#")
    }

    assert md_calc.frames == frames
    assert md_calc.equilibration_frames == 0
    assert md_calc.steps == 100_000
    assert md_calc.relax_structure
    assert commands["timestep"] == ["0.002"]
    assert commands["fix"][:7] == ["1", "all", "npt", "temp", "600.5", "600.5", "0.15"]
    assert commands["velocity"] == ["all", "create", "600.5", "27", "mom", "no", "rot", "no", "dist", "gaussian"]
    assert commands["pair_coeff"][2:4] == ["matgl", "TensorNet-PES-MatPES-PBE-2025.2"]


def test_diffusion_backend_errors(matpes_calculator: PESCalculator) -> None:
    with pytest.raises(ValueError, match="backend"):
        DiffusionCalc(matpes_calculator, specie="Li", temperature=1000, timestep=1.0, loginterval=10, backend="unknown")
    calc = DiffusionCalc(
        matpes_calculator, specie="Li", temperature=1000, timestep=1.0, loginterval=10, backend="lammps"
    )
    with pytest.raises(TypeError, match="model name"):
        calc._build_md_calc()  # noqa: SLF001


def test_diffusion_lammps_md(Li2O: Structure) -> None:
    if not which("lmp"):
        pytest.skip("lammps not available")
    calc = DiffusionCalc(
        "TensorNet",
        backend="lammps",
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
    # Each run writes inputs for its own structure, so the fixed-cell MD keeps that volume.
    for structure in (Li2O, expanded):
        result = calc.calc(structure)
        assert len(result["trajectory"]) == 16
        assert np.isfinite(result["diffusivity"])
        assert np.isfinite(result["conductivity"])
        assert np.isfinite(result["total_energy"])
        assert result["final_structure"].volume == pytest.approx(structure.volume)
        np.testing.assert_allclose(result["final_structure"].cart_coords, result["trajectory"][-1].positions)
