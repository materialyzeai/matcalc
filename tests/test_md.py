"""Tests for MDCalc class"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pytest
from ase.io import read

from matcalc import MDCalc

if TYPE_CHECKING:
    from pathlib import Path

    from ase import Atoms
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


@pytest.fixture(scope="module", autouse=True)
def _set_seed() -> None:
    np.random.seed(42)  # noqa: NPY002


@pytest.mark.parametrize(
    ("ensemble", "expected_energy"),
    [
        ("nve", -10.441453988814853),
        ("nvt", -10.429515745917376),
        ("nvt_berendsen", -10.442275652025423),
        ("nvt_langevin", -10.3960898598168),
        ("nvt_andersen", -10.449911725549356),
        ("nvt_bussi", -10.394832352314676),
        ("npt_inhomogeneous", -10.444233085819286),
        ("npt_berendsen", -10.423091635481624),
        ("npt_nose_hoover", -10.39898962348729),
        ("npt_isotropic_mtk", -10.42449303041493),
        ("npt_mtk", -10.45577119876975),
    ],
)
def test_md_calc(
    Si: Structure,
    matpes_calculator: PESCalculator,
    ensemble: str,
    expected_energy: float,
) -> None:
    """Tests for MDCalc class"""
    # Note: fmax is set relatively high for testing purposes only.

    log_file = f"{ensemble}.log"
    traj_file = f"{ensemble}.traj"

    md_calc = MDCalc(
        calculator=matpes_calculator,
        ensemble=ensemble,
        temperature=300,
        steps=10,
        frames=5,
        compressibility_au=1,
        logfile=log_file,
        trajfile=traj_file,
    )
    initial_vol = Si.lattice.volume
    results = md_calc.calc(Si)
    assert results["final_structure"] != Si
    assert isinstance(results, dict)

    assert "trajectory" in results
    assert "potential_energy" in results
    assert "kinetic_energy" in results
    assert "total_energy" in results

    assert results["total_energy"] == pytest.approx(expected_energy, abs=1e-2)

    energies = np.array(results["trajectory"].total_energies)

    if ensemble != "nve":
        assert not np.allclose(energies - energies[0], 0, atol=1e-9), f"Energies are too close for {ensemble}"

    if ensemble.startswith("nvt"):
        # There should be no volume change for NVT simulations.
        assert np.linalg.det(results["trajectory"].cells[-1]) == pytest.approx(initial_vol, rel=1e-2)

    assert len(results["trajectory"]) == 5

    # Verify that the log file and trajectory file have been created
    assert os.path.isfile(str(log_file))
    assert os.path.isfile(str(traj_file))


def test_md_atoms(
    Si_atoms: Atoms,
    matpes_calculator: PESCalculator,
) -> None:
    """Tests for MDCalc class"""
    # Note: fmax is set relatively high for testing purposes only.
    md_calc = MDCalc(
        calculator=matpes_calculator,
        temperature=300,
        taut=0.1,
        taup=0.1,
        steps=10,
        frames=5,
        compressibility_au=1,
    )
    results = md_calc.calc(Si_atoms)

    assert isinstance(results, dict)


def test_md_relax_cell(
    Si: Structure,
    matpes_calculator: PESCalculator,
) -> None:
    """Tests for MDCalc class with cell relaxation"""
    # Note: fmax is set relatively high for testing purposes only.

    # default behavior (relax_cell = False)
    md_calc = MDCalc(
        calculator=matpes_calculator,
        ensemble="npt_mtk",
        temperature=300,
        steps=1,
        compressibility_au=1,
    )
    initial_vol = Si.lattice.volume
    results = md_calc.calc(Si)
    volume_after_relax = np.linalg.det(results["trajectory"].cells[0])
    assert volume_after_relax == pytest.approx(initial_vol, rel=1e-4)

    md_calc = MDCalc(
        calculator=matpes_calculator,
        ensemble="npt_mtk",
        temperature=300,
        steps=1,
        compressibility_au=1,
        relax_calc_kwargs={"relax_cell": True},
    )
    initial_vol = Si.lattice.volume
    results = md_calc.calc(Si)
    volume_after_relax = np.linalg.det(results["trajectory"].cells[0])
    assert abs(volume_after_relax - initial_vol) > 0.1


def test_stationary(Si_atoms: Atoms, matpes_calculator: PESCalculator, tmp_path: Path) -> None:
    """Tests for MDCalc class with and without stationary COM"""
    starting_com = Si_atoms.get_center_of_mass()

    # Test with stationary COM
    md_calc = MDCalc(
        calculator=matpes_calculator,
        ensemble="npt_mtk",
        temperature=300,
        steps=10,
        compressibility_au=1,
        set_com_stationary=True,
        trajfile=tmp_path / "test.traj",
    )
    md_calc.calc(Si_atoms)
    final_com = read(tmp_path / "test.traj", index=":")[-1].get_center_of_mass()
    assert final_com == pytest.approx(starting_com, abs=1e-2)

    # Test with non-stationary COM
    # Note: MTKNPT does not zero out COM momentum
    md_calc = MDCalc(
        calculator=matpes_calculator,
        ensemble="npt_mtk",
        temperature=300,
        steps=10,
        compressibility_au=1,
        set_com_stationary=False,
        trajfile=tmp_path / "test.traj",
    )
    md_calc.calc(Si_atoms)
    final_com = read(tmp_path / "test.traj", index=":")[-1].get_center_of_mass()
    assert final_com != pytest.approx(starting_com, abs=1e-2)


def test_rotation(Si_atoms: Atoms, matpes_calculator: PESCalculator, tmp_path: Path) -> None:
    """Tests for MDCalc class with and without zero rotation"""
    Si_atoms.set_momenta(Si_atoms.get_momenta() + 10.0)  # magnify the effect
    md_calc_kwargs = {
        "calculator": matpes_calculator,
        "temperature": 300,
        "steps": 1,
        "compressibility_au": 1,
    }
    md_calc_zero_rotation = MDCalc(
        trajfile=tmp_path / "test_zero_rotation.traj",
        set_zero_rotation=True,
        **md_calc_kwargs,
    )
    md_calc_rotation = MDCalc(
        trajfile=tmp_path / "test_rotation.traj",
        set_zero_rotation=False,
        **md_calc_kwargs,
    )
    md_calc_zero_rotation.calc(Si_atoms)
    md_calc_rotation.calc(Si_atoms)
    momenta_with_zero_rotation = read(tmp_path / "test_zero_rotation.traj", index=":")[0].get_momenta()
    momenta_with_rotation = read(tmp_path / "test_rotation.traj", index=":")[0].get_momenta()
    assert momenta_with_zero_rotation != pytest.approx(momenta_with_rotation, abs=0.1)


def test_invalid_ensemble(Si: Structure, matpes_calculator: PESCalculator) -> None:
    with pytest.raises(
        ValueError,
        match=r"The specified ensemble is not supported.*",
    ):
        MDCalc(
            calculator=matpes_calculator,
            ensemble="you_know_who",
            temperature=300,
            taut=0.1,
            taup=0.1,
            compressibility_au=10,
            steps=10,
            logfile=None,
            trajfile=None,
        ).calc(Si)
