"""Tests for PhononCalc class"""

from __future__ import annotations

import inspect
import logging
import os
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_allclose

from matcalc import PhononCalc, QHACalc
from matcalc._phonon import _mc_rattle, _phonon_rattle

if TYPE_CHECKING:
    from pathlib import Path

    from ase import Atoms
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


@pytest.mark.parametrize(
    ("force_const_file", "band_struct_file", "dos_file", "phonon_file"),
    [("", "", "", ""), ("fc", "bs.yaml", "dos.dat", "ph.yaml")],
)
def test_phonon_calc(
    Li2O: Structure,
    matpes_calculator: PESCalculator,
    tmp_path: Path,
    force_const_file: str,
    band_struct_file: str,
    dos_file: str,
    phonon_file: str,
) -> None:
    """Tests for PhononCalc class"""
    # Note that the fmax is probably too high. This is for testing purposes only.

    # change dir to tmp_path
    force_constants = tmp_path / force_const_file if force_const_file else False
    band_structure_yaml = tmp_path / band_struct_file if band_struct_file else False
    total_dos_dat = tmp_path / dos_file if dos_file else False
    phonon_yaml = tmp_path / phonon_file if phonon_file else False
    write_kwargs = {
        "write_force_constants": force_constants,
        "write_band_structure": band_structure_yaml,
        "write_total_dos": total_dos_dat,
        "write_phonon": phonon_yaml,
    }
    phonon_calc = PhononCalc(
        calculator=matpes_calculator,
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        fmax=0.1,
        t_step=50,
        t_max=1000,
        **write_kwargs,  # type: ignore[arg-type]
    )
    result = phonon_calc.calc(Li2O)

    # Test values at 100 K
    thermal_props = result["thermal_properties"]
    ind = thermal_props["temperatures"].tolist().index(300)
    assert thermal_props["heat_capacity"][ind] == pytest.approx(56.618323353498866, rel=1e-1)
    assert thermal_props["entropy"][ind] == pytest.approx(43.70960755706888, rel=1e-1)
    assert thermal_props["free_energy"][ind] == pytest.approx(15.620299826689202, rel=1e-1)
    units = result["_units"]
    assert units["thermal_properties.temperatures"] == "K"
    assert units["thermal_properties.heat_capacity"] == "J/(K*mol)"
    assert units["thermal_properties.entropy"] == "J/(K*mol)"
    assert units["thermal_properties.free_energy"] == "kJ/mol"
    assert units["frequencies"] == "THz"
    assert_allclose(
        result["final_structure"].lattice.abc,
        (3.291071792359756, 3.291071792359756, 3.291071792359756),
        rtol=1e-3,
    )
    assert_allclose(
        result["disp_supercells"][0].lattice.abc,
        (
            2 * result["final_structure"].lattice.abc[0],
            2 * result["final_structure"].lattice.abc[1],
            2 * result["final_structure"].lattice.abc[2],
        ),
    )
    pytest.approx(result["disp_supercells"][0].volume, result["disp_supercells"][-1].volume)

    results = list(phonon_calc.calc_many([Li2O, Li2O]))
    assert len(results) == 2

    ph_calc_params = inspect.signature(PhononCalc).parameters
    # get all keywords starting with write_ and their default values
    file_write_defaults = {key: val.default for key, val in ph_calc_params.items() if key.startswith("write_")}
    assert len(file_write_defaults) == 4

    for keyword, default_path in file_write_defaults.items():
        if instance_val := write_kwargs[keyword]:
            assert os.path.isfile(str(instance_val))
        elif not default_path and not instance_val:
            assert not os.path.isfile(default_path)


def test_phonon_calc_atoms(
    Si_atoms: Atoms,
    matpes_calculator: PESCalculator,
) -> None:
    """Tests for PhononCalc class"""
    phonon_calc = PhononCalc(
        calculator=matpes_calculator,
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        fmax=0.1,
        t_step=50,
        t_max=1000,
    )
    result = phonon_calc.calc(Si_atoms)

    # Test values at 100 K
    thermal_props = result["thermal_properties"]
    ind = thermal_props["temperatures"].tolist().index(300)
    assert thermal_props["heat_capacity"][ind] == pytest.approx(43.3138042001517, rel=1e-1)


def test_phonon_calc_lattice(
    Si_atoms: Atoms,
    matpes_calculator: PESCalculator,
) -> None:
    phonon_calc = PhononCalc(
        calculator=matpes_calculator,
        min_length=10.0,
        fmax=1,
        t_step=50,
        t_max=1000,
    )
    result = phonon_calc.calc(Si_atoms)
    assert_allclose(
        result["disp_supercells"][0].lattice.abc,
        (
            3 * result["final_structure"].lattice.abc[0],
            3 * result["final_structure"].lattice.abc[1],
            3 * result["final_structure"].lattice.abc[2],
        ),
    )
    pytest.approx(result["disp_supercells"][0].volume, result["disp_supercells"][-1].volume)


def test_phonon_calc_imaginary_freq_tol(
    Si_atoms: Atoms,
    matpes_calculator: PESCalculator,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that imaginary frequency tolerance check works correctly."""
    # Stable structure with tolerance should pass without error
    phonon_calc = PhononCalc(
        calculator=matpes_calculator,
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        fmax=0.1,
        imaginary_freq_tol=-0.1,
        on_imaginary_modes="error",
    )
    result = phonon_calc.calc(Si_atoms)
    assert "frequencies" in result
    assert np.min(result["frequencies"]) >= -0.1

    # Distort the structure to create imaginary modes, then check that
    # the tolerance check raises ValueError
    distorted_si_atoms = Si_atoms.copy()
    distorted_si_atoms.cell += 0.5
    phonon_calc = PhononCalc(
        calculator=matpes_calculator,
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        fmax=100.0,
        imaginary_freq_tol=-0.1,
        on_imaginary_modes="error",
    )
    with pytest.raises(ValueError, match="modes are imaginary"):
        phonon_calc.calc(distorted_si_atoms)

    # Distort the structure to create imaginary modes, then check that
    # the tolerance check logs a warning
    distorted_si_atoms = Si_atoms.copy()
    distorted_si_atoms.cell += 0.5
    phonon_calc = PhononCalc(
        calculator=matpes_calculator,
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        fmax=100.0,
        imaginary_freq_tol=-0.1,
        on_imaginary_modes="warn",
    )
    with caplog.at_level(logging.WARNING, logger="matcalc"):
        phonon_calc.calc(distorted_si_atoms)
    assert any("modes are imaginary" in r.message for r in caplog.records)

    # Distort the structure and use a very negative tol so no imaginary modes are flagged
    distorted_si_atoms = Si_atoms.copy()
    distorted_si_atoms.cell += 0.5
    phonon_calc = PhononCalc(
        calculator=matpes_calculator,
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        fmax=100.0,
        imaginary_freq_tol=-100.0,
        on_imaginary_modes="warn",
    )
    result = phonon_calc.calc(distorted_si_atoms)
    assert "frequencies" in result
    assert np.min(result["frequencies"]) < -0.1


def test_phonon_calc_fix_imaginary_attempts(
    Si_atoms: Atoms,
    matpes_calculator: PESCalculator,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test fix_imaginary_attempts parameter for resolving imaginary modes."""
    # Stable structure which needs no fixing — correction attempt should NOT be logged
    phonon_calc = PhononCalc(
        calculator=matpes_calculator,
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        fmax=0.1,
        t_step=50,
        t_max=1000,
        fix_imaginary_attempts=1,
        imaginary_freq_tol=-0.01,
        on_imaginary_modes="error",
    )
    with caplog.at_level(logging.INFO, logger="matcalc"):
        result = phonon_calc.calc(Si_atoms)
    assert np.min(result["frequencies"]) > -0.01
    assert not any("Imaginary mode correction attempt" in r.message for r in caplog.records)

    caplog.clear()

    # Distort the structure to create imaginary modes, then check that
    # the tolerance check raises ValueError
    distorted_si_atoms = Si_atoms.copy()
    distorted_si_atoms.cell += 0.5
    phonon_calc = PhononCalc(
        calculator=matpes_calculator,
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        fmax=100.0,
        imaginary_freq_tol=-0.1,
        on_imaginary_modes="error",
        fix_imaginary_attempts=1,
    )
    with caplog.at_level(logging.INFO, logger="matcalc"), pytest.raises(ValueError, match="modes are imaginary"):
        phonon_calc.calc(distorted_si_atoms)
    assert any("Imaginary mode correction attempt" in r.message for r in caplog.records)


@pytest.mark.parametrize(("rattle_method", "stdev_again"), [("random", 0.02), ("mc", 0.01), ("phonon", 0.01)])
def test_phonon_calc_rattle_seed(
    Si_atoms: Atoms,
    matpes_calculator: PESCalculator,
    monkeypatch: pytest.MonkeyPatch,
    rattle_method: str,
    stdev_again: float,
) -> None:
    """Each correction attempt rattles differently, a fixed seed reproduces the sequence, and the
    displacements scale with rattle_stdev."""
    distorted_si_atoms = Si_atoms.copy()
    distorted_si_atoms.cell += 0.5
    rattle = PhononCalc._rattle_structure  # noqa: SLF001
    displacements: list[np.ndarray] = []

    def recording_rattle(self: PhononCalc, structure_in: Structure, **kwargs: object) -> Structure:
        rattled = rattle(self, structure_in, **kwargs)  # type: ignore[arg-type]
        frac_disp = rattled.frac_coords - structure_in.frac_coords
        frac_disp -= np.round(frac_disp)  # undo wrapping back into the cell
        displacements.append(structure_in.lattice.get_cartesian_coords(frac_disp))
        return rattled

    monkeypatch.setattr(PhononCalc, "_rattle_structure", recording_rattle)

    def run(seed: int | None, rattle_stdev: float = 0.01) -> list[np.ndarray]:
        displacements.clear()
        PhononCalc(
            calculator=matpes_calculator,
            supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
            fmax=100.0,
            imaginary_freq_tol=-0.1,
            fix_imaginary_attempts=2,
            rattle_method=rattle_method,  # type: ignore[arg-type]
            rattle_stdev=rattle_stdev,
            seed=seed,
            write_phonon=False,
        ).calc(distorted_si_atoms)
        return list(displacements)

    first = run(seed=7, rattle_stdev=0.01)
    assert len(first) == 2
    assert not np.allclose(first[0], first[1])
    for disp, disp_again in zip(first, run(seed=7, rattle_stdev=stdev_again), strict=True):
        # rtol absorbs float32 noise in the model forces behind the "phonon" eigenvectors.
        assert_allclose(disp_again, stdev_again / 0.01 * disp, rtol=1e-3, atol=1e-6)
    assert not np.allclose(first[0], run(seed=8)[0])
    if rattle_method == "phonon":
        assert_allclose(np.sqrt(np.mean(np.square(first))), 0.01)


def test_phonon_calc_invalid_rattle_method(matpes_calculator: PESCalculator) -> None:
    with pytest.raises(ValueError, match="rattle_method must be"):
        PhononCalc(calculator=matpes_calculator, rattle_method="MC")  # type: ignore[arg-type]


def test_phonon_calc_rattle_temperature(Si: Structure, matpes_calculator: PESCalculator) -> None:
    phonon_calc = PhononCalc(calculator=matpes_calculator, supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)))
    phonon, _, _ = phonon_calc._run_phonopy(Si)  # noqa: SLF001
    disp_300 = _phonon_rattle(phonon, 0.01, np.random.RandomState(0), temperature=300.0)
    disp_1200 = _phonon_rattle(phonon, 0.5, np.random.RandomState(0), temperature=1200.0)
    assert_allclose(disp_1200, 2 * disp_300)  # amplitudes scale with sqrt(T), stdev is unused
    with pytest.raises(ValueError, match="rattle_temperature is only used"):
        PhononCalc(calculator=matpes_calculator, rattle_temperature=300.0)


def test_phonon_calc_mesh(Si: Structure, matpes_calculator: PESCalculator) -> None:
    def run(**kwargs: object) -> dict:
        return PhononCalc(
            calculator=matpes_calculator,
            supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
            relax_structure=False,
            t_step=100,
            t_max=300,
            write_phonon=False,
            **kwargs,  # type: ignore[arg-type]
        ).calc(Si)

    default = run()
    shifted = run(mesh=(4, 4, 4))
    centred = run(mesh=(4, 4, 4), is_gamma_center=True)
    from_length = run(mesh=20.0)

    assert default["phonon"].mesh.mesh_numbers.tolist() == run(mesh=100.0)["phonon"].mesh.mesh_numbers.tolist()
    assert shifted["phonon"].mesh.mesh_numbers.tolist() == [4, 4, 4]
    assert centred["phonon"].mesh.mesh_numbers.tolist() == [4, 4, 4]
    assert from_length["phonon"].mesh.mesh_numbers.tolist() != default["phonon"].mesh.mesh_numbers.tolist()
    # An even mesh contains Gamma only when it is Gamma-centred; a mesh from a length always does.
    assert not np.any(np.all(shifted["phonon"].mesh.qpoints == 0, axis=1))
    assert np.any(np.all(centred["phonon"].mesh.qpoints == 0, axis=1))
    assert np.any(np.all(from_length["phonon"].mesh.qpoints == 0, axis=1))

    cv_default = default["thermal_properties"]["heat_capacity"][-1]
    for result in (shifted, centred):
        cv = result["thermal_properties"]["heat_capacity"][-1]
        assert cv != cv_default
        assert cv == pytest.approx(cv_default, rel=2e-2)


def test_qha_calc_phonon_mesh(Si: Structure, matpes_calculator: PESCalculator) -> None:
    qha_calc = QHACalc(
        calculator=matpes_calculator,
        t_step=50,
        t_max=300,
        scale_factors=[0.98, 0.99, 1.00, 1.01, 1.02],
        fmax=0.1,
        store_ha_phonon=True,
        phonon_calc_kwargs={
            "supercell_matrix": ((2, 0, 0), (0, 2, 0), (0, 0, 2)),
            "mesh": (4, 4, 4),
            "is_gamma_center": True,
        },
    )
    result = qha_calc.calc(Si)
    for ha_result in result["ha"]:
        assert ha_result["phonon"].mesh.mesh_numbers.tolist() == [4, 4, 4]
        assert np.any(np.all(ha_result["phonon"].mesh.qpoints == 0, axis=1))


@pytest.mark.parametrize(("max_steps", "should_warn"), [(1, True), (500, False)])
def test_phonon_calc_unconverged_prerelax_warns(
    Si: Structure, matpes_calculator: PESCalculator, *, max_steps: int, should_warn: bool
) -> None:
    """A pre-relaxation that runs out of steps warns instead of passing silently (or raising)."""
    structure = Si.copy()
    structure.translate_sites([0], [0.05, 0, 0], frac_coords=False)
    phonon_calc = PhononCalc(
        calculator=matpes_calculator,
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        fmax=1e-3,
        max_steps=max_steps,
        write_phonon=False,
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = phonon_calc.calc(structure)
    messages = [str(w.message) for w in record if "pre-relaxation did not converge" in str(w.message)]
    assert bool(messages) is should_warn
    assert result["is_converged"] is not should_warn


def test_mc_rattle_min_distance(Si_atoms: Atoms) -> None:
    atoms = Si_atoms * (2, 2, 2)
    triu = np.triu_indices(len(atoms), 1)
    d_min = 0.8 * atoms.get_all_distances(mic=True)[triu].min()
    mc_atoms, random_atoms = atoms.copy(), atoms.copy()
    mc_atoms.positions += _mc_rattle(atoms, 0.1, d_min, np.random.RandomState(0))
    # random rattle of the same total size as 10 MC steps of 0.1 Å
    random_atoms.positions += np.random.RandomState(0).normal(0.0, 0.1 * np.sqrt(10), (len(atoms), 3))
    assert mc_atoms.get_all_distances(mic=True)[triu].min() > d_min - 0.2  # d_min - 2 * width
    assert random_atoms.get_all_distances(mic=True)[triu].min() < d_min - 0.2


def test_phonon_calc_imaginary_mode_message(matpes_calculator: PESCalculator) -> None:
    """The percentage of imaginary modes is reported with two decimals."""
    phonon_calc = PhononCalc(calculator=matpes_calculator, on_imaginary_modes="error")
    with pytest.raises(ValueError, match=r"1/3 \(33\.33%\) modes are imaginary"):
        phonon_calc._check_imaginary_modes(np.array([-1.0, 1.0, 2.0]))  # noqa: SLF001


def test_phonon_calc_batch_size(Si: Structure, matpes_calculator: PESCalculator) -> None:
    """Batching the displaced supercells does not change the phonons."""
    kwargs = {
        "supercell_matrix": ((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        "relax_structure": False,
        "write_phonon": False,
        "t_step": 100,
    }
    serial = PhononCalc(matpes_calculator, **kwargs).calc(Si)  # type: ignore[arg-type]
    batched = PhononCalc(matpes_calculator, batch_size=4, **kwargs).calc(Si)  # type: ignore[arg-type]
    assert_allclose(batched["phonon"].force_constants, serial["phonon"].force_constants, atol=1e-4)
    # The Gamma acoustic modes are ~3e-3 THz; omega = sqrt(eigenvalue) amplifies the float32 force noise there, so
    # they move by a few 1e-3 THz between the two paths. All other modes agree to ~1e-5 THz. A mode that flips sign
    # drops out of the thermal sums, which shifts the heat capacity by ~1e-3 J/(K*mol), i.e. ~6e-5 relative.
    assert_allclose(batched["frequencies"], serial["frequencies"], atol=1e-2)
    assert_allclose(
        batched["thermal_properties"]["heat_capacity"], serial["thermal_properties"]["heat_capacity"], rtol=1e-4
    )
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        PhononCalc(matpes_calculator, batch_size=0)
