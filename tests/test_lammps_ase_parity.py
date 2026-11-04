"""Parity of the LAMMPS simulation backend with the ASE backend, and LAMMPS-specific behavior.

The module runs only with ``MATCALC_BACKEND=LAMMPS`` (the rest of the suite runs on both backends); each test switches
to the ASE backend for its reference values as needed. Tests that run LAMMPS are skipped unless the ``lammps`` Python
module can start LAMMPS. The MatGL tests also need LAMMPS built with MatGL's ML-MATGL package (``pair_style matgl``);
the SNAP tests need maml and an ``lmp`` executable, which maml requires to construct SNAP potentials. Set
``MATCALC_LAMMPS_ARGS`` (e.g. ``-k on t 2 -sf kk -pk kokkos newton on neigh half``) to run the same tests through the
Kokkos styles.
"""

from __future__ import annotations

import functools
import itertools
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.constraints import FixAtoms
from ase.io import read
from ase.optimize import BFGS

from matcalc import ElasticityCalc, EOSCalc, LAMMPSCalculator, MDCalc, PhononCalc, RelaxCalc, backend
from matcalc.backend import get_lammps_potential, run_ase, run_lammps, run_lammps_md, to_backend_calculator
from matcalc.backend._lammps_engine import get_engine, lammps_cell
from matcalc.utils import PESCalculator

if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Structure

    from matcalc.utils import PESCalculator as PESCalculatorType

DIR = Path(__file__).parent.absolute()

# Argon Lennard-Jones parameters; ASE's LennardJones shifts the pair energy to zero at rc like pair_modify shift yes.
LJ_EPS, LJ_SIGMA, LJ_RC = 0.0104, 3.4, 8.0


@functools.cache
def _lammps_styles() -> frozenset[str] | None:
    """Styles of the importable LAMMPS build used by these tests, or None if LAMMPS cannot be started."""
    try:
        from lammps import lammps

        lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    except Exception:  # noqa: BLE001
        return None
    try:
        pairs = {style for style in ("lj/cut", "snap", "matgl") if lmp.has_style("pair", style)}
        return frozenset(pairs | ({"fix temp/csvr"} if lmp.has_style("fix", "temp/csvr") else set()))
    finally:
        lmp.close()


def _has_style(style: str) -> bool:
    styles = _lammps_styles()
    return styles is not None and style in styles


pytestmark = pytest.mark.skipif(
    not backend.is_lammps_backend(), reason="LAMMPS backend tests run with MATCALC_BACKEND=LAMMPS"
)

# CI sets this in its LAMMPS leg so that a missing LAMMPS feature fails the tests instead of skipping them.
REQUIRE_LAMMPS = os.environ.get("MATCALC_TEST_REQUIRE_LAMMPS") == "1"
needs_lammps = pytest.mark.skipif(
    not (REQUIRE_LAMMPS or _has_style("lj/cut")), reason="LAMMPS Python module not available"
)
needs_snap = pytest.mark.skipif(
    not (REQUIRE_LAMMPS or (_has_style("snap") and shutil.which("lmp"))),
    reason="needs LAMMPS with ML-SNAP and an lmp executable (required by maml)",
)
needs_matgl = pytest.mark.skipif(
    not (REQUIRE_LAMMPS or _has_style("matgl")), reason="LAMMPS not built with MatGL's ML-MATGL"
)


@pytest.fixture
def lammps_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend, "SIMULATION_BACKEND", "LAMMPS")


def _lj_lammps() -> LAMMPSCalculator:
    return LAMMPSCalculator(
        [f"pair_style lj/cut {LJ_RC}", f"pair_coeff * * {LJ_EPS} {LJ_SIGMA}", "pair_modify shift yes"]
    )


def _lj_ase() -> LennardJones:
    return LennardJones(epsilon=LJ_EPS, sigma=LJ_SIGMA, rc=LJ_RC, smooth=False)


def _skewed(atoms: Atoms, *, left_handed: bool = False, seed: int = 0, shear: float = 1.0) -> Atoms:
    """Strain, rotate and rattle a cell so that it needs rotation and tilt reduction in LAMMPS."""
    rng = np.random.default_rng(seed)
    atoms = atoms.copy()
    strain = np.eye(3) + np.array([[0.02, 0.6 * shear, -0.4 * shear], [0.0, -0.01, 0.7 * shear], [0.0, 0.0, 0.03]])
    atoms.set_cell(atoms.cell @ strain, scale_atoms=True)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation *= np.sign(np.linalg.det(rotation))
    atoms.set_cell(atoms.cell @ rotation, scale_atoms=True)
    if left_handed:
        atoms.set_cell(atoms.cell[[1, 0, 2]], scale_atoms=False)
    atoms.rattle(0.05, seed=seed)
    return atoms


@pytest.fixture
def ar_cell() -> Atoms:
    return bulk("Ar", "fcc", a=5.3, cubic=True).repeat((2, 1, 1))


@pytest.fixture
def ar_supercell() -> Atoms:
    atoms = bulk("Ar", "fcc", a=5.4, cubic=True).repeat((2, 2, 2))
    atoms.rattle(0.02, seed=0)
    return atoms


# --- Pure-Python pieces -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("left_handed", [False, True])
def test_lammps_cell(left_handed: bool) -> None:
    cell = np.array(_skewed(bulk("Cu", "fcc", a=3.6, cubic=True), left_handed=left_handed).cell)
    rot, lower = lammps_cell(cell)

    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
    assert np.all(lower[np.triu_indices(3, 1)] == 0)
    assert np.all(np.diag(lower) > 0)
    lx, ly = lower[0, 0], lower[1, 1]
    assert abs(lower[1, 0]) <= lx / 2 + 1e-9
    assert abs(lower[2, 0]) <= lx / 2 + 1e-9
    assert abs(lower[2, 1]) <= ly / 2 + 1e-9
    # The reduced cell spans the same lattice as the rotated input cell.
    transform = np.linalg.solve((cell @ rot).T, lower.T).T
    np.testing.assert_allclose(transform, np.round(transform), atol=1e-9)
    assert abs(round(np.linalg.det(transform))) == 1


def test_get_lammps_potential_unsupported() -> None:
    with pytest.raises(TypeError, match="EMT has no LAMMPS representation"):
        get_lammps_potential(EMT())


def test_get_lammps_potential_matgl(matpes_calculator: PESCalculatorType) -> None:
    potential = get_lammps_potential(matpes_calculator)
    assert potential is get_lammps_potential(matpes_calculator)  # exported once per process
    assert potential.commands[0] == "pair_style matgl"
    assert potential.commands[1].endswith("{symbols}")
    model_file = potential.commands[1].split('"')[1]
    assert Path(model_file).stat().st_size > 0
    assert potential.pre_commands == ("newton on",)
    assert potential.elements is not None
    assert {"Li", "O", "Si"} <= set(potential.elements)


@pytest.mark.usefixtures("lammps_backend")
def test_prop_calc_calculator_lammps_backend(matpes_calculator: PESCalculatorType) -> None:
    """Under the LAMMPS backend, PropCalc.calculator is the LAMMPSCalculator, resolved once at construction."""
    calc = RelaxCalc(matpes_calculator)
    assert isinstance(calc.calculator, LAMMPSCalculator)
    assert calc.calculator.source is matpes_calculator
    assert calc.calculator.potential is get_lammps_potential(matpes_calculator)
    assert to_backend_calculator(calc.calculator) is calc.calculator
    # Downstream PropCalcs built from it reuse the same calculator.
    assert ElasticityCalc(calc.calculator).calculator is calc.calculator

    lj = _lj_lammps()
    assert RelaxCalc(lj).calculator is lj
    assert lj.source is None
    with pytest.raises(TypeError, match="EMT has no LAMMPS representation"):
        RelaxCalc(EMT())


@pytest.mark.usefixtures("lammps_backend")
def test_prop_calc_model_name_lammps_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model name becomes a LAMMPSCalculator directly (no ASE calculator), exported once per model."""

    def no_ase_calculator(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("an ASE calculator was built")

    monkeypatch.setattr(PESCalculator, "load_universal", staticmethod(no_ase_calculator))
    calc = RelaxCalc("TensorNet-MatPES-PBE-2025.2").calculator
    assert isinstance(calc, LAMMPSCalculator)
    assert calc.source is None
    assert calc.potential.commands[0] == "pair_style matgl"
    # Case variants and aliases of the same model reuse the export.
    assert ElasticityCalc("tensornet-matpes-pbe-2025.2").calculator.potential is calc.potential
    # Other providers are rejected before their model (or package) is loaded.
    with pytest.raises(TypeError, match=r"'MACE-MPtrj-0a-small' \(mace_mp\) has no LAMMPS representation"):
        RelaxCalc("MACE-MPtrj-0a-small")
    with pytest.raises(ValueError, match="Unrecognized"):
        RelaxCalc("NotAModel-1.0")


def test_prop_calc_calculator_ase_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend, "SIMULATION_BACKEND", "ASE")
    emt = EMT()
    assert RelaxCalc(emt).calculator is emt
    assert to_backend_calculator(emt) is emt


# --- Static calculations ----------------------------------------------------------------------------------------------


@needs_lammps
@pytest.mark.parametrize("left_handed", [False, True])
def test_lj_parity(ar_cell: Atoms, left_handed: bool) -> None:
    atoms = _skewed(ar_cell, left_handed=left_handed)
    ref = run_ase(atoms, _lj_ase())
    res = run_lammps(atoms, _lj_lammps())

    assert res.converged is None
    assert res.potential_energy == pytest.approx(ref.potential_energy, abs=1e-8)
    np.testing.assert_allclose(res.forces, ref.forces, atol=1e-8)
    np.testing.assert_allclose(res.stress, ref.stress, atol=1e-9)
    assert np.abs(ref.forces).max() > 1e-3  # non-trivial check
    assert np.abs(ref.stress).max() > 1e-4


@needs_lammps
def test_engine_scoped_failed_setup(ar_cell: Atoms) -> None:
    """After a setup command fails, cleanup undoes what was set up and the cached instance stays usable."""
    engine = get_engine(_lj_lammps().potential, ar_cell)
    engine.load(ar_cell)
    with (
        pytest.raises(Exception, match="no_such_style"),
        engine.scoped(
            ["fix matcalc_a all nve", "fix matcalc_b all no_such_style"], ["unfix matcalc_b", "unfix matcalc_a"]
        ),
    ):
        pass
    with engine.scoped(["fix matcalc_a all nve"], ["unfix matcalc_a"]):  # fix matcalc_a was removed
        engine.lmp.command("run 1")
    assert engine.compute(ar_cell)[0] < 0


@needs_lammps
def test_lammps_calculator_with_ase_backend(ar_cell: Atoms) -> None:
    """LAMMPSCalculator is a regular ASE calculator: engines are reused across cell sizes."""
    for atoms in (_skewed(ar_cell), _skewed(ar_cell.repeat((1, 2, 1)), seed=1)):
        ref = run_ase(atoms, _lj_ase())
        res = run_ase(atoms, _lj_lammps())
        assert res.potential_energy == pytest.approx(ref.potential_energy, abs=1e-8)
        np.testing.assert_allclose(res.forces, ref.forces, atol=1e-8)
        np.testing.assert_allclose(res.stress, ref.stress, atol=1e-9)


@needs_lammps
def test_type_mapping(ar_cell: Atoms) -> None:
    """Only Kr-Kr pairs interact, so the energy equals that of the Kr sublattice alone."""
    atoms = _skewed(ar_cell)
    atoms.symbols[::3] = "Kr"
    calc = LAMMPSCalculator(
        [
            f"pair_style lj/cut {LJ_RC}",
            f"pair_coeff * * 0.0 {LJ_SIGMA}",
            f"pair_coeff 2 2 {LJ_EPS} {LJ_SIGMA}",
            "pair_modify shift yes",
        ],
        elements=["Ar", "Kr"],
    )
    kr = atoms[[i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Kr"]]
    res = run_lammps(atoms, calc)
    ref = run_ase(kr, _lj_ase())
    assert res.potential_energy == pytest.approx(ref.potential_energy, abs=1e-8)
    np.testing.assert_allclose(res.forces[::3], ref.forces, atol=1e-8)
    with pytest.raises(ValueError, match=r"Elements \['Xe'\] are not supported"):
        run_lammps(bulk("Xe", "fcc", a=6.2), calc)


@needs_lammps
def test_constraints_applied(ar_cell: Atoms) -> None:
    atoms = _skewed(ar_cell)
    atoms.set_constraint(FixAtoms(indices=[0, 2]))
    ref = run_ase(atoms, _lj_ase())
    res = run_lammps(atoms, _lj_lammps())
    np.testing.assert_allclose(res.forces[[0, 2]], 0.0)
    np.testing.assert_allclose(res.forces, ref.forces, atol=1e-8)


@needs_lammps
@pytest.mark.usefixtures("lammps_backend")
def test_static_prop_calcs(ar_cell: Atoms, monkeypatch: pytest.MonkeyPatch) -> None:
    """Static-only PropCalcs give the same results with both backends."""

    def run(calc_cls: type, calculator: object, **kwargs: object) -> dict:
        return calc_cls(calculator, relax_structure=False, **kwargs).calc(ar_cell)

    phonon_kwargs = {"supercell_matrix": ((2, 0, 0), (0, 2, 0), (0, 0, 2)), "t_max": 100, "t_step": 50}
    elastic_kwargs = {"relax_deformed_structures": False}
    phonon = run(PhononCalc, _lj_lammps(), **phonon_kwargs)
    elastic = run(ElasticityCalc, _lj_lammps(), **elastic_kwargs)

    monkeypatch.setattr(backend, "SIMULATION_BACKEND", "ASE")
    ref_phonon = run(PhononCalc, _lj_ase(), **phonon_kwargs)
    ref_elastic = run(ElasticityCalc, _lj_ase(), **elastic_kwargs)

    np.testing.assert_allclose(phonon["phonon"].force_constants, ref_phonon["phonon"].force_constants, atol=1e-7)
    # Thermal properties are not compared: round-off decides whether the ~1e-7 THz acoustic modes at Gamma enter them.
    q_point = [0.5, 0.25, 0.1]
    np.testing.assert_allclose(
        phonon["phonon"].get_frequencies(q_point), ref_phonon["phonon"].get_frequencies(q_point), atol=1e-6
    )
    np.testing.assert_allclose(elastic["elastic_tensor"], ref_elastic["elastic_tensor"], atol=1e-6)
    assert elastic["bulk_modulus_vrh"] == pytest.approx(ref_elastic["bulk_modulus_vrh"], rel=1e-6)


@needs_snap
@pytest.mark.parametrize("name", ["SNAP", "qSNAP"])
@pytest.mark.parametrize("left_handed", [False, True])
def test_maml_snap_invariance(name: str, left_handed: bool) -> None:
    """A rotated, re-based copy of an orthogonal cell gives the same energy and co-rotated forces and stress.

    maml's own single-point path is broken upstream (see test_utils.py), so it cannot serve as a reference.
    """
    calc = PESCalculator.load_snap(
        param_file=DIR / "pes" / f"{name}-Cu-2020.1-PES" / "SNAPotential.snapparam",
        coeff_file=DIR / "pes" / f"{name}-Cu-2020.1-PES" / "SNAPotential.snapcoeff",
    )
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    atoms.rattle(0.05, seed=0)
    ref = run_lammps(atoms, calc)  # orthogonal cell: already in LAMMPS form, no rotation or tilt reduction

    rotation, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
    rotation *= np.sign(np.linalg.det(rotation))
    basis_change = np.array([[1, 0, 0], [2, 1, 0], [-1, 3, 1]])  # unimodular: same lattice, large tilts
    if left_handed:
        basis_change = basis_change[[1, 0, 2]]
    other = atoms.copy()
    other.set_cell(basis_change @ np.array(atoms.cell) @ rotation, scale_atoms=False)
    other.positions = atoms.positions @ rotation
    res = run_lammps(other, calc)

    assert -4.5 < ref.potential_energy / len(atoms) < -3.0
    assert res.potential_energy == pytest.approx(ref.potential_energy, abs=1e-6)
    np.testing.assert_allclose(res.forces, ref.forces @ rotation, atol=1e-6)
    np.testing.assert_allclose(res.stress, rotation.T @ ref.stress @ rotation, atol=1e-7)
    assert np.abs(ref.forces).max() > 1e-2


# --- Relaxations ------------------------------------------------------------------------------------------------------


@needs_lammps
@pytest.mark.parametrize("left_handed", [False, True])
def test_relax_atoms_parity(ar_cell: Atoms, left_handed: bool) -> None:
    atoms = _skewed(ar_cell, left_handed=left_handed, shear=0.5)
    kwargs: dict[str, Any] = {"relax_atoms": True, "fmax": 1e-3, "max_steps": 2000}
    ref = run_ase(atoms, _lj_ase(), **kwargs)
    res = run_lammps(atoms, _lj_lammps(), **kwargs)

    assert ref.converged
    assert res.converged
    assert np.linalg.norm(res.forces, axis=1).max() <= 1e-3
    assert res.potential_energy == pytest.approx(ref.potential_energy, abs=1e-4)
    # Same cell, lattice basis and periodic images as the input.
    np.testing.assert_allclose(res.structure.lattice.matrix, np.array(atoms.cell), atol=1e-10)
    np.testing.assert_allclose(res.structure.cart_coords, ref.structure.cart_coords, atol=2e-2)


@needs_lammps
@pytest.mark.parametrize("left_handed", [False, True])
def test_relax_cell_parity(ar_cell: Atoms, left_handed: bool) -> None:
    atoms = _skewed(ar_cell, left_handed=left_handed, shear=0.5)
    kwargs: dict[str, Any] = {"relax_atoms": True, "relax_cell": True, "fmax": 1e-3, "max_steps": 3000}
    ref = run_ase(atoms, _lj_ase(), **kwargs)
    res = run_lammps(atoms, _lj_lammps(), **kwargs)

    assert ref.converged
    assert res.converged
    assert res.potential_energy == pytest.approx(ref.potential_energy, abs=2e-4 * len(atoms))
    # ASE's FrechetCellFilter criterion: cell virial per atom below fmax.
    volume = res.structure.volume
    assert np.linalg.norm(volume * res.stress / len(atoms), axis=1).max() <= 1e-3
    assert res.structure.volume == pytest.approx(ref.structure.volume, rel=1e-2)
    # Same lattice basis as the input: the metric tensor (basis-dependent, rotation-invariant) matches ASE's result.
    metric, ref_metric = (s.lattice.matrix @ s.lattice.matrix.T for s in (res.structure, ref.structure))
    np.testing.assert_allclose(metric, ref_metric, atol=0.05 * ref_metric.max())
    # No rigid shift: fix box/relax scales the box about its center, which moves the LAMMPS box origin.
    np.testing.assert_allclose(res.structure.frac_coords, ref.structure.frac_coords, atol=1e-2)


@needs_lammps
def test_relax_cell_pressure(ar_supercell: Atoms) -> None:
    pressure = 0.005  # eV/A^3, about 0.8 GPa
    kwargs: dict[str, Any] = {"relax_atoms": True, "relax_cell": True, "fmax": 1e-4, "max_steps": 3000}
    free = run_lammps(ar_supercell, _lj_lammps(), **kwargs)
    hydro = run_lammps(
        ar_supercell,
        _lj_lammps(),
        cell_filter_kwargs={"scalar_pressure": pressure, "hydrostatic_strain": True},
        **kwargs,
    )
    assert hydro.converged
    assert hydro.structure.volume < 0.95 * free.structure.volume
    np.testing.assert_allclose(np.trace(hydro.stress) / 3, -pressure, atol=1e-4 * len(ar_supercell) / 1000)
    np.testing.assert_allclose(hydro.structure.lattice.angles, 90, atol=1e-6)  # hydrostatic: shape kept


@needs_lammps
def test_relax_constant_volume(ar_cell: Atoms) -> None:
    """Shape-only relaxation (as used by EOSCalc) keeps the volume and matches ASE's result."""
    atoms = _skewed(ar_cell, shear=0.3)
    kwargs: dict[str, Any] = {
        "relax_atoms": True,
        "relax_cell": True,
        "fmax": 1e-3,
        "max_steps": 3000,
        "cell_filter_kwargs": {"constant_volume": True},
    }
    ref = run_ase(atoms, _lj_ase(), **kwargs)
    res = run_lammps(atoms, _lj_lammps(), **kwargs)

    assert res.converged
    assert res.structure.volume == pytest.approx(atoms.get_volume(), rel=1e-10)
    assert res.potential_energy == pytest.approx(ref.potential_energy, abs=2e-4 * len(atoms))
    # Converged shape: the deviatoric stress vanishes (to fmax), the mean stress need not.
    deviatoric = res.stress - np.trace(res.stress) / 3 * np.eye(3)
    assert np.linalg.norm(res.structure.volume * deviatoric / len(atoms), axis=1).max() <= 1e-3
    assert abs(np.trace(res.stress)) > 1e-4


@needs_lammps
@pytest.mark.usefixtures("lammps_backend")
def test_relax_calc_options(ar_cell: Atoms, tmp_path: Path) -> None:
    atoms = _skewed(ar_cell, shear=0.3)

    # Fixed atoms do not move.
    atoms.set_constraint(FixAtoms(indices=[0, 3]))
    result = run_lammps(atoms, _lj_lammps(), relax_atoms=True, fmax=1e-3)
    np.testing.assert_allclose(result.structure.cart_coords[[0, 3]], atoms.positions[[0, 3]], atol=1e-10)
    assert np.abs(result.structure.cart_coords - atoms.positions).max() > 1e-2
    atoms.set_constraint()

    # Cell-only relaxation: fractional coordinates follow the cell.
    cell_only = RelaxCalc(_lj_lammps(), relax_atoms=False, relax_cell=True, fmax=1e-3, max_steps=2000).calc(atoms)
    np.testing.assert_allclose(
        cell_only["final_structure"].frac_coords, atoms.get_scaled_positions(wrap=False), atol=1e-8
    )
    assert cell_only["final_structure"].volume != pytest.approx(atoms.get_volume(), rel=1e-3)

    # Optimizer classes map to LAMMPS minimizers; non-convergence is reported.
    assert RelaxCalc(_lj_lammps(), optimizer=BFGS, fmax=1e-3, max_steps=2000).calc(atoms)["is_converged"]
    assert not RelaxCalc(_lj_lammps(), fmax=1e-6, max_steps=2).calc(atoms)["is_converged"]
    with pytest.raises(ValueError, match="has no LAMMPS equivalent"):
        RelaxCalc(_lj_lammps(), optimizer="GPMin").calc(atoms)
    with pytest.raises(NotImplementedError, match="FixSymmetry"):
        RelaxCalc(_lj_lammps(), fix_symmetry=True).calc(atoms)
    with pytest.raises(NotImplementedError, match="mask"):
        RelaxCalc(_lj_lammps(), cell_filter_kwargs={"mask": [1, 1, 0, 0, 0, 0]}).calc(atoms)

    # Trajectory frames, ending at the relaxed structure.
    traj_file = tmp_path / "relax.traj"
    result = RelaxCalc(_lj_lammps(), fmax=1e-3, max_steps=2000, traj_file=str(traj_file), interval=5).calc(atoms)
    frames = read(traj_file, ":")
    assert len(frames) > 2
    assert frames[0].get_potential_energy() > frames[-1].get_potential_energy()
    assert frames[-1].get_potential_energy() == pytest.approx(result["energy"], abs=1e-6)
    np.testing.assert_allclose(frames[-1].cell, result["final_structure"].lattice.matrix, atol=1e-6)
    np.testing.assert_allclose(frames[-1].get_forces(), result["forces"], atol=1e-6)


@needs_lammps
@pytest.mark.usefixtures("lammps_backend")
def test_relaxing_prop_calcs(ar_cell: Atoms, monkeypatch: pytest.MonkeyPatch) -> None:
    """PropCalcs that relax give the same results with both backends."""
    elastic = ElasticityCalc(_lj_lammps(), fmax=1e-4).calc(ar_cell)
    eos = EOSCalc(_lj_lammps(), fmax=1e-4).calc(ar_cell)
    monkeypatch.setattr(backend, "SIMULATION_BACKEND", "ASE")
    ref_elastic = ElasticityCalc(_lj_ase(), fmax=1e-4).calc(ar_cell)
    ref_eos = EOSCalc(_lj_ase(), fmax=1e-4).calc(ar_cell)

    assert elastic["bulk_modulus_vrh"] == pytest.approx(ref_elastic["bulk_modulus_vrh"], rel=1e-2)
    assert elastic["shear_modulus_vrh"] == pytest.approx(ref_elastic["shear_modulus_vrh"], rel=2e-2)
    assert eos["bulk_modulus_bm"] == pytest.approx(ref_eos["bulk_modulus_bm"], rel=1e-2)


# --- Molecular dynamics -----------------------------------------------------------------------------------------------


@needs_lammps
@pytest.mark.parametrize("left_handed", [False, True])
def test_md_unwrapped_positions(ar_cell: Atoms, left_handed: bool) -> None:
    """Sparse trajectory frames retain motion across multiple periodic images."""
    atoms = _skewed(ar_cell, left_handed=left_handed)
    atoms.positions += np.array([2, -1, 3]) @ atoms.cell
    velocity = np.array([1.25, -0.8, 2.1]) @ atoms.cell / (100 * units.fs)
    atoms.set_velocities(np.tile(velocity, (len(atoms), 1)))
    calculator = LAMMPSCalculator(["pair_style zero 2.0", "pair_coeff * *"])
    frames = run_lammps_md(
        atoms,
        calculator,
        ensemble="nve",
        temperature=50,
        timestep=1,
        steps=100,
        loginterval=50,
        taut=10,
        taup=100,
    )
    assert len(frames) == 3
    for step, frame in zip((0, 50, 100), frames, strict=True):
        np.testing.assert_allclose(frame.positions, atoms.positions + velocity * step * units.fs, atol=1e-10)


def _md(calculator: object, atoms: Atoms, **kwargs: Any) -> dict:
    return MDCalc(calculator, relax_structure=False, seed=7, **kwargs).calc(atoms)  # type: ignore[arg-type]


@needs_lammps
@pytest.mark.usefixtures("lammps_backend")
def test_md_nve_parity(ar_cell: Atoms, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Velocity Verlet in LAMMPS reproduces ASE's trajectory from the same initial velocities."""
    atoms = _skewed(ar_cell, shear=0.3)
    kwargs: dict[str, Any] = {"ensemble": "nve", "temperature": 50, "timestep": 2.0, "steps": 53, "loginterval": 5}
    res = _md(_lj_lammps(), atoms, trajfile=tmp_path / "md.traj", **kwargs)
    monkeypatch.setattr(backend, "SIMULATION_BACKEND", "ASE")
    ref = _md(_lj_ase(), atoms, **kwargs)

    frames, ref_frames = res["trajectory"], ref["trajectory"]
    assert len(frames) == len(ref_frames) == 11
    assert len(read(tmp_path / "md.traj", ":")) == 11
    for frame, ref_frame in zip(frames, ref_frames, strict=True):
        np.testing.assert_allclose(frame.positions, ref_frame.positions, atol=1e-7)
        np.testing.assert_allclose(frame.get_velocities(), ref_frame.get_velocities(), atol=1e-7)
        np.testing.assert_allclose(frame.get_forces(), ref_frame.get_forces(), atol=1e-7)
        np.testing.assert_allclose(frame.cell, ref_frame.cell, atol=1e-10)
    for key in ("potential_energy", "kinetic_energy", "total_energy"):
        assert res[key] == pytest.approx(ref[key], abs=1e-7)


@needs_lammps
@pytest.mark.usefixtures("lammps_backend")
@pytest.mark.parametrize(
    ("ensemble", "kwargs"),
    [
        ("nvt", {}),
        ("nvt_berendsen", {}),
        ("nvt_langevin", {"friction": 0.02}),
        pytest.param(
            "nvt_bussi",
            {},
            marks=pytest.mark.skipif(
                not (REQUIRE_LAMMPS or _has_style("fix temp/csvr")), reason="needs LAMMPS EXTRA-FIX"
            ),
        ),
    ],
)
def test_md_thermostats(ar_supercell: Atoms, ensemble: str, kwargs: dict) -> None:
    temperature = 40.0
    result = _md(
        _lj_lammps(),
        ar_supercell,
        ensemble=ensemble,
        temperature=temperature,
        timestep=5.0,
        steps=2000,
        loginterval=10,
        **kwargs,
    )
    frames = result["trajectory"]
    mean_t = np.mean([frame.get_temperature() for frame in frames[len(frames) // 2 :]])
    assert mean_t == pytest.approx(temperature, rel=0.25)
    # Positions are unwrapped: no jumps across periodic boundaries between frames.
    steps = [np.abs(b.positions - a.positions).max() for a, b in itertools.pairwise(frames)]
    assert max(steps) < 1.0


@needs_lammps
@pytest.mark.usefixtures("lammps_backend")
@pytest.mark.parametrize(
    ("ensemble", "kwargs"),
    [
        ("npt", {"pressure": 0.0}),  # external_stress defaults to pressure
        ("npt_mtk", {"pressure": 0.0}),
        ("npt_isotropic_mtk", {"pressure": 0.0}),
        ("npt_berendsen", {"pressure": 0.0, "compressibility_au": 50.0}),
        ("npt_inhomogeneous", {"pressure": 0.0, "compressibility_au": 50.0}),
    ],
)
def test_md_barostats(ensemble: str, kwargs: dict) -> None:
    """A compressed crystal expands at zero pressure."""
    atoms = bulk("Ar", "fcc", a=5.0, cubic=True).repeat((2, 2, 2))
    result = _md(
        _lj_lammps(),
        atoms,
        ensemble=ensemble,
        temperature=20,
        timestep=5.0,
        steps=400,
        loginterval=20,
        set_com_stationary=True,
        **kwargs,
    )
    frames = result["trajectory"]
    volumes = [frame.get_volume() for frame in frames]
    assert volumes[-1] > 1.05 * volumes[0]
    # The barostat scales the box about its center, which moves the LAMMPS box origin; with zero momentum, the
    # mean fractional position must stay put.
    mean_frac = [frame.get_scaled_positions(wrap=False).mean(axis=0) for frame in (frames[0], frames[-1])]
    np.testing.assert_allclose(mean_frac[1], mean_frac[0], atol=1e-3)


@needs_lammps
@pytest.mark.usefixtures("lammps_backend")
def test_md_npt_fixed_cell() -> None:
    """npt with pfactor=None keeps the cell fixed, as ASE NPT does."""
    atoms = bulk("Ar", "fcc", a=5.0, cubic=True).repeat((2, 2, 2))
    result = _md(_lj_lammps(), atoms, ensemble="npt", pressure=0.0, pfactor=None, temperature=20, steps=100)
    for frame in result["trajectory"]:
        np.testing.assert_allclose(frame.cell, result["trajectory"][0].cell, atol=1e-10)


@needs_lammps
@pytest.mark.usefixtures("lammps_backend")
def test_md_unsupported(ar_supercell: Atoms) -> None:
    with pytest.raises(NotImplementedError, match="Andersen"):
        _md(_lj_lammps(), ar_supercell, ensemble="nvt_andersen", steps=2)
    with pytest.raises(ValueError, match="requires compressibility_au"):
        _md(_lj_lammps(), ar_supercell, ensemble="npt_berendsen", steps=2)
    with pytest.raises(NotImplementedError, match="orthogonal cell"):
        _md(_lj_lammps(), _skewed(ar_supercell), ensemble="npt_berendsen", compressibility_au=50.0, steps=2)
    # The engine is usable again after a failed setup.
    assert _md(_lj_lammps(), ar_supercell, ensemble="nve", steps=2)["total_energy"] < 0


# --- MatGL ------------------------------------------------------------------------------------------------------------


@needs_matgl
def test_matgl_parity(matpes_calculator: PESCalculatorType, LiFePO4: Structure) -> None:
    for structure in (LiFePO4, _skewed(bulk("Si", "diamond", a=5.43, cubic=True), shear=0.5)):
        ref = run_ase(structure, matpes_calculator)
        res = run_lammps(structure, matpes_calculator)
        assert res.potential_energy / len(structure) == pytest.approx(ref.potential_energy / len(structure), abs=1e-5)
        np.testing.assert_allclose(res.forces, ref.forces, atol=5e-4)
        np.testing.assert_allclose(res.stress, ref.stress, atol=1e-5)


@needs_matgl
@pytest.mark.usefixtures("lammps_backend")
def test_matgl_relax_and_md(matpes_calculator: PESCalculatorType, monkeypatch: pytest.MonkeyPatch) -> None:
    atoms = _skewed(bulk("Si", "diamond", a=5.5, cubic=True), shear=0.2)
    relax = RelaxCalc(matpes_calculator, fmax=0.01).calc(atoms)
    md_kwargs: dict[str, Any] = {"ensemble": "nve", "temperature": 300, "timestep": 1.0, "steps": 20, "loginterval": 5}
    md = _md(matpes_calculator, relax["final_structure"], **md_kwargs)

    monkeypatch.setattr(backend, "SIMULATION_BACKEND", "ASE")
    ref_relax = RelaxCalc(matpes_calculator, fmax=0.01).calc(atoms)
    ref_md = _md(matpes_calculator, relax["final_structure"], **md_kwargs)

    assert relax["is_converged"]
    assert ref_relax["is_converged"]
    assert relax["energy"] / len(atoms) == pytest.approx(ref_relax["energy"] / len(atoms), abs=1e-3)
    assert relax["final_structure"].volume == pytest.approx(ref_relax["final_structure"].volume, rel=5e-3)
    for frame, ref_frame in zip(md["trajectory"], ref_md["trajectory"], strict=True):
        np.testing.assert_allclose(frame.positions, ref_frame.positions, atol=1e-4)
    assert md["total_energy"] == pytest.approx(ref_md["total_energy"], abs=1e-3)
