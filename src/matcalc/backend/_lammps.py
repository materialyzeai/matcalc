"""LAMMPS backend: static calculations, relaxations and molecular dynamics.

The three protocols run on the shared, cached LAMMPS instances of ``_lammps_engine.py``:

* ``run_lammps`` without relaxation is a single ``run 0``.
* ``run_lammps`` with relaxation uses ``minimize`` (with ``fix box/relax`` for the cell), judged by ASE's convergence
  criterion.
* ``run_lammps_md`` runs the ensembles of ``MDCalc`` with the equivalent LAMMPS time-integration fixes.

Relaxations and MD are separate drivers because LAMMPS treats them differently: ``minimize`` and ``run`` accept
different fixes (``fix box/relax`` vs ``fix npt``), and MD can be run in chunks without changing the trajectory, while
restarting ``minimize`` resets its search direction.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.io import Trajectory
from ase.stress import full_3x3_to_voigt_6_stress
from pymatgen.io.lammps.outputs import parse_lammps_dumps, parse_lammps_log

from matcalc.utils import to_ase_atoms, to_pmg_structure

from ._ase import VALID_OPTIMIZERS
from ._base import SimulationResult
from ._lammps_engine import (
    ASE_TIME_TO_PS,
    ASE_VELOCITY_TO_LAMMPS,
    BAR_TO_EVA3,
    get_engine,
    get_lammps_potential,
    pressure_matrix,
    scratch_dir,
)

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from pymatgen.core.structure import Structure

    from ._lammps_engine import Frame, LAMMPSEngine

logger = logging.getLogger(__name__)

# Cell relaxations restart the minimizer every this many iterations to check ASE's convergence criterion (see _relax).
_CELL_RELAX_CHUNK = 50

# ASE optimizer (lower case) -> LAMMPS min_style. LAMMPS has no quasi-Newton minimizer; cg is the closest.
_MIN_STYLES = {
    "fire": "fire",
    "fire2": "fire",
    "mdmin": "quickmin",
    "bfgs": "cg",
    "lbfgs": "cg",
    "bfgslinesearch": "cg",
    "lbfgslinesearch": "cg",
    "goodoldquasinewton": "cg",
    "cg": "cg",
    "sd": "sd",
    "quickmin": "quickmin",
    "hftn": "hftn",
}


# --- Static calculations ----------------------------------------------------------------------------------------------


def run_lammps(
    structure: Structure | Atoms,
    calculator: Calculator,
    *,
    relax_atoms: bool = False,
    relax_cell: bool = False,
    optimizer: Any = "FIRE",
    max_steps: int = 500,
    traj_file: str | None = None,
    interval: int = 1,
    fmax: float = 0.1,
    cell_filter: Any = None,  # noqa: ARG001
    cell_filter_kwargs: dict | None = None,
) -> SimulationResult:
    """Run a static calculation or relaxation in LAMMPS with the force field of ``calculator``.

    Relaxations use ``minimize`` and, with ``relax_cell``, ``fix box/relax``. ``FixAtoms`` constraints become
    ``fix setforce``. Convergence is judged by the same criterion as ``run_ase`` with ``FrechetCellFilter``: the largest
    atomic force and the largest row of the cell virial divided by the number of atoms must not exceed ``fmax``.

    Args:
        structure: Input structure. Must be periodic in all three directions.
        calculator: ASE calculator with a LAMMPS representation: a MatGL ``PESCalculator`` (TensorNet or M3GNet), a
            maml-backed ``matcalc.utils.PESCalculator``, or a :class:`LAMMPSCalculator`.
        relax_atoms: Whether to relax atomic positions.
        relax_cell: Whether to also relax the cell (only with ``relax_atoms``, as in ``run_ase``).
        optimizer: ASE optimizer name or class, mapped to a LAMMPS ``min_style`` (FIRE -> fire, BFGS-type -> cg,
            MDMin -> quickmin); LAMMPS names (cg, sd, fire, quickmin, hftn) are also accepted. Cell relaxations always
            use cg or sd, since ``fix box/relax`` requires a line-search minimizer.
        max_steps: Maximum number of minimizer iterations.
        traj_file: Optional ASE trajectory file for the relaxation frames.
        interval: Frame interval for ``traj_file``.
        fmax: Force convergence criterion (eV/Å).
        cell_filter: Ignored; the cell is relaxed with ``fix box/relax``.
        cell_filter_kwargs: Supported keys: ``scalar_pressure`` (eV/Å^3), ``hydrostatic_strain`` and
            ``exp_cell_factor`` (``cell_factor`` for UnitCellFilter).

    Returns:
        SimulationResult with energies (eV), forces (eV/Å) and stress (eV/Å^3, 3x3, ASE sign convention) in the
        Cartesian frame and lattice basis of ``structure``. ``converged`` is None for a static calculation.

    Raises:
        TypeError: If ``calculator`` has no LAMMPS representation.
        NotImplementedError: For constraints other than ``FixAtoms`` or unsupported ``cell_filter_kwargs``.
    """
    potential = get_lammps_potential(calculator)
    atoms = to_ase_atoms(structure)
    engine = get_engine(potential, atoms)
    if not relax_atoms:
        energy, forces, stress = engine.compute(atoms)
        return _result(atoms, energy, forces, stress, structure=structure)
    final, energy, forces, stress, converged = _relax(
        engine,
        atoms,
        relax_cell=relax_cell,
        optimizer=optimizer,
        max_steps=max_steps,
        traj_file=traj_file,
        interval=interval,
        fmax=fmax,
        cell_filter_kwargs=cell_filter_kwargs,
    )
    if not converged:
        logger.warning(
            "Maximum steps reached in structure relaxation. Max|F| = %s eV/Å", np.linalg.norm(forces, axis=1).max()
        )
    return _result(final, energy, forces, stress, converged=converged)


def _result(
    atoms: Atoms,
    energy: float,
    forces: np.ndarray,
    stress: np.ndarray,
    *,
    structure: Structure | Atoms | None = None,
    converged: bool | None = None,
) -> SimulationResult:
    """Build a SimulationResult, applying the constraints of ``atoms`` to forces and stress as ``run_ase`` does."""
    atoms.calc = SinglePointCalculator(
        atoms, energy=energy, free_energy=energy, forces=forces, stress=full_3x3_to_voigt_6_stress(stress)
    )
    return SimulationResult(
        to_pmg_structure(atoms if structure is None else structure),
        atoms.get_potential_energy(),
        atoms.get_kinetic_energy(),
        atoms.get_total_energy(),
        atoms.get_forces(),
        atoms.get_stress(voigt=False),
        converged=converged,
    )


# --- Relaxation -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _CellTarget:
    """Target and convergence settings of a cell relaxation, from ASE cell-filter keyword arguments."""

    pressure: float = 0.0  # eV/Å^3
    hydrostatic: bool = False
    constant_volume: bool = False
    exp_cell_factor: float | None = None

    @classmethod
    def from_kwargs(cls, kwargs: dict | None) -> _CellTarget:
        kwargs = dict(kwargs or {})
        mask = kwargs.pop("mask", None)
        if mask is not None and not np.all(np.asarray(mask) == 1):
            raise NotImplementedError("A cell filter mask is not supported by the LAMMPS backend.")
        factor = kwargs.pop("exp_cell_factor", None)
        factor = kwargs.pop("cell_factor", factor)
        target = cls(
            pressure=float(kwargs.pop("scalar_pressure", 0.0)),
            hydrostatic=bool(kwargs.pop("hydrostatic_strain", False)),
            constant_volume=bool(kwargs.pop("constant_volume", False)),
            exp_cell_factor=factor,
        )
        if kwargs:
            raise NotImplementedError(f"cell_filter_kwargs {sorted(kwargs)} are not supported by the LAMMPS backend.")
        return target


def _min_style(optimizer: Any, *, relax_cell: bool, kokkos: bool = False) -> str:
    """Map an ASE optimizer (name or class) to a LAMMPS min_style."""
    name = optimizer if isinstance(optimizer, str) else getattr(optimizer, "__name__", str(optimizer))
    style = _MIN_STYLES.get(name.lower())
    if style is None:
        if isinstance(optimizer, str) and optimizer not in VALID_OPTIMIZERS:
            raise ValueError(f"Unknown {optimizer=}, must be one of {VALID_OPTIMIZERS}")  # as run_ase
        raise ValueError(f"Optimizer {name!r} has no LAMMPS equivalent. Choose from {sorted(_MIN_STYLES)}.")
    if kokkos and style != "cg":
        logger.info("cg is the only Kokkos minimizer; using min_style cg instead of %s.", style)
        return "cg"
    if relax_cell and style not in ("cg", "sd"):
        logger.info("fix box/relax requires a line-search minimizer; using min_style cg instead of %s.", style)
        return "cg"
    if style != name.lower():
        logger.info("Using LAMMPS min_style %s for optimizer %s.", style, name)
    return style


def _fixed_atoms(atoms: Atoms) -> list[int]:
    """Indices of atoms fixed by ``FixAtoms``; other constraints are not supported."""
    fixed: set[int] = set()
    for constraint in atoms.constraints:
        if not isinstance(constraint, FixAtoms):
            raise NotImplementedError(
                f"{type(constraint).__name__} is not supported in LAMMPS relaxations; use MATCALC_BACKEND=ASE."
            )
        fixed.update(int(i) for i in constraint.get_indices())
    return sorted(fixed)


def _residual_force(
    forces: np.ndarray, stress: np.ndarray, cell: np.ndarray, target: _CellTarget, *, relax_cell: bool
) -> float:
    """Largest generalized force as seen by an ASE optimizer on ``FrechetCellFilter`` (at zero strain).

    Rotation-dependent (row norms of the cell virial), so ``stress`` must be in the frame of the input structure.
    """
    residual = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0
    if not relax_cell:
        return residual
    virial = -abs(np.linalg.det(cell)) * (stress + target.pressure * np.eye(3))
    if target.hydrostatic:
        virial = np.trace(virial) / 3 * np.eye(3)
    if target.constant_volume:
        virial = virial - np.trace(virial) / 3 * np.eye(3)
    cell_forces = virial / (target.exp_cell_factor or len(forces))
    return max(residual, float(np.linalg.norm(cell_forces, axis=1).max()))


def _relax(
    engine: LAMMPSEngine,
    atoms: Atoms,
    *,
    relax_cell: bool,
    optimizer: Any,
    max_steps: int,
    traj_file: str | None,
    interval: int,
    fmax: float,
    cell_filter_kwargs: dict | None,
) -> tuple[Atoms, float, np.ndarray, np.ndarray, bool]:
    """Relax ``atoms`` with LAMMPS ``minimize``; returns the relaxed atoms, energy, forces, stress and verdict.

    LAMMPS scales the box degrees of freedom of ``fix box/relax`` by the volume rather than by the volume per atom, so
    its own stopping criterion is far stricter than ASE's for cells. Cell relaxations therefore run in chunks of
    ``_CELL_RELAX_CHUNK`` iterations and stop once ASE's criterion is met; restarting also resets the reference cell of
    ``fix box/relax``, as the LAMMPS documentation recommends.

    ``fix box/relax`` has no volume constraint. A constant-volume relaxation (``constant_volume``, as in ASE's cell
    filters) targets the current mean pressure in each chunk, which removes the drive to change the volume, and rescales
    the cell isotropically back to the initial volume after each chunk.
    """
    fixed = _fixed_atoms(atoms)
    target = _CellTarget.from_kwargs(cell_filter_kwargs) if relax_cell else _CellTarget()
    style = _min_style(optimizer, relax_cell=relax_cell, kokkos=engine.kokkos)
    frame = engine.load(atoms)
    lmp = engine.lmp

    setup = ["timestep 0.001", f"min_style {style}", "min_modify norm max"]
    cleanup = []
    if fixed:
        setup += [
            f"group matcalc_fixed id {' '.join(str(i + 1) for i in fixed)}",
            "fix matcalc_setforce matcalc_fixed setforce 0.0 0.0 0.0",
        ]
        cleanup += ["unfix matcalc_setforce", "group matcalc_fixed delete"]
    recorder = _MinimizeRecorder(engine, interval) if traj_file else None

    with engine.scoped(setup, cleanup):
        converged = _minimize(
            engine, frame, fixed, target, relax_cell=relax_cell, max_steps=max_steps, fmax=fmax, recorder=recorder
        )

    lmp.command("run 0")
    cell = engine.cell()
    final = atoms.copy()
    final.set_cell(frame.cell(cell))
    final.positions, _ = frame.positions(engine.positions(), cell)
    energy, forces, stress = engine.energy(), frame.vectors(engine.gather("f")), frame.tensor(engine.stress())
    if recorder is not None and traj_file is not None:
        last = final.copy()
        last.calc = SinglePointCalculator(
            last, energy=energy, free_energy=energy, forces=forces, stress=full_3x3_to_voigt_6_stress(stress)
        )
        recorder.write(traj_file, frame, atoms, last)
    return final, energy, forces, stress, converged


def _minimize(
    engine: LAMMPSEngine,
    frame: Frame,
    fixed: list[int],
    target: _CellTarget,
    *,
    relax_cell: bool,
    max_steps: int,
    fmax: float,
    recorder: _MinimizeRecorder | None,
) -> bool:
    """Run ``minimize`` (in chunks for cell relaxations, see ``_relax``) until ASE's criterion is met; return it."""
    lmp = engine.lmp
    volume = abs(np.linalg.det(engine.cell()))
    couple = "iso" if target.hydrostatic else "tri"
    if target.constant_volume:
        lmp.command("run 0")
    steps_left = max_steps
    converged = False
    while steps_left > 0:
        chunk = min(steps_left, _CELL_RELAX_CHUNK) if relax_cell else steps_left
        if relax_cell:
            pressure = -np.trace(engine.stress()) / 3 if target.constant_volume else target.pressure
            lmp.command(f"fix matcalc_box all box/relax {couple} {pressure / BAR_TO_EVA3} vmax 0.001")
        start = engine.step()
        try:
            with recorder.record() if recorder else contextlib.nullcontext():
                lmp.command(f"minimize 0.0 {fmax} {chunk} {100 * chunk}")
        finally:
            if relax_cell:
                lmp.command("unfix matcalc_box")
        used = engine.step() - start
        steps_left -= used
        if target.constant_volume:
            engine.set_volume(volume)
        if fixed and relax_cell:
            # The minimizer lets fix box/relax shift atoms held by fix setforce; ASE keeps their fractional
            # coordinates, so restore them before checking convergence.
            engine.pin(fixed, frame)
        lmp.command("run 0")  # forces of fixed atoms are zeroed by fix setforce
        residual = _residual_force(
            engine.gather("f"), frame.tensor(engine.stress()), engine.cell(), target, relax_cell=relax_cell
        )
        converged = residual <= fmax
        if converged or used == 0:
            break
    return converged


class _MinimizeRecorder:
    """Records minimization frames through a dump file and the thermo log, and writes them as an ASE trajectory."""

    def __init__(self, engine: LAMMPSEngine, interval: int) -> None:
        self.engine = engine
        self.interval = interval
        self.directory = Path(tempfile.mkdtemp(prefix="min-", dir=scratch_dir()))
        self.files: list[tuple[Path, Path]] = []

    @contextlib.contextmanager
    def record(self) -> Any:
        dump, log = (self.directory / f"min{len(self.files)}.{ext}" for ext in ("dump", "log"))
        self.files.append((dump, log))
        self.engine.lmp.commands_list(
            [
                f'dump matcalc_traj all custom {self.interval} "{dump}" id x y z fx fy fz',
                "dump_modify matcalc_traj sort id format float %.15g",
                f"thermo {self.interval}",
                "thermo_modify format float %.15g",
                f'log "{log}"',
            ]
        )
        try:
            yield
        finally:
            self.engine.lmp.commands_list(["undump matcalc_traj", "log none", "thermo 0", "thermo_modify format none"])

    def write(self, traj_file: str, frame: Frame, atoms: Atoms, final: Atoms) -> None:
        """Write the recorded frames, ending with ``final``, the accepted relaxed state.

        The last minimizer evaluation (e.g. a rejected line-search step) need not be the accepted state, so the final
        frame is written explicitly, as ASE's optimizers do.
        """
        seen: set[int] = set()
        with Trajectory(traj_file, "w") as traj:
            for dump, log in self.files:
                thermo = {int(row.Step): row for row in _read_thermo(log).itertuples()}
                for snapshot in parse_lammps_dumps(str(dump)):
                    row = thermo.get(snapshot.timestep)
                    if row is None or snapshot.timestep in seen:
                        continue
                    seen.add(snapshot.timestep)
                    cell = _cell_from_thermo(row)
                    data = snapshot.data.sort_values("id")
                    image = atoms.copy()
                    image.set_cell(frame.cell(cell))
                    origin = np.array([row.Xlo, row.Ylo, row.Zlo])
                    image.positions, _ = frame.positions(data[["x", "y", "z"]].to_numpy() - origin, cell)
                    stress = -BAR_TO_EVA3 * pressure_matrix(
                        [getattr(row, f"c_matcalc_press_{i}_") for i in range(1, 7)]
                    )
                    image.calc = SinglePointCalculator(
                        image,
                        energy=row.PotEng,
                        free_energy=row.PotEng,
                        forces=frame.vectors(data[["fx", "fy", "fz"]].to_numpy()),
                        stress=full_3x3_to_voigt_6_stress(frame.tensor(stress)),
                    )
                    traj.write(image)
            traj.write(final)
        shutil.rmtree(self.directory, ignore_errors=True)


def _read_thermo(log: Path) -> Any:
    """Thermo output of all runs in a LAMMPS log, with column names made attribute-safe."""
    import pandas as pd

    frames = parse_lammps_log(str(log))
    table = pd.concat(frames, ignore_index=True)
    table.columns = [c.replace("[", "_").replace("]", "_") for c in table.columns]
    return table


def _cell_from_thermo(row: Any) -> np.ndarray:
    return np.array([[row.Lx, 0.0, 0.0], [row.Xy, row.Ly, 0.0], [row.Xz, row.Yz, row.Lz]])


# --- Molecular dynamics -----------------------------------------------------------------------------------------------

# Fix styles that measure the temperature, and so accept ``fix_modify temp``.
_TEMPERATURE_STYLES = ("nvt", "npt", "temp/berendsen", "temp/csvr", "press/berendsen")
_ORTHO_ENSEMBLES = ("npt_berendsen", "npt_inhomogeneous")


def run_lammps_md(
    atoms: Atoms,
    calculator: Calculator,
    *,
    ensemble: str,
    temperature: float,
    timestep: float,
    steps: int,
    loginterval: int = 1,
    pressure: float = 0.0,
    taut: float,
    taup: float,
    friction: float = 1.0e-3,
    ttime: float = 25.0,
    fixed_cell: bool = False,
    external_stress: float | np.ndarray | None = None,
    compressibility_au: float | None = None,
    mask: Any = None,
    tchain: int = 3,
    pchain: int = 3,
    tloop: int = 1,
    ploop: int = 1,
    seed: int | None = None,
    logfile: str | None = None,
) -> list[Atoms]:
    """Run MD in LAMMPS with the force field of ``calculator``; parameters follow ``MDCalc`` and ASE units.

    Initial velocities come from ``atoms``, so both backends start from identical conditions. The NVE integrator is the
    same velocity Verlet scheme as ASE's; thermostats and barostats sample the same ensembles with different integrators
    (e.g. LAMMPS ``fix npt`` is MTK, ASE ``NPT`` is Melchionna), so those trajectories only agree statistically.

    Args:
        atoms: Initial structure with velocities.
        calculator: ASE calculator with a LAMMPS representation (see ``run_lammps``).
        ensemble: ``MDCalc`` ensemble name. ``nvt_andersen`` has no LAMMPS equivalent.
        temperature: Target temperature (K).
        timestep: Time step (fs).
        steps: Number of MD steps.
        loginterval: Interval between returned frames.
        pressure: Target pressure (eV/Å^3) for the Berendsen and MTK barostats.
        taut: Thermostat time constant (ASE time units).
        taup: Barostat time constant (ASE time units); also used for ``npt``/``npt_nose_hoover``, whose ASE
            ``pfactor`` has no LAMMPS equivalent.
        friction: Langevin friction (1 / ASE time units).
        ttime: Thermostat time for ``npt``/``npt_nose_hoover`` (fs).
        fixed_cell: For ``npt``/``npt_nose_hoover``, keep the cell fixed (Nose-Hoover NVT with ``ttime``), as ASE
            ``NPT`` does with ``pfactor=None``.
        external_stress: Target pressure (eV/Å^3, scalar) for ``npt``/``npt_nose_hoover``.
        compressibility_au: Compressibility (Å^3/eV) for the Berendsen barostats; required, as in ASE.
        mask: Cell mask for ``npt``/``npt_nose_hoover``: None or diagonal (orthogonal cell changes, LAMMPS ``aniso``)
            or all ones (fully flexible cell, LAMMPS ``tri``).
        tchain: Thermostat chain length (MTK ensembles).
        pchain: Barostat chain length (MTK ensembles).
        tloop: Thermostat sub-steps (MTK ensembles).
        ploop: Barostat sub-steps (MTK ensembles).
        seed: Seed for stochastic thermostats.
        logfile: Optional LAMMPS log file with thermo output every ``loginterval`` steps.

    Returns:
        Frames at steps 0, loginterval, 2 * loginterval, ..., as ASE MD observers record them, each with velocities
        and a SinglePointCalculator holding energy, forces and stress. Positions are unwrapped.
    """
    engine = get_engine(get_lammps_potential(calculator), atoms)
    lmp = engine.lmp
    masses = _type_masses(atoms, engine)
    momenta = atoms.get_momenta()
    if ensemble.lower() in ("npt", "npt_nose_hoover"):
        # ASE's NPT (Melchionna) zeroes the center-of-mass momentum on construction, in equal shares per atom.
        momenta = momenta - momenta.sum(axis=0) / len(atoms)
    velocities = momenta / atoms.get_masses()[:, None]
    frame = engine.load(atoms, velocities=velocities * ASE_VELOCITY_TO_LAMMPS)
    fixes = _ensemble_fixes(
        ensemble,
        temperature=temperature,
        taut=taut * ASE_TIME_TO_PS,
        taup=taup * ASE_TIME_TO_PS,
        damp=ASE_TIME_TO_PS / friction,
        ttime=ttime * 1e-3,
        fixed_cell=fixed_cell,
        pressure=pressure / BAR_TO_EVA3,
        external_stress=external_stress,
        compressibility_au=compressibility_au,
        mask=mask,
        chains=f"tchain {tchain} pchain {pchain} tloop {tloop} ploop {ploop}",
        seed=seed,
    )
    setup = [*masses, "reset_timestep 0", f"timestep {timestep * 1e-3}"]
    cleanup = []
    if ensemble.lower() in _ORTHO_ENSEMBLES:
        # fix press/berendsen does not support triclinic boxes.
        cell = engine.cell()
        if np.abs(cell[np.tril_indices(3, -1)]).max() > 1e-8 * np.abs(np.diag(cell)).max():
            raise NotImplementedError(f"{ensemble} with the LAMMPS backend requires an orthogonal cell.")
        setup.append("change_box all xy final 0.0 xz final 0.0 yz final 0.0 ortho")
        cleanup.append("change_box all triclinic")
    # ASE's thermostats count 3N degrees of freedom; LAMMPS removes 3 for the center of mass by default, which halves
    # the target kinetic energy of a 2-atom cell. Thermostats and barostats use a 3N temperature instead.
    setup += ["compute matcalc_temp all temp", "compute_modify matcalc_temp extra/dof 0", *fixes]
    fix_ids = [(cmd.split()[1], cmd.split()[3]) for cmd in fixes]  # "fix ID group style ..."
    setup += [f"fix_modify {fix} temp matcalc_temp" for fix, style in fix_ids if style in _TEMPERATURE_STYLES]
    if logfile:
        setup += [f'log "{logfile}"', f"thermo {loginterval}"]
        cleanup += ["log none", "thermo 0"]

    cleanup = [f"unfix {fix}" for fix, _ in fix_ids] + ["uncompute matcalc_temp", *cleanup]
    with engine.scoped(setup, cleanup):
        lmp.command("run 0")
        frames = [_snapshot(engine, frame, atoms)]
        for _ in range(steps // loginterval):
            lmp.command(f"run {loginterval} pre no post no")
            frames.append(_snapshot(engine, frame, atoms))
        if steps % loginterval:
            lmp.command(f"run {steps % loginterval} pre no post no")
    return frames


def _type_masses(atoms: Atoms, engine: LAMMPSEngine) -> list[str]:
    """``mass`` commands so that LAMMPS uses the masses of ``atoms`` (one mass per element)."""
    commands = []
    masses = atoms.get_masses()
    symbols = np.array(atoms.get_chemical_symbols())
    for el in sorted(set(symbols)):
        el_masses = masses[symbols == el]
        if not np.allclose(el_masses, el_masses[0]):
            raise NotImplementedError(f"The LAMMPS backend needs one mass per element; {el} atoms have several.")
        commands.append(f"mass {engine.type_of[el]} {el_masses[0]}")
    return commands


def _ensemble_fixes(  # noqa: C901, PLR0911, PLR0912
    ensemble: str,
    *,
    temperature: float,
    taut: float,
    taup: float,
    damp: float,
    ttime: float,
    fixed_cell: bool,
    pressure: float,
    external_stress: float | np.ndarray | None,
    compressibility_au: float | None,
    mask: Any,
    chains: str,
    seed: int | None,
) -> list[str]:
    """LAMMPS fixes for an ``MDCalc`` ensemble; times in ps, pressures in bar."""
    t = f"{temperature} {temperature}"
    nve = "fix matcalc_md all nve"
    # LAMMPS' Marsaglia RNG takes seeds below 9e8. Without a seed, draw from NumPy's global RNG as ASE's integrators
    # do, so that np.random.seed makes runs reproducible.
    if seed is None:
        lmp_seed = int(np.random.randint(1, 900_000_000))  # noqa: NPY002
    else:
        lmp_seed = int(np.random.default_rng(seed).integers(1, 900_000_000))
    ensemble = ensemble.lower()

    if ensemble == "nve":
        return [nve]
    if ensemble in ("nvt", "nvt_nose_hoover"):
        return [f"fix matcalc_md all nvt temp {t} {taut} tchain 3"]
    # ASE's Berendsen integrators remove the center-of-mass momentum every step (fixcm=True); so does fix momentum.
    zero_momentum = "fix matcalc_momentum all momentum 1 linear 1 1 1"
    if ensemble == "nvt_berendsen":
        return [nve, f"fix matcalc_thermostat all temp/berendsen {t} {taut}", zero_momentum]
    if ensemble == "nvt_langevin":
        return [nve, f"fix matcalc_thermostat all langevin {t} {damp} {lmp_seed} zero yes"]
    if ensemble == "nvt_bussi":
        return [nve, f"fix matcalc_thermostat all temp/csvr {t} {taut} {lmp_seed}"]
    if ensemble == "nvt_andersen":
        raise NotImplementedError("LAMMPS has no Andersen thermostat; use MATCALC_BACKEND=ASE for nvt_andersen.")
    if ensemble in ("npt", "npt_nose_hoover"):
        if fixed_cell:
            return [f"fix matcalc_md all nvt temp {t} {ttime}"]
        if external_stress is None:
            target = 0.0
        elif np.ndim(external_stress) == 0:
            target = float(external_stress) / BAR_TO_EVA3  # type: ignore[arg-type]
        else:
            raise NotImplementedError("The LAMMPS backend supports a scalar external_stress (pressure) only.")
        return [f"fix matcalc_md all npt temp {t} {ttime} {_npt_coupling(mask)} {target} {target} {taup}"]
    if ensemble in ("npt_berendsen", "npt_inhomogeneous"):
        if compressibility_au is None:
            raise ValueError(f"{ensemble} requires compressibility_au.")
        modulus = 1 / compressibility_au / BAR_TO_EVA3
        coupling = "iso" if ensemble == "npt_berendsen" else "aniso"
        return [
            nve,
            f"fix matcalc_thermostat all temp/berendsen {t} {taut}",
            f"fix matcalc_barostat all press/berendsen {coupling} {pressure} {pressure} {taup} modulus {modulus}",
            zero_momentum,
        ]
    if ensemble in ("npt_mtk", "npt_isotropic_mtk"):
        coupling = "tri" if ensemble == "npt_mtk" else "iso"
        return [f"fix matcalc_md all npt temp {t} {taut} {coupling} {pressure} {pressure} {taup} {chains}"]
    raise ValueError(f"Unsupported ensemble {ensemble!r}.")


def _npt_coupling(mask: Any) -> str:
    """LAMMPS barostat coupling for an ASE ``NPT`` cell mask."""
    if mask is None:
        return "aniso"
    mask = np.asarray(mask)
    if mask.shape == (3,):
        mask = np.diag(mask)
    if np.array_equal(mask, np.eye(3)):
        return "aniso"
    if np.all(mask == 1):
        return "tri"
    raise NotImplementedError("The LAMMPS backend supports NPT masks that are diagonal ones or all ones only.")


def _snapshot(engine: LAMMPSEngine, frame: Frame, atoms: Atoms) -> Atoms:
    """Current LAMMPS state as ASE Atoms in the input frame.

    Image flags retain boundary crossings between saved frames; the initial lattice offset restores the input images.
    """
    cell = engine.cell()
    # Fractional coordinates relative to the (possibly moved) box origin, unwrapped with the image flags, then mapped
    # to the input basis and periodic images.
    frac_lammps = np.linalg.solve(cell.T, engine.positions().T).T + engine.gather("image")
    positions = (frac_lammps @ frame.basis + frame.image_offset) @ frame.cell(cell)
    image = Atoms(
        atoms.get_chemical_symbols(),
        positions=positions,
        cell=frame.cell(cell),
        pbc=True,
        masses=atoms.get_masses(),
    )
    image.set_velocities(frame.vectors(engine.gather("v")) / ASE_VELOCITY_TO_LAMMPS)
    energy = engine.energy()
    image.calc = SinglePointCalculator(
        image,
        energy=energy,
        free_energy=energy,
        forces=frame.vectors(engine.gather("f")),
        stress=full_3x3_to_voigt_6_stress(frame.tensor(engine.stress())),
    )
    return image
