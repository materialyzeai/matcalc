"""LAMMPS backend for ``run_pes_calc``.

Calculations run in an in-process LAMMPS instance driven through the ``lammps`` Python module. The LAMMPS force field is
derived from the ASE calculator passed in:

* MatGL TensorNet / M3GNet potentials (``matgl.ext.ase.PESCalculator``) are exported to TorchScript with
  ``matgl.ext.lammps.export_lammps_model`` and evaluated with MatGL's own ``pair_style matgl``, which requires LAMMPS
  built with the ML-MATGL package. To run on a GPU through ``pair_style matgl/kk``, build with Kokkos and set
  ``MATCALC_LAMMPS_ARGS="-k on g 1 -sf kk -pk kokkos newton on neigh half"``.
* maml potentials wrapped in ``matcalc.utils.PESCalculator`` (SNAP, MTP, GAP, NNP) use the pair commands from
  ``Potential.write_param``.
* :class:`LAMMPSCalculator` evaluates arbitrary LAMMPS force-field commands, e.g. EAM or Tersoff potentials.

LAMMPS instances are cached per force field and element set, so repeated calculations (phonon displacements, strained
cells, ...) do not reload the model. Molecular dynamics lives in ``_lammps_md.py``.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import shutil
import tempfile
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.data import atomic_masses, atomic_numbers
from ase.io import Trajectory
from ase.stress import full_3x3_to_voigt_6_stress
from pymatgen.io.lammps.outputs import parse_lammps_dumps, parse_lammps_log

from matcalc.config import LAMMPS_CMDARGS
from matcalc.units import eVA3ToGPa
from matcalc.utils import PESCalculator, resolve_model, to_ase_atoms, to_pmg_structure

from ._base import SimulationResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core.structure import Structure

logger = logging.getLogger(__name__)

# LAMMPS ``metal`` units: pressure in bar (1 bar = 1e-4 GPa), time in ps, velocity in Å/ps.
BAR_TO_EVA3 = 1e-4 / eVA3ToGPa
ASE_TIME_TO_PS = 1e-3 / units.fs
ASE_VELOCITY_TO_LAMMPS = units.fs * 1e3
# Maximum number of live LAMMPS instances per process. Each holds a loaded force field (and, for MatGL, a model).
_MAX_ENGINES = 4
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


@dataclass(frozen=True)
class LAMMPSPotential:
    """LAMMPS force-field specification.

    Attributes:
        commands: Force-field commands (``pair_style``, ``pair_coeff``, ...). The placeholder ``{symbols}`` is
            replaced by the element symbols in LAMMPS atom-type order.
        elements: Element symbols in LAMMPS atom-type order. ``None`` uses the elements of each structure, sorted by
            atomic number, which creates one LAMMPS instance per element set.
        directory: Working directory while issuing ``commands``, for potential files given by relative paths.
        pre_commands: Commands issued before the simulation box is created (e.g. ``newton on``).
    """

    commands: tuple[str, ...]
    elements: tuple[str, ...] | None = None
    directory: str | None = None
    pre_commands: tuple[str, ...] = ()


class LAMMPSCalculator(Calculator):
    """ASE calculator that evaluates a LAMMPS force field in an in-process LAMMPS instance.

    It works with both simulation backends: with ``MATCALC_BACKEND=ASE`` ASE drives the optimizers and MD and calls
    LAMMPS for energies, forces and stresses; with ``MATCALC_BACKEND=LAMMPS`` everything runs in LAMMPS. Under the
    LAMMPS backend, ``PropCalc.calculator`` holds the ``LAMMPSCalculator`` for the given calculator (see
    :func:`to_lammps_calculator`), with the original in ``source``.

    Attributes:
        potential: The LAMMPS force field.
        source: The ASE calculator this one was converted from, or None.

    Example:
        >>> calc = LAMMPSCalculator(
        ...     ["pair_style eam/alloy", "pair_coeff * * Cu_u3.eam.alloy {symbols}"],
        ...     elements=["Cu"],
        ... )
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]  # noqa: RUF012

    def __init__(
        self,
        commands: Sequence[str],
        *,
        elements: Sequence[str] | None = None,
        potential_dir: str | Path | None = None,
        pre_commands: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        """
        Args:
            commands: LAMMPS force-field commands. ``{symbols}`` is replaced by the element symbols in atom-type order.
            elements: Element symbols in LAMMPS atom-type order. Defaults to the elements of each structure, sorted by
                atomic number.
            potential_dir: Directory in which ``commands`` are issued, for potential files given by relative paths.
                Defaults to the directory current at construction.
            pre_commands: Commands issued before the simulation box is created.
            **kwargs: Forwarded to ``ase.calculators.calculator.Calculator``.
        """
        super().__init__(**kwargs)
        self.potential = LAMMPSPotential(
            commands=tuple(commands),
            elements=tuple(elements) if elements is not None else None,
            directory=str(Path(potential_dir or Path.cwd()).resolve()),
            pre_commands=tuple(pre_commands),
        )
        self.source: Calculator | None = None

    @classmethod
    def from_potential(
        cls, potential: LAMMPSPotential, *, source: Calculator | None = None, **kwargs: Any
    ) -> LAMMPSCalculator:
        """Create a calculator for an existing force-field specification.

        Args:
            potential: The LAMMPS force field.
            source: The ASE calculator ``potential`` was derived from, kept for reference.
            **kwargs: Forwarded to ``ase.calculators.calculator.Calculator``.

        Returns:
            LAMMPSCalculator.
        """
        calc = cls(potential.commands, **kwargs)
        calc.potential = potential
        calc.source = source
        return calc

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        """Compute energy, forces and stress with LAMMPS.

        Args:
            atoms: Structure to evaluate.
            properties: Requested properties; all implemented properties are always computed.
            system_changes: ASE change list.
        """
        super().calculate(atoms, properties, system_changes)
        energy, forces, stress = get_engine(self.potential, self.atoms).compute(self.atoms)  # type: ignore[arg-type]
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
            "stress": full_3x3_to_voigt_6_stress(stress),
        }


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
    engine: _LAMMPSEngine,
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

    lmp.commands_list(setup)
    try:
        converged = _minimize(
            engine, frame, fixed, target, relax_cell=relax_cell, max_steps=max_steps, fmax=fmax, recorder=recorder
        )
    finally:
        lmp.commands_list(cleanup)

    lmp.command("run 0")
    cell = engine.cell()
    final = atoms.copy()
    final.set_cell(frame.cell(cell))
    final.positions, _ = frame.positions(engine.gather("x"), cell)
    energy, forces, stress = engine.energy(), frame.vectors(engine.gather("f")), frame.tensor(engine.stress())
    if recorder is not None and traj_file is not None:
        last = final.copy()
        last.calc = SinglePointCalculator(
            last, energy=energy, free_energy=energy, forces=forces, stress=full_3x3_to_voigt_6_stress(stress)
        )
        recorder.write(traj_file, frame, atoms, last)
    return final, energy, forces, stress, converged


def _minimize(
    engine: _LAMMPSEngine,
    frame: _Frame,
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

    def __init__(self, engine: _LAMMPSEngine, interval: int) -> None:
        self.engine = engine
        self.interval = interval
        self.directory = Path(tempfile.mkdtemp(prefix="min-", dir=_scratch_dir()))
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

    def write(self, traj_file: str, frame: _Frame, atoms: Atoms, final: Atoms) -> None:
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
                    image.positions, _ = frame.positions(data[["x", "y", "z"]].to_numpy(), cell)
                    stress = -BAR_TO_EVA3 * _pressure_matrix(
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


def _pressure_matrix(p: Sequence[float]) -> np.ndarray:
    """3x3 pressure tensor from LAMMPS (xx, yy, zz, xy, xz, yz) components."""
    pxx, pyy, pzz, pxy, pxz, pyz = p
    return np.array([[pxx, pxy, pxz], [pxy, pyy, pyz], [pxz, pyz, pzz]])


# --- Force-field resolution -----------------------------------------------------------------------------------------

# Exported force fields keyed by the underlying potential object, so that each model is exported once per process.
_POTENTIAL_CACHE: weakref.WeakKeyDictionary[Any, LAMMPSPotential] = weakref.WeakKeyDictionary()
# Exported MatGL models keyed by model name or resolved model directory (see load_lammps_calculator).
_NAMED_POTENTIALS: dict[str, LAMMPSPotential] = {}
_SCRATCH_DIR: Path | None = None


def get_lammps_potential(calculator: Calculator) -> LAMMPSPotential:
    """Return the LAMMPS force field equivalent to an ASE calculator.

    Args:
        calculator: A :class:`LAMMPSCalculator`, a maml-backed ``matcalc.utils.PESCalculator``, or a MatGL
            ``PESCalculator`` wrapping a TensorNet or M3GNet potential.

    Returns:
        The LAMMPS force-field specification.

    Raises:
        TypeError: If ``calculator`` has no LAMMPS representation.
    """
    if isinstance(calculator, LAMMPSCalculator):
        return calculator.potential
    if isinstance(calculator, PESCalculator):
        return _from_maml(calculator.potential)
    if type(calculator).__module__.startswith("matgl.") and hasattr(calculator, "potential"):
        return _from_matgl(calculator.potential)
    raise TypeError(
        f"{type(calculator).__name__} has no LAMMPS representation. The LAMMPS backend supports MatGL TensorNet/"
        "M3GNet calculators, maml potentials loaded through matcalc.utils.PESCalculator and "
        "matcalc.LAMMPSCalculator. Use MATCALC_BACKEND=ASE for other calculators."
    )


def to_lammps_calculator(calculator: Calculator) -> LAMMPSCalculator:
    """Return a :class:`LAMMPSCalculator` with the force field of an ASE calculator.

    Args:
        calculator: See :func:`get_lammps_potential`. A ``LAMMPSCalculator`` is returned unchanged.

    Returns:
        LAMMPSCalculator whose ``source`` is ``calculator``.

    Raises:
        TypeError: If ``calculator`` has no LAMMPS representation.
    """
    if isinstance(calculator, LAMMPSCalculator):
        return calculator
    return LAMMPSCalculator.from_potential(get_lammps_potential(calculator), source=calculator)


def _scratch_dir() -> Path:
    """Per-process directory for exported models, potential files and trajectories, removed at exit."""
    global _SCRATCH_DIR
    if _SCRATCH_DIR is None:
        _SCRATCH_DIR = Path(tempfile.mkdtemp(prefix="matcalc-lammps-"))
        atexit.register(shutil.rmtree, _SCRATCH_DIR, ignore_errors=True)
    return _SCRATCH_DIR


def load_lammps_calculator(name: str) -> LAMMPSCalculator:
    """Return a :class:`LAMMPSCalculator` for a model name, without building an ASE calculator for it.

    MatGL models (by registry name, alias, raw MatGL name or model directory) are loaded and exported to TorchScript
    for ``pair_style matgl`` once per process. Models of other providers have no LAMMPS representation and are
    rejected before anything is loaded.

    Args:
        name: Model name, as for ``PESCalculator.load_universal``.

    Returns:
        LAMMPSCalculator (``source`` is None).

    Raises:
        TypeError: If the model has no LAMMPS representation.
        ValueError: If ``name`` is not a recognized model.
    """
    provider, kwargs = resolve_model(name)
    if provider != "matgl":
        raise TypeError(
            f"{name!r} ({provider}) has no LAMMPS representation. By name, the LAMMPS backend supports MatGL TensorNet/"
            "M3GNet models; use MATCALC_BACKEND=ASE for other models."
        )
    path = kwargs["path"]
    key = str(Path(path).resolve()) if Path(path).is_dir() else path
    if (potential := _NAMED_POTENTIALS.get(key)) is None:
        import matgl

        potential = _NAMED_POTENTIALS[key] = _from_matgl(matgl.load_model(path=path))
    return LAMMPSCalculator.from_potential(potential)


def _from_matgl(potential: Any) -> LAMMPSPotential:
    """Export a MatGL ``Potential`` to TorchScript for MatGL's ``pair_style matgl``."""
    if (cached := _POTENTIAL_CACHE.get(potential)) is not None:
        return cached
    try:
        from matgl.ext.lammps import export_lammps_model
    except ImportError as err:
        raise ImportError("The LAMMPS backend needs a MatGL version that provides matgl.ext.lammps.") from err

    # Export in the precision the model runs in under ASE, so that both backends agree.
    dtype = next(potential.parameters()).dtype
    fd, path = tempfile.mkstemp(suffix=".pt", prefix="matgl-", dir=_scratch_dir())
    os.close(fd)
    export_lammps_model(potential, path, dtype=dtype)
    lmp_potential = LAMMPSPotential(
        commands=("pair_style matgl", f'pair_coeff * * "{path}" {{symbols}}'),
        # Map every element of the model to a type, so one LAMMPS instance serves all compositions.
        elements=tuple(potential.model.element_types),
        # pair_style matgl requires newton on (ghost-atom force contributions).
        pre_commands=("newton on",),
    )
    _POTENTIAL_CACHE[potential] = lmp_potential
    return lmp_potential


def _from_maml(potential: Any) -> LAMMPSPotential:
    """Build a LAMMPS force field from a maml ``Potential`` via ``write_param``."""
    if (cached := _POTENTIAL_CACHE.get(potential)) is not None:
        return cached
    directory = tempfile.mkdtemp(prefix="maml-", dir=_scratch_dir())
    # write_param writes the parameter files to the working directory and returns commands referring to them.
    with contextlib.chdir(directory):
        commands = potential.write_param()
    elements = getattr(potential, "elements", None)
    lmp_potential = LAMMPSPotential(
        commands=tuple(commands),
        elements=tuple(elements) if elements else None,
        directory=directory,
    )
    _POTENTIAL_CACHE[potential] = lmp_potential
    return lmp_potential


# --- LAMMPS instances -----------------------------------------------------------------------------------------------

_ENGINES: OrderedDict[tuple[LAMMPSPotential, tuple[str, ...]], _LAMMPSEngine] = OrderedDict()


def get_engine(potential: LAMMPSPotential, atoms: Atoms) -> _LAMMPSEngine:
    """Return a cached LAMMPS instance with ``potential`` loaded whose atom types cover the elements of ``atoms``."""
    if not atoms.pbc.all():
        raise ValueError("The LAMMPS backend requires structures that are periodic in all three directions.")
    symbols = atoms.get_chemical_symbols()
    elements = potential.elements or tuple(sorted(set(symbols), key=lambda s: atomic_numbers[s]))
    if missing := set(symbols) - set(elements):
        raise ValueError(f"Elements {sorted(missing)} are not supported by the LAMMPS force field ({elements}).")
    key = (potential, elements)
    engine = _ENGINES.get(key)
    if engine is None:
        engine = _ENGINES[key] = _LAMMPSEngine(potential, elements)
        while len(_ENGINES) > _MAX_ENGINES:
            _ENGINES.popitem(last=False)[1].close()
    else:
        _ENGINES.move_to_end(key)
    return engine


@dataclass(frozen=True)
class _Frame:
    """Maps quantities between the Cartesian frame and lattice basis of a structure and its LAMMPS representation.

    Attributes:
        rot: Orthogonal matrix; Cartesian vectors map as ``v_lammps = v @ rot``.
        basis: Integer unimodular matrix with ``cell_lammps = basis @ (cell @ rot)``.
        ref_frac: Fractional coordinates of the input structure, to which positions are unwrapped.
    """

    rot: np.ndarray
    basis: np.ndarray
    ref_frac: np.ndarray
    lammps_frac: np.ndarray  # wrapped fractional coordinates in the LAMMPS cell, as loaded

    def cell(self, cell_lammps: np.ndarray) -> np.ndarray:
        """LAMMPS cell -> cell in the input frame and lattice basis."""
        return np.linalg.solve(self.basis, cell_lammps) @ self.rot.T

    def vectors(self, vectors_lammps: np.ndarray) -> np.ndarray:
        """Per-atom vectors (forces, velocities) -> input frame."""
        return vectors_lammps @ self.rot.T

    def tensor(self, tensor_lammps: np.ndarray) -> np.ndarray:
        """Rank-2 tensor (stress) -> input frame."""
        return self.rot @ tensor_lammps @ self.rot.T

    def positions(
        self, x_lammps: np.ndarray, cell_lammps: np.ndarray, ref_frac: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Wrapped LAMMPS positions -> Cartesian positions and fractional coordinates in the input frame and basis.

        Each atom takes the periodic image closest to ``ref_frac`` (default: the input structure), which undoes the
        wrapping into the LAMMPS box.
        """
        cell = self.cell(cell_lammps)
        frac = np.linalg.solve(cell.T, self.vectors(x_lammps).T).T
        ref = self.ref_frac if ref_frac is None else ref_frac
        frac += np.round(ref - frac)
        return frac @ cell, frac


class _LAMMPSEngine:
    """A LAMMPS instance with a force field loaded, reused for any cell and number of atoms."""

    def __init__(self, potential: LAMMPSPotential, elements: tuple[str, ...]) -> None:
        try:
            from lammps import LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR, lammps
        except ImportError as err:
            raise ImportError(
                "The LAMMPS backend requires the LAMMPS Python module (`import lammps`), built with the packages "
                "for the force field in use, e.g. ML-MATGL for MatGL potentials."
            ) from err

        self._vector_args = (LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR)
        self.lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite", *LAMMPS_CMDARGS])
        self.kokkos = bool(self.lmp.extract_setting("kokkos_active"))
        self.elements = elements
        self.type_of = {el: i + 1 for i, el in enumerate(elements)}
        self.lmp.commands_list(
            [
                "units metal",
                "atom_style atomic",
                "boundary p p p",
                "atom_modify map yes",
                *potential.pre_commands,
                "region box prism 0 1 0 1 0 1 0 0 0",
                f"create_box {len(elements)} box",
                *(f"mass {i + 1} {atomic_masses[atomic_numbers[el]]}" for i, el in enumerate(elements)),
            ]
        )
        symbols = " ".join(elements)
        with contextlib.chdir(potential.directory) if potential.directory else contextlib.nullcontext():
            self.lmp.commands_list([cmd.replace("{symbols}", symbols) for cmd in potential.commands])
        # Virial-only pressure, so the stress excludes the kinetic contribution, as for ASE calculators.
        self.lmp.commands_list(
            [
                "compute matcalc_press all pressure NULL virial",
                "thermo_style custom step pe c_matcalc_press[*] lx ly lz xy xz yz",
                "thermo_modify norm no",
            ]
        )

    def load(self, atoms: Atoms, velocities: np.ndarray | None = None) -> _Frame:
        """Replace the atoms and box with ``atoms`` (velocities in Å/ps, input frame)."""
        cell = np.asarray(atoms.cell)
        rot, cell_lammps = lammps_cell(cell)
        basis = np.rint(cell_lammps @ np.linalg.inv(cell @ rot))
        frac = np.linalg.solve(cell_lammps.T, (atoms.positions @ rot).T).T
        frac -= np.floor(frac)
        positions = frac @ cell_lammps
        n_atoms = len(atoms)
        types = [self.type_of[s] for s in atoms.get_chemical_symbols()]

        lmp = self.lmp
        lmp.command("delete_atoms group all")
        lmp.reset_box(
            [0.0, 0.0, 0.0], np.diag(cell_lammps).tolist(), cell_lammps[1, 0], cell_lammps[2, 1], cell_lammps[2, 0]
        )
        v = None if velocities is None else (velocities @ rot).ravel().tolist()
        created = lmp.create_atoms(n_atoms, list(range(1, n_atoms + 1)), types, positions.ravel().tolist(), v=v)
        if created != n_atoms:
            raise RuntimeError(f"LAMMPS created {created} of {n_atoms} atoms.")
        return _Frame(rot, basis, atoms.cell.scaled_positions(atoms.positions), frac)

    def set_volume(self, volume: float) -> None:
        """Scale the cell (and atoms) isotropically to ``volume``."""
        cell = self.cell() * (volume / abs(np.linalg.det(self.cell()))) ** (1 / 3)
        self.lmp.command(
            f"change_box all x final 0 {cell[0, 0]} y final 0 {cell[1, 1]} z final 0 {cell[2, 2]} "
            f"xy final {cell[1, 0]} xz final {cell[2, 0]} yz final {cell[2, 1]} remap units box"
        )

    def pin(self, indices: Sequence[int], frame: _Frame) -> None:
        """Put atoms ``indices`` back on their loaded fractional coordinates in the current cell."""
        from ctypes import c_double

        x = self.gather("x")
        x[indices] = frame.lammps_frac[indices] @ self.cell()
        self.lmp.scatter_atoms("x", 1, 3, (c_double * x.size)(*x.ravel()))

    def compute(self, atoms: Atoms) -> tuple[float, np.ndarray, np.ndarray]:
        """Single point on ``atoms``; forces and stress are returned in the Cartesian frame of ``atoms``."""
        frame = self.load(atoms)
        self.lmp.command("run 0")
        return self.energy(), frame.vectors(self.gather("f")), frame.tensor(self.stress())

    def energy(self) -> float:
        """Potential energy (eV) of the last evaluated step."""
        return float(self.lmp.get_thermo("pe"))

    def stress(self) -> np.ndarray:
        """Virial stress (eV/Å^3, ASE sign convention) of the last evaluated step, in the LAMMPS frame."""
        press = self.lmp.extract_compute("matcalc_press", *self._vector_args)
        return -BAR_TO_EVA3 * _pressure_matrix([press[i] for i in range(6)])

    def gather(self, name: str) -> np.ndarray:
        """Per-atom 3-vectors (``x``, ``v`` or ``f``) ordered by atom ID, in the LAMMPS frame."""
        n_atoms = int(self.lmp.get_natoms())
        return np.ctypeslib.as_array(self.lmp.gather_atoms(name, 1, 3)).reshape(n_atoms, 3).copy()

    def cell(self) -> np.ndarray:
        """Current lower-triangular LAMMPS cell."""
        boxlo, boxhi, xy, yz, xz, *_ = self.lmp.extract_box()
        lx, ly, lz = np.subtract(boxhi, boxlo)
        return np.array([[lx, 0.0, 0.0], [xy, ly, 0.0], [xz, yz, lz]])

    def step(self) -> int:
        """Current timestep (minimizer iterations also advance it)."""
        return int(self.lmp.extract_global("ntimestep"))

    def close(self) -> None:
        """Release the LAMMPS instance."""
        self.lmp.close()


def lammps_cell(cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map a cell onto the restricted triclinic form required by LAMMPS.

    The lattice vectors (rows of ``cell``) are transformed by an orthogonal matrix ``rot`` into a lower-triangular cell
    with a positive diagonal, and the tilt factors are then reduced to ``|xy|, |xz| <= lx/2`` and ``|yz| <= ly/2``
    by integer combinations of lattice vectors. The reduced cell describes the same periodic system, so energies,
    per-atom forces and the stress are unchanged. A left-handed cell gives an improper ``rot``, which is fine for any
    O(3)-invariant potential.

    Args:
        cell: 3x3 matrix with lattice vectors as rows.

    Returns:
        ``(rot, lammps_cell)``: Cartesian vectors map as ``v_lammps = v @ rot``, and ``lammps_cell`` is the reduced
        lower-triangular cell in that frame.
    """
    q, r = np.linalg.qr(cell.T)
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    rot = q * signs
    a, b, c = (r * signs[:, None]).T
    b = b - np.round(b[0] / a[0]) * a
    c = c - np.round(c[1] / b[1]) * b
    c = c - np.round(c[0] / a[0]) * a
    lower = np.array([a, b, c])
    lower[np.triu_indices(3, 1)] = 0.0
    return rot, lower
