"""LAMMPS engine shared by the static, relaxation and MD drivers of the LAMMPS backend (``_lammps.py``).

Calculations run in an in-process LAMMPS instance driven through the ``lammps`` Python module. The LAMMPS force field is
derived from the ASE calculator or model name in use:

* MatGL TensorNet / M3GNet potentials are exported to TorchScript with ``matgl.ext.lammps.export_lammps_model`` and
  evaluated with MatGL's own ``pair_style matgl``, which requires LAMMPS built with the ML-MATGL package. To run
  through ``pair_style matgl/kk``, build with Kokkos and set e.g.
  ``MATCALC_LAMMPS_ARGS="-k on g 1 -sf kk -pk kokkos newton on neigh half"``.
* maml potentials wrapped in ``matcalc.utils.PESCalculator`` (SNAP, MTP, GAP, NNP) use the pair commands from
  ``Potential.write_param``.
* :class:`LAMMPSCalculator` evaluates arbitrary LAMMPS force-field commands, e.g. EAM or Tersoff potentials.

LAMMPS instances (:class:`LAMMPSEngine`) are cached per force field and element set, so repeated calculations (phonon
displacements, strained cells, ...) do not reload the model. :class:`Frame` maps quantities between the input structure
and its restricted-triclinic LAMMPS representation.
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
from ase.data import atomic_masses, atomic_numbers
from ase.stress import full_3x3_to_voigt_6_stress

from matcalc.config import LAMMPS_CMDARGS
from matcalc.units import eVA3ToGPa
from matcalc.utils import PESCalculator, resolve_model

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

logger = logging.getLogger(__name__)

# LAMMPS ``metal`` units: pressure in bar (1 bar = 1e-4 GPa), time in ps, velocity in Å/ps.
BAR_TO_EVA3 = 1e-4 / eVA3ToGPa
ASE_TIME_TO_PS = 1e-3 / units.fs
ASE_VELOCITY_TO_LAMMPS = units.fs * 1e3
# Maximum number of live LAMMPS instances per process. Each holds a loaded force field (and, for MatGL, a model).
_MAX_ENGINES = 4


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


def scratch_dir() -> Path:
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
    fd, path = tempfile.mkstemp(suffix=".pt", prefix="matgl-", dir=scratch_dir())
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
    directory = tempfile.mkdtemp(prefix="maml-", dir=scratch_dir())
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

_ENGINES: OrderedDict[tuple[LAMMPSPotential, tuple[str, ...]], LAMMPSEngine] = OrderedDict()


def get_engine(potential: LAMMPSPotential, atoms: Atoms) -> LAMMPSEngine:
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
        engine = _ENGINES[key] = LAMMPSEngine(potential, elements)
        while len(_ENGINES) > _MAX_ENGINES:
            _ENGINES.popitem(last=False)[1].close()
    else:
        _ENGINES.move_to_end(key)
    return engine


@dataclass(frozen=True)
class Frame:
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


class LAMMPSEngine:
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

    def load(self, atoms: Atoms, velocities: np.ndarray | None = None) -> Frame:
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
        return Frame(rot, basis, atoms.cell.scaled_positions(atoms.positions), frac)

    @contextlib.contextmanager
    def scoped(self, setup: Sequence[str], cleanup: Sequence[str]) -> Iterator[None]:
        """Issue ``setup``, then ``cleanup`` on exit, also after an error, so that the cached instance stays reusable.

        Cleanup commands run one at a time and failures are logged, not raised: after a failed setup command, cleanup
        for what was never set up (e.g. ``unfix`` of a fix that was not created) is expected to fail.
        """
        try:
            self.lmp.commands_list(list(setup))
            yield
        finally:
            for command in cleanup:
                try:
                    self.lmp.command(command)
                except Exception:
                    logger.debug("LAMMPS cleanup command failed: %s", command, exc_info=True)

    def set_volume(self, volume: float) -> None:
        """Scale the cell (and atoms) isotropically to ``volume``."""
        cell = self.cell() * (volume / abs(np.linalg.det(self.cell()))) ** (1 / 3)
        self.lmp.command(
            f"change_box all x final 0 {cell[0, 0]} y final 0 {cell[1, 1]} z final 0 {cell[2, 2]} "
            f"xy final {cell[1, 0]} xz final {cell[2, 0]} yz final {cell[2, 1]} remap units box"
        )

    def pin(self, indices: Sequence[int], frame: Frame) -> None:
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
        return -BAR_TO_EVA3 * pressure_matrix([press[i] for i in range(6)])

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


def pressure_matrix(p: Sequence[float]) -> np.ndarray:
    """3x3 pressure tensor from LAMMPS (xx, yy, zz, xy, xz, yz) components."""
    pxx, pyy, pzz, pxy, pxz, pyz = p
    return np.array([[pxx, pxy, pxz], [pxy, pyy, pyz], [pxz, pyz, pzz]])
