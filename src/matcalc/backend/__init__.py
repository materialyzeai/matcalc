"""Provides various backends of running simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matcalc.config import SIMULATION_BACKEND
from matcalc.utils import PESCalculator, to_ase_atoms

from ._ase import run_ase
from ._base import SimulationResult  # noqa: TC001
from ._batch import (
    BatchAdapter,
    get_batch_adapter,
    register_batch_adapter,
    supports_batched_forces,
)
from ._lammps import run_lammps, run_lammps_md
from ._lammps_engine import (
    LAMMPSCalculator,
    LAMMPSPotential,
    get_lammps_potential,
    load_lammps_calculator,
    to_lammps_calculator,
)
from .batch_adapters import GRACEBatchAdapter, MACEBatchAdapter, MatGLBatchAdapter

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure


def is_lammps_backend() -> bool:
    """Whether calculations run with the LAMMPS backend (``MATCALC_BACKEND`` is not ``ASE``)."""
    return SIMULATION_BACKEND != "ASE"


def to_backend_calculator(calculator: str | Calculator) -> Calculator:
    """Return the calculator to use with the active backend.

    Under the ASE backend, a model name is loaded with ``PESCalculator.load_universal`` and a calculator is returned
    unchanged. Under the LAMMPS backend the result is a :class:`LAMMPSCalculator`, so that the force field is resolved
    once (e.g. a MatGL model exported to TorchScript), unsupported models fail immediately, and code that evaluates the
    calculator directly through ASE also uses LAMMPS: a model name is built directly with
    :func:`~matcalc.backend._lammps_engine.load_lammps_calculator` (no ASE calculator is created), and a calculator
    object is converted with :func:`~matcalc.backend._lammps_engine.to_lammps_calculator`.

    Args:
        calculator: Model name or ASE calculator.

    Returns:
        Calculator for the active backend.

    Raises:
        TypeError: Under the LAMMPS backend, if the model or calculator has no LAMMPS representation.
    """
    if not is_lammps_backend():
        return PESCalculator.load_universal(calculator) if isinstance(calculator, str) else calculator
    if isinstance(calculator, str):
        return load_lammps_calculator(calculator)
    return to_lammps_calculator(calculator)


def run_pes_calc(*arg, **kwargs) -> SimulationResult:  # noqa:ANN002,ANN003
    """
    Executes the potential energy surface (PES) calculation using the appropriate backend.

    This function determines the backend to use for the PES calculation based on the
    environment variable `MATCALC_BACKEND`. If the variable is set to `"ASE"` (case-insensitive)
    or is not explicitly provided, the ASE backend is used. Otherwise, the LAMMPS backend is
    selected. The function then forwards the arguments and keyword arguments to the respective
    backend function for execution.

    Args:
        *arg: Positional arguments forwarded to the selected backend.
        **kwargs: Keyword arguments forwarded to the selected backend.

    Returns:
        SimulationResult from the ASE or LAMMPS backend.
    """
    if not is_lammps_backend():
        return run_ase(*arg, **kwargs)
    return run_lammps(*arg, **kwargs)


def calc_batch(
    structures: Sequence[Structure | Atoms],
    calculator: Calculator,
    *,
    properties: Sequence[str] = ("energy", "forces", "stress"),
    batch_size: int = 1,
) -> list[dict[str, Any]]:
    """Static energies, forces and stresses of many independent structures, batched through the model where possible.

    With ``batch_size > 1``, the ASE backend and a calculator with a batch adapter that implements the requested
    properties (see :mod:`matcalc.backend._batch`; matcalc ships adapters for MatGL, MACE and GRACE), the structures
    are evaluated ``batch_size`` at a time in one model call each. This mainly pays off on a GPU with small cells;
    very large batches can be slower because of memory pressure. Otherwise each structure goes through
    :func:`run_pes_calc`.

    Args:
        structures: Structures to evaluate.
        calculator: ASE calculator.
        properties: Properties to return, from ``"energy"``, ``"forces"`` and ``"stress"``.
        batch_size: Number of structures per model call. 1 evaluates them one at a time.

    Returns:
        One dict per structure, in input order, with each requested property: ``"energy"`` (eV), ``"forces"``
        (``(n_atoms, 3)``, eV/Å) and ``"stress"`` (``(3, 3)``, eV/Å³, ASE sign convention).

    Raises:
        ValueError: If ``batch_size`` is less than 1 or a property is unknown.
        RuntimeError: If a batch adapter returns the wrong number of results.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}.")
    if unknown := set(properties) - {"energy", "forces", "stress"}:
        raise ValueError(f"Unknown properties {sorted(unknown)}; choose from energy, forces and stress.")
    adapter = get_batch_adapter(calculator) if batch_size > 1 and not is_lammps_backend() else None
    if adapter is not None and set(properties) <= set(adapter.implemented_properties):
        atoms = [to_ase_atoms(s) for s in structures]
        results: list[dict[str, Any]] = []
        for start in range(0, len(atoms), batch_size):
            chunk = atoms[start : start + batch_size]
            batch_results = adapter.calculate(chunk, properties)
            if len(batch_results) != len(chunk):
                raise RuntimeError(
                    f"{type(adapter).__name__} returned {len(batch_results)} results for {len(chunk)} structures."
                )
            results.extend(batch_results)
        return results
    serial = (run_pes_calc(structure, calculator) for structure in structures)
    return [
        {
            key: value
            for key, value in (("energy", r.potential_energy), ("forces", r.forces), ("stress", r.stress))
            if key in properties
        }
        for r in serial
    ]


def calc_forces(
    structures: Sequence[Structure | Atoms], calculator: Calculator, *, batch_size: int = 1
) -> list[np.ndarray]:
    """Static forces on many independent structures, batched through the model where possible.

    Shorthand for :func:`calc_batch` with ``properties=("forces",)``.

    Args:
        structures: Structures to evaluate.
        calculator: ASE calculator.
        batch_size: Number of structures per model call. 1 evaluates them one at a time.

    Returns:
        Forces (eV/Å), one ``(n_atoms, 3)`` array per structure, in input order.
    """
    return [r["forces"] for r in calc_batch(structures, calculator, properties=("forces",), batch_size=batch_size)]
