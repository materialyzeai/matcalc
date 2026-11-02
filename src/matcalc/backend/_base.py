from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import numpy as np
    from pymatgen.core.structure import Structure


class SimulationResult(NamedTuple):
    """Container for results from PES calculators.

    ``converged`` is the optimizer's own convergence verdict when a relaxation was run (for a cell relaxation this
    covers the cell degrees of freedom, i.e. the stress, as well as the atomic forces) and ``None`` for a static
    calculation.
    """

    structure: Structure
    potential_energy: float
    kinetic_energy: float
    energy: float
    forces: np.typing.NDArray[np.float64]
    stress: np.typing.NDArray[np.float64]
    converged: bool | None = None
