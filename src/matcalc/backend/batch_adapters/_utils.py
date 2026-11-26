"""Helpers shared by the batch adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms


def split_results(
    atoms: Sequence[Atoms],
    properties: Sequence[str],
    energies: np.ndarray | None = None,
    forces: np.ndarray | None = None,
    stresses: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Per-structure result dicts from batch outputs.

    Args:
        atoms: Structures of the batch.
        properties: Requested properties.
        energies: Energies (eV), at least ``len(atoms)`` values; extra trailing entries (padding) are ignored.
        forces: Forces (eV/Å) of all atoms concatenated in batch order, ``(>= total atoms, 3)``.
        stresses: Stresses (eV/Å³, ASE sign), ``(>= len(atoms), 3, 3)``. Symmetrized, as ASE does.

    Returns:
        One dict per structure with the requested properties.
    """
    results: list[dict[str, Any]] = [{} for _ in atoms]
    if "energy" in properties and energies is not None:
        for result, energy in zip(results, np.asarray(energies).reshape(-1), strict=False):
            result["energy"] = float(energy)
    if "forces" in properties and forces is not None:
        bounds = np.cumsum([0, *(len(a) for a in atoms)])
        for i, result in enumerate(results):
            result["forces"] = np.asarray(forces[bounds[i] : bounds[i + 1]], dtype=np.float64)
    if "stress" in properties and stresses is not None:
        for result, stress in zip(results, np.asarray(stresses, dtype=np.float64), strict=False):
            result["stress"] = 0.5 * (stress + stress.T)
    return results
