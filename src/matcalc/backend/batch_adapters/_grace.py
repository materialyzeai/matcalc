"""Batch adapter for GRACE (tensorpotential) models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._utils import split_results

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms
    from ase.calculators.calculator import Calculator


class GRACEBatchAdapter:  # pragma: no cover (tensorpotential is not in the main CI env; see test_backend.py)
    """Evaluates a batch with the calculator's data builders joining the structures, as in GRACE training.

    GRACE models are compiled for fixed input shapes, and ``TPCalculator``'s own padding holds one structure per call.
    This adapter pads each batch with one fake structure that holds the padding atoms, and keeps one padded shape
    (number of structures, atoms and neighbors, with 5% neighbor headroom) for all batches of a run, growing it only
    when a batch exceeds it. A run therefore pays one compilation for that shape. Declines the dense ("uniform") engine,
    whose per-atom-uniform neighbor layout is padded per structure.
    """

    implemented_properties: tuple[str, ...] = ("energy", "forces", "stress")

    def __init__(self, calculator: Calculator) -> None:
        """
        Args:
            calculator: GRACE ``TPCalculator``.
        """
        self.calc: Any = calculator
        self.n_structures = 0
        self.bound: tuple[int, int] | None = None  # (atoms, neighbors) of the padded shape

    @classmethod
    def from_calculator(cls, calculator: Calculator) -> GRACEBatchAdapter | None:
        """The adapter for ``calculator``, or None in dense mode."""
        if getattr(calculator, "dense_reshape", False):
            return None
        return cls(calculator)

    def calculate(self, atoms: Sequence[Atoms], properties: Sequence[str]) -> list[dict[str, Any]]:
        """Evaluate one batch. See :meth:`matcalc.backend.BatchAdapter.calculate`.

        Raises:
            ValueError: If a structure contains an element the model does not support.
        """
        from tensorflow.data import Dataset
        from tensorpotential import constants
        from tensorpotential.data.databuilder import get_number_of_real_atoms, get_number_of_real_neigh

        calc = self.calc
        for a in atoms:
            if unknown := set(a.symbols.species()) - set(calc.element_map):
                raise ValueError(f"Elements {sorted(unknown)} are not supported by this GRACE model.")
        data = []
        for a in atoms:
            entry: dict[str, Any] = {}
            for builder in calc.data_builders:
                entry.update(builder.extract_from_ase_atoms(a))
            data.append(entry)
        batch: dict[str, Any] = {}
        for builder in calc.data_builders:
            batch.update(builder.join_to_batch(data))

        n_atoms, n_neigh = get_number_of_real_atoms(batch), get_number_of_real_neigh(batch)
        self.n_structures = max(self.n_structures, len(atoms) + 1)  # + 1 fake structure for the padding atoms
        if self.bound is None or n_atoms >= self.bound[0] or n_neigh > self.bound[1]:
            self.bound = (n_atoms + 1, int(np.ceil(n_neigh * 1.05)) + 1)
        pad = {
            constants.PAD_MAX_N_STRUCTURES: self.n_structures,
            constants.PAD_MAX_N_ATOMS: self.bound[0],
            constants.PAD_MAX_N_NEIGHBORS: self.bound[1],
        }
        for builder in calc.data_builders:
            builder.pad_batch(batch, pad)
        inputs = Dataset.from_tensors({k: v for k, v in batch.items() if k in calc.data_keys}).get_single_element()

        # The model's total energy and virial are summed over the whole batch, so per-structure values are rebuilt from
        # the per-atom energies and the per-bond pair forces, segmented by structure.
        outputs = [model.compute(inputs) for model in calc.models]
        atom_structure = np.asarray(batch[constants.ATOMS_TO_STRUCTURE_MAP])
        atomic_energies = np.mean([o[constants.PREDICT_ATOMIC_ENERGY].numpy() for o in outputs], axis=0).reshape(-1)
        energies = np.bincount(atom_structure, weights=atomic_energies, minlength=self.n_structures)
        forces = np.mean([o[constants.PREDICT_FORCES].numpy() for o in outputs], axis=0)
        stresses = None
        if "stress" in properties:
            pair_key = "z_" + constants.PREDICT_PAIR_FORCES
            if any(pair_key not in o for o in outputs):
                raise RuntimeError(f"GRACE model outputs no {pair_key!r}, which per-structure stresses need.")
            pair_forces = np.mean([o[pair_key].numpy() for o in outputs], axis=0)
            bond_vectors = np.asarray(batch[constants.BOND_VECTOR])
            bond_structure = atom_structure[np.asarray(batch[constants.BOND_IND_I])]
            stresses = np.array(
                [
                    _structure_stress(
                        pair_forces[bond_structure == i], bond_vectors[bond_structure == i], a.get_volume()
                    )
                    for i, a in enumerate(atoms)
                ]
            )
        return split_results(atoms, properties, energies=energies, forces=forces, stresses=stresses)


def _structure_stress(  # pragma: no cover
    pair_forces: np.ndarray, bond_vectors: np.ndarray, volume: float
) -> np.ndarray:
    """ASE stress (3x3, eV/Å³) of one structure from its bonds, following ``TPCalculator``.

    The virial is GRACE's ``compute_structure_virials_from_pair_forces``: sum over bonds of ``f_a * r_b`` for (xx, yy,
    zz, xy, xz, yz) = (ab) in (00, 11, 22, 10, 20, 21). The stress is ``-virial / volume``.
    """
    f, r = pair_forces, bond_vectors
    xx, yy, zz = (f * r).sum(axis=0)
    xy, xz, yz = (f[:, 1] * r[:, 0]).sum(), (f[:, 2] * r[:, 0]).sum(), (f[:, 2] * r[:, 1]).sum()
    virial = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    return -virial / volume
