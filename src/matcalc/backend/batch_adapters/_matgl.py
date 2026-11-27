"""Batch adapter for MatGL potentials."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._utils import split_results

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms
    from ase.calculators.calculator import Calculator


class MatGLBatchAdapter:
    """Evaluates a batch as one PyG ``Batch`` through ``Potential.forward``, which accepts batched graphs natively.

    Declines potentials that predict charges (they need a total charge per structure) or the Hessian (whose size grows
    with the square of the batch).
    """

    def __init__(self, calculator: Calculator) -> None:
        """
        Args:
            calculator: MatGL ``PESCalculator``.
        """
        from matgl.ext.ase import Atoms2Graph

        calc: Any = calculator
        self.potential = calc.potential
        self.state_attr = calc.state_attr
        self.stress_factor = calc.conversion_factor
        self.converter = Atoms2Graph(calc.element_types, calc.cutoff)
        self.implemented_properties: tuple[str, ...] = (
            ("energy", "forces", "stress") if calc.compute_stress else ("energy", "forces")
        )

    @classmethod
    def from_calculator(cls, calculator: Calculator) -> MatGLBatchAdapter | None:
        """The adapter for ``calculator``, or None if it predicts charges or the Hessian."""
        calc: Any = calculator
        if calc.compute_charge or calc.compute_hessian:
            return None
        return cls(calculator)

    def calculate(self, atoms: Sequence[Atoms], properties: Sequence[str]) -> list[dict[str, Any]]:
        """Evaluate one batch. See :meth:`matcalc.backend.BatchAdapter.calculate`."""
        import torch
        from torch_geometric.data import Batch

        graphs, lattices, states = zip(*(self.converter.get_graph(a) for a in atoms), strict=True)
        batch = Batch.from_data_list(list(graphs))
        lattice = torch.cat([torch.as_tensor(lat).reshape(-1, 3, 3) for lat in lattices])
        if self.state_attr is not None:
            state = torch.as_tensor(self.state_attr).reshape(1, -1).expand(len(atoms), -1)
        else:
            state = torch.stack([torch.as_tensor(s) for s in states])
        # Shapes: energies (B,), forces (N, 3) concatenated in graph order, stresses (3B, 3) in GPa.
        energies, forces, stresses = (t.detach().cpu().numpy() for t in self.potential(batch, lattice, state)[:3])
        return split_results(
            atoms,
            properties,
            energies=energies,
            forces=forces,
            stresses=stresses.reshape(-1, 3, 3) * self.stress_factor if "stress" in properties else None,
        )
