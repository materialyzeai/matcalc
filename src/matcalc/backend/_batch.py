"""Batched force evaluation for many structures in one model call.

Finite-displacement workflows (phonopy, phono3py) evaluate forces on many independent supercells. Calling an ASE
calculator once per supercell leaves a GPU mostly idle on small cells, because each call pays the full kernel-launch
and graph-construction overhead. Models that accept batched graphs can instead evaluate several supercells per
forward pass.

Each supported provider has an adapter: a ``supports`` check and a ``forces`` function that builds the provider's own
batched input from the calculator's settings and calls its model directly. The checks look at the calculator's module
before importing anything, so that probing a calculator never imports another provider's stack (e.g. TensorFlow).
Supported: MatGL ``PESCalculator``, MACE ``MACECalculator`` and GRACE ``TPCalculator``. Every other calculator, and
configurations an adapter does not handle, use the serial path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from matcalc.utils import to_ase_atoms

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure

    BatchedForcesFn = Callable[[Sequence[Structure | Atoms], Calculator, int], list[np.ndarray]]


def _from_module(calculator: Calculator, prefix: str) -> bool:
    """Whether the calculator's class is defined in the package ``prefix``."""
    module = type(calculator).__module__
    return module == prefix or module.startswith(prefix + ".")


def _split_by_atoms(forces: np.ndarray, atoms: Sequence[Atoms]) -> list[np.ndarray]:
    """Split forces concatenated over a batch into one array per structure."""
    return np.split(forces[: sum(len(a) for a in atoms)], np.cumsum([len(a) for a in atoms])[:-1])


def _chunks(structures: Sequence[Structure | Atoms], batch_size: int) -> list[list[Atoms]]:
    """Consecutive batches of at most ``batch_size`` structures, as ASE atoms."""
    return [
        [to_ase_atoms(s) for s in structures[start : start + batch_size]]
        for start in range(0, len(structures), batch_size)
    ]


# MatGL


def _matgl_supports(calculator: Calculator) -> bool:
    """MatGL ``PESCalculator``, unless it predicts charges (needs a total charge per structure) or the Hessian."""
    if not _from_module(calculator, "matgl"):
        return False
    from matgl.ext.ase import PESCalculator

    return isinstance(calculator, PESCalculator) and not (calculator.compute_charge or calculator.compute_hessian)


def matgl_batched_forces(
    structures: Sequence[Structure | Atoms], calculator: Calculator, batch_size: int
) -> list[np.ndarray]:
    """Forces from a MatGL potential, evaluating ``batch_size`` structures per forward pass.

    Args:
        structures: Periodic structures.
        calculator: MatGL ``PESCalculator``.
        batch_size: Number of structures per forward pass.

    Returns:
        Forces (eV/Å), one ``(n_atoms, 3)`` array per structure, in input order.
    """
    import torch
    from matgl.ext.ase import Atoms2Graph
    from torch_geometric.data import Batch

    converter = Atoms2Graph(calculator.element_types, calculator.cutoff)  # type: ignore[attr-defined]
    potential = calculator.potential  # type: ignore[attr-defined]
    fixed_state = calculator.state_attr  # type: ignore[attr-defined]
    forces: list[np.ndarray] = []
    for atoms in _chunks(structures, batch_size):
        graphs, lattices, states = zip(*(converter.get_graph(a) for a in atoms), strict=True)
        batch = Batch.from_data_list(list(graphs))
        lattice = torch.cat([torch.as_tensor(lat).reshape(-1, 3, 3) for lat in lattices])
        if fixed_state is not None:
            state = torch.as_tensor(fixed_state).reshape(1, -1).expand(len(atoms), -1)
        else:
            state = torch.stack([torch.as_tensor(s) for s in states])
        # Potential.forward returns forces for all atoms of the batch, concatenated in graph order.
        batch_forces = potential(batch, lattice, state)[1].detach().cpu().numpy()
        forces.extend(_split_by_atoms(batch_forces, atoms))
    return forces


# MACE


def _mace_supports(calculator: Calculator) -> bool:
    """A plain MACE energy model (``model_type="MACE"``), not compiled and without an external field.

    ``torch.compile`` mode pads every input to a fixed size and is skipped, as are subclasses (e.g. magnetic MACE)
    whose models take extra inputs.
    """
    if not _from_module(calculator, "mace"):
        return False
    from mace.calculators import MACECalculator

    calc: Any = calculator
    return (
        type(calc) is MACECalculator
        and calc.model_type == "MACE"
        and not calc.use_compile
        and getattr(calc, "external_field", None) is None
    )


def mace_batched_forces(
    structures: Sequence[Structure | Atoms], calculator: Calculator, batch_size: int
) -> list[np.ndarray]:
    """Forces from a MACE model, evaluating ``batch_size`` structures per forward pass.

    Mirrors ``MACECalculator.calculate``: the graphs are built with the calculator's key specification, head, atomic
    number table and cutoff, and the forces of a committee of models are averaged.

    Args:
        structures: Periodic structures.
        calculator: MACE ``MACECalculator``.
        batch_size: Number of structures per forward pass.

    Returns:
        Forces (eV/Å), one ``(n_atoms, 3)`` array per structure, in input order.
    """
    from mace import data as mace_data
    from mace.tools import torch_geometric, torch_tools

    calc: Any = calculator
    keyspec = mace_data.KeySpecification(
        info_keys=calc.info_keys, arrays_keys={**calc.arrays_keys, "charges": calc.charges_key}
    )
    unit = calc.energy_units_to_eV / calc.length_units_to_A
    forces: list[np.ndarray] = []
    for atoms in _chunks(structures, batch_size):
        with torch_tools.default_dtype(calc.default_dtype):
            graphs = [
                mace_data.AtomicData.from_config(
                    mace_data.config_from_atoms(a, key_specification=keyspec, head_name=calc.head),
                    z_table=calc.z_table,
                    cutoff=calc.r_max,
                    heads=calc.available_heads,
                )
                for a in atoms
            ]
        batch = torch_geometric.Batch.from_data_list(graphs).to(calc.device)
        total: Any = None
        for model in calc.models:
            model_batch = batch.clone()
            model_dtype = next(model.parameters()).dtype
            for key in model_batch.keys:
                value = model_batch[key]
                if value is not None and hasattr(value, "is_floating_point") and value.is_floating_point():
                    model_batch[key] = value.to(dtype=model_dtype)
            with torch_tools.default_dtype(calc.default_dtype):
                out = model(model_batch.to_dict(), compute_stress=False, training=False)
            model_forces = out["forces"].detach()
            total = model_forces if total is None else total + model_forces
        batch_forces = (total / len(calc.models)).cpu().numpy() * unit
        forces.extend(_split_by_atoms(batch_forces, atoms))
    return forces


# GRACE


def _grace_supports(calculator: Calculator) -> bool:
    """A GRACE ``TPCalculator`` in the default segment-sum ("diverse") mode.

    The dense ("uniform") engine uses a per-atom-uniform neighbor layout that is padded per structure, so it is left
    to the serial path.
    """
    if not _from_module(calculator, "tensorpotential"):
        return False
    from tensorpotential.calculator.asecalculator import TPCalculator

    return isinstance(calculator, TPCalculator) and not calculator.dense_reshape


def grace_batched_forces(
    structures: Sequence[Structure | Atoms], calculator: Calculator, batch_size: int
) -> list[np.ndarray]:
    """Forces from a GRACE model, evaluating ``batch_size`` structures per call.

    The structures of a batch are joined with the calculator's data builders, as in GRACE training, and padded with
    one fake structure. GRACE models are compiled for fixed input shapes, so all batches of a call share one padded
    shape: the bound is set by the first batch with 5% neighbor headroom and only grows if a later batch exceeds it.
    The first call therefore pays one compilation for that shape.

    Args:
        structures: Periodic structures.
        calculator: GRACE ``TPCalculator``.
        batch_size: Number of structures per call.

    Returns:
        Forces (eV/Å), one ``(n_atoms, 3)`` array per structure, in input order.

    Raises:
        ValueError: If a structure contains an element the model does not support.
    """
    from tensorflow.data import Dataset
    from tensorpotential import constants
    from tensorpotential.data.databuilder import get_number_of_real_atoms, get_number_of_real_neigh

    calc: Any = calculator
    n_structures = min(batch_size, len(structures)) + 1  # + 1 fake structure that holds the padding atoms
    bound: tuple[int, int] | None = None
    forces: list[np.ndarray] = []
    for atoms in _chunks(structures, batch_size):
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
        if bound is None or n_atoms >= bound[0] or n_neigh > bound[1]:
            bound = (n_atoms + 1, int(np.ceil(n_neigh * 1.05)) + 1)
        pad = {
            constants.PAD_MAX_N_STRUCTURES: n_structures,
            constants.PAD_MAX_N_ATOMS: bound[0],
            constants.PAD_MAX_N_NEIGHBORS: bound[1],
        }
        for builder in calc.data_builders:
            builder.pad_batch(batch, pad)
        inputs = Dataset.from_tensors({k: v for k, v in batch.items() if k in calc.data_keys}).get_single_element()

        batch_forces = np.mean(
            [model.compute(inputs)[constants.PREDICT_FORCES].numpy() for model in calc.models], axis=0
        )
        forces.extend(_split_by_atoms(batch_forces, atoms))
    return forces


class _Adapter(NamedTuple):
    supports: Callable[[Calculator], bool]
    forces: BatchedForcesFn


_ADAPTERS: tuple[_Adapter, ...] = (
    _Adapter(_matgl_supports, matgl_batched_forces),
    _Adapter(_mace_supports, mace_batched_forces),
    _Adapter(_grace_supports, grace_batched_forces),
)


def batched_forces_fn(calculator: Calculator) -> BatchedForcesFn | None:
    """The batched force function for ``calculator``, or None if it has none.

    Args:
        calculator: ASE calculator.

    Returns:
        A function ``(structures, calculator, batch_size) -> list of forces``, or None.
    """
    for adapter in _ADAPTERS:
        try:
            if adapter.supports(calculator):
                return adapter.forces
        except ImportError:
            continue
    return None


def supports_batched_forces(calculator: Calculator) -> bool:
    """Whether ``calculator`` can evaluate forces on a batch of structures in one model call.

    Args:
        calculator: ASE calculator.

    Returns:
        Whether a batched path applies (see the module docstring for the supported calculators).
    """
    return batched_forces_fn(calculator) is not None
