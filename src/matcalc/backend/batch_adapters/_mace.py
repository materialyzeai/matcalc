"""Batch adapter for MACE models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._utils import split_results

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms
    from ase.calculators.calculator import Calculator


class MACEBatchAdapter:
    """Evaluates a batch as one MACE ``Batch``, mirroring ``MACECalculator.calculate``.

    The graphs are built with the calculator's key specification, head, atomic number table and cutoff, and the outputs
    of a committee of models are averaged. Declines subclasses (e.g. magnetic MACE, whose models take extra inputs),
    model types other than ``"MACE"``, ``torch.compile`` mode (which pads every input to a fixed size) and calculators
    with an external field.
    """

    def __init__(self, calculator: Calculator) -> None:
        """
        Args:
            calculator: MACE ``MACECalculator``.
        """
        from mace import data as mace_data

        calc: Any = calculator
        self.calc = calc
        self.keyspec = mace_data.KeySpecification(
            info_keys=calc.info_keys, arrays_keys={**calc.arrays_keys, "charges": calc.charges_key}
        )
        self.energy_unit = calc.energy_units_to_eV
        self.length_unit = calc.length_units_to_A
        self.implemented_properties: tuple[str, ...] = (
            ("energy", "forces", "stress") if calc.compute_stress else ("energy", "forces")
        )

    @classmethod
    def from_calculator(cls, calculator: Calculator) -> MACEBatchAdapter | None:
        """The adapter for ``calculator``, or None for a configuration it does not handle."""
        from mace.calculators import MACECalculator

        calc: Any = calculator
        if (
            type(calc) is not MACECalculator
            or calc.model_type != "MACE"
            or calc.use_compile
            or getattr(calc, "external_field", None) is not None
        ):
            return None
        return cls(calculator)

    def calculate(self, atoms: Sequence[Atoms], properties: Sequence[str]) -> list[dict[str, Any]]:
        """Evaluate one batch. See :meth:`matcalc.backend.BatchAdapter.calculate`."""
        from mace import data as mace_data
        from mace.tools import torch_geometric, torch_tools

        calc = self.calc
        with torch_tools.default_dtype(calc.default_dtype):
            graphs = [
                mace_data.AtomicData.from_config(
                    mace_data.config_from_atoms(a, key_specification=self.keyspec, head_name=calc.head),
                    z_table=calc.z_table,
                    cutoff=calc.r_max,
                    heads=calc.available_heads,
                )
                for a in atoms
            ]
        batch = torch_geometric.Batch.from_data_list(graphs).to(calc.device)
        compute_stress = "stress" in properties
        totals: dict[str, Any] = {}
        for model in calc.models:
            model_batch = batch.clone()
            model_dtype = next(model.parameters()).dtype
            for key in model_batch.keys:
                value = model_batch[key]
                if value is not None and hasattr(value, "is_floating_point") and value.is_floating_point():
                    model_batch[key] = value.to(dtype=model_dtype)
            with torch_tools.default_dtype(calc.default_dtype):
                out = model(model_batch.to_dict(), compute_stress=compute_stress, training=False)
            for key in ("energy", "forces", "stress"):
                if out.get(key) is not None:
                    value = out[key].detach()
                    totals[key] = value if key not in totals else totals[key] + value
        mean = {key: (value / len(calc.models)).cpu().numpy() for key, value in totals.items()}
        return split_results(
            atoms,
            properties,
            energies=mean["energy"] * self.energy_unit,
            forces=mean["forces"] * self.energy_unit / self.length_unit,
            stresses=mean["stress"] * self.energy_unit / self.length_unit**3 if compute_stress else None,
        )
