"""Batch adapters shipped with matcalc, registered like any third-party plugin.

Each adapter builds its provider's batched input from the calculator's own settings and calls the model directly. The
provider packages are imported only when an adapter is created, i.e. once a calculator of that provider exists.
"""

from __future__ import annotations

from matcalc.backend._batch import register_batch_adapter

from ._grace import GRACEBatchAdapter
from ._mace import MACEBatchAdapter
from ._matgl import MatGLBatchAdapter

__all__ = ["GRACEBatchAdapter", "MACEBatchAdapter", "MatGLBatchAdapter"]

register_batch_adapter("matgl.ext.ase:PESCalculator", MatGLBatchAdapter.from_calculator)
register_batch_adapter("mace.calculators:MACECalculator", MACEBatchAdapter.from_calculator)
register_batch_adapter("tensorpotential.calculator:TPCalculator", GRACEBatchAdapter.from_calculator)
