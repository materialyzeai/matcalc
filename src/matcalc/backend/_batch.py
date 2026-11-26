"""Plugin interface for evaluating many structures in one model call.

Finite-displacement workflows (phonopy, phono3py) and other static evaluations of many independent structures call an
ASE calculator once per structure, which leaves a GPU mostly idle on small cells. A potential can instead provide a
*batch adapter* that evaluates a list of structures in one forward pass.

A calculator gets batched evaluation in one of two ways, checked in this order:

1. **Natively.** The calculator has a ``calculate_batch(atoms, properties)`` method with the semantics of
   :meth:`BatchAdapter.calculate`. Nothing has to be registered.
2. **Through a registered adapter factory.** :func:`register_batch_adapter` maps a calculator class to a factory that
   returns a :class:`BatchAdapter` for a calculator instance, or None to decline it (e.g. an unsupported configuration).
   Later registrations take precedence, so a package can override matcalc's built-in adapters. Packages can register
   at import time or through the ``matcalc.batch_adapters`` entry-point group, whose entries are loaded on the first
   lookup and called without arguments::

       [project.entry-points."matcalc.batch_adapters"]
       mypotential = "mypotential.matcalc_plugin:register"

matcalc ships adapters for MatGL, MACE and GRACE (:mod:`matcalc.backend.batch_adapters`). Splitting structures into
batches is done by the caller (:func:`matcalc.backend.calc_batch`); an adapter only evaluates the batch it is given.
"""

from __future__ import annotations

import logging
import sys
from importlib import import_module
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ase import Atoms
    from ase.calculators.calculator import Calculator

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "matcalc.batch_adapters"


@runtime_checkable
class BatchAdapter(Protocol):
    """Evaluates batches of structures with one calculator's model.

    An adapter is created once per batched run (one call of :func:`matcalc.backend.calc_batch`) and called once per
    batch, so it can keep state across batches, such as a padded input shape that avoids recompilation.
    """

    implemented_properties: tuple[str, ...]
    """The properties :meth:`calculate` can return, from ``"energy"``, ``"forces"`` and ``"stress"``."""

    def calculate(self, atoms: Sequence[Atoms], properties: Sequence[str]) -> list[dict[str, Any]]:
        """Evaluate one batch.

        Args:
            atoms: Structures of the batch.
            properties: Requested properties, a subset of ``implemented_properties``.

        Returns:
            One dict per structure, in input order, with each requested property: ``"energy"`` (float, eV),
            ``"forces"`` (``(n_atoms, 3)``, eV/Å) and ``"stress"`` (``(3, 3)``, eV/Å³, ASE sign convention, i.e.
            positive is tensile).
        """
        ...  # pragma: no cover


# (target, factory) in registration order. A target is a class, or a "module:QualName" path resolved lazily.
_REGISTRY: list[tuple[type | str, Callable[[Calculator], BatchAdapter | None]]] = []
_entry_points_loaded = False


def register_batch_adapter(target: type | str, factory: Callable[[Calculator], BatchAdapter | None]) -> None:
    """Register a batch adapter factory for a calculator class and its subclasses.

    Args:
        target: Calculator class, or its import path as ``"package.module:ClassName"``. A path is resolved only once
            its module has been imported by someone else, so registering never imports the provider's stack.
        factory: Called with a calculator instance; returns a :class:`BatchAdapter`, or None to decline it (the
            next matching registration, or the serial path, is then used).
    """
    _REGISTRY.append((target, factory))


def _resolve(target: type | str) -> type | None:
    """The class for a registration target, or None if its module has not been imported."""
    if isinstance(target, type):
        return target
    module_name, _, qualname = target.partition(":")
    if module_name not in sys.modules:
        return None
    obj: Any = import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part, None)
    return obj if isinstance(obj, type) else None


def _load_entry_points() -> None:
    """Call the ``register`` hooks of installed plugins, once. A failing plugin is logged and skipped."""
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        try:
            entry_point.load()()
        except Exception:
            logger.warning("Failed to load matcalc batch adapter plugin %r", entry_point.name, exc_info=True)


class _NativeAdapter:
    """Adapter for a calculator that implements ``calculate_batch`` itself."""

    def __init__(self, calculator: Calculator) -> None:
        self.calculator = calculator
        properties: Any = getattr(calculator, "batch_properties", None) or calculator.implemented_properties
        self.implemented_properties: tuple[str, ...] = tuple(properties)

    def calculate(self, atoms: Sequence[Atoms], properties: Sequence[str]) -> list[dict[str, Any]]:
        return self.calculator.calculate_batch(atoms, properties)  # type: ignore[attr-defined]


def get_batch_adapter(calculator: Calculator) -> BatchAdapter | None:
    """The batch adapter for ``calculator``, or None if it has none.

    Args:
        calculator: ASE calculator.

    Returns:
        A new :class:`BatchAdapter` for this calculator, or None.
    """
    if callable(getattr(calculator, "calculate_batch", None)):
        return _NativeAdapter(calculator)
    _load_entry_points()
    for target, factory in reversed(_REGISTRY):
        cls = _resolve(target)
        if cls is None or not isinstance(calculator, cls):
            continue
        adapter = factory(calculator)
        if adapter is not None:
            return adapter
    return None


def supports_batched_forces(calculator: Calculator) -> bool:
    """Whether ``calculator`` can evaluate forces on a batch of structures in one model call.

    Args:
        calculator: ASE calculator.

    Returns:
        Whether a batch adapter applies and implements forces.
    """
    adapter = get_batch_adapter(calculator)
    return adapter is not None and "forces" in adapter.implemented_properties
