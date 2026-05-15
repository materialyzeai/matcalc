"""Define basic API."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

from joblib import Parallel, delayed

from .utils import PESCalculator

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure


class PropCalc(abc.ABC):
    """
    Abstract base class for property calculations.

    This class defines the interface for performing property calculations on
    structures (using pymatgen's Structure objects or a dictionary containing a
    pymatgen structure). Subclasses are expected to implement the `calc` method
    to define specific property calculation logic. Additionally, this class provides
    an implementation of the `calc_many` method, which enables concurrent calculations
    on multiple structures using joblib.
    """

    @property
    def calculator(self) -> Calculator:
        """
        Provides access to the `Calculator` instance used internally.

        This property retrieves an instance of the `Calculator` class, which is
        associated with the current object and utilized internally for
        performing operations or calculations. The property ensures encapsulation
        and controlled access to the underlying calculator.

        Returns:
            The internal ASE calculator instance.
        """
        return self._pes_calculator

    @calculator.setter
    def calculator(self, val: str | Calculator) -> None:
        """
        Sets the calculator property for the current object. If a string is provided, it
        loads the universal PESCalculator using the given value. Otherwise, it sets
        the provided Calculator instance.

        Args:
            val: Universal model name string or an existing ASE calculator instance.
        """
        self._pes_calculator = PESCalculator.load_universal(val) if isinstance(val, str) else val

    @staticmethod
    def _merge_units(
        prev: dict[str, Any] | None,
        new: dict[str, str],
    ) -> dict[str, str]:
        """Merge unit annotations, preserving upstream entries.

        Use this when building the ``_units`` field on a result dict that may
        already contain ``_units`` from an earlier step in a chain. Keys present
        in ``new`` overwrite the upstream value (the calc emitting them owns the
        unit string for those fields).

        Args:
            prev: The upstream result dict (or its ``_units`` value), or ``None``.
            new: The units this calc is contributing.

        Returns:
            A merged dict suitable for use as the ``_units`` field.
        """
        base: dict = prev.get("_units", prev) if isinstance(prev, dict) else {}
        return {**base, **new}

    @abc.abstractmethod
    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict[str, Any]:
        """
        Abstract method to calculate and process a given structure.

        Processes a provided structure in the form of a pymatgen `Structure`, ASE `Atoms`,
        or a dictionary with specific keys, and returns a processed dictionary. If the input
        structure is a dictionary, it must contain the key `final_structure` or `structure`;
        otherwise, a `ValueError` is raised. The returned dictionary includes the key
        `final_structure` pointing to the provided or derived structure.

        Args:
            structure: A `Structure`, `Atoms`, or `dict` containing structural data to be
                processed. If a dictionary is provided, it must include either `final_structure`
                or `structure` keys.

        Returns:
            A dictionary with the key `final_structure`, representing the provided structure
            or the structure derived from the dictionary input.

        Raises:
            ValueError: If the provided dictionary does not contain the required keys `final_structure`
                or `structure`.
        """
        if isinstance(structure, dict):
            if "final_structure" in structure:
                # Return a shallow copy so downstream ``result |= {...}`` updates
                # do not mutate the caller's input dict.
                return dict(structure)
            if "structure" in structure:
                return structure | {"final_structure": structure["structure"]}

            raise ValueError(
                "Structure must be either a pymatgen Structure, ASE Atoms, or a dict containing a Structure/Atoms in "
                "the final_structure or structure"
            )
        return {"final_structure": structure}

    def calc_many(
        self,
        structures: Sequence[Structure | dict[str, Any] | Atoms],
        n_jobs: None | int = None,
        allow_errors: bool = False,  # noqa: FBT001,FBT002
        **kwargs: Any,
    ) -> Generator[dict | None, None, None]:
        """
        Calculates multiple structures in parallel with optional error tolerance.

        This method processes a sequence of structures, allowing them to be calculated in
        parallel. It offers the ability to handle errors on a per-structure basis. If error
        tolerance is enabled, calculation failures for individual structures result in returning
        `None` for those structures, without raising exceptions.

        Args:
            structures: Sequence of structures or structure dicts.
            n_jobs: ``joblib`` worker count; ``None`` uses joblib defaults.
            allow_errors: If True, failed structures yield ``None`` instead of raising.
            **kwargs: Forwarded to ``joblib.Parallel``.

        Yields:
            Result dict from ``calc``, or ``None`` when ``allow_errors`` and a structure fails.

        Raises:
            Exception: Any error from ``calc`` when ``allow_errors`` is False.
        """
        parallel = Parallel(n_jobs=n_jobs, return_as="generator", **kwargs)

        def _func(s: Structure | Atoms) -> dict | None:
            try:
                return self.calc(s)
            except Exception:
                if allow_errors:
                    return None
                raise

        return parallel(delayed(_func)(s) for s in structures)


class ChainedCalc(PropCalc):
    """A chained calculator that runs a series of PropCalcs on a structure or set of structures.

    Often, you may want to obtain multiple properties at once, e.g., perform a relaxation followed by a formation
    energy computation and an elasticity calculation. This can be done by supplying a list of calculators. Each
    calculator receives the full result dict from the previous step, so downstream calculators can skip redundant
    relaxations by setting ``relax_structure=False``.
    """

    def __init__(self, prop_calcs: Sequence[PropCalc]) -> None:
        """
        Initialize a chained calculator.

        Args:
            prop_calcs: Sequence of prop calcs.
        """
        self.prop_calcs = tuple(prop_calcs)

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict[str, Any]:
        """
        Runs the series of PropCalcs on a structure.

        Args:
            structure: Pymatgen structure or a dict containing a pymatgen Structure under a "final_structure" or
                "structure" key. Allowing dicts provide the means to chain calculators, e.g., do a relaxation followed
                by an elasticity calculation.

        Returns:
            Merged dict from all steps (final keys depend on the chain).
        """
        results = structure  # type:ignore[assignment]
        for prop_calc in self.prop_calcs:
            results = prop_calc.calc(results)  # type:ignore[assignment]
        return results  # type:ignore[return-value]

    def calc_many(
        self,
        structures: Sequence[Structure | Atoms | dict[str, Any]],
        n_jobs: None | int = None,
        allow_errors: bool = False,  # noqa: FBT001,FBT002
        **kwargs: Any,
    ) -> Generator[dict | None, None, None]:
        """Runs the sequence of PropCalc on many structures.

        Args:
            structures: List or generator of Structures.
            n_jobs: The maximum number of concurrently running jobs. If -1 all CPUs are used. For n_jobs below -1,
                (n_cpus + 1 + n_jobs) are used. None is a marker for `unset` that will be interpreted as n_jobs=1
                unless the call is performed under a parallel_config() context manager that sets another value for
                n_jobs.
            allow_errors: Whether to skip failed calculations. For these calculations, None will be returned. For
                large scale calculations, you may want this to be True to avoid the entire calculation failing.
                Defaults to False.
            **kwargs: Passthrough to calc_many method of all PropCalcs.

        Returns:
            Generator of dicts.
        """
        results = structures  # type:ignore[assignment]
        for prop_calc in self.prop_calcs:
            results = prop_calc.calc_many(results, n_jobs=n_jobs, allow_errors=allow_errors, **kwargs)  # type:ignore[assignment]
        return results  # type:ignore[return-value]
