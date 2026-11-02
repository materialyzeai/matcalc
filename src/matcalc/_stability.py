"""Calculator for stability related properties."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from monty.serialization import loadfn

from ._base import PropCalc
from .backend import run_pes_calc
from .utils import _resolve_canonical, to_pmg_structure

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Element, Species, Structure

ELEMENTAL_REFS_DIR = Path(__file__).parent / "elemental_refs"


def _infer_elemental_refs(calculator: Calculator | str) -> Literal["MatPES-PBE", "MatPES-r2SCAN"]:
    """Pick the builtin MatPES elemental reference set matching the calculator's training functional.

    The functional is read from the ``-r2SCAN`` / ``-PBE`` token of the model name (resolved to its canonical name
    first, so aliases such as ``"r2scan"`` work). ``"MP-PBE"`` is never inferred; pass it explicitly.

    Args:
        calculator: ASE calculator instance or model name string, as passed to ``EnergeticsCalc``.

    Returns:
        ``"MatPES-r2SCAN"`` or ``"MatPES-PBE"``. Falls back to ``"MatPES-PBE"`` with a ``UserWarning`` when the
        functional cannot be determined (a Calculator instance, or a name without a PBE/r2SCAN token).
    """
    if isinstance(calculator, str):
        name = _resolve_canonical(calculator) or calculator
        tokens = {t.lower() for t in Path(name).name.split("-")}
        if "r2scan" in tokens:
            return "MatPES-r2SCAN"
        if "pbe" in tokens:  # Exact token, so PBEsol models are not matched.
            return "MatPES-PBE"
        source = f"model {name!r}"
    else:
        source = f"a {type(calculator).__name__} instance"
    warnings.warn(
        f"Cannot infer the DFT functional of {source}; using the 'MatPES-PBE' elemental references. Cohesive "
        "energies (and formation energies with use_gs_reference=True) combine the MLIP energy with these DFT "
        "references, so pass elemental_refs='MatPES-r2SCAN', 'MatPES-PBE' or 'MP-PBE' to match the potential's "
        "training functional.",
        UserWarning,
        stacklevel=3,
    )
    return "MatPES-PBE"


class EnergeticsCalc(PropCalc):
    """
    Handles formation energy per atom, cohesive energy per atom, and relaxed structures
    using reference elemental data and optional relaxation.

    Attributes:
        calculator: ASE calculator (or universal model name) for energies and forces.
        elemental_refs: Elemental reference data (loaded from the builtin set inferred from, or named by, the
            ``elemental_refs`` argument, or the custom dict passed).
        use_gs_reference: If True, use tabulated DFT ground-state energies from refs.
        relax_structure: Whether to relax the input structure before energetics.
        relax_calc_kwargs: Optional kwargs forwarded to ``RelaxCalc``.
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        elemental_refs: Literal["MP-PBE", "MatPES-PBE", "MatPES-r2SCAN"] | dict | None = None,
        fmax: float = 0.1,
        optimizer: str = "FIRE",
        perturb_distance: float | None = None,
        use_gs_reference: bool = False,
        relax_structure: bool = True,
        relax_calc_kwargs: dict | None = None,
    ) -> None:
        """
        Args:
            calculator: ASE calculator or universal model name string.
            elemental_refs: ``"MP-PBE"``, ``"MatPES-PBE"``, ``"MatPES-r2SCAN"``, a custom dict, or None (default).
                None infers the set from the model name: ``"MatPES-r2SCAN"`` for r2SCAN-trained models and
                ``"MatPES-PBE"`` for PBE-trained models. If the functional cannot be inferred (an ASE Calculator
                instance, or a name without a PBE/r2SCAN token), ``"MatPES-PBE"`` is used and a ``UserWarning``
                is emitted. The DFT ``energy_atomic`` (cohesive energy) and ``energy_per_atom``
                (``use_gs_reference=True``) entries are only meaningful if they match the potential's training
                functional. The default used to be ``"MatPES-PBE"`` regardless of the model; pass it explicitly to
                keep that behavior.
            fmax: Force convergence tolerance (eV/Å) for relaxation.
            optimizer: ASE optimizer name for relaxation.
            perturb_distance: Random displacement (Å) before relaxation, or None.
            use_gs_reference: Use tabulated ground-state energies from references when True.
            relax_structure: Relax input structure before computing energetics.
            relax_calc_kwargs: Optional kwargs for ``RelaxCalc``.
        """
        self.calculator = calculator  # type: ignore[assignment]
        self.fmax = fmax
        self.optimizer = optimizer
        self.perturb_distance = perturb_distance
        if elemental_refs is None:
            elemental_refs = _infer_elemental_refs(calculator)
        if isinstance(elemental_refs, str):
            self.elemental_refs = loadfn(ELEMENTAL_REFS_DIR / f"{elemental_refs}-Element-Refs.json.gz")
        else:
            self.elemental_refs = elemental_refs
        self.use_gs_reference = use_gs_reference
        self.relax_structure = relax_structure
        self.relax_calc_kwargs = relax_calc_kwargs
        # Relaxed elemental reference energies, keyed by element and the relaxation settings, so that
        # repeated ``calc`` calls (e.g. ``calc_many``) do not re-relax the same reference structures.
        self._ref_energy_cache: dict[tuple, float] = {}

    def _relaxed_ref_energy_per_atom(self, symbol: str) -> float:
        """Lowest energy per atom of the relaxed reference structure(s) for an element.

        Results are cached per element, calculator, reference set and relaxation settings.

        Args:
            symbol: Element symbol.

        Returns:
            Minimum energy per atom (eV/atom) over the relaxed reference structures.

        Raises:
            RuntimeError: If no reference relaxation succeeded.
        """
        key = (
            symbol,
            id(self.calculator),
            id(self.elemental_refs),
            self.fmax,
            str(self.optimizer),
            repr(self.relax_calc_kwargs),
        )
        if key in self._ref_energy_cache:
            return self._ref_energy_cache[key]

        relaxer = self._relaxer(fmax=self.fmax, optimizer=self.optimizer)
        # References are always relaxed from their tabulated geometry, without perturbation.
        relaxer.perturb_distance = None
        structure = self.elemental_refs[symbol]["structure"]
        structure_list = [structure] if not isinstance(structure, list) else structure
        energies_per_atom = [
            r["energy"] / r["final_structure"].num_sites for r in relaxer.calc_many(structure_list) if r is not None
        ]
        if not energies_per_atom:
            raise RuntimeError(
                f"All elemental reference relaxations failed for {symbol!r}. "
                f"Cannot compute a formation energy. Either pass "
                f"use_gs_reference=True or supply pre-relaxed elemental_refs."
            )
        self._ref_energy_cache[key] = min(energies_per_atom)
        return self._ref_energy_cache[key]

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict[str, Any]:
        """
        Args:
            structure: Pymatgen structure, ASE atoms, or chained-result dict.

        Returns:
            Dict with ``formation_energy_per_atom`` (eV/atom), ``cohesive_energy_per_atom``
            (eV/atom, or None if reference data missing), ``final_structure``, ``_units``,
            and other keys from relaxation / PES when applicable.
        """
        result = super().calc(structure)
        result, structure_in = self._check_and_prelax(
            result["final_structure"],
            result,
            fmax=self.fmax,
            optimizer=self.optimizer,
            perturb_distance=self.perturb_distance,
        )
        energy = result["energy"] if self.relax_structure else run_pes_calc(structure_in, self.calculator).energy
        nsites = len(structure_in)

        def get_gs_energy(el: Element | Species) -> float:
            """
            Args:
                el: Element or species in the composition.

            Returns:
                Reference energy per atom for that element.
            """
            if el.symbol not in self.elemental_refs:
                raise KeyError(f"No elemental reference for {el.symbol!r} in elemental_refs.")
            if self.use_gs_reference:
                gs_energy = self.elemental_refs[el.symbol]["energy_per_atom"]
                return min(gs_energy) if isinstance(gs_energy, list) else gs_energy
            return self._relaxed_ref_energy_per_atom(el.symbol)

        comp = to_pmg_structure(structure_in).composition
        e_form = energy - sum([get_gs_energy(el) * amt for el, amt in comp.items()])

        try:
            e_coh = energy - sum([self.elemental_refs[el.symbol]["energy_atomic"] * amt for el, amt in comp.items()])
            result = result | {"cohesive_energy_per_atom": e_coh / nsites}
        except KeyError:
            result = result | {"cohesive_energy_per_atom": None}

        existing_units = result.get("_units")
        units_map: dict[str, str] = dict(existing_units) if isinstance(existing_units, dict) else {}
        units_map["formation_energy_per_atom"] = "eV/atom"
        units_map["cohesive_energy_per_atom"] = "eV/atom"
        return result | {
            "formation_energy_per_atom": e_form / nsites,
            "_units": units_map,
        }
