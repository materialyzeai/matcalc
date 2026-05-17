"""Calculator for stability related properties."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from monty.serialization import loadfn

from ._base import PropCalc
from ._relaxation import RelaxCalc
from .backend import run_pes_calc
from .utils import to_pmg_structure

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Element, Species, Structure

ELEMENTAL_REFS_DIR = Path(__file__).parent / "elemental_refs"


class EnergeticsCalc(PropCalc):
    """
    Handles formation energy per atom, cohesive energy per atom, and relaxed structures
    using reference elemental data and optional relaxation.

    Attributes:
        calculator: ASE calculator (or universal model name) for energies and forces.
        elemental_refs: Builtin ref key or custom dict mapping elements to reference data.
        use_gs_reference: If True, use tabulated DFT ground-state energies from refs.
        relax_structure: Whether to relax the input structure before energetics.
        relax_calc_kwargs: Optional kwargs forwarded to ``RelaxCalc``.
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        elemental_refs: Literal["MP-PBE", "MatPES-PBE", "MatPES-r2SCAN"] | dict = "MatPES-PBE",
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
            elemental_refs: ``"MP-PBE"``, ``"MatPES-PBE"``, ``"MatPES-r2SCAN"``, or a custom dict.
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
        if isinstance(elemental_refs, str):
            self.elemental_refs = loadfn(ELEMENTAL_REFS_DIR / f"{elemental_refs}-Element-Refs.json.gz")
        else:
            self.elemental_refs = elemental_refs
        self.use_gs_reference = use_gs_reference
        self.relax_structure = relax_structure
        self.relax_calc_kwargs = relax_calc_kwargs

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
        relaxer = RelaxCalc(
            self.calculator,
            fmax=self.fmax,
            optimizer=self.optimizer,
            perturb_distance=self.perturb_distance,
            **(self.relax_calc_kwargs or {}),
        )
        result, structure_in = self._prerelax(result["final_structure"], result, relaxer=relaxer)
        energy = result["energy"] if self.relax_structure else run_pes_calc(structure_in, self.calculator).energy
        nsites = len(structure_in)

        def get_gs_energy(el: Element | Species) -> float:
            """
            Args:
                el: Element or species in the composition.

            Returns:
                Reference energy per atom for that element.
            """
            if self.use_gs_reference:
                gs_energy = self.elemental_refs[el.symbol]["energy_per_atom"]
                return min(gs_energy) if isinstance(gs_energy, list) else gs_energy

            relaxer.perturb_distance = None
            structure = self.elemental_refs[el.symbol]["structure"]
            structure_list = [structure] if not isinstance(structure, list) else structure
            energies_per_atom = [
                r["energy"] / r["final_structure"].num_sites
                for r in list(relaxer.calc_many(structure_list))
                if r is not None
            ]
            if not energies_per_atom:
                raise RuntimeError(
                    f"All elemental reference relaxations failed for {el.symbol!r}. "
                    f"Cannot compute a formation energy. Either pass "
                    f"use_gs_reference=True or supply pre-relaxed elemental_refs."
                )
            return min(energies_per_atom)

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
