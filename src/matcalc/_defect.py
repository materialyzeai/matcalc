"""Calculator for neutral point-defect formation energies."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from monty.serialization import loadfn
from pymatgen.core import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ._base import PropCalc
from ._relaxation import RelaxCalc
from ._stability import ELEMENTAL_REFS_DIR
from .utils import to_pmg_structure

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.optimize.optimize import Optimizer
    from numpy.typing import ArrayLike
    from pymatgen.core import Structure


class DefectCalc(PropCalc):
    """Calculate neutral point-defect formation energies in supercells.

    ``calc_defects`` relaxes the bulk, builds a supercell and enumerates vacancies, substitutions,
    antisites and interstitials; ``calc`` evaluates a single defect structure. The formation energy is

        E_f = E_defect - (N_defect - sum_i dn_i) * e_bulk - sum_i dn_i * mu_i,

    where ``e_bulk`` is the bulk energy per atom and ``dn_i`` the number of atoms of element i added
    (positive) or removed (negative). Unless given in ``chemical_potentials``, ``mu_i`` is the bulk
    energy per atom for the host of an elemental crystal, and otherwise the lowest energy per atom of
    the relaxed elemental references (the element-rich limit).
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        relax_bulk: bool = True,
        relax_defect: bool = True,
        fmax: float = 0.1,
        optimizer: str | Optimizer = "BFGS",
        max_steps: int = 500,
        chemical_potentials: dict[str, float] | None = None,
        elemental_refs: Literal["MP-PBE", "MatPES-PBE", "MatPES-r2SCAN"] | dict = "MatPES-PBE",
        relax_calc_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the defect protocol.

        Args:
            calculator: ASE calculator or universal model name.
            relax_bulk: Relax bulk atoms and cell. When False, the bulk energy is a single point.
            relax_defect: Relax defect atoms at fixed cell. When False, the defect energy is a
                single point on the unrelaxed defect structure.
            fmax: Force convergence (eV/A) for all relaxations.
            optimizer: ASE optimizer name or class.
            max_steps: Maximum relaxation steps.
            chemical_potentials: Chemical potentials (eV/atom) by element symbol, overriding the
                defaults described in the class docstring.
            elemental_refs: ``"MP-PBE"``, ``"MatPES-PBE"``, ``"MatPES-r2SCAN"`` or a custom dict
                mapping element symbols to ``{"structure": ...}``. The reference structures are
                relaxed with this calculator when a chemical potential is needed.
            relax_calc_kwargs: Extra arguments for RelaxCalc. ``relax_atoms`` and ``relax_cell``
                are set by the protocol.
        """
        self.calculator = calculator  # type: ignore[assignment]
        self.relax_bulk = relax_bulk
        self.relax_defect = relax_defect
        self.fmax = fmax
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.chemical_potentials = chemical_potentials
        self.elemental_refs = elemental_refs
        self.relax_calc_kwargs = relax_calc_kwargs
        self._reference_energies: dict[str, float] = {}

    def calc_defects(
        self,
        structure: Structure | Atoms,
        *,
        supercell_matrix: ArrayLike | None = None,
        min_length: float = 10.0,
        vacancies: bool | str | Sequence[str] = True,
        substitutions: Mapping[str, str | Sequence[str]] | None = None,
        antisites: bool = False,
        interstitials: str | Sequence[str] | Mapping[str, Sequence[ArrayLike]] | None = None,
        symprec: float = 0.01,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generate point defects in a supercell of the relaxed bulk and compute their formation energies.

        Defects are placed on one site of each symmetry-inequivalent orbit of the conventional cell.

        Args:
            structure: Ordered bulk structure, converted to the conventional cell.
            supercell_matrix: Scaling of the relaxed conventional cell. None repeats each lattice
                vector until it is at least ``min_length`` long.
            min_length: Minimum supercell lattice length (A) when ``supercell_matrix`` is None.
            vacancies: True for vacancies of every element, False for none, or element symbols.
            substitutions: Dopants by host element, e.g. ``{"Cu": ["Ni", "Zn"]}``.
            antisites: Place every other host element on each host site.
            interstitials: Element symbols for interstitial sites found by Voronoi analysis
                (requires the ``defects`` extra), or fractional coordinates in the relaxed
                conventional cell by element, e.g. ``{"H": [[0.5, 0.5, 0.5]]}``.
            symprec: Symmetry tolerance for finding inequivalent sites.
            **kwargs: Forwarded to ``calc_many``.

        Returns:
            Per-defect ``calc`` results, each with ``defect_name`` (e.g. ``v_Cu``, ``Ni_Cu``,
            ``H_i``; numbered when several sites share a name), ``element_changes``,
            ``defect_frac_coords`` (defect position in the supercell) and ``supercell_matrix``.
        """
        bulk = to_pmg_structure(structure).to_conventional()
        bulk.remove_oxidation_states()

        bulk_opt = self._relaxer(relax_cell=self.relax_bulk, relax_atoms=self.relax_bulk).calc(bulk)
        final_bulk = bulk_opt["final_structure"]
        if supercell_matrix is None:
            supercell_matrix = [math.ceil(min_length / length) for length in final_bulk.lattice.abc]
        matrix = np.array(supercell_matrix, dtype=int)
        if matrix.ndim < 2:  # noqa: PLR2004
            matrix = np.eye(3, dtype=int) * matrix
        supercell = final_bulk.make_supercell(matrix, in_place=False)

        generated = _site_defects(
            final_bulk,
            supercell,
            vacancies=vacancies,
            substitutions=substitutions,
            antisites=antisites,
            symprec=symprec,
        )
        for symbol, frac_coords in _interstitial_sites(final_bulk, interstitials).items():
            for frac in frac_coords:
                defect = supercell.copy()
                defect.append(symbol, final_bulk.lattice.get_cartesian_coords(frac), coords_are_cartesian=True)
                generated.append((f"{symbol}_i", defect, {symbol: 1}, defect[-1].frac_coords))

        totals = Counter(name for name, *_ in generated)
        seen: Counter[str] = Counter()
        entries = []
        for name, defect, changes, frac in generated:
            seen[name] += 1
            entries.append(
                {
                    "defect": defect,
                    "defect_name": f"{name}_{seen[name]}" if totals[name] > 1 else name,
                    "element_changes": changes,
                    "defect_frac_coords": frac,
                    "supercell_matrix": matrix,
                    "final_bulk": final_bulk,
                    "bulk_energy_per_atom": bulk_opt["energy"] / len(final_bulk),
                }
            )
        return list(self.calc_many(entries, **kwargs))  # type: ignore[arg-type]

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict[str, Any]:
        """Compute the formation energy of one defect structure.

        Args:
            structure: Dict with ``defect`` (defect supercell), ``element_changes`` (atoms added per
                element, negative for removed, e.g. ``{"Cu": -2}`` for a divacancy) and either
                ``bulk`` (relaxed here) or ``bulk_energy_per_atom`` with the relaxed ``final_bulk``.

        Returns:
            The input dict updated with ``final_defect``, ``defect_energy`` (eV),
            ``formation_energy`` (eV), ``chemical_potentials`` (eV/atom) of the changed elements,
            ``final_bulk`` and ``bulk_energy_per_atom`` (eV/atom).
        """
        if not (
            isinstance(structure, dict)
            and {"defect", "element_changes"} <= structure.keys()
            and ("bulk" in structure or {"bulk_energy_per_atom", "final_bulk"} <= structure.keys())
        ):
            raise ValueError(
                "For defect calculations, structure must be a dict with 'defect', 'element_changes' and "
                "either 'bulk' or both 'bulk_energy_per_atom' and 'final_bulk'."
            )

        result = structure.copy()
        if "bulk_energy_per_atom" in structure and "final_bulk" in structure:
            bulk_energy_per_atom = structure["bulk_energy_per_atom"]
        else:
            bulk_opt = self._relaxer(relax_cell=self.relax_bulk, relax_atoms=self.relax_bulk).calc(structure["bulk"])
            bulk_energy_per_atom = bulk_opt["energy"] / len(bulk_opt["final_structure"])
            result |= {"final_bulk": bulk_opt["final_structure"], "bulk_energy_per_atom": bulk_energy_per_atom}
        host = to_pmg_structure(result["final_bulk"]).composition

        defect = to_pmg_structure(structure["defect"])
        changes = {el: n for el, n in structure["element_changes"].items() if n}
        pristine = defect.composition.get_el_amt_dict()
        for el, n in changes.items():
            pristine[el] = pristine.get(el, 0) - n
        if any(n < 0 for n in pristine.values()) or (
            Composition({el: n for el, n in pristine.items() if n}).reduced_formula != host.reduced_formula
        ):
            raise ValueError(
                f"element_changes {changes} do not relate {defect.composition.formula} to the host "
                f"{host.reduced_formula}."
            )

        mu = {el: self._chemical_potential(el, host, bulk_energy_per_atom) for el in changes}
        defect_opt = self._relaxer(relax_cell=False, relax_atoms=self.relax_defect).calc(defect)
        n_pristine = len(defect) - sum(changes.values())
        formation_energy = (
            defect_opt["energy"] - n_pristine * bulk_energy_per_atom - sum(n * mu[el] for el, n in changes.items())
        )

        return result | {
            "defect": defect,
            "final_defect": defect_opt["final_structure"],
            "defect_energy": defect_opt["energy"],
            "formation_energy": formation_energy,
            "chemical_potentials": mu,
            "_units": self._merge_units(
                result,
                {
                    "bulk_energy_per_atom": "eV/atom",
                    "defect_energy": "eV",
                    "formation_energy": "eV",
                    "chemical_potentials": "eV/atom",
                },
            ),
        }

    def _relaxer(self, *, relax_cell: bool, relax_atoms: bool) -> RelaxCalc:
        kwargs: dict[str, Any] = {"fmax": self.fmax, "optimizer": self.optimizer, "max_steps": self.max_steps}
        kwargs |= (self.relax_calc_kwargs or {}) | {"relax_cell": relax_cell, "relax_atoms": relax_atoms}
        return RelaxCalc(self.calculator, **kwargs)

    def _chemical_potential(self, symbol: str, host: Composition, bulk_energy_per_atom: float) -> float:
        if self.chemical_potentials and symbol in self.chemical_potentials:
            return self.chemical_potentials[symbol]
        if host.is_element and host.elements[0].symbol == symbol:
            return bulk_energy_per_atom
        if symbol not in self._reference_energies:
            refs = self.elemental_refs
            if isinstance(refs, str):
                refs = self.elemental_refs = loadfn(ELEMENTAL_REFS_DIR / f"{refs}-Element-Refs.json.gz")
            if symbol not in refs:
                raise ValueError(f"No elemental reference for {symbol}; pass it in chemical_potentials.")
            structures = refs[symbol]["structure"]
            relaxer = self._relaxer(relax_cell=True, relax_atoms=True)
            self._reference_energies[symbol] = min(
                (opt := relaxer.calc(ref))["energy"] / len(opt["final_structure"])
                for ref in (structures if isinstance(structures, list) else [structures])
            )
        return self._reference_energies[symbol]


def _site_defects(
    bulk: Structure,
    supercell: Structure,
    *,
    vacancies: bool | str | Sequence[str],
    substitutions: Mapping[str, str | Sequence[str]] | None,
    antisites: bool,
    symprec: float,
) -> list[tuple[str, Structure, dict[str, int], np.ndarray]]:
    """Vacancies and substitutions on one site of each inequivalent orbit, as (name, defect, changes, frac)."""
    elements = sorted(bulk.symbol_set)
    vacancy_elements = set(
        elements if vacancies is True else [vacancies] if isinstance(vacancies, str) else vacancies or []
    )
    dopants = {host: [d] if isinstance(d, str) else list(d) for host, d in (substitutions or {}).items()}
    unknown = (vacancy_elements | set(dopants)) - set(elements)
    if unknown:
        raise ValueError(f"Elements {sorted(unknown)} are not in the host {bulk.composition.reduced_formula}.")

    defects = []
    for orbit in SpacegroupAnalyzer(bulk, symprec=symprec).get_symmetrized_structure().equivalent_indices:
        symbol = bulk[orbit[0]].specie.symbol
        index = _site_index(supercell, bulk[orbit[0]].coords)
        frac = supercell[index].frac_coords
        if symbol in vacancy_elements:
            defect = supercell.copy()
            defect.remove_sites([index])
            defects.append((f"v_{symbol}", defect, {symbol: -1}, frac))
        for dopant in dopants.get(symbol, []) + ([el for el in elements if el != symbol] if antisites else []):
            defect = supercell.copy()
            defect.replace(index, dopant)
            defects.append((f"{dopant}_{symbol}", defect, {symbol: -1, dopant: 1}, frac))
    return defects


def _site_index(supercell: Structure, coords: np.ndarray) -> int:
    """Index of the supercell site at the Cartesian position of a unit-cell site."""
    frac = supercell.lattice.get_fractional_coords(coords)
    return int(np.argmin(supercell.lattice.get_all_distances(frac, supercell.frac_coords)[0]))


def _interstitial_sites(
    bulk: Structure, interstitials: str | Sequence[str] | Mapping[str, Sequence[ArrayLike]] | None
) -> dict[str, list[np.ndarray]]:
    """Fractional interstitial coordinates in ``bulk`` by element."""
    if interstitials is None:
        return {}
    if isinstance(interstitials, Mapping):
        return {symbol: [np.asarray(frac, dtype=float) for frac in coords] for symbol, coords in interstitials.items()}
    try:
        from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator
    except ImportError as exc:
        raise ImportError(
            "Automatic interstitial sites require pymatgen-analysis-defects: pip install 'matcalc[defects]'."
        ) from exc
    generator = VoronoiInterstitialGenerator()
    symbols = [interstitials] if isinstance(interstitials, str) else interstitials
    return {symbol: [d.site.frac_coords for d in generator.generate(bulk, [symbol])] for symbol in symbols}
