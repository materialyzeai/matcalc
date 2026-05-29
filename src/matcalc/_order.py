"""Monte Carlo ordering of disordered structures."""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import TYPE_CHECKING

from ase.units import kB

from ._base import PropCalc
from .backend import run_pes_calc
from .utils import to_pmg_structure

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Species, Structure


class OrderCalc(PropCalc):
    """
    Order a disordered structure with Metropolis Monte Carlo.

    Given a structure with fractional site occupancies, ``OrderCalc`` builds a
    random ordering whose species counts are commensurate with the input
    occupancies (partial occupancy summing to < 1 is treated as vacancies),
    evaluates its total energy with the supplied PES calculator, then performs
    ``nsteps`` of Monte Carlo in which the species occupying two disordered sites
    are swapped. Each proposal is accepted with the standard Metropolis
    criterion ``min(1, exp(-ΔE / (kB·T)))``. Swaps exchange species between
    sites, so the overall composition is conserved throughout.

    The lowest-energy ordering encountered is returned as ``final_structure``.

    Attributes:
        calculator: ASE calculator (or universal model name) for energies.
        nsteps: Number of Monte Carlo swap proposals.
        temperature: Monte Carlo temperature (K).
        seed: Seed for the random number generator (None for nondeterministic).
        relax_structure: Relax the best ordering before returning it.
        relax_calc_kwargs: Optional kwargs forwarded to ``RelaxCalc``.
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        nsteps: int = 1000,
        temperature: float = 1000.0,
        seed: int | None = None,
        relax_structure: bool = False,
        relax_calc_kwargs: dict | None = None,
    ) -> None:
        """
        Args:
            calculator: ASE calculator or universal model name string.
            nsteps: Number of Monte Carlo swap proposals.
            temperature: Monte Carlo temperature in K. Must be >= 0; at 0 K only
                downhill (non-positive ΔE) swaps are accepted.
            seed: Seed for the random number generator. ``None`` gives a
                nondeterministic run.
            relax_structure: Relax the lowest-energy ordering before returning.
            relax_calc_kwargs: Optional kwargs forwarded to ``RelaxCalc`` when
                ``relax_structure`` is True.

        Raises:
            ValueError: If ``temperature`` is negative.
        """
        if temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {temperature}.")
        self.calculator = calculator  # type: ignore[assignment]
        self.nsteps = nsteps
        self.temperature = temperature
        self.seed = seed
        self.relax_structure = relax_structure
        self.relax_calc_kwargs = relax_calc_kwargs

    def _initial_tokens(
        self,
        structure: Structure,
        disordered_indices: list[int],
        rng: random.Random,
        tol: float = 1e-4,
    ) -> list[Species | None]:
        """Build a randomly shuffled species assignment for the disordered sites.

        Args:
            structure: The disordered input structure.
            disordered_indices: Indices of the sites with partial occupancy.
            rng: Random number generator used to shuffle the assignment.
            tol: Tolerance for rounding species amounts to integer counts.

        Returns:
            A list (one entry per disordered site) of the species to place at
            that site, or ``None`` for a vacancy.

        Raises:
            ValueError: If the total occupancy over the disordered sites is not
                commensurate with an integer number of atoms (within ``tol``).
        """
        n_sites = len(disordered_indices)
        amounts: Counter[Species] = Counter()
        for idx in disordered_indices:
            for sp, occ in structure[idx].species.items():
                amounts[sp] += occ

        counts: dict[Species, int] = {}
        for sp, amt in amounts.items():
            rounded = round(amt)
            if abs(amt - rounded) > tol:
                raise ValueError(
                    f"Occupancy of {sp} over the disordered sublattice is {amt:.4f}, which is not "
                    f"commensurate with an integer number of atoms. Use a supercell that makes the "
                    f"composition integral before ordering."
                )
            counts[sp] = rounded

        n_atoms = sum(counts.values())
        if n_atoms > n_sites:
            raise ValueError(f"Rounded species count ({n_atoms}) exceeds the number of disordered sites ({n_sites}).")

        tokens: list[Species | None] = []
        for sp, count in counts.items():
            tokens.extend([sp] * count)
        tokens.extend([None] * (n_sites - n_atoms))  # remaining sites are vacancies
        rng.shuffle(tokens)
        return tokens

    def _make_structure(
        self,
        structure: Structure,
        disordered_indices: list[int],
        tokens: list[Species | None],
    ) -> Structure:
        """Construct an ordered structure from a species assignment.

        Args:
            structure: The disordered input structure.
            disordered_indices: Indices of the sites with partial occupancy.
            tokens: Species (or ``None`` for a vacancy) to place at each
                disordered site, aligned with ``disordered_indices``.

        Returns:
            An ordered ``Structure``; ordered sites are untouched and vacancies
            are removed.
        """
        ordered = structure.copy()
        to_remove = []
        for sp, idx in zip(tokens, disordered_indices, strict=True):
            if sp is None:
                to_remove.append(idx)
            else:
                ordered[idx] = sp
        if to_remove:
            ordered.remove_sites(to_remove)
        return ordered

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict[str, Any]:
        """
        Args:
            structure: A disordered pymatgen ``Structure`` (or a dict carrying
                one under ``final_structure`` / ``structure``).

        Returns:
            Dict with ``final_structure`` (the lowest-energy ordering, optionally
            relaxed), ``energy`` (eV, total energy of that ordering),
            ``energies`` (eV, the accepted-energy trajectory of length
            ``nsteps + 1``), ``acceptance_ratio`` (fraction of accepted swaps),
            ``_units``, plus any relaxation keys when ``relax_structure`` is True.

        Raises:
            ValueError: If the input structure is already fully ordered, or has
                fewer than two distinct species to swap among its disordered
                sites.
        """
        result = super().calc(structure)
        structure_in = to_pmg_structure(result["final_structure"])

        if structure_in.is_ordered:
            raise ValueError("Structure is already ordered; nothing to do.")

        disordered_indices = [i for i, site in enumerate(structure_in) if not site.is_ordered]
        rng = random.Random(self.seed)  # noqa: S311 — Monte Carlo sampling, not cryptographic
        tokens = self._initial_tokens(structure_in, disordered_indices, rng)

        n_sites = len(tokens)

        def energy(toks: list[Species | None]) -> float:
            ordered = self._make_structure(structure_in, disordered_indices, toks)
            return run_pes_calc(ordered, self.calculator).energy

        e_curr = energy(tokens)
        e_best = e_curr
        best_tokens = list(tokens)
        energies = [e_curr]
        n_accept = 0

        for _ in range(self.nsteps):
            i = rng.randrange(n_sites)
            # Pick a partner whose species differs, so the swap actually changes the ordering.
            others = [k for k in range(n_sites) if tokens[k] != tokens[i]]
            if not others:
                raise ValueError("Disordered sites hold only one species; no swaps are possible.")
            j = rng.choice(others)

            tokens[i], tokens[j] = tokens[j], tokens[i]
            e_new = energy(tokens)
            delta = e_new - e_curr

            if delta <= 0 or (self.temperature > 0 and rng.random() < math.exp(-delta / (kB * self.temperature))):
                e_curr = e_new
                n_accept += 1
                if e_curr < e_best:
                    e_best, best_tokens = e_curr, list(tokens)
            else:
                tokens[i], tokens[j] = tokens[j], tokens[i]  # reject: revert the swap
            energies.append(e_curr)

        best_structure = self._make_structure(structure_in, disordered_indices, best_tokens)
        result.update(
            {
                "final_structure": best_structure,
                "energy": e_best,
                "energies": energies,
                "acceptance_ratio": n_accept / self.nsteps if self.nsteps else 0.0,
                "_units": self._merge_units(result, {"energy": "eV", "energies": "eV"}),
            }
        )

        if self.relax_structure:
            result, _ = self._prerelax(best_structure, result, **(self.relax_calc_kwargs or {}))

        return result
