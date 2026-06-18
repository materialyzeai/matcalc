"""Monte Carlo ordering of disordered structures."""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import TYPE_CHECKING

from ase.units import kB
from joblib import Parallel, delayed

from ._base import PropCalc
from .backend import run_pes_calc
from .utils import to_pmg_structure

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Species, Structure

# A single ordering, represented as the per-site species assignment over the
# disordered sites (``None`` denotes a vacancy). Made hashable (a tuple) so
# distinct orderings can be deduplicated when collecting the lowest-energy set.
Tokens = tuple["Species | None", ...]


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

    Several independent chains can be launched from different random starts
    (``n_init``) and run in parallel (``n_jobs``); a chain stops early once its
    best energy has not improved for ``patience`` consecutive steps. The
    ``n_lowest`` lowest-energy distinct orderings found across all chains are
    returned, with the overall best as ``final_structure``.

    Attributes:
        calculator: ASE calculator (or universal model name) for energies.
        nsteps: Maximum number of Monte Carlo swap proposals per chain.
        temperature: Monte Carlo temperature (K).
        seed: Base seed for the random number generator (None for nondeterministic).
        patience: Stop a chain early if the best energy has not improved for this
            many consecutive steps (None disables early stopping).
        n_init: Number of independent Monte Carlo chains (random initializations).
        n_lowest: Number of lowest-energy distinct orderings to return.
        n_jobs: Worker count for running chains in parallel.
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
        patience: int | None = None,
        n_init: int = 1,
        n_lowest: int = 1,
        n_jobs: int | None = None,
        relax_structure: bool = False,
        relax_calc_kwargs: dict | None = None,
    ) -> None:
        """
        Args:
            calculator: ASE calculator or universal model name string.
            nsteps: Maximum number of Monte Carlo swap proposals per chain.
            temperature: Monte Carlo temperature in K. Must be >= 0; at 0 K only
                downhill (non-positive ΔE) swaps are accepted.
            seed: Base seed for the random number generator. ``None`` gives a
                nondeterministic run; when set, chain ``k`` uses ``seed + k`` so
                runs are reproducible.
            patience: Stop a chain early once its best (lowest) energy has not
                improved for ``patience`` consecutive steps. ``None`` (default)
                runs the full ``nsteps``. Must be a positive integer when set.
            n_init: Number of independent Monte Carlo chains launched from
                different random initial orderings. Must be >= 1.
            n_lowest: Number of lowest-energy distinct orderings to return. Must
                be >= 1.
            n_jobs: Number of joblib workers for running chains in parallel.
                ``None`` uses the joblib default; only relevant when ``n_init``
                > 1.
            relax_structure: Relax the lowest-energy ordering before returning.
            relax_calc_kwargs: Optional kwargs forwarded to ``RelaxCalc`` when
                ``relax_structure`` is True.

        Raises:
            ValueError: If ``temperature`` is negative, ``n_init`` < 1,
                ``n_lowest`` < 1, or ``patience`` is set but not a positive int.
        """
        if temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {temperature}.")
        if n_init < 1:
            raise ValueError(f"n_init must be >= 1, got {n_init}.")
        if n_lowest < 1:
            raise ValueError(f"n_lowest must be >= 1, got {n_lowest}.")
        if patience is not None and patience < 1:
            raise ValueError(f"patience must be a positive integer or None, got {patience}.")
        self.calculator = calculator  # type: ignore[assignment]
        self.nsteps = nsteps
        self.temperature = temperature
        self.seed = seed
        self.patience = patience
        self.n_init = n_init
        self.n_lowest = n_lowest
        self.n_jobs = n_jobs
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
        if n_atoms > n_sites:  # pragma: no cover
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
        tokens: list[Species | None] | Tokens,
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

    def _run_chain(
        self,
        structure: Structure,
        disordered_indices: list[int],
        seed: int | None,
    ) -> dict[str, Any]:
        """Run a single Monte Carlo chain.

        Args:
            structure: The disordered input structure.
            disordered_indices: Indices of the sites with partial occupancy.
            seed: Seed for this chain's random number generator.

        Returns:
            Dict with ``best_energy`` (eV), ``energies`` (the accepted-energy
            trajectory), ``acceptance_ratio``, ``steps_run`` (steps actually
            taken before early stopping), and ``candidates`` (the chain's
            ``n_lowest`` lowest-energy distinct orderings as ``{Tokens: energy}``).

        Raises:
            ValueError: If the disordered sites hold only a single species, so no
                swap can change the ordering.
        """
        rng = random.Random(seed)  # noqa: S311 — Monte Carlo sampling, not cryptographic
        tokens = self._initial_tokens(structure, disordered_indices, rng)
        n_sites = len(tokens)

        def energy(toks: list[Species | None]) -> float:
            ordered = self._make_structure(structure, disordered_indices, toks)
            return run_pes_calc(ordered, self.calculator).energy

        e_curr = energy(tokens)
        e_best = e_curr
        # Distinct orderings visited, keyed by token tuple -> energy. Keeping the
        # whole set of accepted configs lets us extract this chain's n lowest.
        candidates: dict[Tokens, float] = {tuple(tokens): e_curr}
        energies = [e_curr]
        n_accept = 0
        no_improve = 0
        steps_run = 0

        for _ in range(self.nsteps):
            steps_run += 1
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
                candidates[tuple(tokens)] = e_curr
            else:
                tokens[i], tokens[j] = tokens[j], tokens[i]  # reject: revert the swap
            energies.append(e_curr)

            if e_curr < e_best - 1e-12:  # tolerance guards against float noise
                e_best = e_curr
                no_improve = 0
            else:
                no_improve += 1
            if self.patience is not None and no_improve >= self.patience:
                break

        # Trim to this chain's n_lowest distinct orderings. This is sufficient to
        # recover the global n_lowest after merging: a config in the global top-n
        # cannot be outranked by n others within its own chain.
        lowest = dict(sorted(candidates.items(), key=lambda kv: kv[1])[: self.n_lowest])
        return {
            "best_energy": e_best,
            "energies": energies,
            "acceptance_ratio": n_accept / steps_run if steps_run else 0.0,
            "steps_run": steps_run,
            "candidates": lowest,
        }

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict[str, Any]:
        """
        Args:
            structure: A disordered pymatgen ``Structure`` (or a dict carrying
                one under ``final_structure`` / ``structure``).

        Returns:
            Dict with ``final_structure`` (the lowest-energy ordering, optionally
            relaxed), ``energy`` (eV, its total energy), ``lowest_structures``
            (the ``n_lowest`` lowest-energy distinct orderings, ascending) and
            ``lowest_energies`` (eV, aligned with them), ``energies`` (eV, the
            accepted-energy trajectory of the chain that found the best ordering),
            ``acceptance_ratio`` for that chain, ``_units``, plus any relaxation
            keys when ``relax_structure`` is True.

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

        # Distinct seeds per chain so parallel inits explore different orderings
        # while staying reproducible when a base seed is given.
        seeds = [None] * self.n_init if self.seed is None else [self.seed + k for k in range(self.n_init)]

        if self.n_init == 1:
            chains = [self._run_chain(structure_in, disordered_indices, seeds[0])]
        else:
            chains = Parallel(n_jobs=self.n_jobs)(
                delayed(self._run_chain)(structure_in, disordered_indices, s) for s in seeds
            )

        # Merge each chain's lowest orderings and keep the n_lowest distinct ones.
        merged: dict[Tokens, float] = {}
        for chain in chains:
            for toks, energy in chain["candidates"].items():
                if toks not in merged or energy < merged[toks]:
                    merged[toks] = energy
        lowest = sorted(merged.items(), key=lambda kv: kv[1])[: self.n_lowest]
        lowest_structures = [self._make_structure(structure_in, disordered_indices, toks) for toks, _ in lowest]
        lowest_energies = [energy for _, energy in lowest]

        # Report the trajectory/acceptance of the chain that found the best ordering.
        best_chain = min(chains, key=lambda chain: chain["best_energy"])
        best_structure = lowest_structures[0]

        result.update(
            {
                "final_structure": best_structure,
                "energy": lowest_energies[0],
                "lowest_structures": lowest_structures,
                "lowest_energies": lowest_energies,
                "energies": best_chain["energies"],
                "acceptance_ratio": best_chain["acceptance_ratio"],
                "_units": self._merge_units(result, {"energy": "eV", "energies": "eV", "lowest_energies": "eV"}),
            }
        )

        if self.relax_structure:
            result, _ = self._check_and_prelax(best_structure, result, **(self.relax_calc_kwargs or {}))

        return result
