"""Intercalation and voltage-profile calculations via Monte Carlo."""

from __future__ import annotations

import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from monty.serialization import dumpfn
from pymatgen.core import Structure
from pymatgen.transformations.site_transformations import RemoveSitesTransformation
from tqdm import tqdm

from ._base import PropCalc
from ._mc import MCCalc
from .utils import to_pmg_structure

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms
    from ase.calculators.calculator import Calculator

# scipy.linalg.logm (called by ASE's FrechetCellFilter on every relaxation step) emits a
# benign RuntimeWarning whenever its error estimate is non-zero, even at ~1e-13. Silence it.
warnings.filterwarnings("ignore", message="logm result may be inaccurate")

# Number of species on a binary (species <-> species) active site.
_BINARY_SPECIES = 2


class _RemoveKSites:
    """
    Transformation that removes ``k`` randomly chosen sites from a fixed candidate set.

    Sites are drawn without replacement so that exactly ``k`` distinct sites are removed on
    every call. Intended to be driven by :class:`MCCalc` with ``transform_initial=True`` so that
    each proposal removes ``k`` ions from the pristine structure (fixed-concentration sampling).
    """

    def __init__(self, indices: Sequence[int], k: int, rng: np.random.Generator) -> None:
        """
        Initialize the transformation.

        :param indices: Candidate site indices eligible for removal.
        :type indices: Sequence[int]
        :param k: Number of sites to remove on each call.
        :type k: int
        :param rng: Random number generator used to choose which sites to remove.
        :type rng: numpy.random.Generator
        """
        self.indices = indices
        self.k = k
        self.rng = rng

    def apply_transformation(self, structure: Structure) -> Structure:
        """
        Return a copy of ``structure`` with ``k`` randomly chosen sites removed.

        :param structure: The structure to deintercalate.
        :type structure: Structure
        :return: A new structure with the selected sites removed.
        :rtype: Structure
        """
        indices_to_remove = self.rng.choice(self.indices, self.k, replace=False)
        return RemoveSitesTransformation(indices_to_remove).apply_transformation(structure)


class _SwapSites:
    """
    Composition-conserving swap move on a configurable sublattice (a true MCMC move).

    Given an ``ordered_structure`` (the fully occupied host) and a ``disordered_structure``
    (the *same cell* with partial occupancy marking the configurable sublattice), each call
    relocates ``n_swaps`` (``l``) occupants among the active sites:

    - **species ↔ vacancy** (active site has one species with occupancy < 1): ``l`` occupied
      sites are vacated and ``l`` vacant sites are filled, conserving the species count.
    - **species ↔ species** (active site has two species summing to 1): ``l`` pairs exchange
      identity, conserving each species count.

    Because the move is a local perturbation of the *current* configuration and conserves
    composition, driving it with :class:`MCCalc` and ``transform_initial=False`` yields a true
    canonical Markov chain. The two structures must share the same lattice and sites; the
    disordered structure's partially occupied (``not site.is_ordered``) sites define the
    active sublattice.
    """

    def __init__(
        self,
        ordered_structure: Structure,
        disordered_structure: Structure,
        n_swaps: int,
        rng: np.random.Generator,
        *,
        tol: float = 1.0,
    ) -> None:
        """
        Initialize the swap transformation.

        :param ordered_structure: The fully occupied host providing the active-site positions and
            inert framework.
        :type ordered_structure: Structure
        :param disordered_structure: The same cell with partial occupancy marking the configurable
            sublattice (its ``not site.is_ordered`` sites define the active sublattice and states).
        :type disordered_structure: Structure
        :param n_swaps: Number of sites to swap on each call (``l``).
        :type n_swaps: int
        :param rng: Random number generator used to choose which sites to swap.
        :type rng: numpy.random.Generator
        :param tol: Distance tolerance in Å for mapping current sites onto the active sublattice.
        :type tol: float, optional
        :raises ValueError: If the structures differ in size, there are no active sites, or the
            active sites have an unsupported number of species.
        """
        if len(ordered_structure) != len(disordered_structure):
            raise ValueError("ordered_structure and disordered_structure must have the same sites")
        self.lattice = ordered_structure.lattice
        self.n_swaps = n_swaps
        self.rng = rng
        self.tol = tol

        self.active_idx = [i for i, site in enumerate(disordered_structure) if not site.is_ordered]
        if not self.active_idx:
            raise ValueError("disordered_structure has no disordered (active) sites")
        self.active_fcoords = ordered_structure.frac_coords[self.active_idx]

        elements = [el.symbol for el in disordered_structure[self.active_idx[0]].species.elements]
        if len(elements) == 1:
            self.species_a: str = elements[0]  # species <-> vacancy
            self.species_b: str | None = None
        elif len(elements) == _BINARY_SPECIES:
            self.species_a, self.species_b = elements  # species <-> species
        else:
            raise ValueError("active sites must have one (species/vacancy) or two (binary species)")

    def _match(self, fcoords: np.ndarray) -> np.ndarray:
        """Map fractional coords to the nearest active-sublattice positions (list indices 0..M-1)."""
        dists = self.lattice.get_all_distances(fcoords, self.active_fcoords)
        nearest = dists.argmin(axis=1)
        if (dists[np.arange(len(fcoords)), nearest] > self.tol).any():
            raise ValueError("could not map a site onto the active sublattice within tolerance")
        return nearest

    def apply_transformation(self, structure: Structure) -> Structure:
        """
        Return a copy of ``structure`` with ``n_swaps`` sites swapped on the active sublattice.

        :param structure: The current configuration (an ordered structure at the target composition).
        :type structure: Structure
        :return: A new configuration differing by the swap move, with composition conserved.
        :rtype: Structure
        """
        if self.species_b is None:
            return self._swap_vacancy(structure)
        return self._swap_binary(structure)

    def _swap_vacancy(self, structure: Structure) -> Structure:
        occ_current = [i for i, site in enumerate(structure) if site.specie.symbol == self.species_a]
        occ_pos = self._match(structure.frac_coords[occ_current])
        pos_to_current = dict(zip(occ_pos.tolist(), occ_current, strict=True))
        occupied = sorted(pos_to_current)
        vacant = [p for p in range(len(self.active_idx)) if p not in pos_to_current]
        if self.n_swaps > min(len(occupied), len(vacant)):
            raise ValueError(f"n_swaps={self.n_swaps} exceeds available occupied/vacant active sites")

        chosen_occ = self.rng.choice(occupied, self.n_swaps, replace=False)
        chosen_vac = self.rng.choice(vacant, self.n_swaps, replace=False)

        new = structure.copy()
        new.remove_sites(sorted((pos_to_current[int(p)] for p in chosen_occ), reverse=True))
        for p in chosen_vac:
            new.append(self.species_a, self.active_fcoords[int(p)], coords_are_cartesian=False)
        return new

    def _swap_binary(self, structure: Structure) -> Structure:
        assert self.species_b is not None  # noqa: S101  # only called when apply_transformation found a binary site
        swap_current = [i for i, site in enumerate(structure) if site.specie.symbol in (self.species_a, self.species_b)]
        pos = self._match(structure.frac_coords[swap_current])
        pos_to_current = {int(p): ci for p, ci in zip(pos.tolist(), swap_current, strict=True)}
        a_positions = [p for p, ci in pos_to_current.items() if structure[ci].specie.symbol == self.species_a]
        b_positions = [p for p, ci in pos_to_current.items() if structure[ci].specie.symbol == self.species_b]
        if self.n_swaps > min(len(a_positions), len(b_positions)):
            raise ValueError(f"n_swaps={self.n_swaps} exceeds available {self.species_a}/{self.species_b} sites")

        chosen_a = self.rng.choice(a_positions, self.n_swaps, replace=False)
        chosen_b = self.rng.choice(b_positions, self.n_swaps, replace=False)

        new = structure.copy()
        for p in chosen_a:
            new.replace(pos_to_current[int(p)], self.species_b)
        for p in chosen_b:
            new.replace(pos_to_current[int(p)], self.species_a)
        return new


def _build_disordered(structure: Structure, species: str | None) -> Structure:
    """
    Build a disordered structure marking the ``species`` sublattice as active (species/vacancy).

    :param structure: The fully occupied host structure.
    :type structure: Structure
    :param species: Symbol of the active (swappable) species.
    :type species: str | None
    :return: A copy with the ``species`` sites given partial occupancy (active) and the rest left
        fully occupied (inert framework).
    :rtype: Structure
    :raises ValueError: If ``species`` is not set.
    """
    if species is None:
        raise ValueError("algorithm='markov' without disordered_structure requires species")
    occupancies = [{species: 0.5} if site.specie.symbol == species else site.species for site in structure]
    return Structure(structure.lattice, occupancies, structure.frac_coords)


class IntercalationCalc(PropCalc):
    """
    Calculator for intercalation and voltage-profile predictions via Monte Carlo.

    For each target concentration derived from ``concentration_range``, a number ``k`` of
    intercalating ions is removed from the (optionally supercelled) host structure and the
    resulting configurations are sampled with an :class:`MCCalc` Metropolis-Hastings run at fixed
    composition. The accepted energies are used downstream to predict voltage profiles. A
    trajectory is written per concentration level and the per-level results are serialized to
    ``results.json.gz``.

    :param calculator: An ASE calculator object used to perform energy and force
        calculations. If a string is provided, the corresponding universal calculator is loaded.
    :type calculator: Calculator | str
    """

    def __init__(
        self,
        calculator: Calculator | str,
        nsteps: int,
        temperature: float,
        concentration_range: Sequence[float],
        *,
        save_freq: int = 100,
        trajfile: str | None = None,
        supercell: Sequence[int] | None = None,
        relax: bool = True,
        species: str | None = None,
        indices: Sequence[int] | None = None,
        algorithm: Literal["independent", "markov"] = "independent",
        disordered_structure: Structure | None = None,
        swap_size: int = 1,
        seed: int | None = None,
        relax_calc_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the IntercalationCalc.

        :param calculator: An ASE calculator object used to perform energy and force
            calculations. If a string is provided, the corresponding universal calculator is loaded.
        :type calculator: Calculator | str
        :param nsteps: Number of Monte Carlo steps to run at each concentration level.
        :type nsteps: int
        :param temperature: Temperature in Kelvin used in the Metropolis-Hastings acceptance test.
        :type temperature: float
        :param concentration_range: Arguments forwarded to :func:`numpy.arange` (start, stop, step)
            defining the fractional concentrations of removed ions to sample.
        :type concentration_range: Sequence[float]
        :param save_freq: Append a frame to the trajectory every ``save_freq`` Monte Carlo steps.
            Default is 100.
        :type save_freq: int, optional
        :param trajfile: Base trajectory filename. A ``_k{k}`` suffix is inserted before the
            extension for each concentration level. Defaults to a timestamped name.
        :type trajfile: str | None, optional
        :param supercell: Optional supercell scaling applied to the input structure before site
            indices are resolved. Only valid together with ``species``. Default is None.
        :type supercell: Sequence[int] | None, optional
        :param relax: Whether to relax each configuration before scoring it. If False a single
            point energy is used. Default is True.
        :type relax: bool, optional
        :param species: Symbol of the intercalating species to remove. Mutually exclusive with
            ``indices``. Default is None.
        :type species: str | None, optional
        :param indices: Explicit site indices eligible for removal. Mutually exclusive with
            ``species``. Default is None.
        :type indices: Sequence[int] | None, optional
        :param algorithm: Sampling algorithm. ``"independent"`` draws each proposal afresh from the
            pristine host (fixed-concentration independent sampling); ``"markov"`` runs a true
            Metropolis Markov chain of local swap moves (requires ``species`` or
            ``disordered_structure``). Default is ``"independent"``.
        :type algorithm: Literal["independent", "markov"], optional
        :param disordered_structure: For ``algorithm="markov"``, a structure on the same cell as the
            (supercelled) host whose partially occupied sites define the active sublattice. If omitted,
            one is built from ``species`` (species/vacancy). Default is None.
        :type disordered_structure: Structure | None, optional
        :param swap_size: Number of sites swapped per Markov step (``l``). Default is 1.
        :type swap_size: int, optional
        :param seed: Seed for the random number generator, for reproducible sampling. Default is None.
        :type seed: int | None, optional
        :param relax_calc_kwargs: Additional keyword arguments passed to :class:`RelaxCalc`.
            ``max_steps``, ``optimizer`` and ``fmax`` are honored if present. Default is None.
        :type relax_calc_kwargs: dict[str, Any] | None, optional
        """
        self.calculator = calculator  # type: ignore[assignment]
        self.nsteps = nsteps
        self.temperature = temperature
        self.concentration_range = concentration_range
        self.supercell = supercell
        self.save_freq = save_freq
        self.trajfile = trajfile or f"traj-{datetime.now(tz=UTC).strftime('%H%Mhrs_%d-%m-%Y')}.traj"
        self.species = species
        self.indices = indices
        self.algorithm = algorithm
        self.disordered_structure = disordered_structure
        self.swap_size = swap_size
        self.relax = relax
        self.relax_calc_kwargs = relax_calc_kwargs or {}
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)

    def calc(
        self,
        structure: Structure | Atoms | dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run the Monte Carlo deintercalation sweep over the requested concentrations.

        :param structure: The fully occupied host structure to deintercalate, or a dict carrying
            one under ``final_structure`` / ``structure``.
        :type structure: Structure | Atoms | dict[str, Any]
        :return: A dictionary keyed by concentration index, each value being the accepted
            configuration's :class:`MCCalc` results augmented with ``Num_removed`` and
            ``concentration``.
        :rtype: dict[str, Any]
        :raises ValueError: If neither or both of ``species`` and ``indices`` are provided, or if
            ``supercell`` is combined with explicit ``indices``.
        """
        result = super().calc(structure)
        structure = to_pmg_structure(result["final_structure"])

        if self.indices is None and self.species is None:
            raise ValueError("Provide one of indices or species")
        if self.species is not None and self.indices is None:
            if self.supercell:
                structure = structure.make_supercell(self.supercell)
            self.indices = structure.indices_from_symbol(self.species)
        elif self.indices is not None and self.supercell is not None:
            raise ValueError("Provide supercell as input structure if specifying supercell argument")

        assert self.indices is not None  # noqa: S101  # resolved above; satisfies type checker

        disordered = None
        if self.algorithm == "markov":
            disordered = self.disordered_structure or _build_disordered(structure, self.species)

        n_indices = len(self.indices)
        concentrations = np.arange(*self.concentration_range)
        ks = np.unique(np.round(concentrations * n_indices)).astype(int)

        results = {}
        for ik, k in enumerate(tqdm(ks, desc="concentration levels")):
            concentration = k / n_indices
            traj_path = Path(self.trajfile)
            k_trajfile = str(traj_path.with_name(f"{traj_path.stem}_k{int(k)}{traj_path.suffix}"))

            if self.algorithm == "independent":
                # Independent fixed-concentration sampling: each proposal removes k from the pristine host.
                transformation: Any = _RemoveKSites(self.indices, int(k), self._rng)
                transform_initial = True
                mc_structure = structure
            elif self.algorithm == "markov":
                # True Markov chain: seed a configuration at concentration k, then swap-move locally.
                assert disordered is not None  # noqa: S101  # set above when algorithm == "markov"
                transformation = _SwapSites(structure, disordered, self.swap_size, self._rng)
                transform_initial = False
                mc_structure = _RemoveKSites(self.indices, int(k), self._rng).apply_transformation(structure)
            else:
                raise ValueError("algorithm must be one of {'independent', 'markov'}")

            mc = MCCalc(
                self.calculator,
                transformation,
                transform_initial=transform_initial,
                nsteps=self.nsteps,
                temperature=self.temperature,
                save_freq=self.save_freq,
                trajfile=k_trajfile,
                relax=self.relax,
                seed=self.seed,
                relax_calc_kwargs=self.relax_calc_kwargs,
            )
            mc_results = mc.calc(mc_structure)
            results[f"{ik}"] = mc_results | {
                "Num_removed": int(k),
                "concentration": concentration,
            }

        dumpfn(results, "results.json.gz")
        return results
