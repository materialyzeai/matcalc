"""Calculator for Monte Carlo sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from tqdm import tqdm

from ._base import PropCalc
from ._relaxation import RelaxCalc
from .utils import to_ase_atoms, to_pmg_structure

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure

KB = 8.617330337217213e-05  # eV/K


def _atoms_from_results(results: dict[str, Any]) -> Atoms:
    """
    Build an ASE ``Atoms`` carrying a ``SinglePointCalculator`` from a results dict.

    :param results: A results dictionary containing ``final_structure``, ``energy``,
        ``forces`` and ``stress`` keys (as returned by :class:`RelaxCalc`).
    :type results: dict[str, Any]
    :return: An ``Atoms`` object with energy/forces/stress attached, suitable for writing
        to an ASE trajectory.
    :rtype: Atoms
    """
    atoms = to_ase_atoms(results["final_structure"])
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=results["energy"],
        forces=results["forces"],
        stress=results["stress"],
    )
    return atoms


class MCCalc(PropCalc):
    """
    Metropolis-Hastings Monte Carlo sampler driven by a generic transformation.

    At each step the (duck-typed) ``transformation`` proposes a new configuration via its
    ``apply_transformation(structure)`` method; the proposal is scored with a single-point
    energy or a relaxation and accepted/rejected by the Metropolis criterion. When
    ``transform_initial`` is True the transformation is applied to the pristine structure
    every step, so all sampled configurations share a fixed composition (configurational
    sampling at a single concentration); otherwise it is applied to the current state,
    yielding a random walk in configuration space. The trajectory and acceptance ratio are
    recorded every ``save_freq`` steps, and the lowest-energy configuration visited is tracked
    (``min_energy`` / ``min_structure``).

    :param calculator: An ASE calculator object used to perform energy and force
        calculations. If a string is provided, the corresponding universal calculator is loaded.
    :type calculator: Calculator | str
    """

    def __init__(
        self,
        calculator: Calculator | str,
        transformation: Any,
        ensemble: Literal["canonical", "grand_canonical"] = "canonical",
        *,
        transform_initial: bool,
        nsteps: int = 100,
        temperature: float = 300,
        save_freq: int = 10,
        trajfile: str = "mc.traj",
        relax: bool = True,
        seed: int | None = None,
        relax_calc_kwargs: dict | None = None,
    ) -> None:
        """
        Initialize the MCCalc.

        :param calculator: An ASE calculator object used to perform energy and force
            calculations. If a string is provided, the corresponding universal calculator is loaded.
        :type calculator: Calculator | str
        :param transformation: Any object exposing ``apply_transformation(structure) -> Structure``
            that proposes a new configuration (e.g. a pymatgen transformation).
        :type transformation: Any
        :param transform_initial: If True, apply the transformation to the pristine input structure
            every step (fixed-composition sampling); if False, apply it to the current accepted state.
        :type transform_initial: bool
        :param ensemble: Monte Carlo ensemble. Only ``"canonical"`` is implemented;
            ``"grand_canonical"`` is reserved. Default is ``"canonical"``.
        :type ensemble: Literal["canonical", "grand_canonical"], optional
        :param nsteps: Number of Monte Carlo steps to run. Default is 100.
        :type nsteps: int, optional
        :param temperature: Temperature in Kelvin used in the Metropolis acceptance test. Default is 300.
        :type temperature: float, optional
        :param save_freq: Append a frame to the trajectory every ``save_freq`` steps. Default is 10.
        :type save_freq: int, optional
        :param trajfile: Trajectory filename. Default is ``"mc.traj"``.
        :type trajfile: str, optional
        :param relax: Whether to relax each configuration before scoring it. If False a single
            point energy is used. Default is True.
        :type relax: bool, optional
        :param seed: Seed for the random number generator, for reproducible sampling. Default is None.
        :type seed: int | None, optional
        :param relax_calc_kwargs: Additional keyword arguments passed to :class:`RelaxCalc`.
            ``max_steps``, ``optimizer`` and ``fmax`` are honored if present. Default is None.
        :type relax_calc_kwargs: dict | None, optional
        """
        self.calculator = calculator
        self.transformation = transformation
        self.nsteps = nsteps
        self.temperature = temperature
        self.ensemble = ensemble
        self.save_freq = save_freq
        self.trajfile = trajfile
        self.relax = relax
        self.relax_calc_kwargs = relax_calc_kwargs or {}
        self._rng = np.random.default_rng(seed)
        self.transform_initial = transform_initial
        self.trajectory: list[Atoms] = []
        self.acceptance_ratio = 0.0
        self.min_energy: float | None = None
        self.min_structure: Structure | None = None
        self._n_accepted = 0
        self._structure: Structure | None = None
        self._initial_structure: Structure | None = None
        self._results: dict[str, Any] = {}
        self._min_results: dict[str, Any] = {}

    def _score(self, structure: Structure) -> dict[str, Any]:
        """
        Score a structure with a relaxation (or single point if ``relax`` is False).

        :param structure: The configuration to evaluate.
        :type structure: Structure
        :return: The :class:`RelaxCalc` results dictionary (``final_structure``, ``energy``, ...).
        :rtype: dict[str, Any]
        """
        kwargs = dict(self.relax_calc_kwargs)
        max_steps = kwargs.pop("max_steps", 200) if self.relax else kwargs.pop("max_steps", 0)
        relaxer = RelaxCalc(
            self.calculator,
            optimizer=kwargs.pop("optimizer", "FIRE"),
            fmax=kwargs.pop("fmax", 0.02),
            max_steps=max_steps,
            **kwargs,
        )
        return relaxer.calc(structure)

    def step_canonical(self) -> bool:
        """
        Take a single canonical Monte Carlo step and update the accepted state.

        Proposes a new configuration, scores it, and accepts it with the Metropolis
        criterion, updating the current structure and results on acceptance.

        :return: Whether the proposed configuration was accepted.
        :rtype: bool
        """
        if self.transform_initial:
            proposal = self.transformation.apply_transformation(self._initial_structure)
        else:
            proposal = self.transformation.apply_transformation(self._structure)
        results = self._score(proposal)
        if results["energy"] < self._min_results["energy"]:
            self._min_results = results
        delta = results["energy"] - self._results["energy"]
        accept = delta < 0 or self._rng.random() < np.exp(-delta / (KB * self.temperature))
        if accept:
            self._structure = results["final_structure"]
            self._results = results
            self._n_accepted += 1
        return accept

    def step_grand_canonical(self) -> bool:
        """
        Take a single grand-canonical Monte Carlo step.

        :raises NotImplementedError: Grand-canonical sampling is not yet implemented.
        :rtype: bool
        """
        raise NotImplementedError

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict[str, Any]:
        """
        Run the Monte Carlo simulation on the input structure.

        :param structure: The starting structure to sample from, or a dict carrying one under
            ``final_structure`` / ``structure``.
        :type structure: Structure | Atoms | dict[str, Any]
        :return: The final accepted configuration's results augmented with ``acceptance_ratio`` and
            the lowest-energy configuration visited (``min_energy`` and ``min_structure``).
        :rtype: dict[str, Any]
        :raises ValueError: If ``ensemble`` is not a recognized value.
        """
        result = super().calc(structure)
        self._initial_structure = to_pmg_structure(result["final_structure"])
        if self.transform_initial:
            self._structure = self.transformation.apply_transformation(self._initial_structure)
        else:
            self._structure = self._initial_structure
        self._results = self._score(self._structure)
        self._min_results = self._results
        self.trajectory = [_atoms_from_results(self._results)]
        if self.ensemble == "canonical":
            step = self.step_canonical
        elif self.ensemble == "grand_canonical":
            step = self.step_grand_canonical
        else:
            raise ValueError("Only canonical Ensemble accepted")
        for i in tqdm(range(self.nsteps), desc="Step"):
            step()
            if i % self.save_freq == 0:
                self.trajectory.append(_atoms_from_results(self._results))
                write(self.trajfile, self.trajectory)
        self.acceptance_ratio = self._n_accepted / self.nsteps
        self.min_energy = self._min_results["energy"]
        self.min_structure = self._min_results["final_structure"]
        return self._results | {
            "acceptance_ratio": self.acceptance_ratio,
            "min_energy": self.min_energy,
            "min_structure": self.min_structure,
        }
