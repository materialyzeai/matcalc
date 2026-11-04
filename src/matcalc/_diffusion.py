"""Calculator for ionic diffusivity and conductivity via MD and DiffusionAnalyzer."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

from ase import units
from pymatgen.core import Element
from scipy import constants

from ._base import PropCalc
from ._md import MDCalc
from .utils import to_pmg_structure

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure

_ASE_ENSEMBLES = Literal[
    "nve",
    "nvt",
    "nvt_nose_hoover",
    "nvt_berendsen",
    "nvt_langevin",
    "nvt_andersen",
    "nvt_bussi",
    "npt",
    "npt_nose_hoover",
    "npt_berendsen",
    "npt_inhomogeneous",
    "npt_mtk",
    "npt_isotropic_mtk",
]


class DiffusionCalc(PropCalc):
    """Calculate ionic diffusivity and conductivity from an MD trajectory.

    Runs MD with :class:`MDCalc` (on the simulation backend selected by ``MATCALC_BACKEND``) and analyses the
    equilibrated trajectory with
    pymatgen's DiffusionAnalyzer. An input dict containing ``trajectory`` skips
    MD and uses those frames directly. Requires the ``diffusion`` extra.
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        specie: str,
        charge: float | None = None,
        temperature: float,
        timestep: float,
        loginterval: int,
        ensemble: _ASE_ENSEMBLES = "nvt_nose_hoover",
        steps: int = 100_000,
        equilibration_frames: int = 0,
        taut: float = 100.0,
        seed: int | None = None,
        relax_structure: bool = True,
        relax_atoms_only: bool = True,
        set_com_stationary: bool = True,
        set_zero_rotation: bool = False,
        relax_calc_kwargs: dict | None = None,
        analyzer_kwargs: dict[str, Any] | None = None,
        **md_kwargs: Any,
    ) -> None:
        """Initialize the MD protocol and diffusion analysis.

        Args:
            calculator: ASE calculator instance or universal model name.
            specie: Symbol of the mobile ion species, e.g. ``"Li"``.
            charge: Charge of the mobile ion in units of e, for the Nernst-Einstein conductivity. None
                takes the oxidation state of ``specie`` in the input structure if it is decorated, else the
                most likely oxidation state for its composition (e.g. +1 for Cu in Cu2Se), else its first
                common oxidation state with a warning.
            temperature: Simulation temperature in Kelvin, also used for the conductivity. For a supplied
                trajectory, the temperature it was run at.
            timestep: MD time step in femtoseconds. For a supplied trajectory, the step it was run with.
            loginterval: Steps between saved frames, also used as the analyzer's ``step_skip``. For a
                supplied trajectory, the interval it was saved with.
            ensemble: ``MDCalc`` ensemble. Use NVT or NVE: the analysis unwraps
                displacements on each frame's cell, which is only approximate when the cell fluctuates.
            steps: Total number of MD steps.
            equilibration_frames: Leading saved frames discarded by the internal MD run. The default of 0
                keeps the thermostat transient from the Maxwell-Boltzmann start in the MSD, which biases
                the linear fit for short runs. Set it to cover at least the equilibration period, e.g.
                a few ps (``equilibration_frames * loginterval * timestep`` fs).
            taut: Thermostat time constant in femtoseconds.
            seed: Seed for initial velocities on each MD run. None uses a random seed.
            relax_structure: Whether to relax the structure before MD.
            relax_atoms_only: Relax atomic positions only, keeping the input cell. Set False to also
                relax the cell; this gives the 0 K volume, which excludes thermal expansion and
                typically lowers D at the elevated temperatures used for Arrhenius fits.
            set_com_stationary: Remove initial center-of-mass momentum.
            set_zero_rotation: Remove initial angular momentum.
            relax_calc_kwargs: Extra arguments for RelaxCalc when relaxation is enabled. A
                ``relax_cell`` entry here takes precedence over ``relax_atoms_only``.
            analyzer_kwargs: Extra arguments for DiffusionAnalyzer, e.g. ``smoothed``.
            **md_kwargs: Additional arguments forwarded to ``MDCalc`` in its units.
        """
        self.calculator = calculator
        self.specie = specie
        self.charge = charge
        self.temperature = temperature
        self.ensemble: _ASE_ENSEMBLES = ensemble
        self.timestep = timestep
        self.steps = steps
        self.loginterval = loginterval
        self.equilibration_frames = equilibration_frames
        self.taut = taut
        self.seed = seed
        self.relax_structure = relax_structure
        self.relax_atoms_only = relax_atoms_only
        self.set_com_stationary = set_com_stationary
        self.set_zero_rotation = set_zero_rotation
        self.relax_calc_kwargs = {"relax_cell": not relax_atoms_only, **(relax_calc_kwargs or {})}
        self.analyzer_kwargs = analyzer_kwargs or {}
        self.md_kwargs = md_kwargs

    def _build_md_calc(self) -> MDCalc:
        """Build the MD calculator with a fresh initial-velocity generator."""
        if self.ensemble.lower().startswith("npt"):
            warnings.warn(
                f"Ensemble {self.ensemble!r} varies the cell, but DiffusionAnalyzer unwraps displacements "
                "assuming a fixed lattice, so the MSD is only approximate. Use an NVT or NVE ensemble.",
                UserWarning,
                stacklevel=3,
            )
        return MDCalc(
            self.calculator,
            ensemble=self.ensemble,
            temperature=self.temperature,
            timestep=self.timestep,
            steps=self.steps,
            taut=self.taut * units.fs,
            seed=self.seed,
            loginterval=self.loginterval,
            equilibration_frames=self.equilibration_frames,
            relax_structure=self.relax_structure,
            set_com_stationary=self.set_com_stationary,
            set_zero_rotation=self.set_zero_rotation,
            relax_calc_kwargs=self.relax_calc_kwargs,
            **self.md_kwargs,
        )

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict[str, Any]:
        """Compute diffusivity and conductivity.

        Args:
            structure: Structure, ASE Atoms, or a result dict. If the dict contains
                ``trajectory`` (ASE Atoms or pymatgen Structures), all supplied frames
                are analysed without MD or further equilibration trimming. Set
                temperature, timestep and loginterval to match that trajectory.

        Returns:
            Dict with ``diffusivity`` (cm²/s), ``conductivity`` (mS/cm),
            ``final_structure``, ``trajectory`` and ``_units``, preserving upstream
            fields and the energy outputs from any internal MD run.
        """
        from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer

        result = super().calc(structure)
        # Resolved before MD so that an undeterminable charge fails fast.
        charge = self._mobile_ion_charge(to_pmg_structure(result["final_structure"]))
        if "trajectory" not in result:
            md_calc = self._build_md_calc()
            result.update(md_calc.calc(result))
        trajectory = [to_pmg_structure(frame) for frame in result["trajectory"]]
        analyzer = DiffusionAnalyzer.from_structures(
            trajectory,
            specie=self.specie,
            temperature=self.temperature,
            time_step=self.timestep,  # type: ignore[arg-type]
            step_skip=self.loginterval,
            **self.analyzer_kwargs,
        )
        diffusivity = float(analyzer.diffusivity)
        # Nernst-Einstein with the tracer diffusivity. DiffusionAnalyzer.conductivity takes the charge of an
        # undecorated element from its last subshell occupancy, e.g. 10 for Ag/Cu, 4 for O and 5 for F.
        n_mobile = sum(site.specie.symbol == self.specie for site in trajectory[0])
        volume = trajectory[0].volume * 1e-24  # cm^3
        conductivity = (
            1000 * n_mobile / volume * charge**2 * constants.e**2 * diffusivity / (constants.k * self.temperature)
        )
        result["diffusivity"] = diffusivity
        result["conductivity"] = float(conductivity)
        result["_units"] = self._merge_units(
            result.get("_units"),
            {"diffusivity": "cm^2/s", "conductivity": "mS/cm"},
        )
        return result

    def _mobile_ion_charge(self, structure: Structure) -> float:
        """Charge of the mobile ion for the conductivity, as described for ``charge`` in ``__init__``."""
        if self.charge is not None:
            return self.charge
        oxi_states = {
            getattr(site.specie, "oxi_state", None) for site in structure if site.specie.symbol == self.specie
        }
        if len(oxi_states) == 1 and (oxi_state := oxi_states.pop()):
            return oxi_state
        try:
            guesses = structure.composition.element_composition.oxi_state_guesses(max_sites=-50)
        except ValueError:  # reduced composition above 50 atoms
            guesses = ()
        if guesses and guesses[0].get(self.specie):
            return guesses[0][self.specie]
        common = Element(self.specie).common_oxidation_states
        if not common:
            raise ValueError(f"Cannot determine the charge of {self.specie}; pass it as charge.")
        warnings.warn(
            f"No charge-balanced oxidation states for {structure.composition.reduced_formula}; using the "
            f"common oxidation state {common[0]} of {self.specie} for the conductivity. Pass charge to override.",
            UserWarning,
            stacklevel=3,
        )
        return common[0]
