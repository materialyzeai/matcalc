"""Calculator for ionic diffusivity and conductivity via MD and DiffusionAnalyzer."""

from __future__ import annotations

from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from ase import units

from ._base import PropCalc
from ._lammps import LAMMPSMDCalc
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

    Runs MD with :class:`MDCalc` or :class:`LAMMPSMDCalc` and analyses the equilibrated trajectory with
    pymatgen's DiffusionAnalyzer. An input dict containing ``trajectory`` skips
    MD and uses those frames directly. Requires the ``diffusion`` extra.
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        specie: str,
        temperature: float,
        timestep: float,
        loginterval: int,
        backend: Literal["ase", "lammps"] = "ase",
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
            temperature: Simulation temperature in Kelvin, also used for the conductivity. For a supplied
                trajectory, the temperature it was run at.
            timestep: MD time step in femtoseconds. For a supplied trajectory, the step it was run with.
            loginterval: Steps between saved frames, also used as the analyzer's ``step_skip``. For a
                supplied trajectory, the interval it was saved with.
            backend: MD engine. LAMMPS requires a model name string and runs in a temporary directory.
            ensemble: Ensemble supported by the selected MD engine. Use NVT or NVE: the analysis unwraps
                displacements on each frame's cell, which is only approximate when the cell fluctuates.
            steps: Total number of MD steps.
            equilibration_frames: Leading saved frames discarded by the internal MD run. The default of 0
                keeps the thermostat transient from the Maxwell-Boltzmann start in the MSD, which biases
                the linear fit for short runs. Set it to cover at least the equilibration period, e.g.
                a few ps (``equilibration_frames * loginterval * timestep`` fs).
            taut: Thermostat time constant in femtoseconds.
            seed: Seed for initial velocities on each MD run; must be positive for LAMMPS.
                None uses a random seed.
            relax_structure: Whether to relax the structure before MD.
            relax_atoms_only: Relax atomic positions only, keeping the input cell. Set False to also
                relax the cell; this gives the 0 K volume, which excludes thermal expansion and
                typically lowers D at the elevated temperatures used for Arrhenius fits.
            set_com_stationary: Remove initial center-of-mass momentum.
            set_zero_rotation: Remove initial angular momentum.
            relax_calc_kwargs: Extra arguments for RelaxCalc when relaxation is enabled. A
                ``relax_cell`` entry here takes precedence over ``relax_atoms_only``.
            analyzer_kwargs: Extra arguments for DiffusionAnalyzer, e.g. ``smoothed``.
            **md_kwargs: Additional arguments forwarded to the selected MD calculator in its native units.
                Relative LAMMPS file paths are temporary; absolute paths retain explicit outputs.
        """
        if backend not in ("ase", "lammps"):
            raise ValueError(f"Unknown backend {backend!r}. Choose 'ase' or 'lammps'.")
        self.calculator = calculator
        self._calculator_arg = calculator
        self.backend = backend
        self.specie = specie
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

    def _build_md_calc(self) -> MDCalc | LAMMPSMDCalc:
        """Build the MD calculator with a fresh initial-velocity generator."""
        if self.backend == "lammps":
            if not isinstance(self._calculator_arg, str):
                raise TypeError("LAMMPS requires a model name string.")
            seed = self.seed if self.seed is not None else int(np.random.default_rng().integers(1, 2**31 - 1))
            return LAMMPSMDCalc(
                self._calculator_arg,
                ensemble=self.ensemble,  # type: ignore[arg-type]
                temperature=self.temperature,
                timestep=self.timestep / 1000,
                taut=self.taut / 1000,
                steps=self.steps,
                loginterval=self.loginterval,
                equilibration_frames=self.equilibration_frames,
                seed=seed,
                relax_structure=self.relax_structure,
                set_com_stationary=self.set_com_stationary,
                set_zero_rotation=self.set_zero_rotation,
                relax_calc_kwargs=self.relax_calc_kwargs,
                **self.md_kwargs,
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
        if "trajectory" not in result:
            md_calc = self._build_md_calc()
            if isinstance(md_calc, LAMMPSMDCalc):
                with TemporaryDirectory(prefix="matcalc-diffusion-") as directory:
                    result.update(md_calc.calc(result, directory=directory))
            else:
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
        result["diffusivity"] = float(analyzer.diffusivity)
        result["conductivity"] = float(analyzer.conductivity)
        result["_units"] = self._merge_units(
            result.get("_units"),
            {"diffusivity": "cm^2/s", "conductivity": "mS/cm"},
        )
        return result
