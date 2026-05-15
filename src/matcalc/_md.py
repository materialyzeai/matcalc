"""Calculator for MD properties."""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from ase import Atoms, units
from ase.io import read as ase_read
from ase.io.trajectory import TrajectoryWriter
from ase.md import Langevin
from ase.md.andersen import Andersen
from ase.md.bussi import Bussi
from ase.md.nose_hoover_chain import MTKNPT, IsotropicMTKNPT, NoseHooverChainNVT
from ase.md.npt import NPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet

from ._base import PropCalc
from .utils import to_ase_atoms, to_pmg_structure

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure


class MDCalc(PropCalc):
    """
    Performs molecular dynamics (MD) simulations on the input structure to calculate the final energy and
    obtain the final structure. Optionally, the structure is relaxed prior to the MD simulation.
    The simulation setup is flexible, supporting various ensembles (e.g., "nvt", "nve", "npt", etc.) and
    user-defined parameters such as temperature, time step, and number of steps.
    """

    def __init__(
        self,
        calculator: Calculator,
        *,
        ensemble: Literal[
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
        ] = "nvt",
        temperature: int = 300,
        timestep: float = 1.0,
        steps: int = 100,
        pressure: float = 1.01325 * units.bar,
        taut: float | None = None,
        taup: float | None = None,
        friction: float = 1.0e-3,
        andersen_prob: float = 1.0e-2,
        ttime: float = 25.0,
        pfactor: float = 75.0**2.0,
        external_stress: float | np.ndarray | None = None,
        compressibility_au: float | None = None,
        tchain: int = 3,
        pchain: int = 3,
        tloop: int = 1,
        ploop: int = 1,
        trajfile: Any = None,
        logfile: str | None = None,
        loginterval: int = 1,
        append_trajectory: bool = False,
        mask: tuple | np.ndarray | None = None,
        relax_structure: bool = True,
        fmax: float = 0.1,
        optimizer: str = "FIRE",
        frames: int | None = None,
        equilibration_frames: int = 0,
        relax_calc_kwargs: dict | None = None,
        set_com_stationary: bool = False,
        set_zero_rotation: bool = False,
    ) -> None:
        """
        Initializes an MDCalc instance with the specified simulation parameters and relaxation settings.

        Parameters:
            calculator (Calculator): The calculator used for energy, force, and stress evaluations.
                Default to the provided calculator.
            ensemble (str): Ensemble for MD simulation. Options include "nve", "nvt_langevin",
                "nvt_andersen", "nvt_bussi", "npt", "npt_berendsen", "npt_nose_hoover", "npt_mtk",
                "npt_isotropic_mtk". Default to "nvt".
            temperature (int): Simulation temperature in Kelvin. Default to 300.
            timestep (float): Time step in femtoseconds. Default to 1.0.
            steps (int): Number of MD simulation steps. Default to 100.
            pressure (float): External pressure for NPT simulations (in eV/Å³). Default to 1.01325 * units.bar.
            taut (float | None): Time constant for temperature coupling. If None, defaults to 100 * timestep * fs.
                For npt_mtk and npt_isotropic_mtk, this is the time constant for temperature damping.
            taup (float | None): Time constant for pressure coupling. If None, defaults to 1000 * timestep * fs.
                For npt_mtk and npt_isotropic_mtk, this is the time constant for pressure damping.
            friction (float): Friction coefficient for Langevin dynamics. Default to 1.0e-3.
            andersen_prob (float): Collision probability for Andersen thermostat. Default to 1.0e-2.
            ttime (float): Characteristic time scale for the thermostat in ASE units (fs). Default to 25.0.
            pfactor (float): Barostat differential equation constant. Default to 75.0**2.0.
            external_stress (float | np.ndarray | None): External stress applied to the system.
                If not provided, defaults to 0.0.
            compressibility_au (float | None): Material compressibility in Å³/eV. Default to None.
            tchain (int): The number of thermostat variables in the Nose-Hoover thermostat. Default to 3.
                Only used by IsotropicMTKNPT and MTKNPT.
            pchain (int): The number of barostat variables in the Nose-Hoover barostat. Default to 3.
                Only used by IsotropicMTKNPT and MTKNPT.
            tloop (int): The number of sub-steps in thermostat integration. Default to 1.
                Only used by IsotropicMTKNPT and MTKNPT.
            ploop (int): T The number of sub-steps in barostat integration. Default to 1.
                Only used by IsotropicMTKNPT and MTKNPT.
            trajfile (Any): Trajectory object or file for storing simulation data. Default to None.
            logfile (str | None): Filename for simulation logs. Default to None.
            loginterval (int): Interval (in steps) for logging simulation data. Default to 1.
            append_trajectory (bool): Whether to append to an existing trajectory file. Default to False.
            mask (tuple | np.ndarray | None): Constraint mask for NPT simulation cell deformations.
                If not provided, defaults to np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)]).
            relax_structure (bool): Whether to relax the input structure before MD simulation. Default to True.
            fmax (float): Maximum force tolerance for structure relaxation (in eV/Å). Default to 0.1.
            optimizer (str): Optimizer used for structure relaxation. Default to "FIRE".
            frames (int): Number of trailing MD frames to average energies over. Default to None,
                which means all frames (i.e. frames = steps).
            equilibration_frames (int): Skip this many leading frames before the trailing
                ``frames`` window. Use this to discard non-equilibrated state at the start of an MD
                run. Default 0.
            relax_calc_kwargs (dict | None): Additional keyword arguments for the relaxation calculation.
                Default to None.
            set_com_stationary (bool): Whether to set the center-of-mass momentum to zero after setting up the
                Maxwell-Boltzmann distribution.
                Default to False.
            set_zero_rotation (bool): Whether to set the total angular momentum to zero after setting up the
                Maxwell-Boltzmann distribution.
                Default to False.
        """
        self.calculator = calculator
        self.ensemble = ensemble
        self.temperature = temperature
        self.timestep = timestep
        self.steps = steps
        self.pressure = pressure
        self.taut = taut
        self.taup = taup
        self.friction = friction
        self.andersen_prob = andersen_prob
        self.ttime = ttime
        self.pfactor = pfactor
        self.external_stress = external_stress
        self.compressibility_au = compressibility_au
        self.tchain = tchain
        self.pchain = pchain
        self.tloop = tloop
        self.ploop = ploop
        self.trajfile = trajfile
        self.logfile = logfile
        self.loginterval = loginterval
        self.append_trajectory = append_trajectory
        self.mask = mask
        self.relax_structure = relax_structure
        self.fmax = fmax
        self.optimizer = optimizer
        self.frames = frames if frames is not None else self.steps
        self.equilibration_frames = equilibration_frames
        self.relax_calc_kwargs = relax_calc_kwargs
        self.set_com_stationary = set_com_stationary
        self.set_zero_rotation = set_zero_rotation

    def _initialize_md(self, atoms: Atoms) -> Any:  # noqa: C901, PLR0911
        """
        Initializes the MD simulation object based on the provided ASE atoms object and simulation parameters.

        Parameters:
            atoms (Atoms): ASE atoms object representing the structure.

        Returns:
            An MD simulation object (e.g., an instance of NVTBerendsen, VelocityVerlet, etc.)
        """
        atoms.calc = self.calculator

        timestep_fs = self.timestep * units.fs
        taut = self.taut if self.taut is not None else 100 * self.timestep * units.fs
        taup = self.taup if self.taup is not None else 1000 * self.timestep * units.fs
        mask = self.mask if self.mask is not None else np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        external_stress = self.external_stress if self.external_stress is not None else 0.0
        ensemble = self.ensemble.lower()

        if ensemble == "nve":
            return VelocityVerlet(
                atoms,
                timestep_fs,
                logfile=self.logfile,
                loginterval=self.loginterval,
            )
        if ensemble in ("nvt", "nvt_nose_hoover"):
            return NoseHooverChainNVT(
                atoms,
                timestep_fs,
                tdamp=taut,
                temperature_K=self.temperature,
                logfile=self.logfile,
                loginterval=self.loginterval,
            )
        if ensemble == "nvt_berendsen":
            return NVTBerendsen(
                atoms,
                timestep_fs,
                temperature_K=self.temperature,
                taut=taut,
                logfile=self.logfile,
                loginterval=self.loginterval,
            )
        if ensemble == "nvt_langevin":
            return Langevin(
                atoms,
                timestep_fs,
                temperature_K=self.temperature,
                friction=self.friction,
                logfile=self.logfile,
                loginterval=self.loginterval,
            )
        if ensemble == "nvt_andersen":
            return Andersen(
                atoms,
                timestep_fs,
                temperature_K=self.temperature,
                andersen_prob=self.andersen_prob,
                logfile=self.logfile,
                loginterval=self.loginterval,
            )
        if ensemble == "nvt_bussi":
            return Bussi(
                atoms,
                timestep_fs,
                temperature_K=self.temperature,
                taut=taut,
                logfile=self.logfile,
                loginterval=self.loginterval,
            )
        if ensemble in ("npt", "npt_nose_hoover"):
            self._upper_triangular_cell(atoms)
            return NPT(
                atoms,
                timestep_fs,
                temperature_K=self.temperature,
                externalstress=external_stress,  # type: ignore[arg-type]
                ttime=self.ttime * units.fs,
                pfactor=self.pfactor * units.fs,
                logfile=self.logfile,
                loginterval=self.loginterval,
                mask=mask,
            )
        if ensemble == "npt_berendsen":
            return NPTBerendsen(
                atoms,
                timestep_fs,
                temperature_K=self.temperature,
                pressure_au=self.pressure,
                taut=taut,
                taup=taup,
                compressibility_au=self.compressibility_au,
                logfile=self.logfile,
                loginterval=self.loginterval,
            )
        if ensemble == "npt_inhomogeneous":
            return Inhomogeneous_NPTBerendsen(
                atoms,
                timestep_fs,
                temperature_K=self.temperature,
                pressure_au=self.pressure,
                taut=taut,
                taup=taup,
                compressibility_au=self.compressibility_au,
                logfile=self.logfile,
                loginterval=self.loginterval,
            )
        if ensemble == "npt_mtk":
            return MTKNPT(
                atoms,
                timestep=timestep_fs,
                temperature_K=self.temperature,
                pressure_au=self.pressure,
                tdamp=taut,
                pdamp=taup,
                tchain=self.tchain,
                pchain=self.pchain,
                tloop=self.tloop,
                ploop=self.ploop,
                logfile=self.logfile,
                loginterval=self.loginterval,
            )
        if ensemble == "npt_isotropic_mtk":
            return IsotropicMTKNPT(
                atoms,
                timestep=timestep_fs,
                temperature_K=self.temperature,
                pressure_au=self.pressure,
                tdamp=taut,
                pdamp=taup,
                tchain=self.tchain,
                pchain=self.pchain,
                tloop=self.tloop,
                ploop=self.ploop,
                logfile=self.logfile,
                loginterval=self.loginterval,
            )

        raise ValueError(
            "The specified ensemble is not supported, choose from 'nve', 'nvt',"
            " 'nvt_nose_hoover', 'nvt_berendsen', 'nvt_langevin', 'nvt_andersen',"
            " 'nvt_bussi', 'npt', 'npt_nose_hoover', 'npt_berendsen', 'npt_inhomogeneous',"
            " 'npt_mtk', 'npt_isotropic_mtk'."
        )

    def _upper_triangular_cell(self, atoms: Atoms) -> None:
        """
        Transforms the cell of the given atoms object to an upper triangular form if it is not already,
        as required by the Nose-Hoover NPT implementation.

        Parameters:
            atoms (Atoms): ASE atoms object.
        """
        if not atoms.cell[1, 0] == atoms.cell[2, 0] == atoms.cell[2, 1] == 0.0:
            a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
            angles = np.radians((alpha, beta, gamma))
            sin_a, sin_b, _ = np.sin(angles)
            cos_a, cos_b, cos_g = np.cos(angles)
            cos_p = (cos_g - cos_a * cos_b) / (sin_a * sin_b)
            cos_p = np.clip(cos_p, -1, 1)
            sin_p = np.sqrt(1 - cos_p**2)
            new_basis = [
                (a * sin_b * sin_p, a * sin_b * cos_p, a * cos_b),
                (0, b * sin_a, b * cos_a),
                (0, 0, c),
            ]
            atoms.set_cell(new_basis, scale_atoms=True)

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict[str, Any]:
        """
        Performs the MD simulation calculation for the input structure. Prior to generating initial velocities,
        this method calls the superclass's calc method for preprocessing and optionally relaxes the structure.
        It then initializes the atomic velocities using the Maxwell-Boltzmann distribution, runs the MD simulation,
        and returns the final energy along with the final atomic configuration.

        Parameters:
            structure (Structure | Atoms | dict[str, Any]): Input structure as a Structure instance or a dictionary.

        Returns:
            dict: A dictionary with ``final_structure``, ``trajectory`` (list of ASE Atoms),
                ``potential_energy`` (eV), ``kinetic_energy`` (eV), ``total_energy`` (eV)
                averaged over the last ``frames`` frames, ``_units`` mapping each numeric
                output to its unit string, plus any relaxation fields merged in.
        """
        # Preprocess the input structure using the superclass's calc method.
        # This initial processing returns a dictionary containing a "final_structure".
        result = super().calc(structure)
        # If structure relaxation is enabled, relax the structure (atoms only,
        # fixed cell) before the MD simulation.
        result, structure_in = self._prerelax(
            result["final_structure"],
            result,
            fmax=self.fmax,
            optimizer=self.optimizer,
            relax_atoms=True,
            relax_cell=False,
        )

        # Convert the structure to an ASE atoms object,
        # which is required for subsequent molecular dynamics (MD) simulation.
        atoms = to_ase_atoms(structure_in)

        # Initialize the atomic velocities based on the Maxwell-Boltzmann distribution
        # at the specified temperature, ensuring proper kinetic energy.
        MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature)

        if self.set_com_stationary:
            Stationary(atoms)

        if self.set_zero_rotation:
            ZeroRotation(atoms)

        # Initialize the molecular dynamics (MD) simulation and set up the simulation parameters.
        md = self._initialize_md(atoms)

        # Set up ASE native Trajectory as observer, writing frames at loginterval steps.
        use_temp = self.trajfile is None
        if use_temp:
            with tempfile.NamedTemporaryFile(suffix=".traj", delete=False) as tmp_file:
                traj_path = tmp_file.name
        else:
            traj_path = str(self.trajfile)

        mode = "a" if self.append_trajectory else "w"
        with TrajectoryWriter(traj_path, mode, atoms) as ase_traj:
            md.attach(ase_traj.write, interval=self.loginterval)
            # Run the MD simulation for the specified number of steps.
            md.run(self.steps)

        # Read back all frames.
        all_frames = ase_read(traj_path, index=":")

        if use_temp:
            os.unlink(traj_path)

        result["final_structure"] = to_pmg_structure(all_frames[-1])  # type: ignore[arg-type]

        # Drop equilibration_frames from the start, then take the trailing ``frames`` window.
        eq_skipped = all_frames[self.equilibration_frames :] if self.equilibration_frames else all_frames
        traj = eq_skipped[-self.frames :]
        n_avg = len(traj)
        if n_avg == 0:
            raise ValueError(
                f"No frames left to average after dropping {self.equilibration_frames=} "
                f"and taking the last {self.frames=}; got {len(all_frames)} total frames."
            )

        # Calculate the average potential, kinetic and total energies over the selected window.
        energy_pot = sum(a.get_potential_energy() for a in traj) / n_avg
        energy_kin = sum(a.get_kinetic_energy() for a in traj) / n_avg
        energy_tot = sum(a.get_total_energy() for a in traj) / n_avg

        # Update the result dictionary with the simulation trajectory and computed energy.
        result |= {
            "trajectory": traj,
            "potential_energy": energy_pot,
            "kinetic_energy": energy_kin,
            "total_energy": energy_tot,
            "_units": {
                **result.get("_units", {}),
                "potential_energy": "eV",
                "kinetic_energy": "eV",
                "total_energy": "eV",
            },
        }

        # Return the complete result dictionary.
        return result
