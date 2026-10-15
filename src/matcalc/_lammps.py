"""Calculator for MD properties using LAMMPS."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.lammps.inputs import LammpsRun
from pymatgen.io.lammps.outputs import parse_lammps_dumps, parse_lammps_log

from ._base import PropCalc
from ._relaxation import RelaxCalc
from .utils import MODEL_REGISTRY, _resolve_canonical, to_pmg_structure

if TYPE_CHECKING:
    from typing import Any, Literal

    from ase import Atoms

LAMMPS_TEMPLATES_DIR = Path(__file__).parent / "lammps_templates"


class LAMMPSMDCalc(PropCalc):
    """
    Class to manage molecular dynamics simulations using LAMMPS.

    Attributes:
        calculator: Universal model name string (resolved for MatGL potentials).
        temperature: Target temperature (K).
        ensemble: Thermostat/barostat choice.
        timestep: MD timestep (ps).
        steps: Total MD steps.
        pressure: Target pressure for NPT (bar).
        taut, taup: Thermo/barostat time constants (ps).
        infile, trajfile, logfile: LAMMPS IO paths.
        loginterval: Steps between saved frames and between thermo outputs.
        relax_structure: Pre-MD ``RelaxCalc`` when True.
        fmax, optimizer: Relaxation settings.
        frames: Number of trailing trajectory frames to return, or None for all frames.
        equilibration_frames: Leading saved frames to discard before selecting trailing frames.
        seed: Positive integer seed for initial velocities.
        set_com_stationary, set_zero_rotation: Remove initial linear and angular momentum.
        settings: Extra template placeholders for ``LammpsRun``.
        relax_calc_kwargs: Kwargs for pre-MD ``RelaxCalc``.
    """

    def __init__(
        self,
        calculator: str,
        *,
        temperature: float = 300,
        ensemble: Literal["nve", "nvt", "npt", "nvt_nose_hoover", "npt_nose_hoover"] = "nvt",
        timestep: float = 0.001,
        steps: int = 100,
        pressure: float = 1,
        taut: float | None = None,
        taup: float | None = None,
        infile: str = "in.md",
        trajfile: str = "md.lammpstrj",
        logfile: str = "log.lammps",
        loginterval: int = 1,
        relax_structure: bool = True,
        fmax: float = 0.1,
        optimizer: str = "FIRE",
        frames: int | None = None,
        equilibration_frames: int = 0,
        seed: int = 142857,
        set_com_stationary: bool = True,
        set_zero_rotation: bool = True,
        settings: dict | None = None,
        relax_calc_kwargs: dict | None = None,
    ) -> None:
        """Initialize LAMMPSMDCalc with simulation parameters."""
        self.model_name = calculator
        self.calculator = calculator  # type: ignore[assignment]
        self.ensemble = ensemble
        self.timestep = timestep
        self.temperature = temperature
        self.pressure = pressure
        self.steps = steps
        self.taut = taut if taut is not None else 100 * timestep
        self.taup = taup if taup is not None else 1000 * timestep
        self.infile = infile
        self.trajfile = trajfile
        self.logfile = logfile
        self.loginterval = loginterval
        self.relax_structure = relax_structure
        self.fmax = fmax
        self.optimizer = optimizer
        self.frames = frames
        self.equilibration_frames = equilibration_frames
        self.seed = seed
        self.set_com_stationary = set_com_stationary
        self.set_zero_rotation = set_zero_rotation
        self.settings = settings
        self.relax_calc_kwargs = relax_calc_kwargs

        canonical = _resolve_canonical(self.model_name)
        if canonical is not None and MODEL_REGISTRY[canonical]["provider"] == "matgl":
            self.model_name = MODEL_REGISTRY[canonical]["path"]
            self.gnnp_type = "matgl"
        else:
            self.gnnp_type = self.model_name.lower()

        self.fix_command = self._generate_fix_command()

    def _generate_fix_command(self) -> str:
        """Generate LAMMPS fix command based on the selected ensemble."""
        if self.ensemble in ("nvt", "nvt_nose_hoover"):
            return f"fix             1 all nvt temp {self.temperature} {self.temperature} {self.taut}"
        if self.ensemble in ("npt", "npt_nose_hoover"):
            return (
                f"fix             1 all npt temp {self.temperature} {self.temperature} {self.taut} "
                f"iso {self.pressure} {self.pressure} {self.taup}"
            )
        if self.ensemble == "nve":
            return "fix             1 all nve"
        raise ValueError(
            "The specified ensemble is not supported, choose from 'nve', 'nvt',"
            " 'nvt_nose_hoover', 'npt', 'npt_nose_hoover'."
        )

    def write_inputs(
        self,
        structure: Structure | Atoms | dict[str, Any],
        script_template: str | Path = Path(LAMMPS_TEMPLATES_DIR / "md.template"),
        *,
        directory: str | Path = ".",
    ) -> LammpsRun:
        """Write LAMMPS input files based on a given structure.

        Args:
            structure: Structure, atoms, or dict accepted by ``PropCalc.calc``.
            script_template: Path or string content of the LAMMPS script template.
            directory: Directory for input files and relative output paths.

        Returns:
            ``LammpsRun`` with inputs written to the selected directory.
        """
        # Preprocess the input structure using the superclass's calc method.
        # This initial processing returns a dictionary containing a "final_structure".
        result = super().calc(structure)
        structure_in = to_pmg_structure(result["final_structure"])

        # If structure relaxation is enabled, relax the structure (atoms only) before the MD simulation.
        if self.relax_structure:
            # Create a RelaxCalc instance with the specified calculator, convergence criteria (fmax),
            # optimizer, and any additional keyword arguments for the relaxation calculation.
            relaxer = RelaxCalc(
                self.calculator,
                fmax=self.fmax,
                optimizer=self.optimizer,
                **{"relax_atoms": True, "relax_cell": False, **(self.relax_calc_kwargs or {})},
            )
            # Run the relaxation calculation and update the result dictionary.
            result |= relaxer.calc(structure_in)
            # Update the input structure with the relaxed final structure.
            structure_in = result["final_structure"]

        lmpdata = LammpsData.from_structure(structure_in, atom_style="atomic")

        default_settings = {
            "logfile": self.logfile,
            "trajfile": self.trajfile,
            "gnnp_type": self.gnnp_type,
            "model_name": self.model_name,
            "symbol_set": " ".join(element.symbol for element in sorted(structure_in.composition.element_composition)),
            "temperature": self.temperature,
            "timestep": self.timestep,
            "steps": self.steps,
            "loginterval": self.loginterval,
            "taut": self.taut,
            "taup": self.taup,
            "pressure": self.pressure,
            "fix_command": self.fix_command,
            "seed": self.seed,
            "mom": "yes" if self.set_com_stationary else "no",
            "rot": "yes" if self.set_zero_rotation else "no",
        }
        full_settings = {**default_settings, **(self.settings or {})}

        if not isinstance(script_template, str):
            with open(script_template) as f:
                script_template = f.read()

        lammps_run = LammpsRun(
            script_template=script_template,
            settings=full_settings,
            data=lmpdata,
            script_filename=self.infile,
        )
        lammps_run.write_inputs(output_dir=str(directory))
        return lammps_run

    def calc(self, structure: Structure | Atoms | dict[str, Any], *, directory: str | Path = ".") -> dict:
        """Run MD and return the selected frames and the mean energies over the steps they span.

        Args:
            structure: Structure, ASE Atoms, or upstream result dict.
            directory: Working directory for LAMMPS and its relative input/output paths. If the input
                script and ``md.data`` already exist there, e.g. from ``write_inputs`` with a custom
                ``script_template``, they are run as is and ``structure`` is not written.

        Returns:
            Result dict including the selected trajectory and the mean thermo energies from its first
            to its last frame.
        """
        # Preprocess the input structure using the superclass's calc method.
        # This initial processing returns a dictionary containing a "final_structure".
        result = super().calc(structure)
        structure_in: Structure = result["final_structure"]
        directory = Path(directory)

        if not ((directory / self.infile).exists() and (directory / "md.data").exists()):
            self.write_inputs(structure_in, directory=directory)

        subprocess.run(["lmp", "-in", self.infile], check=True, cwd=directory)  # noqa: S603, S607

        # Create a list to record the state of atoms at each simulation step.
        traj = []
        frame_steps = []
        for snapshot in parse_lammps_dumps(str(directory / self.trajfile)):
            lattice = snapshot.box.to_lattice()
            coords = snapshot.data[["x", "y", "z"]].to_numpy()
            species = snapshot.data["element"].tolist()
            struct = Structure(lattice, species, coords, coords_are_cartesian=True)

            traj.append(AseAtomsAdaptor.get_atoms(struct))
            frame_steps.append(snapshot.timestep)
        traj = traj[self.equilibration_frames :]
        frame_steps = frame_steps[self.equilibration_frames :]
        if self.frames is not None:
            traj = traj[-self.frames :]
            frame_steps = frame_steps[-self.frames :]
        if not traj:
            raise ValueError(
                f"No frames left after dropping {self.equilibration_frames=} and selecting {self.frames=}."
            )

        thermo = parse_lammps_log(str(directory / self.logfile))[0]
        # Average all thermo output over the steps spanned by the selected frames.
        thermo = thermo[thermo["Step"].between(frame_steps[0], frame_steps[-1])]
        if thermo.empty:
            raise ValueError(f"No thermo output between steps {frame_steps[0]} and {frame_steps[-1]}.")
        energy_pot = thermo["PotEng"].mean()
        energy_kin = thermo["KinEng"].mean()
        energy_tot = thermo["TotEng"].mean()

        # Update the result dictionary with the simulation trajectory and computed energy.
        result |= {
            "final_structure": to_pmg_structure(traj[-1]),
            "trajectory": traj,
            "potential_energy": energy_pot,
            "kinetic_energy": energy_kin,
            "total_energy": energy_tot,
            "_units": self._merge_units(
                result,
                {
                    "potential_energy": "eV",
                    "kinetic_energy": "eV",
                    "total_energy": "eV",
                },
            ),
        }

        # Return the complete result dictionary.
        return result
