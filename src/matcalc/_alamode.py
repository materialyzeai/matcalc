"""Calculator for phonon properties using alamode."""

from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import phonopy
from ase.units import Bohr, Rydberg
from lxml import etree
from phonopy.file_IO import (
    parse_FORCE_CONSTANTS,
    write_FORCE_CONSTANTS,
)
from phonopy.harmonic.force_constants import (
    compact_fc_to_full_fc,
)
from phonopy.interface.vasp import write_vasp
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

# import pymatgen libraries to determine supercell
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)

from ._base import PropCalc
from ._relaxation import RelaxCalc

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from ase.calculators.calculator import Calculator
    from numpy.typing import ArrayLike
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


class AlamodeCalc(PropCalc):
    """
    A class for phonon, higher-order force constants and thermal property calculations using pheasy.

    The `PhononCalc` class extends the `PropCalc` class to provide
    functionalities for calculating phonon properties and thermal properties
    of a given structure using the phonopy library. It includes options for
    structural relaxation before phonon property and higher-order force constants
    determination, as well as methods to export calculated properties to
    various output files for further analysis or visualization.

    :ivar calculator: A calculator object or a string specifying the
        computational backend to be used.
    :type calculator: Calculator | str
    :ivar atom_disp: Magnitude of atomic displacements for phonon
        calculations.
    :type atom_disp: float
    :ivar supercell_matrix: Array defining the transformation matrix to
        construct supercells for phonon calculations.
    :type supercell_matrix: ArrayLike
    :ivar t_step: Temperature step for thermal property calculations in
        Kelvin.
    :type t_step: float
    :ivar t_max: Maximum temperature for thermal property calculations in
        Kelvin.
    :type t_max: float
    :ivar t_min: Minimum temperature for thermal property calculations in
        Kelvin.
    :type t_min: float
    :ivar fmax: Maximum force convergence criterion for structural relaxation.
    :type fmax: float
    :ivar optimizer: String specifying the optimizer type to be used for
        structural relaxation.
    :type optimizer: str
    :ivar relax_structure: Boolean flag to determine whether to relax the
        structure before phonon calculation.
    :type relax_structure: bool
    :ivar relax_calc_kwargs: Optional dictionary containing additional
        arguments for the structural relaxation calculation.
    :type relax_calc_kwargs: dict | None
    :ivar write_force_constants: Path, boolean, or string specifying whether
        to write the calculated force constants to an output file, and the
        path or name of the file if applicable.
    :type write_force_constants: bool | str | Path
    :ivar write_band_structure: Path, boolean, or string specifying whether
        to write the calculated phonon band structure to an output file,
        and the path or name of the file if applicable.
    :type write_band_structure: bool | str | Path
    :ivar write_total_dos: Path, boolean, or string specifying whether to
        write the calculated total density of states (DOS) to an output
        file, and the path or name of the file if applicable.
    :type write_total_dos: bool | str | Path
    :ivar write_phonon: Path, boolean, or string specifying whether to write
        the calculated phonon properties (e.g., phonon.yaml) to an output
        file, and the path or name of the file if applicable.
    :type write_phonon: bool | str | Path
    :ivar fitting_method: Method for fitting force constants. Options are
        "FDM", "LASSO", or "MD".
    :type fitting_method: str
    :ivar num_harmonic_snapshots: Number of snapshots for harmonic fitting.
        If None, it is set to double the number of displacements.
    :type num_harmonic_snapshots: Optional[int]
    :ivar num_anharmonic_snapshots: Number of snapshots for anharmonic
        fitting. If None, it is set to ten times the number of displacements.
    :type num_anharmonic_snapshots: Optional[int]
    :ivar calc_anharmonic: Boolean flag to determine whether to perform
        anharmonic calculations.
    :type calc_anharmonic: bool
    :ivar symprec: Symmetry precision for structure symmetry analysis.
    :type symprec: float
    :ivar min_length: Minimum length for the supercell.
    :type min_length: float
    :ivar force_diagonal: Boolean flag to determine whether to force the
        diagonalization of the supercell.
    :type force_diagonal: bool
    :ivar cutoff_distance_cubic: Cutoff distance for cubic force constants.
    :type cutoff_distance_cubic: float
    :ivar cutoff_distance_quartic: Cutoff distance for quartic force
        constants.
    :type cutoff_distance_quartic: float
    :ivar cutoff_distance_quintic: Cutoff distance for quintic force
        constants.
    :type cutoff_distance_quintic: float
    :ivar cutoff_distance_sextic: Cutoff distance for sextic force
        constants.
    :type cutoff_distance_sextic: float
    :ivar no_lasso_fitting_anhar: Boolean flag to determine whether to
        disable LASSO fitting for anharmonic calculations.
    :type no_lasso_fitting_anhar: bool
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        atom_disp: float = 0.015,
        atom_disp_anhar: float = 0.1,
        supercell_matrix: ArrayLike | None = None,
        t_step: float = 10,
        t_max: float = 1000,
        t_min: float = 0,
        fmax: float = 0.1,
        optimizer: str = "FIRE",
        relax_structure: bool = True,
        relax_calc_kwargs: dict | None = None,
        write_force_constants: bool | str | Path = False,
        write_band_structure: bool | str | Path = False,
        write_total_dos: bool | str | Path = False,
        write_phonon: bool | str | Path = True,
        fitting_method: str = "LASSO",
        num_harmonic_snapshots: int | None = None,
        num_anharmonic_snapshots: int | None = None,
        calc_anharmonic: bool = False,
        symprec: float = 1e-5,
        min_length: float = 12.0,
        force_diagonal: bool = True,
        cutoff_distance_cubic: float = 6.3,
        cutoff_distance_quartic: float = 5.3,
        cutoff_distance_quintic: float = 3.3,
        cutoff_distance_sextic: float = 3.3,
        no_lasso_fitting_anhar: bool = False,
    ) -> None:
        """
        Initializes the class with configuration for the phonon calculations. The initialization parameters control
        the behavior of structural relaxation, thermal properties, force calculations, and output file generation.
        The class allows for customization of key parameters to facilitate the study of material behaviors.

        :param calculator: The calculator object or string name specifying the calculation backend to use.
        :param atom_disp: Atom displacement to be used for finite difference calculation of force constants.
        :param atom_disp_anhar: Atom displacement to be used for anharmonic calculation of force constants.
        :param supercell_matrix: Transformation matrix to define the supercell for the calculation.
        :param t_step: Temperature step for thermal property calculations.
        :param t_max: Maximum temperature for thermal property calculations.
        :param t_min: Minimum temperature for thermal property calculations.
        :param fmax: Maximum force during structure relaxation, used as a convergence criterion.
        :param optimizer: Name of the optimization algorithm for structural relaxation.
        :param relax_structure: Flag to indicate whether structure relaxation should be performed before calculations.
        :param relax_calc_kwargs: Additional keyword arguments for relaxation phase calculations.
        :param write_force_constants: File path or boolean flag to write force constants.
            Defaults to "force_constants".
        :param write_band_structure: File path or boolean flag to write band structure data.
            Defaults to "band_structure.yaml".
        :param write_total_dos: File path or boolean flag to write total density of states (DOS) data.
            Defaults to "total_dos.dat".
        :param write_phonon: File path or boolean flag to write phonon data. Defaults to "phonon.yaml".
        :param fitting_method: Method for fitting force constants. Options are "FDM", "LASSO", or "MD".
        :param num_harmonic_snapshots: Number of snapshots for harmonic fitting. If None, it is set to double the
            number of displacements.
        :param num_anharmonic_snapshots: Number of snapshots for anharmonic fitting. If None, it is set to ten times
            the number of displacements.
        :param calc_anharmonic: Flag to indicate whether anharmonic calculations should be performed.
        :param symprec: Symmetry precision for structure symmetry analysis.
        :param min_length: Minimum length for the supercell.
        :param force_diagonal: Flag to indicate whether to force the diagonalization of the supercell.
        :param cutoff_distance_cubic: Cutoff distance for cubic force constants.
        :param cutoff_distance_quartic: Cutoff distance for quartic force constants.
        :param cutoff_distance_quintic: Cutoff distance for quintic force constants.
        :param cutoff_distance_sextic: Cutoff distance for sextic force constants.
        """
        self.calculator = calculator  # type: ignore[assignment]
        self.atom_disp = atom_disp
        self.atom_disp_anhar = atom_disp_anhar
        self.supercell_matrix = supercell_matrix
        self.t_step = t_step
        self.t_max = t_max
        self.t_min = t_min
        self.fmax = fmax
        self.optimizer = optimizer
        self.relax_structure = relax_structure
        self.relax_calc_kwargs = relax_calc_kwargs
        self.write_force_constants = write_force_constants
        self.write_band_structure = write_band_structure
        self.write_total_dos = write_total_dos
        self.write_phonon = write_phonon

        # some new parameters for pheasy
        self.fitting_method = fitting_method
        self.num_harmonic_snapshots = num_harmonic_snapshots
        self.num_anharmonic_snapshots = num_anharmonic_snapshots
        self.calc_anharmonic = calc_anharmonic
        self.symprec = symprec
        self.min_length = min_length
        self.force_diagonal = force_diagonal
        self.cutoff_distance_cubic = cutoff_distance_cubic
        self.cutoff_distance_quartic = cutoff_distance_quartic
        self.cutoff_distance_quintic = cutoff_distance_quintic
        self.cutoff_distance_sextic = cutoff_distance_sextic
        self.no_lasso_fitting_anhar = no_lasso_fitting_anhar

        # Set default paths for output files.
        for key, val, default_path in (
            ("write_force_constants", self.write_force_constants, "force_constants"),
            ("write_band_structure", self.write_band_structure, "band_structure.yaml"),
            ("write_total_dos", self.write_total_dos, "total_dos.dat"),
            ("write_phonon", self.write_phonon, "phonon.yaml"),
        ):
            setattr(self, key, str({True: default_path, False: ""}.get(val, val)))  # type: ignore[arg-type]

    def calc(self, structure: Structure | dict[str, Any]) -> dict:
        """Calculates thermal properties of Pymatgen structure with phonopy.

        Args:
            structure: Pymatgen structure.

        Returns:
        {
            phonon: Phonopy object with force constants produced
            thermal_properties:
                {
                    temperatures: list of temperatures in Kelvin,
                    free_energy: list of Helmholtz free energies at corresponding temperatures in kJ/mol,
                    entropy: list of entropies at corresponding temperatures in J/K/mol,
                    heat_capacity: list of heat capacities at constant volume at corresponding temperatures in J/K/mol,
                    The units are originally documented in phonopy.
                    See phonopy.Phonopy.run_thermal_properties()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/api_phonopy.py#L2591)
                    -> phonopy.phonon.thermal_properties.ThermalProperties.run()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L498)
                    -> phonopy.phonon.thermal_properties.ThermalPropertiesBase.run_free_energy()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L217)
                    phonopy.phonon.thermal_properties.ThermalPropertiesBase.run_entropy()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L233)
                    phonopy.phonon.thermal_properties.ThermalPropertiesBase.run_heat_capacity()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L225)
                }
        }
        """
        result = super().calc(structure)
        structure_in: Structure = result["final_structure"]

        if self.relax_structure:
            relaxer = RelaxCalc(
                self.calculator, fmax=self.fmax, optimizer=self.optimizer, **(self.relax_calc_kwargs or {})
            )
            result |= relaxer.calc(structure_in)
            structure_in = result["final_structure"]

        # generate the primitive cell from the structure
        # let's start the calculation from the primitive cell

        """I donot know why the following code does not work.
        if i apply it and will give a huge error in force calculation"""

        # sga = SpacegroupAnalyzer(structure, symprec=self.symprec)
        # structure_in = sga.get_primitive_standard_structure()

        cell = get_phonopy_structure(structure_in)

        # If the supercell matrix is not provided, we need to determine the
        # supercell matrix from the structure. We use the
        # CubicSupercellTransformation to determine the supercell matrix.
        # The supercell matrix is a 3x3 matrix that defines the transformation
        # from the primitive cell to the supercell. The supercell matrix is
        # used to generate the supercell for the phonon calculations.

        if self.supercell_matrix is None:
            transformation = CubicSupercellTransformation(
                min_length=self.min_length, force_diagonal=self.force_diagonal
            )
            supercell = transformation.apply_transformation(structure_in)
            self.supercell_matrix = np.array(transformation.transformation_matrix.transpose().tolist())
            # transfer it to array

            # self.supercell_matrix = supercell.lattice.matrix
        else:
            transformation = None

        phonon = phonopy.Phonopy(cell, self.supercell_matrix)  # type: ignore[arg-type]

        if self.fitting_method == "FDM":
            phonon.generate_displacements(distance=self.atom_disp)

        elif self.fitting_method == "LASSO":
            if self.num_harmonic_snapshots is None:
                phonon.generate_displacements(distance=self.atom_disp)

                self.num_harmonic_snapshots = len(phonon.displacements) * 2

                phonon.generate_displacements(
                    distance=self.atom_disp, number_of_snapshots=self.num_harmonic_snapshots, random_seed=42
                )

        elif self.fitting_method == "MD":
            # pass
            # phonon.generate_displacements(distance=self.atom_disp, number_of_snapshots=self.num_snapshots)

            logger.info("MD fitting method is not implemented yet.")

        else:
            raise ValueError(f"Unknown fitting method: {self.fitting_method}")

        disp_supercells = phonon.supercells_with_displacements

        disp_array = []

        phonon.forces = [  # type: ignore[assignment]
            _calc_forces(self.calculator, supercell)
            for supercell in disp_supercells  # type:ignore[union-attr]
            if supercell is not None
        ]

        force_equilibrium = _calc_forces(self.calculator, phonon.supercell)  # type: ignore[union-attr]
        phonon.forces = np.array(phonon.forces) - force_equilibrium  # type: ignore[assignment]

        for i, supercell in enumerate(disp_supercells):
            disp = supercell.get_positions() - phonon.supercell.get_positions()
            disp_array.append(np.array(disp))

        logger.info("...Forces calculated for the supercells...")
        logger.info("..Producing force constants...")

        disp_array = np.array(disp_array)

        logger.info("Saving disp_array and phonon.forces in files...")

        # Modify the script to include comment headers for each supercell block

        bohr_per_angstrom = 1.8897259886
        ryd_per_ev_angstrom = 0.036749309

        # Output file path
        output_file = "DFSET_harmonic"

        with open(output_file, "w") as f:
            for i, (disp, force) in enumerate(zip(disp_array, phonon.forces, strict=False)):
                f.write(f"# supercell {i + 1}\n")
                for d, fr in zip(disp, force, strict=False):
                    d_bohr = d * bohr_per_angstrom
                    fr_ryd_bohr = fr * ryd_per_ev_angstrom
                    line = " ".join(f"{val:.8f}" for val in np.concatenate((d_bohr, fr_ryd_bohr)))
                    f.write(line + "\n")

        supercell = phonon.get_supercell()

        logger.info("Writing POSCAR and SPOSCAR files for Pheasy to read...")
        write_vasp("POSCAR", cell)
        write_vasp("SPOSCAR", supercell)

        num_har = disp_array.shape[0]
        supercell_matrix = self.supercell_matrix

        logger.info("start running pheasy for second order force constants in cluster")
        logger.info("if you use this function, please cite the following pheasy paper:")
        logger.info(
            "Lin, Changpeng, Samuel Poncé, and Nicola Marzari. General "
            "invariance and equilibrium conditions for lattice dynamics in 1D, 2D, and "
            "3D materials. npj Computational Materials 8.1 (2022): 236."
        )

        # alamode setting

        # Define the path to your SPOSCAR file
        sposcar_path = "SPOSCAR"

        # Read SPOSCAR content
        with open(sposcar_path) as f:
            lines = f.readlines()

        scaling_factor = float(lines[1].strip())
        lattice_vectors = [list(map(float, lines[i].split())) for i in range(2, 5)]
        elements = lines[5].split()
        num_atoms = list(map(int, lines[6].split()))
        total_atoms = sum(num_atoms)

        # Read atomic positions
        position_lines = lines[8 : 8 + total_atoms]
        positions = []
        for element_idx, count in enumerate(num_atoms):
            for _ in range(count):
                line = position_lines.pop(0)
                coords = list(map(float, line.split()[:3]))
                positions.append([element_idx + 1] + coords)

        # Define namelists (manually write all to control formatting)
        with open("alamode.in", "w") as f:
            # &GENERAL
            f.write("&general\n")
            f.write("  PREFIX = EuZnAs_harmonic\n")
            f.write("  MODE = optimize\n")
            f.write(f"  NAT = {total_atoms}\n")
            f.write(f"  NKD = {len(elements)}\n")
            f.write("  KD = " + " ".join(elements) + "\n")
            f.write("/\n\n")

            # &INTERACTION
            f.write("&interaction\n")
            f.write("  NORDER = 1\n")
            f.write("/\n\n")

            # &CUTOFF
            f.write("&cutoff\n")
            f.write("  *-*  18\n")
            f.write("/\n\n")

            # &OPTIMIZE
            f.write("&optimize\n")
            f.write("  DFSET = DFSET_harmonic\n")
            f.write("/\n\n")

            # &CELL
            f.write("&cell\n")
            f.write(f"  {scaling_factor:.16f} # factor \n")
            f.writelines(f"  {vec[0]:.10f}   {vec[1]:.10f}   {vec[2]:.10f}\n" for vec in lattice_vectors)
            f.write("# cell matrix\n/\n\n")

            # &POSITION
            f.write("&position\n")
            f.writelines("  {:d}   {:.16f}   {:.16f}   {:.16f}\n".format(*pos) for pos in positions)
            f.write("/\n")

        # subprocess.run(["mpirun", "-n", "1", "/home/jzheng4/alamode/_build/alm/alm alamode.in"], check=True)
        # subprocess.run(["mpirun -n 1 /home/jzheng4/alamode/_build/alm/alm alamode.in"], check=True)
        subprocess.run("mpirun -n 1 /home/jzheng4/alamode/_build/alm/alm alamode.in", shell=True, check=False)

        map_p2s, fc2_compact = _get_forceconstants_xml("EuZnAs_harmonic.xml")
        _write_fc2_phonopy(map_p2s, fc2_compact, filename="FORCE_CONSTANTS")

        primitive = phonon.primitive

        # symmetry = phonon.get_symmetry()

        # p2s = primitive.get_primitive_to_supercell_map()

        # write_force_constants_to_hdf5(f,p2s_map=p2s)

        # fc = read_force_constants_hdf5(filename="force_constants.hdf5")
        # write_FORCE_CONSTANTS(force_constants=fc)
        fc = parse_FORCE_CONSTANTS("FORCE_CONSTANTS")

        fc_full = compact_fc_to_full_fc(primitive, fc, log_level=0)

        # print(np.shape(fc_full))
        # print (np.shape())

        # write_force_constants_to_hdf5(fc_full,filename='force_constants_300K.hdf5') # change to the filename what you want
        write_FORCE_CONSTANTS(force_constants=fc_full, filename="FORCE_CONSTANTS_2ND")  # the same as the above comment

        if self.calc_anharmonic:
            logger.info("...Start to Calculate anharmonic force constants using LASSO in pheasy...")

            if self.num_anharmonic_snapshots is None:
                phonon.generate_displacements(distance=self.atom_disp_anhar)
                self.num_anharmonic_snapshots = len(phonon.displacements) * 10
                phonon.generate_displacements(
                    distance=self.atom_disp_anhar, number_of_snapshots=self.num_anharmonic_snapshots, random_seed=42
                )
            else:
                phonon.generate_displacements(
                    distance=self.atom_disp_anhar, number_of_snapshots=self.num_anharmonic_snapshots, random_seed=42
                )

            disp_supercells = phonon.supercells_with_displacements
            disp_array = []
            phonon.forces = [  # type: ignore[assignment]
                _calc_forces(self.calculator, supercell)
                for supercell in disp_supercells  # type:ignore[union-attr]
                if supercell is not None
            ]
            force_equilibrium = _calc_forces(self.calculator, phonon.supercell)
            phonon.forces = np.array(phonon.forces) - force_equilibrium
            for i, supercell in enumerate(disp_supercells):
                disp = supercell.get_positions() - phonon.supercell.get_positions()
                disp_array.append(np.array(disp))
            disp_array = np.array(disp_array)
            num_anh = disp_array.shape[0]

            output_file = "DFSET_anharmonic"
            logger.info("Saving disp_array and phonon.forces in files...")
            with open(output_file, "w") as f:
                for i, (disp, force) in enumerate(zip(disp_array, phonon.forces, strict=False)):
                    f.write(f"# supercell {i + 1}\n")
                    for d, fr in zip(disp, force, strict=False):
                        d_bohr = d * bohr_per_angstrom
                        fr_ryd_bohr = fr * ryd_per_ev_angstrom
                        line = " ".join(f"{val:.8f}" for val in np.concatenate((d_bohr, fr_ryd_bohr)))
                        f.write(line + "\n")

            with open("alamode_anhar.in", "w") as f:
                # &GENERAL
                f.write("&general\n")
                f.write("  PREFIX = EuZnAs_harmonic\n")
                f.write("  MODE = optimize\n")
                f.write(f"  NAT = {total_atoms}\n")
                f.write(f"  NKD = {len(elements)}\n")
                f.write("  KD = " + " ".join(elements) + "\n")
                f.write("  FC3_SHENGBTE=1\n")
                f.write("  FC4_SHENGBTE=1\n")
                f.write("/\n\n")

                # &INTERACTION
                f.write("&interaction\n")
                f.write("  NORDER = 3\n")
                f.write("  NBODY = 2 3 3\n")
                f.write("/\n\n")

                # &CUTOFF
                f.write("&cutoff\n")
                f.write("  *-* 18 10 7.5\n")
                f.write("/\n\n")

                # &OPTIMIZE
                f.write("&optimize\n")
                f.write(f"  NDATA ={num_anh}\n")
                f.write(" LMODEL = enet\n")
                f.write("  DFSET = DFSET_anharmonic\n")
                f.write("  FC2XML = EuZnAs_harmonic.xml\n")
                f.write("  ICONST = 11\n")
                f.write("  L1_ALPHA = 3.82925e-06\n")
                f.write("  STANDARDIZE = 1\n")
                f.write("  CONV_TOL = 1.0e-8\n")

                f.write("/\n\n")

                # &CELL
                f.write("&cell\n")
                f.write(f"  {scaling_factor:.16f} # factor \n")
                f.writelines(f"  {vec[0]:.10f}   {vec[1]:.10f}   {vec[2]:.10f}\n" for vec in lattice_vectors)
                f.write("# cell matrix\n/\n\n")

                # &POSITION
                f.write("&position\n")
                f.writelines("  {:d}   {:.16f}   {:.16f}   {:.16f}\n".format(*pos) for pos in positions)
                f.write("/\n")

        # subprocess.run(["mpirun", "-n", "1", "/home/jzheng4/alamode/_build/alm/alm alamode_anhar.in"], check=True)
        subprocess.run("mpirun -n 1 /home/jzheng4/alamode/_build/alm/alm alamode_anhar.in", shell=True, check=False)

        logger.info("...Finished running Alamode and higher-order FCs are ready...")

        return result | {"phonon": phonon}


def _calc_forces(calculator: Calculator, supercell: PhonopyAtoms) -> ArrayLike:
    """Helper to compute forces on a structure.

    Args:
        calculator: ASE Calculator
        supercell: Supercell from phonopy.

    Return:
        forces
    """
    struct = get_pmg_structure(supercell)
    atoms = AseAtomsAdaptor.get_atoms(struct)
    atoms.calc = calculator
    return atoms.get_forces()


def _parse_xml(fname_xml):
    try:
        # Attempt to parse the XML file directly with lxml
        tree = etree.parse(fname_xml)
        return tree
    except etree.XMLSyntaxError:
        # If a parsing error occurs with lxml,
        # attempt to repair by re-reading the file with the 'recover' parser
        repair_parser = etree.XMLParser(recover=True)
        tree = etree.parse(fname_xml, parser=repair_parser)
        return tree


def _get_forceconstants_xml(fname_xml):
    xml = _parse_xml(fname_xml)
    root = xml.getroot()

    natom_super = int(root.find("Structure/NumberOfAtoms").text)
    ntrans = int(root.find("Symmetry/NumberOfTranslations").text)
    natom_prim = natom_super // ntrans

    # parse mapping table
    map_p2s = np.zeros((ntrans, natom_prim), dtype=int)
    for elems in root.findall("Symmetry/Translations/map"):
        itran = int(elems.get("tran")) - 1
        iatom = int(elems.get("atom")) - 1
        map_p2s[itran, iatom] = int(elems.text) - 1

    # parse harmonic force constants
    fc2_compact = np.zeros((natom_prim, natom_super, 3, 3), dtype=float)

    for elems in root.findall("ForceConstants/HARMONIC/FC2"):
        atom1, xyz1 = [int(t) - 1 for t in elems.get("pair1").split()]
        atom2, xyz2, icell2 = [int(t) - 1 for t in elems.get("pair2").split()]
        fcsval = float(elems.text)

        fc2_compact[atom1, atom2, xyz1, xyz2] += fcsval

    fc2_compact *= Rydberg / (Bohr**2)

    return map_p2s, fc2_compact


def _write_fc2_phonopy(map_p2s, fc2_compact, filename="FORCE_CONSTANTS"):
    natom_prim, natom_super = fc2_compact.shape[:2]

    with open(filename, "w") as f:
        f.write(f"{natom_prim:5d} {natom_super:5d}\n")
        for i in range(natom_prim):
            for j in range(natom_super):
                f.write(f"{map_p2s[0, i] + 1:5d} {j + 1:5d}\n")
                for k in range(3):
                    f.writelines(f"{fc2_compact[i, j, k, l]:20.15f}" for l in range(3))
                    f.write("\n")
