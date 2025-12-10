"""Some utility methods, e.g., for getting calculators from well-known sources."""

from __future__ import annotations

import warnings
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from ase import Atoms
from ase.calculators.calculator import Calculator
from pymatgen.core import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from .units import eVA3ToGPa

if TYPE_CHECKING:
    from pathlib import Path

    from maml.apps.pes import LMPStaticCalculator
    from pyace.basis import ACEBBasisSet, ACECTildeBasisSet, BBasisConfiguration
    from pymatgen.core import IMolecule, IStructure


# Listing of supported universal calculators.
# If you update UNIVERSAL_CALCULATORS, you must also update the mapping in
# map_calculators_to_packages in test_utils.py, unless already covered.

_universal_calculators = [
    "M3GNet",
    "CHGNet",
    "MACE",
    "SevenNet",
    "TensorNet",
    "GRACE",
    "ORB",
    "PBE",
    "r2SCAN",
    "MatterSim",
    "FAIRChem",
    "PETMAD",
    "DeePMD",
]

try:
    # Auto-load all available PES models from matgl if installed.
    import matgl

    _universal_calculators += [
        m for m in matgl.get_available_pretrained_models() if "PES" in m and "ANI-1x-Subset-PES" not in m
    ]
    _universal_calculators = sorted(set(_universal_calculators))
except Exception:  # noqa: BLE001
    warnings.warn("Unable to get pre-trained MatGL universal calculators.", stacklevel=1)

# Different backend libraries (MatGL, MACE, GRACE, etc.) use
# inconsistent or non-canonical model names (e.g., "small-omat-0",
# "TensorNet-MatPES-PBE-v2025.1-PES", "GRACE-1L-OMAT-medium-base").

# Users and developers are encouraged to use the model naming convention below.
# MatCalc Model Naming Convention (Unified model ID format):
# [Model]-(#Layers)-[Dataset]-(Functional)-(Cutoff)-(Version)-(Size)-(Postprocess)
# Not all fields must appear.

# Examples:
# TensorNet-MatPES-PBE-v2025.1-PES
# MACE-MP-0-medium
# GRACE-1L-MP-r6
# GRACE-1L-OMAT-medium-base
# GRACE-2L-OMAT-large-ft-AM

# This table maps such identifiers into the backend model names that
# each library expects.

# This tabel will be gradually expanded as new models are released.

# Keys must be lowercase and represent canonical identifiers
# Values are the actual model names passed to the backend libraries.

MODEL_ALIASES = {
    # Using abbreviations will load the most advanced model.
    "tensornet": "TensorNet-MatPES-PBE-v2025.1-PES",
    "m3gnet": "M3GNet-MatPES-PBE-v2025.1-PES",
    "chgnet": "CHGNet-MatPES-PBE-2025.2.10-2.7M-PES",
    "mace": "medium-mpa-0",
    "grace": "GRACE-2L-OAM",
    "pbe": "TensorNet-MatPES-PBE-v2025.1-PES",
    "r2scan": "TensorNet-MatPES-r2SCAN-v2025.1-PES",
    "mace-mp-0-small": "small",
    "mace-mp-0-medium": "medium",
    "mace-mp-0-large": "large",
    "mace-mp-0b-small": "small-0b",
    "mace-mp-0b-medium": "medium-0b",
    "mace-mp-0b2-small": "small-0b2",
    "mace-mp-0b2-medium": "medium-0b2",
    "mace-mp-0b2-large": "large-0b2",
    "mace-mp-0b3-medium": "medium-0b3",
    "mace-mpa-0-medium": "medium-mpa-0",
    "mace-omat-0-small": "small-omat-0",
    "mace-omat-0-medium": "medium-omat-0",
}


UNIVERSAL_CALCULATORS = Enum("UNIVERSAL_CALCULATORS", {k: k for k in _universal_calculators})  # type: ignore[misc]


class PESCalculator(Calculator):
    """
    Class for simulating and calculating potential energy surfaces (PES) using various
    machine learning and classical potentials. It extends the ASE `Calculator` API,
    allowing integration with the ASE framework for molecular dynamics and structure
    optimization.

    PESCalculator provides methods to perform energy, force, and stress calculations
    using potentials such as MTP, GAP, NNP, SNAP, ACE, NequIP, DeePMD and MatGL (M3GNet, TensorNet, CHGNet). The class
    includes utilities to load compatible models for each potential type, making it
    a versatile tool for materials modeling and molecular simulations.

    :ivar potential: The potential model used for PES calculations.
    :type potential: LMPStaticCalculator
    :ivar stress_weight: The stress weight factor to convert between units.
    :type stress_weight: float
    """

    implemented_properties = ["energy", "forces", "stress"]  # noqa:RUF012

    def __init__(
        self,
        potential: LMPStaticCalculator,
        stress_unit: Literal["eV/A3", "GPa"] = "GPa",
        stress_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize PESCalculator with a potential from maml.

        Args:
            potential (LMPStaticCalculator): maml.apps.pes._lammps.LMPStaticCalculator
            stress_unit (str): The unit of stress. Default to "GPa"
            stress_weight (float): The conversion factor from GPa to eV/A^3, if it is set to 1.0, the unit is in GPa.
                Default to 1.0.
            **kwargs: Additional keyword arguments passed to super().__init__().
        """
        super().__init__(**kwargs)
        self.potential = potential

        # Handle stress unit conversion
        if stress_unit == "eV/A3":
            conversion_factor = 1 / eVA3ToGPa  # Conversion factor from GPa to eV/A^3
        elif stress_unit == "GPa":
            conversion_factor = 1.0  # No conversion needed if stress is already in GPa
        else:
            raise ValueError(f"Unsupported stress_unit: {stress_unit}. Must be 'GPa' or 'eV/A3'.")

        self.stress_weight = stress_weight * conversion_factor

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list | None = None,
        system_changes: list | None = None,
    ) -> None:
        """
        Perform calculation for an input Atoms.

        Args:
            atoms (ase.Atoms): ase Atoms object
            properties (list): The list of properties to calculate
            system_changes (list): monitor which properties of atoms were
                changed for new calculation. If not, the previous calculation
                results will be loaded.
        """
        from ase.calculators.calculator import all_changes, all_properties
        from maml.apps.pes import EnergyForceStress
        from pymatgen.io.ase import AseAtomsAdaptor

        properties = properties or all_properties
        system_changes = system_changes or all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)

        structure: Structure | IStructure = AseAtomsAdaptor.get_structure(atoms)  # type: ignore[arg-type,assignment]
        efs_calculator = EnergyForceStress(ff_settings=self.potential)
        energy, forces, stresses = efs_calculator.calculate([structure])[0]

        self.results = {
            "energy": energy,
            "forces": forces,
            "stress": stresses * self.stress_weight,
        }

    @staticmethod
    def load_matgl(path: str | Path, **kwargs: Any) -> Calculator:
        """
        Loads a MATGL model from the specified path and initializes a PESCalculator
        with the loaded model and additional optional parameters.

        This method uses the MATGL library to load a model from the given file path
        or directory. It then configures a calculator using the loaded model and
        the provided keyword arguments.

        :param path: The path to the MATGL model file or directory.
        :type path: str | Path
        :param kwargs: Additional keyword arguments used to configure the calculator.
        :return: An instance of the PESCalculator initialized with the loaded MATGL
            model and configured with the given parameters.
        :rtype: Calculator
        """
        import matgl
        from matgl.ext.ase import PESCalculator as PESCalculator_

        model = matgl.load_model(path=path)  # type:ignore[arg-type]
        kwargs.setdefault("stress_unit", "eV/A3")
        return PESCalculator_(potential=model, **kwargs)

    @staticmethod
    def load_mtp(filename: str | Path, elements: list, **kwargs: Any) -> Calculator:
        """
        Load a machine-learned potential (MTPotential) from a configuration file and
        create a calculator object to interface with it.

        This method initializes an instance of MTPotential using a provided
        configuration file and elements. It returns a PESCalculator instance,
        which wraps the initialized potential model.

        :param filename: Path to the configuration file for the MTPotential.
        :type filename: str | Path
        :param elements: List of element symbols used in the model. Each element
            should be a string representing a chemical element (e.g., "H", "O").
        :type elements: list
        :param kwargs: Additional keyword arguments to configure the PESCalculator.
        :type kwargs: Any
        :return: A calculator object wrapping the MTPotential.
        :rtype: Calculator
        """
        from maml.apps.pes import MTPotential

        model = MTPotential.from_config(filename=filename, elements=elements)
        return PESCalculator(potential=model, **kwargs)

    @staticmethod
    def load_gap(filename: str | Path, **kwargs: Any) -> Calculator:
        """
        Loads a Gaussian Approximation Potential (GAP) model from the given file and
        returns a corresponding Calculator instance. GAP is a machine learning-based
        potential used for atomistic simulations and requires a specific config file as
        input. Any additional arguments for the calculator can be passed via kwargs,
        allowing customization.

        :param filename: Path to the configuration file for the GAP model.
        :type filename: str | Path
        :param kwargs: Additional keyword arguments for configuring the calculator.
        :type kwargs: Any
        :return: An instance of PESCalculator initialized with the GAPotential model.
        :rtype: Calculator
        """
        from maml.apps.pes import GAPotential

        model = GAPotential.from_config(filename=str(filename))
        return PESCalculator(potential=model, **kwargs)

    @staticmethod
    def load_nnp(
        input_filename: str | Path, scaling_filename: str | Path, weights_filenames: list, **kwargs: Any
    ) -> Calculator:
        """
        Loads a neural network potential (NNP) from specified configuration files and
        creates a Calculator object configured with the potential. This function allows
        for customizable keyword arguments to modify the behavior of the resulting
        Calculator.

        :param input_filename: Path to the primary input file containing NNP configuration.
        :type input_filename: str | Path
        :param scaling_filename: Path to the scaling parameters file required for the NNP.
        :type scaling_filename: str | Path
        :param weights_filenames: List of paths to weight files for the NNP.
        :type weights_filenames: list
        :param kwargs: Additional keyword arguments passed to the Calculator constructor.
        :type kwargs: Any
        :return: A Calculator object initialized with the loaded NNP settings.
        :rtype: Calculator
        """
        from maml.apps.pes import NNPotential

        model = NNPotential.from_config(
            input_filename=input_filename,
            scaling_filename=scaling_filename,
            weights_filenames=weights_filenames,
        )
        return PESCalculator(potential=model, **kwargs)

    @staticmethod
    def load_snap(param_file: str | Path, coeff_file: str | Path, **kwargs: Any) -> Calculator:
        """
        Load a SNAP (Spectral Neighbor Analysis Potential) configuration and create a
        corresponding Calculator instance.

        This static method initializes a SNAPotential instance using the provided
        configuration files and subsequently generates a PESCalculator based on the
        created potential model and additional keyword arguments.

        :param param_file: Path to the parameter file required for SNAPotential configuration.
        :param coeff_file: Path to the coefficient file required for SNAPotential configuration.
        :param kwargs: Additional keyword arguments passed to the PESCalculator.
        :return: A PESCalculator instance configured with the SNAPotential model.
        :rtype: Calculator
        """
        from maml.apps.pes import SNAPotential

        model = SNAPotential.from_config(param_file=param_file, coeff_file=coeff_file)
        return PESCalculator(potential=model, **kwargs)

    @staticmethod
    def load_ace(  # pragma: no cover
        basis_set: str | Path | ACEBBasisSet | ACECTildeBasisSet | BBasisConfiguration, **kwargs: Any
    ) -> Calculator:
        """
        Load an ACE (Atomic Cluster Expansion) calculator using the specified basis set.

        This method utilizes the PyACE library to create and initialize a PyACECalculator
        instance with a given basis set. The provided basis set can take various forms including
        file paths, basis set objects, or configurations. Additional customization options
        can be passed through keyword arguments.

        :param basis_set: The basis set used for initializing the ACE calculator. This can
            be provided as a string, Path object, ACEBBasisSet, ACECTildeBasisSet, or
            BBasisConfiguration.
        :param kwargs: Additional configuration parameters to customize the ACE
            calculator. These keyword arguments are passed directly to the PyACECalculator
            instance during initialization.
        :return: An instance of the Calculator class representing the initialized ACE
            calculator.
        """
        from pyace import PyACECalculator

        return PyACECalculator(basis_set=basis_set, **kwargs)

    @staticmethod
    def load_nequip(  # pragma: no cover
        model_path: str | Path, **kwargs: Any
    ) -> Calculator:
        """
        Loads and returns a NequIP `Calculator` instance from the specified model path.
        This method facilitates the integration of machine learning models into ASE
        by loading a model for atomic-scale simulations.

        :param model_path: The file path to the serialized NequIP model.
        :type model_path: str | Path
        :param kwargs: Additional keyword arguments to be passed to the
            `NequIPCalculator.from_deployed_model` method.
        :type kwargs: Any
        :return: A `Calculator` instance initialized with the given model and parameters,
            suitable for ASE simulations.
        :rtype: Calculator
        """
        from nequip.ase import NequIPCalculator

        return NequIPCalculator.from_deployed_model(model_path=model_path, **kwargs)

    @staticmethod
    def load_deepmd(  # pragma: no cover
        model_path: str | Path, **kwargs: Any
    ) -> Calculator:
        """
        Loads a Deep Potential Molecular Dynamics (DeePMD) model and returns a `Calculator`
        object for molecular dynamics simulations.

        This method imports the `deepmd.calculator.DP` class and initializes it with the
        given model path and optional keyword arguments. The resulting `Calculator` object
        is used to perform molecular simulations based on the specified DeePMD model.

        The function requires the DeePMD-kit library to be installed to properly import
        and utilize the `DP` class.

        :param model_path: Path to the trained DeePMD model file, provided as a string
                           or a Path object.
        :param kwargs: Additional options and configurations to pass into the DeePMD
                       `Calculator` during initialization.
        :return: An instance of the Calculator object initialized with the specified
                 DeePMD model and optional configurations.
        :rtype: Calculator
        """
        from deepmd.calculator import DP

        return DP(model=model_path, **kwargs)

    @staticmethod
    def load_universal(name: str | Calculator, **kwargs: Any) -> Calculator:  # noqa: C901
        """
        Loads a calculator instance based on the provided name or an existing calculator object. The
        method supports multiple pre-built universal models and aliases for ease of use. If an existing calculator
        object is passed instead of a name, it will directly return that calculator instance. Supported FPs
        include SOTA potentials such as M3GNet, CHGNet, TensorNet, MACE, GRACE, SevenNet, ORB, etc.

        This method is designed to provide a universal interface to load various calculator types, which
        may belong to different domains and packages. It auto-resolves aliases, provides default options
        for certain calculators, and raises errors for unsupported inputs.

        :param name: The name of the calculator to load or an instance of a Calculator.
        :param kwargs: Keyword arguments that are passed to the internal calculator initialization routines
                    for models matching the specified name. These options are calculator dependent.
        :return: An instance of the loaded calculator.

        :raises ValueError: If the name provided does not match any recognized calculator type.
        """
        result: Calculator

        if not isinstance(name, str):  # e.g. already an ase Calculator instance
            result = name

        elif any(name.lower().startswith(m) for m in ("m3gnet", "chgnet", "tensornet", "pbe", "r2scan")):
            name = MODEL_ALIASES.get(name.lower(), name)
            result = PESCalculator.load_matgl(name, **kwargs)

        elif name.lower().startswith("mace"):
            name = MODEL_ALIASES.get(name.lower(), name)
            from mace.calculators import mace_mp

            result = mace_mp(model=name, **kwargs)

        elif name.lower() == "sevennet":
            from sevenn.calculator import SevenNetCalculator

            result = SevenNetCalculator(**kwargs)

        elif name.lower().startswith("grace"):
            name = MODEL_ALIASES.get(name.lower(), name)
            from tensorpotential.calculator.foundation_models import grace_fm

            result = grace_fm(model=name, **kwargs)

        elif name.lower() == "orb":
            from orb_models.forcefield.calculator import ORBCalculator
            from orb_models.forcefield.pretrained import ORB_PRETRAINED_MODELS

            model = kwargs.pop("model", "orb-v2")
            device = kwargs.get("device", "cpu")

            orbff = ORB_PRETRAINED_MODELS[model](device=device)
            result = ORBCalculator(orbff, **kwargs)

        elif name.lower() == "mattersim":  # pragma: no cover
            from mattersim.forcefield import MatterSimCalculator

            result = MatterSimCalculator(**kwargs)

        elif name.lower() == "fairchem":  # pragma: no cover
            from fairchem.core import FAIRChemCalculator, pretrained_mlip

            device = kwargs.pop("device", "cpu")
            model = kwargs.pop("model", "uma-s-1")
            task_name = kwargs.pop("task_name", "omat")
            predictor = pretrained_mlip.get_predict_unit(model, device=device)
            result = FAIRChemCalculator(predictor, task_name=task_name, **kwargs)

        elif name.lower() == "petmad":  # pragma: no cover
            from pet_mad.calculator import PETMADCalculator

            result = PETMADCalculator(**kwargs)

        elif name.lower().startswith("deepmd"):  # pragma: no cover
            from pathlib import Path

            from deepmd.calculator import DP

            cwd = Path(__file__).parent.absolute()
            model_path = cwd / "../../tests/pes/DPA3-LAM-2025.3.14-PES" / "2025-03-14-dpa3-openlam.pth"
            model_path = model_path.resolve()
            kwargs.setdefault("model", model_path)
            result = DP(**kwargs)

        else:
            raise ValueError(f"Unrecognized {name=}, must be one of {UNIVERSAL_CALCULATORS}")

        return result


def to_ase_atoms(structure: Atoms | Structure | Molecule) -> Atoms:
    """
    Converts a given structure into an ASE Atoms object. This function checks
    if the input structure is already an ASE Atoms object. If not, it converts
    a pymatgen Structure object to an ASE Atoms object using the AseAtomsAdaptor.

    :param structure: The input structure, which can be either an ASE Atoms object
        or a pymatgen Structure object.
    :type structure: Atoms | Structure
    :return: An ASE Atoms object representing the given structure.
    :rtype: Atoms
    """
    return structure if isinstance(structure, Atoms) else AseAtomsAdaptor.get_atoms(structure)


def to_pmg_structure(structure: Atoms | Structure) -> Structure:
    """
    Converts a given structure of type Atoms or Structure into a Structure
    object. If the input structure is already of type Structure, it is
    returned unchanged. If the input structure is of type Atoms, it is
    converted to a Structure using the AseAtomsAdaptor.

    :param structure: The input structure to be converted. This can be of
        type Atoms or Structure.
    :type structure: Atoms | Structure
    :return: A Structure object corresponding to the input structure. If the
        input is already a Structure, it is returned as-is. Otherwise, it is
        converted.
    :rtype: Structure
    """
    return structure if isinstance(structure, Structure) else AseAtomsAdaptor.get_structure(structure)  # type: ignore[return-value]


def to_pmg_molecule(structure: Atoms | Structure | Molecule | IMolecule) -> IMolecule:
    """
    Converts a given structure of type Atoms or Structure into a Molecule
    object. If the input structure is already of type Molecule, it is
    returned unchanged. If the input structure is of type Atoms, it is
    converted to a Molecule using the AseAtomsAdaptor.

    :param structure: The input structure to be converted. This can be of
        type Atoms or Structure or Molecule.
    :type structure: Atoms | Structure | Molecule
    :return: A Molecule object corresponding to the input structure. If the
        input is already a Molecule, it is returned as-is. Otherwise, it is
        converted.
    :rtype: Molecule
    """
    if isinstance(structure, Atoms):
        structure = AseAtomsAdaptor.get_molecule(structure)

    return Molecule.from_sites(structure)  # type: ignore[return-value]