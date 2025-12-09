"""NEB calculations."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from ase.io import Trajectory
from ase.mep import NEBTools

try:
    from ase.mep import NEB
except ImportError:
    from ase.neb import NEB
from ase.utils.forcecurve import fit_images
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Species

from ._base import PropCalc
from .backend._ase import get_ase_optimizer
from .utils import to_ase_atoms, to_pmg_structure

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.optimize.optimize import Optimizer


@dataclass
class MEP:
    """Minimum Energy Path dataclass for NEB calculations.

    Stores MEP results in a data-efficient format where labels are stored
    once, and only fractional coordinates and lattices vary between images.
    """

    labels: list[Species] = field(default_factory=list)
    """Species labels for all atoms (same for all images)."""
    lattices: np.ndarray | list[np.ndarray] = field(default_factory=lambda: np.eye(3))
    """Lattice matrix(ces). If a single array, applies to all images. If a list, one per image."""
    frac_coords: list[np.ndarray] = field(default_factory=list)
    """Fractional coordinates for each image."""
    energies: list[float] = field(default_factory=list)
    """Energy for each image."""

    def __init__(
        self,
        structures: list[Structure],
        energies: list[float],
    ) -> None:
        """Initialize MEP from structures and energies.

        Parameters:
            structures: List of pymatgen Structures.
            energies: List of energies for each image.
        """
        if len(structures) != len(energies):
            n_structs = len(structures)
            n_energies = len(energies)
            raise ValueError(f"Number of structures ({n_structs}) must match number of energies ({n_energies})")

        first_struct = structures[0]
        self.labels = list(first_struct.species)  # type: ignore[arg-type]

        lattices_list = [struct.lattice.matrix for struct in structures]
        frac_coords_list = [struct.frac_coords for struct in structures]

        # Check if all lattices are the same (within numerical precision)
        first_lattice = lattices_list[0]
        all_same = all(np.allclose(first_lattice, lat) for lat in lattices_list[1:])
        self.lattices = first_lattice if all_same else lattices_list
        self.frac_coords = frac_coords_list
        self.energies = list(energies)

    def get_lattices_list(self) -> list[np.ndarray]:
        """Get lattices as a list, expanding a single lattice if needed."""
        if isinstance(self.lattices, np.ndarray):
            return [self.lattices] * len(self.frac_coords) if self.frac_coords else []
        return self.lattices

    def as_dict(self) -> dict[str, Any]:
        """Convert MEP to a data-efficient dictionary representation.

        Returns:
            Dictionary with:
                - "labels": List of species strings (stored once)
                - "lattice": Lattice matrix as a 3x3 array (only if same for all images)
                - "images": List of dictionaries, each containing:
                    - "lattice": Lattice matrix as a 3x3 array (only if different per image)
                    - "frac_coords": Fractional coordinates for the image
                    - "energy": Energy for the image
        """
        lattices_list = self.get_lattices_list()

        # Check if all lattices are the same
        if lattices_list and all(np.allclose(lattices_list[0], lat) for lat in lattices_list[1:]):
            # Store single lattice at top level
            return {
                "labels": [str(spec) for spec in self.labels],
                "lattice": lattices_list[0].tolist(),
                "images": [
                    {
                        "frac_coords": frac_coords.tolist(),
                        "energy": energy,
                    }
                    for frac_coords, energy in zip(self.frac_coords, self.energies, strict=False)
                ],
            }
        # Store lattice per image
        return {
            "labels": [str(spec) for spec in self.labels],
            "images": [
                {
                    "lattice": lattice.tolist(),
                    "frac_coords": frac_coords.tolist(),
                    "energy": energy,
                }
                for lattice, frac_coords, energy in zip(lattices_list, self.frac_coords, self.energies, strict=False)
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MEP:
        """Reconstruct MEP from a dictionary representation.

        Parameters:
            d: Dictionary with keys "labels" and "images" as returned by as_dict().
                May also contain "lattice" at top level if same for all images.

        Returns:
            MEP instance reconstructed from the dictionary.
        """
        labels = [Species(label) for label in d["labels"]]
        frac_coords = [np.array(img["frac_coords"]) for img in d["images"]]
        energies = [img["energy"] for img in d["images"]]

        # Check if lattice is stored at top level (same for all) or per image
        if "lattice" in d:
            lattices_list = [np.array(d["lattice"])] * len(frac_coords)
        else:
            lattices_list = [np.array(img["lattice"]) for img in d["images"]]

        # Reconstruct structures from labels, lattices, and fractional coordinates
        structures = []
        for lattice, frac_coord in zip(lattices_list, frac_coords, strict=False):
            lattice_obj = Lattice(lattice)
            structure = Structure(lattice_obj, labels, frac_coord)
            structures.append(structure)

        return cls(structures, energies)

    def get_structures(self) -> list[Structure]:
        """Get all images as a list of pymatgen Structures.

        Returns:
            List of Structure objects, one for each image in the MEP.
        """
        structures = []
        lattices_list = self.get_lattices_list()
        for lattice, frac_coords in zip(lattices_list, self.frac_coords, strict=False):
            lattice_obj = Lattice(lattice)
            structure = Structure(lattice_obj, self.labels, frac_coords)
            structures.append(structure)
        return structures


class NEBCalc(PropCalc):
    """NEB calculator."""

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        optimizer: str | Optimizer = "BFGS",
        traj_folder: str | None = None,
        interval: int = 1,
        climb: bool = True,
        fmax: float = 0.1,
        max_steps: int = 1000,
        **kwargs: Any,
    ) -> None:
        """Initialize the instance of the class.

        Parameters:
            calculator (Calculator | str): An ASE calculator object used to perform energy and force
                calculations. If string is provided, the corresponding universal calculator is loaded.
            optimizer (str | Optimizer, optional): The optimization algorithm to use. Defaults to "BFGS".
            traj_folder (str | None, optional): The folder to save trajectory information. Defaults to None.
            interval (int, optional): The interval for recording trajectory information. Defaults to 1.
            climb (bool, optional): Whether to perform climbing image nudged elastic band (CI-NEB). Defaults to True.
            fmax (float, optional): The maximum force tolerance for convergence. Defaults to 0.1.
            max_steps (int, optional): The maximum number of optimization steps. Defaults to 1000.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            None
        """
        self.calculator = calculator  # type: ignore[assignment]

        self.traj_folder = traj_folder
        self.interval = interval
        self.climb = climb
        self.optimizer = get_ase_optimizer(optimizer)
        self.fmax = fmax
        self.max_steps = max_steps
        self.kwargs = kwargs

    def calc_images(
        self,
        start_struct: Structure,
        end_struct: Structure,
        *,
        n_images: int = 7,
        interpolate_lattices: bool = False,
        autosort_tol: float = 0.5,
    ) -> dict[str, Any]:
        """Calculate NEB images between given start and end structures.

        Parameters:
            start_struct (Structure): Initial structure.
            end_struct (Structure): Final structure.
            n_images (int): Number of images to calculate (default is 7).
            interpolate_lattices (bool): Whether to interpolate lattices between start and end structures (default is
                False).
            autosort_tol (float): Tolerance for autosorting the images (default is 0.5).

        Returns:
            NEBCalc: NEB calculation object containing the interpolated images.
        """
        images = start_struct.interpolate(
            end_struct,
            nimages=n_images + 1,
            interpolate_lattices=interpolate_lattices,
            pbc=False,
            autosort_tol=autosort_tol,
        )
        return self.calc({f"image{i:02d}": s for i, s in enumerate(images)})

    def calc(
        self,
        structure: Structure | Atoms | dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate the energy barrier using the nudged elastic band method.

        Parameters:
            - structure: A dictionary containing the images with keys 'image0', 'image1', etc. Must be of type dict.

        Returns:
                dict:
                    - "barrier" (float): The energy barrier of the reaction pathway.
                    - "force" (float): The force exerted on the atoms during the NEB calculation.
                    - "mep" (MEP): An MEP dataclass containing the images and their respective energies.
        """
        if not isinstance(structure, dict):
            raise ValueError(  # noqa:TRY004
                "For NEB calculations, structure must be a dict containing the images with keys image00, image01, etc."
            )
        images: list[Atoms] = []
        for _, image in sorted(structure.items(), key=lambda x: x[0]):
            atoms = to_ase_atoms(image)
            atoms.calc = self.calculator
            images.append(atoms)

        self.neb = NEB(images, climb=self.climb, allow_shared_calculator=True, **self.kwargs)
        optimizer = self.optimizer(self.neb)  # type:ignore[operator]
        if self.traj_folder is not None:
            os.makedirs(self.traj_folder, exist_ok=True)
            for idx, img in enumerate(images):
                optimizer.attach(
                    Trajectory(f"{self.traj_folder}/image-{idx}.traj", "w", img),
                    interval=self.interval,
                )

        optimizer.run(fmax=self.fmax, steps=self.max_steps)
        neb_tool = NEBTools(self.neb.images)
        data = neb_tool.get_barrier()  # add structures
        result = {"barrier": data[0], "force": data[1]}

        energies = fit_images(self.neb.images).energies
        # Convert images to pymatgen structures
        structures = [to_pmg_structure(image) for image in self.neb.images]

        # Create MEP instance from structures and energies
        mep = MEP(structures, list(energies))
        result["mep"] = mep
        return result
