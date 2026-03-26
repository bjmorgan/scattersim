"""I/O adapters for scattersim.

Unpacks external structure formats (ASE Atoms) into scattersim's internal
representation (plain NumPy arrays). Also provides DISCUS .stru file I/O
for validation.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def from_ase(atoms: Any, species_map: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack an ASE Atoms object into scattersim's internal representation.

    Parameters
    ----------
    atoms : ase.Atoms
        Full supercell structure.
    species_map : dict
        Mapping from element symbols to form factor species labels.
        Required -- no default. Example: {'Nb': 'Nb5+', 'O': 'O2-', 'F': 'F1-'}.

    Returns
    -------
    positions : np.ndarray, shape (N, 3)
        Cartesian coordinates in Angstroms.
    species : np.ndarray, shape (N,)
        Form factor species labels.
    cell : np.ndarray, shape (3, 3)
        Cell vectors in Angstroms.

    Raises
    ------
    KeyError
        If an element symbol in atoms is not in species_map.
    """
    positions = atoms.positions.copy()
    symbols = atoms.get_chemical_symbols()
    missing = set(symbols) - set(species_map.keys())
    if missing:
        raise KeyError(
            f"Elements {missing} not in species_map. "
            f"Provide mappings for all elements. "
            f"species_map keys: {set(species_map.keys())}"
        )
    species = np.array([species_map[s] for s in symbols])
    cell = np.array(atoms.cell)
    return positions, species, cell


def write_stru(filename: str, positions: np.ndarray, species: np.ndarray,
               cell: np.ndarray, title: str = "") -> None:
    """Write a DISCUS .stru file.

    Parameters
    ----------
    filename : str
        Output file path.
    positions : np.ndarray, shape (N, 3)
        Cartesian coordinates in Angstroms.
    species : np.ndarray, shape (N,)
        Species labels (e.g. 'NB', 'O', 'F'). Padded to 4-char field width on output.
    cell : np.ndarray, shape (3, 3)
        Cell vectors in Angstroms. Assumes orthorhombic (uses diagonal).
    title : str
        Title line for the file.
    """
    # Cell must be a diagonal matrix (Cartesian-aligned orthorhombic).
    # Rotated orthorhombic cells are not supported — coordinates would
    # also need rotation, which this function does not do.
    off_diag = cell - np.diag(np.diag(cell))
    if np.any(np.abs(off_diag) > 1e-6):
        raise ValueError(
            "write_stru requires a diagonal cell matrix (Cartesian-aligned "
            "orthorhombic). Got off-diagonal elements: "
            f"{off_diag[np.abs(off_diag) > 1e-6]}"
        )
    a, b, c = np.linalg.norm(cell, axis=1)
    # Convert to fractional coordinates
    inv_cell = np.linalg.inv(cell)
    frac = positions @ inv_cell.T

    with open(filename, "w") as f:
        f.write(f"title {title}\n")
        f.write("spcgr P 1\n")
        f.write(f"cell  {a:10.6f},{b:10.6f},{c:10.6f}, 90.000000, 90.000000, 90.000000\n")
        f.write("ncell  1, 1, 1, {}\n".format(len(species)))
        f.write("atoms\n")
        for sp, xyz in zip(species, frac):
            name = f"{sp:<4s}"
            cx = f"{xyz[0]:14.6f},"
            cy = f"{xyz[1]:14.6f},"
            cz = f"{xyz[2]:14.6f},"
            biso = f"{0.5:14.6f},"
            f.write(f"{name}{cx:>16s}{cy:>16s}{cz:>16s}{biso:>16s}"
                    f"       1,       0,       0,   1.000000\n")


def read_stru(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a DISCUS .stru file.

    Parameters
    ----------
    filename : str
        Input file path.

    Returns
    -------
    positions : np.ndarray, shape (N, 3)
        Cartesian coordinates in Angstroms.
    species : np.ndarray, shape (N,)
        Species labels.
    cell : np.ndarray, shape (3, 3)
        Cell vectors in Angstroms (diagonal, orthorhombic assumed).
    """
    species_list = []
    frac_list = []
    cell = None

    with open(filename) as f:
        in_atoms = False
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if line.startswith("cell"):
                try:
                    parts = line.replace("cell", "").strip().split(",")
                    a = float(parts[0])
                    b = float(parts[1])
                    c = float(parts[2])
                except (IndexError, ValueError) as e:
                    raise ValueError(
                        f"{filename}:{line_no}: cannot parse cell line: {line!r}"
                    ) from e
                cell = np.diag([a, b, c])
            elif line == "atoms":
                in_atoms = True
            elif in_atoms and line:
                try:
                    tokens = line.split()
                    name = tokens[0]
                    rest = line[len(name):].strip()
                    vals = [v.strip() for v in rest.split(",")]
                    x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
                except (IndexError, ValueError) as e:
                    raise ValueError(
                        f"{filename}:{line_no}: cannot parse atom line: {line!r}"
                    ) from e
                species_list.append(name)
                frac_list.append([x, y, z])

    if cell is None:
        raise ValueError(f"No cell line found in {filename}")

    if not species_list:
        raise ValueError(f"No atoms found in {filename}")

    frac = np.array(frac_list)
    positions = frac @ cell  # fractional to Cartesian
    species = np.array(species_list)

    return positions, species, cell
