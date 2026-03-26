"""I/O adapters for scattersim.

Unpacks external structure formats (ASE Atoms) into scattersim's internal
representation (plain NumPy arrays). Also provides DISCUS .stru file I/O
for validation.
"""
import numpy as np


def from_ase(atoms, species_map: dict):
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
    species = np.array([species_map[s] for s in symbols])
    cell = np.array(atoms.cell)
    return positions, species, cell


def write_stru(filename: str, positions: np.ndarray, species: np.ndarray,
               cell: np.ndarray, title: str = ""):
    """Write a DISCUS .stru file.

    Parameters
    ----------
    filename : str
        Output file path.
    positions : np.ndarray, shape (N, 3)
        Cartesian coordinates in Angstroms.
    species : np.ndarray, shape (N,)
        Species labels (4-char DISCUS names, e.g. 'NB', 'O', 'F').
    cell : np.ndarray, shape (3, 3)
        Cell vectors in Angstroms. Assumes orthorhombic (uses diagonal).
    title : str
        Title line for the file.
    """
    # Check for orthorhombic cell (off-diagonal elements must be zero)
    off_diag = cell - np.diag(np.diag(cell))
    if np.any(np.abs(off_diag) > 1e-6):
        raise ValueError(
            "write_stru only supports orthorhombic cells. "
            f"Off-diagonal elements are non-zero: {off_diag}"
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
        for i in range(len(species)):
            name = f"{species[i]:<4s}"
            cx = f"{frac[i, 0]:14.6f},"
            cy = f"{frac[i, 1]:14.6f},"
            cz = f"{frac[i, 2]:14.6f},"
            biso = f"{0.5:14.6f},"
            f.write(f"{name}{cx:>16s}{cy:>16s}{cz:>16s}{biso:>16s}"
                    f"       1,       0,       0,   1.000000\n")


def read_stru(filename: str):
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
        for line in f:
            line = line.strip()
            if line.startswith("cell"):
                parts = line.replace("cell", "").strip().split(",")
                a = float(parts[0])
                b = float(parts[1])
                c = float(parts[2])
                cell = np.diag([a, b, c])
            elif line == "atoms":
                in_atoms = True
            elif in_atoms and line:
                tokens = line.split()
                name = tokens[0]
                # Rejoin and split by comma
                rest = line[len(name):].strip()
                vals = [v.strip() for v in rest.split(",")]
                x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
                species_list.append(name)
                frac_list.append([x, y, z])

    if cell is None:
        raise ValueError(f"No cell found in {filename}")

    frac = np.array(frac_list)
    positions = frac @ cell  # fractional to Cartesian
    species = np.array(species_list)

    return positions, species, cell
