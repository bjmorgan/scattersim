# scattersim/qgrid.py
"""Reciprocal-space Q-vector grid construction.

Builds arrays of Q-vectors in Cartesian inverse Angstroms for the Fourier
engine. Supports 2D zone-axis grids and 1D reciprocal-space lines.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class QGrid:
    """2D grid of Q-vectors for a zone-axis diffraction pattern.

    Attributes
    ----------
    Q : np.ndarray
        Shape (npts, npts, 3). Cartesian Q-vectors in inverse Angstroms.
        First axis is v2 (vertical), second axis is v1 (horizontal).
    extent : tuple
        (v1_min, v1_max, v2_min, v2_max) in inverse Angstroms, for matplotlib imshow.
    v1_label : str
        Label for the horizontal in-plane basis vector.
    v2_label : str
        Label for the vertical in-plane basis vector.
    """
    Q: np.ndarray
    extent: tuple
    v1_label: str
    v2_label: str


@dataclass
class QLine:
    """1D grid of Q-vectors along a reciprocal-space line.

    Attributes
    ----------
    Q : np.ndarray
        Shape (npts, 3). Cartesian Q-vectors in inverse Angstroms.
    q_scalar : np.ndarray
        Shape (npts,). Scalar coordinate along the line (distance from start).
    label : str
        Description of the line, e.g. "(0,0,0) to (4,0,0)".
    """
    Q: np.ndarray
    q_scalar: np.ndarray
    label: str


def _reciprocal_lattice(cell: np.ndarray) -> np.ndarray:
    """Compute reciprocal lattice vectors (with 2*pi factor).

    Parameters
    ----------
    cell : np.ndarray
        Shape (3, 3). Real-space cell vectors as rows.

    Returns
    -------
    np.ndarray
        Shape (3, 3). Reciprocal lattice vectors as rows.
        b_i . a_j = 2*pi * delta_ij.
    """
    return 2 * np.pi * np.linalg.inv(cell).T


def _find_in_plane_basis(uvw: np.ndarray, recip: np.ndarray) -> tuple:
    """Find two short reciprocal lattice vectors satisfying the zone condition.

    Searches for hkl vectors satisfying u*h + v*k + w*l = 0, then applies
    2D Lagrange (Gauss) reduction to find the two shortest independent vectors.

    Parameters
    ----------
    uvw : np.ndarray
        Zone axis direction in direct-space Miller indices.
    recip : np.ndarray
        Shape (3, 3). Reciprocal lattice vectors as rows.

    Returns
    -------
    v1, v2 : np.ndarray
        Cartesian in-plane vectors in inverse Angstroms.
    v1_hkl, v2_hkl : np.ndarray
        Miller indices of v1 and v2.
    """
    # Search for hkl satisfying the zone condition: u*h + v*k + w*l = 0
    u, v, w = uvw
    max_idx = 5
    candidates = []
    for h in range(-max_idx, max_idx + 1):
        for k in range(-max_idx, max_idx + 1):
            for l in range(-max_idx, max_idx + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                if u * h + v * k + w * l == 0:
                    q_cart = np.array([h, k, l]) @ recip
                    q_mag = np.linalg.norm(q_cart)
                    candidates.append((q_mag, h, k, l, q_cart))

    # Sort by magnitude, then lexicographic for deterministic tie-breaking
    candidates.sort(key=lambda x: (x[0], abs(x[1]), abs(x[2]), abs(x[3])))

    if not candidates:
        raise ValueError(
            f"No reciprocal lattice vectors satisfy zone condition for uvw={uvw} "
            f"within |hkl| <= {max_idx}"
        )

    # Take shortest as v1
    _, h1, k1, l1, v1_cart = candidates[0]
    v1_hkl = np.array([h1, k1, l1])

    # Find shortest independent vector as v2
    for _, h2, k2, l2, v2_cart in candidates[1:]:
        # Check independence: cross product of hkl indices should be non-zero
        cross = np.cross(v1_hkl, np.array([h2, k2, l2]))
        if np.any(cross != 0):
            v2_hkl = np.array([h2, k2, l2])
            break
    else:
        raise ValueError(f"Could not find two independent in-plane vectors for uvw={uvw}")

    # 2D Lagrange reduction to get shortest pair
    v1_cart, v2_cart = _lagrange_reduce_2d(
        v1_hkl.astype(float) @ recip,
        v2_hkl.astype(float) @ recip
    )

    # Recover Miller indices from reduced Cartesian vectors
    inv_recip = np.linalg.inv(recip)
    v1_hkl_reduced = np.rint(v1_cart @ inv_recip).astype(int)
    v2_hkl_reduced = np.rint(v2_cart @ inv_recip).astype(int)

    return v1_cart, v2_cart, v1_hkl_reduced, v2_hkl_reduced


def _lagrange_reduce_2d(v1: np.ndarray, v2: np.ndarray) -> tuple:
    """Lagrange (Gauss) reduction of a 2D lattice basis.

    Returns the two shortest basis vectors spanning the same lattice.
    """
    for _ in range(100):  # safety limit
        if np.dot(v1, v1) > np.dot(v2, v2):
            v1, v2 = v2, v1
        # Reduce v2 by v1
        mu = round(np.dot(v2, v1) / np.dot(v1, v1))
        v2 = v2 - mu * v1
        if np.dot(v2, v2) >= np.dot(v1, v1):
            break
    return _orient_positive(v1), _orient_positive(v2)


def _orient_positive(v: np.ndarray) -> np.ndarray:
    """Flip vector so its first non-negligible component is positive."""
    idx = np.flatnonzero(np.abs(v) > 1e-10)
    if len(idx) > 0 and v[idx[0]] < 0:
        return -v
    return v


def _fmt_point(p: np.ndarray) -> str:
    """Format a reciprocal-space point as a label string."""
    if np.allclose(p, np.rint(p)):
        return ",".join(str(int(x)) for x in np.rint(p))
    return ",".join(f"{x:.2g}" for x in p)


def _format_hkl(hkl: np.ndarray) -> str:
    """Format Miller indices as a label string."""
    h, k, l = hkl.astype(int)
    return f"[{h},{k},{l}]"


def zone_axis_grid(uvw, cell: np.ndarray, extent: float, npts: int) -> QGrid:
    """Build a 2D Q-grid for a zone-axis diffraction pattern.

    Parameters
    ----------
    uvw : array_like
        Zone axis direction as integer Miller indices, e.g. [0, 0, 1].
    cell : np.ndarray
        Shape (3, 3). Real-space cell vectors as rows, in Angstroms.
    extent : float
        Range in inverse Angstroms from origin in each in-plane direction.
        Grid spans [-extent, +extent] along both axes.
    npts : int
        Number of grid points per axis.

    Returns
    -------
    QGrid
        2D grid of Q-vectors with metadata for plotting.
    """
    uvw = np.asarray(uvw, dtype=int)
    cell = np.asarray(cell, dtype=float)
    recip = _reciprocal_lattice(cell)

    v1_cart, v2_cart, v1_hkl, v2_hkl = _find_in_plane_basis(uvw, recip)

    # Normalise to unit vectors in Cartesian space
    e1 = v1_cart / np.linalg.norm(v1_cart)
    e2 = v2_cart / np.linalg.norm(v2_cart)

    # Orthogonalise e2 w.r.t. e1 (Gram-Schmidt) to ensure perpendicular grid axes
    e2 = e2 - np.dot(e2, e1) * e1
    e2 = e2 / np.linalg.norm(e2)

    # Build uniform Cartesian grid
    c = np.linspace(-extent, extent, npts)
    C1, C2 = np.meshgrid(c, c, indexing='xy')

    Q = (C1[:, :, np.newaxis] * e1[np.newaxis, np.newaxis, :] +
         C2[:, :, np.newaxis] * e2[np.newaxis, np.newaxis, :])

    return QGrid(
        Q=Q,
        extent=(-extent, extent, -extent, extent),
        v1_label=_format_hkl(v1_hkl),
        v2_label=_format_hkl(v2_hkl),
    )


def line_grid(start, end, cell: np.ndarray, npts: int) -> QLine:
    """Build a 1D Q-grid along a line in reciprocal space.

    Parameters
    ----------
    start, end : array_like
        Endpoints in reciprocal lattice units (RLU).
    cell : np.ndarray
        Shape (3, 3). Real-space cell vectors as rows, in Angstroms.
    npts : int
        Number of points along the line.

    Returns
    -------
    QLine
        1D grid of Q-vectors with scalar coordinate and label.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    cell = np.asarray(cell, dtype=float)
    recip = _reciprocal_lattice(cell)

    Q_start = start @ recip
    Q_end = end @ recip

    t = np.linspace(0, 1, npts)
    Q = Q_start[np.newaxis, :] + t[:, np.newaxis] * (Q_end - Q_start)[np.newaxis, :]

    q_scalar = np.linalg.norm(Q - Q_start, axis=1)

    label = f"({_fmt_point(start)}) to ({_fmt_point(end)})"

    return QLine(Q=Q, q_scalar=q_scalar, label=label)
