# scattersim/fourier.py
"""Direct Fourier sum engine for kinematic scattering intensity.

Computes I(Q) = |F(Q)|^2 where F(Q) = sum_j f_j(|Q|) exp(i Q . r_j).

Uses Numba JIT compilation with parallel row iteration when available,
falling back to pure NumPy otherwise.
"""
import math

import numpy as np

from scattersim import form_factors as ff_module
from scattersim.qgrid import QGrid, QLine


_has_tqdm = True
try:
    from tqdm import trange
except ImportError:
    _has_tqdm = False

_has_numba = True
try:
    import numba
except ImportError:
    _has_numba = False


def _progress_range(n: int, **kwargs):
    """Row iterator with optional tqdm progress bar."""
    if _has_tqdm:
        return trange(n, **kwargs)
    return range(n)


def _resolve_ff(ff):
    """Resolve form factor argument to a callable."""
    if callable(ff):
        return ff
    dispatch = {
        'xray': ff_module.waasmaier_kirfel,
        'xray-discus': ff_module.waasmaier_kirfel_discus,
        'electron': ff_module.peng,
    }
    if ff not in dispatch:
        raise ValueError(
            f"Unknown form factor model '{ff}'. "
            f"Choose from {list(dispatch.keys())} or pass a callable."
        )
    return dispatch[ff]


# ============================================================
# Numba kernel
# ============================================================

if _has_numba:
    @numba.njit(parallel=True)
    def _numba_kernel_2d(positions, Q_array, f_species, atom_sp_idx, I_out):  # type: ignore[misc]
        """Numba-accelerated Fourier sum for 2D grids.

        Parameters
        ----------
        positions : (N, 3) float64
        Q_array : (n2, n1, 3) float64
        f_species : (n_species, n2, n1) float64 — pre-evaluated form factors
        atom_sp_idx : (N,) int32 — species index per atom
        I_out : (n2, n1) float64 — output (modified in place)
        """
        n2, n1, _ = Q_array.shape
        N = positions.shape[0]
        for i2 in numba.prange(n2):
            for i1 in range(n1):
                F_re = 0.0
                F_im = 0.0
                qx = Q_array[i2, i1, 0]
                qy = Q_array[i2, i1, 1]
                qz = Q_array[i2, i1, 2]
                for j in range(N):
                    phase = (positions[j, 0] * qx
                             + positions[j, 1] * qy
                             + positions[j, 2] * qz)
                    f = f_species[atom_sp_idx[j], i2, i1]
                    F_re += f * math.cos(phase)
                    F_im += f * math.sin(phase)
                I_out[i2, i1] = F_re * F_re + F_im * F_im


# ============================================================
# Form factor pre-evaluation (cached across frames)
# ============================================================

def _build_ff_cache(unique_species, Q_array, ff_func):
    """Pre-evaluate form factors for all species at all Q-points.

    This only depends on the Q-grid and species list, not on atom positions,
    so it can be computed once and reused across MC frames.

    Parameters
    ----------
    unique_species : list of str
        Sorted unique species labels.
    Q_array : np.ndarray
        Shape (n2, n1, 3) for 2D or (npts, 3) for 1D.
    ff_func : callable

    Returns
    -------
    f_species : np.ndarray
        For 2D: shape (n_species, n2, n1). For 1D: shape (n_species, npts).
    sp_to_idx : dict
        Maps species label to index in f_species.
    """
    sp_to_idx = {sp: i for i, sp in enumerate(unique_species)}
    n_sp = len(unique_species)

    if Q_array.ndim == 3:
        n2, n1, _ = Q_array.shape
        # Compute s for entire grid at once: (n2, n1)
        s_grid = np.linalg.norm(Q_array, axis=2) / (4 * np.pi)
        f_species = np.empty((n_sp, n2, n1), dtype=np.float64)
        for i_sp, sp in enumerate(unique_species):
            for i2 in range(n2):
                f_species[i_sp, i2, :] = ff_func(sp, s_grid[i2, :])
    else:
        npts = Q_array.shape[0]
        s = np.linalg.norm(Q_array, axis=1) / (4 * np.pi)
        f_species = np.empty((n_sp, npts), dtype=np.float64)
        for i_sp, sp in enumerate(unique_species):
            f_species[i_sp, :] = ff_func(sp, s)

    return f_species, sp_to_idx


# ============================================================
# Single-frame computation
# ============================================================

def _compute_2d_numba(positions, atom_sp_idx, Q_array, f_species):
    """Compute I(Q) for a 2D grid using Numba."""
    n2, n1, _ = Q_array.shape
    I_out = np.empty((n2, n1), dtype=np.float64)
    _numba_kernel_2d(positions, Q_array, f_species, atom_sp_idx, I_out)
    return I_out


def _compute_2d_numpy(positions, atom_sp_idx, Q_array, f_species, progress, desc):
    """Compute I(Q) for a 2D grid using pure NumPy."""
    n2, n1, _ = Q_array.shape
    N = positions.shape[0]
    I_out = np.zeros((n2, n1))

    for i2 in _progress_range(n2, disable=not progress, desc=desc):
        Q_row = Q_array[i2, :, :]  # (n1, 3)
        phases = positions @ Q_row.T  # (N, n1)
        exp_phases = np.exp(1j * phases)

        # Build f_row[j, i1] from pre-evaluated species form factors
        f_row = f_species[atom_sp_idx, i2, :]  # (N, n1)

        F = np.sum(f_row * exp_phases, axis=0)  # (n1,)
        I_out[i2, :] = np.abs(F) ** 2

    return I_out


def _compute_1d(positions, atom_sp_idx, Q_array, f_species):
    """Compute I(Q) for a 1D line."""
    phases = positions @ Q_array.T  # (N, npts)
    exp_phases = np.exp(1j * phases)

    # f_line[j, i] from pre-evaluated species form factors
    f_line = f_species[atom_sp_idx, :]  # (N, npts)

    F = np.sum(f_line * exp_phases, axis=0)  # (npts,)
    return np.abs(F) ** 2


# ============================================================
# Public API
# ============================================================

def intensity(source, Q, ff, progress=True):
    """Compute scattering intensity I(Q) = |F(Q)|^2.

    Parameters
    ----------
    source : tuple or list of tuples
        Single frame: (positions, species) where positions is (N, 3) array
        in Angstroms and species is (N,) array of string labels.
        Multiple frames: list of (positions, species) tuples. Returns
        incoherent average (<|F|^2>, not |<F>|^2).
    Q : QGrid or QLine
        Q-vector grid from qgrid module.
    ff : str or callable
        Form factor model: 'xray', 'xray-discus', 'electron', or
        callable(species, s) -> array.
    progress : bool, default True
        Show progress bar (requires tqdm; silently disabled if not installed).
        Only used for the NumPy fallback path; Numba path runs without progress.

    Returns
    -------
    np.ndarray
        Scattering intensity. Shape (n2, n1) for QGrid, (npts,) for QLine.
    """
    ff_func = _resolve_ff(ff)
    Q_array = Q.Q
    is_2d = Q_array.ndim == 3

    # Determine frames
    if isinstance(source, list):
        frames = source
    else:
        frames = [source]

    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError("source must contain at least one frame.")

    # Gather unique species from all frames
    all_species: set[str] = set()
    for _, sp in frames:
        all_species.update(np.asarray(sp))
    unique_species = sorted(all_species)

    # Pre-evaluate form factors once for the Q-grid (cached across frames)
    f_species, sp_to_idx = _build_ff_cache(unique_species, Q_array, ff_func)

    I_sum = None

    for i, (positions, species) in enumerate(frames):
        positions = np.asarray(positions, dtype=np.float64)
        species = np.asarray(species)

        # Map atoms to species indices
        atom_sp_idx = np.array([sp_to_idx[s] for s in species], dtype=np.int32)

        if is_2d:
            if _has_numba:
                I_frame = _compute_2d_numba(positions, atom_sp_idx, Q_array, f_species)
            else:
                desc = (f"Frame {i+1}/{n_frames}" if n_frames > 1
                        else "Computing I(Q)")
                I_frame = _compute_2d_numpy(
                    positions, atom_sp_idx, Q_array, f_species, progress, desc
                )
        else:
            I_frame = _compute_1d(positions, atom_sp_idx, Q_array, f_species)

        if I_sum is None:
            I_sum = I_frame
        else:
            I_sum += I_frame

    return I_sum / n_frames
