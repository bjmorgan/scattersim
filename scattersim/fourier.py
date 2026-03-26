# scattersim/fourier.py
"""Direct Fourier sum engine for kinematic scattering intensity.

Computes I(Q) = |F(Q)|^2 where F(Q) = sum_j f_j(|Q|) exp(i Q . r_j).
"""
import numpy as np

from scattersim import form_factors as ff_module
from scattersim.qgrid import QGrid, QLine


_has_tqdm = True
try:
    from tqdm import trange
except ImportError:
    _has_tqdm = False


def _progress_range(n: int, **kwargs):
    """Row iterator with optional tqdm progress bar."""
    if _has_tqdm:
        return trange(n, **kwargs)
    return range(n)


def _resolve_ff(ff):
    """Resolve form factor argument to a callable.

    Parameters
    ----------
    ff : str or callable
        'xray', 'xray-discus', 'electron', or callable(species, s) -> array.

    Returns
    -------
    callable
        Function with signature (species: str, s: np.ndarray) -> np.ndarray.
    """
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


def _compute_single_frame(positions, species, Q_array, ff_func, progress, desc):
    """Compute I(Q) for a single frame.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
    species : np.ndarray, shape (N,)
    Q_array : np.ndarray, shape (..., 3) — either (n2, n1, 3) or (npts, 3)
    ff_func : callable
    progress : bool
    desc : str

    Returns
    -------
    np.ndarray — I(Q), shape matches Q_array leading dimensions.
    """
    is_2d = Q_array.ndim == 3
    if is_2d:
        n2, n1, _ = Q_array.shape
        I = np.zeros((n2, n1))
        unique_species = sorted(set(species))
        masks = {sp: (species == sp) for sp in unique_species}

        for i2 in _progress_range(n2, disable=not progress, desc=desc):
            Q_row = Q_array[i2, :, :]  # (n1, 3)
            s_row = np.linalg.norm(Q_row, axis=1) / (4 * np.pi)  # (n1,)
            phases = positions @ Q_row.T  # (N, n1)
            exp_phases = np.exp(1j * phases)

            # Build weighted sum: F = sum_j f_j * exp(i Q . r_j)
            F = np.zeros(n1, dtype=complex)
            for sp in unique_species:
                f_vals = ff_func(sp, s_row)  # (n1,)
                F += np.sum(f_vals[np.newaxis, :] * exp_phases[masks[sp], :], axis=0)

            I[i2, :] = np.abs(F) ** 2
        return I
    else:
        # 1D line: Q_array shape (npts, 3)
        npts = Q_array.shape[0]
        s = np.linalg.norm(Q_array, axis=1) / (4 * np.pi)
        phases = positions @ Q_array.T  # (N, npts)
        exp_phases = np.exp(1j * phases)

        unique_species = sorted(set(species))
        masks = {sp: (species == sp) for sp in unique_species}

        F = np.zeros(npts, dtype=complex)
        for sp in unique_species:
            f_vals = ff_func(sp, s)  # (npts,)
            F += np.sum(f_vals[np.newaxis, :] * exp_phases[masks[sp], :], axis=0)

        return np.abs(F) ** 2


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

    Returns
    -------
    np.ndarray
        Scattering intensity. Shape (npts, npts) for QGrid, (npts,) for QLine.
    """
    ff_func = _resolve_ff(ff)
    Q_array = Q.Q

    # Determine if single frame or multi-frame
    if isinstance(source, list):
        frames = source
    else:
        frames = [source]

    n_frames = len(frames)
    I_sum = None

    for i, (positions, species) in enumerate(frames):
        positions = np.asarray(positions, dtype=float)
        species = np.asarray(species)
        desc = f"Frame {i+1}/{n_frames}" if n_frames > 1 else "Computing I(Q)"
        show_progress = progress and (Q_array.ndim == 3)  # only for 2D grids

        I_frame = _compute_single_frame(
            positions, species, Q_array, ff_func, show_progress, desc
        )

        if I_sum is None:
            I_sum = I_frame
        else:
            I_sum += I_frame

    return I_sum / n_frames
