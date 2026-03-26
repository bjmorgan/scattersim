"""Atomic form factor evaluation.

Supports X-ray (Waasmaier-Kirfel) and electron (Peng) form factors
via 5-Gaussian parameterisation: f(s) = sum_i a_i exp(-b_i s^2) + c
where s = |Q| / (4 pi).
"""
import json
import re
import warnings
from functools import lru_cache
from importlib import resources

import numpy as np


def _load_table(filename: str) -> dict[str, dict]:
    """Load a form factor JSON table from the data directory."""
    ref = resources.files("scattersim").joinpath("data", filename)
    with resources.as_file(ref) as path:
        data: dict = json.loads(path.read_text())
    result: dict[str, dict] = data["elements"]
    return result


@lru_cache(maxsize=1)
def _wk_table() -> dict[str, dict]:
    return _load_table("waasmaier_kirfel.json")


@lru_cache(maxsize=1)
def _peng_table() -> dict[str, dict]:
    return _load_table("peng.json")


def _eval_five_gaussian(a: list[float], b: list[float], c: float, s: np.ndarray) -> np.ndarray:
    """Evaluate the 5-Gaussian form factor: f(s) = sum_i a_i exp(-b_i s^2) + c."""
    s2 = s ** 2
    result = np.full_like(s, c, dtype=np.float64)
    for ai, bi in zip(a, b):
        result += ai * np.exp(-bi * s2)
    return result


def _strip_charge(species: str) -> str:
    """Strip ionic charge suffix from a species label.

    'NB5+' -> 'NB', 'O2-' -> 'O', 'F1-' -> 'F', 'FE' -> 'FE'.
    """
    return re.sub(r'\d*[+-]$', '', species)


def waasmaier_kirfel(species: str, s: np.ndarray) -> np.ndarray:
    """Evaluate X-ray form factor (Waasmaier-Kirfel).

    Parameters
    ----------
    species : str
        Element or ion label, e.g. 'NB', 'O', 'Nb5+', 'O2-'.
    s : np.ndarray
        Scattering vector magnitudes s = |Q| / (4 pi) in inverse Angstroms.

    Returns
    -------
    np.ndarray
        Form factor values, same shape as s.
    """
    table = _wk_table()
    key = species.upper()
    if key not in table:
        raise KeyError(
            f"Species '{species}' not found in Waasmaier-Kirfel table. "
            f"Available: {sorted(table.keys())[:10]}..."
        )
    entry = table[key]
    return _eval_five_gaussian(entry["a"], entry["b"], entry["c"], s)


def waasmaier_kirfel_discus(species: str, s: np.ndarray) -> np.ndarray:
    """Evaluate X-ray form factor with DISCUS-matching degradation.

    Replicates two DISCUS numerical artefacts:
    1. Float32 truncation in Fortran literal constants (~1e-7 per coefficient)
    2. Discretisation of s to 0.001 increments (lookup table)

    For validation against DISCUS output only.

    Parameters
    ----------
    species : str
        Element or ion label, e.g. 'NB', 'O2-'.
    s : np.ndarray
        Scattering vector magnitudes s = |Q| / (4 pi) in inverse Angstroms.

    Returns
    -------
    np.ndarray
        Form factor values, same shape as s.
    """
    table = _wk_table()
    key = species.upper()
    if key not in table:
        raise KeyError(f"Species '{species}' not found in Waasmaier-Kirfel table.")
    entry = table[key]

    # Apply float32 degradation
    a = [float(np.float64(np.float32(v))) for v in entry["a"]]
    b = [float(np.float64(np.float32(v))) for v in entry["b"]]
    c = float(np.float64(np.float32(entry.get("c", 0.0))))

    # Discretise s to 0.001
    s_disc = np.round(s / 0.001) * 0.001

    return _eval_five_gaussian(a, b, c, s_disc)


def peng(species: str, s: np.ndarray) -> np.ndarray:
    """Evaluate electron form factor (Peng et al.).

    Parameters
    ----------
    species : str
        Element label, e.g. 'NB', 'O', 'F'. Ion labels (e.g. 'NB5+')
        are accepted but fall back to the neutral atom with a warning,
        since the Peng parameterisation covers neutral atoms only.
    s : np.ndarray
        Scattering vector magnitudes s = |Q| / (4 pi) in inverse Angstroms.

    Returns
    -------
    np.ndarray
        Form factor values, same shape as s.
    """
    table = _peng_table()
    key = species.upper()
    if key not in table:
        neutral = _strip_charge(key)
        if neutral in table:
            warnings.warn(
                f"Peng table has neutral atoms only; using '{neutral}' for '{species}'. "
                f"Electron form factors are parameterised for neutral atoms.",
                stacklevel=2,
            )
            key = neutral
        else:
            raise KeyError(
                f"Species '{species}' not found in Peng table. "
                f"Available: {sorted(table.keys())[:10]}..."
            )
    entry = table[key]
    return _eval_five_gaussian(entry["a"], entry["b"], entry.get("c", 0.0), s)
