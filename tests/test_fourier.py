# tests/test_fourier.py
import numpy as np
import pytest
from scattersim import fourier
from scattersim.qgrid import QGrid, QLine, zone_axis_grid, line_grid


# Simple cubic cell
CELL = np.diag([4.0, 4.0, 4.0])

# Single atom at origin
SINGLE_ATOM_POS = np.array([[0.0, 0.0, 0.0]])
SINGLE_ATOM_SPECIES = np.array(["NB"])


def _constant_ff(species, s):
    """Constant form factor for testing (f=1 for all species, all s)."""
    return np.ones_like(s)


class TestIntensitySingleAtom:
    """Single atom at origin: I(Q) = |f|^2 for all Q."""

    def test_2d_grid_uniform_intensity(self):
        Q = zone_axis_grid([0, 0, 1], CELL, extent=3.0, npts=11)
        I = fourier.intensity(
            (SINGLE_ATOM_POS, SINGLE_ATOM_SPECIES), Q, ff=_constant_ff, progress=False
        )
        assert I.shape == (11, 11)
        # With f=1 and one atom, I = 1 everywhere
        np.testing.assert_allclose(I, 1.0, atol=1e-10)

    def test_1d_line_uniform_intensity(self):
        Q = line_grid([0, 0, 0], [4, 0, 0], CELL, npts=101)
        I = fourier.intensity(
            (SINGLE_ATOM_POS, SINGLE_ATOM_SPECIES), Q, ff=_constant_ff, progress=False
        )
        assert I.shape == (101,)
        np.testing.assert_allclose(I, 1.0, atol=1e-10)


class TestIntensityTwoAtoms:
    """Two atoms: interference pattern with unit form factor."""

    def test_constructive_interference(self):
        """Two atoms separated by d=2A along x. At Q_x=0, both in phase: I=4."""
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        species = np.array(["O", "O"])
        Q = line_grid([0, 0, 0], [1, 0, 0], CELL, npts=1001)
        I = fourier.intensity(
            (positions, species), Q, ff=_constant_ff, progress=False
        )
        np.testing.assert_allclose(I[0], 4.0, atol=1e-10)

    def test_destructive_interference(self):
        """Two atoms separated by d=2A. Destructive at Q_x = pi/2 (Q*d = pi).
        I = |1 + exp(i*pi)|^2 = 0.
        Q_x = pi/2 corresponds to h = pi/2 / (2*pi/a) = a/4 = 1.0 in RLU for a=4.
        """
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        species = np.array(["O", "O"])
        # Single point at h=1 (Q_x = 2*pi/4 = pi/2, so Q*d = pi/2*2 = pi)
        Q = line_grid([1, 0, 0], [1.001, 0, 0], CELL, npts=2)
        I = fourier.intensity(
            (positions, species), Q, ff=_constant_ff, progress=False
        )
        np.testing.assert_allclose(I[0], 0.0, atol=1e-10)


class TestIntensityMultiFrame:
    """Multi-frame averaging."""

    def test_average_of_identical_frames(self):
        Q = zone_axis_grid([0, 0, 1], CELL, extent=2.0, npts=5)
        frame = (SINGLE_ATOM_POS, SINGLE_ATOM_SPECIES)
        I_single = fourier.intensity(frame, Q, ff=_constant_ff, progress=False)
        I_avg = fourier.intensity([frame, frame, frame], Q, ff=_constant_ff, progress=False)
        np.testing.assert_allclose(I_avg, I_single, atol=1e-10)


class TestIntensityStringFF:
    """Form factor string resolution."""

    def test_xray_string(self):
        Q = zone_axis_grid([0, 0, 1], CELL, extent=1.0, npts=3)
        # Should not raise
        I = fourier.intensity(
            (SINGLE_ATOM_POS, SINGLE_ATOM_SPECIES), Q, ff='xray', progress=False
        )
        assert I.shape == (3, 3)

    def test_unknown_string_raises(self):
        Q = zone_axis_grid([0, 0, 1], CELL, extent=1.0, npts=3)
        with pytest.raises(ValueError):
            fourier.intensity(
                (SINGLE_ATOM_POS, SINGLE_ATOM_SPECIES), Q, ff='unknown', progress=False
            )


class TestNumbaVsNumpy:
    """Verify Numba and NumPy paths produce identical results."""

    def test_2d_grid_matches(self):
        """Numba and NumPy paths should agree to machine precision."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.95, 0.0, 0.0],
            [0.0, 1.95, 0.0],
            [0.0, 0.0, 1.95],
        ])
        species = np.array(["NB", "O", "O", "F"])
        Q = zone_axis_grid([0, 0, 1], CELL, extent=3.0, npts=21)

        # Numba path (if available)
        I_default = fourier.intensity(
            (positions, species), Q, ff=_constant_ff, progress=False
        )

        # Force NumPy path
        saved = fourier._has_numba
        fourier._has_numba = False
        I_numpy = fourier.intensity(
            (positions, species), Q, ff=_constant_ff, progress=False
        )
        fourier._has_numba = saved

        np.testing.assert_allclose(I_default, I_numpy, rtol=1e-12)
