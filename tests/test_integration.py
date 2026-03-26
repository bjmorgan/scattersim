# tests/test_integration.py
"""Integration tests: full pipeline from positions to intensity."""
import numpy as np
from scattersim import fourier
from scattersim.qgrid import zone_axis_grid, line_grid


def _constant_ff(species, s):
    """Unit form factor for all species."""
    return np.ones_like(s)


class TestSimpleCubicBraggPeaks:
    """Simple cubic lattice should produce Bragg peaks at reciprocal lattice points."""

    def setup_method(self):
        """Build a 4x4x4 simple cubic lattice, a=4A, one atom per cell."""
        a = 4.0
        N = 4
        self.cell = np.diag([a * N, a * N, a * N])
        positions = []
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    positions.append([i * a, j * a, k * a])
        self.positions = np.array(positions)
        self.species = np.array(["X"] * len(positions))

    def test_bragg_peak_at_origin(self):
        """I(Q=0) = N^2 for N atoms with unit form factor."""
        Q = line_grid([0, 0, 0], [0.001, 0, 0], self.cell, npts=2)
        I = fourier.intensity(
            (self.positions, self.species), Q, ff=_constant_ff, progress=False
        )
        N = len(self.positions)
        np.testing.assert_allclose(I[0], N ** 2, rtol=1e-6)

    def test_bragg_peak_at_100(self):
        """Intensity at (1,0,0) reciprocal lattice point of the unit cell
        should also be N^2 (all atoms in phase)."""
        # Q at h=4 of supercell = h=1 of unit cell
        Q = line_grid([3.99, 0, 0], [4.01, 0, 0], self.cell, npts=3)
        I = fourier.intensity(
            (self.positions, self.species), Q, ff=_constant_ff, progress=False
        )
        N = len(self.positions)
        # Middle point is closest to the Bragg peak
        np.testing.assert_allclose(I[1], N ** 2, rtol=0.01)

    def test_intensity_between_peaks_is_small(self):
        """Between Bragg peaks, I should be much smaller than N^2."""
        # h=2.0 in supercell RLU = h=0.5 in unit-cell RLU, truly between Bragg peaks
        Q = line_grid([2.0, 0, 0], [2.1, 0, 0], self.cell, npts=10)
        I = fourier.intensity(
            (self.positions, self.species), Q, ff=_constant_ff, progress=False
        )
        N = len(self.positions)
        # Should be orders of magnitude smaller than N^2
        assert I.max() < N ** 2 * 0.1
