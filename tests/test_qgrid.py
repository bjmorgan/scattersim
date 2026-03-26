# tests/test_qgrid.py
import numpy as np
import pytest
from scattersim.qgrid import QGrid, QLine, zone_axis_grid, line_grid


# Simple cubic cell for testing (a = 4.0 A)
CUBIC_CELL = np.diag([4.0, 4.0, 4.0])


class TestQGrid:
    def test_shape(self):
        qg = zone_axis_grid([0, 0, 1], CUBIC_CELL, extent=3.0, npts=11)
        assert qg.Q.shape == (11, 11, 3)

    def test_centred_on_origin(self):
        qg = zone_axis_grid([0, 0, 1], CUBIC_CELL, extent=3.0, npts=11)
        centre = qg.Q[5, 5, :]
        np.testing.assert_allclose(centre, [0, 0, 0], atol=1e-10)

    def test_extent_matches(self):
        qg = zone_axis_grid([0, 0, 1], CUBIC_CELL, extent=3.0, npts=11)
        assert qg.extent == (-3.0, 3.0, -3.0, 3.0)

    def test_q_perpendicular_to_zone_axis(self):
        """All Q-vectors should have zero component along the zone axis."""
        qg = zone_axis_grid([0, 0, 1], CUBIC_CELL, extent=3.0, npts=11)
        # For [001] zone axis, all Q should have Q_z = 0
        np.testing.assert_allclose(qg.Q[:, :, 2], 0, atol=1e-10)

    def test_max_q_magnitude_matches_extent(self):
        qg = zone_axis_grid([0, 0, 1], CUBIC_CELL, extent=3.0, npts=101)
        # Corner Q-vectors have magnitude sqrt(2) * extent
        Q_mags = np.linalg.norm(qg.Q, axis=-1)
        # Edge midpoint should be exactly extent
        assert abs(Q_mags[50, 0] - 3.0) < 0.1  # left edge, middle row

    def test_has_labels(self):
        qg = zone_axis_grid([0, 0, 1], CUBIC_CELL, extent=3.0, npts=11)
        assert isinstance(qg.v1_label, str)
        assert isinstance(qg.v2_label, str)


class TestZoneAxisNonTrivial:
    """Test zone-axis grid with non-trivial geometry."""

    def test_110_zone_axis_perpendicular(self):
        """[1,1,0] zone axis: all Q should be perpendicular to [1,1,0] direction."""
        qg = zone_axis_grid([1, 1, 0], CUBIC_CELL, extent=3.0, npts=11)
        # [1,1,0] in Cartesian is [1,1,0] (cubic cell)
        uvw_cart = np.array([1.0, 1.0, 0.0])
        uvw_cart = uvw_cart / np.linalg.norm(uvw_cart)
        # Dot product of all Q vectors with zone axis should be zero
        dots = qg.Q @ uvw_cart
        np.testing.assert_allclose(dots, 0, atol=1e-10)

    def test_110_zone_basis_orthogonal(self):
        """In-plane basis vectors should be orthogonal after Gram-Schmidt."""
        qg = zone_axis_grid([1, 1, 0], CUBIC_CELL, extent=3.0, npts=11)
        # Extract two edge vectors from the grid
        e1 = qg.Q[0, 1, :] - qg.Q[0, 0, :]  # horizontal step
        e2 = qg.Q[1, 0, :] - qg.Q[0, 0, :]  # vertical step
        dot = np.dot(e1, e2)
        np.testing.assert_allclose(dot, 0, atol=1e-10)

    def test_deterministic_orientation(self):
        """Same inputs should always produce the same grid."""
        qg1 = zone_axis_grid([1, 1, 0], CUBIC_CELL, extent=3.0, npts=5)
        qg2 = zone_axis_grid([1, 1, 0], CUBIC_CELL, extent=3.0, npts=5)
        np.testing.assert_array_equal(qg1.Q, qg2.Q)


class TestQLine:
    def test_shape(self):
        ql = line_grid([0, 0, 0], [4, 0, 0], CUBIC_CELL, npts=201)
        assert ql.Q.shape == (201, 3)
        assert ql.q_scalar.shape == (201,)

    def test_starts_at_origin(self):
        ql = line_grid([0, 0, 0], [4, 0, 0], CUBIC_CELL, npts=201)
        np.testing.assert_allclose(ql.Q[0], [0, 0, 0], atol=1e-10)

    def test_endpoint_rlu_conversion(self):
        """End point [4,0,0] in RLU should be 4 * 2*pi/a in Cartesian."""
        ql = line_grid([0, 0, 0], [4, 0, 0], CUBIC_CELL, npts=201)
        a = 4.0
        expected_qx = 4 * 2 * np.pi / a
        np.testing.assert_allclose(ql.Q[-1, 0], expected_qx, rtol=1e-10)

    def test_q_scalar_monotonic(self):
        ql = line_grid([0, 0, 0], [4, 0, 0], CUBIC_CELL, npts=201)
        assert np.all(np.diff(ql.q_scalar) > 0)

    def test_has_label(self):
        ql = line_grid([0, 0, 0], [4, 0, 0], CUBIC_CELL, npts=201)
        assert isinstance(ql.label, str)
