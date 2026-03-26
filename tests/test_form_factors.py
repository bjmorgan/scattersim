# tests/test_form_factors.py
import numpy as np
import pytest
from scattersim import form_factors


class TestWaasmaierKirfel:
    """Test X-ray form factor evaluation."""

    def test_at_s_zero_equals_Z_approximately(self):
        """f(s=0) = sum(a) + c, which should approximate Z."""
        s = np.array([0.0])
        f = form_factors.waasmaier_kirfel("O", s)
        # Oxygen Z=8, WK fit gives ~7.97
        assert abs(f[0] - 8.0) < 0.1

    def test_returns_array_matching_input_shape(self):
        s = np.linspace(0, 2, 100)
        f = form_factors.waasmaier_kirfel("NB", s)
        assert f.shape == (100,)

    def test_monotonically_decreasing(self):
        """X-ray form factors decrease with s."""
        s = np.linspace(0.01, 2.0, 100)
        f = form_factors.waasmaier_kirfel("NB", s)
        assert np.all(np.diff(f) < 0)

    def test_unknown_species_raises(self):
        with pytest.raises(KeyError):
            form_factors.waasmaier_kirfel("XX", np.array([0.0]))

    def test_scalar_s_works(self):
        """Should handle scalar input gracefully."""
        f = form_factors.waasmaier_kirfel("O", np.array([0.5]))
        assert isinstance(f, np.ndarray)
        assert f.shape == (1,)


class TestWaasmaierKirfelDiscus:
    """Test DISCUS-matching form factor with float32 degradation."""

    def test_differs_from_exact(self):
        """DISCUS mode should differ due to float32 bug."""
        s = np.array([0.5])
        f_exact = form_factors.waasmaier_kirfel("NB", s)
        f_discus = form_factors.waasmaier_kirfel_discus("NB", s)
        assert f_exact[0] != f_discus[0]
        # But only by ~1e-5 or less
        assert abs(f_exact[0] - f_discus[0]) < 1e-3

    def test_s_discretisation(self):
        """DISCUS mode discretises s to 0.001 increments."""
        s1 = np.array([0.5001])
        s2 = np.array([0.5004])
        f1 = form_factors.waasmaier_kirfel_discus("NB", s1)
        f2 = form_factors.waasmaier_kirfel_discus("NB", s2)
        # Both round to s=0.500, so should be identical
        np.testing.assert_array_equal(f1, f2)


class TestPeng:
    """Test electron form factor evaluation."""

    def test_returns_array(self):
        s = np.linspace(0, 2, 50)
        f = form_factors.peng("NB", s)
        assert f.shape == (50,)

    def test_at_s_zero_positive(self):
        f = form_factors.peng("O", np.array([0.0]))
        assert f[0] > 0

    def test_unknown_species_raises(self):
        with pytest.raises(KeyError):
            form_factors.peng("XX", np.array([0.0]))

    def test_ion_label_falls_back_to_neutral_with_warning(self):
        """Peng table is neutral-atom only; ion labels should fall back with a warning."""
        s = np.array([0.5])
        with pytest.warns(UserWarning, match="neutral atoms only"):
            f = form_factors.peng("NB5+", s)
        f_neutral = form_factors.peng("NB", s)
        np.testing.assert_array_equal(f, f_neutral)
