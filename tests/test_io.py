import numpy as np
import pytest
import tempfile
import os
from scattersim import io


class TestFromAse:
    def test_basic_unpacking(self):
        """Test unpacking an ASE Atoms object."""
        ase = pytest.importorskip("ase")
        from ase import Atoms

        atoms = Atoms("NbOF", positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0]],
                       cell=[4, 4, 4], pbc=True)
        species_map = {"Nb": "Nb5+", "O": "O2-", "F": "F1-"}
        positions, species, cell = io.from_ase(atoms, species_map)

        assert positions.shape == (3, 3)
        assert list(species) == ["Nb5+", "O2-", "F1-"]
        assert cell.shape == (3, 3)
        np.testing.assert_allclose(positions[0], [0, 0, 0])

    def test_species_map_required(self):
        """Should raise if species_map is missing a symbol."""
        ase = pytest.importorskip("ase")
        from ase import Atoms

        atoms = Atoms("NbO", positions=[[0, 0, 0], [2, 0, 0]],
                       cell=[4, 4, 4], pbc=True)
        with pytest.raises(KeyError):
            io.from_ase(atoms, species_map={"Nb": "Nb5+"})  # missing O


class TestStruIO:
    def test_roundtrip(self):
        """Write then read a .stru file and recover the same data."""
        positions = np.array([[0.0, 0.0, 0.0], [1.95, 0.0, 0.0]])
        species = np.array(["NB", "O"])
        cell = np.diag([3.9, 3.9, 3.9])

        with tempfile.NamedTemporaryFile(suffix=".stru", delete=False) as f:
            fname = f.name
        try:
            io.write_stru(fname, positions, species, cell, title="test")
            pos2, spec2, cell2 = io.read_stru(fname)

            np.testing.assert_allclose(positions, pos2, atol=1e-5)
            assert list(species) == list(spec2)
            np.testing.assert_allclose(cell, cell2, atol=1e-5)
        finally:
            os.unlink(fname)

    def test_write_stru_non_orthorhombic_raises(self):
        """write_stru should raise ValueError for non-orthorhombic cells."""
        positions = np.array([[0.0, 0.0, 0.0]])
        species = np.array(["NB"])
        cell = np.array([[3.9, 0.5, 0.0],
                         [0.0, 3.9, 0.0],
                         [0.0, 0.0, 3.9]])

        with tempfile.NamedTemporaryFile(suffix=".stru", delete=False) as f:
            fname = f.name
        try:
            with pytest.raises(ValueError, match="orthorhombic"):
                io.write_stru(fname, positions, species, cell)
        finally:
            os.unlink(fname)

    def test_read_stru_missing_cell_raises(self):
        """read_stru should raise ValueError if no cell line is found."""
        with tempfile.NamedTemporaryFile(suffix=".stru", mode='w', delete=False) as f:
            f.write("title test\nspcgr P 1\natoms\nNB  0.0, 0.0, 0.0, 0.5\n")
            fname = f.name
        try:
            with pytest.raises(ValueError, match="No cell"):
                io.read_stru(fname)
        finally:
            os.unlink(fname)

    def test_read_stru_malformed_atom_raises(self):
        """read_stru should give a context-aware error for bad atom lines."""
        with tempfile.NamedTemporaryFile(suffix=".stru", mode='w', delete=False) as f:
            f.write("title test\ncell 3.9, 3.9, 3.9, 90, 90, 90\natoms\nNB  bad_data\n")
            fname = f.name
        try:
            with pytest.raises(ValueError, match="cannot parse atom line"):
                io.read_stru(fname)
        finally:
            os.unlink(fname)
