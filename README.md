# scattersim

A Python library for computing kinematic scattering patterns from supercell structures, using the direct Fourier sum method.

## Features

- **Zone-axis electron diffraction patterns** from arbitrary crystal structures
- **X-ray and electron form factors** (Waasmaier-Kirfel and Peng parameterisations)
- **Multi-frame averaging** for Monte Carlo / molecular dynamics snapshots
- **1D reciprocal-space line profiles** for quantitative analysis
- Accepts **ASE Atoms** objects or plain NumPy arrays
- Validated against [DISCUS](https://tproffen.github.io/DiffuseCode/)

## Installation

```bash
git clone <repo-url>
cd scattersim
pip install -e ".[dev]"
```

Requires Python >= 3.11 and NumPy. Optional dependencies: `tqdm` (progress bars), `ase` (structure I/O), `matplotlib` and `cmcrameri` (plotting).

## Quick start

```python
import numpy as np
from scattersim.io import from_ase
from scattersim import qgrid, fourier
from ase.io import read

# Load structure and map elements to form factor species
atoms = read('POSCAR')
positions, species, cell = from_ase(atoms, species_map={
    'Nb': 'Nb5+', 'O': 'O2-', 'F': 'F1-',
})

# Build a [001] zone-axis Q-grid
Q = qgrid.zone_axis_grid(uvw=[0, 0, 1], cell=cell, extent=5.0, npts=201)

# Compute intensity (single frame)
I = fourier.intensity((positions, species), Q, ff='xray')

# Multi-frame averaging from MC snapshots
from glob import glob
frames = []
for path in sorted(glob('mc_snapshots/POSCAR_*')):
    p, s, c = from_ase(read(path), species_map={...})
    frames.append((p, s))
I_avg = fourier.intensity(frames, Q, ff='electron')
```

You can also pass raw NumPy arrays directly, without ASE:

```python
positions = np.array([[0, 0, 0], [1.95, 0, 0], ...])  # Angstroms
species = np.array(['Nb5+', 'O2-', ...])
cell = np.diag([46.8, 46.8, 46.8])  # supercell vectors in Angstroms

Q = qgrid.zone_axis_grid([1, 1, 4], cell, extent=5.0, npts=201)
I = fourier.intensity((positions, species), Q, ff='xray')
```

## Modules

| Module | Purpose |
|--------|---------|
| `scattersim.io` | Unpack ASE Atoms (`from_ase`), read/write DISCUS `.stru` files |
| `scattersim.form_factors` | X-ray (Waasmaier-Kirfel), electron (Peng), and DISCUS-matching form factors |
| `scattersim.qgrid` | Build 2D zone-axis grids and 1D reciprocal-space line grids |
| `scattersim.fourier` | Compute scattering intensity via direct Fourier sum |

## Form factors

The `ff` argument to `fourier.intensity()` selects the form factor model:

- `'xray'` — Waasmaier-Kirfel 5-Gaussian X-ray form factors (all elements and ions)
- `'electron'` — Peng 5-Gaussian electron form factors (neutral atoms; ion labels fall back to neutral with a warning)
- `'xray-discus'` — DISCUS-matching mode for numerical validation (replicates float32 coefficient bug and s-discretisation)
- Any callable `f(species, s) -> array` for custom form factors

### Species labels

The `species_map` passed to `from_ase()` maps element symbols to form factor table keys. This is **required** — there are no default oxidation states.

```python
# X-ray: use ionic species
species_map = {'Nb': 'Nb5+', 'O': 'O2-', 'F': 'F1-'}

# The same map works for electron diffraction — Peng falls back
# to neutral atoms with a warning, since the parameterisation
# covers neutral atoms only.
```

## Units

All inputs and outputs use consistent units:

- Positions: Angstroms
- Cell vectors: Angstroms
- Q-vectors: inverse Angstroms (physics convention: |Q| = 2pi/d)
- Form factors: electrons

## Examples

See `examples/nbo2f_demo.py` for a complete worked example computing zone-axis electron diffraction patterns from an NbO2F supercell with anion ordering and Nb displacements.

## Licence

MIT. See [LICENCE](LICENCE).

## References

- Waasmaier & Kirfel, Acta Cryst. A51 (1995) 416-431 — X-ray form factor parameterisation
- Peng, Ren, Dudarev & Whelan, Acta Cryst. A52 (1996) 257-276 — electron form factor parameterisation
- Proffen & Neder, J. Appl. Cryst. 30 (1997) 171-175 — DISCUS
