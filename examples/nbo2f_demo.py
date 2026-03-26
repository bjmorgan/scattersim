"""NbO2F diffuse scattering demo using scattersim.

Builds a 12x12x12 supercell with uncorrelated OOF anion ordering and
parametrised Nb displacements (delta=0.144 A, away from F toward O),
then computes zone-axis electron diffraction patterns.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

from scattersim import qgrid, fourier

np.random.seed(42)

# ============================================================
# Parameters
# ============================================================
N = 12
a = 3.90  # lattice parameter (Angstroms)
delta_nb = 0.144  # Nb displacement magnitude (Angstroms)

# ============================================================
# Build OOF-ordered structure with Nb displacements
# ============================================================

def build_nbo2f(N, a, delta_nb, seed=42):
    """Build an NbO2F supercell with OOF ordering and Nb displacements.

    Returns (positions, species) as plain NumPy arrays.
    """
    rng = np.random.RandomState(seed)

    # Anion arrays: 0=O, 1=F
    anion_x = np.zeros((N, N, N), dtype=int)
    anion_y = np.zeros((N, N, N), dtype=int)
    anion_z = np.zeros((N, N, N), dtype=int)

    # X-columns: along a-axis, indexed [j,k,i]
    for j in range(N):
        for k in range(N):
            offset = rng.randint(0, 3)
            for i in range(N):
                anion_x[j, k, i] = 1 if (i + offset) % 3 == 2 else 0

    # Y-columns: along b-axis, indexed [i,k,j]
    for i in range(N):
        for k in range(N):
            offset = rng.randint(0, 3)
            for j in range(N):
                anion_y[i, k, j] = 1 if (j + offset) % 3 == 2 else 0

    # Z-columns: along c-axis, indexed [i,j,k]
    for i in range(N):
        for j in range(N):
            offset = rng.randint(0, 3)
            for k in range(N):
                anion_z[i, j, k] = 1 if (k + offset) % 3 == 2 else 0

    # Nb displacements: away from F (s=1), toward O (s=0)
    nb_disp = np.zeros((N, N, N, 3))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                nb_disp[i, j, k, 0] = delta_nb * (anion_x[j, k, (i-1) % N] - anion_x[j, k, i])
                nb_disp[i, j, k, 1] = delta_nb * (anion_y[i, k, (j-1) % N] - anion_y[i, k, j])
                nb_disp[i, j, k, 2] = delta_nb * (anion_z[i, j, (k-1) % N] - anion_z[i, j, k])

    elements = []
    positions = []

    # Nb atoms
    for i in range(N):
        for j in range(N):
            for k in range(N):
                elements.append('NB')
                positions.append([
                    i * a + nb_disp[i, j, k, 0],
                    j * a + nb_disp[i, j, k, 1],
                    k * a + nb_disp[i, j, k, 2],
                ])

    # X-anions at (i+0.5, j, k)
    for j in range(N):
        for k in range(N):
            for i in range(N):
                el = 'F' if anion_x[j, k, i] else 'O'
                dx = 0.5 * (nb_disp[i, j, k, 0] + nb_disp[(i+1) % N, j, k, 0])
                elements.append(el)
                positions.append([(i + 0.5) * a + dx, j * a, k * a])

    # Y-anions at (i, j+0.5, k)
    for i in range(N):
        for k in range(N):
            for j in range(N):
                el = 'F' if anion_y[i, k, j] else 'O'
                dy = 0.5 * (nb_disp[i, j, k, 1] + nb_disp[i, (j+1) % N, k, 1])
                elements.append(el)
                positions.append([i * a, (j + 0.5) * a + dy, k * a])

    # Z-anions at (i, j, k+0.5)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                el = 'F' if anion_z[i, j, k] else 'O'
                dz = 0.5 * (nb_disp[i, j, k, 2] + nb_disp[i, j, (k+1) % N, 2])
                elements.append(el)
                positions.append([i * a, j * a, (k + 0.5) * a + dz])

    return np.array(positions), np.array(elements)


# ============================================================
# Build structure
# ============================================================
print("Building 12x12x12 NbO2F supercell...")
positions, species = build_nbo2f(N, a, delta_nb)
cell = np.diag([N * a, N * a, N * a])

n_nb = np.sum(species == 'NB')
n_o = np.sum(species == 'O')
n_f = np.sum(species == 'F')
print(f"  Atoms: {len(positions)} (Nb:{n_nb}, O:{n_o}, F:{n_f}, O:F={n_o/n_f:.2f})")

# ============================================================
# Constant electron form factors (for comparison with Brink)
# ============================================================
def electron_const(species_label, s):
    """Constant electron scattering factors."""
    ff_map = {'NB': 41.0, 'O': 8.0, 'F': 9.0}
    return np.full_like(s, ff_map[species_label])

# ============================================================
# Plotting helper
# ============================================================
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

def plot_zone_axis(I, Q, title, filename_stem):
    """Plot zone-axis pattern."""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        np.log10(I + 1),
        extent=Q.extent,
        origin='lower',
        cmap=cmc.lipari,
        aspect='equal',
    )
    ax.set_xlabel(f'Q along {Q.v1_label} ($\\AA^{{-1}}$)')
    ax.set_ylabel(f'Q along {Q.v2_label} ($\\AA^{{-1}}$)')
    ax.set_title(title)
    # Colourbar matched to plot height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.1)
    plt.colorbar(im, cax=cax, label='$\\log_{10}(I+1)$')
    plt.tight_layout()
    fname = f'examples/{filename_stem}.png'
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ============================================================
# Compute and plot zone axis patterns
# ============================================================
zones = [
    ([0, 0, 1],   'NbO$_2$F [001] zone axis',   'nbo2f_001'),
    ([1, 1, 4],   'NbO$_2$F [114] zone axis',   'nbo2f_114'),
    ([-1, 1, 3],  'NbO$_2$F [$\\bar{1}$13] zone axis', 'nbo2f_m113'),
]

for uvw, title, stem in zones:
    label = str(uvw)
    print(f"\nComputing {label} zone axis pattern...")
    Q = qgrid.zone_axis_grid(uvw, cell, extent=5.0, npts=401)
    I = fourier.intensity((positions, species), Q, ff=electron_const, progress=True)
    plot_zone_axis(I, Q, title, stem)

print("\nDone.")
