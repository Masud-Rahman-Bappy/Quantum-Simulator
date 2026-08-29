"""Two interacting bosons in a rotating 2D harmonic trap (LLL model).

This program is derived from the supplied assignment.  It reproduces its basis
table, Hamiltonian, eigenvalues/eigenvectors, two-particle probability maps for
five fixed positions of particle 2, and one-particle density maps.  It also
exports numerical CSV tables and a compact summary table figure.

=============================================================================
1. PHYSICAL MODEL
=============================================================================

The particles are identical bosons confined to a rapidly rotating isotropic
two-dimensional harmonic trap.  In the lowest-Landau-level (LLL) approximation,
only single-particle orbitals with radial quantum number zero and non-negative
angular momentum m are retained.  In oscillator units (hbar*omega for energy
and a_perp for length), their energies are

    epsilon_m = 1 + m(1-xi),

where xi=Omega/omega is the rotation frequency divided by the trap frequency.
As xi approaches 1, states of different angular momentum become nearly
degenerate before interactions are included.

The second-quantized Hamiltonian is

    H = sum_m epsilon_m a_m^dagger a_m
        + contact-interaction terms.

For N=2 bosons, the bosonic Laughlin angular momentum is

    L_max = N(N-1) = 2.

Keeping orbitals m=0,1,2 gives the four Fock states

    |200>  (L=0),   |110>  (L=1),
    |020>  (L=2),   |101>  (L=2).

Here |n0 n1 n2> specifies the occupation of orbitals m=0,1,2.

=============================================================================
2. HAMILTONIAN USED IN THE ASSIGNMENT
=============================================================================

Define the dimensionless contact-interaction scale

    X = g_2D/(4*pi*a_perp^2).

In the ordered basis [|200>,|110>,|020>,|101>], conservation of total angular
momentum makes H block diagonal:

        [2+2X          0              0              0       ]
        [  0       3-xi+2X           0              0       ]
    H = [  0           0         4-2xi+X           X       ]
        [  0           0             X          4-2xi+X    ].

The eigenpairs are therefore

    E(L=0)       = 2+2X,           |200>,
    E(L=1)       = 3-xi+2X,        |110>,
    E(L=2,+)     = 4-2xi+2X,       (|020>+|101>)/sqrt(2),
    E(L=2,-)     = 4-2xi,          (|101>-|020>)/sqrt(2).

The last state avoids contact interaction and is the two-boson bosonic
Laughlin state.  It is the lower state in the L=2 block for repulsive X>0.

=============================================================================
3. COORDINATE-SPACE WAVEFUNCTIONS
=============================================================================

Use the complex coordinate z=(x+i y)/a_perp.  The normalized LLL orbital is

    phi_m(z) = z^m exp(-|z|^2/2)/sqrt(pi*m!).

Fock states of identical bosons already imply symmetrized coordinate-space
wavefunctions.  The normalized states required for the assignment are

    Psi_L0(z1,z2) = exp[-(|z1|^2+|z2|^2)/2]/pi,

    Psi_L1(z1,z2) = (z1+z2) exp[-...]/(pi*sqrt(2)),

    Psi_L2_ground = (z1-z2)^2 exp[-...]/(2*sqrt(2)*pi),

    Psi_L2_excited = (z1+z2)^2 exp[-...]/(2*sqrt(2)*pi).

The ground-state factor (z1-z2)^2 makes the probability exactly zero when the
two particles coincide.  This correlation hole is how the Laughlin state
removes the contact-interaction energy while remaining symmetric under
z1<->z2 (the square removes the sign change).

The plotted two-particle probability for fixed z2 is |Psi(z1,z2)|^2.  It is not
renormalized as a function of z1, because the assignment displays slices of
the full joint probability density.

=============================================================================
4. ONE-PARTICLE DENSITY AND A CORRECTION TO THE ASSIGNMENT
=============================================================================

The marginal density for the labelled coordinate z1 is

    rho(z1) = integral |Psi(z1,z2)|^2 d^2z2.

For the four states above, with r^2=|z1|^2,

    rho_L0(r) = exp(-r^2)/pi,

    rho_L1(r) = (r^2+1) exp(-r^2)/(2*pi),

    rho_L2_ground(r) = rho_L2_excited(r)
                     = (r^4+4r^2+2) exp(-r^2)/(8*pi).

Every expression integrates to one over d^2z1.  If the conventional total
particle density is desired, multiply these marginals by N=2 so that its
integral is two.

The assignment's final L=2 density derivation omits interference contributions
from |z1 +/- z2|^4.  Its minus-sign expression can even become negative, which
is impossible for a probability density.  Exact integration shows that the
ground and excited L=2 states have the same one-particle marginal density;
their different two-particle correlations remain visible in the conditional
probability maps.

=============================================================================
5. GENERATED RESULTS
=============================================================================

The output folder contains:

  basis_states.csv
  hamiltonian_matrix.csv
  eigensystem.csv
  radial_density_profiles.csv
  Table_basis_hamiltonian_eigensystem.png
  Figure_energy_versus_rotation.png
  Figure_joint_probability_all_z2.png
  Figure_joint_probability_z2_*.png       (five assignment cases)
  Figure_one_particle_density_maps.png
  Figure_radial_density_profiles.png

"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STATE_LABELS = ["|200>", "|110>", "|020>", "|101>"]
TOTAL_L = np.array([0, 1, 2, 2], dtype=int)
OCCUPATIONS = np.array([[2, 0, 0], [1, 1, 0], [0, 2, 0], [1, 0, 1]], dtype=int)


@dataclass(frozen=True)
class ModelParameters:
    """Dimensionless parameters for the two-boson Hamiltonian."""

    xi: float = 0.75
    g2d: float = 1.0
    a_perp: float = 1.0

    @property
    def interaction_X(self) -> float:
        return self.g2d / (4.0 * np.pi * self.a_perp**2)


def hamiltonian(parameters: ModelParameters) -> np.ndarray:
    """Return the 4x4 Hamiltonian shown in the assignment."""
    xi = parameters.xi
    X = parameters.interaction_X
    H = np.zeros((4, 4), dtype=float)
    H[0, 0] = 2.0 + 2.0 * X
    H[1, 1] = 3.0 - xi + 2.0 * X
    H[2, 2] = H[3, 3] = 4.0 - 2.0 * xi + X
    H[2, 3] = H[3, 2] = X
    return H


def analytic_eigensystem(parameters: ModelParameters) -> list[dict[str, object]]:
    """Return the four analytic energy branches and normalized state vectors."""
    xi, X = parameters.xi, parameters.interaction_X
    invsqrt2 = 1.0 / np.sqrt(2.0)
    return [
        {"name": "L=0", "energy": 2.0 + 2.0 * X,
         "vector": np.array([1.0, 0.0, 0.0, 0.0]), "state": "|200>"},
        {"name": "L=1", "energy": 3.0 - xi + 2.0 * X,
         "vector": np.array([0.0, 1.0, 0.0, 0.0]), "state": "|110>"},
        {"name": "L=2 ground", "energy": 4.0 - 2.0 * xi,
         "vector": np.array([0.0, 0.0, -invsqrt2, invsqrt2]),
         "state": "(|101>-|020>)/sqrt(2)"},
        {"name": "L=2 excited", "energy": 4.0 - 2.0 * xi + 2.0 * X,
         "vector": np.array([0.0, 0.0, invsqrt2, invsqrt2]),
         "state": "(|101>+|020>)/sqrt(2)"},
    ]


def gaussian_factor(z1: np.ndarray, z2: complex | np.ndarray) -> np.ndarray:
    """Common LLL Gaussian exp[-(|z1|^2+|z2|^2)/2]."""
    return np.exp(-0.5 * (np.abs(z1) ** 2 + np.abs(z2) ** 2))


def psi_l0(z1: np.ndarray, z2: complex | np.ndarray) -> np.ndarray:
    """Normalized coordinate wavefunction corresponding to |200>."""
    return gaussian_factor(z1, z2) / np.pi


def psi_l1(z1: np.ndarray, z2: complex | np.ndarray) -> np.ndarray:
    """Normalized symmetric coordinate wavefunction corresponding to |110>."""
    return (z1 + z2) * gaussian_factor(z1, z2) / (np.pi * np.sqrt(2.0))


def psi_l2_ground(z1: np.ndarray, z2: complex | np.ndarray) -> np.ndarray:
    """Laughlin state: (z1-z2)^2 produces a second-order coincidence zero."""
    return (z1 - z2) ** 2 * gaussian_factor(z1, z2) / (2.0 * np.sqrt(2.0) * np.pi)


def psi_l2_excited(z1: np.ndarray, z2: complex | np.ndarray) -> np.ndarray:
    """Center-of-mass-like L=2 partner proportional to (z1+z2)^2."""
    return (z1 + z2) ** 2 * gaussian_factor(z1, z2) / (2.0 * np.sqrt(2.0) * np.pi)


WAVEFUNCTIONS = [psi_l0, psi_l1, psi_l2_ground, psi_l2_excited]
STATE_TITLES = ["L=0", "L=1", "L=2 ground (Laughlin)", "L=2 excited"]


def joint_probability(function, z1: np.ndarray, z2: complex) -> np.ndarray:
    """Return the joint probability slice |Psi(z1,z2)|^2."""
    return np.abs(function(z1, z2)) ** 2


def marginal_density(state_index: int, z: np.ndarray) -> np.ndarray:
    """Analytic one-particle marginal rho(z), normalized to integral rho=1."""
    r2 = np.abs(z) ** 2
    if state_index == 0:
        return np.exp(-r2) / np.pi
    if state_index == 1:
        return (r2 + 1.0) * np.exp(-r2) / (2.0 * np.pi)
    if state_index in (2, 3):
        return (r2**2 + 4.0 * r2 + 2.0) * np.exp(-r2) / (8.0 * np.pi)
    raise ValueError("state_index must be 0, 1, 2, or 3")


def export_csv_tables(output: Path, parameters: ModelParameters) -> None:
    """Write the basis, Hamiltonian, eigensystem, and radial profiles as CSV."""
    with (output / "basis_states.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["total_L", "n_m=0", "n_m=1", "n_m=2", "Fock_state"])
        for L, occ, label in zip(TOTAL_L, OCCUPATIONS, STATE_LABELS):
            writer.writerow([int(L), *map(int, occ), label])

    H = hamiltonian(parameters)
    with (output / "hamiltonian_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["bra/ket", *STATE_LABELS])
        for label, row in zip(STATE_LABELS, H):
            writer.writerow([label, *[f"{value:.12g}" for value in row]])

    with (output / "eigensystem.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["branch", "energy", "basis_expression", *[f"coefficient_{s}" for s in STATE_LABELS]])
        for item in analytic_eigensystem(parameters):
            writer.writerow([item["name"], f"{item['energy']:.12g}", item["state"],
                             *[f"{v:.12g}" for v in item["vector"]]])

    radius = np.linspace(0.0, 4.0, 401)
    z = radius.astype(complex)
    profiles = [marginal_density(i, z) for i in range(4)]
    with (output / "radial_density_profiles.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["radius", *STATE_TITLES])
        for row in zip(radius, *profiles):
            writer.writerow([f"{value:.12g}" for value in row])


def plot_summary_table(output: Path, parameters: ModelParameters) -> None:
    """Render basis, Hamiltonian, and analytic eigenpairs in a readable table figure."""
    H = hamiltonian(parameters)
    eig = analytic_eigensystem(parameters)
    fig = plt.figure(figsize=(13.2, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.35, 1.3])

    ax0 = fig.add_subplot(grid[0]); ax0.axis("off")
    basis_rows = [[int(L), *map(int, occ), label]
                  for L, occ, label in zip(TOTAL_L, OCCUPATIONS, STATE_LABELS)]
    table0 = ax0.table(cellText=basis_rows,
                       colLabels=["Total L", "n0", "n1", "n2", "Fock state"],
                       cellLoc="center", loc="center", colWidths=[0.13, 0.1, 0.1, 0.1, 0.25])
    table0.auto_set_font_size(False); table0.set_fontsize(10); table0.scale(1, 1.5)
    ax0.set_title("Two-boson Fock basis", fontsize=14, pad=8)

    ax1 = fig.add_subplot(grid[1]); ax1.axis("off")
    matrix_text = [[f"{v:.6f}" for v in row] for row in H]
    table1 = ax1.table(cellText=matrix_text, rowLabels=STATE_LABELS, colLabels=STATE_LABELS,
                       cellLoc="center", loc="center")
    table1.auto_set_font_size(False); table1.set_fontsize(9.5); table1.scale(1, 1.6)
    ax1.set_title(fr"Hamiltonian ($\xi={parameters.xi:g}$, $X={parameters.interaction_X:.6f}$)",
                  fontsize=14, pad=8)

    ax2 = fig.add_subplot(grid[2]); ax2.axis("off")
    eig_rows = [[item["name"], f"{item['energy']:.8f}", item["state"]] for item in eig]
    table2 = ax2.table(cellText=eig_rows, colLabels=["Branch", "Energy", "Eigenstate"],
                       cellLoc="center", loc="center", colWidths=[0.2, 0.18, 0.45])
    table2.auto_set_font_size(False); table2.set_fontsize(10); table2.scale(1, 1.55)
    ax2.set_title("Analytic eigensystem", fontsize=14, pad=8)
    fig.suptitle("Assignment numerical tables", fontsize=17)
    fig.savefig(output / "Table_basis_hamiltonian_eigensystem.png", dpi=230)


def plot_energy_versus_rotation(output: Path, parameters: ModelParameters) -> None:
    """Plot all four analytic energy branches and the lowest-energy envelope."""
    xi = np.linspace(0.0, 1.0, 501)
    X = parameters.interaction_X
    energies = np.vstack([
        np.full_like(xi, 2.0 + 2.0 * X),
        3.0 - xi + 2.0 * X,
        4.0 - 2.0 * xi,
        4.0 - 2.0 * xi + 2.0 * X,
    ])
    fig, ax = plt.subplots(figsize=(8.0, 5.4), constrained_layout=True)
    colors = ["#2f67b2", "#db7c20", "#26834f", "#9a47a8"]
    for E, label, color in zip(energies, ["L=0", "L=1", "L=2 ground", "L=2 excited"], colors):
        ax.plot(xi, E, lw=2.1, label=label, color=color)
    ax.plot(xi, np.min(energies, axis=0), "k--", lw=2.5, label="global ground energy")
    ax.axvline(parameters.xi, color="gray", lw=1.2, ls=":", label=fr"chosen $\xi={parameters.xi:g}$")
    ax.set(xlabel=r"Rotation ratio $\xi=\Omega/\omega$", ylabel=r"Energy $E/(\hbar\omega)$",
           title=fr"Two-boson energy branches ($X={X:.5f}$)")
    ax.grid(alpha=0.23); ax.legend(ncol=2, fontsize=9)
    fig.savefig(output / "Figure_energy_versus_rotation.png", dpi=240)


def safe_z2_name(z2: complex) -> str:
    """Return a Windows-safe filename token for a complex coordinate."""
    return f"x{z2.real:+g}_y{z2.imag:+g}".replace("+", "p").replace("-", "m").replace(".", "p")


def plot_probability_row(axes, z1: np.ndarray, z2: complex, extent: float) -> None:
    """Populate four axes with assignment probability maps for one fixed z2."""
    for ax, function, title in zip(axes, WAVEFUNCTIONS, STATE_TITLES):
        probability = joint_probability(function, z1, z2)
        image = ax.imshow(probability, origin="lower", extent=[-extent, extent] * 2,
                          cmap="turbo", interpolation="bilinear")
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1=\mathrm{Re}(z_1)$")
        ax.set_ylabel(r"$y_1=\mathrm{Im}(z_1)$")
        # Attach the colorbar to the same figure as its axis.  Using plt.colorbar
        # would depend on whichever figure is currently active when several
        # assignment figures are being assembled in the same loop.
        ax.figure.colorbar(image, ax=ax, shrink=0.72)


def plot_joint_probabilities(output: Path, extent: float, points: int) -> None:
    """Create individual and combined probability figures for all assignment z2 values."""
    axis = np.linspace(-extent, extent, points)
    Xgrid, Ygrid = np.meshgrid(axis, axis)
    z1 = Xgrid + 1j * Ygrid
    z2_values = [0 + 0j, 1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]

    combined, combined_axes = plt.subplots(5, 4, figsize=(14.0, 16.2), constrained_layout=True)
    for row, z2 in enumerate(z2_values):
        plot_probability_row(combined_axes[row], z1, z2, extent)
        combined_axes[row, 0].annotate(fr"fixed $z_2={z2.real:g}{z2.imag:+g}i$",
                                       xy=(-0.48, 0.5), xycoords="axes fraction",
                                       rotation=90, va="center", ha="center", fontsize=10)

        fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.8), constrained_layout=True)
        plot_probability_row(axes, z1, z2, extent)
        fig.suptitle(fr"Joint probability slice for fixed $z_2={z2.real:g}{z2.imag:+g}i$", fontsize=14)
        fig.savefig(output / f"Figure_joint_probability_z2_{safe_z2_name(z2)}.png", dpi=230)

    combined.suptitle(r"Assignment probability maps: $|\Psi(z_1,z_2)|^2$", fontsize=16)
    combined.savefig(output / "Figure_joint_probability_all_z2.png", dpi=230)


def plot_density_maps(output: Path, extent: float, points: int) -> None:
    """Plot corrected analytic one-particle marginal densities for all four states."""
    axis = np.linspace(-extent, extent, points)
    Xgrid, Ygrid = np.meshgrid(axis, axis)
    z = Xgrid + 1j * Ygrid
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.9), constrained_layout=True)
    for index, (ax, title) in enumerate(zip(axes, STATE_TITLES)):
        rho = marginal_density(index, z)
        image = ax.imshow(rho, origin="lower", extent=[-extent, extent] * 2,
                          cmap="turbo", interpolation="bilinear")
        ax.set(title=title, xlabel=r"$x/a_\perp$", ylabel=r"$y/a_\perp$")
        ax.set_aspect("equal")
        ax.figure.colorbar(image, ax=ax, shrink=0.72)
    fig.suptitle(r"Normalized one-particle marginal density $\rho(z)$", fontsize=15)
    fig.savefig(output / "Figure_one_particle_density_maps.png", dpi=240)


def plot_radial_profiles(output: Path) -> None:
    """Line chart comparing the radial density profiles and confirming L=2 equality."""
    radius = np.linspace(0.0, 4.0, 700)
    z = radius.astype(complex)
    fig, ax = plt.subplots(figsize=(8.0, 5.3), constrained_layout=True)
    styles = ["-", "--", "-.", ":"]
    for index, (title, style) in enumerate(zip(STATE_TITLES, styles)):
        ax.plot(radius, marginal_density(index, z), style, lw=2.4, label=title)
    ax.set(xlabel=r"Radius $r=|z|$", ylabel=r"Marginal density $\rho(r)$",
           title="Radial one-particle density profiles")
    ax.grid(alpha=0.23); ax.legend(fontsize=9)
    ax.text(0.98, 0.70, "The two L=2 curves overlap exactly", transform=ax.transAxes,
            ha="right", fontsize=9)
    fig.savefig(output / "Figure_radial_density_profiles.png", dpi=240)


def numerical_checks(parameters: ModelParameters) -> None:
    """Check Hermiticity, analytic eigenpairs, normalization, and correlation zero."""
    H = hamiltonian(parameters)
    assert np.allclose(H, H.T.conj())
    for item in analytic_eigensystem(parameters):
        vector = item["vector"]
        assert np.isclose(np.vdot(vector, vector), 1.0)
        assert np.allclose(H @ vector, item["energy"] * vector, atol=1e-12)

    # Numerically verify full two-particle normalization on a large square.
    axis = np.linspace(-5.0, 5.0, 501)
    x1, y1 = np.meshgrid(axis, axis)
    z1 = x1 + 1j * y1
    # First integrate over z1 for many z2 grid rows using separability through
    # a coarser four-dimensional check would be expensive.  Instead verify the
    # exact one-particle marginals integrate to one and spot-check the Laughlin zero.
    for index in range(4):
        rho = marginal_density(index, z1)
        integral_x = np.trapezoid(rho, axis, axis=1)
        integral = np.trapezoid(integral_x, axis)
        assert np.isclose(integral, 1.0, atol=2e-6)
    test_points = np.array([0j, 1 + 1j, -0.7 + 0.3j])
    assert np.allclose(psi_l2_ground(test_points, test_points), 0.0)

    # Confirm numerical and analytic matrix eigenvalues agree.
    numerical = np.linalg.eigvalsh(H)
    analytic = np.sort([item["energy"] for item in analytic_eigensystem(parameters)])
    assert np.allclose(numerical, analytic)


def print_results(parameters: ModelParameters) -> None:
    """Print the same main numerical results that are exported to tables."""
    print("\nModel parameters")
    print(f"  xi = {parameters.xi:.8g}")
    print(f"  g2D = {parameters.g2d:.8g}")
    print(f"  a_perp = {parameters.a_perp:.8g}")
    print(f"  X = g2D/(4*pi*a_perp^2) = {parameters.interaction_X:.10g}")
    print("\nHamiltonian in basis [|200>, |110>, |020>, |101>]:")
    print(np.array2string(hamiltonian(parameters), precision=8, suppress_small=True))
    print("\nAnalytic eigensystem:")
    for item in analytic_eigensystem(parameters):
        print(f"  {item['name']:12s} E={item['energy']:.10f}  {item['state']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xi", type=float, default=0.75, help="rotation ratio Omega/omega")
    parser.add_argument("--g2d", type=float, default=1.0)
    parser.add_argument("--a-perp", type=float, default=1.0)
    parser.add_argument("--extent", type=float, default=3.0)
    parser.add_argument("--points", type=int, default=241)
    parser.add_argument("--no-show", action="store_true")
    default_output = Path(__file__).resolve().parent / "results_two_boson_assignment"
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()

    if args.a_perp <= 0:
        parser.error("--a-perp must be positive")
    parameters = ModelParameters(args.xi, args.g2d, args.a_perp)
    args.output.mkdir(parents=True, exist_ok=True)

    numerical_checks(parameters)
    print("All checks passed: Hermiticity, eigensystem, density normalization, and Laughlin zero.")
    print_results(parameters)
    export_csv_tables(args.output, parameters)
    plot_summary_table(args.output, parameters)
    plot_energy_versus_rotation(args.output, parameters)
    plot_joint_probabilities(args.output, args.extent, args.points)
    plot_density_maps(args.output, args.extent, args.points)
    plot_radial_profiles(args.output)

    print(f"\nAll plots and tables saved in: {args.output.resolve()}")
    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
