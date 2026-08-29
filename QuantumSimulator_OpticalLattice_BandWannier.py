"""Quantum Simulator Assignment 1 - optical-lattice bands and Wannier states.

=============================================================================
THEORY USED IN THE CODE
=============================================================================

1. One-dimensional optical lattice
----------------------------------
The single-particle Hamiltonian is

    H = -(hbar^2/2m) d^2/dx^2 + V0 sin^2(k x).

The recoil energy is E_R=hbar^2 k^2/(2m).  Introduce dimensionless variables

    X = k x,       qbar = q/k,       s = V0/E_R.

Then

    H/E_R = -d^2/dX^2 + s sin^2(X).

The potential period is a=pi/k.  Therefore the first Brillouin zone is
-k <= q <= k, or -1 <= qbar <= 1.

2. Bloch theorem and plane-wave basis
-------------------------------------
A Bloch eigenstate is expanded as

    psi_(band,qbar)(X) = sum_j C_(band,j)(qbar)
                               exp[i(qbar+2j)X].

The reciprocal lattice vector is 2k, which explains the separation 2j in the
plane-wave momenta.  Since

    sin^2(X) = 1/2 - (exp(i2X)+exp(-i2X))/4,

the lattice potential couples only neighbouring Fourier orders j and j+/-1.
The dimensionless Hamiltonian matrix is consequently tridiagonal:

    H_(j,j)     = (qbar+2j)^2 + s/2,
    H_(j,j+/-1) = -s/4.

Diagonalizing this Hermitian matrix at every qbar gives E_n(qbar)/E_R and the
Fourier coefficients C_(n,j)(qbar).  An odd basis size 2J+1 retains orders
j=-J,...,+J.  Comparing 3, 5, and 7 orders demonstrates truncation error.

3. Bloch wavefunctions
----------------------
At a selected qbar, each eigenvector returned by diagonalization contains the
coefficients C_(n,j).  Substitution into the Fourier expansion reconstructs
the complex Bloch wavefunction.  At qbar=0 the Hamiltonian is real and has
inversion symmetry, so properly gauged eigenstates can be chosen almost purely
real and with definite parity.

4. Wannier function
-------------------
The lowest-band Wannier function centered at lattice site R_l=l*pi is the
discrete Fourier transform of Bloch states across the Brillouin zone:

    w_l(X) = (1/sqrt(N_q)) sum_q exp(-i qbar R_l) psi_(0,qbar)(X).

Numerical eigensolvers give every eigenvector an arbitrary complex phase.  A
direct sum without correcting those phases may produce a noisy, delocalized
"Wannier function."  This code uses a parallel-transport gauge: successive
eigenvectors are phase-aligned so their overlap is positive.  The resulting
Wannier function is normalized numerically and shifted so its main peak is at
the chosen lattice minimum.

5. Tight-binding approximation
------------------------------
If the wells are deep and neighbouring Wannier orbitals overlap weakly, the
lowest band has the form

    E_TB(qbar) = E_c - 2t cos(pi qbar).

For a one-dimensional lattice the bandwidth is W=4t (equivalently W=2zt with
coordination z=2).  This code extracts E_c and t from the exact band edges and
compares the fitted tight-binding curve with plane-wave diagonalization.

For a deep sinusoidal lattice, the standard asymptotic tunnelling expression is

    t/E_R ~= (4/sqrt(pi)) s^(3/4) exp(-2 sqrt(s)).

It decreases exponentially for deep lattices.  The shallow-lattice maximum of
this asymptotic expression occurs at s=(3/4)^2=0.5625; the formula itself is
quantitatively reliable primarily in the deep-lattice regime.

6. Harmonic approximation
-------------------------
Near a minimum X=0,

    s sin^2(X) ~= s X^2.

Thus each sufficiently deep well behaves as a harmonic oscillator.  Its local
energies and normalized ground state in dimensionless X are

    E_n/E_R ~= (2n+1)sqrt(s),
    phi_0(X) = s^(1/8)/pi^(1/4) exp[-sqrt(s) X^2/2].

As s grows, the exact bands become flatter, their first gap approaches
2sqrt(s) E_R, and the lowest Wannier orbital approaches this Gaussian.

=============================================================================
OUTPUTS
=============================================================================

  FigureA_band_convergence.png
      Bands for s=1,5,10 using 3, 5, and 7 plane waves.

  FigureB1_B3_bloch_states.png
      First two qbar=0 Bloch states and Fourier coefficients for s=1,5,10.

  FigureB4_Wannier_functions.png
      Lowest-band Wannier functions for s=1,5,10.

  FigureC1_tight_binding_comparison.png
      Exact lowest band and fitted tight-binding dispersion.

  FigureC2_tunneling_versus_depth.png
      Deep-lattice tunnelling estimate versus s=V0/E_R.

  FigureC3_harmonic_band_comparison.png
      First two exact bands and harmonic energies for s=10,20,30.

  FigureC4_Wannier_harmonic_comparison.png
      Exact Wannier function and harmonic Gaussian for s=10,20,30.

"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def lattice_hamiltonian(qbar: float, depth: float, orders: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Return the dimensionless plane-wave Hamiltonian H/E_R and order labels.

    Parameters
    ----------
    qbar:
        Quasimomentum q/k in the reduced Brillouin zone [-1,1].
    depth:
        Dimensionless lattice depth s=V0/E_R.
    orders:
        Odd number of plane waves (3,5,7,...).  The Fourier orders are
        j=-(orders//2),...,+(orders//2).
    """
    if orders < 3 or orders % 2 == 0:
        raise ValueError("orders must be an odd integer >= 3")
    J = orders // 2
    j = np.arange(-J, J + 1)
    diagonal = (qbar + 2.0 * j) ** 2 + depth / 2.0
    H = np.diag(diagonal)
    coupling = -depth / 4.0
    H += np.diag(np.full(orders - 1, coupling), 1)
    H += np.diag(np.full(orders - 1, coupling), -1)
    return H, j


def solve_bands(qbar: np.ndarray, depth: float, orders: int = 7) -> np.ndarray:
    """Return all plane-wave eigenenergies E/E_R for every qbar."""
    energies = np.empty((len(qbar), orders))
    for iq, q in enumerate(qbar):
        H, _ = lattice_hamiltonian(float(q), depth, orders)
        energies[iq] = np.linalg.eigvalsh(H)
    return energies


def solve_state(qbar: float, depth: float, orders: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sorted energies, eigenvectors (columns), and Fourier orders j."""
    H, j = lattice_hamiltonian(qbar, depth, orders)
    energy, vectors = np.linalg.eigh(H)
    return energy, vectors, j


def reconstruct_bloch(coefficients: np.ndarray, j: np.ndarray,
                      qbar: float, X: np.ndarray) -> np.ndarray:
    r"""Compute psi(X)=sum_j C_j exp[i(qbar+2j)X]."""
    phase = np.exp(1j * (qbar + 2.0 * j[:, None]) * X[None, :])
    return coefficients @ phase


def phase_align_coefficients(coefficients: np.ndarray) -> np.ndarray:
    """Apply the parallel-transport gauge along increasing quasimomentum.

    Eigenvectors differing only by a sign/phase describe the same Bloch state,
    but inconsistent phases destroy a Fourier-constructed Wannier function.
    This routine makes each overlap <u_(q-dq)|u_q> real and positive.
    """
    aligned = np.asarray(coefficients, dtype=complex).copy()
    # Give the first vector a deterministic phase.
    pivot = np.argmax(np.abs(aligned[0]))
    aligned[0] *= np.exp(-1j * np.angle(aligned[0, pivot]))
    for iq in range(1, len(aligned)):
        overlap = np.vdot(aligned[iq - 1], aligned[iq])
        if abs(overlap) > 1e-14:
            aligned[iq] *= np.exp(-1j * np.angle(overlap))
    return aligned


def lowest_band_wannier(depth: float, X: np.ndarray, orders: int = 15,
                        nq: int = 401, site: int = 0) -> np.ndarray:
    """Construct and normalize the lowest-band Wannier function w_site(X).

    The qbar endpoint +1 is excluded because -1 and +1 are the same point of
    the Brillouin-zone circle.  The site coordinate is R_site=site*pi in X=kx.
    """
    q_values = np.linspace(-1.0, 1.0, nq, endpoint=False)
    coefficients = np.empty((nq, orders), dtype=complex)
    _, j = lattice_hamiltonian(0.0, depth, orders)
    for iq, q in enumerate(q_values):
        _, vectors, _ = solve_state(float(q), depth, orders)
        coefficients[iq] = vectors[:, 0]
    coefficients = phase_align_coefficients(coefficients)

    wannier = np.zeros_like(X, dtype=complex)
    R_site = site * np.pi
    for q, coeff in zip(q_values, coefficients):
        psi_q = reconstruct_bloch(coeff, j, float(q), X)
        wannier += np.exp(-1j * q * R_site) * psi_q
    wannier /= np.sqrt(nq)

    # Fix one remaining global phase so the central peak is positive and real.
    center = np.argmin(np.abs(X - R_site))
    wannier *= np.exp(-1j * np.angle(wannier[center]))
    norm = np.trapezoid(np.abs(wannier) ** 2, X)
    return wannier / np.sqrt(norm)


def harmonic_ground_state(depth: float, X: np.ndarray) -> np.ndarray:
    """Normalized ground state of -d^2/dX^2+s X^2."""
    return depth ** 0.125 / np.pi ** 0.25 * np.exp(-0.5 * np.sqrt(depth) * X**2)


def tight_binding_from_band(qbar: np.ndarray, lowest_band: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Fit band edges to E_c-2t cos(pi qbar); return curve, t, and E_c."""
    i_center = np.argmin(np.abs(qbar))
    E_min = lowest_band[i_center]
    E_max = 0.5 * (lowest_band[0] + lowest_band[-1])
    t_hop = (E_max - E_min) / 4.0  # W=4t in one dimension
    E_center = 0.5 * (E_max + E_min)
    return E_center - 2.0 * t_hop * np.cos(np.pi * qbar), t_hop, E_center


def asymptotic_tunneling(depth: np.ndarray) -> np.ndarray:
    r"""Deep-lattice estimate t/E_R=(4/sqrt(pi))s^(3/4)exp(-2sqrt(s))."""
    return 4.0 / np.sqrt(np.pi) * depth**0.75 * np.exp(-2.0 * np.sqrt(depth))


def figure_a_band_convergence(output: Path, qbar: np.ndarray) -> None:
    """Assignment Figure A: 3-, 5-, and 7-wave convergence."""
    depths = [1.0, 5.0, 10.0]
    styles = {3: ("o", None), 5: (None, "--"), 7: (None, "-")}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), constrained_layout=True, sharey=True)
    for ax, depth in zip(axes, depths):
        for orders in [3, 5, 7]:
            energy = solve_bands(qbar, depth, orders)
            marker, linestyle = styles[orders]
            for band in range(orders):
                label = f"{orders} plane waves" if band == 0 else None
                if orders == 3:
                    ax.plot(qbar[::8], energy[::8, band], linestyle="none", marker=marker,
                            markersize=2.7, color="#ce6b27", label=label)
                else:
                    ax.plot(qbar, energy[:, band], linestyle=linestyle, linewidth=1.3,
                            color="#4771b2" if orders == 5 else "black", label=label)
        ax.set(title=fr"$V_0={depth:g}E_R$", xlabel=r"Reduced quasimomentum $q/k$",
               xlim=(-1, 1), ylim=(0, 30))
        ax.grid(alpha=0.18)
    axes[0].set_ylabel(r"Energy $E/E_R$")
    axes[-1].legend(fontsize=8, loc="upper center")
    fig.suptitle("Plane-wave truncation and reduced-zone energy bands", fontsize=15)
    fig.savefig(output / "FigureA_band_convergence.png", dpi=240)


def figure_b_bloch_states(output: Path) -> None:
    """Assignment Figures B1-B3: qbar=0 wavefunctions and coefficients."""
    X = np.linspace(-4.0, 4.0, 1801)
    for depth in [1.0, 5.0, 10.0]:
        energy, vectors, j = solve_state(0.0, depth, 7)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.1), constrained_layout=True)
        for band in [0, 1]:
            coeff = vectors[:, band].astype(complex)
            # Deterministic gauge: make the largest coefficient positive.
            pivot = np.argmax(np.abs(coeff))
            coeff *= np.exp(-1j * np.angle(coeff[pivot]))
            psi = reconstruct_bloch(coeff, j, 0.0, X)
            axes[0, band].plot(X, psi.real, lw=2, label="Real part")
            axes[0, band].plot(X, psi.imag, lw=1.6, ls="--", label="Imaginary part")
            axes[0, band].set(title=fr"Band {band}: $E/E_R={energy[band]:.5f}$",
                              xlabel=r"$X=kx$", ylabel=r"$\psi_{n,0}(X)$")
            axes[0, band].grid(alpha=0.2)
            axes[0, band].legend(fontsize=8)
            axes[1, band].stem(j, coeff.real, linefmt="#b5282f", markerfmt="o", basefmt="k-")
            axes[1, band].set(xticks=j, xlabel=r"Fourier order $j$",
                              ylabel=r"Coefficient $C_{n,j}$",
                              title=fr"Fourier coefficients, band {band}")
            axes[1, band].grid(alpha=0.2)
        fig.suptitle(fr"Bloch states at $q=0$, $V_0={depth:g}E_R$", fontsize=15)
        fig.savefig(output / f"FigureB_Bloch_states_V{depth:g}ER.png", dpi=240)


def figure_b4_wannier(output: Path) -> None:
    """Assignment Figure B4: lowest-band Wannier orbitals."""
    X = np.linspace(-4.0 * np.pi, 4.0 * np.pi, 3001)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True, sharey=True)
    for ax, depth in zip(axes, [1.0, 5.0, 10.0]):
        w = lowest_band_wannier(depth, X)
        ax.plot(X / np.pi, w.real, lw=2, color="#2b7a55", label="Re(w)")
        ax.plot(X / np.pi, w.imag, lw=1.3, ls="--", color="#8b4ea1", label="Im(w)")
        ax.set(title=fr"$V_0={depth:g}E_R$", xlabel=r"Position $X/\pi=x/a$",
               xlim=(-4, 4))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(r"Normalized Wannier function $w_0(X)$")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Lowest-band Wannier function at the central lattice site", fontsize=15)
    fig.savefig(output / "FigureB4_Wannier_functions.png", dpi=240)


def figure_c1_tight_binding(output: Path, qbar: np.ndarray) -> None:
    """Assignment Figure C1: exact lowest band versus tight binding."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True, sharey=False)
    for ax, depth in zip(axes, [1.0, 5.0, 10.0]):
        exact = solve_bands(qbar, depth, 15)[:, 0]
        tb, t_hop, _ = tight_binding_from_band(qbar, exact)
        ax.plot(qbar, exact, lw=2.2, color="black", label="Plane-wave result")
        ax.plot(qbar, tb, lw=2, ls="--", color="#d07020", label="Tight binding")
        ax.set(title=fr"$V_0={depth:g}E_R$,  $t/E_R={t_hop:.4g}$",
               xlabel=r"$q/k$")
        ax.set_ylabel(r"Lowest-band energy $E/E_R$")
        ax.grid(alpha=0.2)
    axes[-1].legend(fontsize=8)
    fig.suptitle("Tight-binding approximation improves as the lattice deepens", fontsize=15)
    fig.savefig(output / "FigureC1_tight_binding_comparison.png", dpi=240)


def figure_c2_tunneling(output: Path) -> None:
    """Assignment Figure C2: asymptotic tunnelling versus lattice depth."""
    depth = np.linspace(0.02, 10.0, 900)
    tunneling = asymptotic_tunneling(depth)
    peak_depth = (3.0 / 4.0) ** 2
    fig, ax = plt.subplots(figsize=(7.6, 5.0), constrained_layout=True)
    ax.plot(depth, tunneling, lw=2.5, color="#b02a37")
    ax.axvline(peak_depth, color="gray", ls="--", lw=1.3,
               label=fr"formal maximum $s=(3/4)^2={peak_depth:.4f}$")
    ax.set(xlabel=r"Lattice depth $s=V_0/E_R$", ylabel=r"Estimated tunnelling $t/E_R$",
           title="Deep-lattice tunnelling estimate")
    ax.grid(alpha=0.23)
    ax.legend(fontsize=9)
    fig.savefig(output / "FigureC2_tunneling_versus_depth.png", dpi=240)


def figure_c3_harmonic_bands(output: Path, qbar: np.ndarray) -> None:
    """Assignment Figure C3: exact first bands versus harmonic energies."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), constrained_layout=True, sharey=True)
    for ax, depth in zip(axes, [10.0, 20.0, 30.0]):
        exact = solve_bands(qbar, depth, 21)
        ax.plot(qbar, exact[:, 0], color="#1d5fa7", lw=2.2, label="Exact band 0")
        ax.plot(qbar, exact[:, 1], color="#b52a33", lw=2.2, label="Exact band 1")
        ax.axhline(np.sqrt(depth), color="#1d5fa7", ls="--", lw=1.5,
                   label=r"HO $E_0=\sqrt{s}$")
        ax.axhline(3.0 * np.sqrt(depth), color="#b52a33", ls="--", lw=1.5,
                   label=r"HO $E_1=3\sqrt{s}$")
        ax.set(title=fr"$V_0={depth:g}E_R$", xlabel=r"$q/k$", xlim=(-1, 1), ylim=(0, 20))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(r"Energy $E/E_R$")
    axes[-1].legend(fontsize=7.5)
    fig.suptitle("Flattening of deep-lattice bands and harmonic-well energies", fontsize=15)
    fig.savefig(output / "FigureC3_harmonic_band_comparison.png", dpi=240)


def figure_c4_harmonic_wannier(output: Path) -> None:
    """Assignment Figure C4: exact Wannier orbital versus HO Gaussian."""
    X = np.linspace(-2.5 * np.pi, 2.5 * np.pi, 2401)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True, sharey=True)
    for ax, depth in zip(axes, [10.0, 20.0, 30.0]):
        w = lowest_band_wannier(depth, X, orders=21)
        ho = harmonic_ground_state(depth, X)
        # Both functions have unit integral in X and a positive central peak.
        ax.plot(X / np.pi, w.real, lw=2.2, color="black", label="Exact Wannier")
        ax.plot(X / np.pi, ho, lw=2, ls="--", color="#d17220", label="Harmonic Gaussian")
        ax.set(title=fr"$V_0={depth:g}E_R$", xlabel=r"$X/\pi=x/a$", xlim=(-2.5, 2.5))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Normalized wavefunction")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Deep-lattice Wannier orbital approaches the harmonic ground state", fontsize=15)
    fig.savefig(output / "FigureC4_Wannier_harmonic_comparison.png", dpi=240)


def numerical_checks() -> None:
    """Verify matrix Hermiticity, free-particle limit, and Wannier normalization."""
    H, _ = lattice_hamiltonian(0.37, 5.0, 7)
    assert np.allclose(H, H.conj().T)

    # At V0=0 the plane waves are uncoupled and energies are (qbar+2j)^2.
    q_test = 0.23
    H0, j = lattice_hamiltonian(q_test, 0.0, 7)
    assert np.allclose(np.sort(np.diag(H0)), np.linalg.eigvalsh(H0))
    assert np.allclose(np.diag(H0), (q_test + 2.0 * j) ** 2)

    X = np.linspace(-3.0 * np.pi, 3.0 * np.pi, 1801)
    w = lowest_band_wannier(5.0, X, orders=11, nq=201)
    assert np.isclose(np.trapezoid(np.abs(w) ** 2, X), 1.0, atol=2e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="save figures without opening windows")
    parser.add_argument("--q-points", type=int, default=401)
    default_output = Path(__file__).resolve().parent / "results_assignment1"
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    qbar = np.linspace(-1.0, 1.0, args.q_points)
    numerical_checks()
    print("Numerical checks passed: Hermiticity, free-particle limit, and Wannier normalization.")

    figure_a_band_convergence(args.output, qbar)
    figure_b_bloch_states(args.output)
    figure_b4_wannier(args.output)
    figure_c1_tight_binding(args.output, qbar)
    figure_c2_tunneling(args.output)
    figure_c3_harmonic_bands(args.output, qbar)
    figure_c4_harmonic_wannier(args.output)

    print(f"All figures saved in: {args.output.resolve()}")
    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
