"""Exact diagonalization of three rotating, interacting bosons in 2D.

This is a clean Python implementation of the supplied assignment and MATLAB
files.  Units are chosen so that energy is measured in hbar*omega and length
in the oscillator length a_perp.  Only lowest-Landau-level/Fock-Darwin orbitals
with non-negative angular momentum m are retained.

Run examples
------------
    python RotatingBosonsExactDiagonalization.py --task energy
    python RotatingBosonsExactDiagonalization.py --task density
    python RotatingBosonsExactDiagonalization.py --task pair --separations 0 0.5 1
    python RotatingBosonsExactDiagonalization.py --task all

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from math import factorial, pi, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def occupations_with_N_and_L(N: int, L: int, m_max: int) -> list[tuple[int, ...]]:
    """Return all bosonic occupations |n_0,n_1,...> with sum n=N, sum m*n_m=L.

    This recursion directly generates the Fock basis.  It replaces the MATLAB
    integer-partition/permutation routines, which are difficult to generalize.
    """
    states: list[tuple[int, ...]] = []

    def fill(m: int, particles_left: int, angular_left: int, occ: list[int]) -> None:
        if m == m_max:
            if angular_left == m * particles_left:
                states.append(tuple(occ + [particles_left]))
            return
        for n_m in range(particles_left + 1):
            used_L = m * n_m
            if used_L <= angular_left:
                fill(m + 1, particles_left - n_m, angular_left - used_L,
                     occ + [n_m])

    fill(0, N, L, [])
    return states


def apply_operators(state: tuple[int, ...], operations: list[tuple[str, int]]):
    """Apply operators from right to left and return (new_state, amplitude).

    Each operation is ("ann" or "cre", orbital).  The list is supplied in the
    actual order of action on the ket.  Bosonic square-root factors are included:
    a_m|n_m> = sqrt(n_m)|n_m-1>, a_m^dag|n_m> = sqrt(n_m+1)|n_m+1>.
    """
    out = list(state)
    amplitude = 1.0
    for kind, m in operations:
        if kind == "ann":
            if out[m] == 0:
                return None, 0.0
            amplitude *= sqrt(out[m])
            out[m] -= 1
        elif kind == "cre":
            amplitude *= sqrt(out[m] + 1)
            out[m] += 1
        else:
            raise ValueError(f"Unknown operator {kind!r}")
    return tuple(out), amplitude


def contact_integral(k: int, l: int, p: int, q: int) -> float:
    r"""LLL contact matrix element used by the supplied MATLAB Hamiltonian.

    I_klpq = delta_(k+l,p+q) (k+l)! /
             [2^(k+l) 2*pi sqrt(k!l!p!q!)].
    """
    if k + l != p + q:
        return 0.0
    s = k + l
    return factorial(s) / (
        2.0**s * 2.0 * pi
        * sqrt(factorial(k) * factorial(l) * factorial(p) * factorial(q))
    )


@dataclass
class SectorResult:
    L: int
    basis: list[tuple[int, ...]]
    H: np.ndarray
    energies: np.ndarray
    vectors: np.ndarray

    @property
    def ground_energy(self) -> float:
        return float(self.energies[0])

    @property
    def ground_state(self) -> np.ndarray:
        return self.vectors[:, 0]


def solve_sector(N: int, L: int, xi: float, g2d: float = 1.0,
                 a_perp: float = 1.0) -> SectorResult:
    r"""Construct and diagonalize the Hamiltonian in one fixed-L sector.

    H = sum_m [1+m(1-xi)] a_m^dag a_m
        + (g2d/(2*a_perp^2)) sum_klpq I_klpq a_k^dag a_l^dag a_p a_q.

    Because [H,L]=0, every L sector is independent.  The explicit ordered sum
    over k,l,p,q is intentional: it implements the displayed assignment formula
    without hand-added factors of 2 and therefore avoids double counting.
    """
    m_max = max(L, 0)
    basis = occupations_with_N_and_L(N, L, m_max)
    index = {state: i for i, state in enumerate(basis)}
    H = np.zeros((len(basis), len(basis)), dtype=float)

    for col, ket in enumerate(basis):
        H[col, col] += sum((1.0 + m * (1.0 - xi)) * n
                           for m, n in enumerate(ket))
        for k, l, p, q in product(range(m_max + 1), repeat=4):
            integral = contact_integral(k, l, p, q)
            if integral == 0.0:
                continue
            # a_k^dag a_l^dag a_p a_q |ket>: q acts first.
            out, amp = apply_operators(
                ket, [("ann", q), ("ann", p), ("cre", l), ("cre", k)]
            )
            if out in index:
                H[index[out], col] += g2d / (2.0 * a_perp**2) * integral * amp

    H = 0.5 * (H + H.T)  # removes only roundoff-level asymmetry
    energies, vectors = np.linalg.eigh(H)
    return SectorResult(L, basis, H, energies, vectors)


def orbital(m: int, z: np.ndarray) -> np.ndarray:
    r"""Normalized Fock-Darwin/LLL orbital phi_m(z)=z^m exp(-|z|^2/2)/sqrt(pi m!)."""
    return z**m * np.exp(-0.5 * np.abs(z)**2) / sqrt(pi * factorial(m))


def expectation_operator(result: SectorResult, operations: list[tuple[str, int]]) -> complex:
    """Return <G|O|G> for an operator sequence acting on the ket."""
    c = result.ground_state
    lookup = {state: i for i, state in enumerate(result.basis)}
    value = 0.0j
    for col, ket in enumerate(result.basis):
        out, amp = apply_operators(ket, operations)
        row = lookup.get(out)
        if row is not None:
            value += np.conjugate(c[row]) * c[col] * amp
    return value


def one_body_density_matrix(result: SectorResult) -> np.ndarray:
    r"""gamma_ij=<a_i^dag a_j>.  Its trace must equal particle number N."""
    M = len(result.basis[0])
    gamma = np.empty((M, M), dtype=complex)
    for i, j in product(range(M), repeat=2):
        gamma[i, j] = expectation_operator(
            result, [("ann", j), ("cre", i)])
    return gamma


def two_body_density_matrix(result: SectorResult) -> np.ndarray:
    r"""Gamma_ijkl=<a_i^dag a_j^dag a_k a_l> with all ordered indices.

    Both i,j permutations and k,l permutations are included by the four full
    sums in the definition.  They must NOT also be inserted manually afterward.
    This directly resolves the double-counting question in the assignment.
    """
    M = len(result.basis[0])
    gamma2 = np.empty((M, M, M, M), dtype=complex)
    for i, j, k, l in product(range(M), repeat=4):
        gamma2[i, j, k, l] = expectation_operator(
            result, [("ann", l), ("ann", k), ("cre", j), ("cre", i)])
    return gamma2


def density_on_grid(result: SectorResult, z: np.ndarray) -> np.ndarray:
    r"""rho(z)=sum_ij phi_i*(z) phi_j(z) <a_i^dag a_j>."""
    gamma = one_body_density_matrix(result)
    phi = np.array([orbital(m, z) for m in range(gamma.shape[0])])
    rho = np.einsum("ixy,ij,jxy->xy", phi.conj(), gamma, phi, optimize=True)
    return np.real_if_close(rho).real


def pair_correlation_on_grid(result: SectorResult, z1: np.ndarray,
                             z2: np.ndarray, normalize: bool = False) -> np.ndarray:
    r"""Evaluate eta(z1,z2)=<psi^dag(z1)psi^dag(z2)psi(z1)psi(z2)>.

    If normalize=True, return g2=eta/[rho(z1)rho(z2)] where the denominator is
    nonzero.  The unnormalized eta is the quantity written in the assignment.
    """
    G = two_body_density_matrix(result)
    M = G.shape[0]
    p1 = np.array([orbital(m, z1) for m in range(M)])
    p2 = np.array([orbital(m, z2) for m in range(M)])
    eta = np.einsum("ixy,jxy,ijkl,kxy,lxy->xy",
                    p1.conj(), p2.conj(), G, p1, p2, optimize=True).real
    eta = np.maximum(eta, 0.0)  # suppress tiny negative roundoff
    if not normalize:
        return eta
    rho1, rho2 = density_on_grid(result, z1), density_on_grid(result, z2)
    return np.divide(eta, rho1 * rho2, out=np.zeros_like(eta),
                     where=rho1 * rho2 > 1e-12)


def print_sector_table(results: list[SectorResult]) -> None:
    print("\nFixed-angular-momentum ground states")
    print(" L   basis dimension       E0")
    print("--   ---------------   -----------")
    for r in results:
        print(f"{r.L:2d}   {len(r.basis):15d}   {r.ground_energy:11.7f}")


def plot_energy(N: int, xis: np.ndarray, outdir: Path) -> None:
    L_values = np.arange(N * (N - 1) + 1)
    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    for xi in xis:
        E = [solve_sector(N, int(L), float(xi)).ground_energy for L in L_values]
        ax.plot(L_values, E, marker="o", lw=2, label=fr"$\xi={xi:.2f}$")
    ax.set(xlabel=r"Total angular momentum $L$",
           ylabel=r"Ground-state energy $E/(\hbar\omega)$")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.savefig(outdir / "energy_versus_angular_momentum.png", dpi=220)


def plot_densities(results: list[SectorResult], extent: float,
                   points: int, outdir: Path) -> None:
    axis = np.linspace(-extent, extent, points)
    X, Y = np.meshgrid(axis, axis)
    Z = X + 1j * Y
    fig, axs = plt.subplots(2, 4, figsize=(13, 6.4), constrained_layout=True)
    for ax, result in zip(axs.flat, results):
        rho = density_on_grid(result, Z)
        image = ax.imshow(rho, origin="lower", extent=[-extent, extent] * 2,
                          cmap="viridis", interpolation="bilinear")
        ax.set_title(fr"$L={result.L}$")
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax, shrink=0.82)
    for ax in axs.flat[len(results):]:
        ax.axis("off")
    fig.supxlabel(r"$x/a_\perp$")
    fig.supylabel(r"$y/a_\perp$")
    fig.suptitle(r"Ground-state one-particle density $\rho(x,y)$")
    fig.savefig(outdir / "ground_state_density.png", dpi=220)


def plot_pair_correlations(results: list[SectorResult], separations: list[float],
                           extent: float, points: int, outdir: Path,
                           normalize: bool) -> None:
    axis = np.linspace(-extent, extent, points)
    X, Y = np.meshgrid(axis, axis)
    Z = X + 1j * Y
    fig, axs = plt.subplots(len(separations), len(results),
                            figsize=(2.7 * len(results), 2.6 * len(separations)),
                            squeeze=False, constrained_layout=True)
    for row, d in enumerate(separations):
        z1, z2 = Z - d / 2.0, Z + d / 2.0
        for col, result in enumerate(results):
            corr = pair_correlation_on_grid(result, z1, z2, normalize)
            ax = axs[row, col]
            image = ax.imshow(corr, origin="lower", extent=[-extent, extent] * 2,
                              cmap="magma", interpolation="bilinear")
            ax.set_title(fr"$L={result.L},\ d={d:g}$", fontsize=9)
            ax.set_aspect("equal")
            fig.colorbar(image, ax=ax, shrink=0.72)
    fig.supxlabel(r"midpoint $x/a_\perp$")
    fig.supylabel(r"midpoint $y/a_\perp$")
    label = r"normalized $g^{(2)}$" if normalize else r"pair density $\eta(z_1,z_2)$"
    fig.suptitle("Ground-state " + label)
    fig.savefig(outdir / ("normalized_pair_correlation.png" if normalize
                          else "pair_correlation.png"), dpi=220)


def sanity_checks(results: list[SectorResult], N: int) -> None:
    """Check Hermiticity, eigenvector normalization, and density-matrix traces."""
    for r in results:
        assert np.allclose(r.H, r.H.T.conj(), atol=1e-12)
        assert np.isclose(np.vdot(r.ground_state, r.ground_state), 1.0)
        gamma = one_body_density_matrix(r)
        assert np.isclose(np.trace(gamma).real, N, atol=1e-10)
        G = two_body_density_matrix(r)
        # sum_ij <a_i^dag a_j^dag a_j a_i> = N(N-1)
        pair_number = sum(G[i, j, j, i] for i in range(G.shape[0])
                          for j in range(G.shape[1])).real
        assert np.isclose(pair_number, N * (N - 1), atol=1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["energy", "density", "pair", "all"],
                        default="all")
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--xi", type=float, default=0.75,
                        help="rotation ratio Omega/omega (called gita in MATLAB)")
    parser.add_argument("--points", type=int, default=161)
    parser.add_argument("--extent", type=float, default=3.0)
    parser.add_argument("--separations", nargs="+", type=float,
                        default=[0.0, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--normalized-pair", action="store_true")
    parser.add_argument("--no-show", action="store_true",
                        help="save plots without opening interactive windows")
    # Use the script's own folder, not the terminal's current working folder.
    # This matters on Windows: VS Code may start PowerShell inside its protected
    # installation directory (C:\Program Files\Microsoft VS Code), where a
    # non-administrator cannot create result folders.
    default_output = Path(__file__).resolve().parent / "results_rotating_bosons"
    parser.add_argument("--output", type=Path, default=default_output,
                        help="result folder (default: beside this Python file)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    Lmax = args.N * (args.N - 1)  # Laughlin angular momentum for bosonic nu=1/2
    results = [solve_sector(args.N, L, args.xi) for L in range(Lmax + 1)]
    print_sector_table(results)
    sanity_checks(results, args.N)
    print("All algebraic checks passed.")

    if args.task in {"energy", "all"}:
        plot_energy(args.N, np.arange(0.75, 1.001, 0.05), args.output)
    if args.task in {"density", "all"}:
        plot_densities(results, args.extent, args.points, args.output)
    if args.task in {"pair", "all"}:
        plot_pair_correlations(results, args.separations, args.extent,
                               args.points, args.output, args.normalized_pair)
    print(f"Results saved in: {args.output.resolve()}")

    if args.no_show:
        plt.close("all")
    else:
        print("Displaying plots. Close all figure windows to finish the program.")
        plt.show()


if __name__ == "__main__":
    main()
