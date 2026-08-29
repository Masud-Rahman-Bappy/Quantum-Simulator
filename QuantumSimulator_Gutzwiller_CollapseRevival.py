"""Quantum Simulator Assignment 2 - Gutzwiller theory and collapse/revival.

It produces:

1. Four Gutzwiller-amplitude distributions f_n (assignment Fig. 1).
2. Long-range first-order coherence g1 versus U/t (assignment Fig. 2).
3. The initial number distribution and its collapse/revival after a sudden
   interaction quench (assignment Figs. 3-4).

Theory used in the program
--------------------------
1. Bose-Hubbard model

   The assignment considers bosons in an optical lattice described by

     H - mu*N = -t sum_<i,j> a_j^dagger a_i
                + (U/2) sum_i n_i(n_i-1) - mu sum_i n_i.

   Here t is the nearest-neighbour tunnelling energy, U is the repulsive
   on-site interaction, mu is the chemical potential, and z is the number of
   nearest neighbours of each site (z=2 for a one-dimensional chain).

2. Homogeneous Gutzwiller approximation

   The many-body state is approximated as a product of identical single-site
   states,

     |Psi_G> = product_i |phi>_i,
     |phi>_i = sum_(n=0)^Nmax f_n |n>_i.

   Therefore, inter-site entanglement is neglected, while local number
   fluctuations are retained through the coefficients f_n.  For the uniform,
   unfrustrated ground state, all f_n can be chosen real and non-negative.
   Aligned phases maximize |<a>| and hence make the negative hopping energy as
   low as possible.

3. Energy functional and constraints

   The single-site superfluid order parameter is

     alpha = <a> = sum_(n=0)^(Nmax-1) sqrt(n+1) f_n f_(n+1).

   The energy per site becomes

  E = -z t |sum_(n=0)^(Nmax-1) sqrt(n+1) f_n f_(n+1)|^2
      + sum_n [U n(n-1)/2 - mu n] f_n^2,

   It is minimized subject to

     sum_n f_n^2 = 1,                 (normalization)
     sum_n n f_n^2 = nbar.            (specified mean occupation)

   Since nbar is fixed, -mu*nbar is a constant.  It shifts the reported energy
   but does not change the optimized f_n.  Nmax=8 is the Fock-space truncation
   specified in the assignment.

4. Physical interpretation of Figure 1

   When U/t is small, tunnelling favours a number-fluctuating superfluid, so
   f_n spreads over several Fock states.  When U/t is large at integer filling,
   repulsion suppresses number fluctuations and f_n approaches delta_(n,nbar),
   characteristic of a Mott-like state.  For non-integer nbar=1.1, at least two
   neighbouring occupations are needed to satisfy the mean-number constraint.

5. Long-range first-order coherence

   In a homogeneous product state, <a_i^dagger a_j>=|alpha|^2 for i != j.
   The normalized coherence plotted in Figure 2 is

     g1 = |alpha|^2/nbar.

   It approaches one for a coherent superfluid and zero when number is fixed
   and phase coherence is lost.

6. Sudden quench, collapse, and revival

   The initial state is the U=0, t=1 superfluid.  The lattice is then suddenly
   deepened so tunnelling is negligible and the post-quench Hamiltonian is

     H_f = (U_f/2) n(n-1).

   A sudden quench is too fast to change the initial probabilities |f_n|^2.
   Each Fock component only acquires its own phase,

     f_n(time) = f_n(0) exp[-i U_f n(n-1) time/(2 hbar)].

   Consequently,

  g1(time) = |sum_n sqrt(n+1) f_n* f_(n+1) exp(-i n U time/hbar)|^2/nbar.

   The different number-state phases first dephase (collapse) and then rephase
   (revival).  The exact revival period is T_rev=2*pi*hbar/U_f.  Re-minimizing
   after setting t=0 would instead construct a new equilibrium Mott state and
   would incorrectly erase the sudden-quench collapse/revival physics.

"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def order_parameter(f: np.ndarray) -> float:
    """Return <a> = sum sqrt(n+1) f_n f_(n+1) for real aligned f_n."""
    n = np.arange(len(f) - 1)
    return float(np.sum(np.sqrt(n + 1.0) * f[:-1] * f[1:]))


def energy_per_site(
    f: np.ndarray, U: float, t_hop: float, zcoord: int, mu: float
) -> float:
    """Homogeneous Gutzwiller energy per lattice site."""
    n = np.arange(len(f), dtype=float)
    kinetic = -zcoord * t_hop * order_parameter(f) ** 2
    onsite = np.sum((0.5 * U * n * (n - 1.0) - mu * n) * f**2)
    return float(kinetic + onsite)


def feasible_initial_amplitudes(Nmax: int, nbar: float) -> np.ndarray:
    """Construct an exactly normalized starting point with the requested mean."""
    if not 0.0 <= nbar <= Nmax:
        raise ValueError("nbar must lie between 0 and Nmax")
    lower = int(np.floor(nbar))
    upper = min(lower + 1, Nmax)
    probability = np.zeros(Nmax + 1)
    if lower == upper:
        probability[lower] = 1.0
    else:
        probability[lower] = upper - nbar
        probability[upper] = nbar - lower
    return np.sqrt(probability)


def optimize_gutzwiller(
    nbar: float,
    U: float,
    t_hop: float = 1.0,
    Nmax: int = 8,
    zcoord: int = 2,
    mu: float = 1.0,
    starts: int = 8,
    seed: int = 7,
) -> tuple[np.ndarray, float]:
    """Minimize E with normalization and mean-number constraints.

    Several deterministic/randomized starts are used because constrained local
    minimizers can depend on their initial point.  Only the lowest valid result
    is returned.  Non-negative amplitudes are sufficient for the unfrustrated
    homogeneous ground state: relative phases would only reduce |<a>|.
    """
    rng = np.random.default_rng(seed)
    n = np.arange(Nmax + 1, dtype=float)
    constraints = [
        {"type": "eq", "fun": lambda f: np.dot(f, f) - 1.0},
        {"type": "eq", "fun": lambda f: np.dot(n, f * f) - nbar},
    ]
    bounds = [(0.0, 1.0)] * (Nmax + 1)
    base = feasible_initial_amplitudes(Nmax, nbar)
    candidates = [base]

    # Broad positive starts; SLSQP projects them onto the equality constraints.
    for _ in range(starts - 1):
        trial = 0.65 * base + 0.35 * rng.random(Nmax + 1)
        trial /= np.linalg.norm(trial)
        candidates.append(trial)

    valid = []
    for x0 in candidates:
        result = minimize(
            energy_per_site,
            x0,
            args=(U, t_hop, zcoord, mu),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 3000, "disp": False},
        )
        norm_error = abs(np.dot(result.x, result.x) - 1.0)
        mean_error = abs(np.dot(n, result.x**2) - nbar)
        if result.success and norm_error < 2e-8 and mean_error < 2e-8:
            valid.append(result)

    if not valid:
        raise RuntimeError(f"Constrained minimization failed for U={U}, nbar={nbar}")
    best = min(valid, key=lambda result: result.fun)
    f = best.x / np.linalg.norm(best.x)
    return f, float(best.fun)


def first_order_coherence(f: np.ndarray, nbar: float) -> float:
    """g1=|<a>|^2/<n>; equals one for an ideal coherent state."""
    return order_parameter(f) ** 2 / nbar if nbar > 0 else 0.0


def quenched_coherence(
    f_initial: np.ndarray, nbar: float, U_final: float, time: np.ndarray, hbar: float = 1.0
) -> np.ndarray:
    """Exact Gutzwiller coherence after quenching to H=U*n*(n-1)/2."""
    n = np.arange(len(f_initial) - 1, dtype=float)
    weights = np.sqrt(n + 1.0) * f_initial[:-1] * f_initial[1:]
    amplitude = np.sum(
        weights[:, None] * np.exp(-1j * n[:, None] * U_final * time[None, :] / hbar),
        axis=0,
    )
    return np.abs(amplitude) ** 2 / nbar


def plot_assignment_figure1(output: Path, Nmax: int, zcoord: int, mu: float) -> None:
    cases = [
        (1.0, 0.0, 1.0),
        (2.0, 0.0, 1.0),
        (1.0, 10.0, 1.0),
        (1.1, 100.0, 1.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    n = np.arange(Nmax + 1)
    for label, ax, (nbar, U, t_hop) in zip("abcd", axes.flat, cases):
        f, energy = optimize_gutzwiller(nbar, U, t_hop, Nmax, zcoord, mu)
        ax.bar(n, f, color="#3569b7", edgecolor="black", linewidth=0.6)
        ax.set(
            xticks=n,
            xlabel=r"Fock index $n$",
            ylabel=r"Amplitude $f_n$",
            title=fr"({label}) $\langle n\rangle={nbar:g},\ U={U:g},\ t={t_hop:g}$",
        )
        ax.text(0.98, 0.93, fr"$E/M={energy:.4f}$", transform=ax.transAxes,
                ha="right", va="top", fontsize=9)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Optimized single-site Gutzwiller amplitudes", fontsize=15)
    fig.savefig(output / "Figure1_Gutzwiller_amplitudes.png", dpi=240)


def plot_assignment_figure2(output: Path, Nmax: int, zcoord: int, mu: float) -> None:
    ratios = np.linspace(0.0, 20.0, 101)
    coherence = np.empty_like(ratios)
    previous = None
    for i, ratio in enumerate(ratios):
        # t=1 makes U numerically identical to U/t.
        f, _ = optimize_gutzwiller(1.0, ratio, 1.0, Nmax, zcoord, mu,
                                   starts=5, seed=100 + i)
        coherence[i] = first_order_coherence(f, 1.0)
        previous = f
    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    ax.plot(ratios, coherence, color="#b5282f", linewidth=2.5)
    ax.set(xlabel=r"Interaction ratio $U/t$", ylabel=r"$g_1=|\langle a\rangle|^2/\langle n\rangle$",
           title=r"Loss of long-range coherence at $\langle n\rangle=1$")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.25)
    fig.savefig(output / "Figure2_coherence_versus_U_over_t.png", dpi=240)


def plot_collapse_revival(output: Path, Nmax: int, zcoord: int, mu: float) -> None:
    nbar, U_initial, t_initial = 1.0, 0.0, 1.0
    U_final, hbar = 10.0, 1.0
    f_initial, _ = optimize_gutzwiller(
        nbar, U_initial, t_initial, Nmax, zcoord, mu, starts=12
    )

    # Several full revival periods T_rev=2*pi*hbar/U_final.
    time = np.linspace(0.0, 4.0 * 2.0 * np.pi * hbar / U_final, 1000)
    g1 = quenched_coherence(f_initial, nbar, U_final, time, hbar)
    phase_time = U_final * time / hbar

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    n = np.arange(Nmax + 1)
    axes[0].bar(n, f_initial**2, color="#447f55", edgecolor="black", linewidth=0.6)
    axes[0].set(xticks=n, xlabel=r"Fock index $n$", ylabel=r"Probability $|f_n|^2$",
                title=r"Initial superfluid number distribution ($U/t=0$)")
    axes[0].grid(axis="y", alpha=0.22)

    axes[1].plot(phase_time, g1, color="#7b3fa1", linewidth=2.2)
    for k in range(5):
        axes[1].axvline(2 * np.pi * k, color="gray", alpha=0.18, linewidth=1)
    axes[1].set(xlabel=r"Dimensionless time $U_f t/\hbar$", ylabel=r"$g_1(t)$",
                title=r"Collapse and revival after quench to $t_{hop}=0$, $U_f=10$")
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].grid(alpha=0.22)
    fig.savefig(output / "Figure3_4_collapse_and_revival.png", dpi=240)

    np.savetxt(
        output / "collapse_revival_data.csv",
        np.column_stack((time, phase_time, g1)),
        delimiter=",",
        header="time,U_final*time/hbar,g1",
        comments="",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Nmax", type=int, default=8)
    parser.add_argument("--z", type=int, default=2, help="coordination number (2 for 1D)")
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--no-show", action="store_true")
    default_output = Path(__file__).resolve().parent / "results_assignment2"
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    plot_assignment_figure1(args.output, args.Nmax, args.z, args.mu)
    plot_assignment_figure2(args.output, args.Nmax, args.z, args.mu)
    plot_collapse_revival(args.output, args.Nmax, args.z, args.mu)
    print(f"All graphs and data saved in: {args.output.resolve()}")
    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
