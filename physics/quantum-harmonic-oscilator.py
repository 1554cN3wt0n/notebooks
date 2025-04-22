

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import hermite, factorial
    return factorial, hermite, np, plt


@app.cell
def _():
    # Constants (in atomic units where ħ = 1, m = 1)
    hbar = 1.0
    m = 1.0
    omega = 1.0
    n_levels = 6  # Number of energy levels to show
    return hbar, m, n_levels, omega


@app.cell
def _(hbar, m, n_levels, np, omega):
    # x range for potential plot
    x = np.linspace(-4, 4, 1000)
    V = 0.5 * m * omega**2 * x**2  # Harmonic oscillator potential

    # Energy levels
    n = np.arange(n_levels)
    E_n = hbar * omega * (n + 0.5)
    return E_n, V, n, x


@app.cell
def _(factorial, hbar, hermite, m, np, omega):
    # Prefactors for wavefunctions
    alpha = np.sqrt(m * omega / hbar)
    norm = lambda n: 1.0 / np.sqrt((2**n) * factorial(n)) * (alpha / np.pi)**0.25

    def psi_n(n, x):
        Hn = hermite(n)
        return norm(n) * np.exp(-0.5 * (alpha * x)**2) * Hn(alpha * x)
    return (psi_n,)


@app.cell
def _(E_n, V, n, np, plt, psi_n, x):
    # Plot setup
    plt.figure(figsize=(10, 6))
    plt.plot(x, V, color='black', label='Potential $V(x)$')

    # Plot energy levels and probability densities
    for i in n:
        E = E_n[i]
        psi_sq = psi_n(i, x)**2
        # Normalize and shift the probability density to align with energy level
        scale = 0.5 / np.max(psi_sq)  # scale for visual clarity
        plt.plot(x, scale * psi_sq + E, label=f'$|\\psi_{i}(x)|^2 + E_{i}$')

        # Energy level line
        plt.hlines(E, -4, 4, color='gray', linestyle='--', linewidth=0.5)
        plt.text(4.2, E, f'$E_{i}$', va='center')

    # Final plot decorations
    plt.title("Quantum Harmonic Oscillator: Energy Levels and Probability Densities")
    plt.xlabel("$x$")
    plt.ylabel("Energy / $|\\psi_n(x)|^2$")
    plt.ylim(0, E_n[-1] + 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
