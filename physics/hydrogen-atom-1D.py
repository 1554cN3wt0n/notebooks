

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import genlaguerre, factorial
    return factorial, genlaguerre, np, plt


@app.cell
def _(np):
    # Constants
    rydberg_energy = 13.6  # eV
    n_max = 10  # number of energy levels to show

    # Energy levels
    n_values = np.arange(1, n_max + 1)
    E_n = -rydberg_energy / n_values**2
    return E_n, n_values, rydberg_energy


@app.cell
def _(E_n, n_values, plt, rydberg_energy):
    # Plot
    plt.figure(figsize=(6, 8))
    for i, E in enumerate(E_n):
        plt.hlines(E, xmin=0, xmax=1, color='blue')
        plt.text(1.05, E, f'n={n_values[i]}', va='center', fontsize=10)

    # Ionization level (E = 0)
    plt.hlines(0, 0, 1, color='red', linestyle='--', linewidth=1)
    plt.text(1.05, 0, 'Ionization\nlimit', va='bottom', color='red')

    plt.title("Hydrogen Atom Energy Levels")
    plt.ylabel("Energy (eV)")
    plt.xticks([])
    plt.ylim(-rydberg_energy - 1, 1)
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(factorial, genlaguerre, np):
    # Constants (atomic units: ħ = 1, m_e = 1, e = 1)
    a0 = 1.0  # Bohr radius in atomic units

    # Radial part R_{n,l}(r) of the hydrogen wavefunction (atomic units)
    def R_nl(n, l, r):
        rho = 2 * r / (n * a0)
        norm = np.sqrt((2.0 / (n * a0))**3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
        L = genlaguerre(n - l - 1, 2 * l + 1)(rho)
        return norm * np.exp(-rho / 2) * rho**l * L

    # Radial probability density: P(r) = r^2 * |R_{n,l}(r)|^2
    def P_r(n, l, r):
        R = R_nl(n, l, r)
        return r**2 * np.abs(R)**2
    return (P_r,)


@app.cell
def _(P_r, np, plt):
    # Plot radial probability densities
    r = np.linspace(0, 20, 1000)

    levels = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]  # (n, l)

    plt.figure(figsize=(10, 6))
    for n, l in levels:
        plt.plot(r, P_r(n, l, r), label=f'n={n}, ℓ={l}')

    plt.title("Hydrogen Atom Radial Probability Densities")
    plt.xlabel("r (Bohr radius)")
    plt.ylabel(r"$P(r) = r^2 |R_{n\ell}(r)|^2$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
