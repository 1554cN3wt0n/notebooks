import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import sph_harm, genlaguerre, factorial
    return factorial, genlaguerre, np, plt, sph_harm


@app.cell
def _(np):
    # Constants
    a0 = 1.0  # Bohr radius in atomic units

    # Grid in Cartesian space
    N = 300
    x = np.linspace(-20, 20, N)
    z = np.linspace(-20, 20, N)
    X, Z = np.meshgrid(x, z)
    Y = np.zeros_like(X)  # cross section at y = 0
    return X, Y, Z, a0, x, z


@app.cell
def _(X, Y, Z, a0, factorial, genlaguerre, np, sph_harm):
    # Convert to spherical coordinates
    R = np.sqrt(X**2 + Y**2 + Z**2)
    Theta = np.arccos(np.divide(Z, R, where=R != 0))
    Phi = np.arctan2(Y, X)

    # Hydrogen wavefunction: n, l, m
    n, l, m = 3, 2, 0

    # Radial part
    def R_nl(n, l, r):
        rho = 2 * r / (n * a0)
        norm = np.sqrt((2.0 / (n * a0))**3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
        L = genlaguerre(n - l - 1, 2 * l + 1)(rho)
        return norm * np.exp(-rho / 2) * rho**l * L

    # Angular part: complex spherical harmonics
    Y_lm = sph_harm(m, l, Phi, Theta)

    # Total wavefunction
    Psi = R_nl(n, l, R) * Y_lm
    P = np.abs(Psi)**2

    # Normalize and mask large r
    P[R > 20] = 0
    return P, l, m, n


@app.cell
def _(P, l, m, n, plt, x, z):
    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(P, extent=[x.min(), x.max(), z.min(), z.max()], origin='lower', cmap='inferno')
    plt.colorbar(label='Probability Density')
    plt.title(f'Cross Section of Hydrogen Wavefunction |ψₙₗₘ|² (n={n}, ℓ={l}, m={m})\nin the xz-plane (y=0)')
    plt.xlabel('x (Bohr radii)')
    plt.ylabel('z (Bohr radii)')
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
