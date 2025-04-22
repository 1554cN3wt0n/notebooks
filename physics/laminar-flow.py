

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    return np, plt


@app.cell
def _():
    # Parameters
    height = 1.0            # channel height (m)
    length = 5.0            # channel length (m)
    ny = 100                # number of vertical points
    nx = 300                # number of horizontal points
    mu = 1e-3               # dynamic viscosity (Pa·s)
    dpdx = -10              # pressure gradient (Pa/m)
    return dpdx, height, length, mu, nx, ny


@app.cell
def _(height, length, np, nx, ny):
    # Grid
    y = np.linspace(0, height, ny)
    x = np.linspace(0, length, nx)
    Y, X = np.meshgrid(y, x)
    return X, Y, y


@app.cell
def _(dpdx, height, mu, np, nx, y):
    # Analytical solution for steady Poiseuille flow: u(y) = (1/(2*mu)) * dp/dx * (y(h - y))
    u_max = -(dpdx) * height**2 / (8 * mu)
    u = (1 / (2 * mu)) * dpdx * (y * (height - y))  # 1D profile

    # Broadcast u to 2D (same profile along x)
    U = np.tile(u, (nx, 1))
    return (U,)


@app.cell
def _(U, X, Y, plt):
    # Plot
    plt.figure(figsize=(10, 4))
    cp = plt.contourf(X, Y, U, levels=50, cmap='viridis')
    plt.colorbar(cp, label='Velocity (m/s)')
    plt.title("Laminar Poiseuille Flow in a 2D Channel")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
