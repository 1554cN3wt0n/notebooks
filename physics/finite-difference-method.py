

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo
    return marimo, np, plt


@app.cell
def _():
    # Parameters
    alpha = 0.01    # thermal diffusivity
    L = 1.0         # length of the rod
    T = 2.0         # total time
    nx = 50         # number of spatial points
    nt = 500        # number of time steps
    dx = L / (nx - 1)
    dt = T / nt
    return L, alpha, dt, dx, nt, nx


@app.cell
def _(L, alpha, dt, dx, np, nx):
    # Stability condition (important!)
    if alpha * dt / dx**2 > 0.5:
        raise ValueError("Stability condition not met. Reduce dt or increase dx.")

    # Discretized spatial grid
    x = np.linspace(0, L, nx)

    # Initial condition: a bump in the center
    u = np.zeros(nx)
    u[int(nx/2)] = 1.0
    return u, x


@app.cell
def _(u):
    # To store the solution at each time step (optional)
    u_history = [u.copy()]
    return (u_history,)


@app.cell
def _(marimo):
    marimo.md(
        r'''
       \[u_i^{n+1} = u_i^n + \frac{\alpha \Delta t}{\Delta x^2} \left( u_{i+1}^n - 2u_i^n + u_{i-1}^n \right)\]
        '''
    )
    return


@app.cell
def _(alpha, dt, dx, nt, nx):
    def solve(u, u_history):
        # Time stepping loop
        for n in range(nt):
            u_new = u.copy()
            for i in range(1, nx-1):
                u_new[i] = u[i] + alpha * dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
            u = u_new
            u_history.append(u.copy())
        return u
    return (solve,)


@app.cell
def _(solve, u, u_history):
    u_solve = solve(u, u_history)
    return


@app.cell
def _(dt, nt, plt, u_history, x):
    # Plotting
    plt.figure(figsize=(10, 6))
    times_to_plot = [0, int(nt/4), int(nt/2), int(3*nt/4), nt]
    for idx in times_to_plot:
        plt.plot(x, u_history[idx], label=f't = {idx*dt:.2f}s')

    plt.title('Heat Equation Solution using Finite Difference Method')
    plt.xlabel('Position along the rod [x]')
    plt.ylabel('Temperature [u]')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return


if __name__ == "__main__":
    app.run()
