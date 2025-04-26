

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
    # Lattice parameters
    nx, ny = 100, 100          # Grid size
    tau = 0.6                  # Relaxation time
    omega = 1 / tau            # Relaxation parameter
    num_iters = 10000          # Simulation steps
    u_lid = 0.1                # Lid velocity
    return num_iters, nx, ny, omega, u_lid


@app.cell
def _(np, nx, ny):
    # D2Q9 model
    c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]])
    w = np.array([4/9] + [1/9]*4 + [1/36]*4)  # Weights

    # Distribution functions
    f = np.ones((9, nx, ny)) * (1/9)
    feq = np.copy(f)
    rho = np.ones((nx, ny))
    u = np.zeros((2, nx, ny))
    return c, f, feq, u, w


@app.cell
def _(u_lid):
    # Bounce-back for no-slip boundary
    def bounce_back(f):
        f[1,:,0] = f[3,:,0]    # Bottom
        f[1,:,-1] = f[3,:,-1]  # Top
        f[2,0,:] = f[4,0,:]    # Left
        f[2,-1,:] = f[4,-1,:]  # Right

    # Lid velocity on top
    def apply_lid(u):
        u[0,:, -1] = u_lid
        u[1,:, -1] = 0
    return apply_lid, bounce_back


@app.cell
def _(apply_lid, bounce_back, feq, np, num_iters, omega):
    def solve(f, u, c, w):
        for it in range(num_iters):
            # Compute macroscopic variables
            rho = np.sum(f, axis=0)
            u[0] = np.sum(f * c[:, 0].reshape((9, 1, 1)), axis=0) / rho
            u[1] = np.sum(f * c[:, 1].reshape((9, 1, 1)), axis=0) / rho
    
            apply_lid(u)
    
            # Compute equilibrium distribution
            for i in range(9):
                cu = 3 * (c[i, 0] * u[0] + c[i, 1] * u[1])
                feq[i] = w[i] * rho * (1 + cu + 0.5 * cu**2 - 1.5 * (u[0]**2 + u[1]**2))
    
            # Collision step
            f += -omega * (f - feq)
    
            # Streaming step
            for i in range(9):
                f[i] = np.roll(np.roll(f[i], c[i, 0], axis=0), c[i, 1], axis=1)
    
            bounce_back(f)
    
            if it % 1000 == 0:
                print(f"Step {it}")
        return u
    return (solve,)


@app.cell
def _(c, f, solve, u, w):
    u_sol = solve(f, u, c, w)
    return (u_sol,)


@app.cell
def _(plt, u_sol):
    # Plot velocity field
    plt.quiver(u_sol[0].T, u_sol[1].T)
    plt.title("Velocity Field (Lid-Driven Cavity)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().invert_yaxis()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
