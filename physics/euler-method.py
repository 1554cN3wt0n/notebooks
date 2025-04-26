

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    return np, plt


@app.cell
def _(np):
    # Physical constants
    g = 9.81
    L = 1.0
    theta0 = np.pi / 2     # initial angle (radians)
    omega0 = 0.0           # initial angular velocity
    return L, g, omega0, theta0


@app.cell
def _(np, omega0, theta0):
    # Time setup
    dt = 0.01
    t_max = 10
    t = np.arange(0, t_max, dt)

    # Initialize arrays
    theta = np.zeros(len(t))
    omega = np.zeros(len(t))
    theta[0] = theta0
    omega[0] = omega0
    return dt, omega, t, theta


@app.cell
def _(L, g, np):
    def euler_method(theta, omega, t, dt):
        # Euler integration
        for i in range(len(t) - 1):
            omega[i+1] = omega[i] - (g / L) * np.sin(theta[i]) * dt
            theta[i+1] = theta[i] + omega[i] * dt
        return theta, omega
    return (euler_method,)


@app.cell
def _(dt, euler_method, omega, t, theta):
    theta_sol, omega_sol = euler_method(theta, omega, t, dt)
    return (theta_sol,)


@app.cell
def _(plt, t, theta_sol):
    # Plot angle over time
    plt.figure(figsize=(10, 4))
    plt.plot(t, theta_sol, label='θ(t)')
    plt.title("Simple Pendulum Using Euler Method")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.grid(True)
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
