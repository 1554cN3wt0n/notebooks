

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
    g = 9.81       # gravity (m/s^2)
    L = 1.0        # pendulum length (m)
    theta0 = np.pi / 2  # initial angle (rad)
    omega0 = 0.0         # initial angular velocity (rad/s)
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
    # Derivatives
    def dtheta_dt(theta, omega):
        return omega

    def domega_dt(theta, omega):
        return - (g / L) * np.sin(theta)
    return domega_dt, dtheta_dt


@app.cell
def _(domega_dt, dtheta_dt):
    def runge_kutta(theta, omega, t, dt):
        # Runge-Kutta 4th order
        for i in range(len(t) - 1):
            k1_theta = dtheta_dt(theta[i], omega[i])
            k1_omega = domega_dt(theta[i], omega[i])
    
            k2_theta = dtheta_dt(theta[i] + 0.5*dt*k1_theta, omega[i] + 0.5*dt*k1_omega)
            k2_omega = domega_dt(theta[i] + 0.5*dt*k1_theta, omega[i] + 0.5*dt*k1_omega)
    
            k3_theta = dtheta_dt(theta[i] + 0.5*dt*k2_theta, omega[i] + 0.5*dt*k2_omega)
            k3_omega = domega_dt(theta[i] + 0.5*dt*k2_theta, omega[i] + 0.5*dt*k2_omega)
    
            k4_theta = dtheta_dt(theta[i] + dt*k3_theta, omega[i] + dt*k3_omega)
            k4_omega = domega_dt(theta[i] + dt*k3_theta, omega[i] + dt*k3_omega)
    
            theta[i+1] = theta[i] + (dt / 6.0) * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
            omega[i+1] = omega[i] + (dt / 6.0) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)
        return theta, omega
    return (runge_kutta,)


@app.cell
def _(dt, omega, runge_kutta, t, theta):
    theta_sol, omega_sol = runge_kutta(theta, omega, t, dt)
    return omega_sol, theta_sol


@app.cell
def _(omega_sol, plt, t, theta_sol):
    # Plotting
    plt.figure(figsize=(12, 5))

    # Angle vs time
    plt.subplot(1, 2, 1)
    plt.plot(t, theta_sol)
    plt.title("Nonlinear Pendulum: θ vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")

    # Phase space
    plt.subplot(1, 2, 2)
    plt.plot(theta_sol, omega_sol)
    plt.title("Phase Space: ω vs θ")
    plt.xlabel("Angle (rad)")
    plt.ylabel("Angular Velocity (rad/s)")

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
