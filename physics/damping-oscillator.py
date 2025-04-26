

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
    def damped_oscillator_position_and_state_space(time, mass, damping_coefficient, spring_constant, initial_position, initial_velocity):
      omega_n = np.sqrt(spring_constant / mass)  # Natural frequency
      zeta = damping_coefficient / (2 * np.sqrt(mass * spring_constant)) # Damping ratio

      position = np.zeros_like(time)
      velocity = np.zeros_like(time)
      state_space = np.zeros((len(time), 2))
      position[0] = initial_position
      velocity[0] = initial_velocity
      state_space[0, :] = [initial_position, initial_velocity]

      dt = time[1] - time[0] # Time step (assuming uniform time array)

      for i in range(len(time) - 1):
        # Equations of motion in state space form:
        # dx/dt = v
        # dv/dt = -(k/m)x - (b/m)v

        acceleration = -(spring_constant / mass) * position[i] - (damping_coefficient / mass) * velocity[i]

        # Using Euler's method for integration (simple but can be less accurate for larger dt)
        velocity[i+1] = velocity[i] + acceleration * dt
        position[i+1] = position[i] + velocity[i] * dt
        state_space[i+1, :] = [position[i+1], velocity[i+1]]

      return position, velocity, state_space
    return (damped_oscillator_position_and_state_space,)


@app.cell
def _(damped_oscillator_position_and_state_space, np, plt):
    # --- Simulation Parameters ---
    mass = 1.0          # kg
    damping_coefficient = 0.5 # Ns/m
    spring_constant = 4.0   # N/m
    initial_position = 1.0  # m
    initial_velocity = 0.0  # m/s
    time = np.linspace(0, 10, 500) # Time array from 0 to 10 seconds with 500 points

    # --- Calculate Position, Velocity, and State Space ---
    position, velocity, state_space = damped_oscillator_position_and_state_space(
        time, mass, damping_coefficient, spring_constant, initial_position, initial_velocity
    )

    # --- Plotting ---
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot Position vs Time
    axs[0].plot(time, position, label='Position (x)')
    axs[0].plot(time, velocity, label='Velocity (v)', linestyle='--')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Position (m) / Velocity (m/s)')
    axs[0].set_title('Position and Velocity of a Damped Harmonic Oscillator')
    axs[0].grid(True)
    axs[0].legend()

    # Plot State Space (Velocity vs Position)
    axs[1].plot(state_space[:, 0], state_space[:, 1], label='State Space Trajectory')
    axs[1].scatter(initial_position, initial_velocity, color='red', marker='o', label='Initial State')
    axs[1].set_xlabel('Position (m)')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].set_title('State Space Representation')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
