import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sympy.abc import t,m,g,k
    from sympy import diff,symbols,Function,Derivative,dsolve
    return Derivative, Function, diff, dsolve, g, k, m, np, plt, symbols, t


@app.cell
def _(Function, g, m, symbols, t):
    # u = dx/dt, x, t
    x = symbols('x',cls=Function)
    u = symbols('u',cls=Function)
    K = m * u(t) ** 2 / 2 # Kinetic Energy
    V = m * g * x(t)      # Potential Energy
    L = K - V
    return K, L, V, u, x


@app.cell
def _(L, diff, t, u, x):
    # d/dt(dL/du) - dL/dx = 0
    # eq = d/dt(dL/du) - dL/dx = 0
    eq = diff(diff(L,u(t)),t) - diff(L,x(t))
    eq
    return (eq,)


@app.cell
def _(diff, eq, t, u, x):
    f = eq.replace(u(t),diff(x(t),t))
    f
    return (f,)


@app.cell
def _(dsolve, f, t, x):
    dsolve(f,x(t))
    return


if __name__ == "__main__":
    app.run()
