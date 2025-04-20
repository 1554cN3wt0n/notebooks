import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium")


@app.cell
def _():
    import sympy as sp
    return (sp,)


@app.cell
def _(sp):
    # Define coordinates
    t, x, y, z = sp.symbols('t x y z')
    coords = [t, x, y, z]
    return coords, t, x, y, z


@app.cell
def _(sp):
    # Define the metric tensor g_{mu, nu}
    # Example: Minkowski metric (signature: -+++)
    g = sp.Matrix([
        [-1,  0,  0,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  1,  0],
        [ 0,  0,  0,  1]
    ])

    # Compute the inverse metric g^{mu, nu}
    g_inv = g.inv()
    return g, g_inv


@app.cell
def _(coords, g, g_inv, sp):
    # Compute the Christoffel symbols
    n = len(coords)
    Gamma = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]

    for _l in range(n):
        for _mu in range(n):
            for _nu in range(n):
                term = 0
                for sigma in range(n):
                    term += g_inv[_l, sigma] * (
                        sp.diff(g[sigma, _mu], coords[_nu]) +
                        sp.diff(g[sigma, _nu], coords[_mu]) -
                        sp.diff(g[_mu, _nu], coords[sigma])
                    )
                Gamma[_l][_mu][_nu] = sp.simplify(0.5 * term)
    return Gamma, n, sigma, term


@app.cell
def _(n, sp):
    # Define the velocity vector dx^mu/dτ
    tau = sp.symbols('tau')
    x_tau = [sp.Function(f'x{i}')(tau) for i in range(n)]  # x^mu(τ)
    dx_tau = [sp.diff(xi, tau) for xi in x_tau]
    ddx_tau = [sp.diff(dxi, tau) for dxi in dx_tau]
    return ddx_tau, dx_tau, tau, x_tau


@app.cell
def _(Gamma, coords, ddx_tau, dx_tau, n, sp, x_tau):
    # Construct the geodesic equations
    geodesic_eqs = []
    for _l in range(n):
        eq = ddx_tau[_l]
        for _mu in range(n):
            for _nu in range(n):
                eq += Gamma[_l][_mu][_nu].subs(
                    [(coords[i], x_tau[i]) for i in range(n)]
                ) * dx_tau[_mu] * dx_tau[_nu]
        geodesic_eqs.append(sp.simplify(eq))
    return eq, geodesic_eqs


@app.cell
def _(geodesic_eqs, sp):
    for _i, _eq in enumerate(geodesic_eqs):
        print(f"Geodesic equation for x^{_i}:")
        sp.pprint(sp.Eq(_eq, 0))
        print()
    return


if __name__ == "__main__":
    app.run()
