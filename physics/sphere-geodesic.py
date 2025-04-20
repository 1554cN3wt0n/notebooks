

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sympy as sp
    return (sp,)


@app.cell
def _(sp):
    # Define symbols and functions
    R, tau = sp.symbols('R tau')
    theta = sp.Function('theta')(tau)
    phi = sp.Function('phi')(tau)
    coords = [theta, phi]
    return R, phi, tau, theta


@app.cell
def _(R, sp, theta):
    # Define the metric tensor on the 2D sphere
    g = sp.Matrix([
        [R**2,                     0],
        [0,      R**2 * sp.sin(theta)**2]
    ])

    # Compute the inverse metric
    g_inv = g.inv()
    return g, g_inv


@app.cell
def _(g, g_inv, phi, sp, theta):
    # Prepare Christoffel symbols
    n = 2
    Gamma = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]

    # Use dummy symbols to take derivatives
    theta_sym, phi_sym = sp.symbols('theta_sym phi_sym')
    coord_syms = [theta_sym, phi_sym]
    g_sym = g.subs({theta: theta_sym, phi: phi_sym})

    # Calculate Christoffel symbols
    for _l in range(n):
        for _mu in range(n):
            for _nu in range(n):
                total = 0
                for sigma in range(n):
                    total += g_inv[_l, sigma] * (
                        sp.diff(g_sym[sigma, _mu], coord_syms[_nu]) +
                        sp.diff(g_sym[sigma, _nu], coord_syms[_mu]) -
                        sp.diff(g_sym[_mu, _nu], coord_syms[sigma])
                    )
                Gamma[_l][_mu][_nu] = sp.simplify(0.5 * total.subs({theta_sym: theta, phi_sym: phi}))

    return Gamma, coord_syms, n


@app.cell
def _(phi, sp, tau, theta):
    # Define derivatives of coordinates
    dtheta = sp.diff(theta, tau)
    dphi = sp.diff(phi, tau)
    ddtheta = sp.diff(dtheta, tau)
    ddphi = sp.diff(dphi, tau)
    dcoords = [dtheta, dphi]
    ddcoords = [ddtheta, ddphi]
    return dcoords, ddcoords


@app.cell
def _(Gamma, coord_syms, dcoords, ddcoords, n, phi, sp, theta):
    # Construct geodesic equations
    geodesic_eqs = []
    for _l in range(n):
        eq = ddcoords[_l]
        for _mu in range(n):
            for _nu in range(n):
                # Substitute symbolic coords with actual functions
                Gamma_subs = Gamma[_l][_mu][_nu].subs({
                    coord_syms[0]: theta,
                    coord_syms[1]: phi
                })
                eq += Gamma_subs * dcoords[_mu] * dcoords[_nu]
        geodesic_eqs.append(sp.simplify(eq))
    return (geodesic_eqs,)


@app.cell
def _(geodesic_eqs, sp):
    # Output geodesic equations
    for _i, _eq in enumerate(geodesic_eqs):
        print(f"\nGeodesic equation for {['theta', 'phi'][_i]}(τ):")
        sp.pprint(sp.Eq(_eq, 0))

    return


if __name__ == "__main__":
    app.run()
