import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium")


@app.cell
def _():
    import sympy
    from sympy.abc import alpha, phi, theta, r

    from einsteinpy.symbolic import MetricTensor, ChristoffelSymbols, RiemannCurvatureTensor, RicciTensor, RicciScalar
    from einsteinpy.geodesic import Geodesic
    sympy.init_printing()
    return (
        ChristoffelSymbols,
        Geodesic,
        MetricTensor,
        RicciScalar,
        RicciTensor,
        RiemannCurvatureTensor,
        alpha,
        phi,
        r,
        sympy,
        theta,
    )


@app.cell
def _(MetricTensor, phi, r, sympy, theta):
    g = MetricTensor([[r**2,0],[0,r**2 * sympy.sin(phi)**2]],[phi,theta])
    g.tensor()
    return (g,)


@app.cell
def _(ChristoffelSymbols, g):
    cs = ChristoffelSymbols.from_metric(g)
    cs.tensor()
    return (cs,)


@app.cell
def _(RiemannCurvatureTensor, g):
    rt = RiemannCurvatureTensor.from_metric(g)
    rt.tensor()
    return (rt,)


@app.cell
def _(RicciTensor, g):
    R = RicciTensor.from_metric(g)
    R.tensor()
    return (R,)


@app.cell
def _(RicciScalar, g):
    rs = RicciScalar.from_metric(g)
    rs.tensor()
    return (rs,)


if __name__ == "__main__":
    app.run()
