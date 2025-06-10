import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import scipy.sparse as sp
    from numpy.polynomial.polynomial import Polynomial as Poly

    from scipy.optimize import minimize, LinearConstraint, Bounds
    import scipy.optimize as opt

    import matplotlib.pyplot as plt
    return Poly, mo, np


@app.cell
def _(np):
    # Definition of the configuration

    r_int: float = 1.00
    r_ext: float = 1.01

    sigY: float = 360*10**4
    E: float = 210000*10**4
    nu: float = 0.3
    pression1: float = 2*10**4
    rho: float = 7.8*10**-2
    #
    S = np.zeros((2, 2), dtype=float)
    Ebar = E/(1.-nu**2)
    nubar = nu/(1.-nu)
    S[1, 1] = S[0, 0] = 1./Ebar
    S[1, 0] = S[0, 1] = -nubar/Ebar

    # space discretization
    nel = 1
    degU = 3  # u_r, sigma_r, degree per element
    degS = degU
    degL = degS + 1  # lambda, nb unknwon per element

    # loading sequence
    nt_part1 = 2
    nt_part2 = 2
    nt = nt_part1 + nt_part2 - 1
    # free time (not init)
    ntf = nt - 1
    temps_part1 = np.linspace(0, 0.0001, nt_part1)
    temps_part2 = np.linspace(0.0001, 0.0002, nt_part2)
    temps = np.concatenate((temps_part1, temps_part2[1:])).T

    p_int = np.zeros(nt, dtype=float)
    p_int[0:nt_part1] = np.linspace(0., pression1, nt_part1)
    p_int[nt_part1-1:] = np.linspace(pression1, 2*pression1, nt_part2)

    # p_ext = np.zeros(nt, dtype=float)
    # p_ext[0:nt_part1] = pression1  # np.linspace(0., pression1, nt_part1)
    # p_ext[nt_part1:] = pression1

    p_ext = p_int

    return degL, degS, degU, rho


@app.cell
def _(mo):
    mo.md(
        r"""
    SBEN functional:
    $$\begin{aligned}\Pi(u,\sigma,\lambda)&=\iint \sigma_Y\dot{\lambda} + \sigma:S:\dot{\sigma} - \dot\epsilon:\sigma \mathrm{\,dx\,dt}\\
    &=\iint \sigma_Y\dot{\lambda} \mathrm{\,dx\,dt}+ \frac{1}{2} \int \left[\sigma:S:\sigma\right]_0^T\mathrm{\,dx}- \int \left[\sigma\cdot n\cdot \dot{u}\right]_{r_i}^{r_e}\mathrm{\,dt} \\
    &=\iint \sigma_Y\dot{\lambda} \mathrm{\,dx\,dt} + \frac{1}{2} \int \left[\sigma:S:\sigma\right]_0^T\mathrm{\,dx} + \int \left(p_e(t)\dot{u}(r_e,t)-p_i(t)\dot{u}(r_i,t)\right)\mathrm{\,dt}
    \end{aligned}$$
    For cylindrical coordinates $\mathrm{dx}=2\pi r \mathrm{dr}$.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Constraints (Tresca): 

    $$\dot{\lambda}\geqslant 0$$

    $$\begin{pmatrix}-1 \\ 1\end{pmatrix}{\dot{\lambda}} = \dot{\hat{\epsilon}} - S\dot{\hat{\sigma}}$$

    $$ -\sigma_Y\leqslant \sigma_r -\sigma_\theta \leqslant \sigma_Y$$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Let $\phi_\sigma$ and $\phi_u$ be the Lagrange shape functions for $\sigma_r$ and $u_r$, thanks to equilibrium, we have:

    $$\hat{\sigma}_E = \begin{pmatrix}\sigma_r\\\sigma_\theta\end{pmatrix}=\underbrace{\begin{pmatrix}\phi_\sigma \\ (r\phi_\sigma)')\end{pmatrix}}_{\mathbf{A}_{s,E}}\boldsymbol{\sigma}_E+\underbrace{\begin{pmatrix}0 \\ -r\rho\phi_u\end{pmatrix}}_{\mathbf{A}_{u,E}}\mathbf{\ddot{u}}_E $$

    Regarding strains, we have (we multiply by $r$ to have polynomials): 

    $$ r\hat{\epsilon} = \begin{pmatrix}r\epsilon_r\\ r\epsilon_\theta\end{pmatrix}=\underbrace{\begin{pmatrix} r\phi_u' \\ \phi_u \end{pmatrix}}_{\mathbf{B}_u}\mathbf{u}$$

    Constraints are considered at Gauss points.
    """
    )
    return


@app.cell
def _(Poly, degL, degS, degU, np, rho):
    # Gauss Integration
    rg, wg = np.polynomial.legendre.leggauss(degL)

    def refLagrangeShapes(deg: int = 3):
        xi = np.linspace(-1., 1., deg+1)
        shapes = []
        deriv = []
        for i in range(deg+1):
            roots = np.delete(xi, i)
            shape = Poly.fromroots(roots)
            shapes.append(shape / shape(xi[i]))
            deriv.append(Poly.deriv(shapes[-1]))
        return np.reshape(np.asarray(shapes), (1, deg+1)), np.reshape(np.asarray(deriv), (1, deg+1))

    def polytimesr(x): return x * Poly.basis(1)
    def polytimesr2(x): return x * Poly.basis(2)
    arr_polytimesr = np.vectorize(polytimesr)
    arr_polytimesr2 = np.vectorize(polytimesr2)
    apply_vectorized = np.vectorize(lambda f, x: f(x))

    phiU, dphiU = refLagrangeShapes(degU)
    phiS, dphiS = refLagrangeShapes(degS)

    As = np.vstack([phiS,phiS+arr_polytimesr(dphiS)])
    Au = np.vstack([np.zeros_like(phiU),-rho*phiS+arr_polytimesr(dphiU)])
    Bu = np.vstack([arr_polytimesr(dphiU),phiU])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Computation of SBEN
    We assume all initial conditions are 0.

    The quadratic term is: $\pi \int\sigma(r,T):S:\sigma(r,T)r\mathrm{\,dr}$
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
