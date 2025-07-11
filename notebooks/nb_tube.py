import marimo

__generated_with = "0.14.9"
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

    apply_vectorized = np.vectorize(lambda f, x: f(x))

    return LinearConstraint, Poly, apply_vectorized, minimize, mo, np, plt


@app.cell
def _(np):
    # Definition of the configuration
    class geometry:
        def __init__(self, ri: float = 100., re: float = 101.):
            self.ri: float = ri
            self.re: float = re

        def discretize(self, di):
            # coordinates of dofs
            self.coor_bord_el = np.linspace(self.ri, self.re, di.nel+1)
            self.size_el = np.diff(self.coor_bord_el)
            self.coorL = np.zeros(di.nbL)
            for el0 in range(di.nel):
                self.coorL[di.indL[el0]] = self.coor_bord_el[el0] + \
                    di.rg_01*self.size_el[el0]
            self.coorU = np.linspace(self.ri, self.re, di.nbU)
            self.coorS = np.linspace(self.ri, self.re, di.nbS)

    class material:
        def __init__(self, sY: float = 360, E: float = 210000, nu: float = 0.3, rho: float = 7.8e-9):
            self.sigY = sY
            self.E = E
            self.nu = nu
            self.rho = rho
            self.S = np.zeros((2, 2), dtype=float)
            Ebar = E/(1.-nu**2)
            nubar = nu/(1.-nu)
            self.S[1, 1] = self.S[0, 0] = 1./Ebar
            self.S[1, 0] = self.S[0, 1] = -nubar/Ebar

    class discretization:
        # indexing for continuous Lagrange
        def locToGloLag(self, ne: int, deg: int):
            ind = np.empty((ne, deg+1), dtype=int)
            for i in range(ne):
                ind[i, :] = np.arange(i*deg, (i+1)*deg+1)
            return ind

        # indexing for discontinuous Lagrange
        def locToGloDG(self, ne: int, nl: int):
            ind = np.empty((ne, nl), dtype=int)
            for i in range(ne):
                ind[i, :] = np.arange(i*nl, (i+1)*nl)
            return ind

        def __init__(self, nel: int = 1, degU: int = 3, degS: int = 3, degL: int = 4, nt: int = 4):
            assert (nt > 3)
            self.nel = nel
            self.degU = degU
            self.degS = degS
            self.degL = degL
            self.nt = nt  # including initial time step (0)
            self.ntf = nt-1  # free time steps
            #
            self.nbS = nel*(degS+1) - (nel-1)
            self.nbU = nel*(degU+1) - (nel-1)
            self.nbL = nel * degL
            # f is for free (only sigma has bcs)
            self.nbSf = nel*(degS+1) - (nel-1) - 2
            self.nbUf = self.nbU
            #  Total
            self.nbT = self.nbU + self.nbS + self.nbL
            self.nbTf = self.nbUf + self.nbSf + self.nbL

            self.indU = self.locToGloLag(nel, degU)
            self.indS = self.locToGloLag(nel, degS)
            self.indL = self.locToGloDG(nel, degL)

            # Gauss Integration
            # quadrature on [-1,1]
            self.rg, self.wg = np.polynomial.legendre.leggauss(degL)
            # quadrature on [0,1]
            self.rg_01 = (self.rg+1)/2
            self.wg_01 = self.wg/2

            def iLeftS(tt): return self.gIndS(tt)[0]
            def iRightS(tt): return self.gIndS(tt)[-1]
            self.ibcs = np.array([f(tt) for tt in range(self.ntf)
                                  for f in (iLeftS, iRightS)], dtype=int)
            self.iNbcs = np.setdiff1d(np.arange(self.nbT*self.ntf), self.ibcs)

        # global space-time numbering bcs not removed but ics removed
        def gIndU(self, tt):
            return tt*self.nbT+np.arange(self.nbU)

        def gIndS(self, tt):
            return tt*self.nbT+self.nbU+np.arange(self.nbS)

        def gIndL(self, tt):
            return tt*self.nbT+self.nbU+self.nbS+np.arange(self.nbL)
        # global space-time numbering bcs and ics removed

        def gIndUf(self, tt):
            return tt*self.nbTf+np.arange(self.nbUf)

        def gIndSf(self, tt):
            return tt*self.nbTf+self.nbUf+np.arange(self.nbSf)

        def gIndLf(self, tt):
            return tt*self.nbTf+self.nbUf+self.nbSf+np.arange(self.nbL)
        # element contributions with bcs kept

        def gIndUe(self, tt, el):
            return tt*self.nbT+self.indU[el]

        def gIndSe(self, tt, el):
            return tt*self.nbT+self.nbU+self.indS[el]

        def gIndLe(self, tt, el):
            return tt*self.nbT+self.nbU+self.nbS+self.indL[el]

    # The loading definition is a bit crude

    class loading:
        def __init__(self, T: float = .0002, pmax: float = 2., form: str = "ramp"):
            self.T = T
            self.form = form
            self.pmax = pmax
            self.ini_u = lambda r: 0.
            self.ini_v = lambda r: 0.
            self.ini_a = lambda r: 0.
            self.ini_s = lambda r: 0.
            self.ini_l = lambda r: 0.

        def pint(self, t: float) -> float:
            if self.form == "ramp" or self.form == "symramp":
                if 0 <= t <= self.T:
                    return self.pmax*t/self.T
                elif t < 0:
                    return 0.
                else:
                    return np.nan
            elif self.form in ("haltedramp","symhaltedramp"):
                if t < 0:
                    return 0.
                elif t < self.T/2:
                    return 2*self.pmax*t/self.T
                else:
                    return self.pmax
            else:
                return np.nan

        def pext(self, t: float) -> float:
            if self.form in ("ramp","haltedramp"):
                return 0.
            elif self.form == "symramp" or self.form == "symhaltedramp":
                return self.pint(t)
            else:
                return np.nan

        def discretize(self, di: discretization, ge: geometry):
            self.temps = np.linspace(0., self.T, di.nt)
            self.p_int = np.vectorize(self.pint)(self.temps)
            self.p_ext = np.vectorize(self.pext)(self.temps)
            self.dt = self.T/di.ntf
            self.u_ini = np.vectorize(self.ini_u)(ge.coorU)
            self.v_ini = np.vectorize(self.ini_v)(ge.coorU)
            self.a_ini = np.vectorize(self.ini_a)(ge.coorU)
            self.s_ini = np.vectorize(self.ini_s)(ge.coorS)
            self.l_ini = np.vectorize(self.ini_l)(ge.coorL)
            assert (np.isclose(self.p_int[0], - self.s_ini[0]))
            assert (np.isclose(self.p_ext[0], - self.s_ini[-1]))

            def vLeftS(tt): return -self.p_int[tt+1]
            def vRightS(tt): return -self.p_ext[tt+1]
            self.vbcs = np.array([f(tt) for tt in range(di.ntf)
                                  for f in (vLeftS, vRightS)], dtype=float, ndmin=2).T

    return discretization, geometry, loading, material


@app.cell
def _(mo):
    mo.md(
        r"""
    SBEN functional:
    $$\begin{aligned}\Pi(u,\sigma,\lambda)&=\iint \sigma_Y\dot{\lambda} + \sigma:S:\dot{\sigma} - \dot\epsilon:\sigma \mathrm{\,dx\,dt}\\
    &=\iint \sigma_Y\dot{\lambda} \mathrm{\,dx\,dt}+ \frac{1}{2} \int \left[\sigma:S:\sigma\right]_0^T\mathrm{\,dx}- 2\pi \int \left[r\sigma\cdot n\cdot \dot{u}\right]_{r_i}^{r_e}\mathrm{\,dt} +\iint \rho \ddot{u}\dot{u} \mathrm{\,dx\, dt} \\
    &=\int \sigma_Y\left[\lambda\right]_0^T+ \frac{1}{2}  \left[\sigma:S:\sigma+\rho \dot{u} ^2 \right]_0^T\mathrm{\,dx} + 2\pi \int \left(r_e p_e(t)\dot{u}(r_e,t)-r_i p_i(t)\dot{u}(r_i,t)\right)\mathrm{\,dt}
    \end{aligned}$$
    For cylindrical coordinates $\mathrm{dx}=2\pi r \mathrm{dr}$.
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
def _(Poly, apply_vectorized, np):
    class spacemat:
        # Lagrange shape functions
        def lagrangeShapes(self, a: float = 0., b: float = 1., deg: int = 3):
            xi = np.linspace(a, b, deg+1)
            shapes = []
            deriv = []
            for i in range(deg+1):
                roots = np.delete(xi, i)
                shape = Poly.fromroots(roots)
                shapes.append(shape / shape(xi[i]))
                deriv.append(Poly.deriv(shapes[-1]))
            return np.reshape(np.asarray(shapes), (1, deg+1)), np.reshape(np.asarray(deriv), (1, deg+1))

        def __init__(self, di, ge, mat):
            self.Lambda = np.zeros((di.nbL), dtype=float)
            self.matFss = np.zeros((di.nbS, di.nbS), dtype=float)
            self.matFsu = np.zeros((di.nbS, di.nbU), dtype=float)
            self.matFuu = np.zeros((di.nbU, di.nbU), dtype=float)
            self.matFus = np.zeros((di.nbU, di.nbS), dtype=float)
            self.matMuu = np.zeros((di.nbU, di.nbU), dtype=float)

            phiSref, dphiSref = self.lagrangeShapes(-1., 1., di.degS)
            phiUref, dphiUref = self.lagrangeShapes(-1., 1., di.degU)

            for el in range(di.nel):

                self.Lambda[di.indL[el]] = mat.sigY * 2 * np.pi * \
                    ge.size_el[el] * ge.coorL[di.indL[el]]*di.wg_01

                # sides of the element
                rl = ge.coor_bord_el[el]
                rr = ge.coor_bord_el[el+1]
                h = rr-rl

                arr_polytimesR = np.vectorize(
                    lambda P: P * (Poly([1, 1])*h/2+rl))

                Asref = np.vstack(
                    (phiSref, phiSref+arr_polytimesR(dphiSref)*2/h))
                Auref = np.vstack(
                    (np.zeros_like(phiUref), -mat.rho*arr_polytimesR(phiUref)))

                def my_integRef(P): return Poly.integ(
                    h * np.pi * P * (Poly([1, 1])*h/2+rl), lbnd=-1)
                arr_integRef = np.vectorize(my_integRef)

                M1 = np.dot(phiUref.T, phiUref)*mat.rho
                M1i = arr_integRef(M1)
                self.matMuu[np.ix_(di.indU[el, :], di.indU[el, :])
                            ] += apply_vectorized(M1i, 1)

                F1 = np.dot(Asref.T, np.dot(mat.S, Asref))
                F1i = arr_integRef(F1)
                F2 = np.dot(Asref.T, np.dot(mat.S, Auref))
                F2i = arr_integRef(F2)
                F3 = np.dot(Auref.T, np.dot(mat.S, Auref))
                F3i = arr_integRef(F3)
                F4 = np.dot(Auref.T, np.dot(mat.S, Asref))
                F4i = arr_integRef(F4)

                self.matFss[np.ix_(di.indS[el, :], di.indS[el, :])
                            ] += apply_vectorized(F1i, 1)

                self.matFsu[np.ix_(di.indS[el, :], di.indU[el, :])
                            ] += apply_vectorized(F2i, 1)

                self.matFuu[np.ix_(di.indU[el, :], di.indU[el, :])
                            ] += apply_vectorized(F3i, 1)

                self.matFus[np.ix_(di.indU[el, :], di.indS[el, :])
                            ] += apply_vectorized(F4i, 1)

    return (spacemat,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Computation of SBEN

    The space-time unknown is $\mathbf{x}=\begin{pmatrix} \begin{pmatrix} \mathbf{u} \\ \boldsymbol{\sigma} \\ \boldsymbol{\lambda} \end{pmatrix}_{t=1} \\ \vdots \\ \begin{pmatrix} \mathbf{u} \\ \boldsymbol{\sigma} \\ \boldsymbol{\lambda} \end{pmatrix}_{t=N} \end{pmatrix}$

    The sBEN functional can be written as: $\Pi(\mathbf{x})=\mathbf{x}^T\mathbf{Q}\mathbf{x}+\mathbf{x}^T\mathbf{L}+c$.
    ### Space integration
    #### Quadratic term:
    $\mathbf{x}^T\mathbf{Q}\mathbf{x} = \pi \left( \int\sigma(r,T):S:\sigma(r,T)r\mathrm{\,dr} + \rho \dot{u}^2(r,T)\right)$, good news it is symmetric.
    #### Linear term
    $\mathbf{x}^T\mathbf{L} = \int \sigma_Y\lambda(T) \mathrm{\,dx}+ \int \left(p_e(t)\dot{u}(r_e,t)-p_i(t)\dot{u}(r_i,t)\right)\mathrm{\,dt}$
    #### Constant term
    $c = -  \int \sigma_Y\lambda(0) \mathrm{\,dx} - \frac{1}{2} \int \left(\sigma(0):S:\sigma(0)+\rho \dot{u}(0) ^2 \right)\mathrm{\,dx}$

    ### Time integration
    We suppose constant time step $\delta$.

    Backward Euler leads to : $\dot{\mathbf{u}}_t = (\mathbf{u}_t - \mathbf{u}_{t-1})/\delta$ and $\ddot{\mathbf{u}}_t = (\mathbf{u}_t - 2\mathbf{u}_{t-1} + \mathbf{u}_{t-2})/\delta^2$.

    Trapezoidal integration with constant time step is $\int p \mathrm{\,dt} = \delta\left(\sum_{T-1}  p_i\right)+ \displaystyle\frac{\delta}{2} (p_T-p_0)$

    The known bcs are removed later.
    """
    )
    return


@app.cell
def _(np):
    class sben:
        def __init__(self, ge, di, sm, lo):
            self.matTQforK = np.zeros((di.nbT*di.ntf, di.nbT*di.ntf))
            self.matTQforC = np.zeros((di.nbT*di.ntf, di.nbT*di.ntf))
            self.matTQ = np.zeros((di.nbT*di.ntf, di.nbT*di.ntf))
            self.matTL = np.zeros((di.nbTf*di.ntf, 1))
            self.constant = 0

            # Quadratic term, energy at last time step (lts for short). BCs taken into account later.
            lts = di.ntf-1

            # Kinetic energy
            self.matTQforK[np.ix_(di.gIndU(lts),
                                  di.gIndU(lts))] += .5 * sm.matMuu/lo.dt**2
            self.matTQforK[np.ix_(di.gIndU(lts),
                                  di.gIndU(lts-1))] -= .5 * sm.matMuu/lo.dt**2
            self.matTQforK[np.ix_(di.gIndU(lts-1),
                                  di.gIndU(lts))] -= .5 * sm.matMuu/lo.dt**2
            self.matTQforK[np.ix_(di.gIndU(lts-1),
                                  di.gIndU(lts-1))] += .5 * sm.matMuu/lo.dt**2

            # Elastic energy, including contribution from acceleration
            # (s_T + (u_T -2u_{T-1}+u_T-2)/delta^2)^2
            # ss term
            self.matTQforC[np.ix_(di.gIndS(lts),
                                  di.gIndS(lts))] += .5 * sm.matFss
            #  su terms
            self.matTQforC[np.ix_(di.gIndS(lts),
                                  di.gIndU(lts))] += .5 * sm.matFsu/lo.dt**2
            self.matTQforC[np.ix_(di.gIndS(lts),
                                  di.gIndU(lts-1))] -= sm.matFsu/lo.dt**2
            self.matTQforC[np.ix_(di.gIndS(lts),
                                  di.gIndU(lts-2))] += .5*sm.matFsu/lo.dt**2
            # us terms (sym)
            self.matTQforC[np.ix_(di.gIndU(lts),
                                  di.gIndS(lts))] += .5 * sm.matFus/lo.dt**2
            self.matTQforC[np.ix_(di.gIndU(lts-1),
                                  di.gIndS(lts))] -= sm.matFus/lo.dt**2
            self.matTQforC[np.ix_(di.gIndU(lts-2),
                                  di.gIndS(lts))] += .5*sm.matFus/lo.dt**2
            # uu terms
            self.matTQforC[np.ix_(di.gIndU(lts),
                                  di.gIndU(lts))] += .5 * sm.matFuu/lo.dt**4
            self.matTQforC[np.ix_(di.gIndU(lts-1),
                                  di.gIndU(lts))] -= sm.matFuu/lo.dt**4
            self.matTQforC[np.ix_(di.gIndU(lts),
                                  di.gIndU(lts-1))] -= sm.matFuu/lo.dt**4
            self.matTQforC[np.ix_(di.gIndU(lts-2),
                                  di.gIndU(lts))] += .5*sm.matFuu/lo.dt**4
            self.matTQforC[np.ix_(di.gIndU(lts),
                                  di.gIndU(lts-2))] += .5*sm.matFuu/lo.dt**4

            self.matTQforC[np.ix_(di.gIndU(lts-1),
                                  di.gIndU(lts-1))] += 2*sm.matFuu/lo.dt**4
            self.matTQforC[np.ix_(di.gIndU(lts-1),
                                  di.gIndU(lts-2))] -= sm.matFuu/lo.dt**4
            self.matTQforC[np.ix_(di.gIndU(lts-2),
                                  di.gIndU(lts-1))] -= sm.matFuu/lo.dt**4
            self.matTQforC[np.ix_(di.gIndU(lts-2),
                                  di.gIndU(lts-2))] += .5*sm.matFuu/lo.dt**4

            self.matTQ = self.matTQforC + self.matTQforK

            self.matTL[di.gIndLf(lts)] += sm.Lambda[:, np.newaxis]
            # The quadratic term was already integrated in time

            # For the trapezoidal integration, multiplies by the time step.
            # The contribution of the initial and last time steps must be halved

            self.matTLfromBCs = np.zeros_like(self.matTL)
            self.constantfromBCs = 0.
            for tt in range(1, di.ntf-1):
                self.matTLfromBCs[di.gIndUf(tt)[0]] -= 2 * \
                    np.pi*ge.ri*lo.p_int[tt+1]/lo.dt
                self.matTLfromBCs[di.gIndUf(tt)[-1]] += 2 * \
                    np.pi*ge.re*lo.p_ext[tt+1]/lo.dt
                self.matTLfromBCs[di.gIndUf(tt-1)[0]] += 2 * \
                    np.pi*ge.ri*lo.p_int[tt+1]/lo.dt
                self.matTLfromBCs[di.gIndUf(tt-1)[-1]] -= 2 * \
                    np.pi*ge.re*lo.p_ext[tt+1]/lo.dt

            #  first free time step
            self.matTLfromBCs[di.gIndUf(0)[0]] -= 2 * \
                np.pi*ge.ri*lo.p_int[1]/lo.dt
            self.matTLfromBCs[di.gIndUf(0)[-1]] += 2 * \
                np.pi*ge.re*lo.p_ext[1]/lo.dt
            self.constantfromBCs += 2 * np.pi * \
                (ge.ri*lo.p_int[0]*lo.u_ini[0] -
                 ge.re*lo.p_ext[0]*lo.u_ini[-1])
            # last time step (halved for integration)
            self.matTLfromBCs[di.gIndUf(lts)[0]] -= \
                np.pi*ge.ri*lo.p_int[lts+1]/lo.dt
            self.matTLfromBCs[di.gIndUf(lts)[-1]] += \
                np.pi*ge.re*lo.p_ext[lts+1]/lo.dt
            self.matTLfromBCs[di.gIndUf(lts-1)[0]] +=\
                np.pi*ge.ri*lo.p_int[lts+1]/lo.dt
            self.matTLfromBCs[di.gIndUf(lts-1)[-1]] -= \
                np.pi*ge.re*lo.p_ext[lts+1]/lo.dt
            # initial time step (halved for integration)
            self.constantfromBCs -= 2 * np.pi * \
                (ge.ri*lo.p_int[0]*lo.v_ini[0] -
                 ge.re*lo.p_ext[0]*lo.v_ini[-1])*lo.dt/2

            # Integration
            self.matTLfromBCs *= lo.dt
            #
            self.matTL += self.matTLfromBCs
            self.constant += self.constantfromBCs

            # Now we take into account the bcs in Q
            self.matTLfromQ = self.matTQ[np.ix_(di.iNbcs, di.ibcs)] @ lo.vbcs + \
                self.matTQ[np.ix_(di.ibcs, di.iNbcs)].T @ lo.vbcs
            self.constantfromQ = lo.vbcs.T @ (
                self.matTQ[np.ix_(di.ibcs, di.ibcs)] @ lo.vbcs)
            self.matTQ = np.delete(self.matTQ, di.ibcs, axis=0)
            self.matTQ = np.delete(self.matTQ, di.ibcs, axis=1)

            self.matTQforK = np.delete(self.matTQforK, di.ibcs, axis=0)
            self.matTQforK = np.delete(self.matTQforK, di.ibcs, axis=1)
            self.matTQforC = np.delete(self.matTQforC, di.ibcs, axis=0)
            self.matTQforC = np.delete(self.matTQforC, di.ibcs, axis=1)

            self.constant += self.constantfromQ
            self.matTL += self.matTLfromQ

            # initial value of mechanical energy
            self.initialC = .5 * (lo.s_ini.T @ sm.matFss @ lo.s_ini + 2 * lo.s_ini.T @
                                  sm.matFsu @ lo.a_ini + lo.a_ini.T @ sm.matFuu @ lo.a_ini)
            # initial value of kinetic energy
            self.initialK = .5 * lo.v_ini.T @ sm.matMuu @ lo.v_ini
            self.constant -= self.initialK + self.initialC
            # inital value of dissipation
            self.constant -= sm.Lambda.T @ lo.l_ini

        def Pi(self, x):
            return (x.T @ (self.matTQ @ x) + x.T @ self.matTL + self.constant).item()

        def absPi(self, x):
            return (x.T @ (self.matTQ @ x) + np.abs(x.T @ self.matTL) + np.abs(self.constant)).item()
    
        def piecesOfPi(self, x):
            print("Kinetic energy last:", x.T @ (self.matTQforK @ x), ", Kinetic energy init:",
                  self.initialK, "var: ", x.T @ (self.matTQforK @ x)-self.initialK)
            print("Elastic energy last:", x.T @ (self.matTQforC @ x)+self.matTLfromQ.T@x+self.constantfromQ,
                  ', quad', x.T @ (self.matTQforC @ x), ', lin', self.matTLfromQ.T@x, ', cte', self.constantfromQ)
            print("Elastic energy init:", self.initialC, ", var:", x.T @
                  (self.matTQforC @ x)+self.matTLfromQ.T@x+self.constantfromQ-self.initialC)
            print("From BCs:", ", lin:", self.matTLfromBCs.T @
                  x, ", const:", self.constantfromBCs)
            return
    return (sben,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Verification
    We can choose any $u(r,t)$ and $\sigma_r(r,t)$, deduce $\sigma_\theta$ by the equilibrium and compute $\Pi$. If the chosen fields are representable by our approximation (Lagrange FEM in space, continuous piece-wise linear in time), the computation should match.
    ### Ramp for stress case
    $\displaystyle\sigma_r=-\frac{t}{T}p_{max}$ and $u=0$, then $\displaystyle\Pi=\frac{\pi(r_e^2-r_i^2)}{2}p_{max}\left(\sum S_{ij}\right)$.
    We just need to provide the pressure and fill the $\sigma$ dof of $x$.
    """
    )
    return


@app.cell
def _(discretization, geometry, loading, material, np, sben, spacemat):
    ge = geometry()
    mat = material()
    di = discretization(nel=4, nt=60)
    ge.discretize(di)
    sm = spacemat(di, ge, mat)
    ##
    # Validation with a symramp (default loading):
    # We suppose homogeneous stress (opposite to pressure), zero displacement (Hooke law is not satisfied)

    lo = loading()
    lo.discretize(di, ge)

    ##
    from sympy import Symbol, diff, integrate, Expr
    # Define functions here
    def lam(r, t): return 0
    def ur(r, t): return (r/ge.ri)*t**2
    def sigr(r, t): return (r/ge.ri)*t**2
    # Definition of sigma_theta
    def sigt(r, t): return diff(r*sigr(r, t), r)-r*mat.rho*diff(ur(r, t), t, 2)
    # Ensuring the load is consistant (bcs and ics)
    lo.p_ext = -np.vectorize(sigr)(ge.re, lo.temps)
    lo.p_int = -np.vectorize(sigr)(ge.ri, lo.temps)
    lo.u_ini = np.vectorize(ur)(ge.coorU, 0.)
    lo.l_ini = np.vectorize(lam)(ge.coorL, 0.)
    lo.s_ini = np.vectorize(sigr)(ge.coorS, 0.)
    def vLeftS(tt): return -lo.p_int[tt+1]
    def vRightS(tt): return -lo.p_ext[tt+1]
    lo.vbcs = np.array([f(tt) for tt in range(di.ntf)
                                  for f in (vLeftS, vRightS)], dtype=float, ndmin=2).T

    r = Symbol('r')
    t = Symbol('t')
    vr = diff(ur(r, t), t).subs(t, 0)
    ar = diff(ur(r, t), t, 2).subs(t, 0)
    def v0(rr): return vr.subs(r, rr)
    def a0(rr): return ar.subs(r, rr)
    lo.v_ini = np.vectorize(v0)(ge.coorU).astype(float)
    lo.a_ini = np.vectorize(a0)(ge.coorU).astype(float)

    ##

    pb = sben(ge, di, sm, lo)
    x = np.zeros_like(pb.matTL)
    for itt in range(di.ntf):
        x[di.gIndSf(itt)] = np.vectorize(sigr)(
            [ge.coorS[1:-1, None]], lo.temps[itt+1])
        x[di.gIndUf(itt)] = np.vectorize(ur)(
            ge.coorU[:, None], lo.temps[itt+1])
        x[di.gIndLf(itt)] = np.vectorize(lam)(
            ge.coorL[:, None], lo.temps[itt+1])
    sbenvalue = pb.Pi(x)
    print("numerical sBEN:", sbenvalue)

    ##
    symPi: Expr = 2*np.pi*(integrate(
        (mat.sigY*diff(lam(r, t), t))*r
        + r * sigr(r, t)*(diff(mat.S[0, 0]*sigr(r, t) +
                               mat.S[0, 1]*sigt(r, t), t)-diff(ur(r, t), r, t))
        + sigt(r, t)*(r*diff(mat.S[1, 0]*sigr(r, t) +
                             mat.S[1, 1]*sigt(r, t), t)-diff(ur(r, t), t)),
        (r, ge.ri, ge.re), (t, 0, lo.T))).evalf()
    print('Symbolic sBEN ', symPi)
    print('Is it matching ?', np.isclose(sbenvalue, float(symPi)))
    pb.piecesOfPi(x)

    return di, ge, mat, pb, sm


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Constraints

    ### Equality

    $$\begin{pmatrix}-1 \\ 1\end{pmatrix}{\dot{\lambda}} = \dot{\hat{\epsilon}} - S\dot{\hat{\sigma}}$$

    $$\forall t, \ \begin{pmatrix}-1 \\ 1\end{pmatrix}[\lambda]_0^t  = [ \hat{\epsilon} ]_0^t - S[ \hat{\sigma} ]_0^t $$

    ### Inequality

    $$\dot{\lambda}\geqslant 0$$

    Tresca:

    $$ -\sigma_Y\leqslant \sigma_r -\sigma_\theta \leqslant \sigma_Y$$
    """
    )
    return


@app.cell
def _(LinearConstraint, Poly, apply_vectorized, np):
    # Equality constraints
    def Ceq(di, ge, mat, sm, lo):
        # we do not eliminate bcs at first but we do not take into account initial step
        matCeq = np.zeros(
            (2*di.nbL*di.ntf, di.nbT*di.ntf), dtype=float)
        lb = np.zeros(
            (2*di.nbL*di.ntf), dtype=float)
        ub = np.zeros(
            (2*di.nbL*di.ntf), dtype=float)

        phiSref, dphiSref = sm.lagrangeShapes(-1., 1., di.degS)
        phiUref, dphiUref = sm.lagrangeShapes(-1., 1., di.degU)

        for el in range(di.nel):
            # sides of the element
            rl = ge.coor_bord_el[el]
            rr = ge.coor_bord_el[el+1]
            h = rr-rl

            arr_polytimesR = np.vectorize(
                lambda P: P * (Poly([1, 1])*h/2+rl))
            arr_polytimesR2 = np.vectorize(
                lambda P: P * (Poly([1, 1])*h/2+rl)**2)
            # Everything is multiplied by r to keep polynomials
            rAsref = mat.S@np.vstack(
                (arr_polytimesR(phiSref), arr_polytimesR(phiSref)+arr_polytimesR2(dphiSref)*2/h))
            rAuref = mat.S@np.vstack(
                (np.zeros_like(phiUref), -mat.rho*arr_polytimesR2(phiUref)))
            rBuref = np.vstack(
                (arr_polytimesR(dphiUref)*2/h, phiUref))
    # TODO MANAGE FIRST TIME STEPS WITH ICS and CREATE bound vectors
            for tt in range(di.ntf):
                # lambda(t)
                matCeq[np.ix_(tt*di.nbL+di.indL[el], di.gIndLe(tt, el))
                       ] -= np.diag(ge.coorL[di.indL[el]])
                matCeq[np.ix_((di.ntf+tt)*di.nbL+di.indL[el],
                              di.gIndLe(tt, el))] += np.diag(ge.coorL[di.indL[el]])
                # sigma(t)
                matCeq[np.ix_(tt*di.nbL+di.indL[el], di.gIndSe(tt, el))
                       ] += apply_vectorized(rAsref[0, np.newaxis].T, di.rg).T
                matCeq[np.ix_((di.ntf+tt)*di.nbL+di.indL[el], di.gIndSe(tt, el))
                       ] += apply_vectorized(rAsref[1, np.newaxis].T, di.rg).T
                matCeq[np.ix_(tt*di.nbL+di.indL[el], di.gIndUe(tt, el))
                       ] += apply_vectorized(rAuref[0, np.newaxis].T, di.rg).T/lo.dt**2
                matCeq[np.ix_((di.ntf+tt)*di.nbL+di.indL[el], di.gIndUe(tt, el))
                       ] += apply_vectorized(rAuref[1, np.newaxis].T, di.rg).T/lo.dt**2
                # epsilon(t)
                matCeq[np.ix_(tt*di.nbL+di.indL[el], di.gIndUe(tt, el))
                       ] -= apply_vectorized(rBuref[0, np.newaxis].T, di.rg).T
                matCeq[np.ix_((di.ntf+tt)*di.nbL+di.indL[el], di.gIndUe(tt, el))
                       ] -= apply_vectorized(rBuref[1, np.newaxis].T, di.rg).T
                # contribution of acceleration
                if tt > 0:
                    matCeq[np.ix_(tt*di.nbL+di.indL[el], di.gIndUe(tt-1, el))
                           ] -= 2*apply_vectorized(rAuref[0, np.newaxis].T, di.rg).T/lo.dt**2
                    matCeq[np.ix_((di.ntf+tt)*di.nbL+di.indL[el], di.gIndUe(tt-1, el))
                           ] -= 2*apply_vectorized(rAuref[1, np.newaxis].T, di.rg).T/lo.dt**2
                if tt > 1:
                    matCeq[np.ix_(tt*di.nbL+di.indL[el], di.gIndUe(tt-2, el))
                           ] += apply_vectorized(rAuref[0, np.newaxis].T, di.rg).T/lo.dt**2
                    matCeq[np.ix_((di.ntf+tt)*di.nbL+di.indL[el], di.gIndUe(tt-2, el))
                           ] += apply_vectorized(rAuref[1, np.newaxis].T, di.rg).T/lo.dt**2
                #
                lb[tt*di.nbL+di.indL[el]] += - ge.coorL[di.indL[el]]*lo.l_ini[di.indL[el]] + apply_vectorized(rAsref[0, np.newaxis].T, di.rg).T @ lo.s_ini[di.indS[el]] + apply_vectorized(
                    rAuref[0, np.newaxis].T, di.rg).T @ lo.a_ini[di.indU[el]] - apply_vectorized(rBuref[0, np.newaxis].T, di.rg).T @ lo.u_ini[di.indU[el]]
                lb[(di.ntf+tt)*di.nbL+di.indL[el]] += ge.coorL[di.indL[el]]*lo.l_ini[di.indL[el]] + apply_vectorized(rAsref[1, np.newaxis].T, di.rg).T @ lo.s_ini[di.indS[el]
                                                                                                                                                                ] + apply_vectorized(rAuref[1, np.newaxis].T, di.rg).T @ lo.a_ini[di.indU[el]] - apply_vectorized(rBuref[1, np.newaxis].T, di.rg).T @ lo.u_ini[di.indU[el]]
            # tt == 0:
            lb[di.indL[el]] += apply_vectorized(rAuref[0, np.newaxis].T, di.rg).T @ (
                lo.u_ini[di.indS[el]] / lo.dt + lo.v_ini[di.indS[el]])/lo.dt
            lb[(di.ntf)*di.nbL+di.indL[el]] += apply_vectorized(rAuref[1,
                                                                       np.newaxis].T, di.rg).T @ (lo.u_ini[di.indS[el]] / lo.dt + lo.v_ini[di.indU[el]])/lo.dt
            # tt == 1:
            lb[di.nbL+di.indL[el]] -= apply_vectorized(
                rAuref[0, np.newaxis].T, di.rg).T @ lo.u_ini[di.indS[el]] / lo.dt**2
            lb[(di.ntf+1)*di.nbL+di.indL[el]] -= apply_vectorized(rAuref[1,
                                                                         np.newaxis].T, di.rg).T @ lo.u_ini[di.indS[el]] / lo.dt**2

        lb -= np.ravel(matCeq[:, di.ibcs] @ lo.vbcs)
        ub = lb
        matCeq = np.delete(matCeq, di.ibcs, axis=1)

        return LinearConstraint(matCeq, lb, ub)

    # Inequality constraints
    def Cineq(di, ge, mat, sm, lo):
        # we do not eliminate bcs at first but we do not take into account initial step
        matCineq = np.zeros(
            (2*di.nbL*di.ntf, di.nbT*di.ntf), dtype=float)
        lb = np.zeros(
            (2*di.nbL*di.ntf), dtype=float)
        ub = np.zeros(
            (2*di.nbL*di.ntf), dtype=float)

        phiSref, dphiSref = sm.lagrangeShapes(-1., 1., di.degS)
        phiUref, dphiUref = sm.lagrangeShapes(-1., 1., di.degU)

        for el in range(di.nel):
            rl = ge.coor_bord_el[el]
            rr = ge.coor_bord_el[el+1]
            h = rr-rl

            arr_polytimesR = np.vectorize(
                lambda P: P * (Poly([1, 1])*h/2+rl))

            for tt in range(di.ntf):
                matCineq[np.ix_(tt*di.nbL+di.indL[el], di.gIndSe(tt, el))
                         ] = apply_vectorized(arr_polytimesR(dphiSref).T,  di.rg).T
                matCineq[np.ix_(tt*di.nbL+di.indL[el], di.gIndUe(tt, el))] -= apply_vectorized(
                    mat.rho*arr_polytimesR(phiUref).T / lo.dt**2,  di.rg).T
                if tt > 0:
                    matCineq[np.ix_(tt*di.nbL+di.indL[el], di.gIndUe(tt-1, el))] += apply_vectorized(
                        2*mat.rho*arr_polytimesR(phiUref).T / lo.dt**2, di.rg).T
                if tt > 1:
                    matCineq[np.ix_(tt*di.nbL+di.indL[el], di.gIndUe(tt-2, el))] -= apply_vectorized(
                        mat.rho*arr_polytimesR(phiUref).T / lo.dt**2, di.rg).T
                ub[tt*di.nbL+di.indL[el]] = mat.sigY
                lb[tt*di.nbL+di.indL[el]] = -mat.sigY
                #
                matCineq[np.ix_((di.ntf+tt)*di.nbL+di.indL[el], di.gIndLe(tt, el))
                         ] += np.eye(di.degL)/lo.dt
                if tt > 0:
                    matCineq[np.ix_((di.ntf+tt-1)*di.nbL+di.indL[el], di.gIndLe(tt, el))
                             ] -= np.eye(di.degL)/lo.dt
                lb[(di.ntf+tt)*di.nbL+di.indL[el]] = 0
                ub[(di.ntf+tt)*di.nbL+di.indL[el]] = np.inf
            # tt=0
                lb[(di.ntf)*di.nbL+di.indL[el]] = lo.l_ini[di.indL[el]]/lo.dt

            # tt==0:
            lb[di.indL[el]] += apply_vectorized(mat.rho*arr_polytimesR(phiUref).T, di.rg).T @ (
                lo.u_ini[di.indS[el]] / lo.dt + lo.v_ini[di.indS[el]])/lo.dt
            ub[di.indL[el]] += apply_vectorized(mat.rho*arr_polytimesR(phiUref).T, di.rg).T @ (
                lo.u_ini[di.indS[el]] / lo.dt + lo.v_ini[di.indS[el]])/lo.dt
            # tt==1:
            lb[di.nbL+di.indL[el]] -= apply_vectorized(
                arr_polytimesR(phiUref).T, di.rg).T @ lo.u_ini[di.indS[el]] / lo.dt**2
            ub[di.nbL+di.indL[el]] -= apply_vectorized(
                arr_polytimesR(phiUref).T, di.rg).T @ lo.u_ini[di.indS[el]] / lo.dt**2

        ub -= np.ravel(matCineq[:, di.ibcs]@lo.vbcs)
        lb -= np.ravel(matCineq[:, di.ibcs]@lo.vbcs)
        matCineq = np.delete(matCineq, di.ibcs, axis=1)
        return LinearConstraint(matCineq, lb, ub)
    return Ceq, Cineq


@app.cell
def _(Ceq, Cineq, di, ge, loading, mat, minimize, np, pb, sben, sm):
    lo2 = loading(pmax=3.2,form="haltedramp")
    lo2.discretize(di, ge)
    pb2 = sben(ge, di, sm, lo2)
    x2 = np.ravel(np.zeros_like(pb.matTL))

    # eqc = {'type'='eq', 'fun' = Ceq(di,ge,mat,sm,lo) }
    # ineqc = {'type'='ineq', 'fun' = Cineq(di,ge,mat,sm,lo) }

    myConstraints = [Ceq(di, ge, mat, sm, lo2), Cineq(di, ge, mat, sm, lo2)]

    def dPi(x):
        return np.ravel(pb2.matTL) + 2*(pb2.matTQ@x)

    def HPi(x):
        return 2*pb2.matTQ

    res = minimize(pb2.Pi, x2, method='trust-constr',  constraints=myConstraints,
                   options={'disp': True}, jac=dPi, hess=HPi)

    return lo2, pb2, res


@app.cell
def _(di, ge, lo2, np, pb2, plt, res):
    print(pb2.Pi(res.x)/pb2.absPi(res.x))

    U = np.zeros((di.nbU, di.nt), dtype=float)
    Sig = np.zeros((di.nbS, di.nt), dtype=float)
    Lams = np.zeros((di.nbL, di.nt), dtype=float)
    for tt in range(di.ntf):
        U[:, tt+1] = res.x[tt*di.nbTf+np.arange(di.nbU)]
        Sig[1:-1, tt+1] = res.x[tt*di.nbTf+di.nbU+np.arange(di.nbSf)]
        Sig[0, tt+1] = -lo2.p_int[tt+1]
        Sig[-1, tt+1] = -lo2.p_ext[tt+1]
        Lams[:, tt+1] = res.x[tt*di.nbTf+di.nbU+di.nbSf+np.arange(di.nbL)]

    plt.plot(ge.coorS,Sig)
    plt.legend('sigma vs r')
    plt.show()

    plt.plot(ge.coorU,U)
    plt.legend('disp vs r')
    plt.show()

    plt.plot(ge.coorL,Lams)
    plt.legend('lam vs r')
    plt.show()

    plt.plot(lo2.temps,Sig[0,:])
    plt.legend('sigma(r_i) vs time')
    plt.show()

    plt.plot(lo2.temps,U[0,:])
    plt.legend('U(r_i) vs time')
    plt.show()

    '''
        phiSref, dphiSref = sm.lagrangeShapes(-1., 1., di.degS)
        phiUref, dphiUref = sm.lagrangeShapes(-1., 1., di.degU)


    coorUvisu = np.linspace(ge.ri, ge.re, di.nel*di.degU*10)
    Uvisu = np.zeros((di.nel*di.degU*10, di.nt), dtype=float)
    for tt in range(di.ntf):
        for el in range(di.nel):                    
            rl = ge.coor_bord_el[el]
            rr = ge.coor_bord_el[el+1]
            h = rr-rl
            arr_polytimesR = np.vectorize(
                lambda P: P * (Poly([1, 1])*h/2+rl))
        
    #        phiU, dphiU = lagrangeShapes(
    #            ge.coor_bord_el[el], ge.coor_bord_el[el+1], di.degU)
            Uvisu[el*di.degU*10:(el+1)*di.degU*10, tt +
                  1] = apply_vectorized(phiUref.T, np.arange(10*di.degU)).T@U[di.indU[el, :], tt+1]
    
    coorSvisu = np.linspace(ge.ri, ge.re, di.nel*di.degS*10)
    Svisu = np.zeros((di.nel*di.degS*10, di.nt), dtype=float)
    for tt in range(di.ntf):
        for el in range(di.nel):
    #        phiS, dphiS = lagrangeShapes( 
    #            ge.coor_bord_el[el], ge.coor_bord_el[el+1], di.degS)
            Svisu[el*di.degS*10:(el+1)*di.degS*10, tt +
                  1] = apply_vectorized(phiSref.T, np.arange(10*di.degS)).T@Sig[di.indS[el, :], tt+1]
    
    Sthetavisu = np.gradient(Svisu, axis=0)+np.diag(coorSvisu)@Svisu - \
        mat.rho * np.diag(coorSvisu)@np.gradient(np.gradient(Uvisu,
                                                         axis=1), axis=1)/lo.dt**2
    print(Sig)
    print(Svisu)
    '''
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
