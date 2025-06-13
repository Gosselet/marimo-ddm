import marimo

__generated_with = "0.13.15"
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
    return Bounds, LinearConstraint, Poly, minimize, mo, np


@app.cell
def _(T, np):
    # Definition of the configuration
    class geometry:
        def __init__(self, ri : float =100., re : float = 101.):
            self.ri: float = ri
            self.re: float = re

        def discretize(self,di):
            # coordinates of dofs
            self.coor_bord_el = np.linspace(self.ri, self.re, di.nel+1)
            self.size_el = np.diff(self.coor_bord_el)
            self.coorL = np.zeros(di.nbL)
            for el0 in range(di.nel):
                self.coorL[di.indL[el0]] = self.coor_bord_el[el0] + di.rg_01*self.size_el[el0]
            self.coorU = np.linspace(self.ri, self.re, di.nbU)
            self.coorS = np.linspace(self.ri, self.re, di.nbS)

    class material:
        def __init__(self, sY : float = 360, E : float = 210000, nu : float = 0.3, rho : float = 7.8e-9):
            self.sigY = sY
            self.E = E
            self.nu = nu
            self.rho=rho
            self.S = np.zeros((2, 2), dtype=float)
            Ebar = E/(1.-nu**2)
            nubar = nu/(1.-nu)
            self.S[1, 1] = self.S[0, 0] = 1./Ebar
            self.S[1, 0] = self.S[0, 1] = -nubar/Ebar

    class discretization:
        # indexing for continuous Lagrange
        def locToGloLag(ne: int, deg: int):
            ind = np.empty((ne, deg+1), dtype=int)
            for i in range(ne):
                ind[i, :] = np.arange(i*deg, (i+1)*deg+1)
            return ind

        # indexing for discontinuous Lagrange
        def locToGloDG(ne: int, nl: int):
            ind = np.empty((ne, nl), dtype=int)
            for i in range(ne):
                ind[i, :] = np.arange(i*nl, (i+1)*nl)
            return ind 

        def __init__(self, nel : int = 1, degU : int = 3, degS : int = 3, degL : int = 4, nt : int = 4):
            self.nel = nel
            self.degU = degU
            self.degS = degS
            self.degL = degL
            self.nt = nt #including initial time step (0)
            self.ntf = nt-1 #free time steps
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

            self.indU = discretization.locToGloLag(nel, degU)
            self.indS = discretization.locToGloLag(nel, degS)
            self.indL = discretization.locToGloDG(nel, degL)

            # Gauss Integration
            # quadrature on [-1,1]
            self.rg, self.wg = np.polynomial.legendre.leggauss(degL)
            # quadrature on [0,1]
            self.rg_01 = (self.rg+1)/2
            self.wg_01 = self.wg/2


        # global space-time numbering bcs not removed but ics removed
        def gIndU(self,tt):
            return tt*self.nbT+np.arange(self.nbU)
        def gIndS(self,tt):
            return tt*self.nbT+self.nbU+np.arange(self.nbS)
        def gIndL(self,tt):
            return tt*self.nbT+self.nbU+self.nbS+np.arange(self.nbL)
        # global space-time numbering bcs and ics removed
        def gIndUf(self,tt):
            return tt*self.nbTf+np.arange(self.nbUf)
        def gIndSf(self,tt):
            return tt*self.nbTf+self.nbUf+np.arange(self.nbSf)
        def gIndLf(self,tt):
            return tt*self.nbTf+self.nbUf+self.nbSf+np.arange(self.nbL)
        # element contributions with bcs kept
        def gIndUe(self,tt, el):
            return tt*self.nbT+self.indU[el]
        def gIndSe(self,tt, el):
            return tt*self.nbT+self.nbU+self.indS[el]
        def gIndLe(self,tt, el):
            return tt*self.nbT+self.nbU+self.nbS+self.indL[el]

    class loading:
        def __init__(self, T : float = .0002, pmax : float = 2., form : str = "symramp"):
            self.T = T
            self.form = form
            self.pmax = pmax

        def pint(self, t : float) -> float:
            if self.form == "ramp" or self.form == "symramp" :
                if 0 <= t <= self.T :
                    return self.pmax*t/self.T
                elif t<0 :
                    return 0.
                elif t>T :
                    return np.nan

        def pext(self, t : float) -> float:
            if self.form == "ramp":
                return 0.
            elif self.form == "symramp":
                return self.pint(t)

        def discretize(self, di:discretization):
            self.temps = np.linspace(0.,self.T,di.nt)
            self.p_int = np.vectorize(self.pint)(self.temps)
            self.p_ext = np.vectorize(self.pext)(self.temps)
            self.dt = self.T/di.ntf


    return discretization, geometry, loading, material


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
    Let $\phi_\sigma$ and $\phi_u$ be the Lagrange shape functions for $\sigma_r$ and $u_r$, thanks to equilibrium, we have:

    $$\hat{\sigma}_E = \begin{pmatrix}\sigma_r\\\sigma_\theta\end{pmatrix}=\underbrace{\begin{pmatrix}\phi_\sigma \\ (r\phi_\sigma)')\end{pmatrix}}_{\mathbf{A}_{s,E}}\boldsymbol{\sigma}_E+\underbrace{\begin{pmatrix}0 \\ -r\rho\phi_u\end{pmatrix}}_{\mathbf{A}_{u,E}}\mathbf{\ddot{u}}_E $$

    Regarding strains, we have (we multiply by $r$ to have polynomials): 

    $$ r\hat{\epsilon} = \begin{pmatrix}r\epsilon_r\\ r\epsilon_\theta\end{pmatrix}=\underbrace{\begin{pmatrix} r\phi_u' \\ \phi_u \end{pmatrix}}_{\mathbf{B}_u}\mathbf{u}$$

    Constraints are considered at Gauss points.
    """
    )
    return


@app.cell
def _(Poly, np):



    def polytimesr(x): return x * Poly.basis(1)
    def polytimesr2(x): return x * Poly.basis(2)
    arr_polytimesr = np.vectorize(polytimesr)
    arr_polytimesr2 = np.vectorize(polytimesr2)
    arr_polyderiv = np.vectorize(Poly.deriv)
    apply_vectorized = np.vectorize(lambda f, x: f(x))


    return apply_vectorized, arr_polytimesr, arr_polytimesr2


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Computation of SBEN
    ### Space integration

    We need to compute the quadratic term: $F = \pi \int\sigma(r,T):S:\sigma(r,T)r\mathrm{\,dr}$, good news it is symetric.

    And the linear term associated with $\lambda$.
    """
    )
    return


@app.cell
def _(Poly, apply_vectorized, np):
    class spacemat:
        # Lagrange shape functions
        def lagrangeShapes(self,a: float = 0., b: float = 1., deg: int = 3):
            xi = np.linspace(a, b, deg+1)
            shapes = []
            deriv = []
            for i in range(deg+1):
                roots = np.delete(xi, i)
                shape = Poly.fromroots(roots)
                shapes.append(shape / shape(xi[i]))
                deriv.append(Poly.deriv(shapes[-1]))
            return np.reshape(np.asarray(shapes), (1, deg+1)), np.reshape(np.asarray(deriv), (1, deg+1))
    
        def __init__(self,di, ge, mat):
            self.Lambda = np.zeros((di.nbL), dtype=float)
            self.matFss = np.zeros((di.nbS, di.nbS), dtype=float)
            self.matFsu = np.zeros((di.nbS, di.nbU), dtype=float)
            self.matFuu = np.zeros((di.nbU, di.nbU), dtype=float)
            self.matFus = np.zeros((di.nbU, di.nbS), dtype=float)
        
            phiSref, dphiSref = self.lagrangeShapes(-1., 1., di.degS)
            phiUref, dphiUref = self.lagrangeShapes(-1., 1., di.degU)
        
            for el in range(di.nel):
        
                self.Lambda[di.indL[el]] = mat.sigY * 2 * np.pi * ge.size_el[el] * ge.coorL[di.indL[el]]*di.wg_01
        
                # sides of the element
                rl = ge.coor_bord_el[el]
                rr = ge.coor_bord_el[el+1]
                h = rr-rl
        
                arr_polytimesR = np.vectorize(lambda P : P * (Poly([1,1])*h/2+rl))
        
                Asref =  np.vstack((phiSref, phiSref+arr_polytimesR(dphiSref)*2/h))
                Auref =  np.vstack((np.zeros_like(phiUref), -mat.rho*arr_polytimesR(phiUref)))
        
                def my_integRef(P): return Poly.integ(
                    h * np.pi * P * (Poly([1,1])*h/2+rl), lbnd=-1)
                arr_integRef = np.vectorize(my_integRef)
        
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
    ### Time integration
    All initial conditions are assumed to be zero. We suppose constant time step $\delta$.

    The space-time unknown is $\mathbf{x}=\begin{pmatrix} \begin{pmatrix} \mathbf{u} \\ \boldsymbol{\sigma} \\ \boldsymbol{\lambda} \end{pmatrix}_{t=1} \\ \vdots \\ \begin{pmatrix} \mathbf{u} \\ \boldsymbol{\sigma} \\ \boldsymbol{\lambda} \end{pmatrix}_{t=N} \end{pmatrix}$

    The sBEN functional can be written as: $\Pi(\mathbf{x})=\mathbf{x}^T\mathbf{Q}\mathbf{x}+\mathbf{x}^T\mathbf{L}$.

    The known bcs are removed later.

    Backward Euler leads to : $\dot{\mathbf{u}}_t = (\mathbf{u}_t - \mathbf{u}_{t-1})/\delta$ and $\ddot{\mathbf{u}}_t = (\mathbf{u}_t - 2\mathbf{u}_{t-1} + \mathbf{u}_{t-2})/\delta^2$.

    Trapezoidal integration with constant time step and zero initial conditions is $\int p \mathrm{\,dt} = \delta\left(\sum_{T-1}  p_i\right)+ \displaystyle\frac{\delta}{2} p_T$
    """
    )
    return


@app.cell
def _(np):
    class sben:
        def __init__(self,di,sm,lo):
            self.matTQ = np.zeros((di.nbT*di.ntf, di.nbT*di.ntf))
            self.matTL = np.zeros((di.nbTf*di.ntf, 1))
        
            # contribution of F, only at last time step (+ acceleration)
            self.matTQ[np.ix_(di.gIndS(di.ntf-1), di.gIndS(di.ntf-1))] += .5 * sm.matFss
        
            self.matTQ[np.ix_(di.gIndS(di.ntf-1), di.gIndU(di.ntf-1))] += .5 * sm.matFsu/lo.dt**2
            self.matTQ[np.ix_(di.gIndU(di.ntf-1), di.gIndS(di.ntf-1))] += .5 * sm.matFus/lo.dt**2
            self.matTQ[np.ix_(di.gIndS(di.ntf-1), di.gIndU(di.ntf-2))] -= sm.matFsu/lo.dt**2
            self.matTQ[np.ix_(di.gIndU(di.ntf-1), di.gIndS(di.ntf-2))] -= sm.matFus/lo.dt**2
            self.matTQ[np.ix_(di.gIndS(di.ntf-1), di.gIndU(di.ntf-3))] += .5 * sm.matFsu/lo.dt**2
            self.matTQ[np.ix_(di.gIndU(di.ntf-3), di.gIndS(di.ntf-1))] += .5 * sm.matFus/lo.dt**2
        
            self.matTQ[np.ix_(di.gIndU(di.ntf-1), di.gIndU(di.ntf-1))] += .5 * sm.matFuu/lo.dt**4
            self.matTQ[np.ix_(di.gIndU(di.ntf-1), di.gIndU(di.ntf-2))] -= sm.matFuu/lo.dt**4
            self.matTQ[np.ix_(di.gIndU(di.ntf-2), di.gIndU(di.ntf-1))] -= sm.matFuu/lo.dt**4
            self.matTQ[np.ix_(di.gIndU(di.ntf-1), di.gIndU(di.ntf-3))] += .5 * sm.matFuu/lo.dt**4
            self.matTQ[np.ix_(di.gIndU(di.ntf-3), di.gIndU(di.ntf-1))] += .5 * sm.matFuu/lo.dt**4
            self.matTQ[np.ix_(di.gIndU(di.ntf-2), di.gIndU(di.ntf-2))] += 2. * sm.matFuu/lo.dt**4
            self.matTQ[np.ix_(di.gIndU(di.ntf-2), di.gIndU(di.ntf-3))] -= sm.matFuu/lo.dt**4
            self.matTQ[np.ix_(di.gIndU(di.ntf-3), di.gIndU(di.ntf-2))] -= sm.matFuu/lo.dt**4
            self.matTQ[np.ix_(di.gIndU(di.ntf-3), di.gIndU(di.ntf-3))] += .5 * sm.matFuu/lo.dt**4
        
            # The quadratic term was already integrated in time
        
            # For the trapezoidal integration, multiplies by the time step.
            # The contribution of the last time step must be halved
        
            for tt in range(di.ntf):
                self.matTL[di.gIndLf(tt), 0] += sm.Lambda
                self.matTL[di.gIndUf(tt)[0]] -= lo.p_int[tt+1]/lo.dt
                self.matTL[di.gIndUf(tt)[-1]] += lo.p_ext[tt+1]/lo.dt
                if tt>0:
                    self.matTL[di.gIndUf(tt-1)[0]] += lo.p_int[tt+1]/lo.dt
                    self.matTL[di.gIndUf(tt-1)[-1]] -= lo.p_ext[tt+1]/lo.dt
            # Integration
            self.matTL *= lo.dt
            self.matTL[di.nbTf*(di.ntf-1):] /= 2.
        
            # Now we take into account the bcs in Q
        
            def iLeftS(tt): return di.gIndS(tt)[0]
            def iRightS(tt): return di.gIndS(tt)[-1]
            def vLeftS(tt): return -lo.p_int[tt+1]
            def vRightS(tt): return -lo.p_ext[tt+1]
        
        
            ibcs = np.array([f(tt) for tt in range(di.ntf)
                            for f in (iLeftS, iRightS)], dtype=int)
            vbcs = np.array([f(tt) for tt in range(di.ntf)
                            for f in (vLeftS, vRightS)], dtype=float,ndmin=2).T
            iNbcs = np.setdiff1d(np.arange(di.nbT*di.ntf), ibcs)
        
            self.matTL += self.matTQ[np.ix_(iNbcs, ibcs)] @ vbcs + \
                self.matTQ[np.ix_(ibcs, iNbcs)].T @ vbcs
            self.constant = vbcs.T @ (self.matTQ[np.ix_(ibcs, ibcs)] @ vbcs)
            self.matTQ = np.delete(self.matTQ, ibcs, axis=0)
            self.matTQ = np.delete(self.matTQ, ibcs, axis=1)
    
        def Pi(self,x):
            return x.T @ (self.matTQ @ x) + x.T @ self.matTL + self.constant
    return (sben,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Verification
    We can choose any $u(r,t)$ and $\sigma_r(r,t)$, deduce $\sigma_\theta$ by the equilibrium and compte $\Pi$. If the chosen fields are representable by or approximation (Lagrange FEM is space, continuous piece-wise linear in time), the computation should match.
    ### Ramp for stress case
    $\displaystyle\sigma_r=-\frac{t}{T}p_{max}$ and $u=0$, then $\displaystyle\Pi=\frac{\pi(r_e^2-r_i^2)}{2}p_{max}\left(\sum S_{ij}\right)$.
    We just need to provide the pressure and fill the $\sigma$ dof of $x$.

    ### Ramp for stress + constant displacement
    """
    )
    return


@app.cell
def _(discretization, geometry, loading, material, np, sben, spacemat):
    ge = geometry()
    mat = material()
    di = discretization(nel=2,nt=6)
    ge.discretize(di)
    sm = spacemat(di, ge, mat)


    # Validation with a symramp (default loading):
    # We suppose homogeneous stress (opposite to pressure), zero displacement (Hooke law is not satisfied)

    lo = loading()
    lo.discretize(di)
    symramp = sben(di,sm,lo)

    x = np.zeros_like(symramp.matTL)
    for itt in range(di.ntf):
        x[di.gIndSf(itt)]=-lo.p_int[itt+1]

    sbenvalue = symramp.Pi(x)
    print("numerical Pi:", sbenvalue)
    sfinal = np.array([[-lo.p_ext[-1]], [-lo.p_ext[-1]]])
    aPi = (sfinal.T@mat.S@sfinal) * np.pi * (ge.re**2-ge.ri**2)/2
    print("analytical Pi:", aPi)
    assert(np.isclose(sbenvalue,aPi))
    return di, lo, x


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Constraints and bounds

    $$\dot{\lambda}\geqslant 0$$

    $$\begin{pmatrix}-1 \\ 1\end{pmatrix}{\dot{\lambda}} = \dot{\hat{\epsilon}} - S\dot{\hat{\sigma}}$$

    Tresca:

    $$ -\sigma_Y\leqslant \sigma_r -\sigma_\theta \leqslant \sigma_Y$$
    """
    )
    return


@app.cell
def _(
    Bounds,
    LinearConstraint,
    S,
    apply_vectorized,
    arr_polytimesr,
    arr_polytimesr2,
    coorL,
    coor_bord_el,
    degL,
    degS,
    degU,
    di,
    dphiS,
    gIndLe,
    gIndSe,
    gIndUe,
    indL,
    lagrangeShapes,
    lo,
    nbL,
    nbS,
    nbT,
    nbU,
    nel,
    np,
    ntf,
    p_ext,
    p_int,
    phiU,
    rho,
    sigY,
):
    # Bounds
    def Beq():
        lb = np.full(
            (di.nbTf*di.ntf), -np.inf)
        ub = np.full(
            (di.nbTf*di.ntf), np.inf)
        for tt in range(di.ntf):
            lb[di.gIndLf(tt)] = 0.
        return Bounds(lb, ub)  # type: ignore

    # Equality constraints
    def Ceq():
        # we do not eliminate bcs at first but we do not take into account initial step
        matCeq = np.zeros(
            (2*di.nbL*ntf, nbT*ntf), dtype=float)
        lb = np.zeros(
            (2*nbL*ntf), dtype=float)
        ub = np.zeros(
            (2*nbL*ntf), dtype=float)

        for tt in range(ntf):
            for el in range(nel):
                # first set of constraints : lambda +  dot(epsilon_r) - S dot(sigma)
                # warning we assume 0 initial conditions and same time interval
                phiU, dphiU = lagrangeShapes(
                    coor_bord_el[el], coor_bord_el[el+1], degU)
                phiS, dphiS = lagrangeShapes(
                    coor_bord_el[el], coor_bord_el[el+1], degS)

                matCeq[np.ix_(tt*nbL+indL[el], gIndUe(tt, el))] += apply_vectorized(
                    dphiU.T, coorL[el*degL:(el+1)*degL]).T/lo.dt
                matCeq[np.ix_(tt*nbL+indL[el], gIndSe(tt, el))] -= S[0, 0]*apply_vectorized(
                    phiS.T, coorL[el*degL:(el+1)*degL]).T/lo.dt
                matCeq[np.ix_(tt*nbL+indL[el], gIndLe(tt, el))] = np.eye(degL)
                # addign terms of sigma_theta
                matCeq[np.ix_(tt*nbL+indL[el], gIndSe(tt, el))] -= S[0, 1]*apply_vectorized(
                    arr_polytimesr(dphiS).T, coorL[el*degL:(el+1)*degL]).T/lo.dt
                matCeq[np.ix_(tt*nbL+indL[el], gIndSe(tt, el))] -= S[0, 1]*apply_vectorized(
                    phiS.T, coorL[el*degL:(el+1)*degL]).T/lo.dt
                matCeq[np.ix_(tt*nbL+indL[el], gIndUe(tt, el))] = S[0, 1]*apply_vectorized(
                    rho*arr_polytimesr(phiU).T, coorL[el*degL:(el+1)*degL]).T/lo.dt**3

                if tt > 0:
                    matCeq[np.ix_(tt*nbL+indL[el], gIndUe(tt-1, el))] -= apply_vectorized(
                        dphiU.T, coorL[el*degL:(el+1)*degL]).T/lo.dt
                    matCeq[np.ix_(tt*nbL+indL[el], gIndSe(tt, el))] += S[0, 0]*apply_vectorized(
                        phiS.T, coorL[el*degL:(el+1)*degL]).T/lo.dt
                # adding terms for sigma_theta
                    matCeq[np.ix_(tt*nbL+indL[el], gIndSe(tt-1, el))] += S[0, 1]*apply_vectorized(
                        arr_polytimesr(dphiS).T, coorL[el*degL:(el+1)*degL]).T/lo.dt
                    matCeq[np.ix_(tt*nbL+indL[el], gIndSe(tt-1, el))] += S[0, 1]*apply_vectorized(
                        phiS.T, coorL[el*degL:(el+1)*degL]).T/lo.dt
                    matCeq[np.ix_(tt*nbL+indL[el], gIndUe(tt-1, el))] -= 3*S[0, 1]*apply_vectorized(
                        rho*arr_polytimesr(phiU).T, coorL[el*degL:(el+1)*degL]).T/lo.dt**3
                if tt > 1:
                    matCeq[np.ix_(tt*nbL+indL[el], gIndUe(tt-2, el))] += 3*S[0, 1]*apply_vectorized(
                        rho*arr_polytimesr(phiU).T, coorL[el*degL:(el+1)*degL]).T/lo.dt**3
                if tt > 2:
                    matCeq[np.ix_(tt*nbL+indL[el], gIndUe(tt-3, el))] -= S[0, 1]*apply_vectorized(
                        rho*arr_polytimesr(phiU).T, coorL[el*degL:(el+1)*degL]).T/lo.dt**3

                # second set of constraints : r*lambda = r*dot(epsilon)_p_theta-r*s[1,0]*dot(sigma_r)*-r*S[1,1]dot(sigma_theta)
                # warning we assume 0 initial conditions and same time interval
                matCeq[np.ix_(nbL*ntf+tt*nbL+indL[el], gIndUe(tt, el))] += apply_vectorized(
                    phiU.T, coorL[el*degL:(el+1)*degL]).T/lo.dt
                matCeq[np.ix_(nbL*ntf+tt*nbL+indL[el], gIndSe(tt, el))] -= S[1, 1] * apply_vectorized(
                    arr_polytimesr2(dphiS).T+arr_polytimesr(phiS).T, coorL[el*degL:(el+1)*degL]).T/lo.dt
                matCeq[np.ix_(nbL*ntf+tt*nbL+indL[el], gIndUe(tt, el))] += apply_vectorized(
                    rho*arr_polytimesr2(phiU).T, coorL[el*degL:(el+1)*degL]).T/lo.dt**3
                matCeq[np.ix_(nbL*ntf+tt*nbL+indL[el], gIndSe(tt, el))] -= S[1, 0]*apply_vectorized(  # added missing term of sigma_r
                    arr_polytimesr(phiS).T, coorL[el*degL:(el+1)*degL]).T/lo.dt

                matCeq[np.ix_(nbL*ntf+tt*nbL+indL[el],
                              gIndLe(tt, el))] = -np.diag(coorL[el*degL:(el+1)*degL])
                if tt > 0:
                    matCeq[np.ix_(nbL*ntf+tt*nbL+indL[el], gIndUe(tt-1, el))] -= apply_vectorized(
                        phiU.T, coorL[el*degL:(el+1)*degL]).T/lo.dt
                    matCeq[np.ix_(nbL*ntf+tt*nbL+indL[el], gIndSe(tt-1, el))] += S[1, 1] * apply_vectorized(
                        arr_polytimesr2(dphiS).T+arr_polytimesr(phiS).T, coorL[el*degL:(el+1)*degL]).T/lo.dt
                    matCeq[np.ix_(nbL*ntf+tt*nbL+indL[el], gIndUe(tt-1, el))] -= 3*apply_vectorized(
                        rho*arr_polytimesr2(phiU).T, coorL[el*degL:(el+1)*degL]).T/lo.dt**3
                    matCeq[np.ix_(nbL*ntf+tt*nbL+indL[el], gIndSe(tt-1, el))] += S[1, 0]*apply_vectorized(  # added missing term of sigma_r
                        arr_polytimesr(phiS).T, coorL[el*degL:(el+1)*degL]).T/lo.dt
                if tt > 1:
                    matCeq[np.ix_(nbL*ntf+tt*nbL+indL[el], gIndUe(tt-2, el))] += 3*apply_vectorized(
                        rho*arr_polytimesr2(phiU).T, coorL[el*degL:(el+1)*degL]).T/lo.dt**3
                if tt > 2:
                    matCeq[np.ix_(nbL*ntf+tt*nbL+indL[el], gIndUe(tt-3, el))] -= apply_vectorized(
                        rho*arr_polytimesr2(phiU).T, coorL[el*degL:(el+1)*degL]).T/lo.dt**3

        indC = np.zeros(2*ntf, dtype=int)
        sd = np.zeros(2*ntf, dtype=float)
        for tt in range(ntf):
            indC[2*tt:2*(tt+1)] = [tt*nbT+nbU, tt*nbT+nbU+nbS]
            sd[2*tt:2*(tt+1)] = [-p_int[tt], -p_ext[tt]]
        ub -= matCeq[:, indC]@sd
        lb -= matCeq[:, indC]@sd
        matCeq = np.delete(matCeq, indC, axis=1)
        return LinearConstraint(matCeq, lb, ub)  # type: ignore

    # Inequality constraints
    def Cineq():
        # we do not eliminate bcs at first but we do not take into account initial step
        matCeq = np.zeros(
            (nbL*ntf, nbT*ntf), dtype=float)
        lb = np.zeros(
            (nbL*ntf), dtype=float)
        ub = np.zeros(
            (nbL*ntf), dtype=float)

        for tt in range(ntf):
            for el in range(nel):
                matCeq[np.ix_(2*nbL*ntf+tt*nbL+indL[el], gIndSe(tt, el))
                       ] = apply_vectorized(arr_polytimesr(dphiS).T, coorL[el*degL:(el+1)*degL]).T
                matCeq[np.ix_(2*nbL*ntf+tt*nbL+indL[el], gIndUe(tt, el))] = - apply_vectorized(
                    rho*arr_polytimesr(phiU).T / lo.dt**2, coorL[el*degL:(el+1)*degL]).T
                if tt > 0:
                    matCeq[np.ix_(2*nbL*ntf+tt*nbL+indL[el], gIndUe(tt-1, el))] = + apply_vectorized(
                        2*rho*arr_polytimesr(phiU).T / lo.dt**2, coorL[el*degL:(el+1)*degL]).T
                if tt > 1:
                    matCeq[np.ix_(2*nbL*ntf+tt*nbL+indL[el], gIndUe(tt-2, el))] = - apply_vectorized(
                        rho*arr_polytimesr(phiU).T / lo.dt**2, coorL[el*degL:(el+1)*degL]).T
                ub[2*nbL*ntf+tt*nbL+indL[el]] = sigY
                lb[2*nbL*ntf+tt*nbL+indL[el]] = -sigY

        indC = np.zeros(2*ntf, dtype=int)
        sd = np.zeros(2*ntf, dtype=float)
        for tt in range(ntf):
            indC[2*tt:2*(tt+1)] = [tt*nbT+nbU, tt*nbT+nbU+nbS]
            sd[2*tt:2*(tt+1)] = [-p_int[tt], -p_ext[tt]]
        ub -= matCeq[:, indC]@sd
        lb -= matCeq[:, indC]@sd
        matCeq = np.delete(matCeq, indC, axis=1)
        return LinearConstraint(matCeq, lb, ub)  # type: ignore


    return Beq, Ceq, Cineq


@app.cell
def _(Beq, Ceq, Cineq, Pi, matQ, matTL, minimize, myBounds, x):
    myBound = Beq()
    myConstraints = {"eq" : Ceq(), "ineq" : Cineq()}
    def dPi(x):
        return matTL + 2*(matQ@x)
    def HPi(x):
        return 2*matQ

    res = minimize(Pi, x, method='trust-ncg', constraints=myConstraints, bounds=myBounds,
                   options={'disp': True}, jac=dPi, hess=HPi)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
