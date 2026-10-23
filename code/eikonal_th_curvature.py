"""
Explicit curvature corrections for the Taylor-Hood surface Eikonal LVPP system.

TH           : original ambient surface-divergence residual, bcs=[].
TH-curvature : exactly that residual minus BOTH curvature couplings, bcs=[].

All modes use continuous Q2 ambient vectors and continuous Q1 scalars,
with degree-2 or degree-3 quadrilateral geometry and matched solver settings. In the
explicit correction, c_h = div_Gamma_h(n_h) * n_h is computed on each cell
from the actual geometric mesh; it is unchanged by reversal of the local normal.
No interior-edge correction is silently added to TH-curvature.
"""

import argparse
import json
import time
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

import basix
import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from dolfinx.fem import Constant, Expression, Function, form
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

R_MOBIUS, W_MOBIUS = 2.0, 1.0


# --------------------------------------------------------------------------------------
# Meshes
# --------------------------------------------------------------------------------------
def _tensor_index(nsub):
    grids = np.meshgrid(*[np.arange(n) for n in nsub], indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


def _orient(F, tdim, gdim):
    """Swap the first two parameters of a block map if it is negatively oriented (tdim == gdim)."""
    if tdim != gdim:
        return F
    xc = np.full((1, tdim), 0.5)
    hstep = 1e-6
    Jc = np.column_stack(
        [(F(xc + hstep * e) - F(xc - hstep * e))[0] / (2 * hstep) for e in np.eye(tdim)]
    )
    if np.linalg.det(Jc) > 0:
        return F
    perm = np.arange(tdim)
    perm[[0, 1]] = perm[[1, 0]]
    return lambda xi: F(xi[:, perm])


def build_block_mesh(comm, cell_type, gdeg, blocks, tol=1e-9):
    """Structured multi-block mesh.  blocks = [(F, nsub)], F maps (N, tdim) points of [0,1]^tdim
    to (N, gdim) physical points; each block is split into prod(nsub) cells with Lagrange geometry
    of degree gdeg evaluated from the exact map.  Coincident nodes (block interfaces, periodic
    identifications such as the Moebius twist) are merged geometrically."""
    ctype = getattr(basix.CellType, cell_type)
    ref = basix.create_element(
        basix.ElementFamily.P, ctype, gdeg, basix.LagrangeVariant.equispaced
    ).points
    nref, tdim = ref.shape
    if comm.rank == 0:
        X = []
        for F, nsub in blocks:
            gdim = F(np.full((1, tdim), 0.5)).shape[1]
            F = _orient(F, tdim, gdim)
            idx = _tensor_index(nsub)
            xi = (idx[:, None, :] + ref[None, :, :]) / np.asarray(nsub, dtype=float)
            X.append(F(xi.reshape(-1, tdim)))
        X = np.vstack(X)
        pairs = cKDTree(X).query_pairs(tol, output_type="ndarray")
        n = X.shape[0]
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _, labels = connected_components(g, directed=False)
        x = np.empty((labels.max() + 1, X.shape[1]))
        x[labels] = X
        cells = labels.reshape(-1, nref).astype(np.int64)
        gdim = X.shape[1]
    else:
        gdim = None
    gdim = comm.bcast(gdim, root=0)
    if comm.rank != 0:
        cells = np.empty((0, nref), dtype=np.int64)
        x = np.empty((0, gdim), dtype=np.float64)
    # The coordinate element must use the SAME nodes as `ref` above. At degree
    # two the usual variants coincide; at degree three equispaced and GLL nodes
    # differ, so leaving this implicit distorts the geometric map.
    domain = ufl.Mesh(basix.ufl.element(
        "Lagrange", cell_type, gdeg,
        lagrange_variant=basix.LagrangeVariant.equispaced, shape=(gdim,)))
    partitioner = dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.none)
    return dolfinx.mesh.create_mesh(comm, cells=cells, x=x, e=domain, partitioner=partitioner)


def mobius_mesh(comm, n, R=R_MOBIUS, w=W_MOBIUS, gdeg=2):
    """Moebius strip X(th, s) = ((R + s cos(th/2)) cos th, (R + s cos(th/2)) sin th, s sin(th/2)),
    s in [-w, w]; (8n) x (2n) quadrilaterals (MFEM mobius-strip.mesh refined l times: n = 2^l)."""

    def F(xi):
        th = 2 * np.pi * xi[:, 0]
        s = w * (2 * xi[:, 1] - 1)
        rho = R + s * np.cos(th / 2)
        return np.stack([rho * np.cos(th), rho * np.sin(th), s * np.sin(th / 2)], axis=1)

    return build_block_mesh(comm, "quadrilateral", gdeg, [(F, (8 * n, 2 * n))])


# --------------------------------------------------------------------------------------
# Exact viscosity solutions u = dist(x, dOmega), grad u, and distance to the singular set
# --------------------------------------------------------------------------------------


def mobius_mean_curvature_x2(x, R=R_MOBIUS):
    """2H = tr(shape operator) of the exact Moebius strip at the point x (sign depends on the
    normal orientation).  With F = 0, G = 1, X_ss = 0:  2H = X_tt . (X_t x X_s) / E^{3/2}."""
    th = ufl.atan2(x[1], x[0])
    rho = ufl.sqrt(x[0] ** 2 + x[1] ** 2)
    c, sn = ufl.cos(th / 2), ufl.sin(th / 2)
    s = (rho - R) * c + x[2] * sn
    r = R + s * c
    Xt = ufl.as_vector([-r * ufl.sin(th) - 0.5 * s * sn * ufl.cos(th),
                        r * ufl.cos(th) - 0.5 * s * sn * ufl.sin(th), 0.5 * s * c])
    Xs = ufl.as_vector([c * ufl.cos(th), c * ufl.sin(th), sn])
    rt = -0.5 * s * sn  # d r / d th
    rtt = -0.25 * s * c
    Xtt = ufl.as_vector([rtt * ufl.cos(th) - 2 * rt * ufl.sin(th) - r * ufl.cos(th),
                         rtt * ufl.sin(th) + 2 * rt * ufl.cos(th) - r * ufl.sin(th), -0.25 * s * sn])
    E = ufl.dot(Xt, Xt)
    return ufl.dot(Xtt, ufl.cross(Xt, Xs)) / E**1.5


def mobius_curvature_vector(x, R=R_MOBIUS):
    """Exact smooth curvature vector div_Gamma(n)*n, extended near the surface.

    With E=|X_theta|^2, F=0, G=1 and X_ss=0, the sign convention is
    kappa=-X_theta,theta dot (X_theta cross X_s) / E**(3/2).
    Multiplying by n cancels all local normal-orientation dependence.
    """
    theta = ufl.atan2(x[1], x[0])
    rho = ufl.sqrt(x[0]**2 + x[1]**2)
    c, sn = ufl.cos(theta/2), ufl.sin(theta/2)
    s = (rho-R)*c + x[2]*sn
    r = R+s*c
    rt, rtt = -0.5*s*sn, -0.25*s*c
    Xt = ufl.as_vector([-r*ufl.sin(theta)-0.5*s*sn*ufl.cos(theta),
                        r*ufl.cos(theta)-0.5*s*sn*ufl.sin(theta), 0.5*s*c])
    Xs = ufl.as_vector([c*ufl.cos(theta), c*ufl.sin(theta), sn])
    Xtt = ufl.as_vector([rtt*ufl.cos(theta)-2*rt*ufl.sin(theta)-r*ufl.cos(theta),
                         rtt*ufl.sin(theta)+2*rt*ufl.cos(theta)-r*ufl.sin(theta), -0.25*s*sn])
    cross = ufl.cross(Xt, Xs)
    E = ufl.dot(Xt, Xt)
    return -ufl.dot(Xtt, cross) / E**2 * cross


def exact_solution(domain, x):
    if domain != "mobius":
        raise ValueError(f"Unsupported domain: {domain}")
    rho = ufl.sqrt(x[0] ** 2 + x[1] ** 2)
    s_abs = ufl.sqrt((rho - R_MOBIUS) ** 2 + x[2] ** 2)
    u = W_MOBIUS - s_abs
    g = -ufl.as_vector([
        (rho - R_MOBIUS) * x[0] / rho,
        (rho - R_MOBIUS) * x[1] / rho,
        x[2],
    ]) / s_abs
    return u, g, s_abs


# --------------------------------------------------------------------------------------
# LVPP solver
# --------------------------------------------------------------------------------------
class EikonalLVPP:
    def __init__(self, mesh, domain, disc, qdeg, qdeg_err, delta=0.1, eps_quasi=1e-6, factor="cholesky"):
        self.mesh, self.comm = mesh, mesh.comm
        cell, gdim = mesh.basix_cell(), mesh.geometry.dim
        if disc not in ("TH", "TH-curvature"):
            raise ValueError(f"Unsupported formulation: {disc}")
        el_psi = basix.ufl.element("Lagrange", cell, 2, shape=(gdim,))
        el_u = basix.ufl.element("Lagrange", cell, 1)
        self.disc = disc
        self.factor = factor
        W = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([el_psi, el_u]))
        self.W = W
        self.w, self.w_old, self.dw, self.w_bak = (Function(W) for _ in range(4))
        U, self.u_map = W.sub(1).collapse()
        _, self.psi_map = W.sub(0).collapse()
        self.U = U
        self.ndofs = W.dofmap.index_map.size_global * W.dofmap.index_map_bs

        psi, u = ufl.split(self.w)
        psi_old, u_old = ufl.split(self.w_old)
        tau, v = ufl.TestFunctions(W)
        wt = ufl.TrialFunction(W)
        dpsi, _ = ufl.split(wt)
        self.alpha = Constant(mesh, PETSc.ScalarType(0.0))
        dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": qdeg})

        # Geometry is differentiated cellwise on the actual geometric surface.
        n_h = ufl.CellNormal(mesh)
        kappa_h = ufl.div(n_h)
        curvature_vector = kappa_h * n_h
        self.curvature_vector = curvature_vector
        Z = psi / ufl.sqrt(1 + ufl.dot(psi, psi))
        F = (
            ufl.inner(Z, tau) * dx
            + u * ufl.div(tau) * dx
            + (ufl.div(psi) - ufl.div(psi_old)) * v * dx
            + self.alpha * v * dx
        )
        if disc == "TH-curvature":
            # Keep the ORIGINAL divergence residual and boundary handling.
            # Add the two negative curvature terms explicitly.
            F -= u * ufl.dot(curvature_vector, tau) * dx
            F -= v * ufl.dot(curvature_vector, psi - psi_old) * dx
        self.bcs = []
        J = ufl.derivative(F, self.w, wt)
        self.F_form = form(F)
        self.audit_quadrature_degree = max(qdeg + 4, 16)
        self.F_audit_form = form(ufl.Form([
            integral.reconstruct(metadata={"quadrature_degree": self.audit_quadrature_degree})
            for integral in F.integrals()
        ]))
        self.J_form = form(J)
        self.Jq_form = form(J + eps_quasi * ufl.inner(dpsi, tau) * dx)

        # Jacobians are symmetric indefinite (saddle point): MUMPS LDL^T with 2x2 pivoting
        self.A = dolfinx.fem.petsc.create_matrix(self.J_form)
        if factor == "cholesky":
            self.A.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        self.b = dolfinx.fem.petsc.create_vector(W)
        self.b_audit = dolfinx.fem.petsc.create_vector(W)
        self.icntl14 = 200
        self.ksp = self._make_ksp()

        # ---- error functionals -------------------------------------------------------
        x = ufl.SpatialCoordinate(mesh)
        u_ex, g_ex, dsing = exact_solution(domain, x)
        dxe = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": qdeg_err})
        chi = ufl.conditional(ufl.gt(dsing, delta), 1.0, 0.0)
        eu = (u_ex - u) ** 2
        es = ufl.dot(g_ex - Z, g_ex - Z)
        funcs = {
            "eu": eu * dxe,
            "es": es * dxe,
            "eu_loc": chi * eu * dxe,
            "es_loc": chi * es * dxe,
            "psi2": ufl.dot(psi, psi) * dxe,
            "du": (u - u_old) ** 2 * dxe,
            "int_u": u * dxe,
            "int_uex": u_ex * dxe,
            "vol": 1.0 * dxe,
            "curvature2": ufl.dot(curvature_vector, curvature_vector) * dxe,
            "curvature_error2": ufl.dot(curvature_vector - mobius_curvature_vector(x),
                                         curvature_vector - mobius_curvature_vector(x)) * dxe,
            "Z2max_viol": ufl.max_value(ufl.dot(Z, Z) - 1.0, 0.0) * dxe,
        }
        if mesh.topology.dim < gdim:
            nrm = ufl.CellNormal(mesh)
            funcs["Zn"] = ufl.dot(Z, nrm) ** 2 * dxe
            if domain == "mobius":
                # predicted normal component for the ambient-vector mixed form:
                # (Z(psi), tau) + (u, div_G tau) = 0  =>  Z(psi) = grad_G u - (div_G n) u n,  |div_G n| = |2H|
                H2 = mobius_mean_curvature_x2(x)
                funcs["Zn_pred"] = (abs(ufl.dot(Z, nrm)) - abs(H2) * u) ** 2 * dxe
        funcs["egu"] = ufl.dot(g_ex - ufl.grad(u), g_ex - ufl.grad(u)) * dxe
        self.funcs = {k: form(f) for k, f in funcs.items()}
        # distance of an iterate (w) to a reference state (w_ref), e.g. the discrete limit S -> S_max
        self.w_ref = Function(W)
        psi_r, u_r = ufl.split(self.w_ref)
        Z_r = psi_r / ufl.sqrt(1 + ufl.dot(psi_r, psi_r))
        self.ref_forms = {"du_ref": form((u - u_r) ** 2 * dxe), "dZ_ref": form(ufl.dot(Z - Z_r, Z - Z_r) * dxe)}
        self.uI = Function(U)
        self.uI.interpolate(Expression(u_ex, U.element.interpolation_points))
        self.uh = Function(U)

    # ---- helpers ----------------------------------------------------------------------
    def _make_ksp(self):
        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(self.A)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType(self.factor)
        pc.setFactorSolverType("mumps")
        pc.setFactorSetUpSolverType()
        pc.getFactorMatrix().setMumpsIcntl(14, self.icntl14)
        return ksp

    def _scalar(self, key):
        return self.comm.allreduce(dolfinx.fem.assemble_scalar(self.funcs[key]), op=MPI.SUM)

    def _residual(self):
        with self.b.localForm() as bl:
            bl.set(0.0)
        dolfinx.fem.petsc.assemble_vector(self.b, self.F_form)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        for bc in self.bcs:  # homogeneous Dirichlet data, kept exactly by every iterate
            bc.set(self.b.array_w, alpha=0.0)
        return self.b.norm()


    def _newton_direction(self, quasi):
        """Solve J dw = -F with MUMPS.  If the factorisation fails (|psi| large makes the psi-block
        of J nearly singular along psi: eigenvalues ~ (1+|psi|^2)^{-3/2}), retry with more MUMPS
        workspace, then with the eps-regularised Jacobian J + eps (dpsi, tau)."""
        rnorm = None
        for attempt, use_q in enumerate([quasi, quasi, True]):
            self.A.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix(self.A, self.Jq_form if use_q else self.J_form, bcs=self.bcs)
            self.A.assemble()
            rnorm = self._residual()
            self.b.scale(-1.0)
            self.ksp.solve(self.b, self.dw.x.petsc_vec)
            self.dw.x.scatter_forward()
            if self.ksp.getConvergedReason() >= 0:
                return rnorm
            Fm = self.ksp.getPC().getFactorMatrix()
            info = (Fm.getMumpsInfog(1), Fm.getMumpsInfog(2))
            if self.comm.rank == 0:
                print(f"    MUMPS failure INFOG(1:2)={info} (attempt {attempt}, quasi={use_q})", flush=True)
            self.icntl14 = min(2 * self.icntl14, 1600)  # bounded: memory grows with ICNTL(14)
            self.ksp.destroy()
            self.ksp = self._make_ksp()
        raise RuntimeError("linear solve failed")

    def _audit_residual(self):
        """Check the same subproblem with higher quadrature, without changing it."""
        with self.b_audit.localForm() as entries:
            entries.set(0.0)
        dolfinx.fem.petsc.assemble_vector(self.b_audit, self.F_audit_form)
        self.b_audit.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        for bc in self.bcs:
            bc.set(self.b_audit.array_w, alpha=0.0)
        return self.b_audit.norm()

    def _l2_u(self, arr):
        """L2 norm of a scalar coefficient array, via the mass form."""
        self.uh.x.array[:] = arr
        if not hasattr(self, "_m_form"):
            self._m_form = form(self.uh**2 * ufl.dx(domain=self.mesh))
        return np.sqrt(self.comm.allreduce(dolfinx.fem.assemble_scalar(self._m_form), op=MPI.SUM))

    def diagnostics(self):
        vol = self._scalar("vol")
        d = {
            "e_u_L2": np.sqrt(self._scalar("eu")),
            "e_grad_L2": np.sqrt(self._scalar("es")),
            "e_u_L2_loc": np.sqrt(self._scalar("eu_loc")),
            "e_grad_L2_loc": np.sqrt(self._scalar("es_loc")),
            "psi_rms": np.sqrt(self._scalar("psi2") / vol),
            "J_h": self._scalar("int_u"),
            "J_ex": self._scalar("int_uex"),
            "curvature_L2": np.sqrt(self._scalar("curvature2")),
            "curvature_vector_error_L2": np.sqrt(self._scalar("curvature_error2")),
        }
        uvals = self.w.x.array[self.u_map]
        nloc = self.U.dofmap.index_map.size_local * self.U.dofmap.index_map_bs
        d["e_u_inf_nodal"] = self.comm.allreduce(
            np.max(np.abs(uvals[:nloc] - self.uI.x.array[:nloc]), initial=0.0), op=MPI.MAX
        )
        d["max_u"] = self.comm.allreduce(np.max(uvals[:nloc], initial=-np.inf), op=MPI.MAX)
        if "Zn" in self.funcs:
            d["Z_normal_L2"] = np.sqrt(self._scalar("Zn"))
        if "Zn_pred" in self.funcs:
            d["Z_normal_minus_2Hu_L2"] = np.sqrt(self._scalar("Zn_pred"))
        if "egu" in self.funcs:
            d["e_gradu_L2"] = np.sqrt(self._scalar("egu"))
        return d

    # ---- protocols --------------------------------------------------------------------
    def run_paper(self, tol=1e-4, maxk=100, damping=0.8, max_newton=5, alpha0=10.0, alpha_cap=50.0,
                  log=print):
        """Mirror of MFEM ex40.cpp (Taylor-Hood variant) / Sec. 5.1 of the paper."""
        self.w.x.array[:] = 0.0
        self.w_old.x.array[:] = 0.0
        hist, S, inc, total_newton = [], 0.0, 0.1, 0
        for k in range(1, maxk + 1):
            a_k = min(alpha0 * 2 ** (k - 1), alpha_cap)
            self.alpha.value = a_k
            S += a_k
            t0 = time.perf_counter()
            u_prev = self.w_old.x.array[self.u_map].copy()
            for j in range(max_newton):
                self._newton_direction(quasi=True)
                total_newton += 1
                u_new = self.w.x.array[self.u_map] + self.dw.x.array[self.u_map]
                upd = self._l2_u(u_new - u_prev)
                u_prev = u_new
                self.w.x.array[self.psi_map] += damping * self.dw.x.array[self.psi_map]
                self.w.x.array[self.u_map] = u_new
                if upd < inc:
                    break
            inc = np.sqrt(self._scalar("du"))
            rec = {"k": k, "alpha": a_k, "S": S, "newton": j + 1, "increment": inc,
                   "time": time.perf_counter() - t0}
            rec.update(self.diagnostics())
            hist.append(rec)
            log(_fmt(rec))
            self.w_old.x.array[:] = self.w.x.array
            if inc < tol:
                break
            # |psi^k| ~ S_k |q| with |q| = O(1) for a converging iteration; far larger means breakdown
            if not np.isfinite(inc) or rec["psi_rms"] > 20.0 * S:
                log("  LVPP iterates blew up; stopping")
                return hist, {"outer": k, "newton_total": total_newton, "failed": True}
        return hist, {"outer": k, "newton_total": total_newton, "failed": inc >= tol}

    def _newton_exact(self, rtol=1e-10, atol=1e-12, maxit=80, log=print):
        r0 = self._residual()
        r = r0
        for it in range(maxit + 1):
            if r <= max(atol, rtol * r0) and it > 0:
                return True, it, r
            try:
                self._newton_direction(quasi=False)
            except RuntimeError:
                return False, it + 1, r
            self.w_bak.x.array[:] = self.w.x.array
            t, accepted = 1.0, False
            while t >= 2.0**-12:
                self.w.x.array[:] = self.w_bak.x.array + t * self.dw.x.array
                rn = self._residual()
                if np.isfinite(rn) and rn <= (1 - 1e-4 * t) * r:
                    accepted = True
                    break
                t *= 0.5
            if not accepted:
                self.w.x.array[:] = self.w_bak.x.array
                # residual stagnation at round-off level counts as converged
                return (r <= 1e3 * max(atol, rtol * r0)), it + 1, r
            r = rn
        return False, maxit, r

    def run_exact(self, S_max=2e4, alpha0=10.0, growth=2.0, maxk=80, log=print):
        self.w.x.array[:] = 0.0
        self.w_old.x.array[:] = 0.0
        hist, S, k, total_newton = [], 0.0, 0, 0
        snaps = []
        a_k = alpha0
        while S < S_max * (1 - 1e-12) and k < maxk:
            k += 1
            a_try = min(a_k, S_max - S)
            for attempt in range(4):
                self.alpha.value = a_try
                self.w.x.array[:] = self.w_old.x.array
                t0 = time.perf_counter()
                ok, its, res = self._newton_exact(log=log)
                total_newton += its
                if ok:
                    break
                log(f"  k={k}: Newton failed for alpha={a_try:.3g} (res {res:.2e}); halving")
                a_try *= 0.5
            if not ok:
                log("  giving up")
                break
            S += a_try
            inc = np.sqrt(self._scalar("du"))
            rec = {"k": k, "alpha": a_try, "S": S, "newton": its, "increment": inc, "residual": res,
                   "residual_audit": self._audit_residual(),
                   "time": time.perf_counter() - t0}
            rec.update(self.diagnostics())
            hist.append(rec)
            log(_fmt(rec))
            self.w_old.x.array[:] = self.w.x.array
            snaps.append(self.w.x.array.copy())
            a_k = a_try * growth
        # proximal error on the fixed mesh: distance of each iterate to the final (S = S_max) iterate
        if snaps:
            self.w_ref.x.array[:] = snaps[-1]
            for rec, arr in zip(hist, snaps):
                self.w.x.array[:] = arr
                for key, f in self.ref_forms.items():
                    rec[key] = float(np.sqrt(self.comm.allreduce(dolfinx.fem.assemble_scalar(f), op=MPI.SUM)))
            self.w.x.array[:] = snaps[-1]
        return hist, {"outer": k, "newton_total": total_newton, "failed": bool(S < S_max * (1 - 1e-12))}

    # ---- output -----------------------------------------------------------------------
    def save_fields(self, prefix, gdeg):
        mesh = self.mesh
        gdim = mesh.geometry.dim
        Vo = dolfinx.fem.functionspace(mesh, ("DG", gdeg))
        Qo = dolfinx.fem.functionspace(mesh, ("DG", gdeg, (gdim,)))
        uo, Zo, eo = Function(Vo, name="u_h"), Function(Qo, name="Z(psi_h)"), Function(Vo, name="u_ex-u_h")
        psi, u = ufl.split(self.w)
        uo.interpolate(Expression(u, Vo.element.interpolation_points))
        Zo.interpolate(Expression(psi / ufl.sqrt(1 + ufl.dot(psi, psi)), Qo.element.interpolation_points))
        x = ufl.SpatialCoordinate(mesh)
        u_ex, _, _ = exact_solution(self._domain, x)
        eo.interpolate(Expression(u_ex - u, Vo.element.interpolation_points))
        with dolfinx.io.VTXWriter(mesh.comm, f"{prefix}_u.bp", [uo, eo], engine="BP4") as f:
            f.write(0.0)
        with dolfinx.io.VTXWriter(mesh.comm, f"{prefix}_grad.bp", [Zo], engine="BP4") as f:
            f.write(0.0)
        # cell-wise nodal data for matplotlib (serial only)
        if mesh.comm.size == 1:
            gx = mesh.geometry.x[mesh.geometry.dofmap][:, :, :gdim]
            uv = uo.x.array[Vo.dofmap.list]
            ev = eo.x.array[Vo.dofmap.list]
            np.savez_compressed(f"{prefix}_nodal.npz", x=gx, u=uv, err=ev)


def _fmt(r):
    s = (f"k={r['k']:3d} a={r['alpha']:8.3g} S={r['S']:9.4g} nwt={r['newton']:2d} "
         f"inc={r['increment']:.3e} eu={r['e_u_L2']:.3e} eg={r['e_grad_L2']:.3e} "
         f"einf={r['e_u_inf_nodal']:.3e} maxu={r['max_u']:.5f} |psi|rms={r['psi_rms']:.3e}")
    if "Z_normal_L2" in r:
        s += f" |Z.n|={r['Z_normal_L2']:.3e}"
    if "Z_normal_minus_2Hu_L2" in r:
        s += f" ||Z.n|-|2H|u|={r['Z_normal_minus_2Hu_L2']:.3e}"
    if "e_gradu_L2" in r:
        s += f" |grad(u-uh)|={r['e_gradu_L2']:.3e}"
    return s


DEFAULTS = {
    "mobius": {"disc": "TH", "qdeg": 16, "qdeg_err": 20, "gdeg": 2},
}


def make_mesh(comm, domain, n, variant="quad", gdeg=2):
    if domain != "mobius" or variant != "quad":
        raise ValueError("Only the quadrilateral Mobius surface mesh is supported")
    return mobius_mesh(comm, n, gdeg=gdeg)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", choices=["mobius"], default="mobius")
    p.add_argument("--n", type=int, required=True, help="Mobius mesh refinement index")
    p.add_argument("--protocol", choices=["paper", "exact"], default="exact")
    p.add_argument("--disc", choices=["TH", "TH-curvature"], default=None,
                   help="original divergence form or explicit curvature correction")
    p.add_argument("--S-max", type=float, default=1e4)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--delta", type=float, default=0.1, help="exclusion width for local errors")
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--save-fields", action="store_true")
    p.add_argument("--factor", choices=["cholesky", "lu"], default="cholesky")
    p.add_argument("--maxk", type=int, default=100)
    p.add_argument("--mesh", choices=["quad"], default="quad",
                   help="quadrilateral Mobius surface mesh")
    p.add_argument("--geometry-degree", type=int, choices=[2, 3], default=2)
    p.add_argument("--quadrature-degree", type=int, default=16)
    p.add_argument("--error-quadrature-degree", type=int, default=20)
    a = p.parse_args()

    comm = MPI.COMM_WORLD
    cfg = dict(DEFAULTS[a.domain])
    if a.disc:
        cfg["disc"] = a.disc
    cfg["gdeg"] = a.geometry_degree
    cfg["qdeg"] = a.quadrature_degree
    cfg["qdeg_err"] = a.error_quadrature_degree
    log = print if comm.rank == 0 else (lambda *x, **y: None)

    t0 = time.perf_counter()
    mesh = make_mesh(comm, a.domain, a.n, a.mesh, cfg["gdeg"])
    tdim = mesh.topology.dim
    ncells = mesh.topology.index_map(tdim).size_global
    mesh.topology.create_entities(tdim)
    nloc = mesh.topology.index_map(tdim).size_local
    hmax = comm.allreduce(
        np.max(dolfinx.cpp.mesh.h(mesh._cpp_object, tdim, np.arange(nloc, dtype=np.int32)), initial=0.0),
        op=MPI.MAX,
    )
    solver = EikonalLVPP(mesh, a.domain, cfg["disc"], cfg["qdeg"], cfg["qdeg_err"], delta=a.delta,
                         factor=a.factor)
    solver._domain = a.domain
    log(f"[{a.domain} n={a.n} {cfg['disc']} {a.protocol}] cells={ncells} dofs={solver.ndofs} "
        f"hmax={hmax:.4f} setup={time.perf_counter() - t0:.1f}s")

    t1 = time.perf_counter()
    if a.protocol == "paper":
        hist, summ = solver.run_paper(tol=a.tol, maxk=a.maxk, log=log)
    else:
        hist, summ = solver.run_exact(S_max=a.S_max, log=log)
    summ.update({"domain": a.domain, "mesh": a.mesh, "n": a.n, "disc": cfg["disc"], "protocol": a.protocol,
                 "geometry_degree": cfg["gdeg"], "quadrature_degree": cfg["qdeg"],
                 "error_quadrature_degree": cfg["qdeg_err"],
                 "audit_quadrature_degree": solver.audit_quadrature_degree,
                 "cells": ncells, "dofs": solver.ndofs, "hmax": hmax, "h_nominal": 1.0 / a.n,
                 "solve_time": time.perf_counter() - t1, "nprocs": comm.size, "delta": a.delta,
                 "S_max": a.S_max if a.protocol == "exact" else None})
    log(f"done: outer={summ['outer']} newton={summ['newton_total']} time={summ['solve_time']:.1f}s")

    a.out.mkdir(parents=True, exist_ok=True)
    dom_tag = a.domain if a.mesh == "quad" else f"{a.domain}-{a.mesh}"
    tag = f"{dom_tag}_{cfg['disc']}_{a.protocol}_n{a.n}"
    if comm.rank == 0:
        with open(a.out / f"{tag}.json", "w") as f:
            json.dump({"summary": summ, "history": hist}, f, indent=1, default=float)
    if a.save_fields:
        solver.save_fields(str(a.out / tag), cfg["gdeg"])


if __name__ == "__main__":
    main()
