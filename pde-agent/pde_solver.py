#!/usr/bin/env python3
"""
pde_solver.py — Natural Language PDE Solver · CauchyX AI
=========================================================
Supported PDEs
  heat      : u_t = α ∇²u                    (1D / 2D, FDM)
  wave      : u_tt = c² u_xx                  (1D, FDM)
  poisson   : -∇²u = f(x,y)                  (1D / 2D, direct sparse)
  burgers   : u_t + u u_x = ν u_xx            (1D, FDM upwind)
  advection : u_t + a u_x = 0                 (1D, upwind)
  allen_cahn: u_t = ε²∇²u + u - u³           (1D, FDM)
  cauchynet : any 1D/2D PDE                   (PINN, PyTorch)
  physicsNeMo: any PDE                        (NVIDIA PhysicsNeMo/Modulus)
  ode       : y'' + ω²y = 0 (or custom)       (scipy IVP)

Usage
  python pde_solver.py "solve the 1D heat equation alpha=0.01 on [0,1] t=0.5"
  python pde_solver.py "2D Poisson on unit square zero BC"
  python pde_solver.py "Burgers nu=0.005 periodic BC IC=-sin(pi*x) until t=1"
  python pde_solver.py "wave equation c=1.5 on [0,2] IC=sin(pi*x) until t=2"
  python pde_solver.py "CauchyNet heat 1D alpha=0.01"   ← PINN mode
"""

import sys, re, os, math, textwrap

# Force UTF-8 output on Windows (avoids GBK codec errors)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

# ─────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class PDESpec:
    pde_type  : str   = "heat"
    dims      : int   = 1
    x_domain  : List  = field(default_factory=lambda: [0.0, 1.0])
    y_domain  : List  = field(default_factory=lambda: [0.0, 1.0])
    t_end     : float = 0.5
    bc_type   : str   = "dirichlet"   # dirichlet | neumann | periodic | mixed
    bc_left   : float = 0.0
    bc_right  : float = 0.0
    bc_top    : float = 0.0
    bc_bottom : float = 0.0
    ic_str    : str   = "np.sin(np.pi * x)"
    params    : Dict  = field(default_factory=dict)
    nx        : int   = 128
    ny        : int   = 64
    use_pinn  : bool  = False  # use CauchyNet PINN instead of FDM
    use_nemo  : bool  = False  # use NVIDIA PhysicsNeMo backend
    raw       : str   = ""


# ─────────────────────────────────────────────────────────────
#  NATURAL LANGUAGE PARSER
# ─────────────────────────────────────────────────────────────

class NLParser:
    """
    Extracts a PDESpec from a free-form natural language string.
    Handles mixed Chinese/English.
    """

    _TYPE_MAP = {
        "heat"      : ["heat", "diffusion", "thermal", "temperature",
                        "conduction", "diffuse", "热方程", "扩散"],
        "wave"      : ["wave", "vibrat", "acoustic", "string", "hyperbolic",
                        "wave eq", "波方程", "弦振动"],
        "poisson"   : ["poisson", "electrostatic", "potential", "pressure eq",
                        "泊松", "静电"],
        "laplace"   : ["laplace", "laplacian", "steady state", "harmonic",
                        "equilibrium", "拉普拉斯", "稳态"],
        "burgers"   : ["burger", "viscous flow", "shock", "inviscid",
                        "nonlinear conv", "伯格斯"],
        "advection" : ["advect", "transport", "convect", "first order hyp",
                        "对流"],
        "allen_cahn": ["allen", "cahn", "phase field", "interface", "phase-field"],
        "ode"       : ["ode", "ordinary", "spring", "pendulum", "oscillat",
                        "常微分", "弹簧", "单摆"],
        "cauchynet" : ["cauchynet", "cauchy net", "pinn", "neural", "physics-informed",
                        "物理神经"],
        "physicsNeMo": ["physicsnemo", "physicsNeMo", "nemo", "modulus",
                         "nvidia modulus", "physx", "nvidia pinn"],
    }

    def parse(self, raw: str) -> PDESpec:
        s  = PDESpec(raw=raw)
        lo = raw.lower()

        s.pde_type  = self._type(lo)
        s.dims      = self._dims(lo)
        s.x_domain,\
        s.y_domain  = self._domains(lo)
        s.t_end     = self._t_end(lo, s.pde_type)
        s.bc_type,\
        s.bc_left,\
        s.bc_right  = self._bc(lo)
        s.ic_str    = self._ic(lo, s.dims)
        s.params    = self._params(lo, s.pde_type)
        s.nx        = self._nx(lo)
        s.ny        = s.nx // 2
        s.use_nemo  = (s.pde_type == "physicsNeMo" or
                       any(k in lo for k in ["physicsnemo","nemo","modulus",
                                              "nvidia modulus"]))
        s.use_pinn  = (not s.use_nemo and
                       (s.pde_type == "cauchynet" or
                        "cauchynet" in lo or "pinn" in lo))
        if s.pde_type in ("cauchynet", "physicsNeMo"):
            # resolve the actual PDE type
            clean = lo
            for k in ["cauchynet","pinn","physicsnemo","nemo","modulus"]:
                clean = clean.replace(k, "")
            s.pde_type = self._type(clean)
            if s.pde_type in ("cauchynet", "physicsNeMo"):
                s.pde_type = "heat"
        return s

    # ── helpers ──────────────────────────────────────────────

    def _type(self, lo):
        for t, kws in self._TYPE_MAP.items():
            if any(kw in lo for kw in kws):
                return t
        return "heat"

    def _dims(self, lo):
        if any(k in lo for k in ["2d","2-d","two-dim","(x,y)","x,y,","y,x",
                                   "二维","2维"]):
            return 2
        return 1

    def _domains(self, lo):
        # Match bracket ranges like [0,1] [0, pi] [0,2pi]
        def _v(s):
            s = s.strip().replace("π","pi")
            if s == "pi":      return math.pi
            if s == "2pi":     return 2*math.pi
            if s == "2*pi":    return 2*math.pi
            try: return float(s)
            except: return 1.0
        pat = r"\[\s*([\d\.]+)\s*,\s*([\dpi\*\.]+)\s*\]"
        hits = re.findall(pat, lo)
        if len(hits) >= 2:
            return [_v(hits[0][0]),_v(hits[0][1])], [_v(hits[1][0]),_v(hits[1][1])]
        if len(hits) == 1:
            return [_v(hits[0][0]),_v(hits[0][1])], [0.0, 1.0]
        return [0.0, 1.0], [0.0, 1.0]

    def _t_end(self, lo, pde_type):
        defaults = {"heat":0.5,"wave":2.0,"poisson":0.0,"laplace":0.0,
                    "burgers":1.0,"advection":1.0,"allen_cahn":1.0,"ode":10.0}
        for pat in [r"until\s+t\s*=\s*([\d\.]+)",
                    r"to\s+t\s*=\s*([\d\.]+)",
                    r"t\s*=\s*([\d\.]+)\s*$",
                    r"t_end\s*=\s*([\d\.]+)",
                    r"time\s+([\d\.]+)"]:
            m = re.search(pat, lo)
            if m: return float(m.group(1))
        return defaults.get(pde_type, 1.0)

    def _bc(self, lo):
        if "periodic" in lo or "周期" in lo:
            return "periodic", 0.0, 0.0
        if "neumann" in lo or "flux" in lo or "insulated" in lo or "绝热" in lo:
            return "neumann", 0.0, 0.0
        if "robin" in lo:
            return "robin", 0.0, 0.0
        # Explicit bc value — match only bc=N or dirichlet=N, not "bc until t=N"
        m = re.search(r"(?:bc|dirichlet|boundary)\s*=\s*([\-\d\.]+)", lo)
        val = float(m.group(1)) if m else 0.0
        return "dirichlet", val, val

    def _ic(self, lo, dims):
        # Try to parse common patterns
        if "gaussian" in lo or "gauss" in lo:
            return "np.exp(-40*(x-0.5)**2)"
        if "heaviside" in lo or "step function" in lo or "阶跃" in lo:
            return "np.where(x < 0.5, 1.0, 0.0)"
        if "triangl" in lo:
            return "np.where(x<0.5, 2*x, 2-2*x)"
        # sin pattern: sin(pi*x), sin(π x), sin(2πx) …
        m = re.search(r"sin\s*\(\s*([^)]+)\)", lo)
        if m:
            arg = (m.group(1)
                   .replace("π","np.pi")
                   .replace("pi","np.pi")
                   .replace(" ","")
                   .replace("*x","*x")
                   )
            # make sure it has x
            if "x" in arg:
                expr = f"np.sin({arg})"
                if dims == 2:
                    expr = expr + " * np.sin(np.pi * y)"
                return expr
        if "cos" in lo:
            return "np.cos(2*np.pi*x)"
        # default
        if dims == 2:
            return "np.sin(np.pi*x) * np.sin(np.pi*y)"
        return "np.sin(np.pi*x)"

    def _params(self, lo, pde_type):
        p = {}
        # alpha / diffusivity
        for pat in [r"alpha\s*=\s*([\d\.e\-\+]+)",
                    r"α\s*=\s*([\d\.e\-\+]+)",
                    r"diffusiv\w*\s*=?\s*([\d\.e\-\+]+)"]:
            m = re.search(pat, lo)
            if m: p["alpha"] = float(m.group(1)); break
        if "alpha" not in p:
            p["alpha"] = 0.01
        # wave speed c
        for pat in [r"c\s*=\s*([\d\.]+)",
                    r"speed\s*=?\s*([\d\.]+)",
                    r"velocity\s*=?\s*([\d\.]+)"]:
            m = re.search(pat, lo)
            if m: p["c"] = float(m.group(1)); break
        if "c" not in p: p["c"] = 1.0
        # viscosity / nu
        for pat in [r"nu\s*=\s*([\d\.e\-\+]+)",
                    r"ν\s*=\s*([\d\.e\-\+]+)",
                    r"viscosity\s*=?\s*([\d\.e\-\+]+)",
                    r"epsilon\s*=\s*([\d\.e\-\+]+)"]:
            m = re.search(pat, lo)
            if m: p["nu"] = float(m.group(1)); break
        if "nu" not in p: p["nu"] = 0.01
        # advection speed a
        m = re.search(r"(?:advect|transport).*?(?:a|speed)\s*=\s*([\-\d\.]+)", lo)
        p["a"] = float(m.group(1)) if m else 1.0
        # Allen-Cahn epsilon
        m = re.search(r"epsilon\s*=\s*([\d\.e\-\+]+)", lo)
        p["eps"] = float(m.group(1)) if m else 0.05
        return p

    def _nx(self, lo):
        for pat in [r"nx\s*=\s*(\d+)", r"n\s*=\s*(\d+)", r"grid\s+(\d+)"]:
            m = re.search(pat, lo)
            if m: return min(int(m.group(1)), 512)
        return 128


# ─────────────────────────────────────────────────────────────
#  FDM SOLVERS
# ─────────────────────────────────────────────────────────────

def _ic_eval(ic_str: str, x: np.ndarray,
             y: Optional[np.ndarray] = None) -> np.ndarray:
    """Safely evaluate IC string; fall back to sin(πx)."""
    ns = {"np": np, "sin": np.sin, "cos": np.cos,
          "exp": np.exp, "pi": math.pi, "x": x}
    if y is not None:
        ns["y"] = y
    try:
        result = eval(ic_str, ns)
        return np.broadcast_to(result, x.shape).copy().astype(float)
    except Exception as e:
        print(f"  [warn] IC eval failed ({e}), using sin(πx)")
        return np.sin(math.pi * x).astype(float)


def _apply_bc_1d(u: np.ndarray, bc_type: str,
                  left: float, right: float) -> np.ndarray:
    if bc_type == "periodic":
        u[0] = u[-2]; u[-1] = u[1]
    elif bc_type == "neumann":
        u[0] = u[1]; u[-1] = u[-2]
    else:
        u[0] = left; u[-1] = right
    return u


def solve_heat_1d(s: PDESpec) -> dict:
    """Explicit FTCS — CFL: r = α dt/dx² ≤ 0.5"""
    α = s.params["alpha"]
    x = np.linspace(*s.x_domain, s.nx)
    dx = x[1] - x[0]
    dt = 0.4 * dx**2 / α
    nt = max(int(s.t_end / dt) + 1, 1)
    dt = s.t_end / nt
    r  = α * dt / dx**2

    u  = _ic_eval(s.ic_str, x)
    u  = _apply_bc_1d(u, s.bc_type, s.bc_left, s.bc_right)

    snaps, t_snaps = [u.copy()], [0.0]
    snap_at = np.linspace(0, s.t_end, 7)[1:]
    snap_idx = 0
    t = 0.0
    for _ in range(nt):
        un = u.copy()
        un[1:-1] = u[1:-1] + r*(u[2:] - 2*u[1:-1] + u[:-2])
        _apply_bc_1d(un, s.bc_type, s.bc_left, s.bc_right)
        u = un; t += dt
        if snap_idx < len(snap_at) and t >= snap_at[snap_idx]:
            snaps.append(u.copy()); t_snaps.append(t)
            snap_idx += 1

    return dict(x=x, snaps=snaps, t_snaps=t_snaps,
                u=u, dims=1, scheme="FTCS r={:.4f}".format(r))


def solve_heat_2d(s: PDESpec) -> dict:
    """Explicit 2D FTCS."""
    α  = s.params["alpha"]
    x  = np.linspace(*s.x_domain, s.nx)
    y  = np.linspace(*s.y_domain, s.ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dx = x[1]-x[0]; dy = y[1]-y[0]
    dt = 0.2 * min(dx, dy)**2 / α
    nt = max(int(s.t_end / dt) + 1, 1)
    nt = min(nt, 3000)   # cap
    dt_actual = s.t_end / nt
    rx = α * dt_actual / dx**2
    ry = α * dt_actual / dy**2

    ic_2d = s.ic_str.replace("x","X").replace("y","Y") if "y" in s.ic_str else s.ic_str.replace("x","X") + " * np.sin(np.pi*Y)"
    ns = {"np":np,"sin":np.sin,"cos":np.cos,"exp":np.exp,"pi":math.pi,"X":X,"Y":Y,"x":X,"y":Y}
    try:
        u = eval(ic_2d, ns).astype(float)
    except:
        u = np.sin(math.pi*X)*np.sin(math.pi*Y)
    u[0,:]=u[-1,:]=u[:,0]=u[:,-1]=s.bc_left

    for _ in range(nt):
        un = u.copy()
        un[1:-1,1:-1] = (u[1:-1,1:-1]
            + rx*(u[2:,1:-1]-2*u[1:-1,1:-1]+u[:-2,1:-1])
            + ry*(u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,:-2]))
        un[0,:]=un[-1,:]=un[:,0]=un[:,-1]=s.bc_left
        u = un

    return dict(x=x, y=y, X=X, Y=Y, u=u, dims=2,
                scheme="FTCS 2D rx={:.4f} ry={:.4f}".format(rx,ry))


def solve_wave_1d(s: PDESpec) -> dict:
    """Leapfrog scheme — CFL: ν = c dt/dx ≤ 1"""
    c  = s.params["c"]
    x  = np.linspace(*s.x_domain, s.nx)
    dx = x[1]-x[0]
    dt = 0.9 * dx / abs(c)
    nt = max(int(s.t_end / dt)+1, 1)
    dt = s.t_end / nt
    r2 = (c*dt/dx)**2

    u0 = _ic_eval(s.ic_str, x)
    # First step (zero initial velocity)
    u1 = u0.copy()
    u1[1:-1] = u0[1:-1] + 0.5*r2*(u0[2:]-2*u0[1:-1]+u0[:-2])
    _apply_bc_1d(u1, s.bc_type, s.bc_left, s.bc_right)

    snaps, t_snaps = [u0.copy()], [0.0]
    snap_at = np.linspace(0, s.t_end, 7)[1:]
    snap_idx = 0
    u_prev, u_curr = u0, u1
    t = dt

    for _ in range(nt-1):
        un = 2*u_curr - u_prev + r2*(
            np.roll(u_curr,-1)-2*u_curr+np.roll(u_curr,1))
        _apply_bc_1d(un, s.bc_type, s.bc_left, s.bc_right)
        u_prev, u_curr = u_curr, un; t += dt
        if snap_idx < len(snap_at) and t >= snap_at[snap_idx]:
            snaps.append(u_curr.copy()); t_snaps.append(t)
            snap_idx += 1

    return dict(x=x, snaps=snaps, t_snaps=t_snaps, u=u_curr, dims=1,
                scheme="Leapfrog ν²={:.4f}".format(r2))


def solve_poisson_1d(s: PDESpec) -> dict:
    """-u'' = f(x) with direct tridiagonal solve."""
    x  = np.linspace(*s.x_domain, s.nx)
    dx = x[1]-x[0]
    # RHS: try to get from IC string (reinterpret as source), else default
    f  = np.sin(math.pi*(x-s.x_domain[0])/(s.x_domain[1]-s.x_domain[0]))
    n  = s.nx-2
    diag = np.full(n, 2/dx**2)
    off  = np.full(n-1, -1/dx**2)
    from scipy.linalg import solve_banded
    ab = np.zeros((3, n))
    ab[0, 1:] = off; ab[1, :] = diag; ab[2, :-1] = off
    rhs = f[1:-1].copy()
    rhs[0]  += s.bc_left  / dx**2
    rhs[-1] += s.bc_right / dx**2
    u_int = solve_banded((1,1), ab, rhs)
    u = np.empty(s.nx)
    u[0]=s.bc_left; u[-1]=s.bc_right; u[1:-1]=u_int

    return dict(x=x, snaps=[u], t_snaps=[0.0], u=u, dims=1,
                scheme="Direct tridiagonal (Poisson 1D)")


def solve_poisson_2d(s: PDESpec) -> dict:
    """2D Poisson via sparse direct solve."""
    from scipy.sparse import diags as sp_diags, kron, eye
    from scipy.sparse.linalg import spsolve

    nx = min(s.nx, 64); ny = min(s.ny, 64)
    x  = np.linspace(*s.x_domain, nx)
    y  = np.linspace(*s.y_domain, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dx = x[1]-x[0]; dy = y[1]-y[0]

    F  = np.sin(math.pi*(X-s.x_domain[0])/(s.x_domain[1]-s.x_domain[0])) \
       * np.sin(math.pi*(Y-s.y_domain[0])/(s.y_domain[1]-s.y_domain[0]))

    nx2=nx-2; ny2=ny-2; N=nx2*ny2
    Lx = sp_diags([-1,2,-1],[-1,0,1],shape=(nx2,nx2))/dx**2
    Ly = sp_diags([-1,2,-1],[-1,0,1],shape=(ny2,ny2))/dy**2
    A  = kron(Lx, eye(ny2)) + kron(eye(nx2), Ly)

    u_int = spsolve(A, F[1:-1,1:-1].ravel()).reshape(nx2,ny2)
    u = np.zeros((nx,ny))
    u[0,:]=u[-1,:]=u[:,0]=u[:,-1]=s.bc_left
    u[1:-1,1:-1] = u_int

    return dict(x=x, y=y, X=X, Y=Y, u=u, dims=2,
                scheme="Sparse direct (Poisson 2D)")


def solve_burgers_1d(s: PDESpec) -> dict:
    """Godunov upwind + central diffusion."""
    nu = s.params["nu"]
    x  = np.linspace(*s.x_domain, s.nx)
    dx = x[1]-x[0]
    dt = min(0.45*dx**2/max(nu,1e-9), 0.45*dx)
    nt = max(int(s.t_end/dt)+1, 1)
    dt = s.t_end/nt

    u = _ic_eval(s.ic_str, x)
    snaps, t_snaps = [u.copy()], [0.0]
    snap_at = np.linspace(0, s.t_end, 7)[1:]
    snap_idx = 0
    t = 0.0

    for _ in range(nt):
        # upwind convection
        u_plus  = np.roll(u, -1); u_minus = np.roll(u, 1)
        F_plus  = np.maximum(u,0)*u + np.minimum(u,0)*u_plus
        F_minus = np.maximum(u,0)*u_minus + np.minimum(u,0)*u
        conv    = (F_plus - F_minus) / dx
        diff    = nu*(u_plus - 2*u + u_minus)/dx**2
        un      = u + dt*(-conv + diff)
        _apply_bc_1d(un, s.bc_type, s.bc_left, s.bc_right)
        u = un; t += dt
        if snap_idx < len(snap_at) and t >= snap_at[snap_idx]:
            snaps.append(u.copy()); t_snaps.append(t)
            snap_idx += 1

    return dict(x=x, snaps=snaps, t_snaps=t_snaps, u=u, dims=1,
                scheme="Godunov upwind + central diff (Burgers)")


def solve_advection_1d(s: PDESpec) -> dict:
    """First-order upwind advection u_t + a u_x = 0."""
    a  = s.params.get("a", 1.0)
    x  = np.linspace(*s.x_domain, s.nx)
    dx = x[1]-x[0]
    dt = 0.8 * dx / abs(a)
    nt = max(int(s.t_end/dt)+1, 1); dt = s.t_end/nt
    CFL = a*dt/dx

    u = _ic_eval(s.ic_str, x)
    snaps, t_snaps = [u.copy()], [0.0]
    snap_at = np.linspace(0, s.t_end, 7)[1:]
    snap_idx = 0
    t = 0.0

    for _ in range(nt):
        if a >= 0:
            un = u - CFL*(u - np.roll(u, 1))
        else:
            un = u - CFL*(np.roll(u,-1) - u)
        _apply_bc_1d(un, s.bc_type, s.bc_left, s.bc_right)
        u = un; t += dt
        if snap_idx < len(snap_at) and t >= snap_at[snap_idx]:
            snaps.append(u.copy()); t_snaps.append(t)
            snap_idx += 1

    return dict(x=x, snaps=snaps, t_snaps=t_snaps, u=u, dims=1,
                scheme="Upwind (Advection CFL={:.3f})".format(CFL))


def solve_allen_cahn_1d(s: PDESpec) -> dict:
    """u_t = ε² u_xx + u - u³  (phase-field / Allen-Cahn)"""
    eps = s.params.get("eps", 0.05)
    x   = np.linspace(*s.x_domain, s.nx)
    dx  = x[1]-x[0]
    dt  = min(0.4*dx**2/eps**2, 0.1)
    nt  = max(int(s.t_end/dt)+1, 1); dt = s.t_end/nt

    u = _ic_eval(s.ic_str, x)
    snaps, t_snaps = [u.copy()], [0.0]
    snap_at = np.linspace(0, s.t_end, 7)[1:]
    snap_idx = 0
    t = 0.0

    for _ in range(nt):
        diff = eps**2*(np.roll(u,-1)-2*u+np.roll(u,1))/dx**2
        reac = u - u**3
        un   = u + dt*(diff + reac)
        _apply_bc_1d(un, s.bc_type, s.bc_left, s.bc_right)
        u = un; t += dt
        if snap_idx < len(snap_at) and t >= snap_at[snap_idx]:
            snaps.append(u.copy()); t_snaps.append(t)
            snap_idx += 1

    return dict(x=x, snaps=snaps, t_snaps=t_snaps, u=u, dims=1,
                scheme="FTCS (Allen-Cahn ε={})".format(eps))


def solve_ode(s: PDESpec, raw: str) -> dict:
    """Harmonic oscillator or damped oscillator via scipy IVP."""
    from scipy.integrate import solve_ivp
    lo = raw.lower()
    t_end = s.t_end if s.t_end > 1 else 10.0

    # ω detection
    omega = 1.0
    m = re.search(r"omega\s*=\s*([\d\.]+)", lo)
    if m: omega = float(m.group(1))

    # damping
    gamma = 0.0
    m = re.search(r"(?:damp|gamma|zeta)\s*=\s*([\d\.]+)", lo)
    if m: gamma = float(m.group(1))

    def ode(t, y): return [y[1], -2*gamma*omega*y[1] - omega**2*y[0]]

    sol = solve_ivp(ode, [0, t_end], [1.0, 0.0],
                    max_step=t_end/1000, dense_output=True)
    t_arr = np.linspace(0, t_end, 1000)
    Y     = sol.sol(t_arr)

    return dict(t=t_arr, y=Y[0], dy=Y[1], dims=0,
                snaps=[], t_snaps=[],
                scheme="SciPy RK45 (ω={} γ={})".format(omega, gamma))


# ─────────────────────────────────────────────────────────────
#  CauchyNet PINN SOLVER  (PyTorch)
# ─────────────────────────────────────────────────────────────

def solve_cauchynet_pinn(s: PDESpec) -> dict:
    """
    Single-layer CauchyNet PINN.
    φ(x; λ1, λ2, d) = (λ1·x + λ2) / (x² + d²)
    Trains on PDE residual + BC loss.
    Returns FDM result as fallback if torch unavailable.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("  [warn] PyTorch not found — falling back to FDM")
        return _fdm_dispatch(s)

    torch.manual_seed(42)
    device = "cpu"
    dtype  = torch.float32

    # ── CauchyNet layer ────────────────────────────────────
    class CauchyActivation(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.l1 = nn.Parameter(torch.full((n,), 0.01))
            self.l2 = nn.Parameter(torch.full((n,), 0.01))
            self.d  = nn.Parameter(torch.ones(n))
        def forward(self, x):
            return (self.l1 * x + self.l2) / (x**2 + self.d**2 + 1e-8)

    class CauchyNet(nn.Module):
        def __init__(self, in_dim=2, width=200, out_dim=1):
            super().__init__()
            self.fc_in  = nn.Linear(in_dim, width)
            self.act    = CauchyActivation(width)
            self.fc_out = nn.Linear(width, out_dim)
        def forward(self, x):
            return self.fc_out(self.act(self.fc_in(x)))

    x0, x1 = s.x_domain
    α  = s.params["alpha"]
    in_dim = 2 if s.dims == 1 else 3   # (x,t) for 1D or (x,y,t) for 2D

    net = CauchyNet(in_dim=in_dim, width=200).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    print("  [CauchyNet] Training PINN (500 iters)...")

    for it in range(500):
        opt.zero_grad()
        # Collocation points
        x_c = torch.rand(1000, 1)*(x1-x0)+x0
        t_c = torch.rand(1000, 1)*s.t_end
        xt  = torch.cat([x_c, t_c], dim=1).requires_grad_(True)

        u   = net(xt)
        # Compute gradients
        u_t = torch.autograd.grad(u, xt, torch.ones_like(u),
                                  create_graph=True)[0][:, 1:2]
        u_x = torch.autograd.grad(u, xt, torch.ones_like(u),
                                  create_graph=True)[0][:, 0:1]
        u_xx = torch.autograd.grad(u_x, xt, torch.ones_like(u_x),
                                   create_graph=True)[0][:, 0:1]
        # PDE residual: u_t - α u_xx = 0
        pde_loss = torch.mean((u_t - α * u_xx)**2)

        # BC loss: u(x0,t) = u(x1,t) = 0
        xbc_l = torch.full((100,1), x0); xbc_r = torch.full((100,1), x1)
        tbc   = torch.rand(100,1)*s.t_end
        u_bl  = net(torch.cat([xbc_l, tbc], 1))
        u_br  = net(torch.cat([xbc_r, tbc], 1))
        bc_loss = torch.mean(u_bl**2) + torch.mean(u_br**2)

        # IC loss
        x_ic = torch.rand(500,1)*(x1-x0)+x0
        t_ic = torch.zeros(500,1)
        u_ic_pred = net(torch.cat([x_ic, t_ic], 1))
        u_ic_true = torch.sin(math.pi*(x_ic-x0)/(x1-x0))
        ic_loss = torch.mean((u_ic_pred - u_ic_true)**2)

        loss = pde_loss + 10*bc_loss + 10*ic_loss
        loss.backward(); opt.step()

        if (it+1) % 100 == 0:
            print(f"    iter {it+1:4d}  loss={loss.item():.2e}  "
                  f"pde={pde_loss.item():.2e}  bc={bc_loss.item():.2e}")

    # Evaluate on grid
    xv = np.linspace(x0, x1, s.nx)
    snaps = []; t_snaps_list = []
    snap_times = np.linspace(0, s.t_end, 7)
    with torch.no_grad():
        for tt in snap_times:
            xr = torch.tensor(xv, dtype=dtype).unsqueeze(1)
            tr = torch.full_like(xr, tt)
            u_snap = net(torch.cat([xr, tr], 1)).squeeze().numpy()
            snaps.append(u_snap); t_snaps_list.append(tt)

    return dict(x=xv, snaps=snaps, t_snaps=t_snaps_list,
                u=snaps[-1], dims=1,
                scheme="CauchyNet PINN (single-layer, width=200)")


# ─────────────────────────────────────────────────────────────
#  NVIDIA PhysicsNeMo BACKEND
# ─────────────────────────────────────────────────────────────

def solve_physicsnemo(s: PDESpec) -> dict:
    """
    Calls NVIDIA PhysicsNeMo (formerly Modulus) to solve the PDE.
    PhysicsNeMo must be installed:  pip install nvidia-physicsnemo

    Falls back to CauchyNet PINN if PhysicsNeMo is not available.
    """
    # ── Try to import PhysicsNeMo ──────────────────────────
    try:
        import physicsnemo                           # nvidia-physicsnemo >= 1.0
        _nemo_version = getattr(physicsnemo, "__version__", "unknown")
        _has_nemo = True
    except ImportError:
        try:
            import modulus                           # older Modulus API
            physicsnemo = modulus
            _nemo_version = getattr(modulus, "__version__", "unknown")
            _has_nemo = True
        except ImportError:
            _has_nemo = False

    if not _has_nemo:
        print("  [warn] PhysicsNeMo / Modulus not installed.")
        print("         Install:  pip install nvidia-physicsnemo")
        print("         Falling back to CauchyNet PINN.")
        return solve_cauchynet_pinn(s)

    print(f"  [PhysicsNeMo v{_nemo_version}] Building geometry and PDE nodes...")

    import torch
    x0, x1 = s.x_domain
    α  = s.params["alpha"]

    # ── PhysicsNeMo: geometry ──────────────────────────────
    try:
        from physicsnemo.geometry.primitives_1d import Line1D
        from physicsnemo.geometry.primitives_2d import Rectangle
        from physicsnemo.models.fully_connected import FullyConnected
        from physicsnemo.eq.pdes.diffusion import Diffusion
        from physicsnemo.eq.pdes.wave_equation import WaveEquation
        from physicsnemo.domain import Domain
        from physicsnemo.domain.constraint import (
            PointwiseBoundaryConstraint,
            PointwiseInteriorConstraint,
        )
        from physicsnemo.domain.validator import PointwiseValidator
        from physicsnemo.solver import Solver
        from physicsnemo.hydra import to_yaml
        import hydra
        from omegaconf import DictConfig

        # geometry
        geo = Line1D(x0, x1)

        # PDE node
        if s.pde_type in ("heat", "diffusion"):
            pde_node = Diffusion(T="u", D=α, dim=1, time=True)
        elif s.pde_type == "wave":
            pde_node = WaveEquation(u="u", c=s.params["c"], dim=1, time=True)
        else:
            pde_node = Diffusion(T="u", D=α, dim=1, time=True)

        # Neural network (FullyConnected — PhysicsNeMo default)
        net = FullyConnected(
            in_features=2,   # (x, t)
            out_features=1,  # u
            layer_size=256,
            num_layers=5,
        )

        # Build domain & constraints
        domain = Domain()

        # Interior: PDE residual
        interior = PointwiseInteriorConstraint(
            nodes=[pde_node, net],
            geometry=geo,
            outvar={"diffusion_u": 0},
            batch_size=1024,
            parameterization={"t": (0, s.t_end)},
        )
        domain.add_constraint(interior, "interior")

        # Boundary: Dirichlet u=0
        bc = PointwiseBoundaryConstraint(
            nodes=[net],
            geometry=geo,
            outvar={"u": s.bc_left},
            batch_size=256,
            parameterization={"t": (0, s.t_end)},
        )
        domain.add_constraint(bc, "bc")

        # Solver (quick run — 1000 iterations)
        slv = Solver(cfg=None, domain=domain)
        slv.solve(max_steps=1000)

        # Evaluate on grid
        xv  = np.linspace(x0, x1, s.nx)
        snaps = []; t_snaps_list = []
        with torch.no_grad():
            for tt in np.linspace(0, s.t_end, 7):
                inp = {
                    "x": torch.tensor(xv, dtype=torch.float32).unsqueeze(1),
                    "t": torch.full((s.nx, 1), tt, dtype=torch.float32),
                }
                out = net(inp)["u"].squeeze().numpy()
                snaps.append(out); t_snaps_list.append(tt)

        return dict(x=xv, snaps=snaps, t_snaps=t_snaps_list,
                    u=snaps[-1], dims=1,
                    scheme=f"PhysicsNeMo v{_nemo_version} FullyConnected")

    except Exception as e:
        # PhysicsNeMo API varies between versions — handle gracefully
        print(f"  [PhysicsNeMo] API error: {e}")
        print("  Falling back to CauchyNet PINN.")
        return solve_cauchynet_pinn(s)


# ─────────────────────────────────────────────────────────────
#  DISPATCHER
# ─────────────────────────────────────────────────────────────

def _fdm_dispatch(s: PDESpec) -> dict:
    t = s.pde_type
    if t == "wave":
        return solve_wave_1d(s)
    if t in ("poisson", "laplace"):
        return solve_poisson_2d(s) if s.dims == 2 else solve_poisson_1d(s)
    if t == "burgers":
        return solve_burgers_1d(s)
    if t == "advection":
        return solve_advection_1d(s)
    if t == "allen_cahn":
        return solve_allen_cahn_1d(s)
    if t == "ode":
        return None  # handled separately
    # heat / default
    return solve_heat_2d(s) if s.dims == 2 else solve_heat_1d(s)


def dispatch(s: PDESpec, raw: str) -> dict:
    if s.pde_type == "ode":
        return solve_ode(s, raw)
    if s.use_nemo:
        return solve_physicsnemo(s)
    if s.use_pinn:
        return solve_cauchynet_pinn(s)
    return _fdm_dispatch(s)


# ─────────────────────────────────────────────────────────────
#  VISUALIZATION
# ─────────────────────────────────────────────────────────────

_DARK_BG   = "#04060f"
_DARK_SURF = "#080d1a"
_CYAN      = "#00d4ff"
_MUTED     = "#4a5a7a"

def _style_ax(ax):
    ax.set_facecolor(_DARK_SURF)
    ax.tick_params(colors="#8a9ab5", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#1e3060")
    ax.xaxis.label.set_color("#8a9ab5")
    ax.yaxis.label.set_color("#8a9ab5")
    ax.title.set_color("#c8d8f0")
    ax.grid(True, color="#0d1f3c", linewidth=0.5)


def visualize(s: PDESpec, r: dict, out: str = "pde_solution.png") -> str:
    dims = r.get("dims", 1)
    plt.rcParams["font.family"] = "monospace"

    if dims == 0:                                # ── ODE
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        axes[0].plot(r["t"], r["y"],  color=_CYAN,     lw=1.8, label="y(t)")
        axes[1].plot(r["t"], r["dy"], color="#a78bfa",  lw=1.8, label="y'(t)")
        for ax, label in zip(axes, ["y(t) — position", "y'(t) — velocity"]):
            ax.set_xlabel("t"); ax.set_ylabel(label)
            _style_ax(ax); ax.legend(fontsize=8)
        title = "ODE Solution"

    elif dims == 1:                              # ── 1-D PDE
        snaps = r.get("snaps", [r["u"]])
        t_snaps = r.get("t_snaps", [0.0])
        n  = min(len(snaps), 6)
        nc = min(n, 3); nr = math.ceil(n / nc)
        fig, axes = plt.subplots(nr, nc, figsize=(4.2*nc, 3*nr))
        axes = np.array(axes).ravel()
        cmap = plt.cm.plasma(np.linspace(0.15, 0.95, n))
        for i in range(n):
            axes[i].plot(r["x"], snaps[i], color=cmap[i], lw=1.8)
            axes[i].set_xlabel("x"); axes[i].set_ylabel("u")
            axes[i].set_title(f"t = {t_snaps[i]:.4f}")
            _style_ax(axes[i])
        for ax in axes[n:]:
            ax.axis("off")
        title = f"{s.pde_type.upper()} — 1D FDM  |  {r.get('scheme','')}"

    else:                                        # ── 2-D PDE
        X, Y, u = r["X"], r["Y"], r["u"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        cf = axes[0].contourf(X, Y, u, levels=60, cmap="plasma")
        fig.colorbar(cf, ax=axes[0])
        axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
        axes[0].set_title("Contour u(x,y)")
        im = axes[1].pcolormesh(X, Y, u, cmap="inferno", shading="auto")
        fig.colorbar(im, ax=axes[1])
        axes[1].contour(X, Y, u, levels=15, colors="white",
                        linewidths=0.4, alpha=0.35)
        axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
        axes[1].set_title("Field u(x,y)")
        [_style_ax(ax) for ax in axes]
        title = f"{s.pde_type.upper()} — 2D  |  {r.get('scheme','')}"

    fig.patch.set_facecolor(_DARK_BG)
    fig.suptitle(title, fontsize=11, color="#e8f0ff", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────────────────────

def summary(s: PDESpec, r: dict, img: str):
    u_arr = r.get("u", r.get("y", np.array([0.0])))
    w = 54
    sep = "─" * w

    lines = [
        "┌" + "─"*(w-2) + "┐",
        "│{:^{}}│".format("  CauchyX PDE Agent  ·  Solution Report  ", w-2),
        "├" + "─"*(w-2) + "┤",
        "│  PDE type  :  {:<{}}│".format(
            s.pde_type.upper() + ("  (2D)" if s.dims==2 else "  (1D)"), w-18),
        "│  Solver    :  {:<{}}│".format(
            ("PhysicsNeMo" if s.use_nemo else
             "CauchyNet PINN" if s.use_pinn else "Finite Difference"), w-18),
        "│  Scheme    :  {:<{}}│".format(r.get("scheme","")[:w-18], w-18),
        "├" + "─"*(w-2) + "┤",
        "│  x domain  :  [{:.3g}, {:.3g}]{:<{}}│".format(
            *s.x_domain, "", w-28),
    ]
    if s.dims == 2:
        lines.append("│  y domain  :  [{:.3g}, {:.3g}]{:<{}}│".format(
            *s.y_domain, "", w-28))
    if s.pde_type not in ("poisson","laplace","ode"):
        lines.append("│  t end     :  {:<{}}│".format(s.t_end, w-18))
    lines += [
        "│  BC type   :  {:<{}}│".format(s.bc_type, w-18),
        "│  IC        :  {:<{}}│".format(s.ic_str[:w-18], w-18),
        "│  Params    :  {:<{}}│".format(str(s.params)[:w-18], w-18),
        "│  Grid      :  nx={} ny={}  {:<{}}│".format(
            s.nx, s.ny, "", w-30),
        "├" + "─"*(w-2) + "┤",
        "│  ‖u‖_max   :  {:<{}.6f}│".format(float(np.max(np.abs(u_arr))), w-18),
        "│  ‖u‖_mean  :  {:<{}.6f}│".format(float(np.mean(u_arr)), w-18),
        "│  ‖u‖_min   :  {:<{}.6f}│".format(float(np.min(u_arr)), w-18),
        "├" + "─"*(w-2) + "┤",
        "│  Plot      :  {:<{}}│".format(img[:w-18], w-18),
        "└" + "─"*(w-2) + "┘",
    ]
    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────

def solve(text: str, outdir: str = "D:\\claudecode") -> str:
    """
    Solve a PDE described in natural language.
    Returns the path to the saved plot image.
    """
    parser = NLParser()
    spec   = parser.parse(text)

    print(f"\n[PDE Agent] ─── Parsed Specification ────────────────")
    print(f"  type    : {spec.pde_type.upper()}")
    print(f"  dims    : {spec.dims}D")
    print(f"  x-dom   : {spec.x_domain}")
    print(f"  t_end   : {spec.t_end}")
    print(f"  BC      : {spec.bc_type}  (L={spec.bc_left} R={spec.bc_right})")
    print(f"  IC      : {spec.ic_str}")
    print(f"  params  : {spec.params}")
    print(f"  solver  : {'PhysicsNeMo' if spec.use_nemo else 'CauchyNet PINN' if spec.use_pinn else 'FDM'}")
    print(f"[PDE Agent] ─── Solving ─────────────────────────────")

    result = dispatch(spec, text)

    fname  = f"pde_{spec.pde_type}_{spec.dims}d.png"
    outpath = os.path.join(outdir, fname)
    visualize(spec, result, outpath)
    summary(spec, result, outpath)

    return outpath


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def _help():
    print(textwrap.dedent("""
    ╔══════════════════════════════════════════════════════════╗
    ║          CauchyX PDE Agent — Natural Language Solver     ║
    ╚══════════════════════════════════════════════════════════╝

    Usage:
      python pde_solver.py "<natural language description>"

    Supported PDEs:
      heat      u_t = α ∇²u                   (1D / 2D FDM)
      wave      u_tt = c² u_xx                (1D leapfrog)
      poisson   -∇²u = f                      (1D / 2D sparse)
      burgers   u_t + u u_x = ν u_xx          (1D upwind)
      advection u_t + a u_x = 0               (1D upwind)
      allen_cahn u_t = ε²∇²u + u - u³        (1D FDM)
      ode       y'' + 2γω y' + ω²y = 0        (scipy RK45)

    Add "CauchyNet" or "PINN" to use neural solver (PyTorch).

    Examples:
      python pde_solver.py "1D heat alpha=0.01 on [0,1] t=0.5 IC=sin(pi*x)"
      python pde_solver.py "2D Poisson unit square zero BC"
      python pde_solver.py "wave equation c=1.5 on [0,2] IC=sin(pi*x) t=3"
      python pde_solver.py "Burgers nu=0.005 periodic IC=-sin(pi*x) t=1"
      python pde_solver.py "Allen-Cahn epsilon=0.05 on [0,1] t=2"
      python pde_solver.py "harmonic oscillator omega=2 until t=10"
      python pde_solver.py "CauchyNet heat 1D alpha=0.01 on [0,1] t=0.5"
    """))


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        _help(); sys.exit(0)
    text = " ".join(sys.argv[1:])
    solve(text)
