# /pde — Natural Language PDE Solver (CauchyX PDE Agent)

Solve partial differential equations described in plain natural language.
Supports FDM classical solvers, CauchyNet PINN (PyTorch), and PhysicsNeMo backend.

## Setup (one-time)

After installing, edit the solver path below to match where you saved `pde_solver.py`:

```
SOLVER_PATH = /path/to/pde_solver.py
```

## Trigger

User types `/pde <description>` — description is a free-form natural language
problem statement, in English or Chinese.

## Workflow

1. **Echo the parsed spec** — show a structured box with what was understood:
   PDE type, domain, BC, IC, parameters, solver mode.
   Ask for correction before solving if anything looks wrong.

2. **Run the solver** — execute via Bash (adjust path to your installation):
   ```
   python /path/to/pde_solver.py "<description>"
   ```
   On Windows use PowerShell:
   ```
   python C:\your\path\pde_solver.py "<description>"
   ```

3. **Display results** — show the summary table (printed by the script) and
   display the saved PNG image using the Read tool on the output path.

4. **Offer follow-up** — suggest parameter sweeps, different BCs, or PINN mode.

## Supported PDE types

| Keyword(s) in description | PDE | Solver |
|---|---|---|
| heat, diffusion, thermal | u_t = α∇²u | FTCS FDM |
| wave, vibration, string | u_tt = c²u_xx | Leapfrog FDM |
| poisson, laplace, electrostatic | −∇²u = f | Sparse direct |
| burgers, viscous, shock | u_t + u u_x = ν u_xx | Godunov upwind |
| advection, transport | u_t + a u_x = 0 | First-order upwind |
| allen-cahn, phase field | u_t = ε²∇²u + u−u³ | FTCS FDM |
| ode, spring, pendulum, oscillator | y''+2γωy'+ω²y=0 | SciPy RK45 |
| CauchyNet, PINN, neural | any above | CauchyNet PINN (PyTorch) |
| PhysicsNeMo, modulus | any above | NVIDIA PhysicsNeMo |

## Parameter syntax the parser understands

| Natural language form | Extracted as |
|---|---|
| `alpha=0.01`, `α=0.01`, `diffusivity 0.01` | thermal diffusivity α |
| `c=1.5`, `wave speed 2`, `velocity=3` | wave speed c |
| `nu=0.005`, `ν=0.01`, `viscosity 0.01` | kinematic viscosity ν |
| `[0,1]`, `[0,pi]`, `[0,2]` | spatial domain bounds |
| `t=0.5`, `until t=2`, `to t=3` | simulation end time |
| `sin(pi*x)`, `gaussian`, `step function` | initial condition |
| `Dirichlet`, `Neumann`, `periodic` | boundary condition type |
| `2D`, `two-dimensional`, `(x,y)` | 2D spatial problem |
| `nx=256` | spatial grid resolution |
| `CauchyNet` or `PINN` | use neural network solver |
| `PhysicsNeMo` or `modulus` | use NVIDIA PhysicsNeMo backend |

## Example invocations

```
/pde solve the 1D heat equation with alpha=0.01 on [0,1] until t=0.5, IC=sin(pi*x), Dirichlet BC=0
/pde 2D Poisson equation on unit square with zero Dirichlet boundary conditions
/pde wave equation c=1.5 on [0,2] with IC=sin(pi*x) zero BC until t=3
/pde Burgers equation nu=0.005 periodic BC IC=-sin(pi*x) until t=1
/pde Allen-Cahn epsilon=0.05 on [0,1] until t=2 with tanh initial condition
/pde harmonic oscillator omega=2 gamma=0.1 until t=20
/pde CauchyNet heat 1D alpha=0.01 on [0,1] t=0.5
/pde PhysicsNeMo Navier-Stokes 2D on [0,1]x[0,1]
```

## After the run

- The solution plot is saved to `pde_<type>_<dims>d.png` in the current directory
- Read and display it with the Read tool
- Print the numerical summary (max/mean/min of solution field)
- Ask: "Want to try different parameters, a different solver, or compare FDM vs CauchyNet PINN?"

## Error handling

- If the PDE type is ambiguous, ask the user to clarify before running
- If scipy/torch is missing, inform the user and suggest `pip install scipy torch matplotlib`
- If the solution diverges (inf/nan), suggest reducing t_end or increasing nx
