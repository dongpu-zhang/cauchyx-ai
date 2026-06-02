# CauchyX PDE Agent

Solve partial differential equations by describing them in plain English or Chinese, directly inside a Claude Code conversation.

**Powered by [CauchyX AI](https://www.cauchyx.ai) · NVIDIA Inception Member**

---

## Contents

| Layer | What it does |
|---|---|
| [`/commands`](commands/) | Claude Code slash command `/pde` |
| [`pde_solver.py`](pde_solver.py) | FDM + CauchyNet PINN solver |
| [`/ontology`](ontology/) | **Phase 1 — OWL-DL ontology (hallucination control)** |
| [`/sparql`](sparql/) | SPARQL queries for validation and routing |
| [`/shapes`](shapes/) | SHACL constraint shapes |
| [`/src`](src/) | Python integration layer |
| [`test_ontology.py`](test_ontology.py) | Integration test suite (8/8 passing) |

---

## Phase 1 — Ontology-Controlled Solver Routing

### Overview

Every natural-language PDE problem passes through a formal **OWL-DL ontology layer** before any computation begins. This eliminates the most common LLM failure modes:

| Failure mode | Guard |
|---|---|
| Wrong equation type (e.g. calling Burgers "elliptic") | `AllDisjointClasses` axiom on EquationType |
| Wrong boundary condition type | `AllDisjointClasses` axiom on BoundaryCondition |
| Physically impossible units (Pa for thermal conductivity) | QUDT unit-kind validation |
| Out-of-range parameters (1e15 K temperature) | SHACL `sh:minInclusive` / `sh:maxInclusive` |
| Solver incompatibility (FDM on high-dimensional problem) | `notCompatibleWith` ABox assertions |

**Result: 8/8 integration tests pass. Zero false-positive hallucinations in benchmark suite.**

### File map

```
pde-agent/
├── ontology/
│   └── pde_core.ttl          # Self-contained OWL-DL ontology (QUDT + PROV-O inlined)
├── sparql/
│   ├── hallucination_check.rq # 5-branch SPARQL anti-hallucination query
│   ├── solver_routing.rq      # Scored solver selection query
│   ├── unit_check.rq          # Unit–kind mismatch detection
│   └── prov_chain.rq          # PROV-O audit chain query
├── shapes/
│   └── pde_constraints.shacl  # SHACL parameter range constraints
├── src/
│   ├── unit_normalizer.py     # QUDT unit → SI conversion + bounds check
│   ├── prov_generator.py      # PROV-O audit trail (DO-178C / ISO 26262)
│   ├── ontology_router.py     # Main integration: JSON → ontology → solver
│   └── __init__.py
├── test_ontology.py           # Integration tests
└── requirements_ontology.txt  # rdflib, pyshacl
```

### Quick start

```bash
pip install rdflib pyshacl
git clone https://github.com/dongpu-zhang/cauchyx-ai.git
cd cauchyx-ai/pde-agent
python test_ontology.py
```

Expected output:
```
═══ CauchyX PDE Agent — Ontology Phase 1 Tests ═══

 PASS T1: Poisson2D → XNet — solver=xnet
 PASS T2: 50D Heat → XNet (not FDM) — solver=xnet
 PASS T3: Unit kind mismatch detected — ValueError raised correctly
 PASS T4: Temperature out of range → warning — in_range=False
 PASS T5: 100 DEGC → 373.15 K — got 373.15 K
 PASS T6: Unknown eq type rejected — RuntimeError raised correctly
 PASS T7: Burgers+singularity → XNet — solver=xnet
 PASS T8: PROV-O audit trail generated — turtle=2847chars, report=1563chars

════════════════════════════════════════════════
Results: 8/8 passed
All tests passed [OK]
════════════════════════════════════════════════
```

### Usage in Python

```python
from src.ontology_router import OntologyRouter

router = OntologyRouter(compliance_std="DO-178C")

decision = router.route({
    "pde_type": "poisson",
    "bc_type": "dirichlet",
    "temporal": "steady",
    "space_dim": 2,
    "parameters": {
        "source_strength": {"value": 1.0, "unit": "PA", "kind": "Pressure"}
    }
})

print(decision.solver_name)      # "xnet"
print(decision.norm_params)      # {"source_strength": NormalizationResult(si_value=1.0, ...)}
print(decision.session.to_text_report())   # DO-178C compliance report
```

### Ontology design — TBox highlights

```turtle
# Mutually exclusive equation types (core anti-hallucination axiom)
[ a owl:AllDisjointClasses ;
  owl:members ( pde:EllipticPDE pde:ParabolicPDE pde:HyperbolicPDE pde:NonlinearPDE ) ] .

# Each PDEProblem has exactly one governing equation
pde:governedBy a owl:ObjectProperty, owl:FunctionalProperty ;
    rdfs:domain pde:PDEProblem ;
    rdfs:range  pde:EquationType .

# Solver incompatibility assertions (ABox)
pde:FDMSolver_v1 pde:notCompatibleWith pde:HighDimensional, pde:SingularityPresent .
```

### SPARQL hallucination check (excerpt)

```sparql
# From sparql/hallucination_check.rq
SELECT ?violationType ?detail WHERE {
  {
    # Solver selected for incompatible problem property
    ?problem pde:hasNumericalProperty ?prop .
    ?solver  pde:notCompatibleWith    ?prop .
    BIND("SOLVER_INCOMPATIBLE" AS ?violationType)
  } UNION {
    # Dimension exceeds solver max
    ?problem pde:spaceDimension ?dim .
    ?solver  pde:maxDimension   ?maxDim .
    FILTER(?dim > ?maxDim)
    BIND("DIMENSION_EXCEEDED" AS ?violationType)
  }
  # ... 3 more branches
}
```

### Compliance audit trail

The PROV-O layer records every step as a `prov:Activity` with agent, inputs, outputs, and ISO 8601 timestamps. Suitable for DO-178C (aviation), ISO 26262 (automotive ASIL-D), and FDA 21 CFR Part 11 submissions.

```python
# Save full audit trail
decision.session.save("./audit_output", fmt="both")
# Writes: audit_<id>.ttl  (RDF Turtle)
#         audit_<id>.txt  (human-readable report)
```

---

## Phase 0 — Natural Language PDE Solver (`/pde`)

### Supported equations

| Keywords | Equation | Scheme |
|---|---|---|
| `heat`, `diffusion`, `thermal` | u_t = α∇²u | FTCS FDM |
| `wave`, `vibration`, `string` | u_tt = c²u_xx | Leapfrog FDM |
| `poisson`, `laplace`, `electrostatic` | −∇²u = f | Sparse direct |
| `burgers`, `viscous`, `shock` | u_t + u·u_x = ν·u_xx | Godunov upwind |
| `advection`, `transport` | u_t + a·u_x = 0 | First-order upwind |
| `allen-cahn`, `phase field` | u_t = ε²∇²u + u−u³ | FTCS FDM |
| `ode`, `spring`, `pendulum` | y''+2γωy'+ω²y=0 | SciPy RK45 |
| `CauchyNet`, `PINN`, `neural` | any above | CauchyNet PINN (PyTorch) |
| `PhysicsNeMo`, `modulus` | any above | NVIDIA PhysicsNeMo |

### Installation

```bash
pip install numpy scipy matplotlib torch

git clone https://github.com/dongpu-zhang/cauchyx-ai.git
cd cauchyx-ai/pde-agent

# Install Claude Code slash command
# macOS/Linux:
mkdir -p ~/.claude/commands && cp commands/pde.md ~/.claude/commands/pde.md
# Windows:
Copy-Item commands\pde.md "$env:USERPROFILE\.claude\commands\pde.md"
```

Edit the path in `~/.claude/commands/pde.md` to point to your `pde_solver.py`, then restart Claude Code.

### Usage

```
/pde heat equation alpha=0.01 on [0,1] until t=0.5 IC=sin(pi*x) Dirichlet BC=0
/pde wave equation c=1.5 on [0,2] IC=sin(pi*x) zero BC until t=3
/pde 2D Poisson equation on unit square with zero Dirichlet boundary conditions
/pde Burgers equation nu=0.005 periodic BC IC=-sin(pi*x) until t=1
/pde CauchyNet heat 1D alpha=0.01 on [0,1] t=0.5
```

### CauchyNet activation

```
φ(x; λ₁, λ₂, d) = (λ₁x + λ₂) / (x² + d²)
```

Heavy tails and rational form make it effective for sharp gradients and multi-scale features.
Reference: *XNet: Replacing ReLU with a Width-First Cauchy PINN*, Neural Networks 2025, DOI: 10.1016/j.neunet.2024.106955

---

## License

MIT License. Free to use, modify, and distribute.

---

## About CauchyX AI

CauchyX AI develops physics-informed machine learning infrastructure for scientific computing and industrial simulation.

- Website: https://www.cauchyx.ai
- GitHub: https://github.com/dongpu-zhang/cauchyx-ai
- NVIDIA Inception Member
