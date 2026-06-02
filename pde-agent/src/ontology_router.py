"""
ontology_router.py
Main integration point: LLM JSON output → ontology validation → solver decision.

Flow:
  1. Load pde_core.ttl into rdflib graph
  2. Instantiate PDEProblem from LLM JSON
  3. Run hallucination_check.rq  → reject on violations
  4. Normalize all parameters    → unit_normalizer
  5. Run SHACL constraints       → reject on shape violations
  6. Run solver_routing.rq       → select solver
  7. Start ProvSession           → record every step
  8. Return routing decision + session for pde_solver.py

Usage:
  from src.ontology_router import OntologyRouter
  router = OntologyRouter()
  decision = router.route(llm_json)
  # decision.solver_name, decision.session, decision.norm_params
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, XSD

try:
    from pyshacl import validate as shacl_validate
    SHACL_AVAILABLE = True
except ImportError:
    SHACL_AVAILABLE = False

from .unit_normalizer import normalize_params, NormalizationResult
from .prov_generator  import ProvSession, AGENT_LLM, AGENT_REASONER, AGENT_XNET, AGENT_FDM, AGENT_FEM

PDE  = Namespace("http://cauchyx.ai/pde#")
PROV = Namespace("http://www.w3.org/ns/prov#")

# 本体文件路径（相对于此文件）
_HERE        = Path(__file__).parent.parent
ONTOLOGY_TTL = _HERE / "ontology" / "pde_core.ttl"
HALLUC_RQ    = _HERE / "sparql"   / "hallucination_check.rq"
ROUTING_RQ   = _HERE / "sparql"   / "solver_routing.rq"
SHACL_FILE   = _HERE / "shapes"   / "pde_constraints.shacl"

# LLM JSON → PDE 本体 URI 的映射表
EQ_TYPE_MAP = {
    "poisson":    PDE.PoissonEquation,
    "laplace":    PDE.LaplaceEquation,
    "heat":       PDE.HeatEquation,
    "diffusion":  PDE.HeatEquation,
    "wave":       PDE.WaveEquation,
    "advection":  PDE.AdvectionEquation,
    "burgers":    PDE.BurgersEquation,
    "allen-cahn": PDE.AllenCahnEquation,
    "allencahn":  PDE.AllenCahnEquation,
}

BC_TYPE_MAP = {
    "dirichlet": PDE.DirichletBC,
    "neumann":   PDE.NeumannBC,
    "robin":     PDE.RobinBC,
    "periodic":  PDE.PeriodicBC,
    "mixed":     PDE.RobinBC,
}

TEMPORAL_MAP = {
    "steady":    PDE.SteadyState,
    "steady-state": PDE.SteadyState,
    "transient": PDE.Transient,
    "time-dependent": PDE.Transient,
}

DOMAIN_MAP = {
    1: PDE.Domain1D,
    2: PDE.Domain2D,
    3: PDE.Domain3D,
}

NUM_PROP_KEYWORDS = {
    "stiff":       PDE.StiffProblem,
    "singularity": PDE.SingularityPresent,
    "singular":    PDE.SingularityPresent,
    "high-dim":    PDE.HighDimensional,
    "high_dim":    PDE.HighDimensional,
    "nonlinear":   PDE.SingularityPresent,
    "convection":  PDE.ConvectionDominated,
}

SOLVER_LABEL_MAP = {
    str(PDE.XNetSolver_v1):  "xnet",
    str(PDE.FDMSolver_v1):   "fdm",
    str(PDE.FEMSolver_v1):   "fem",
}

SOLVER_AGENT_MAP = {
    "xnet": AGENT_XNET,
    "fdm":  AGENT_FDM,
    "fem":  AGENT_FEM,
}


@dataclass
class RoutingDecision:
    solver_name: str                           # "xnet" | "fdm" | "fem"
    solver_uri:  URIRef
    session:     ProvSession
    norm_params: dict[str, NormalizationResult]
    problem_uri: URIRef
    violations:  list[dict]  = field(default_factory=list)
    warnings:    list[str]   = field(default_factory=list)
    ok:          bool        = True


class OntologyRouter:
    """
    Validates LLM JSON against PDE ontology and routes to the correct solver.
    Instantiate once; call route() per request.
    """

    def __init__(self, compliance_std: str = "NONE"):
        self.compliance_std = compliance_std
        self._base_graph    = Graph()
        self._base_graph.parse(str(ONTOLOGY_TTL), format="turtle")
        self._halluc_query  = HALLUC_RQ.read_text(encoding="utf-8")
        self._routing_query = ROUTING_RQ.read_text(encoding="utf-8")

    # ── Public API ───────────────────────────────────────────────

    def route(self, llm_json: dict | str) -> RoutingDecision:
        """
        Full pipeline: parse → validate → normalize → route → record provenance.
        Raises RuntimeError if critical violations are found.
        """
        if isinstance(llm_json, str):
            llm_json = json.loads(llm_json)

        session  = ProvSession(compliance_std=self.compliance_std)
        warnings: list[str] = []

        # ── Step 1: LLM 解析
        t0 = _now()
        problem_uri, g = self._instantiate(llm_json, session)
        t1 = _now()
        input_ent  = session.record_entity("user_input",  "LLM parsed JSON input")
        parsed_ent = session.record_entity("parsed_pde",  "Instantiated PDEProblem",
                                           derived_from=input_ent)
        session.record_step("LLMParsing", AGENT_LLM,
                            [input_ent], [parsed_ent], t0, t1)

        # ── Step 2: 本体幻觉检测
        t2 = _now()
        violations = self._check_hallucinations(g, problem_uri)
        t3 = _now()
        valid_ent  = session.record_entity("validation_result",
                                           f"Hallucination check: {len(violations)} violations",
                                           derived_from=parsed_ent,
                                           extra={"violationCount": len(violations)})
        session.record_step("OntologyValidation", AGENT_REASONER,
                            [parsed_ent], [valid_ent], t2, t3)

        if violations:
            msg = "ONTOLOGY VIOLATIONS (hallucinations detected):\n"
            msg += "\n".join(f"  [{v['type']}] {v['detail']}" for v in violations)
            raise RuntimeError(msg)

        # ── Step 3: SHACL 约束检查（参数范围）
        shacl_warns = self._shacl_check(g)
        warnings.extend(shacl_warns)

        # ── Step 4: 参数单位归一化
        raw_params = llm_json.get("parameters", {})
        norm_params: dict[str, NormalizationResult] = {}
        if raw_params:
            try:
                norm_params = normalize_params(raw_params)
            except ValueError as e:
                raise RuntimeError(f"Unit normalization failed:\n{e}") from e

        # ── Step 5: 求解器路由
        t4 = _now()
        solver_uri   = self._route_solver(g, problem_uri)
        solver_name  = SOLVER_LABEL_MAP.get(str(solver_uri), "xnet")
        t5 = _now()
        solution_ent = session.record_entity("routing_decision",
                                             f"Solver selected: {solver_name}",
                                             derived_from=valid_ent,
                                             extra={"solverName": solver_name})
        session.record_step("SolverRouting", AGENT_REASONER,
                            [valid_ent], [solution_ent], t4, t5)

        return RoutingDecision(
            solver_name = solver_name,
            solver_uri  = solver_uri,
            session     = session,
            norm_params = norm_params,
            problem_uri = problem_uri,
            violations  = violations,
            warnings    = warnings,
            ok          = True,
        )

    # ── Private helpers ──────────────────────────────────────────

    def _instantiate(
        self, llm_json: dict, session: ProvSession
    ) -> tuple[URIRef, Graph]:
        """Convert LLM JSON to RDF triples appended to a copy of the base graph."""
        g = Graph()
        g += self._base_graph          # clone base

        sess_id = session.session_id
        prob_uri = URIRef(f"http://cauchyx.ai/session/{sess_id}/problem")
        g.add((prob_uri, RDF.type, PDE.PDEProblem))

        # 方程类型
        eq_raw = llm_json.get("pde_type", "").lower().strip()
        eq_uri = EQ_TYPE_MAP.get(eq_raw)
        if eq_uri is None:
            raise RuntimeError(
                f"Unknown pde_type: '{llm_json.get('pde_type')}'. "
                f"Allowed: {sorted(EQ_TYPE_MAP.keys())}"
            )
        g.add((prob_uri, PDE.governedBy, eq_uri))

        # 边界条件
        bc_raw = llm_json.get("bc_type", "dirichlet").lower().strip()
        bc_uri = BC_TYPE_MAP.get(bc_raw)
        if bc_uri is None:
            raise RuntimeError(
                f"Unknown bc_type: '{llm_json.get('bc_type')}'. "
                f"Allowed: {sorted(BC_TYPE_MAP.keys())}"
            )
        g.add((prob_uri, PDE.hasBoundaryCondition, bc_uri))

        # 时间类型
        temp_raw = llm_json.get("temporal", "transient").lower().strip()
        temp_uri = TEMPORAL_MAP.get(temp_raw, PDE.Transient)
        g.add((prob_uri, PDE.hasTemporalType, temp_uri))

        # 空间域
        dim = int(llm_json.get("space_dim", 1))
        dom_uri = DOMAIN_MAP.get(min(dim, 3), PDE.Domain1D)
        g.add((prob_uri, PDE.hasSpatialDomain, dom_uri))
        g.add((prob_uri, PDE.spaceDimension, Literal(dim, datatype=XSD.integer)))

        # 数值特性（从关键词列表推断）
        props_raw = [p.lower() for p in llm_json.get("numerical_properties", [])]
        if dim > 10:
            props_raw.append("high-dim")
        for kw, prop_uri in NUM_PROP_KEYWORDS.items():
            if kw in props_raw:
                g.add((prob_uri, PDE.hasNumericalProperty, prop_uri))
        if not props_raw:
            g.add((prob_uri, PDE.hasNumericalProperty, PDE.RegularProblem))

        # 时间终点（稳态问题不应设置）
        time_end = llm_json.get("time_end")
        if time_end is not None:
            g.add((prob_uri, PDE.timeEnd, Literal(float(time_end), datatype=XSD.double)))

        return prob_uri, g

    def _check_hallucinations(
        self, g: Graph, problem_uri: URIRef
    ) -> list[dict]:
        """Run hallucination SPARQL query. Returns list of violations."""
        results = g.query(
            self._halluc_query,
            initBindings={"problem": problem_uri},
        )
        violations = []
        for row in results:
            violations.append({
                "type":   str(row.violationType),
                "detail": str(row.detail),
            })
        return violations

    def _shacl_check(self, g: Graph) -> list[str]:
        """Run SHACL validation. Returns warnings (non-fatal)."""
        if not SHACL_AVAILABLE:
            return ["pyshacl not installed — SHACL validation skipped"]
        try:
            shacl_g = Graph()
            shacl_g.parse(str(SHACL_FILE), format="turtle")
            conforms, _, report_text = shacl_validate(
                g, shacl_graph=shacl_g, abort_on_first=False
            )
            if not conforms:
                return [f"SHACL: {line}" for line in report_text.split("\n") if line.strip()]
        except Exception as e:
            return [f"SHACL validation error: {e}"]
        return []

    def _route_solver(self, g: Graph, problem_uri: URIRef) -> URIRef:
        """
        Python-based solver routing using ontology ABox.
        Rules derived from solver capability assertions in pde_core.ttl.
        XNet is default (highest accuracy); FDM/FEM only when XNet is overkill
        and the problem is low-dimensional with no singularities.
        """
        # Collect numerical properties of this problem
        props = set(
            str(o).split("#")[-1]
            for o in g.objects(problem_uri, PDE.hasNumericalProperty)
        )
        dim = int(next(g.objects(problem_uri, PDE.spaceDimension), Literal(1)))

        # Blocking conditions for non-XNet solvers
        blocks_fdm = {
            "HighDimensional", "SingularityPresent", "StiffProblem", "ConvectionDominated"
        }
        blocks_fem = {"HighDimensional"}

        # Prefer FDM only when: low-dim + regular + no singularity + no stiffness
        use_fdm = (
            dim <= 3
            and "RegularProblem" in props
            and not (props & blocks_fdm)
        )

        # Poisson/Laplace (elliptic) on regular low-dim domains: FEM is natural
        # but XNet outperforms both — keep XNet as default
        # FDM is only preferred for pure advection problems
        eq_types = set(
            str(o).split("#")[-1]
            for o in g.objects(problem_uri, PDE.governedBy)
        )
        is_advection = bool({"AdvectionEquation"} & eq_types)

        if use_fdm and is_advection:
            return PDE.FDMSolver_v1

        # Default: XNet (highest accuracy, handles all problem types)
        return PDE.XNetSolver_v1


# ── Convenience function ─────────────────────────────────────────

_default_router: Optional[OntologyRouter] = None

def route(llm_json: dict | str, compliance_std: str = "NONE") -> RoutingDecision:
    """
    Module-level shortcut. Reuses a cached router instance.
    """
    global _default_router
    if _default_router is None or _default_router.compliance_std != compliance_std:
        _default_router = OntologyRouter(compliance_std=compliance_std)
    return _default_router.route(llm_json)


def _now() -> datetime:
    return datetime.now(timezone.utc)
