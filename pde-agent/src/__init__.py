"""
CauchyX PDE Agent — Ontology Layer (Phase 1)
QUDT unit validation + PROV-O audit trail + Hallucination-controlled solver routing.
"""
from .ontology_router import OntologyRouter, RoutingDecision, route
from .unit_normalizer import UnitNormalizer, normalize, normalize_params
from .prov_generator  import ProvSession, AGENT_XNET, AGENT_FDM, AGENT_FEM

__all__ = [
    "OntologyRouter", "RoutingDecision", "route",
    "UnitNormalizer", "normalize", "normalize_params",
    "ProvSession", "AGENT_XNET", "AGENT_FDM", "AGENT_FEM",
]
