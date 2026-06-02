"""
CauchyX PDE Agent — Ontology Layer
Phase 1: QUDT unit validation + PROV-O audit trail + hallucination-controlled solver routing.
Phase 2: EMMO-aligned material library (15 materials, α = k/(ρ·cp) derivation).
"""
from .ontology_router  import OntologyRouter, RoutingDecision, route
from .unit_normalizer  import UnitNormalizer, normalize, normalize_params
from .prov_generator   import ProvSession, AGENT_XNET, AGENT_FDM, AGENT_FEM
from .material_library import (
    MaterialLibrary, MaterialProps,
    get_material, list_materials, list_by_category, derive_diffusivity,
)

__all__ = [
    # Phase 1
    "OntologyRouter", "RoutingDecision", "route",
    "UnitNormalizer", "normalize", "normalize_params",
    "ProvSession", "AGENT_XNET", "AGENT_FDM", "AGENT_FEM",
    # Phase 2
    "MaterialLibrary", "MaterialProps",
    "get_material", "list_materials", "list_by_category", "derive_diffusivity",
]
