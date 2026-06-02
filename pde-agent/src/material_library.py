"""
material_library.py
EMMO-aligned material property library for PDE coefficient resolution.

Loads materials.ttl, optionally runs the derive_diffusivity CONSTRUCT
rule to materialise missing α values, and exposes a clean Python API.

Usage:
    from src.material_library import get_material, list_materials

    mat = get_material("silicon")
    print(mat.alpha)   # thermal diffusivity m²/s
    print(mat.k)       # thermal conductivity W/(m·K)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rdflib import Graph, Namespace, Literal
from rdflib.namespace import RDF, RDFS, XSD

PDE  = Namespace("http://cauchyx.ai/pde#")

_HERE          = Path(__file__).parent.parent
MATERIALS_TTL  = _HERE / "ontology" / "materials.ttl"
DERIVE_RQ      = _HERE / "sparql"   / "derive_diffusivity.rq"
LOOKUP_RQ      = _HERE / "sparql"   / "material_lookup.rq"

# Supported category URI fragments
CATEGORY_LABELS = {
    "Metal", "Semiconductor", "Fluid", "Ceramic", "Composite", "BatteryMaterial"
}


@dataclass
class MaterialProps:
    name:        str            # canonical label (lowercase)
    category:    str            # Metal | Semiconductor | Fluid | …
    k:           float          # thermal conductivity   W/(m·K)
    rho:         float          # density                kg/m³
    cp:          float          # specific heat          J/(kg·K)
    alpha:       float          # thermal diffusivity    m²/s  = k/(rho·cp)
    mu:          Optional[float] = None   # dynamic viscosity  Pa·s (fluids)
    sigma:       Optional[float] = None   # electrical cond.   S/m
    application: str = ""

    def summary(self) -> str:
        lines = [
            f"Material    : {self.name}  ({self.category})",
            f"  k         : {self.k:.4g} W/(m·K)",
            f"  ρ         : {self.rho:.4g} kg/m³",
            f"  cp        : {self.cp:.4g} J/(kg·K)",
            f"  α = k/ρcp : {self.alpha:.4g} m²/s",
        ]
        if self.mu is not None:
            lines.append(f"  μ         : {self.mu:.4g} Pa·s")
        if self.sigma is not None:
            lines.append(f"  σ         : {self.sigma:.4g} S/m")
        if self.application:
            lines.append(f"  Use case  : {self.application}")
        return "\n".join(lines)


class MaterialLibrary:
    """
    Loads materials.ttl, runs the derive_diffusivity CONSTRUCT rule for
    any materials missing explicit α, then caches all properties.
    """

    def __init__(self):
        self._g = Graph()
        self._g.parse(str(MATERIALS_TTL), format="turtle")

        # Run CONSTRUCT rule to materialise missing diffusivity values
        derive_q = DERIVE_RQ.read_text(encoding="utf-8")
        derived  = self._g.query(derive_q)
        self._g += derived          # merge derived triples into graph

        self._cache: dict[str, MaterialProps] = {}
        self._build_cache()

    # ── Internal ────────────────────────────────────────────────

    def _build_cache(self) -> None:
        q = """
        PREFIX pde:  <http://cauchyx.ai/pde#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?mat ?name ?catClass ?k ?rho ?cp ?alpha ?mu ?sigma ?app
        WHERE {
            ?mat rdfs:label ?name ;
                 a ?catClass ;
                 pde:thermalConductivity  ?k ;
                 pde:density               ?rho ;
                 pde:specificHeatCapacity  ?cp ;
                 pde:thermalDiffusivity    ?alpha .

            ?catClass rdfs:subClassOf pde:Material .

            OPTIONAL { ?mat pde:dynamicViscosity       ?mu }
            OPTIONAL { ?mat pde:electricalConductivity ?sigma }
            OPTIONAL { ?mat pde:applicationDomain      ?app }
        }
        """
        for row in self._g.query(q):
            cat_fragment = str(row.catClass).split("#")[-1]
            key = str(row.name).lower().strip()
            self._cache[key] = MaterialProps(
                name        = str(row.name),
                category    = cat_fragment,
                k           = float(row.k),
                rho         = float(row.rho),
                cp          = float(row.cp),
                alpha       = float(row.alpha),
                mu          = float(row.mu)    if row.mu    else None,
                sigma       = float(row.sigma) if row.sigma else None,
                application = str(row.app)     if row.app   else "",
            )

    # ── Public API ───────────────────────────────────────────────

    def get(self, name: str) -> MaterialProps:
        """
        Look up a material by name (case-insensitive).
        Raises ValueError listing available names if not found.
        """
        key = name.lower().strip()
        if key not in self._cache:
            available = sorted(self._cache.keys())
            raise ValueError(
                f"Unknown material '{name}'. "
                f"Available ({len(available)}): {available}"
            )
        return self._cache[key]

    def list_all(self) -> list[MaterialProps]:
        return list(self._cache.values())

    def list_by_category(self, category: str) -> list[MaterialProps]:
        return [m for m in self._cache.values()
                if m.category.lower() == category.lower()]

    @staticmethod
    def derive_diffusivity(k: float, rho: float, cp: float) -> float:
        """First-principles derivation: α = k / (ρ · cp)"""
        return k / (rho * cp)


# ── Module-level cache + convenience functions ─────────────────

_lib: Optional[MaterialLibrary] = None


def _get_lib() -> MaterialLibrary:
    global _lib
    if _lib is None:
        _lib = MaterialLibrary()
    return _lib


def get_material(name: str) -> MaterialProps:
    """Look up a material by name. Raises ValueError if unknown."""
    return _get_lib().get(name)


def list_materials() -> list[MaterialProps]:
    """Return all materials in the library."""
    return _get_lib().list_all()


def list_by_category(category: str) -> list[MaterialProps]:
    """Return all materials of a given category (Metal, Fluid, …)."""
    return _get_lib().list_by_category(category)


def derive_diffusivity(k: float, rho: float, cp: float) -> float:
    """Compute thermal diffusivity α = k / (ρ · cp) from first principles."""
    return MaterialLibrary.derive_diffusivity(k, rho, cp)
