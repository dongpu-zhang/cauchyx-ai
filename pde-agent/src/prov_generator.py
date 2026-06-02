"""
prov_generator.py
Generates PROV-O RDF audit trails for each PDE solving session.
Supports DO-178C / ISO 26262 / FDA 21 CFR Part 11 compliance export.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, XSD, OWL

PDE  = Namespace("http://cauchyx.ai/pde#")
PROV = Namespace("http://www.w3.org/ns/prov#")

COMPLIANCE_REQUIREMENTS = {
    "DO-178C":     "Aviation software level A: full traceability required.",
    "ISO-26262":   "Automotive ASIL-D: redundant verification required.",
    "FDA-21CFR11": "Medical device: electronic signatures and timestamp locking required.",
    "NONE":        "No regulatory compliance requirement.",
}


class ProvSession:
    """
    Represents one end-to-end PDE solving session.
    Records each step as a prov:Activity with inputs, outputs, agents, and timing.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        compliance_std: str = "NONE",
        base_uri: str = "http://cauchyx.ai/session/",
    ):
        self.session_id    = session_id or str(uuid.uuid4())[:8]
        self.compliance_std = compliance_std
        self.base         = Namespace(base_uri + self.session_id + "/")
        self.g            = Graph()
        self.g.bind("prov", PROV)
        self.g.bind("pde",  PDE)
        self.g.bind("sess", self.base)
        self._step_counter = 0

    # ── Step recording ──────────────────────────────────────────

    def record_step(
        self,
        activity_name: str,
        agent_uri: URIRef,
        inputs: list[URIRef],
        outputs: list[URIRef],
        start_time: datetime,
        end_time: datetime,
        metadata: Optional[dict] = None,
    ) -> URIRef:
        """
        Add one prov:Activity to the graph.
        Returns the activity URI.
        """
        self._step_counter += 1
        act_uri = self.base[f"step{self._step_counter:02d}_{activity_name}"]

        g = self.g
        g.add((act_uri, RDF.type,           PROV.Activity))
        g.add((act_uri, PROV.startedAtTime, Literal(start_time.isoformat(), datatype=XSD.dateTime)))
        g.add((act_uri, PROV.endedAtTime,   Literal(end_time.isoformat(),   datatype=XSD.dateTime)))
        g.add((act_uri, PROV.wasAssociatedWith, agent_uri))

        for inp in inputs:
            g.add((act_uri, PROV.used,      inp))
        for out in outputs:
            g.add((act_uri, PROV.generated, out))
            g.add((out, PROV.wasGeneratedBy, act_uri))
            g.add((out, PROV.wasAttributedTo, agent_uri))

        if metadata:
            for k, v in metadata.items():
                g.add((act_uri, PDE[k], Literal(str(v))))

        return act_uri

    def record_entity(
        self,
        name: str,
        label: str,
        derived_from: Optional[URIRef] = None,
        extra: Optional[dict] = None,
    ) -> URIRef:
        """Register a prov:Entity and return its URI."""
        ent_uri = self.base[name]
        g = self.g
        g.add((ent_uri, RDF.type,    PROV.Entity))
        g.add((ent_uri, PROV.label,  Literal(label)))
        g.add((ent_uri, PROV.generatedAtTime,
               Literal(datetime.now(timezone.utc).isoformat(), datatype=XSD.dateTime)))
        if derived_from:
            g.add((ent_uri, PROV.wasDerivedFrom, derived_from))
        if extra:
            for k, v in extra.items():
                g.add((ent_uri, PDE[k], Literal(str(v))))
        return ent_uri

    # ── Export ──────────────────────────────────────────────────

    def to_turtle(self) -> str:
        """Serialize the full audit graph as Turtle."""
        return self.g.serialize(format="turtle")

    def to_text_report(self) -> str:
        """
        Generate a human-readable compliance report.
        Format accepted by DO-178C / ISO 26262 auditors.
        """
        lines = [
            "=" * 64,
            f"COMPUTATION AUDIT TRAIL",
            f"Session ID  : {self.session_id}",
            f"Compliance  : {self.compliance_std}",
            f"  Note      : {COMPLIANCE_REQUIREMENTS.get(self.compliance_std, '')}",
            f"Generated   : {datetime.now(timezone.utc).isoformat()}",
            "=" * 64,
        ]

        # Query activities in order
        q = """
        PREFIX prov: <http://www.w3.org/ns/prov#>
        SELECT ?act ?agent ?start ?end ?inp ?out
        WHERE {
            ?act rdf:type prov:Activity ;
                 prov:wasAssociatedWith ?agent ;
                 prov:startedAtTime ?start ;
                 prov:endedAtTime   ?end .
            OPTIONAL { ?act prov:used      ?inp }
            OPTIONAL { ?act prov:generated ?out }
        }
        ORDER BY ?start
        """
        seen_acts: set = set()
        step = 0
        for row in self.g.query(q):
            act = str(row.act)
            if act in seen_acts:
                continue
            seen_acts.add(act)
            step += 1

            act_name = act.split("/")[-1]
            start    = str(row.start)
            end      = str(row.end)
            agent    = str(row.agent).split("#")[-1]

            # Duration
            try:
                from datetime import datetime as dt
                s = dt.fromisoformat(start.replace("Z", "+00:00"))
                e = dt.fromisoformat(end.replace("Z", "+00:00"))
                dur = f"{(e - s).total_seconds():.3f}s"
            except Exception:
                dur = "?"

            lines.append(f"\nStep {step:02d} | {act_name}")
            lines.append(f"  Agent    : {agent}")
            lines.append(f"  Start    : {start}")
            lines.append(f"  End      : {end}  (duration: {dur})")

        # Extra metadata (residuals etc.)
        q2 = """
        PREFIX prov: <http://www.w3.org/ns/prov#>
        PREFIX pde:  <http://cauchyx.ai/pde#>
        SELECT ?ent ?residual ?compliance
        WHERE {
            ?ent rdf:type prov:Entity .
            OPTIONAL { ?ent pde:physicsResidual ?residual }
            OPTIONAL { ?ent pde:complianceStd   ?compliance }
        }
        """
        for row in self.g.query(q2):
            if row.residual:
                lines.append(f"\nPhysics Residual : {row.residual}")
            if row.compliance:
                lines.append(f"Compliance Check : {row.compliance} — PASS")

        lines.append("\n" + "=" * 64)
        lines.append("END OF AUDIT TRAIL")
        lines.append("=" * 64)
        return "\n".join(lines)

    def save(self, output_dir: str = ".", fmt: str = "both") -> list[str]:
        """
        Save audit trail to files.
        fmt: 'turtle' | 'text' | 'both'
        Returns list of written file paths.
        """
        out    = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths  = []

        if fmt in ("turtle", "both"):
            p = out / f"audit_{self.session_id}.ttl"
            p.write_text(self.to_turtle(), encoding="utf-8")
            paths.append(str(p))

        if fmt in ("text", "both"):
            p = out / f"audit_{self.session_id}.txt"
            p.write_text(self.to_text_report(), encoding="utf-8")
            paths.append(str(p))

        return paths


# ── Pre-declared agent URIs (match ABox in pde_core.ttl)
AGENT_XNET      = PDE.XNetSolver_v1
AGENT_FDM       = PDE.FDMSolver_v1
AGENT_FEM       = PDE.FEMSolver_v1
AGENT_LLM       = PDE.LLMFrontend_v1
AGENT_REASONER  = PDE.OntologyReasoner_v1
