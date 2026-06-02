"""
test_ontology.py
Phase 1 integration tests — run with: python test_ontology.py

Tests cover:
  1. Valid problem → correct solver routed
  2. Unit hallucination → rejected
  3. Solver-dimension mismatch → rejected
  4. Burgers + SingularityPresent → XNet enforced
  5. PROV-O audit trail generation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.ontology_router import OntologyRouter
from src.unit_normalizer import normalize

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"{status} {name}" + (f" — {detail}" if detail else ""))
    return condition


def run_tests():
    print("\n═══ CauchyX PDE Agent — Ontology Phase 1 Tests ═══\n")
    router = OntologyRouter(compliance_std="DO-178C")
    results = []

    # ── Test 1: 正常 Poisson 2D → XNet ────────────────────────
    try:
        d = router.route({
            "pde_type": "poisson",
            "bc_type": "dirichlet",
            "temporal": "steady",
            "space_dim": 2,
            "parameters": {
                "source_strength": {"value": 1.0, "unit": "PA", "kind": "Pressure"}
            }
        })
        results.append(check(
            "T1: Poisson2D → XNet",
            d.solver_name == "xnet" and d.ok,
            f"solver={d.solver_name}"
        ))
    except Exception as e:
        results.append(check("T1: Poisson2D → XNet", False, str(e)))

    # ── Test 2: 高维问题 → XNet (FDM 被排除) ──────────────────
    try:
        d = router.route({
            "pde_type": "heat",
            "bc_type": "dirichlet",
            "temporal": "transient",
            "space_dim": 50,
        })
        results.append(check(
            "T2: 50D Heat → XNet (not FDM)",
            d.solver_name == "xnet",
            f"solver={d.solver_name}"
        ))
    except Exception as e:
        results.append(check("T2: 50D Heat → XNet", False, str(e)))

    # ── Test 3: 单位量种不匹配 → 幻觉被检测 ──────────────────
    try:
        normalize(101325.0, "PA", "ThermalConductivity")  # Pa 不是热导率单位
        results.append(check("T3: Unit kind mismatch detected", False,
                             "Should have raised ValueError"))
    except ValueError as e:
        results.append(check("T3: Unit kind mismatch detected", True,
                             "ValueError raised correctly"))

    # ── Test 4: 温度超物理范围 → 警告 ─────────────────────────
    try:
        r = normalize(1e15, "K", "Temperature")  # 1e15 K 超出范围
        results.append(check(
            "T4: Temperature out of range → warning",
            not r.in_physical_range,
            f"in_range={r.in_physical_range}"
        ))
    except Exception as e:
        results.append(check("T4: Temperature range check", False, str(e)))

    # ── Test 5: 单位转换正确性 ────────────────────────────────
    try:
        r = normalize(100.0, "DEGC", "Temperature")
        expected = 373.15
        results.append(check(
            "T5: 100 DEGC → 373.15 K",
            abs(r.si_value - expected) < 0.01,
            f"got {r.si_value:.2f} K"
        ))
    except Exception as e:
        results.append(check("T5: DEGC→K conversion", False, str(e)))

    # ── Test 6: 未知方程类型 → 拒绝 ───────────────────────────
    try:
        router.route({
            "pde_type": "navier-stokes-turbulent-quantum",  # 不存在
            "bc_type": "dirichlet",
            "temporal": "transient",
            "space_dim": 3,
        })
        results.append(check("T6: Unknown eq type rejected", False,
                             "Should have raised RuntimeError"))
    except RuntimeError as e:
        results.append(check("T6: Unknown eq type rejected", True,
                             "RuntimeError raised correctly"))

    # ── Test 7: Burgers 1D → XNet (奇点特性) ─────────────────
    try:
        d = router.route({
            "pde_type": "burgers",
            "bc_type": "periodic",
            "temporal": "transient",
            "space_dim": 1,
            "numerical_properties": ["stiff", "singularity"],
            "parameters": {
                "viscosity": {"value": 0.01, "unit": "PA-SEC", "kind": "DynamicViscosity"}
            }
        })
        results.append(check(
            "T7: Burgers+singularity → XNet",
            d.solver_name == "xnet",
            f"solver={d.solver_name}"
        ))
    except Exception as e:
        results.append(check("T7: Burgers+singularity → XNet", False, str(e)))

    # ── Test 8: PROV-O 审计链生成 ─────────────────────────────
    try:
        d = router.route({
            "pde_type": "heat",
            "bc_type": "dirichlet",
            "temporal": "transient",
            "space_dim": 1,
        })
        turtle = d.session.to_turtle()
        report = d.session.to_text_report()
        results.append(check(
            "T8: PROV-O audit trail generated",
            "prov:Activity" in turtle and "DO-178C" in report,
            f"turtle={len(turtle)}chars, report={len(report)}chars"
        ))
    except Exception as e:
        results.append(check("T8: PROV-O audit trail", False, str(e)))

    # ════════════════════════════════════════════════════════════
    #  Phase 2 — EMMO Material Library Tests
    # ════════════════════════════════════════════════════════════
    print("\n--- Phase 2: EMMO Material Library ---\n")
    from src.material_library import (
        get_material, list_materials, list_by_category, derive_diffusivity
    )

    # ── Test 9: Silicon — correct thermal diffusivity ─────────
    try:
        si = get_material("silicon")
        expected_alpha = 8.918e-5      # k/(rho*cp) = 148/(2330*712)
        results.append(check(
            "T9: Silicon alpha correct",
            abs(si.alpha - expected_alpha) / expected_alpha < 1e-3,
            f"alpha={si.alpha:.4e} m2/s (expected {expected_alpha:.4e})"
        ))
    except Exception as e:
        results.append(check("T9: Silicon alpha", False, str(e)))

    # ── Test 10: LFP battery material ────────────────────────
    try:
        lfp = get_material("lfp")
        expected_alpha = 2.778e-7     # 1.0/(3600*1000)
        results.append(check(
            "T10: LFP battery alpha correct",
            abs(lfp.alpha - expected_alpha) / expected_alpha < 1e-3
            and lfp.category == "BatteryMaterial",
            f"alpha={lfp.alpha:.4e}, category={lfp.category}"
        ))
    except Exception as e:
        results.append(check("T10: LFP battery", False, str(e)))

    # ── Test 11: Unknown material raises ValueError ───────────
    try:
        get_material("unobtainium")
        results.append(check("T11: Unknown material rejected", False,
                             "Should have raised ValueError"))
    except ValueError as e:
        results.append(check("T11: Unknown material rejected", True,
                             "ValueError raised correctly"))

    # ── Test 12: First-principles α derivation ────────────────
    try:
        alpha = derive_diffusivity(k=385.0, rho=8960.0, cp=385.0)  # Copper
        expected = 1.116e-4
        results.append(check(
            "T12: Copper alpha = k/(rho*cp)",
            abs(alpha - expected) / expected < 1e-3,
            f"alpha={alpha:.4e} (expected {expected:.4e})"
        ))
    except Exception as e:
        results.append(check("T12: alpha derivation", False, str(e)))

    # ── Test 13: Material-aware routing injects alpha ─────────
    try:
        d = router.route({
            "pde_type": "heat",
            "bc_type":  "dirichlet",
            "temporal": "transient",
            "space_dim": 1,
            "material": "ti-6al-4v",
        })
        alpha_injected = d.norm_params.get("thermal_diffusivity")
        results.append(check(
            "T13: Ti-6Al-4V alpha auto-injected",
            alpha_injected is not None
            and abs(alpha_injected.si_value - 2.876e-6) / 2.876e-6 < 1e-3
            and d.material is not None
            and d.material.category == "Metal",
            f"alpha={alpha_injected.si_value:.4e} m2/s, mat={d.material.category}"
        ))
    except Exception as e:
        results.append(check("T13: Material-aware routing", False, str(e)))

    # ── Test 14: Library has all 15 materials ─────────────────
    try:
        all_mats = list_materials()
        results.append(check(
            "T14: Library contains 15 materials",
            len(all_mats) == 15,
            f"found {len(all_mats)}: {sorted(m.name for m in all_mats)}"
        ))
    except Exception as e:
        results.append(check("T14: Library size", False, str(e)))

    # ── Test 15: Category filter — Fluids ────────────────────
    try:
        fluids = list_by_category("Fluid")
        names  = sorted(m.name for m in fluids)
        results.append(check(
            "T15: list_by_category('Fluid') = 3 fluids",
            len(fluids) == 3,
            f"fluids={names}"
        ))
    except Exception as e:
        results.append(check("T15: Category filter", False, str(e)))

    # ── Test 16: PROV-O records material step ─────────────────
    try:
        d = router.route({
            "pde_type": "heat",
            "bc_type":  "dirichlet",
            "temporal": "transient",
            "space_dim": 2,
            "material": "copper",
        })
        turtle = d.session.to_turtle()
        results.append(check(
            "T16: PROV-O records MaterialResolution step",
            "MaterialResolution" in turtle and "copper" in turtle.lower(),
            f"turtle={len(turtle)}chars"
        ))
    except Exception as e:
        results.append(check("T16: PROV-O material step", False, str(e)))

    # ── Summary ───────────────────────────────────────────────
    passed = sum(results)
    total  = len(results)
    print(f"\n{'═'*48}")
    print(f"Results: {passed}/{total} passed")
    if passed == total:
        print("All tests passed [OK]")
    else:
        print(f"{total - passed} test(s) failed [FAIL]")
    print(f"{'═'*48}\n")
    return passed == total


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
