"""
unit_normalizer.py
QUDT-based unit normalization and hallucination detection.
Converts any supported unit to SI, validates quantity kinds.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


# SI 目标单位（每个量种的标准单位）
SI_UNIT: dict[str, str] = {
    "Temperature":          "K",
    "ThermalConductivity":  "W-PER-M-K",
    "ThermalDiffusivity":   "M2-PER-SEC",
    "DynamicViscosity":     "PA-SEC",
    "Pressure":             "PA",
    "Speed":                "M-PER-SEC",
    "Density":              "KG-PER-M3",
    "SpecificHeatCapacity": "J-PER-KG-K",
}

# 每个单位 → (量种, 转换到SI的乘数, 偏移量)
# SI值 = 输入值 * multiplier + offset
UNIT_TABLE: dict[str, tuple[str, float, float]] = {
    # 温度
    "K":     ("Temperature",         1.0,     0.0),
    "DEGC":  ("Temperature",         1.0,   273.15),
    "DEGF":  ("Temperature",         5/9,   255.372),
    "DEGR":  ("Temperature",         5/9,     0.0),
    # 热导率
    "W-PER-M-K":    ("ThermalConductivity", 1.0,    0.0),
    "W-PER-CM-K":   ("ThermalConductivity", 100.0,  0.0),
    "BTU-PER-HR-FT-DEGF": ("ThermalConductivity", 1.7307, 0.0),
    # 热扩散系数
    "M2-PER-SEC":   ("ThermalDiffusivity",  1.0,    0.0),
    "CM2-PER-SEC":  ("ThermalDiffusivity",  1e-4,   0.0),
    "MM2-PER-SEC":  ("ThermalDiffusivity",  1e-6,   0.0),
    # 粘度
    "PA-SEC":  ("DynamicViscosity",  1.0,   0.0),
    "CP":      ("DynamicViscosity",  1e-3,  0.0),   # centipoise
    "P":       ("DynamicViscosity",  0.1,   0.0),   # poise
    # 压力
    "PA":   ("Pressure", 1.0,    0.0),
    "KPA":  ("Pressure", 1e3,    0.0),
    "MPA":  ("Pressure", 1e6,    0.0),
    "BAR":  ("Pressure", 1e5,    0.0),
    "ATM":  ("Pressure", 101325.0, 0.0),
    "PSI":  ("Pressure", 6894.76,  0.0),
    # 速度
    "M-PER-SEC":  ("Speed", 1.0,     0.0),
    "KM-PER-HR":  ("Speed", 1/3.6,   0.0),
    "FT-PER-SEC": ("Speed", 0.3048,  0.0),
    # 密度
    "KG-PER-M3":  ("Density", 1.0,    0.0),
    "G-PER-CM3":  ("Density", 1000.0, 0.0),
    # 比热容
    "J-PER-KG-K":    ("SpecificHeatCapacity", 1.0,    0.0),
    "KJ-PER-KG-K":   ("SpecificHeatCapacity", 1000.0, 0.0),
    "BTU-PER-LB-DEGF": ("SpecificHeatCapacity", 4186.8, 0.0),
}

# 每个量种的物理合理范围（SI 单位）
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "Temperature":          (0.0,    1e8),     # K: 绝对零度 ~ 太阳核心
    "ThermalConductivity":  (1e-3,   2500.0),  # W/(m·K): 气体 ~ 钻石
    "ThermalDiffusivity":   (1e-9,   1e-3),    # m²/s: 绝热材料 ~ 银
    "DynamicViscosity":     (1e-6,   1e6),     # Pa·s: 气体 ~ 沥青
    "Pressure":             (0.0,    1e12),    # Pa: 真空 ~ 中子星表面
    "Speed":                (0.0,    3e8),     # m/s: 静止 ~ 光速
    "Density":              (1e-5,   2e4),     # kg/m³: 极稀薄气体 ~ 金属
    "SpecificHeatCapacity": (100.0,  1e5),     # J/(kg·K): 金属 ~ 氢气
}


@dataclass
class NormalizationResult:
    si_value: float
    si_unit: str
    quantity_kind: str
    original_value: float
    original_unit: str
    in_physical_range: bool
    range_warning: Optional[str] = None


class UnitNormalizer:
    """
    Converts physical parameter values to SI units.
    Detects unit-quantity kind mismatches (LLM hallucinations).
    """

    def normalize(
        self,
        value: float,
        unit: str,
        declared_kind: Optional[str] = None,
    ) -> NormalizationResult:
        """
        Convert value+unit to SI.
        Raises ValueError if unit is unknown or kind mismatches.
        """
        unit_key = unit.upper().replace(" ", "-").replace("/", "-PER-")
        # 规范化常见别名
        unit_key = unit_key.replace("WATT", "W").replace("METER", "M")

        if unit_key not in UNIT_TABLE:
            raise ValueError(
                f"Unknown unit: '{unit}'. "
                f"Supported units: {sorted(UNIT_TABLE.keys())}"
            )

        actual_kind, multiplier, offset = UNIT_TABLE[unit_key]

        # 量种一致性检查（幻觉检测核心）
        if declared_kind and declared_kind != actual_kind:
            raise ValueError(
                f"Unit-kind mismatch (HALLUCINATION DETECTED): "
                f"declared kind='{declared_kind}' but unit '{unit}' "
                f"implies kind='{actual_kind}'. "
                f"For {declared_kind}, use: {SI_UNIT.get(declared_kind, '?')}"
            )

        si_value = value * multiplier + offset
        si_unit  = SI_UNIT.get(actual_kind, unit_key)

        # 物理范围检查
        in_range = True
        warning  = None
        if actual_kind in PHYSICAL_BOUNDS:
            lo, hi = PHYSICAL_BOUNDS[actual_kind]
            if not (lo <= si_value <= hi):
                in_range = False
                warning  = (
                    f"Value {si_value:.3e} {si_unit} is outside physical bounds "
                    f"[{lo:.3e}, {hi:.3e}] for {actual_kind}. "
                    f"Possible unit error or hallucination."
                )

        return NormalizationResult(
            si_value=si_value,
            si_unit=si_unit,
            quantity_kind=actual_kind,
            original_value=value,
            original_unit=unit,
            in_physical_range=in_range,
            range_warning=warning,
        )

    def normalize_params(self, params: dict) -> dict:
        """
        Normalize a dict of {name: {value, unit, kind?}} to SI.
        Returns {name: NormalizationResult}.
        Raises on any hallucination; collects all errors before raising.
        """
        results = {}
        errors  = []

        for name, spec in params.items():
            try:
                result = self.normalize(
                    value=float(spec["value"]),
                    unit=str(spec["unit"]),
                    declared_kind=spec.get("kind"),
                )
                if result.range_warning:
                    errors.append(f"[{name}] {result.range_warning}")
                results[name] = result
            except ValueError as e:
                errors.append(f"[{name}] {e}")

        if errors:
            raise ValueError(
                "Parameter validation failed:\n" + "\n".join(errors)
            )

        return results


# ── 快捷函数（供 ontology_router.py 直接调用）
_normalizer = UnitNormalizer()

def normalize(value: float, unit: str, kind: str | None = None) -> NormalizationResult:
    return _normalizer.normalize(value, unit, kind)

def normalize_params(params: dict) -> dict:
    return _normalizer.normalize_params(params)
