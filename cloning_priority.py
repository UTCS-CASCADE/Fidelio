from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any


EPSILON = 1e-9
PRIORITY_ORDER = (1, 2, 3, 4)

# Default selection of which prioritization profile to use. The historical
# behavior is the frontend-bound hierarchy, so it remains the default. Select an
# alternative profile by exporting CLONE_PRIORITY_MODE (e.g. "backend") or by
# calling set_priority_mode() programmatically.
DEFAULT_PRIORITY_MODE = "frontend"
_PRIORITY_MODE_ENV = "CLONE_PRIORITY_MODE"
_PRIORITY_MODE_OVERRIDE: str | None = None


@dataclass(frozen=True)
class FeatureConstraint:
    feature_name: str
    group: str
    priority_rank: int
    tolerance_ratio: float
    importance_weight: float


# ---------------------------------------------------------------------------
# Frontend-bound profile (default / historical behavior).
# Tiers: T0 IPC, T1 frontend, T2 branch/data cache, T3 iTLB/top-down.
# ---------------------------------------------------------------------------
_FRONTEND_TIER_LABELS = {
    1: "Tier T0 IPC",
    2: "Tier T1 frontend",
    3: "Tier T2 branch/data cache",
    4: "Tier T3 iTLB/top-down",
}

_FRONTEND_CONSTRAINTS = {
    "perf_stat.IPC": FeatureConstraint(
        feature_name="perf_stat.IPC",
        group="ipc",
        priority_rank=1,
        tolerance_ratio=0.05,
        importance_weight=1.0,
    ),
    "perf_stat.L1-icache-load-misses_MPKI": FeatureConstraint(
        feature_name="perf_stat.L1-icache-load-misses_MPKI",
        group="frontend_icache",
        priority_rank=2,
        tolerance_ratio=0.08,
        importance_weight=1.0,
    ),
    "perf_stat.branch-misses_MPKI": FeatureConstraint(
        feature_name="perf_stat.branch-misses_MPKI",
        group="frontend_branch_predictor",
        priority_rank=2,
        tolerance_ratio=0.08,
        importance_weight=0.7,
    ),
    "perf_stat.branch-load-misses_MPKI": FeatureConstraint(
        feature_name="perf_stat.branch-load-misses_MPKI",
        group="btb_proxy",
        priority_rank=3,
        tolerance_ratio=0.15,
        importance_weight=0.5,
    ),
    "perf_stat.L1-dcache-load-misses_MPKI": FeatureConstraint(
        feature_name="perf_stat.L1-dcache-load-misses_MPKI",
        group="data",
        priority_rank=3,
        tolerance_ratio=0.15,
        importance_weight=0.5,
    ),
    "perf_stat.L1-dcache-loads_LPKI": FeatureConstraint(
        feature_name="perf_stat.L1-dcache-loads_LPKI",
        group="data",
        priority_rank=3,
        tolerance_ratio=0.15,
        importance_weight=0.7,
    ),
    "perf_stat.L1-dcache-stores_SPKI": FeatureConstraint(
        feature_name="perf_stat.L1-dcache-stores_SPKI",
        group="data",
        priority_rank=3,
        tolerance_ratio=0.15,
        importance_weight=0.5,
    ),
    "perf_stat.LLC-load-misses_MPKI": FeatureConstraint(
        feature_name="perf_stat.LLC-load-misses_MPKI",
        group="llc_reuse",
        priority_rank=3,
        tolerance_ratio=0.15,
        importance_weight=0.7,
    ),
    "perf_stat.iTLB-load-misses_MPKI": FeatureConstraint(
        feature_name="perf_stat.iTLB-load-misses_MPKI",
        group="tlb_footprint",
        priority_rank=3,
        tolerance_ratio=0.20,
        importance_weight=0.8,
    ),
    "perf_stat.L2-icache-load-misses_MPKI": FeatureConstraint(
        feature_name="perf_stat.L2-icache-load-misses_MPKI",
        group="frontend_icache",
        priority_rank=3,
        tolerance_ratio=0.15,
        importance_weight=0.4,
    ),
    "tma.TMA_Frontend_Bound(%)": FeatureConstraint(
        feature_name="tma.TMA_Frontend_Bound(%)",
        group="topdown_l1",
        priority_rank=4,
        tolerance_ratio=0.20,
        importance_weight=0.3,
    ),
    "tma.TMA_Backend_Bound(%)": FeatureConstraint(
        feature_name="tma.TMA_Backend_Bound(%)",
        group="topdown_l1",
        priority_rank=4,
        tolerance_ratio=0.20,
        importance_weight=0.3,
    ),
    "tma.TMA_Bad_Speculation(%)": FeatureConstraint(
        feature_name="tma.TMA_Bad_Speculation(%)",
        group="topdown_l1",
        priority_rank=4,
        tolerance_ratio=0.20,
        importance_weight=0.3,
    ),
    "tma.TMA_Retiring(%)": FeatureConstraint(
        feature_name="tma.TMA_Retiring(%)",
        group="topdown_l1",
        priority_rank=4,
        tolerance_ratio=0.20,
        importance_weight=0.3,
    ),
}


# ---------------------------------------------------------------------------
# Backend-bound profile.
# Tiers mirror the backend top-down (TMA) drill-down hierarchy:
#   T0: IPC, tma_memory_bound, tma_core_bound
#   T1: L1-dcache-load-misses MPKI, tma_dram_bound, tma_l1/l2/l3_bound
#   T2: dTLB-load-misses MPKI, LLC-load-misses MPKI
# IPC stays the tightest/primary signal and must not be sacrificed to match
# lower tiers.
# ---------------------------------------------------------------------------
_BACKEND_TIER_LABELS = {
    1: "Tier T0 IPC/memory-core bound",
    2: "Tier T1 cache bound (L1d MPKI, DRAM/L1/L2/L3 bound)",
    3: "Tier T2 TLB/LLC (dTLB-load MPKI, LLC-load MPKI)",
}

_BACKEND_CONSTRAINTS = {
    # --- T0 ---
    "perf_stat.IPC": FeatureConstraint(
        feature_name="perf_stat.IPC",
        group="ipc",
        priority_rank=1,
        tolerance_ratio=0.05,
        importance_weight=1.0,
    ),
    "tma.TMA_Memory_Bound(%)": FeatureConstraint(
        feature_name="tma.TMA_Memory_Bound(%)",
        group="topdown_l2_backend",
        priority_rank=1,
        tolerance_ratio=0.10,
        importance_weight=0.9,
    ),
    "tma.TMA_Core_Bound(%)": FeatureConstraint(
        feature_name="tma.TMA_Core_Bound(%)",
        group="topdown_l2_backend",
        priority_rank=1,
        tolerance_ratio=0.10,
        importance_weight=0.7,
    ),
    # --- T1 ---
    "perf_stat.L1-dcache-load-misses_MPKI": FeatureConstraint(
        feature_name="perf_stat.L1-dcache-load-misses_MPKI",
        group="data",
        priority_rank=2,
        tolerance_ratio=0.10,
        importance_weight=0.8,
    ),
    "tma.TMA_DRAM_Bound(%)": FeatureConstraint(
        feature_name="tma.TMA_DRAM_Bound(%)",
        group="topdown_l3_memory",
        priority_rank=2,
        tolerance_ratio=0.15,
        importance_weight=0.8,
    ),
    "tma.TMA_L1_Bound(%)": FeatureConstraint(
        feature_name="tma.TMA_L1_Bound(%)",
        group="topdown_l3_memory",
        priority_rank=2,
        tolerance_ratio=0.15,
        importance_weight=0.6,
    ),
    "tma.TMA_L2_Bound(%)": FeatureConstraint(
        feature_name="tma.TMA_L2_Bound(%)",
        group="topdown_l3_memory",
        priority_rank=2,
        tolerance_ratio=0.15,
        importance_weight=0.5,
    ),
    "tma.TMA_L3_Bound(%)": FeatureConstraint(
        feature_name="tma.TMA_L3_Bound(%)",
        group="topdown_l3_memory",
        priority_rank=2,
        tolerance_ratio=0.15,
        importance_weight=0.6,
    ),
    # --- T2 ---
    "perf_stat.dTLB-load-misses_MPKI": FeatureConstraint(
        feature_name="perf_stat.dTLB-load-misses_MPKI",
        group="tlb_footprint",
        priority_rank=3,
        tolerance_ratio=0.20,
        importance_weight=0.6,
    ),
    "perf_stat.LLC-load-misses_MPKI": FeatureConstraint(
        feature_name="perf_stat.LLC-load-misses_MPKI",
        group="llc_reuse",
        priority_rank=3,
        tolerance_ratio=0.20,
        importance_weight=0.7,
    ),
}


_FRONTEND_PROMPT_FRAGMENT = (
    "Regeneration must prioritize hardware-counter similarity in this order: "
    "Tier T0 IPC within +/-5% as the primary fidelity signal; "
    "Tier T1 frontend counters within +/-8% with L1-icache-load-misses_MPKI ahead of branch-misses_MPKI; "
    "Tier T2 within +/-12% for BTB proxy branch-load-misses_MPKI and data-cache counters "
    "(L1-dcache-load-misses_MPKI, L1-dcache-loads_LPKI, L1-dcache-stores_SPKI); "
    "Tier T3 within +/-20% for iTLB-load-misses_MPKI and the top-down L1 breakdown "
    "(TMA_Frontend_Bound(%), TMA_Backend_Bound(%), TMA_Bad_Speculation(%), TMA_Retiring(%)). "
    "Do not sacrifice IPC to match lower-tier counters."
)

_BACKEND_PROMPT_FRAGMENT = (
    "Regeneration must prioritize hardware-counter similarity in this order: "
    "Tier T0 IPC within +/-5% as the primary fidelity signal, alongside the backend top-down split "
    "TMA_Memory_Bound(%) and TMA_Core_Bound(%) within +/-10%; "
    "Tier T1 within +/-15% for the memory-bound drill-down TMA_DRAM_Bound(%), TMA_L1_Bound(%), "
    "TMA_L2_Bound(%), TMA_L3_Bound(%), and within +/-10% for L1-dcache-load-misses_MPKI; "
    "Tier T2 within +/-20% for dTLB-load-misses_MPKI and LLC-load-misses_MPKI. "
    "Do not sacrifice IPC to match lower-tier counters."
)


@dataclass(frozen=True)
class PriorityProfile:
    name: str
    constraints: dict[str, FeatureConstraint]
    tier_labels: dict[int, str]
    prompt_fragment: str


_PROFILES: dict[str, PriorityProfile] = {
    "frontend": PriorityProfile(
        name="frontend",
        constraints=_FRONTEND_CONSTRAINTS,
        tier_labels=_FRONTEND_TIER_LABELS,
        prompt_fragment=_FRONTEND_PROMPT_FRAGMENT,
    ),
    "backend": PriorityProfile(
        name="backend",
        constraints=_BACKEND_CONSTRAINTS,
        tier_labels=_BACKEND_TIER_LABELS,
        prompt_fragment=_BACKEND_PROMPT_FRAGMENT,
    ),
}

# Accept a few friendly aliases for the mode name.
_MODE_ALIASES = {
    "frontend": "frontend",
    "frontend-bound": "frontend",
    "frontend_bound": "frontend",
    "fe": "frontend",
    "backend": "backend",
    "backend-bound": "backend",
    "backend_bound": "backend",
    "be": "backend",
}


def normalize_priority_mode(mode: str | None) -> str:
    if not mode:
        return DEFAULT_PRIORITY_MODE
    key = str(mode).strip().lower()
    return _MODE_ALIASES.get(key, key if key in _PROFILES else DEFAULT_PRIORITY_MODE)


def set_priority_mode(mode: str | None) -> str:
    """Override the active prioritization profile. Pass None to revert to env/default."""
    global _PRIORITY_MODE_OVERRIDE
    _PRIORITY_MODE_OVERRIDE = normalize_priority_mode(mode) if mode is not None else None
    return get_priority_mode()


def get_priority_mode() -> str:
    if _PRIORITY_MODE_OVERRIDE is not None:
        return _PRIORITY_MODE_OVERRIDE
    return normalize_priority_mode(os.environ.get(_PRIORITY_MODE_ENV))


def active_profile() -> PriorityProfile:
    return _PROFILES.get(get_priority_mode(), _PROFILES[DEFAULT_PRIORITY_MODE])


def active_constraints() -> dict[str, FeatureConstraint]:
    return active_profile().constraints


def feature_constraint(feature_name: str) -> FeatureConstraint | None:
    return active_constraints().get(feature_name)


def relative_error(target_value: float, candidate_value: float) -> float:
    denom = max(abs(float(target_value)), EPSILON)
    return abs(float(candidate_value) - float(target_value)) / denom


def tolerance_bounds(target_value: float, tolerance_ratio: float) -> tuple[float, float]:
    radius = abs(float(target_value)) * float(tolerance_ratio)
    center = float(target_value)
    return center - radius, center + radius


def evaluate_priority_constraints(
    target_features: dict[str, Any],
    candidate_features: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    violations_by_priority = {priority: 0 for priority in PRIORITY_ORDER}
    overflow_by_priority = {priority: 0.0 for priority in PRIORITY_ORDER}
    weighted_violations_by_priority = {priority: 0.0 for priority in PRIORITY_ORDER}
    weighted_overflow_by_priority = {priority: 0.0 for priority in PRIORITY_ORDER}

    for feature_name, constraint in active_constraints().items():
        if feature_name not in target_features or feature_name not in candidate_features:
            continue
        target_value = float(target_features[feature_name])
        candidate_value = float(candidate_features[feature_name])
        rel_err = relative_error(target_value, candidate_value)
        within_tolerance = rel_err <= constraint.tolerance_ratio + 1e-12
        overflow_ratio = max(0.0, rel_err - constraint.tolerance_ratio) / max(constraint.tolerance_ratio, EPSILON)
        lower_bound, upper_bound = tolerance_bounds(target_value, constraint.tolerance_ratio)
        if not within_tolerance:
            violations_by_priority[constraint.priority_rank] += 1
            overflow_by_priority[constraint.priority_rank] += overflow_ratio
            weighted_violations_by_priority[constraint.priority_rank] += constraint.importance_weight
            weighted_overflow_by_priority[constraint.priority_rank] += (
                overflow_ratio * constraint.importance_weight
            )
        rows.append(
            {
                "feature_name": feature_name,
                "group": constraint.group,
                "priority_rank": constraint.priority_rank,
                "importance_weight": constraint.importance_weight,
                "target_value": target_value,
                "candidate_value": candidate_value,
                "delta": candidate_value - target_value,
                "relative_error": rel_err,
                "relative_error_pct": rel_err * 100.0,
                "tolerance_ratio": constraint.tolerance_ratio,
                "tolerance_pct": constraint.tolerance_ratio * 100.0,
                "within_tolerance": within_tolerance,
                "acceptable_min": lower_bound,
                "acceptable_max": upper_bound,
                "overflow_ratio": overflow_ratio,
            }
        )

    rows.sort(
        key=lambda row: (
            int(not bool(row["within_tolerance"])),
            -int(row["priority_rank"]),
            float(row.get("importance_weight", 0.0)),
            float(row["overflow_ratio"]),
            float(row["relative_error"]),
        ),
        reverse=True,
    )
    checked_feature_count = len(rows)
    violated_feature_count = sum(violations_by_priority.values())
    return {
        "checked_feature_count": checked_feature_count,
        "violated_feature_count": violated_feature_count,
        "all_within_tolerance": checked_feature_count > 0 and violated_feature_count == 0,
        "violations_by_priority": violations_by_priority,
        "overflow_by_priority": overflow_by_priority,
        "weighted_violations_by_priority": weighted_violations_by_priority,
        "weighted_overflow_by_priority": weighted_overflow_by_priority,
        "feature_rows": rows,
    }


def candidate_rank_key(
    constraint_report: dict[str, Any],
    weighted_error: float,
) -> tuple[Any, ...]:
    weighted_violations = constraint_report.get("weighted_violations_by_priority", {})
    weighted_overflow = constraint_report.get("weighted_overflow_by_priority", {})
    return (
        *(float(weighted_violations.get(priority, 0.0)) for priority in PRIORITY_ORDER),
        *(float(weighted_overflow.get(priority, 0.0)) for priority in PRIORITY_ORDER),
        float(weighted_error) if math.isfinite(float(weighted_error)) else float("inf"),
    )


def prioritized_gap_key(row: dict[str, Any]) -> tuple[Any, ...]:
    priority_rank = int(row.get("priority_rank", 999))
    within_tolerance = bool(row.get("within_tolerance", False))
    return (
        0 if (priority_rank < 999 and not within_tolerance) else 1 if priority_rank < 999 else 2,
        priority_rank,
        -float(row.get("overflow_ratio", 0.0)),
        -float(row.get("normalized_abs_delta", 0.0)),
    )


def priority_tier_label(priority_rank: int | None) -> str:
    if priority_rank is None:
        return "No active tier"
    try:
        numeric = int(priority_rank)
    except (TypeError, ValueError):
        return f"Tier {priority_rank}"
    return active_profile().tier_labels.get(numeric, f"Tier T{max(numeric - 1, 0)}")


def first_unsatisfied_priority(feature_rows: list[dict[str, Any]]) -> int | None:
    for priority in PRIORITY_ORDER:
        for row in feature_rows:
            if int(row.get("priority_rank", 999) or 999) != priority:
                continue
            if not bool(row.get("within_tolerance")):
                return priority
    return None


def priority_tier_state(feature_rows: list[dict[str, Any]]) -> dict[str, Any]:
    relevant_rows = [
        row
        for row in feature_rows
        if int(row.get("priority_rank", 999) or 999) in PRIORITY_ORDER
    ]
    active_priority_rank = first_unsatisfied_priority(relevant_rows)

    def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            rows,
            key=lambda row: (
                int(not bool(row.get("within_tolerance"))),
                float(row.get("overflow_ratio", 0.0) or 0.0) * float(row.get("importance_weight", 1.0) or 1.0),
                float(row.get("overflow_ratio", 0.0) or 0.0),
                float(row.get("importance_weight", 1.0) or 1.0),
                -int(row.get("priority_rank", 999) or 999),
                float(row.get("relative_error", 0.0) or 0.0),
            ),
            reverse=True,
        )

    matched_rows = [row for row in relevant_rows if bool(row.get("within_tolerance"))]
    if active_priority_rank is None:
        return {
            "active_priority_rank": None,
            "active_tier_label": priority_tier_label(None),
            "higher_priority_matched_rows": sort_rows(matched_rows),
            "active_rows": [],
            "active_focus_rows": [],
            "active_protected_rows": [],
            "deferred_rows": [],
        }

    higher_priority_matched_rows = sort_rows(
        [
            row
            for row in relevant_rows
            if int(row.get("priority_rank", 999) or 999) < active_priority_rank
            and bool(row.get("within_tolerance"))
        ]
    )
    active_rows = sort_rows(
        [
            row
            for row in relevant_rows
            if int(row.get("priority_rank", 999) or 999) == active_priority_rank
        ]
    )
    active_focus_rows = [row for row in active_rows if not bool(row.get("within_tolerance"))]
    active_protected_rows = [row for row in active_rows if bool(row.get("within_tolerance"))]
    deferred_rows = sort_rows(
        [
            row
            for row in relevant_rows
            if int(row.get("priority_rank", 999) or 999) > active_priority_rank
        ]
    )
    return {
        "active_priority_rank": active_priority_rank,
        "active_tier_label": priority_tier_label(active_priority_rank),
        "higher_priority_matched_rows": higher_priority_matched_rows,
        "active_rows": active_rows,
        "active_focus_rows": active_focus_rows,
        "active_protected_rows": active_protected_rows,
        "deferred_rows": deferred_rows,
    }


def hierarchy_prompt_fragment() -> str:
    return active_profile().prompt_fragment
