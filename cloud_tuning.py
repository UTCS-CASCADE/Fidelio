from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.benchmark_sources import (
    parse_benchmark_binary_sources,
    resolve_build_timeout_sec,
    resolve_preferred_build_dir,
)
from scripts.cloning_priority import (
    active_constraints as _active_constraints,
    candidate_rank_key,
    evaluate_priority_constraints,
    feature_constraint,
    priority_tier_label,
    priority_tier_state,
    prioritized_gap_key,
    tolerance_bounds,
)
from agents.diagnostician import build_diagnostician_report
from scripts.feature_prompt_guidance import feature_prompt_guidance
from scripts.feature_schema import (
    CANONICAL_FEATURE_COLUMNS,
    absolute_match_report,
    canonicalize_feature_dict,
    feature_display_name,
    load_feature_table,
    resolve_feature_table_path,
)
from folly_edit_rag.scripts.local_patch_utils import validate_generated_source_output
from scripts.run_feature_extraction_end_to_end import (
    choose_unit_cmd,
    collect_application_seed_features,
    collect_unit_features,
    is_application_seed_unit,
    list_microbenchmarks,
    measure_unit_runtime,
)

REPO_ROOT = Path(__file__).resolve().parent
FOLLY_EDIT_RAG_SCRIPTS = REPO_ROOT / "folly_edit_rag" / "scripts"
PROPERTY_TASKS_DIR = REPO_ROOT / "folly_edit_rag" / "outputs" / "cloud_tuning"
PROPERTY_TASKS_MANIFEST = (
    REPO_ROOT / "folly_edit_rag" / "data" / "property_edit_tasks" / "property_edit_tasks.jsonl"
)
PROPERTY_TASKS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_XLSX = resolve_feature_table_path(REPO_ROOT / "search_engine" / "all_features.xlsx")

_PATCHES_DIR = FOLLY_EDIT_RAG_SCRIPTS / "outputs" / "patches"
_COMPILE_LOGS_DIR = FOLLY_EDIT_RAG_SCRIPTS / "outputs" / "compile_logs"


def _read_token_usage_sidecar(path: Path) -> dict[str, int]:
    """Read a token-usage sidecar JSON written by llm_provider.save_token_usage()."""
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return {
                "input_tokens": int(data.get("input_tokens") or 0),
                "output_tokens": int(data.get("output_tokens") or 0),
                "total_tokens": int(data.get("total_tokens") or 0),
                "call_count": int(data.get("call_count") or 0),
            }
    except Exception:
        pass
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "call_count": 0}


def _accumulate_token_usage(target: dict[str, int], source: dict[str, int]) -> None:
    for key in ("input_tokens", "output_tokens", "total_tokens", "call_count"):
        target[key] = target.get(key, 0) + source.get(key, 0)


def _estimated_llm_cost_usd(usage: dict[str, int]) -> float:
    input_rate = float(os.environ.get("VERTEX_INPUT_USD_PER_MILLION_TOKENS", "2.0"))
    output_rate = float(os.environ.get("VERTEX_OUTPUT_USD_PER_MILLION_TOKENS", "12.0"))
    return (
        int(usage.get("input_tokens", 0)) * input_rate
        + int(usage.get("output_tokens", 0)) * output_rate
    ) / 1_000_000.0


HW_COUNTER_FEATURE_NAMES: tuple[str, ...] = (
    "perf_stat.IPC",
    "perf_stat.L1-icache-load-misses_MPKI",
    "perf_stat.branch-misses_MPKI",
    "perf_stat.branch-load-misses_MPKI",
    "perf_stat.L1-dcache-load-misses_MPKI",
    "perf_stat.L1-dcache-loads_LPKI",
    "perf_stat.L1-dcache-stores_SPKI",
    "perf_stat.LLC-load-misses_MPKI",
    "perf_stat.iTLB-load-misses_MPKI",
)
FRONTEND_CONTROL_FEATURE_NAMES: tuple[str, ...] = (
    "perf_stat.L1-icache-load-misses_MPKI",
    "perf_stat.branch-misses_MPKI",
    "perf_stat.branch-load-misses_MPKI",
    "perf_stat.iTLB-load-misses_MPKI",
)
MEMORY_SIDE_FEATURE_NAMES: tuple[str, ...] = (
    "perf_stat.L1-dcache-load-misses_MPKI",
    "perf_stat.L1-dcache-loads_LPKI",
    "perf_stat.L1-dcache-stores_SPKI",
)
MAGNITUDE_BANDS = {
    "minimally": (0.10, 0.30),
    "moderately": (0.30, 0.60),
    "substantially": (0.60, 1.00),
    "significantly": (1.00, None),
}
MAGNITUDE_TARGET_FACTORS = {
    "minimally": 0.20,
    "moderately": 0.45,
    "substantially": 0.80,
    "significantly": 1.20,
}
MAX_ADJUSTMENT_VERSIONS = int(os.environ.get("CLOUD_MAX_ADJUSTMENT_VERSIONS", "15"))  # legacy compat
MAX_ATTEMPTS_PER_VARIANT = int(os.environ.get("CLOUD_MAX_ATTEMPTS_PER_VARIANT", "8"))
MAX_COMPILE_REPAIR_ATTEMPTS = int(os.environ.get("CLOUD_MAX_COMPILE_REPAIR_ATTEMPTS", "20"))
MAX_LLM_COST_USD_PER_SEGMENT = float(os.environ.get("CLOUD_MAX_LLM_COST_USD_PER_SEGMENT", "0"))
MATCH_ERROR_THRESHOLD = float(os.environ.get("MATCH_ERROR_THRESHOLD", "0.85"))
SOURCE_SWITCH_MIN_ATTEMPTS = int(os.environ.get("CLOUD_SOURCE_SWITCH_MIN_ATTEMPTS", "4"))
STAGNANT_ATTEMPT_WINDOW = int(os.environ.get("CLOUD_STAGNANT_ATTEMPT_WINDOW", "3"))
STOP_ON_STAGNATION = os.environ.get("CLOUD_STOP_ON_STAGNATION", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MATERIAL_IMPROVEMENT_ABS = float(os.environ.get("CLOUD_MATERIAL_IMPROVEMENT_ABS", "0.01"))
MATERIAL_IMPROVEMENT_REL = float(os.environ.get("CLOUD_MATERIAL_IMPROVEMENT_REL", "0.02"))
FAR_SOURCE_BASELINE_ERROR = float(os.environ.get("CLOUD_FAR_SOURCE_BASELINE_ERROR", "0.60"))
FAR_SOURCE_VIOLATIONS = int(os.environ.get("CLOUD_FAR_SOURCE_VIOLATIONS", "8"))
FROM_SCRATCH_IF_BASELINE_LOW = os.environ.get("CLOUD_FROM_SCRATCH_IF_BASELINE_LOW", "0").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}
FROM_SCRATCH_BASELINE_HW_THRESHOLD = float(
    os.environ.get("CLOUD_FROM_SCRATCH_BASELINE_HW_THRESHOLD", "50.0")
)
SEED_BASELINE_SELECTION_MODE = os.environ.get("SEED_BASELINE_SELECTION_MODE", "").strip().lower()
IPC_SAFE_LOCAL_EDIT_ATTEMPTS = max(
    1, int(os.environ.get("CLOUD_IPC_SAFE_LOCAL_EDIT_ATTEMPTS", "2"))
)
IPC_GUARDRAIL_MAX_EXCESS_OVER_TOLERANCE = float(
    os.environ.get("CLOUD_IPC_GUARDRAIL_MAX_EXCESS_OVER_TOLERANCE", "0.12")
)
IPC_GUARDRAIL_MAX_WORSENING = float(
    os.environ.get("CLOUD_IPC_GUARDRAIL_MAX_WORSENING", "0.06")
)
STRICT_IPC_GUARDRAIL = os.environ.get("CLOUD_STRICT_IPC_GUARDRAIL", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MIN_CANDIDATE_IPC_ACCURACY = float(
    os.environ.get("CLOUD_MIN_CANDIDATE_IPC_ACCURACY", "0")
)
_IPC_COUPLED_MIN_ATTEMPTS = int(os.environ.get("CLOUD_IPC_COUPLED_MIN_ATTEMPTS", "8"))
PER_FEATURE_IMPROVEMENT_PCT = float(os.environ.get("CLOUD_PER_FEATURE_IMPROVEMENT_PCT", "0.5"))
DEFAULT_CMAKE_PARALLEL = int(os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "16"))
REGISTRATION_ONLY_ALIAS_REASON = "Generated microbenchmark is a registration-only alias."
REQUEST_BUDGET_SLACK_SECONDS = float(os.environ.get("CLOUD_REQUEST_BUDGET_SLACK_SECONDS", "30"))
MIN_GENERATE_BUDGET_SECONDS = float(os.environ.get("CLOUD_MIN_GENERATE_BUDGET_SECONDS", "45"))
MIN_COMPILE_BUDGET_SECONDS = float(os.environ.get("CLOUD_MIN_COMPILE_BUDGET_SECONDS", "120"))
MIN_EVALUATE_BUDGET_SECONDS = float(os.environ.get("CLOUD_MIN_EVALUATE_BUDGET_SECONDS", "45"))
GENERATION_QUICK_RETRIES = int(os.environ.get("CLOUD_GENERATION_QUICK_RETRIES", "2"))

ADAPTIVE_TIER_STAGNATION_WINDOW = int(os.environ.get("CLOUD_ADAPTIVE_TIER_STAGNATION_WINDOW", "3"))
ADAPTIVE_TIER_TOLERANCE_BOOST = float(os.environ.get("CLOUD_ADAPTIVE_TIER_TOLERANCE_BOOST", "1.5"))

TIER_ESCALATION_WINDOW = int(os.environ.get("CLOUD_TIER_ESCALATION_WINDOW", "8"))

TWO_PHASE_T0_MODEL = os.environ.get("CLOUD_TWO_PHASE_T0_MODEL", "").strip()
TWO_PHASE_EXPLORATION_MODEL = os.environ.get("CLOUD_TWO_PHASE_EXPLORATION_MODEL", "").strip()
TWO_PHASE_ENABLED = bool(TWO_PHASE_T0_MODEL and TWO_PHASE_EXPLORATION_MODEL)


def _two_phase_model_env(objective_priority_rank: int | None) -> dict[str, str] | None:
    """Return env overrides for the two-phase model strategy, or None if disabled."""
    if not TWO_PHASE_ENABLED:
        return None
    # T0 = priority rank 1 (IPC).  Use the strong model.
    # Everything else = use the fast exploration model.
    model = TWO_PHASE_T0_MODEL if objective_priority_rank == 1 else TWO_PHASE_EXPLORATION_MODEL
    return {
        "VERTEX_MODEL_NAME": model,
        "VERTEX_GENERATION_MODEL_NAME": model,
        "VERTEX_REPAIR_MODEL_NAME": model,
    }


def _deadline_from_wall_seconds(max_wall_seconds: float | None) -> float | None:
    if max_wall_seconds is None:
        return None
    try:
        wall_seconds = float(max_wall_seconds)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(wall_seconds) or wall_seconds <= 0.0:
        return None
    return time.monotonic() + wall_seconds


def _remaining_wall_seconds(deadline_monotonic: float | None) -> float | None:
    if deadline_monotonic is None:
        return None
    return max(0.0, float(deadline_monotonic) - time.monotonic())


def _available_step_budget_seconds(
    deadline_monotonic: float | None,
    *,
    reserve_seconds: float = 0.0,
) -> float | None:
    remaining = _remaining_wall_seconds(deadline_monotonic)
    if remaining is None:
        return None
    return max(0.0, remaining - max(0.0, reserve_seconds))


def _wall_budget_stop_reason(
    step_name: str,
    *,
    remaining_seconds: float | None,
    reserve_seconds: float = 0.0,
) -> str:
    if remaining_seconds is None:
        return f"Request wall-clock budget exhausted before {step_name}."
    return (
        f"Request wall-clock budget exhausted before {step_name} "
        f"(remaining={remaining_seconds:.1f}s, reserved_slack={reserve_seconds:.1f}s)."
    )


def _hardware_counter_features(features: dict[str, Any]) -> dict[str, float]:
    canonical = canonicalize_feature_dict({str(k): v for k, v in (features or {}).items()})
    return {
        name: float(canonical[name])
        for name in HW_COUNTER_FEATURE_NAMES
        if name in canonical
    }


SCALE_TARGET_FLOOR_RATIO = float(os.environ.get("CLOUD_SCALE_TARGET_FLOOR_RATIO", "0.50"))


def _hardware_counter_match_report(
    target_features: dict[str, Any],
    candidate_features: dict[str, Any],
    feature_scales: dict[str, float],
) -> dict[str, object]:
    adjusted_scales = feature_scales
    if SCALE_TARGET_FLOOR_RATIO > 0:
        adjusted_scales = dict(feature_scales)
        for name in HW_COUNTER_FEATURE_NAMES:
            target_val = target_features.get(name)
            if target_val is None:
                continue
            try:
                target_abs = abs(float(target_val))
            except (TypeError, ValueError):
                continue
            if target_abs > 1e-9:
                adjusted_scales[name] = max(
                    adjusted_scales.get(name, 1.0),
                    target_abs * SCALE_TARGET_FLOOR_RATIO,
                )
    return absolute_match_report(
        target_features,
        candidate_features,
        adjusted_scales,
        feature_names=HW_COUNTER_FEATURE_NAMES,
    )


def _hardware_counter_constraint_report(
    target_features: dict[str, Any],
    candidate_features: dict[str, Any],
) -> dict[str, Any]:
    return evaluate_priority_constraints(
        _hardware_counter_features(target_features),
        _hardware_counter_features(candidate_features),
    )


def _relaxed_constraint_report(
    target_features: dict[str, Any],
    candidate_features: dict[str, Any],
    active_priority_rank: int,
    tolerance_boost: float,
) -> dict[str, Any]:
    'Re-evaluate constraints with inflated tolerance for *active_priority_rank*.'
    target = _hardware_counter_features(target_features)
    candidate = _hardware_counter_features(candidate_features)
    base_report = evaluate_priority_constraints(target, candidate)
    if tolerance_boost <= 1.0:
        return base_report
    # Patch the feature rows belonging to the active tier.
    patched_rows: list[dict[str, Any]] = []
    patched_violation_delta = 0
    for row in list(base_report.get("feature_rows") or []):
        if int(row.get("priority_rank", 999) or 999) != active_priority_rank:
            patched_rows.append(row)
            continue
        new_tol = float(row["tolerance_ratio"]) * tolerance_boost
        rel_err = float(row["relative_error"])
        was_within = bool(row["within_tolerance"])
        now_within = rel_err <= new_tol + 1e-12
        overflow = max(0.0, rel_err - new_tol) / max(new_tol, 1e-9)
        patched_row = dict(row)
        patched_row["tolerance_ratio"] = new_tol
        patched_row["tolerance_pct"] = new_tol * 100.0
        patched_row["within_tolerance"] = now_within
        patched_row["overflow_ratio"] = overflow
        target_val = float(row["target_value"])
        radius = abs(target_val) * new_tol
        patched_row["acceptable_min"] = target_val - radius
        patched_row["acceptable_max"] = target_val + radius
        patched_rows.append(patched_row)
        if not was_within and now_within:
            patched_violation_delta -= 1
    patched_report = dict(base_report)
    patched_report["feature_rows"] = patched_rows
    new_violated = int(base_report.get("violated_feature_count", 0)) + patched_violation_delta
    patched_report["violated_feature_count"] = max(new_violated, 0)
    patched_report["all_within_tolerance"] = (
        len(patched_rows) > 0 and max(new_violated, 0) == 0
    )
    # Update per-priority violation/overflow aggregates.
    violations_by_p = dict(base_report.get("violations_by_priority") or {})
    overflow_by_p = dict(base_report.get("overflow_by_priority") or {})
    w_violations_by_p = dict(base_report.get("weighted_violations_by_priority") or {})
    w_overflow_by_p = dict(base_report.get("weighted_overflow_by_priority") or {})
    # Recompute for the active tier only.
    tier_violations = 0
    tier_overflow = 0.0
    tier_w_violations = 0.0
    tier_w_overflow = 0.0
    for row in patched_rows:
        if int(row.get("priority_rank", 999) or 999) != active_priority_rank:
            continue
        if not bool(row["within_tolerance"]):
            tier_violations += 1
            tier_overflow += float(row["overflow_ratio"])
            tier_w_violations += float(row.get("importance_weight", 1.0))
            tier_w_overflow += float(row["overflow_ratio"]) * float(row.get("importance_weight", 1.0))
    violations_by_p[active_priority_rank] = tier_violations
    overflow_by_p[active_priority_rank] = tier_overflow
    w_violations_by_p[active_priority_rank] = tier_w_violations
    w_overflow_by_p[active_priority_rank] = tier_w_overflow
    patched_report["violations_by_priority"] = violations_by_p
    patched_report["overflow_by_priority"] = overflow_by_p
    patched_report["weighted_violations_by_priority"] = w_violations_by_p
    patched_report["weighted_overflow_by_priority"] = w_overflow_by_p
    return patched_report


def _priority_tier_view(
    feature_rows: list[dict[str, Any]],
    *,
    locked_priority_rank: int | None = None,
) -> dict[str, Any]:
    if locked_priority_rank is None:
        tier_state = priority_tier_state(feature_rows)
        tier_state["guardrail_rows"] = list(tier_state.get("higher_priority_matched_rows") or [])
        tier_state["guardrail_violated_rows"] = []
        return tier_state

    relevant_rows = [
        row
        for row in feature_rows
        if int(row.get("priority_rank", 999) or 999) < 999
    ]

    def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            rows,
            key=lambda row: (
                int(not bool(row.get("within_tolerance"))),
                float(row.get("overflow_ratio", 0.0) or 0.0)
                * float(row.get("importance_weight", 1.0) or 1.0),
                float(row.get("overflow_ratio", 0.0) or 0.0),
                float(row.get("importance_weight", 1.0) or 1.0),
                -int(row.get("priority_rank", 999) or 999),
                float(row.get("relative_error", 0.0) or 0.0),
            ),
            reverse=True,
        )

    active_priority_rank = int(locked_priority_rank)
    guardrail_rows = sort_rows(
        [
            row
            for row in relevant_rows
            if int(row.get("priority_rank", 999) or 999) < active_priority_rank
        ]
    )
    guardrail_violated_rows = [row for row in guardrail_rows if not bool(row.get("within_tolerance"))]
    higher_priority_matched_rows = [row for row in guardrail_rows if bool(row.get("within_tolerance"))]
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
        "guardrail_rows": guardrail_rows,
        "guardrail_violated_rows": guardrail_violated_rows,
        "higher_priority_matched_rows": higher_priority_matched_rows,
        "active_rows": active_rows,
        "active_focus_rows": active_focus_rows,
        "active_protected_rows": active_protected_rows,
        "deferred_rows": deferred_rows,
    }


def _locked_objective_priority_rank(
    constraint_report: dict[str, Any],
    *,
    previous_priority_rank: int | None = None,
) -> int | None:
    feature_rows = list(constraint_report.get("feature_rows") or [])
    current_priority_rank = priority_tier_state(feature_rows).get("active_priority_rank")
    if previous_priority_rank is None:
        return current_priority_rank
    try:
        locked_priority_rank = int(previous_priority_rank)
    except (TypeError, ValueError):
        return current_priority_rank
    if current_priority_rank is None:
        return locked_priority_rank if not feature_rows else None
    if locked_priority_rank > current_priority_rank:
        return locked_priority_rank
    if current_priority_rank < locked_priority_rank:
        return current_priority_rank
    locked_rows = [
        row
        for row in feature_rows
        if int(row.get("priority_rank", 999) or 999) == locked_priority_rank
    ]
    if any(not bool(row.get("within_tolerance")) for row in locked_rows):
        return locked_priority_rank
    return current_priority_rank


SOLVED_TIER_MAX_REGRESSION = float(os.environ.get("CLOUD_SOLVED_TIER_MAX_REGRESSION", "0.03"))


def _update_solved_tier_snapshot(
    snapshot: dict[str, float],
    candidate_features: dict[str, Any],
    canonical_target: dict[str, float],
    solved_priority_rank: int,
) -> None:
    """Record the relative errors of features in the just-solved tier."""
    for fname, constraint in _active_constraints().items():
        if constraint.priority_rank != solved_priority_rank:
            continue
        target_val = canonical_target.get(fname)
        cand_val = candidate_features.get(fname)
        if target_val is not None and cand_val is not None:
            try:
                gap = abs(float(cand_val) - float(target_val)) / max(abs(float(target_val)), 1e-9)
            except (TypeError, ValueError):
                continue
            snapshot[fname] = gap


def _solved_tier_regression_check(
    snapshot: dict[str, float],
    candidate_features: dict[str, Any],
    canonical_target: dict[str, float],
) -> str:
    """Check that a candidate doesn't regress features in already-solved tiers.

    Returns an empty string on success, or a reason string on regression.
    """
    for fname, locked_gap in snapshot.items():
        target_val = canonical_target.get(fname)
        cand_val = candidate_features.get(fname)
        if target_val is None or cand_val is None:
            continue
        try:
            new_gap = abs(float(cand_val) - float(target_val)) / max(abs(float(target_val)), 1e-9)
        except (TypeError, ValueError):
            continue
        regression = new_gap - locked_gap
        if regression > SOLVED_TIER_MAX_REGRESSION + 1e-12:
            return (
                f"Candidate regresses solved-tier feature {fname} "
                f"(locked gap {locked_gap * 100:.2f}% -> {new_gap * 100:.2f}%, "
                f"allowed regression {SOLVED_TIER_MAX_REGRESSION * 100:.1f}%)."
            )
    return ""


def _priority_rank_score(
    constraint_report: dict[str, Any],
    priority_rank: int | None,
) -> tuple[float, float]:
    if priority_rank is None:
        return (0.0, 0.0)
    try:
        numeric_rank = int(priority_rank)
    except (TypeError, ValueError):
        return (float("inf"), float("inf"))
    weighted_violations = constraint_report.get("weighted_violations_by_priority", {})
    weighted_overflow = constraint_report.get("weighted_overflow_by_priority", {})
    return (
        float(weighted_violations.get(numeric_rank, 0.0) or 0.0),
        float(weighted_overflow.get(numeric_rank, 0.0) or 0.0),
    )


def _objective_tier_improvement_report(
    *,
    anchor_constraint_report: dict[str, Any],
    candidate_constraint_report: dict[str, Any],
    objective_priority_rank: int | None,
) -> dict[str, Any]:
    if objective_priority_rank is None:
        return {
            "passed": True,
            "objective_priority_rank": None,
            "anchor_score": [0.0, 0.0],
            "candidate_score": [0.0, 0.0],
            "reason": "",
        }
    anchor_score = _priority_rank_score(anchor_constraint_report, objective_priority_rank)
    candidate_score = _priority_rank_score(candidate_constraint_report, objective_priority_rank)
    passed = candidate_score < anchor_score
    return {
        "passed": passed,
        "objective_priority_rank": int(objective_priority_rank),
        "objective_tier_label": priority_tier_label(objective_priority_rank),
        "anchor_score": list(anchor_score),
        "candidate_score": list(candidate_score),
        "reason": (
            ""
            if passed
            else (
                f"{priority_tier_label(objective_priority_rank)} did not improve "
                f"(anchor score {anchor_score[0]:.3f}/{anchor_score[1]:.3f}, "
                f"candidate score {candidate_score[0]:.3f}/{candidate_score[1]:.3f})."
            )
        ),
    }


def _ipc_guardrail_report(
    *,
    canonical_target: dict[str, float],
    anchor_features: dict[str, Any],
    candidate_features: dict[str, Any],
    ipc_coupled: bool = False,
) -> dict[str, Any]:
    ipc_constraint = feature_constraint("perf_stat.IPC")
    tolerance_ratio = float(ipc_constraint.tolerance_ratio) if ipc_constraint is not None else 0.05
    anchor_gap = _feature_relative_gap(canonical_target, anchor_features, "perf_stat.IPC")
    candidate_gap = _feature_relative_gap(canonical_target, candidate_features, "perf_stat.IPC")
    if ipc_coupled and not STRICT_IPC_GUARDRAIL:
        max_allowed_gap = max(
            (anchor_gap or 0.0) + 0.10,
            tolerance_ratio + IPC_GUARDRAIL_MAX_EXCESS_OVER_TOLERANCE,
            1.0,  # at least 100 % relative error allowed
        )
    else:
        max_allowed_gap = tolerance_ratio + IPC_GUARDRAIL_MAX_EXCESS_OVER_TOLERANCE
    if MIN_CANDIDATE_IPC_ACCURACY > 0:
        max_allowed_gap = min(
            max_allowed_gap,
            max(0.0, 1.0 - MIN_CANDIDATE_IPC_ACCURACY),
        )
    passed = True
    reason_parts: list[str] = []

    if candidate_gap is None:
        passed = False
        reason_parts.append("IPC guardrail could not be evaluated because candidate IPC is missing.")
    else:
        if candidate_gap > max_allowed_gap + 1e-12:
            passed = False
            reason_parts.append(
                f"IPC relative error {candidate_gap * 100.0:.2f}% exceeded the hard ceiling of {max_allowed_gap * 100.0:.2f}%."
            )
        if (not ipc_coupled or STRICT_IPC_GUARDRAIL) and anchor_gap is not None and candidate_gap > anchor_gap + IPC_GUARDRAIL_MAX_WORSENING + 1e-12:
            passed = False
            reason_parts.append(
                f"IPC relative error worsened from {anchor_gap * 100.0:.2f}% to {candidate_gap * 100.0:.2f}%, beyond the allowed regression of {IPC_GUARDRAIL_MAX_WORSENING * 100.0:.2f}%."
            )

    return {
        "passed": passed,
        "anchor_relative_gap": anchor_gap,
        "candidate_relative_gap": candidate_gap,
        "tolerance_ratio": tolerance_ratio,
        "max_allowed_relative_gap": max_allowed_gap,
        "max_allowed_worsening": IPC_GUARDRAIL_MAX_WORSENING,
        "minimum_candidate_ipc_accuracy": MIN_CANDIDATE_IPC_ACCURACY,
        "reason": " ".join(reason_parts).strip(),
    }


def _slugify(text: str, max_len: int = 96) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")
    slug = "_".join(filter(None, slug.split("_")))
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:10]
    if not slug:
        return digest
    if len(slug) <= max_len:
        return f"{slug}_{digest}"
    keep = max_len - len(digest) - 1
    return f"{slug[:keep].rstrip('_')}_{digest}"


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe_value(item) for item in sorted(value, key=repr)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return item_method()
        except Exception:
            pass
    return value


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe_value(payload), indent=2), encoding="utf-8")


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe_value(row)) + "\n")


def _run_logged(cmd: list[str], log_path: Path, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    child_env = dict(os.environ if env is None else env)
    child_env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(REPO_ROOT / "scripts")]
        + ([child_env["PYTHONPATH"]] if child_env.get("PYTHONPATH") else [])
    )
    label = log_path.stem
    print(f"[cloud-tuning] start {label}: {' '.join(cmd[:3])}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def _pump(stream: Any, chunks: list[str], target: Any) -> None:
        if stream is None:
            return
        for line in stream:
            chunks.append(line)
            target.write(line)
            target.flush()
        stream.close()

    stdout_thread = threading.Thread(target=_pump, args=(proc.stdout, stdout_chunks, sys.stdout), daemon=True)
    stderr_thread = threading.Thread(target=_pump, args=(proc.stderr, stderr_chunks, sys.stderr), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    returncode = proc.wait()
    stdout_thread.join()
    stderr_thread.join()
    completed = subprocess.CompletedProcess(cmd, returncode, "".join(stdout_chunks), "".join(stderr_chunks))
    _json_write(
        log_path,
        {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
    )
    print(f"[cloud-tuning] finish {label}: rc={completed.returncode}", flush=True)
    return completed


def _load_feature_table(xlsx_path: Path = DEFAULT_XLSX) -> pd.DataFrame:
    return load_feature_table(xlsx_path)


def _threshold_met(constraint_report: dict[str, Any], weighted_error: float) -> bool:
    checked_feature_count = int(constraint_report.get("checked_feature_count", 0) or 0)
    if checked_feature_count > 0 and bool(constraint_report.get("all_within_tolerance")):
        return True
    return float(weighted_error) <= MATCH_ERROR_THRESHOLD


def _resolve_binary_path(build_dir: Path, binary_name: str, source_root: Path) -> Path:
    source_info = parse_benchmark_binary_sources().get(binary_name)
    if source_info is not None and source_info.get("binary_path"):
        return Path(source_info["binary_path"]).resolve()
    if "fbthrift" in build_dir.name.lower() or "fbthrift" in str(source_root).lower():
        return (build_dir / "bin" / binary_name).resolve()
    return (build_dir / binary_name).resolve()


def _normalize_build_dir(build_dir: Path, source_root: Path, binary_name: str | None = None) -> Path:
    return resolve_preferred_build_dir(Path(build_dir), Path(source_root), binary_name=binary_name)


def _looks_like_bm_list_fallback(
    binary_name: str,
    listed_microbenchmarks: list[str],
    has_bm_list: bool,
) -> bool:
    unique = sorted(set(str(name).strip() for name in listed_microbenchmarks if str(name).strip()))
    return (not has_bm_list) and unique == [str(binary_name)]


def _capture_baseline_microbenchmarks(
    *,
    binary_path: Path,
    binary_name: str,
    source_microbenchmark: str,
    build_dir: Path,
    source_root: Path,
) -> tuple[set[str], bool]:
    if is_application_seed_unit(source_microbenchmark):
        return {source_microbenchmark}, False
    listed, has_bm_list = list_microbenchmarks(binary_path, timeout_s=60)
    if not _looks_like_bm_list_fallback(binary_name, listed, has_bm_list):
        return set(listed), has_bm_list

    timeout_sec = resolve_build_timeout_sec(binary_name)
    build_cmd = [
        "cmake",
        "--build",
        str(build_dir),
        "--parallel",
        str(DEFAULT_CMAKE_PARALLEL),
        "--target",
        binary_name,
    ]
    print(
        f"[cloud-tuning] baseline --bm_list unavailable for {binary_name}; "
        f"building target once before baseline capture",
        flush=True,
    )
    build_proc = _run_logged(
        build_cmd,
        PROPERTY_TASKS_DIR / "_baseline_bootstrap" / f"{binary_name}.json",
        cwd=FOLLY_EDIT_RAG_SCRIPTS,
    )
    if build_proc.returncode != 0:
        print(
            f"[cloud-tuning] baseline bootstrap build failed for {binary_name}; "
            "falling back to binary-name-only benchmark list",
            flush=True,
        )
        return set(listed), has_bm_list

    listed_after, has_bm_list_after = list_microbenchmarks(binary_path, timeout_s=min(timeout_sec, 120))
    if _looks_like_bm_list_fallback(binary_name, listed_after, has_bm_list_after):
        print(
            f"[cloud-tuning] baseline --bm_list still unavailable for {binary_name} after bootstrap build",
            flush=True,
        )
    else:
        print(
            f"[cloud-tuning] captured {len(set(listed_after))} baseline microbenchmarks for {binary_name}",
            flush=True,
        )
    return set(listed_after), has_bm_list_after


def _rebuild_application_seed_binary(
    *,
    run_dir: Path,
    binary_name: str,
    binary_path: Path,
) -> None:
    source_info = parse_benchmark_binary_sources().get(binary_name)
    if source_info is None or source_info.get("build_system") != "svt_ninja":
        raise RuntimeError(f"No supported application-seed build configuration for {binary_name}")

    build_dir = Path(source_info["build_dir"])
    build_target = str(source_info["build_target"])
    print(
        f"[cloud-tuning] cleaning and rebuilding application seed target {build_target} "
        "before baseline measurement to discard stale candidate objects",
        flush=True,
    )
    clean_proc = _run_logged(
        ["ninja", "-C", str(build_dir), "-t", "clean", "libSvtAv1Enc.so", build_target],
        run_dir / "00_application_seed_baseline_clean.json",
        cwd=Path(source_info["source_root"]),
    )
    if clean_proc.returncode != 0:
        raise RuntimeError(
            f"Application-seed targeted clean failed for {binary_name} with rc={clean_proc.returncode}"
        )
    rebuild_proc = _run_logged(
        ["ninja", "-C", str(build_dir), build_target],
        run_dir / "00_application_seed_baseline_rebuild.json",
        cwd=Path(source_info["source_root"]),
    )
    if rebuild_proc.returncode != 0:
        raise RuntimeError(
            f"Application-seed coherent rebuild failed for {binary_name} "
            f"with rc={rebuild_proc.returncode}"
        )
    if not binary_path.is_file():
        raise RuntimeError(
            f"Application-seed coherent rebuild succeeded but binary is missing: {binary_path}"
        )
    linkage = subprocess.run(
        ["ldd", "-r", str(binary_path)],
        cwd=str(binary_path.parent),
        capture_output=True,
        text=True,
        timeout=60,
    )
    linkage_output = f"{linkage.stdout}\n{linkage.stderr}".strip()
    if (
        linkage.returncode != 0
        or "undefined symbol:" in linkage_output
        or "not found" in linkage_output
    ):
        raise RuntimeError(
            f"Application-seed dynamic linkage is invalid after coherent rebuild: {linkage_output}"
        )


def _trim_feedback_text(text: str, max_lines: int = 24, max_chars: int = 2400) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    lines = cleaned.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    clipped = "\n".join(lines).strip()
    if len(clipped) > max_chars:
        clipped = clipped[-max_chars:]
    return clipped


def _first_nonempty_line(text: str) -> str:
    for line in str(text).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _runtime_estimate_runnable(runtime_estimate: dict[str, Any]) -> bool:
    if bool(runtime_estimate.get("timed_out")):
        return False
    try:
        return int(runtime_estimate.get("returncode")) == 0
    except (TypeError, ValueError):
        return False


def _collect_live_anchor_features(
    *,
    run_dir: Path,
    binary_path: Path,
    microbenchmark_name: str,
    has_bm_list: bool,
    perf_events: str,
    amd: bool,
) -> tuple[dict[str, Any] | None, str]:
    if not is_application_seed_unit(microbenchmark_name):
        try:
            runtime_estimate = measure_unit_runtime(
                binary_path,
                microbenchmark_name,
                has_bm_list=has_bm_list,
            )
        except Exception as exc:
            return None, f"runtime estimate failed: {exc}"
        if not _runtime_estimate_runnable(runtime_estimate):
            return None, "baseline benchmark did not complete a clean preflight run"

    unit_output_dir = run_dir / "feature_eval" / "baseline_anchor" / _slugify(microbenchmark_name)
    try:
        if is_application_seed_unit(microbenchmark_name):
            row = collect_application_seed_features(
                binary_path=binary_path,
                unit_name=microbenchmark_name,
                unit_output_dir=unit_output_dir,
                perf_events=perf_events,
            )
        else:
            row = collect_unit_features(
                binary_path=binary_path,
                unit_name=microbenchmark_name,
                cmd=choose_unit_cmd(binary_path, microbenchmark_name, has_bm_list),
                unit_output_dir=unit_output_dir,
                perfspect_bin="perf",
                perf_events=perf_events,
                skip_intel_pt=(amd or os.environ.get("SEG08_SKIP_PT") == "1"),
            )
    except Exception as exc:
        return None, f"feature extraction failed: {exc}"
    return canonicalize_feature_dict(row), ""


def _select_baseline_features(
    *,
    table_baseline: dict[str, Any],
    live_baseline: dict[str, Any] | None,
    live_baseline_error: str,
    application_seed: bool,
    binary_name: str,
    microbenchmark_name: str,
) -> tuple[dict[str, Any], str]:
    if live_baseline is not None:
        return live_baseline, "live_measurement"
    if application_seed:
        detail = live_baseline_error or "live PMU collection returned no features"
        raise RuntimeError(
            "Live application-seed baseline measurement failed for "
            f"{binary_name}::{microbenchmark_name}; refusing feature-table fallback: {detail}"
        )
    return table_baseline, "feature_table_fallback"


def _registration_only_alias_validation_reason(
    *,
    compile_record: dict[str, Any] | None,
    anchor_source_text: str,
) -> str:
    if not anchor_source_text.strip():
        return ""
    compile_record = compile_record or {}
    final_file_raw = str(compile_record.get("final_file", "")).strip()
    if not final_file_raw:
        return ""
    final_file = Path(final_file_raw)
    if not final_file.exists():
        return ""
    try:
        candidate_text = final_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    validation_reason = validate_generated_source_output(
        candidate_text,
        original_text=anchor_source_text,
    )
    if validation_reason and "registration-only alias" in validation_reason:
        return validation_reason
    return ""


def _feature_relative_gap(
    target_features: dict[str, float],
    candidate_features: dict[str, Any],
    feature_name: str,
) -> float | None:
    if feature_name not in target_features or feature_name not in candidate_features:
        return None
    target_value = float(target_features[feature_name])
    candidate_value = float(candidate_features[feature_name])
    return abs(candidate_value - target_value) / max(abs(target_value), 1e-9)


def _normalized_rank_key(rank_key: Any, *, pad: int = 9) -> tuple[float, ...]:
    values: list[float] = []
    for value in list(rank_key or []):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = float("inf")
        values.append(numeric if math.isfinite(numeric) else float("inf"))
    while len(values) < pad:
        values.append(float("inf"))
    return tuple(values[:pad])


def _working_state_rank_key(
    target_features: dict[str, float],
    candidate_features: dict[str, Any],
    rank_key: Any,
) -> tuple[float, ...]:
    ipc_gap = _feature_relative_gap(target_features, candidate_features, "perf_stat.IPC")
    return (
        *_normalized_rank_key(rank_key),
        float(ipc_gap) if ipc_gap is not None else float("inf"),
    )


def _ipc_within_tolerance_for_features(
    target_features: dict[str, float],
    candidate_features: dict[str, Any],
) -> bool:
    constraint = feature_constraint("perf_stat.IPC")
    if constraint is None:
        return False
    gap = _feature_relative_gap(target_features, candidate_features, "perf_stat.IPC")
    return gap is not None and gap <= constraint.tolerance_ratio + 1e-12


def _corrective_gap_priority_key(row: dict[str, Any], *, freeze_ipc: bool = False) -> tuple[Any, ...]:
    feature_name = str(row.get("feature_name", "")).strip()
    within_tolerance = bool(row.get("within_tolerance", False))
    priority_rank = int(row.get("priority_rank", 999))
    importance_weight = float(row.get("importance_weight", 1.0) or 1.0)
    overflow_ratio = float(row.get("overflow_ratio", 0.0) or 0.0)
    relative_error = float(row.get("relative_error", 0.0) or 0.0)
    normalized_abs_delta = float(row.get("normalized_abs_delta", 0.0) or 0.0)
    severity_score = overflow_ratio * importance_weight
    ipc_frozen_penalty = 1 if freeze_ipc and feature_name == "perf_stat.IPC" else 0
    return (
        0 if not within_tolerance else 1,
        ipc_frozen_penalty,
        -severity_score,
        -overflow_ratio,
        priority_rank,
        -relative_error,
        -normalized_abs_delta,
    )


def _corrective_feature_rows(
    feature_rows: list[dict[str, Any]],
    *,
    freeze_ipc: bool = False,
) -> list[dict[str, Any]]:
    return sorted(
        feature_rows,
        key=lambda row: _corrective_gap_priority_key(row, freeze_ipc=freeze_ipc),
    )


def _selection_error(evaluation: dict[str, Any]) -> float:
    if evaluation.get("selection_eligible") is False:
        return float("inf")
    try:
        value = float(evaluation.get("candidate_error"))
    except (TypeError, ValueError):
        return float("inf")
    return value if math.isfinite(value) else float("inf")


def _evaluation_selection_key(evaluation: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _selection_error(evaluation),
        *_normalized_rank_key(evaluation.get("best_candidate_rank_key")),
    )


def _coupled_evaluation_selection_key(
    evaluation: dict[str, Any],
    canonical_target: dict[str, float],
) -> tuple[Any, ...]:
    'Selection key for IPC-cache/branch coupled runs.'
    base_error = _selection_error(evaluation)
    if not math.isfinite(base_error):
        return (float("inf"),) * 11

    best_candidate = evaluation.get("best_candidate") or {}
    features = best_candidate.get("features") or {}
    ipc_gap = _feature_relative_gap(canonical_target, features, "perf_stat.IPC")
    ipc_penalty = abs(float(ipc_gap)) if ipc_gap is not None and math.isfinite(float(ipc_gap)) else float("inf")

    # Also reward proximity on the key coupled features.
    coupled_feature_gaps: list[float] = []
    for feat in (
        "perf_stat.L1-dcache-load-misses_MPKI",
        "perf_stat.LLC-load-misses_MPKI",
        "perf_stat.branch-misses_MPKI",
        "perf_stat.branch-load-misses_MPKI",
    ):
        gap = _feature_relative_gap(canonical_target, features, feat)
        if gap is not None and math.isfinite(float(gap)):
            coupled_feature_gaps.append(abs(float(gap)))
    mean_coupled_gap = (
        sum(coupled_feature_gaps) / len(coupled_feature_gaps) if coupled_feature_gaps else float("inf")
    )

    return (
        base_error,
        mean_coupled_gap,
        ipc_penalty,
        *_normalized_rank_key(evaluation.get("best_candidate_rank_key")),
    )


def _material_accuracy_improvement(previous_error: float | None, current_error: float | None) -> bool:
    if previous_error is None:
        return True
    if current_error is None or not math.isfinite(current_error):
        return False
    if not math.isfinite(previous_error):
        return True
    improvement = previous_error - current_error
    if improvement <= 1e-12:
        return False
    if improvement >= MATERIAL_IMPROVEMENT_ABS:
        return True
    return (improvement / max(abs(previous_error), 1e-9)) >= MATERIAL_IMPROVEMENT_REL


_COUPLED_PARETO_FEATURES: frozenset[str] = frozenset((
    "perf_stat.IPC",
    "perf_stat.L1-dcache-load-misses_MPKI",
    "perf_stat.LLC-load-misses_MPKI",
    "perf_stat.branch-misses_MPKI",
    "perf_stat.branch-load-misses_MPKI",
))


def _update_per_feature_best(
    per_feature_best: dict[str, float],
    candidate_features: dict[str, Any],
    canonical_target: dict[str, float],
    *,
    ipc_coupled: bool = False,
) -> bool:
    'Update per-feature best-ever accuracy tracker. Return True if any feature improved.'
    any_improved = False
    eps = 1e-12
    for feat, target_val in canonical_target.items():
        if ipc_coupled and feat not in _COUPLED_PARETO_FEATURES:
            continue
        measured = candidate_features.get(feat)
        if measured is None:
            continue
        try:
            measured_f = float(measured)
            target_f = float(target_val)
        except (TypeError, ValueError):
            continue
        accuracy = max(0.0, 1.0 - abs(measured_f - target_f) / max(abs(target_f), eps)) * 100.0
        prev_best = per_feature_best.get(feat)
        if prev_best is None or accuracy > prev_best + PER_FEATURE_IMPROVEMENT_PCT:
            per_feature_best[feat] = accuracy
            any_improved = True
    return any_improved


def _select_diagnostician_attempt(
    attempts: list[dict[str, Any]],
    selected_evaluation: dict[str, Any],
) -> dict[str, Any] | None:
    try:
        selected_attempt_index = int(selected_evaluation.get("attempt_index"))
    except (TypeError, ValueError, AttributeError):
        selected_attempt_index = None
    if selected_attempt_index is not None:
        for attempt in attempts:
            try:
                if int(attempt.get("attempt_index")) == selected_attempt_index:
                    return attempt
            except (TypeError, ValueError, AttributeError):
                continue
    return attempts[-1] if attempts else None


def _stagnation_stop_reason(
    *,
    attempt_index: int,
    baseline_error: float,
    best_candidate_error: float | None,
    last_material_improvement_attempt: int,
    last_per_feature_improvement_attempt: int,
    baseline_constraint_report: dict[str, Any],
    ipc_coupled: bool = False,
) -> str:
    effective_min_attempts = max(SOURCE_SWITCH_MIN_ATTEMPTS, _IPC_COUPLED_MIN_ATTEMPTS) if ipc_coupled else SOURCE_SWITCH_MIN_ATTEMPTS
    if attempt_index < effective_min_attempts:
        return ""

    extended_window = STAGNANT_ATTEMPT_WINDOW * 2
    if (
        attempt_index - last_material_improvement_attempt >= extended_window
        and attempt_index - last_per_feature_improvement_attempt >= extended_window
    ):
        return (
            "No material accuracy improvement (aggregate or any individual feature) in the last "
            f"{extended_window} attempts.  Switch source benchmark or generation strategy to "
            "break out of the plateau."
        )

    if not _source_family_is_far_from_target(
        baseline_error=baseline_error,
        baseline_constraint_report=baseline_constraint_report,
    ):
        return ""
    if best_candidate_error is None or not math.isfinite(best_candidate_error):
        return (
            "Current source benchmark family is far from target and has produced no valid candidate "
            f"after {attempt_index} attempts. "
            "Stop local refinement and switch source benchmark or primary transformation class."
        )
    if best_candidate_error + 1e-12 < baseline_error:
        return ""
    if (attempt_index - last_material_improvement_attempt) < STAGNANT_ATTEMPT_WINDOW:
        return ""
    # Even if aggregate error stagnated, keep going if any individual feature
    # improved recently (Pareto progress).
    if (attempt_index - last_per_feature_improvement_attempt) < STAGNANT_ATTEMPT_WINDOW:
        return ""
    return (
        "Current source benchmark family is not beating the baseline and is no longer improving materially "
        "(neither aggregate error nor any individual feature improved in the last "
        f"{STAGNANT_ATTEMPT_WINDOW} attempts). "
        "Stop local refinement and switch source benchmark or primary transformation class."
    )


def _source_family_is_far_from_target(
    *,
    baseline_error: float,
    baseline_constraint_report: dict[str, Any],
) -> bool:
    return (
        baseline_error >= FAR_SOURCE_BASELINE_ERROR
        or int(baseline_constraint_report.get("violated_feature_count", 0) or 0) >= FAR_SOURCE_VIOLATIONS
    )


def _build_variant_lesson(
    *,
    variant_index: int,
    variant_attempts: list[dict[str, Any]],
    best_variant_evaluation: dict[str, Any] | None,
    best_variant_error: float | None,
    canonical_target: dict[str, Any],
    generation_strategy: str,
    stop_reason: str,
) -> dict[str, Any]:
    """Summarise what a stagnated variant tried and learned, for the next variant."""
    best_candidate = dict((best_variant_evaluation or {}).get("best_candidate") or {})
    best_features = dict(best_candidate.get("features") or {})

    well_matched: list[str] = []
    poorly_matched: list[str] = []
    for feat_name, target_val in canonical_target.items():
        if not isinstance(target_val, (int, float)) or not math.isfinite(float(target_val)):
            continue
        clone_val = best_features.get(feat_name)
        if clone_val is None or not isinstance(clone_val, (int, float)):
            poorly_matched.append(feat_name)
            continue
        if float(target_val) == 0:
            accuracy = 100.0 if float(clone_val) == 0 else 0.0
        else:
            accuracy = max(0.0, 100.0 * (1.0 - abs(float(clone_val) - float(target_val)) / abs(float(target_val))))
        if accuracy >= 80.0:
            well_matched.append(feat_name)
        elif accuracy < 50.0:
            poorly_matched.append(feat_name)

    n_attempts = len(variant_attempts)
    error_text = f"{best_variant_error:.4f}" if best_variant_error is not None else "no valid candidate"

    lesson_lines = [
        f"--- Variant {variant_index} lesson (strategy: {generation_strategy}, "
        f"{n_attempts} attempts, best error: {error_text}) ---",
        f"Stopped because: {stop_reason}",
    ]
    if well_matched:
        lesson_lines.append(
            f"Features that were well-matched (>=80%): {', '.join(well_matched[:6])}"
        )
    if poorly_matched:
        lesson_lines.append(
            f"Features that remained poorly matched (<50%): {', '.join(poorly_matched[:6])}"
        )
    lesson_lines.append(
        "The next variant MUST try a fundamentally different code structure or algorithm. "
        "Do not repeat the same approach. Change the primary hot-path pattern, "
        "data structure sizes, or memory access strategy."
    )

    lesson_text = "\n".join(lesson_lines)
    return {
        "variant_index": variant_index,
        "generation_strategy": generation_strategy,
        "n_attempts": n_attempts,
        "best_error": best_variant_error,
        "well_matched_features": well_matched,
        "poorly_matched_features": poorly_matched,
        "stop_reason": stop_reason,
        "lesson_text": lesson_text,
    }


_CONFLICTING_FROM_SCRATCH_PAIRS: tuple[tuple[str, str], ...] = (
    ("perf_stat.L1-dcache-load-misses_MPKI", "perf_stat.LLC-load-misses_MPKI"),
    ("perf_stat.L1-dcache-load-misses_MPKI", "perf_stat.IPC"),
    ("perf_stat.branch-misses_MPKI", "perf_stat.IPC"),
)


def _baseline_has_feature_coupling(
    baseline_constraint_report: dict[str, Any],
) -> bool:
    """Return True if the top violated features include a known conflicting pair."""
    feature_rows = list(baseline_constraint_report.get("feature_rows") or [])
    violated = {
        str(row.get("feature_name", "")).strip()
        for row in feature_rows
        if not bool(row.get("within_tolerance"))
    }
    return any(
        feat_a in violated and feat_b in violated
        for feat_a, feat_b in _CONFLICTING_FROM_SCRATCH_PAIRS
    )


def _iteration_priority_text(
    *,
    feature_rows: list[dict[str, Any]],
    ipc_frozen: bool,
    objective_priority_rank: int | None = None,
) -> str:
    tier_state = _priority_tier_view(feature_rows, locked_priority_rank=objective_priority_rank)
    active_priority_rank = tier_state.get("active_priority_rank")
    protected_feature_names = _ordered_feature_names_from_rows(
        list(tier_state.get("guardrail_rows") or [])
        + list(tier_state.get("active_protected_rows") or []),
        max_features=4,
    )
    deferred_feature_names = _ordered_feature_names_from_rows(
        list(tier_state.get("deferred_rows") or []),
        max_features=4,
    )
    lines = ["1. buildability"]
    next_index = 2
    if protected_feature_names:
        labels = ", ".join(feature_display_name(name) for name in protected_feature_names)
        lines.append(f"{next_index}. strictly preserve higher-priority guardrails and already-matched counters: {labels}")
        next_index += 1
    if active_priority_rank is not None:
        lines.append(
            f"{next_index}. primary objective remains {priority_tier_label(active_priority_rank)}; do not retarget lower tiers"
        )
        next_index += 1
    elif ipc_frozen:
        lines.append(f"{next_index}. protect IPC and already-good prioritized counters")
        next_index += 1
    else:
        lines.append(f"{next_index}. biggest remaining prioritized miss")
        next_index += 1
    if deferred_feature_names:
        labels = ", ".join(feature_display_name(name) for name in deferred_feature_names)
        lines.append(f"{next_index}. do not target lower tiers yet: {labels}")
    elif active_priority_rank is not None:
        lines.append(f"{next_index}. next miss only after {priority_tier_label(active_priority_rank)} is within tolerance")
    return "Priority:\n" + "\n".join(lines)


def _format_metric_value(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.6g}"


def _relative_accuracy_pct(target_value: Any, candidate_value: Any) -> float | None:
    try:
        target = float(target_value)
        candidate = float(candidate_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(target) or not math.isfinite(candidate):
        return None
    if abs(target) <= 1e-9:
        return 100.0 if abs(candidate) <= 1e-9 else 0.0
    return max(0.0, 100.0 * (1.0 - abs(candidate - target) / abs(target)))


def _summarize_live_accuracy(
    canonical_target: dict[str, float],
    features: dict[str, Any],
) -> dict[str, float]:
    axes = {
        "overall_hw": [
            "perf_stat.IPC",
            "perf_stat.L1-icache-load-misses_MPKI",
            "perf_stat.branch-misses_MPKI",
            "perf_stat.branch-load-misses_MPKI",
            "perf_stat.L1-dcache-load-misses_MPKI",
            "perf_stat.L1-dcache-loads_LPKI",
            "perf_stat.L1-dcache-stores_SPKI",
            "perf_stat.iTLB-load-misses_MPKI",
        ],
        "ipc": ["perf_stat.IPC"],
        "l1i": ["perf_stat.L1-icache-load-misses_MPKI"],
        "branch": ["perf_stat.branch-misses_MPKI", "perf_stat.branch-load-misses_MPKI"],
        "l1d": [
            "perf_stat.L1-dcache-load-misses_MPKI",
            "perf_stat.L1-dcache-loads_LPKI",
            "perf_stat.L1-dcache-stores_SPKI",
        ],
        "itlb": ["perf_stat.iTLB-load-misses_MPKI"],
    }
    summary: dict[str, float] = {}
    for label, keys in axes.items():
        values = [
            _relative_accuracy_pct(canonical_target.get(key), features.get(key))
            for key in keys
            if key in canonical_target and key in features
        ]
        valid = [value for value in values if value is not None]
        if valid:
            summary[label] = sum(valid) / len(valid)
    return summary


def _baseline_hw_accuracy_pct(
    canonical_target: dict[str, float],
    canonical_baseline: dict[str, Any],
) -> float | None:
    summary = _summarize_live_accuracy(canonical_target, canonical_baseline)
    value = summary.get("overall_hw")
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _print_live_accuracy(
    prefix: str,
    canonical_target: dict[str, float],
    features: dict[str, Any],
    weighted_error: float | None,
) -> None:
    summary = _summarize_live_accuracy(canonical_target, features)
    parts: list[str] = []
    if weighted_error is not None and math.isfinite(weighted_error):
        parts.append(f"weighted_error={weighted_error:.4f}")
    for key in ["overall_hw", "ipc", "l1i", "branch", "l1d", "itlb"]:
        value = summary.get(key)
        if value is not None:
            parts.append(f"{key}={value:.1f}%")
    if parts:
        print(f"{prefix} {' '.join(parts)}", flush=True)


def _build_working_state_payload(
    *,
    reference_microbenchmark: str,
    active_source_microbenchmark: str,
    current_file: Path,
    current_features: dict[str, Any],
    current_rank_key: Any,
    canonical_target: dict[str, float],
    feature_scales: dict[str, float],
    origin_label: str,
    objective_priority_rank: int | None = None,
    initial_generation_mode: str = "derived_from_nearest",
    origin_attempt_index: int | None = None,
    carry_forward_reason: str = "",
) -> dict[str, Any]:
    hardware_target = _hardware_counter_features(canonical_target)
    hardware_current = _hardware_counter_features(current_features)
    current_constraint_report = _hardware_counter_constraint_report(canonical_target, current_features)
    objective_priority_rank = _locked_objective_priority_rank(
        current_constraint_report,
        previous_priority_rank=objective_priority_rank,
    )
    ipc_target = hardware_target.get("perf_stat.IPC")
    ipc_current = hardware_current.get("perf_stat.IPC")
    ipc_gap = _feature_relative_gap(hardware_target, hardware_current, "perf_stat.IPC")
    ipc_frozen = _ipc_within_tolerance_for_features(hardware_target, hardware_current)
    top_gaps = _top_feature_gaps(
        hardware_target,
        hardware_current,
        feature_scales,
        limit=len(HW_COUNTER_FEATURE_NAMES),
        freeze_ipc=ipc_frozen,
        severity_weighted=True,
    )
    tier_state = _priority_tier_view(
        list(current_constraint_report.get("feature_rows") or []),
        locked_priority_rank=objective_priority_rank,
    )
    active_priority_rank = objective_priority_rank
    protected_feature_names = _ordered_feature_names_from_rows(
        list(tier_state.get("guardrail_rows") or [])
        + list(tier_state.get("active_protected_rows") or []),
        max_features=5,
    )
    deferred_feature_names = _ordered_feature_names_from_rows(
        list(tier_state.get("deferred_rows") or []),
        max_features=5,
    )
    lines = [
        f"Original reference benchmark: {reference_microbenchmark}.",
        f"Current working benchmark anchor: {active_source_microbenchmark}.",
        f"Current working file: {current_file}.",
        f"Working state origin: {origin_label}.",
    ]
    if initial_generation_mode == "from_scratch":
        lines.append(
            "Initial generation strategy: from-scratch experiment when the nearest retrieved benchmark is too far from the target."
        )
    if origin_attempt_index is not None:
        lines.append(f"Working state came from compiled attempt {origin_attempt_index}.")
    if active_source_microbenchmark == reference_microbenchmark:
        if initial_generation_mode == "from_scratch":
            lines.append(
                "This file is still anchored on the original reference benchmark, but this experiment should not directly clone or minimally edit that benchmark body. Add exactly one new from-scratch benchmark in this file using local style and placement guidance."
            )
        else:
            lines.append(
                "This file is still anchored on the original reference benchmark. Add exactly one derived benchmark."
            )
    else:
        lines.append(
            "This file already contains one carried-forward generated benchmark. Refine that carried-forward benchmark in place and do not add a second generated benchmark."
        )
    if carry_forward_reason:
        lines.append(f"Carry-forward reason: {carry_forward_reason}")
    if ipc_target is not None and ipc_current is not None and ipc_gap is not None:
        lines.append(
            f"IPC target={_format_metric_value(ipc_target)}, current={_format_metric_value(ipc_current)}, "
            f"relative_gap_pct={ipc_gap * 100.0:.2f}."
        )
    if active_priority_rank is not None:
        lines.append(f"Controller objective tier: {priority_tier_label(active_priority_rank)}.")
    if protected_feature_names:
        labels = ", ".join(feature_display_name(name) for name in protected_feature_names)
        lines.append(f"Strictly conserve protected higher-priority or already-matched counters: {labels}.")
    if deferred_feature_names and active_priority_rank is not None:
        labels = ", ".join(feature_display_name(name) for name in deferred_feature_names)
        lines.append(f"Do not target lower tiers yet: {labels}.")
    remaining_rows = list(tier_state.get("active_focus_rows") or [])[:3]
    if not remaining_rows:
        remaining_rows = [
            row
            for row in top_gaps
            if not bool(row.get("within_tolerance"))
        ][:3]
    if remaining_rows:
        lines.append("Highest remaining constrained gaps from the current working state:")
        for row in remaining_rows:
            lines.append(
                "- "
                f"{row['feature_name']}: target={_format_metric_value(row.get('target_value'))}, "
                f"current={_format_metric_value(hardware_current.get(str(row['feature_name'])))}"
                f", relative_error_pct={float(row.get('relative_error_pct', 0.0)):.2f}"
            )

    return {
        "reference_microbenchmark": reference_microbenchmark,
        "active_source_microbenchmark": active_source_microbenchmark,
        "current_file": str(current_file),
        "origin_label": origin_label,
        "objective_priority_rank": objective_priority_rank,
        "objective_tier_label": priority_tier_label(objective_priority_rank),
        "initial_generation_mode": initial_generation_mode,
        "origin_attempt_index": origin_attempt_index,
        "carry_forward_reason": carry_forward_reason,
        "current_rank_key": list(_normalized_rank_key(current_rank_key)),
        "working_state_rank_key": list(
            _working_state_rank_key(canonical_target, current_features, current_rank_key)
        ),
        "ipc_target": ipc_target,
        "ipc_current": ipc_current,
        "ipc_relative_gap": ipc_gap,
        "top_remaining_gaps": top_gaps,
        "summary_text": "\n".join(lines).strip(),
    }


def _promote_working_state(
    *,
    current_state: dict[str, Any],
    evaluation: dict[str, Any],
    compile_record: dict[str, Any],
    canonical_target: dict[str, float],
) -> tuple[dict[str, Any] | None, str]:
    if not bool(evaluation.get("selection_eligible")):
        return None, ""
    best_candidate = evaluation.get("best_candidate") or {}
    candidate_name = str(best_candidate.get("microbenchmark", "")).strip()
    candidate_features = best_candidate.get("features") or {}
    final_file = str(compile_record.get("final_file", "")).strip()
    if not candidate_name or not candidate_features or not final_file:
        missing = []
        if not candidate_name:
            missing.append("candidate_name")
        if not candidate_features:
            missing.append("candidate_features")
        if not final_file:
            missing.append("final_file")
        print(
            f"[cloud-tuning] working state promotion skipped: missing {', '.join(missing)} "
            f"(candidate={candidate_name!r}, compile_record.final_file={compile_record.get('final_file')!r})",
            flush=True,
        )
        return None, ""

    candidate_rank_key = _working_state_rank_key(
        canonical_target,
        candidate_features,
        evaluation.get("best_candidate_rank_key"),
    )
    current_features = current_state.get("current_features") or {}
    old_ipc_gap = _feature_relative_gap(canonical_target, current_features, "perf_stat.IPC")
    new_ipc_gap = _feature_relative_gap(canonical_target, candidate_features, "perf_stat.IPC")
    objective_report = evaluation.get("objective_tier_report") or {}
    objective_label = str(
        objective_report.get("objective_tier_label")
        or priority_tier_label(current_state.get("objective_priority_rank"))
    ).strip()

    if objective_label and bool(objective_report.get("passed")):
        anchor_score = objective_report.get("anchor_score") or [0.0, 0.0]
        candidate_score = objective_report.get("candidate_score") or [0.0, 0.0]
        reason = (
            f"Compiled candidate {candidate_name} safely improved {objective_label} "
            f"(tier score {float(anchor_score[0]):.3f}/{float(anchor_score[1]):.3f} -> "
            f"{float(candidate_score[0]):.3f}/{float(candidate_score[1]):.3f}) while preserving IPC guardrails."
        )
    elif old_ipc_gap is not None and new_ipc_gap is not None and new_ipc_gap + 1e-12 < old_ipc_gap:
        reason = (
            f"Compiled candidate {candidate_name} moves IPC closer to target "
            f"({old_ipc_gap * 100.0:.2f}% -> {new_ipc_gap * 100.0:.2f}% relative gap)."
        )
    else:
        reason = (
            f"Compiled candidate {candidate_name} improved the prioritized hardware-counter match score "
            "and will be used as the next working state."
        )

    promoted_state = {
        "reference_microbenchmark": current_state["reference_microbenchmark"],
        "active_source_microbenchmark": candidate_name,
        "current_file": final_file,
        "current_features": candidate_features,
        "current_rank_key": list(_normalized_rank_key(evaluation.get("best_candidate_rank_key"))),
        "working_state_rank_key": list(candidate_rank_key),
        "origin_label": "compiled carried-forward candidate",
        "objective_priority_rank": _locked_objective_priority_rank(
            _hardware_counter_constraint_report(canonical_target, candidate_features),
            previous_priority_rank=current_state.get("objective_priority_rank"),
        ),
        "initial_generation_mode": current_state.get("initial_generation_mode", "derived_from_nearest"),
        "origin_attempt_index": int(evaluation.get("attempt_index", 0) or 0),
        "carry_forward_reason": reason,
    }
    return promoted_state, reason


def _ipc_guardrail_retry_feedback(
    task_payload: dict[str, Any],
    attempt_index: int,
    evaluation: dict[str, Any],
    canonical_target: dict[str, float] | None = None,
) -> str:
    'Build retry feedback when all candidates failed the IPC guardrail.'
    candidate_evals = list(evaluation.get("candidate_evaluations") or [])
    # Pick the candidate closest to target overall.
    best_ce: dict[str, Any] | None = None
    best_error = float("inf")
    for ce in candidate_evals:
        err = ce.get("candidate_error")
        if isinstance(err, (int, float)) and math.isfinite(float(err)) and float(err) < best_error:
            best_error = float(err)
            best_ce = ce
    if best_ce is None and candidate_evals:
        best_ce = candidate_evals[0]
    if best_ce is None:
        return ""  # fall through to generic path

    features = dict(best_ce.get("features") or {})
    candidate_ipc = features.get("perf_stat.IPC")
    micro = str(best_ce.get("microbenchmark", "")).strip()
    ipc_gr = dict(best_ce.get("ipc_guardrail_report") or {})

    # Collect target values — prefer canonical_target, fall back to top_feature_gaps.
    target_by_name: dict[str, float] = {}
    if canonical_target:
        for fname, val in canonical_target.items():
            if isinstance(val, (int, float)):
                target_by_name[fname] = float(val)
    else:
        for row in list(task_payload.get("top_feature_gaps") or []):
            fname = str(row.get("feature_name", ""))
            tv = row.get("target_value")
            if tv is not None:
                target_by_name[fname] = float(tv)
    target_ipc = target_by_name.get("perf_stat.IPC")

    lines: list[str] = [
        f"Attempt {attempt_index} compiled and ran but the IPC guardrail rejected the candidate."
    ]
    if micro:
        lines.append(f"New benchmark name: {micro}.")

    # Show measured counters so the LLM knows what happened.
    measured_lines: list[str] = []
    for fname in (
        "perf_stat.IPC",
        "perf_stat.L1-dcache-load-misses_MPKI",
        "perf_stat.LLC-load-misses_MPKI",
        "perf_stat.branch-misses_MPKI",
        "perf_stat.branch-load-misses_MPKI",
        "perf_stat.L1-dcache-loads_LPKI",
        "perf_stat.L1-dcache-stores_SPKI",
    ):
        measured = features.get(fname)
        target = target_by_name.get(fname)
        if measured is not None:
            target_text = f", target={target}" if target is not None else ""
            measured_lines.append(f"  {fname}: measured={measured}{target_text}")
    if measured_lines:
        lines.append("\nMeasured candidate counters:")
        lines.extend(measured_lines)

    # Detect IPC-cache/branch coupling.
    ipc_too_high = (
        candidate_ipc is not None
        and target_ipc is not None
        and float(candidate_ipc) > float(target_ipc) * 1.20
    )
    cache_branch_too_low = False
    for fname in ("perf_stat.L1-dcache-load-misses_MPKI", "perf_stat.branch-misses_MPKI"):
        measured = features.get(fname)
        target = target_by_name.get(fname)
        if (
            measured is not None
            and target is not None
            and float(target) > 0
            and float(measured) < float(target) * 0.30
        ):
            cache_branch_too_low = True
            break

    if ipc_too_high and cache_branch_too_low:
        lines.append(
            f"\nROOT CAUSE: Your code is too compute-efficient. "
            f"Measured IPC={candidate_ipc:.2f} but target is {target_ipc:.2f}. "
            f"The seed benchmark's baseline IPC appeared close to target because of framework "
            f"overhead, but stripped-down hot-path code exposes the true compute-bound IPC. "
            f"Cache-miss and branch-miss counters are also far below target because the code "
            f"lacks memory pressure and branch unpredictability."
        )
        lines.append(
            "\nFIX — these changes fix IPC AND cache/branch counters simultaneously:"
            "\n  1) Allocate a permuted-index array sized to exceed L1D (>64 KB) using Fisher-Yates shuffle."
            "\n  2) Use pointer-chasing (next = arr[current]) in the hot loop so each load depends on the "
            "previous load's value — hardware prefetchers cannot predict this."
            "\n  3) Gate a data-dependent branch on the load value (e.g. if (arr[pos] > threshold)) with "
            "~50% taken rate so the branch predictor mispredicts frequently."
            "\n  4) Ensure the inner loop runs 100K+ iterations to dominate instruction count and avoid "
            "MPKI dilution from benchmark framework overhead."
            "\n  5) Do NOT use LCG, hash, or modular-arithmetic random indexing — hardware prefetchers "
            "defeat those patterns."
            "\nThese memory stalls and branch mispredictions will create pipeline bubbles that "
            "naturally lower IPC from ~3.0 toward the target while also raising L1-dcache-load-misses_MPKI, "
            "LLC-load-misses_MPKI, branch-misses_MPKI, and branch-load-misses_MPKI."
        )
    elif ipc_too_high:
        lines.append(
            f"\nPROBLEM: Candidate IPC={candidate_ipc:.2f} is far above target={target_ipc}. "
            "Add stall-inducing patterns: dependent load-use chains, pointer-chasing, or "
            "data-dependent branches to bring IPC down via pipeline bubbles."
        )
    else:
        ipc_reason = str(ipc_gr.get("reason", "")).strip()
        if ipc_reason:
            lines.append(f"\nIPC guardrail violation: {ipc_reason}")
        lines.append(
            "Adjust the hot-path code to bring IPC closer to the target value."
        )

    return "\n".join(lines)


def _build_attempt_retry_feedback(
    task_payload: dict[str, Any],
    attempt_index: int,
    *,
    objective_priority_rank: int | None = None,
    active_source_microbenchmark: str | None = None,
    compile_record: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
    canonical_target: dict[str, float] | None = None,
) -> str:
    compile_record = compile_record or {}
    if compile_record and not bool(compile_record.get("success")):
        if bool(compile_record.get("policy_rejected")):
            policy_reason = _trim_feedback_text(
                str(compile_record.get("policy_rejection_reason", "")),
                max_chars=700,
            )
            detail = f" Reason: {policy_reason}" if policy_reason else ""
            return (
                f"Attempt {attempt_index} was rejected by the application patch policy before compilation."
                f"{detail} Generate a fresh function-scoped unified diff; do not repair or rewrite the complete file."
            )
        compile_attempts = compile_record.get("compile_attempts") or []
        patch_apply_failed = any(
            attempt.get("patch_apply_success") is False for attempt in compile_attempts
        )
        reached_compile = any("compile_rc" in attempt for attempt in compile_attempts)
        error_blob = _trim_feedback_text(
            "\n".join(
                part
                for part in (
                    str(compile_record.get("second_patch_apply_summary", "")),
                    str(compile_record.get("first_patch_apply_summary", "")),
                    str(compile_record.get("second_patch_apply_stderr", "")),
                    str(compile_record.get("second_patch_apply_stdout", "")),
                    str(compile_record.get("first_patch_apply_stderr", "")),
                    str(compile_record.get("first_patch_apply_stdout", "")),
                    str(compile_record.get("second_compile_summary", "")),
                    str(compile_record.get("first_compile_summary", "")),
                    str(compile_record.get("second_compile_stderr", "")),
                    str(compile_record.get("second_compile_stdout", "")),
                    str(compile_record.get("first_compile_stderr", "")),
                    str(compile_record.get("first_compile_stdout", "")),
                )
                if part.strip()
            )
        )
        if patch_apply_failed and not reached_compile:
            if error_blob:
                return (
                    f"Attempt {attempt_index} produced a patch that did not apply cleanly. "
                    f"Do not rewrite from scratch. Make one minimal corrective patch only, preserve existing helpful mechanisms, and fix buildability first. "
                    f"Return only the complete updated source file contents for the target file so the system can regenerate a clean local diff. "
                    f"Do not include prose or code fences.\n"
                    f"{error_blob}"
                )
            return (
                f"Attempt {attempt_index} produced a patch that did not apply cleanly. "
                f"Do not rewrite from scratch. Make one minimal corrective patch only and return only the complete updated source file contents for the target file."
            )
        if error_blob:
            return (
                f"Attempt {attempt_index} did not compile. Do not rewrite from scratch. Make one minimal corrective patch only, preserve existing helpful mechanisms, and fix buildability first.\n"
                f"{error_blob}"
            )
        return (
            f"Attempt {attempt_index} did not compile. Do not rewrite from scratch. Make one minimal corrective patch only and preserve existing helpful mechanisms."
        )

    evaluation = evaluation or {}
    reason = str(evaluation.get("reason", "")).strip()
    if reason == REGISTRATION_ONLY_ALIAS_REASON:
        candidate_names = [
            str(name).strip()
            for name in list(evaluation.get("new_microbenchmarks") or [])
            if str(name).strip()
        ]
        candidate_text = f" New benchmark name: {candidate_names[0]}." if candidate_names else ""
        validation_reason = str(evaluation.get("registration_only_alias_reason", "")).strip()
        detail_text = f" {validation_reason}" if validation_reason else ""
        return (
            f"Attempt {attempt_index} compiled but produced a registration-only alias.{candidate_text} "
            "You added a new benchmark name, but the timed body is unchanged. "
            "Add new hot-path code, not a new registration only."
            f"{detail_text}\n"
            "Keep exactly one generated benchmark and change the executed benchmark body itself with a real timed-path edit "
            "such as new noinline helpers, alternate hot paths, or state-driven control flow on the timed path. "
            "Do not just wrap, forward, or re-register the same work."
        )
    if reason == "No new microbenchmark appeared in --bm_list.":
        reference_microbenchmark = str(task_payload.get("source_microbenchmark", "")).strip()
        active_source = str(active_source_microbenchmark or reference_microbenchmark).strip()
        target_file = str(task_payload.get("target_file", "")).strip()
        source_root = str(task_payload.get("source_root", "")).strip()
        generator_like = False
        if target_file and source_root:
            try:
                target_file_path = Path(source_root) / target_file
                generator_like = "table or macro pipeline" in _registration_idiom_clause(target_file_path).lower()
            except Exception:
                generator_like = False
        if active_source and active_source != reference_microbenchmark:
            source_text = (
                f" by refining the carried-forward benchmark {active_source} in place"
            )
        else:
            source_text = f" derived from {reference_microbenchmark}" if reference_microbenchmark else ""
        if generator_like:
            return (
                f"Attempt {attempt_index} compiled but produced an invalid benchmark set. "
                f"Keep exactly one generated microbenchmark{source_text} as one concrete `addBenchmark(...)` registration with one unique benchmark name, "
                "do not reuse any existing `--bm_list` benchmark name, and do not add a new macro-family invocation, `testOrder.push_back(...)`, or `tests[...]` table entry that expands into many benchmarks. "
                f"{reason}"
            )
        return (
            f"Attempt {attempt_index} compiled but produced an invalid benchmark set. "
            f"Keep exactly one generated microbenchmark{source_text}, place it immediately after the source benchmark when starting from the original file, "
            f"do not reuse any existing `--bm_list` benchmark name, and do not introduce any other new benchmark registrations. {reason}"
        )

    if reason == "No new microbenchmark satisfied the IPC guardrail.":
        ipc_feedback = _ipc_guardrail_retry_feedback(
            task_payload, attempt_index, evaluation,
            canonical_target=canonical_target,
        )
        if ipc_feedback:
            return ipc_feedback

    best_candidate = evaluation.get("best_candidate") or {}
    best_report = evaluation.get("best_candidate_report") or {}
    best_constraint_report = evaluation.get("best_candidate_constraint_report") or {}
    candidate_name = str(best_candidate.get("microbenchmark", "")).strip()
    candidate_error = best_report.get("weighted_mean_normalized_abs_error")
    feature_rows = _prompt_hardware_feature_rows(
        list(best_constraint_report.get("feature_rows") or [])
    )
    ipc_row = next((row for row in feature_rows if str(row.get("feature_name")) == "perf_stat.IPC"), None)
    ipc_frozen = bool(ipc_row and bool(ipc_row.get("within_tolerance")))
    locked_priority_rank = _locked_objective_priority_rank(
        best_constraint_report,
        previous_priority_rank=objective_priority_rank,
    )
    tier_state = _priority_tier_view(
        feature_rows,
        locked_priority_rank=locked_priority_rank,
    )
    active_priority_rank = tier_state.get("active_priority_rank")
    violated_rows = [
        row
        for row in list(tier_state.get("guardrail_violated_rows") or []) + list(tier_state.get("active_focus_rows") or [])
        if not bool(row.get("within_tolerance"))
    ][:3]
    misses = []
    for row in violated_rows:
        feature_name = str(row.get("feature_name", "")).strip()
        target_value = row.get("target_value")
        candidate_value = row.get("candidate_value")
        rel_error_pct = row.get("relative_error_pct")
        if feature_name:
            misses.append(
                f"{feature_name}: target={target_value}, candidate={candidate_value}, relative_error_pct={rel_error_pct}"
            )
    miss_text = "\n".join(misses)
    accuracy_text = ""
    if isinstance(candidate_error, (int, float)) and math.isfinite(float(candidate_error)):
        accuracy_text = f" Current hardware-only weighted normalized error: {float(candidate_error):.4f}."
    benchmark_text = f" New benchmark name: {candidate_name}." if candidate_name else ""
    measurement_lines = _measurement_lines(
        task_payload=task_payload,
        best_constraint_report=best_constraint_report,
    )
    measured_text = "\n".join(measurement_lines) if measurement_lines else "- No measured counter snapshot available."
    priority_text = _iteration_priority_text(
        feature_rows=feature_rows,
        ipc_frozen=ipc_frozen,
        objective_priority_rank=locked_priority_rank,
    )
    correction_lines = _iterative_correction_lines(
        task_payload=task_payload,
        best_constraint_report=best_constraint_report,
        ipc_frozen=ipc_frozen,
        objective_priority_rank=locked_priority_rank,
    )
    correction_text = "\n".join(correction_lines)
    problem_summary = ""
    ipc_guardrail_report = evaluation.get("ipc_guardrail_report") or {}
    objective_report = evaluation.get("objective_tier_report") or {}
    if violated_rows:
        top_row = violated_rows[0]
        tier_prefix = ""
        if active_priority_rank is not None:
            tier_prefix = f"Primary objective remains {priority_tier_label(active_priority_rank)}. "
        problem_summary = (
            tier_prefix
            + f"Highest-priority remaining miss is {feature_display_name(str(top_row.get('feature_name', '')))} "
            f"(current={top_row.get('candidate_value')}, target={top_row.get('target_value')}). "
            "Preserve whatever is already working and change one primary lever at a time."
        )
        if ipc_guardrail_report and not bool(ipc_guardrail_report.get("passed")):
            ipc_reason = str(ipc_guardrail_report.get("reason", "")).strip()
            if ipc_reason:
                problem_summary += f" IPC is a protected metric and this candidate violated the guardrail: {ipc_reason}"
        if objective_report and not bool(objective_report.get("passed", True)):
            objective_reason = str(objective_report.get("reason", "")).strip()
            if objective_reason:
                problem_summary += f" {objective_reason}"
    elif reason:
        problem_summary = reason
    if miss_text:
        return (
            f"Attempt {attempt_index} compiled and ran but missed the hardware-counter target profile.{benchmark_text}{accuracy_text}\n"
            f"{priority_text}\n\n"
            f"Measured current variant:\n{measured_text}\n\n"
            f"Problem summary:\n{problem_summary}\n\n"
            f"Correction rules:\n{correction_text}"
        )
    if reason:
        return (
            f"Attempt {attempt_index} compiled and ran but still missed the hardware-counter target profile.{benchmark_text}{accuracy_text}\n"
            f"{priority_text}\n\n"
            f"Measured current variant:\n{measured_text}\n\n"
            f"Problem summary:\n{problem_summary or reason}\n\n"
            f"Correction rules:\n{correction_text}"
        )
    return (
        f"Attempt {attempt_index} compiled and ran but still missed the hardware-counter target profile.{benchmark_text}{accuracy_text}\n"
        f"{priority_text}\n\n"
        f"Measured current variant:\n{measured_text}\n\n"
        f"Problem summary:\nMake the next adjustment more targeted toward {task_payload['feature_display_name']} while preserving what is already working.\n\n"
        f"Correction rules:\n{correction_text}"
    )


def _evaluate_adjusted_binary(
    *,
    run_dir: Path,
    attempt_index: int,
    binary_path: Path,
    binary_name: str,
    source_microbenchmark: str,
    baseline_binary_microbenchmarks: set[str],
    canonical_target: dict[str, float],
    feature_scales: dict[str, float],
    anchor_features: dict[str, Any],
    anchor_constraint_report: dict[str, Any],
    objective_priority_rank: int | None,
    perf_events: str,
    amd: bool,
    deadline_monotonic: float | None = None,
    compile_record: dict[str, Any] | None = None,
    anchor_source_text: str = "",
    ipc_coupled: bool = False,
    tier_stagnation_attempts: int = 0,
) -> dict[str, Any]:
    application_seed = is_application_seed_unit(source_microbenchmark)
    if application_seed:
        listed, has_bm_list = [source_microbenchmark], False
    else:
        listed, has_bm_list = list_microbenchmarks(binary_path)
    if _looks_like_bm_list_fallback(binary_name, sorted(baseline_binary_microbenchmarks), False) and has_bm_list:
        evaluation: dict[str, Any] = {
            "attempt_index": attempt_index,
            "binary_path": str(binary_path),
            "binary_name": binary_name,
            "source_microbenchmark": source_microbenchmark,
            "has_bm_list": has_bm_list,
            "listed_microbenchmarks": listed,
            "baseline_binary_microbenchmarks": sorted(baseline_binary_microbenchmarks),
            "new_microbenchmarks": [],
            "success": False,
            "reason": (
                "Baseline --bm_list capture failed earlier and fell back to the binary name only; "
                "candidate evaluation cannot distinguish the one generated benchmark from the existing benchmark set."
            ),
            "baseline_capture_failed": True,
        }
        return evaluation
    new_microbenchmarks = (
        [source_microbenchmark]
        if application_seed
        else sorted(name for name in set(listed) if name not in baseline_binary_microbenchmarks)
    )
    evaluation: dict[str, Any] = {
        "attempt_index": attempt_index,
        "binary_path": str(binary_path),
        "binary_name": binary_name,
        "source_microbenchmark": source_microbenchmark,
        "has_bm_list": has_bm_list,
        "listed_microbenchmarks": listed,
        "baseline_binary_microbenchmarks": sorted(baseline_binary_microbenchmarks),
        "new_microbenchmarks": new_microbenchmarks,
        "candidate_evaluations": [],
        "success": False,
    }
    if not new_microbenchmarks:
        evaluation["reason"] = "No new microbenchmark appeared in --bm_list."
        return evaluation
    alias_validation_reason = None if application_seed else _registration_only_alias_validation_reason(
        compile_record=compile_record,
        anchor_source_text=anchor_source_text,
    )
    if alias_validation_reason:
        evaluation["reason"] = REGISTRATION_ONLY_ALIAS_REASON
        evaluation["selection_eligible"] = False
        evaluation["registration_only_alias_reason"] = alias_validation_reason
        evaluation["candidate_evaluations"] = [
            {
                "microbenchmark": microbenchmark_name,
                "selection_eligible": False,
                "reason": REGISTRATION_ONLY_ALIAS_REASON,
                "registration_only_alias_reason": alias_validation_reason,
            }
            for microbenchmark_name in new_microbenchmarks
        ]
        return evaluation
    best_candidate: dict[str, Any] | None = None
    best_report: dict[str, Any] | None = None
    best_constraint_report: dict[str, Any] | None = None
    best_rank_key: tuple[Any, ...] | None = None
    candidate_error: float | None = None
    runnable_candidate_count = 0
    featured_candidate_count = 0
    ipc_safe_candidate_count = 0
    eligible_candidate_count = 0
    application_runtime_failure_reasons: list[str] = []
    best_ipc_guardrail_report: dict[str, Any] | None = None
    best_objective_report: dict[str, Any] | None = None
    _best_ipc_rejected_tier_candidate: dict[str, Any] | None = None
    _best_ipc_rejected_tier_error: float = float("inf")

    def _flush_best_on_budget_exhaustion() -> None:
        """Write the best candidate found so far into `evaluation` before returning."""
        if best_candidate is not None and best_report is not None and best_rank_key is not None and candidate_error is not None:
            evaluation.update({
                "candidate_error": candidate_error,
                "best_candidate": best_candidate,
                "best_candidate_report": best_report,
                "best_candidate_constraint_report": best_constraint_report,
                "best_candidate_rank_key": list(best_rank_key),
                "chosen_runtime_estimate": best_candidate.get("runtime_estimate"),
                "selection_eligible": True,
                "ipc_guardrail_report": best_ipc_guardrail_report,
                "objective_tier_report": best_objective_report,
                "success": _threshold_met(best_constraint_report, candidate_error),
                "threshold_met": _threshold_met(best_constraint_report, candidate_error),
            })

    for microbenchmark_name in new_microbenchmarks:
        remaining_before_candidate = _remaining_wall_seconds(deadline_monotonic)
        if remaining_before_candidate is not None and remaining_before_candidate <= 0.0:
            evaluation["reason"] = _wall_budget_stop_reason(
                "candidate feature evaluation",
                remaining_seconds=remaining_before_candidate,
            )
            evaluation["budget_exhausted"] = True
            _flush_best_on_budget_exhaustion()
            if not evaluation.get("selection_eligible"):
                evaluation["selection_eligible"] = False
            return evaluation
        candidate_dir = run_dir / "feature_eval" / f"attempt_{attempt_index}" / _slugify(microbenchmark_name)
        cmd = choose_unit_cmd(binary_path, microbenchmark_name, has_bm_list)
        runtime_estimate = (
            {
                "command": cmd,
                "elapsed_seconds": float(
                    os.environ.get("APPLICATION_SEED_WINDOW_S", "3")
                ),
                "timed_out": False,
                "returncode": 0,
                "measurement_mode": "pid_attached",
            }
            if application_seed
            else measure_unit_runtime(
                binary_path,
                microbenchmark_name,
                has_bm_list=has_bm_list,
            )
        )
        candidate_eval: dict[str, Any] = {
            "microbenchmark": microbenchmark_name,
            "command": cmd,
            "feature_output_dir": str(candidate_dir),
            "runtime_estimate": runtime_estimate,
            "runnable": _runtime_estimate_runnable(runtime_estimate),
        }
        if not candidate_eval["runnable"]:
            candidate_eval["reason"] = "Benchmark did not complete a clean preflight run."
            evaluation["candidate_evaluations"].append(candidate_eval)
            continue

        remaining_before_collection = _remaining_wall_seconds(deadline_monotonic)
        if remaining_before_collection is not None and remaining_before_collection <= 0.0:
            evaluation["candidate_evaluations"].append(candidate_eval)
            evaluation["reason"] = _wall_budget_stop_reason(
                "candidate feature collection",
                remaining_seconds=remaining_before_collection,
            )
            evaluation["budget_exhausted"] = True
            _flush_best_on_budget_exhaustion()
            if not evaluation.get("selection_eligible"):
                evaluation["selection_eligible"] = False
            return evaluation

        runnable_candidate_count += 1
        try:
            if application_seed:
                row = collect_application_seed_features(
                    binary_path=binary_path,
                    unit_name=microbenchmark_name,
                    unit_output_dir=candidate_dir,
                    perf_events=perf_events,
                )
            else:
                row = collect_unit_features(
                    binary_path=binary_path,
                    unit_name=microbenchmark_name,
                    cmd=cmd,
                    unit_output_dir=candidate_dir,
                    perfspect_bin="perf",
                    perf_events=perf_events,
                    skip_intel_pt=(amd or os.environ.get("SEG08_SKIP_PT") == "1"),
                )
        except Exception as exc:
            candidate_eval["reason"] = f"Feature extraction failed: {exc}"
            if application_seed:
                application_runtime_failure_reasons.append(str(exc))
            evaluation["candidate_evaluations"].append(candidate_eval)
            continue

        featured_candidate_count += 1
        candidate_features = canonicalize_feature_dict(row)
        current_report = _hardware_counter_match_report(canonical_target, candidate_features, feature_scales)
        current_constraint_report = _hardware_counter_constraint_report(canonical_target, candidate_features)
        current_error = float(current_report["weighted_mean_normalized_abs_error"])
        current_rank_key = candidate_rank_key(
            current_constraint_report,
            current_error,
        )
        ipc_guardrail_report = _ipc_guardrail_report(
            canonical_target=canonical_target,
            anchor_features=anchor_features,
            candidate_features=candidate_features,
            ipc_coupled=ipc_coupled,
        )
        _use_adaptive = (
            tier_stagnation_attempts >= ADAPTIVE_TIER_STAGNATION_WINDOW
            and objective_priority_rank is not None
            and int(objective_priority_rank) > 1
        )
        if _use_adaptive:
            relaxed_candidate_cr = _relaxed_constraint_report(
                canonical_target, candidate_features,
                active_priority_rank=int(objective_priority_rank),
                tolerance_boost=ADAPTIVE_TIER_TOLERANCE_BOOST,
            )
            relaxed_anchor_cr = _relaxed_constraint_report(
                canonical_target, anchor_features,
                active_priority_rank=int(objective_priority_rank),
                tolerance_boost=ADAPTIVE_TIER_TOLERANCE_BOOST,
            )
        else:
            relaxed_candidate_cr = current_constraint_report
            relaxed_anchor_cr = anchor_constraint_report
        objective_report = _objective_tier_improvement_report(
            anchor_constraint_report=relaxed_anchor_cr,
            candidate_constraint_report=relaxed_candidate_cr,
            objective_priority_rank=objective_priority_rank,
        )
        if _use_adaptive and bool(objective_report.get("passed")):
            # Log that adaptive tolerance made the difference.
            strict_report = _objective_tier_improvement_report(
                anchor_constraint_report=anchor_constraint_report,
                candidate_constraint_report=current_constraint_report,
                objective_priority_rank=objective_priority_rank,
            )
            if not bool(strict_report.get("passed")):
                print(
                    f"[cloud-tuning] adaptive tolerance accepted tier improvement for "
                    f"{priority_tier_label(objective_priority_rank)} "
                    f"(stagnation={tier_stagnation_attempts} attempts, "
                    f"boost={ADAPTIVE_TIER_TOLERANCE_BOOST:.1f}x)",
                    flush=True,
                )
        ipc_guardrail_passed = bool(ipc_guardrail_report.get("passed"))
        objective_tier_improved = bool(objective_report.get("passed"))
        if ipc_coupled:
            selection_eligible = ipc_guardrail_passed
        else:
            selection_eligible = ipc_guardrail_passed and objective_tier_improved
        candidate_eval.update(
            {
                "features": candidate_features,
                "candidate_error": current_error,
                "threshold_met": _threshold_met(current_constraint_report, current_error),
                "hardware_counter_match_report": current_report,
                "hardware_counter_constraint_report": current_constraint_report,
                "rank_key": list(current_rank_key),
                "ipc_guardrail_report": ipc_guardrail_report,
                "objective_tier_report": objective_report,
                "selection_eligible": selection_eligible,
            }
        )
        if ipc_guardrail_passed:
            ipc_safe_candidate_count += 1
        if not selection_eligible:
            rejection_reasons = []
            if not ipc_guardrail_passed:
                rejection_reasons.append(str(ipc_guardrail_report.get("reason", "")).strip())
            if not objective_tier_improved:
                rejection_reasons.append(str(objective_report.get("reason", "")).strip())
            candidate_eval["reason"] = " ".join(part for part in rejection_reasons if part).strip()
            evaluation["candidate_evaluations"].append(candidate_eval)
            if not ipc_guardrail_passed and objective_tier_improved and current_error < _best_ipc_rejected_tier_error:
                _best_ipc_rejected_tier_error = current_error
                _best_ipc_rejected_tier_candidate = {
                    "microbenchmark": microbenchmark_name,
                    "features": candidate_features,
                    "candidate_error": current_error,
                    "ipc_guardrail_report": ipc_guardrail_report,
                    "objective_tier_report": objective_report,
                }
            continue

        eligible_candidate_count += 1
        evaluation["candidate_evaluations"].append(candidate_eval)
        if best_rank_key is None or current_rank_key < best_rank_key:
            best_rank_key = current_rank_key
            candidate_error = current_error
            best_report = current_report
            best_constraint_report = current_constraint_report
            best_ipc_guardrail_report = ipc_guardrail_report
            best_objective_report = objective_report
            best_candidate = {
                "microbenchmark": microbenchmark_name,
                "command": cmd,
                "features": candidate_features,
                "runtime_estimate": runtime_estimate,
            }

    evaluation["runnable_candidate_count"] = runnable_candidate_count
    evaluation["featured_candidate_count"] = featured_candidate_count
    evaluation["ipc_safe_candidate_count"] = ipc_safe_candidate_count
    evaluation["eligible_candidate_count"] = eligible_candidate_count
    if _best_ipc_rejected_tier_candidate is not None:
        evaluation["ipc_rejected_tier_hint"] = _best_ipc_rejected_tier_candidate
    if best_candidate is None or best_report is None or best_constraint_report is None or best_rank_key is None or candidate_error is None:
        if runnable_candidate_count == 0:
            evaluation["reason"] = "No new microbenchmark completed a clean preflight run."
        elif application_seed and application_runtime_failure_reasons and featured_candidate_count == 0:
            unique_reasons = list(dict.fromkeys(application_runtime_failure_reasons))
            evaluation["reason"] = (
                "Application-seed runtime validation failed: " + " ".join(unique_reasons)
            )
            evaluation["runtime_validation_failed"] = True
        elif ipc_safe_candidate_count == 0:
            evaluation["reason"] = "No new microbenchmark satisfied the IPC guardrail."
        elif eligible_candidate_count == 0 and objective_priority_rank is not None:
            evaluation["reason"] = (
                f"No new microbenchmark improved {priority_tier_label(objective_priority_rank)} "
                "while preserving IPC guardrails."
            )
        else:
            evaluation["reason"] = "No new microbenchmark produced a usable hardware-counter feature profile."
        evaluation["selection_eligible"] = False
        return evaluation

    success = _threshold_met(best_constraint_report, candidate_error)
    evaluation.update(
        {
            "success": success,
            "threshold_met": success,
            "candidate_error": candidate_error,
            "best_candidate": best_candidate,
            "best_candidate_report": best_report,
            "best_candidate_constraint_report": best_constraint_report,
            "best_candidate_rank_key": list(best_rank_key),
            "chosen_runtime_estimate": best_candidate.get("runtime_estimate"),
            "selection_metric": "hardware_counters_only",
            "selected_from_multiple_new_microbenchmarks": len(new_microbenchmarks) > 1,
            "ipc_guardrail_report": best_ipc_guardrail_report,
            "objective_tier_report": best_objective_report,
            "selection_eligible": True,
        }
    )
    if not success:
        selection_prefix = ""
        if len(new_microbenchmarks) > 1:
            selection_prefix = (
                f"Evaluated {featured_candidate_count}/{len(new_microbenchmarks)} new microbenchmarks and selected "
                f"{best_candidate['microbenchmark']}. "
            )
        evaluation["reason"] = (
            selection_prefix
            + "Best IPC-safe active-tier-improving candidate missed the hardware-counter target profile threshold "
            f"(hardware-only weighted normalized error {candidate_error:.4f})."
        )
    elif len(new_microbenchmarks) > 1:
        evaluation["reason"] = (
            f"Evaluated {featured_candidate_count}/{len(new_microbenchmarks)} new microbenchmarks and selected "
            f"{best_candidate['microbenchmark']} as the best IPC-safe hardware-counter match."
        )
    return evaluation


def _feature_scales(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for feature_name in CANONICAL_FEATURE_COLUMNS:
        if feature_name not in df.columns:
            out[feature_name] = 1.0
            continue
        numeric = pd.to_numeric(df[feature_name], errors="coerce").dropna()
        arr = numeric.to_numpy(dtype=float)
        if arr.size == 0:
            out[feature_name] = 1.0
        elif arr.size == 1:
            out[feature_name] = max(abs(float(arr[0])) * 0.05, 1e-6)
        else:
            q10, q25, q50, q75, q90 = pd.Series(arr).quantile([0.10, 0.25, 0.50, 0.75, 0.90]).tolist()
            out[feature_name] = max(q90 - q10, q75 - q25, float(pd.Series(arr).std()), abs(q50) * 0.05, 1e-6)
    return out


@lru_cache(maxsize=1)
def _load_property_task_rows() -> tuple[dict[str, Any], ...]:
    if not PROPERTY_TASKS_MANIFEST.exists():
        return ()

    rows: list[dict[str, Any]] = []
    with PROPERTY_TASKS_MANIFEST.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return tuple(rows)


def _lookup_property_task(
    binary_name: str,
    microbenchmark_name: str | None = None,
    feature_name: str | None = None,
    direction: str | None = None,
    magnitude: str | None = None,
) -> dict[str, Any] | None:
    for row in _load_property_task_rows():
        if str(row.get("binary")) != str(binary_name):
            continue
        if microbenchmark_name is not None and str(row.get("source_microbenchmark")) != str(microbenchmark_name):
            continue
        if feature_name is not None and str(row.get("feature_name")) != str(feature_name):
            continue
        if direction is not None and str(row.get("direction")) != str(direction):
            continue
        if magnitude is not None and str(row.get("magnitude")) != str(magnitude):
            continue
        return dict(row)
    return None


def _resolve_binary_target(binary_name: str, microbenchmark_name: str) -> dict[str, Any]:
    source_map = parse_benchmark_binary_sources()
    info = source_map.get(binary_name)
    if info is None:
        manifest_row = _lookup_property_task(binary_name, microbenchmark_name) or _lookup_property_task(binary_name)
        if manifest_row is None:
            raise KeyError(binary_name)
        source_root = Path(str(manifest_row["source_root"]))
        rel_path = Path(str(manifest_row["target_file"]))
        build_dir = _normalize_build_dir(Path(str(manifest_row["build_dir"])), source_root, binary_name)
        return {
            "source_root": source_root,
            "build_dir": build_dir,
            "target_file": str(rel_path),
            "current_file": source_root / rel_path,
        }

    source_root = Path(info["source_root"])
    rel_paths = list(info["source_rel_paths"])
    if len(rel_paths) == 1:
        rel_path = rel_paths[0]
    else:
        best_path = rel_paths[0]
        best_score = -1
        for rel_path in rel_paths:
            text = (source_root / rel_path).read_text(encoding="utf-8", errors="ignore")
            unit_token = microbenchmark_name.rsplit("::", 1)[-1].replace("_8bpc", "").replace("_16bpc", "")
            score = text.count(microbenchmark_name)
            score += text.count(unit_token)
            score += text.count("BENCHMARK(")
            score += text.count("BENCHMARK_RELATIVE(")
            if score > best_score:
                best_score = score
                best_path = rel_path
        rel_path = best_path
    build_dir = _normalize_build_dir(Path(info["build_dir"]), source_root, binary_name)
    return {
        "source_root": source_root,
        "build_dir": build_dir,
        "target_file": str(rel_path),
        "current_file": source_root / rel_path,
    }


def _top_feature_gaps(
    target_features: dict[str, float],
    baseline_features: dict[str, float],
    feature_scales: dict[str, float],
    limit: int = 12,
    *,
    freeze_ipc: bool = False,
    severity_weighted: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_name in CANONICAL_FEATURE_COLUMNS:
        if feature_name not in target_features or feature_name not in baseline_features:
            continue
        scale = max(float(feature_scales.get(feature_name, 1.0)), 1e-9)
        constraint = feature_constraint(feature_name)
        delta = float(target_features[feature_name]) - float(baseline_features[feature_name])
        rel_err = abs(delta) / max(abs(float(target_features[feature_name])), 1e-9)
        tolerance_ratio = constraint.tolerance_ratio if constraint is not None else None
        within_tolerance = (
            rel_err <= tolerance_ratio + 1e-12 if tolerance_ratio is not None else False
        )
        overflow_ratio = (
            max(0.0, rel_err - tolerance_ratio) / max(tolerance_ratio, 1e-9)
            if tolerance_ratio is not None
            else 0.0
        )
        rows.append(
            {
                "feature_name": feature_name,
                "target_value": float(target_features[feature_name]),
                "baseline_value": float(baseline_features[feature_name]),
                "delta_to_target": delta,
                "normalized_abs_delta": abs(delta) / scale,
                "priority_rank": constraint.priority_rank if constraint is not None else 999,
                "priority_group": constraint.group if constraint is not None else None,
                "importance_weight": constraint.importance_weight if constraint is not None else 0.0,
                "relative_error": rel_err,
                "relative_error_pct": rel_err * 100.0,
                "tolerance_ratio": tolerance_ratio,
                "tolerance_pct": (tolerance_ratio * 100.0) if tolerance_ratio is not None else None,
                "within_tolerance": within_tolerance,
                "overflow_ratio": overflow_ratio,
            }
        )
    if severity_weighted:
        rows.sort(key=lambda row: _corrective_gap_priority_key(row, freeze_ipc=freeze_ipc))
    else:
        rows.sort(key=prioritized_gap_key)
    return rows[:limit]


def _binary_subject(binary_name: str) -> str:
    subject = binary_name.replace("_benchmark", " benchmark")
    subject = subject.replace("_bench", " bench")
    return subject.replace("_", " ").strip()


def _task_bounds(scale: float, magnitude: str) -> tuple[float, float | None, float]:
    lower_factor, upper_factor = MAGNITUDE_BANDS[magnitude]
    return (
        lower_factor * scale,
        None if upper_factor is None else upper_factor * scale,
        MAGNITUDE_TARGET_FACTORS[magnitude] * scale,
    )


def _classify_magnitude(abs_delta: float, scale: float) -> str:
    ratio = abs_delta / max(scale, 1e-9)
    if ratio < 0.30:
        return "minimally"
    if ratio < 0.60:
        return "moderately"
    if ratio < 1.00:
        return "substantially"
    return "significantly"


def _style_anchors(binary_name: str, microbenchmark_name: str) -> list[str]:
    manifest_row = _lookup_property_task(binary_name, microbenchmark_name)
    anchors = manifest_row.get("style_anchors") if manifest_row else []
    if not isinstance(anchors, list):
        return []
    return [str(anchor) for anchor in anchors if str(anchor).strip()]


def _benchmark_family_base(source_microbenchmark: str) -> str:
    family = " ".join(str(source_microbenchmark).split()).strip()
    for separator in ("(", ":"):
        if separator in family:
            family = family.split(separator, 1)[0].strip()
    return family or str(source_microbenchmark).strip()


def _prompt_code_view(text: str) -> str:
    source = str(text)
    pattern = re.compile(r"(?ms)^[ \t]*#if 0\b.*?^[ \t]*#endif[ \t]*\n?")
    rendered: list[str] = []
    last = 0
    for match in pattern.finditer(source):
        block = match.group(0)
        if (
            match.end() >= int(len(source) * 0.70)
            and len(block) >= 1200
            and re.search(r"relative\s+time/iter\s+iters/s", block)
            and re.search(r"\b[A-Za-z0-9_]+:\s+k=", block)
        ):
            rendered.append(source[last:match.start()])
            last = match.end()
    rendered.append(source[last:])
    return "".join(rendered)


def _edit_location_clause(
    target_file_path: Path,
    source_microbenchmark: str,
    *,
    from_scratch: bool = False,
) -> str:
    try:
        file_text = target_file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""

    code_view = _prompt_code_view(file_text)
    lines = code_view.splitlines()
    if not lines:
        return ""

    source_name = " ".join(str(source_microbenchmark).split()).strip()
    family_base = _benchmark_family_base(source_name)
    source_lower = source_name.lower()
    family_lower = family_base.lower()

    def _codeish(line: str) -> bool:
        stripped = line.strip()
        return bool(stripped) and not stripped.startswith("//")

    exact_hits = [line.strip() for line in lines if _codeish(line) and source_lower in line.lower()]
    anchor_hits = [
        line.strip()
        for line in lines
        if _codeish(line)
        and family_lower
        and family_lower in line.lower()
        and ("BENCHMARK" in line or "addBenchmark" in line or "add" in line)
    ]
    generator_like = any("addBenchmark(__FILE__" in line for line in lines) and bool(anchor_hits)

    if not exact_hits and anchor_hits:
        anchor = anchor_hits[0]
        if generator_like:
            if from_scratch:
                return (
                    f"This file appears to generate the {family_base} benchmark family programmatically rather than defining "
                    f"a standalone `{source_name}` benchmark block. Add exactly one concrete `addBenchmark(...)` registration adjacent to "
                    f"`{anchor}` and keep the existing generated family unchanged; use that location for placement and style only, not as a code body to clone. "
                    "Do not add a new macro-family invocation or table entry that expands into many benchmarks."
                )
            return (
                f"This file appears to generate the {family_base} benchmark family programmatically rather than defining "
                f"a standalone `{source_name}` benchmark block. Add exactly one concrete `addBenchmark(...)` registration adjacent to "
                f"`{anchor}` and keep the existing generated family unchanged. Do not add a new macro-family invocation or table entry that expands into many benchmarks."
            )
        if from_scratch:
            return (
                f"The named source benchmark `{source_name}` does not appear as a standalone implementation block. "
                f"Add exactly one new manual benchmark registration adjacent to `{anchor}` and use that area only as a placement/style reference; do not clone its body."
            )
        return (
            f"The named source benchmark `{source_name}` does not appear as a standalone implementation block. "
            f"Add exactly one new benchmark registration adjacent to `{anchor}` instead of inventing a new placement site elsewhere in the file."
        )

    if exact_hits:
        anchor = exact_hits[0]
        if from_scratch:
            return (
                f"Place the new from-scratch benchmark adjacent to the existing `{source_name}` code or registration site, for example near `{anchor}`, "
                "but do not clone or minimally edit that benchmark body. Use it only as a placement and style reference."
            )
        return (
            f"Keep the edit adjacent to the existing `{source_name}` code or registration site, for example near `{anchor}`. "
            "Do not move the change into an unrelated benchmark family."
        )

    if from_scratch:
        return (
            f"Add exactly one new from-scratch benchmark registration near the `{source_name}` family registration to stay consistent with this file's layout, "
            "and avoid modifying unrelated benchmark families."
        )
    return (
        f"Add exactly one new benchmark registration for a variant derived from `{source_name}` close to its existing family registration, "
        "and avoid modifying unrelated benchmark families."
    )


def _registration_idiom_clause(target_file_path: Path) -> str:
    try:
        file_text = target_file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""

    code_view = _prompt_code_view(file_text)
    lines = code_view.splitlines()
    if not lines:
        return ""

    storage_type = ""
    if "std::function<" in code_view:
        storage_type = "std::function"
    elif "folly::Function<" in code_view:
        storage_type = "folly::Function"

    generator_like = (
        any("addBenchmark(__FILE__" in line for line in lines)
        and any("#define " in line for line in lines)
        and any("testOrder" in line or "tests[" in line for line in lines)
    )
    if generator_like:
        parts = [
            "This file registers benchmarks indirectly through a table or macro pipeline and a final addBenchmark loop."
        ]
        if storage_type:
            parts.append(
                f"Preserve the existing callback storage type `{storage_type}` and the final `addBenchmark(...)` call shape."
            )
        parts.append(
            "Do not add a new macro-family invocation, `testOrder.push_back(...)`, or `tests[...]` table assignment that expands into many benchmarks."
        )
        parts.append(
            "When the evaluator requires exactly one new benchmark, add one concrete `addBenchmark(...)` registration with one unique benchmark name, keep the existing generated family unchanged, and do not reuse an existing `--bm_list` benchmark name."
        )
        if storage_type == "std::function":
            parts.append(
                "Do not replace `std::function` with `folly::Function` or add an extra wrapper lambda unless the file already uses that pattern."
            )
        if storage_type == "folly::Function":
            parts.append(
                "Do not replace `folly::Function` with another callable wrapper unless the file already mixes those patterns."
            )
        return " ".join(parts)

    if any("addBenchmark(" in line for line in lines) and storage_type:
        return (
            f"Preserve the existing local `addBenchmark(...)` registration idiom and `{storage_type}` callback storage while adding exactly one new benchmark."
        )

    return ""


def _prompt_hardware_feature_rows(feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in feature_rows
        if str(row.get("feature_name", "")).strip().startswith("perf_stat.")
    ]


def _measurement_lines(
    *,
    task_payload: dict[str, Any],
    best_constraint_report: dict[str, Any],
) -> list[str]:
    feature_rows = _prompt_hardware_feature_rows(
        list(best_constraint_report.get("feature_rows") or [])
    )
    by_name = {str(row.get("feature_name", "")).strip(): row for row in feature_rows}
    # Show IPC first, then primary feature, then all violated, then matched.
    ordered_names: list[str] = []
    for candidate in ["perf_stat.IPC", str(task_payload.get("feature_name", "")).strip()]:
        if candidate and candidate in by_name and candidate not in ordered_names:
            ordered_names.append(candidate)
    # All violated features sorted by overflow (worst first)
    violated = [
        row for row in feature_rows
        if str(row.get("feature_name", "")).strip() not in ordered_names
        and not bool(row.get("within_tolerance"))
    ]
    violated.sort(key=lambda r: float(r.get("overflow_ratio", 0)), reverse=True)
    for row in violated:
        ordered_names.append(str(row.get("feature_name", "")).strip())
    # Then matched features for context
    for row in feature_rows:
        fname = str(row.get("feature_name", "")).strip()
        if fname and fname not in ordered_names:
            ordered_names.append(fname)

    lines: list[str] = []
    for feature_name in ordered_names:
        row = by_name.get(feature_name) or {}
        target_value = row.get("target_value")
        candidate_value = row.get("candidate_value")
        tolerance_ratio = row.get("tolerance_ratio")
        within = bool(row.get("within_tolerance"))
        label = feature_display_name(feature_name)
        status_tag = "OK" if within else "MISS"
        # Direction hint: tell the LLM which way to move
        direction = ""
        try:
            tv, cv = float(target_value), float(candidate_value)
            if not within:
                if cv > tv:
                    direction = " -> REDUCE"
                elif cv < tv:
                    direction = " -> INCREASE"
        except (TypeError, ValueError):
            pass
        if tolerance_ratio is not None:
            try:
                acceptable_min, acceptable_max = tolerance_bounds(float(target_value), float(tolerance_ratio))
                lines.append(
                    f"- [{status_tag}] {label}: current={candidate_value}, target={target_value}, "
                    f"acceptable={acceptable_min:.6g}..{acceptable_max:.6g}{direction}"
                )
            except Exception:
                lines.append(f"- [{status_tag}] {label}: current={candidate_value}, target={target_value}{direction}")
        else:
            lines.append(f"- [{status_tag}] {label}: current={candidate_value}, target={target_value}{direction}")
    return lines


_COUPLING_CACHE_BRANCH_FEATURES = {
    "perf_stat.L1-dcache-load-misses_MPKI",
    "perf_stat.LLC-load-misses_MPKI",
    "perf_stat.branch-misses_MPKI",
    "perf_stat.branch-load-misses_MPKI",
}


def _iterative_coupling_detected(by_name: dict[str, dict]) -> bool:
    'Return True when measured candidate features show IPC-cache/branch coupling.'
    ipc_row = by_name.get("perf_stat.IPC")
    if not ipc_row:
        return False
    try:
        ipc_val = float(ipc_row.get("candidate_value", 0))
        ipc_target = float(ipc_row.get("target_value", 0))
    except (TypeError, ValueError):
        return False
    if ipc_target <= 0 or ipc_val < ipc_target * 1.15:
        return False  # IPC not significantly above target
    far_below = 0
    for feat in _COUPLING_CACHE_BRANCH_FEATURES:
        row = by_name.get(feat)
        if not row:
            continue
        try:
            val = float(row.get("candidate_value", 0))
            tgt = float(row.get("target_value", 0))
        except (TypeError, ValueError):
            continue
        if tgt > 0 and val < tgt * 0.3:
            far_below += 1
    return far_below >= 2


def _coupling_sizing_feedback(by_name: dict[str, dict]) -> list[str]:
    """Return concrete sizing guidance given measured candidate counters."""
    lines: list[str] = []
    lines.append(
        "- IPC-CACHE/BRANCH COUPLING DETECTED in measured candidate. "
        "IPC is too high because the workload lacks memory stalls and branch mispredictions. "
        "Do NOT try to lower IPC directly — fix the cache and branch counters first."
    )
    # L1d-MPKI feedback
    l1d = by_name.get("perf_stat.L1-dcache-load-misses_MPKI")
    if l1d:
        try:
            val = float(l1d.get("candidate_value", 0))
            tgt = float(l1d.get("target_value", 0))
        except (TypeError, ValueError):
            val, tgt = 0.0, 0.0
        if tgt > 0 and val < tgt * 0.3:
            ratio = tgt / max(val, 0.001)
            if ratio > 50:
                size_hint = "1 MB (256K uint32_t entries)"
            elif ratio > 10:
                size_hint = "512 KB (128K uint32_t entries)"
            else:
                size_hint = "256 KB (64K uint32_t entries)"
            lines.append(
                f"- L1d-MPKI is {val:.3f} vs target {tgt:.2f} ({ratio:.0f}× too low). "
                f"Your chase array is too small. Increase it to at least {size_hint}. "
                f"Ensure every loop iteration does a DEPENDENT load: pos = arr[pos]. "
                f"The array MUST exceed L1D cache (48 KB) by a wide margin."
            )
    # LLC-MPKI feedback
    llc = by_name.get("perf_stat.LLC-load-misses_MPKI")
    if llc:
        try:
            val = float(llc.get("candidate_value", 0))
            tgt = float(llc.get("target_value", 0))
        except (TypeError, ValueError):
            val, tgt = 0.0, 0.0
        if tgt > 0 and val < tgt * 0.3:
            lines.append(
                f"- LLC-MPKI is {val:.4f} vs target {tgt:.3f}. "
                f"Add a SECOND large cold pointer-chase chain (128 MB, exceeding LLC) with "
                f"Fisher-Yates shuffled nodes. Access via PARALLEL independent loads (not serial) "
                f"every N iterations to control LLC MPKI. Use branchless scheduling: "
                f"coldMask = -uint64_t(j % N == 0). "
                f"CRITICAL: Allocate the cold chain with 1GB hugepages (mmap with MAP_HUGETLB | MAP_HUGE_1GB) "
                f"to avoid consuming STLB entries that would destroy iTLB accuracy."
            )
    # Branch MPKI feedback
    for feat_name, label in [
        ("perf_stat.branch-misses_MPKI", "branch-misses MPKI"),
        ("perf_stat.branch-load-misses_MPKI", "branch-load-misses MPKI"),
    ]:
        row = by_name.get(feat_name)
        if not row:
            continue
        try:
            val = float(row.get("candidate_value", 0))
            tgt = float(row.get("target_value", 0))
        except (TypeError, ValueError):
            continue
        if tgt > 0 and val < tgt * 0.3:
            lines.append(
                f"- {label} is {val:.3f} vs target {tgt:.2f}. "
                f"The branch condition MUST depend on the randomly-loaded value "
                f"(e.g., if (pos & 1)) so the branch predictor cannot learn the pattern."
            )
    # Iteration count
    lines.append(
        "- Iteration count MUST be >= 500K (preferably 1M) so the chase loop "
        "dominates the instruction mix over fiber/framework overhead."
    )
    return lines


def _iterative_correction_lines(
    *,
    task_payload: dict[str, Any],
    best_constraint_report: dict[str, Any],
    ipc_frozen: bool,
    objective_priority_rank: int | None = None,
) -> list[str]:
    feature_rows = _prompt_hardware_feature_rows(
        list(best_constraint_report.get("feature_rows") or [])
    )
    if not feature_rows:
        top_gaps = list(task_payload.get("top_feature_gaps") or [])
        coupled = _ipc_cache_branch_coupled(top_gaps, ipc_frozen)
        lines = [
            "- Change one primary lever only.",
            "- Protect counters already within tolerance.",
        ]
        if coupled:
            lines.append(
                "- IPC-CACHE/BRANCH COUPLING: The seed benchmark's code is compute-bound "
                "with much higher effective IPC than the target. Add memory stalls and branch "
                "unpredictability to simultaneously lower IPC and raise cache/branch miss counters. "
                "You MUST use a permuted-index chase array > 64 KB with dependent loads "
                "(pos = chase[pos]) and data-dependent branches (if (pos & 1)) in a 100K+ iteration "
                "inner loop. Simple conditionals or linear array walks are defeated by prefetchers."
            )
        elif ipc_frozen:
            lines.append("- IPC is already within tolerance; keep it stable unless buildability requires a change.")
        else:
            lines.append("- Keep IPC ahead of lower-tier counters when goals conflict.")
        return lines

    by_name = {str(row.get("feature_name", "")).strip(): row for row in feature_rows}
    locked_priority_rank = _locked_objective_priority_rank(
        best_constraint_report,
        previous_priority_rank=objective_priority_rank,
    )
    tier_state = _priority_tier_view(
        feature_rows,
        locked_priority_rank=locked_priority_rank,
    )
    active_priority_rank = tier_state.get("active_priority_rank")
    guardrail_feature_names = _ordered_feature_names_from_rows(
        list(tier_state.get("guardrail_rows") or []),
        max_features=5,
    )
    protected_feature_names = _ordered_feature_names_from_rows(
        list(tier_state.get("active_protected_rows") or []),
        max_features=5,
    )
    deferred_feature_names = _ordered_feature_names_from_rows(
        list(tier_state.get("deferred_rows") or []),
        max_features=5,
    )
    prioritized_rows = _corrective_feature_rows(feature_rows, freeze_ipc=ipc_frozen)
    top_row = next((row for row in prioritized_rows if not bool(row.get("within_tolerance"))), None)
    if top_row is None:
        primary_name = str(task_payload.get("feature_name", "")).strip()
        top_row = by_name.get(primary_name) or by_name.get("perf_stat.IPC")

    top_feature_name = str((top_row or {}).get("feature_name", "")).strip()
    branch_rows = [
        row
        for row in [
            by_name.get("perf_stat.branch-misses_MPKI"),
            by_name.get("perf_stat.branch-load-misses_MPKI"),
        ]
        if row
    ]
    branch_overshoot = any(_metric_status(row) == "too high" for row in branch_rows)
    status = _metric_status(top_row or {})
    primary_fix, _ = _recommended_transformations(
        top_feature_name,
        status=status,
        by_name=by_name,
        branch_overshoot=branch_overshoot,
    )

    # Detect coupling from the measured candidate features.
    coupled_from_measured = _iterative_coupling_detected(by_name)

    lines = [
        "- Change one primary lever only.",
    ]
    if active_priority_rank is not None:
        lines.append(
            f"- Primary objective remains {priority_tier_label(active_priority_rank)} until it is within tolerance; do not retarget the run around another tier unless the controller changes modes."
        )
    if guardrail_feature_names:
        labels = ", ".join(feature_display_name(name) for name in guardrail_feature_names)
        lines.append(f"- Higher-priority guardrails that must not regress: {labels}. If a patch breaks one, revert to the previous anchor and make a smaller change.")
    if protected_feature_names:
        labels = ", ".join(feature_display_name(name) for name in protected_feature_names)
        lines.append(f"- Strictly conserve already-matched counters inside the active tier: {labels}.")
    else:
        lines.append("- Protect counters already within tolerance.")
    if coupled_from_measured:
        # Override normal IPC guidance.
        lines.extend(_coupling_sizing_feedback(by_name))
    elif ipc_frozen:
        lines.append("- IPC is already within tolerance; keep it stable unless buildability requires a change.")
    else:
        lines.append("- IPC is a protected metric. If a candidate violates IPC, revert to the previous anchor and apply a smaller patch instead of retargeting the whole run around IPC recovery.")
    if top_feature_name:
        lines.append(
            f"- Biggest remaining miss: {feature_display_name(top_feature_name)} is {status} "
            f"(current={(top_row or {}).get('candidate_value')}, target={(top_row or {}).get('target_value')})."
        )
    if primary_fix and not coupled_from_measured:
        lines.append(f"- Recommended correction: {primary_fix}")
    if deferred_feature_names:
        labels = ", ".join(feature_display_name(name) for name in deferred_feature_names)
        lines.append(f"- Leave lower-tier counters for later attempts: {labels}.")
    return lines


def _metric_status(row: dict[str, Any]) -> str:
    if bool(row.get("within_tolerance")):
        return "within range"
    try:
        candidate_value = float(row.get("candidate_value"))
        target_value = float(row.get("target_value"))
    except Exception:
        return "out of range"
    if candidate_value > target_value:
        return "too high"
    if candidate_value < target_value:
        return "too low"
    return "within range"


def _recommended_transformations(
    feature_name: str,
    *,
    status: str,
    by_name: dict[str, dict[str, Any]] | None = None,
    branch_overshoot: bool = False,
) -> tuple[str, str | None]:
    if feature_name == "perf_stat.IPC":
        if status == "too high":
            by_name = by_name or {}
            frontend_low = any(
                _metric_status(by_name.get(name) or {}) == "too low"
                for name in (
                    "perf_stat.L1-icache-load-misses_MPKI",
                    "perf_stat.branch-misses_MPKI",
                    "perf_stat.branch-load-misses_MPKI",
                )
                if by_name.get(name)
            )
            if frontend_low:
                return (
                    "Add one compact frontend or control-flow perturbation in the existing hot path, such as a small unpredictable branch site or modest hot-footprint expansion, instead of jumping first to a backend stall mechanism.",
                    "Only after frontend and branch counters are materially closer should you add a small backend-latency lever if IPC still remains too high.",
                )
            primary = (
                "PROVEN APPROACH: IPC is primarily controlled by iteration count (determines tuned-code fraction of total instructions) "
                "and the balance of ALU work vs memory stalls. Increase iteration count first — empirical: 50M iters -> IPC 1.96, "
                "100M -> 1.70, 130M -> 1.62. If IPC is still too high, add a dependent pointer-chase load-use chain on the hot path."
            )
            secondary = (
                "If branch counters are already above target, slightly soften branch intensity while keeping the existing "
                "frontend/cache shape stable."
                if branch_overshoot
                else "Keep the current branch/frontend mechanism stable; only add the smallest backend-latency lever needed."
            )
            return primary, secondary
        if status == "too low":
            return (
                "Preserve the current benchmark's throughput-carrying structure and remove only the smallest source of avoidable stall time or dependency depth.",
                "Do not change loop trip counts, benchmark scale, or the main steady-state work volume unless required; if a fix risks lowering IPC elsewhere, reduce its frequency or move it off the hottest path.",
            )
    if feature_name == "perf_stat.branch-misses_MPKI":
        if status == "too low":
            return (
                "PROVEN PATTERN: Use 3 noinline 256-case switch functions rotated with j%3. The switch index (payload & 0xFF) from a cache-missing pointer-chase load makes the indirect branch unpredictable to TAGE. This generates branch misses AND L1i pressure simultaneously. "
                "COMPACT ALTERNATIVE: For branch misses without expanding code footprint, use Collatz iteration: `if (x & 1) { x = 3*x + 1; } else { x >>= 1; }`. This generates ~35% misprediction in ~2 instructions per branch. Insert N_COLLATZ iterations per loop to control branch-miss MPKI (75 proven effective). Seed with pointer-chase load value.",
                "Do not add simple if/else branches on register-computed state; the CPU resolves those speculatively without mispredicting. Only branches gated on high-latency load values reliably cause mispredictions.",
            )
        return (
            "Reduce the number of unpredictable switch targets or make the switch index more predictable to lower branch-miss rate.",
            "Do not add a large dispatcher tree or many helper variants.",
        )
    if feature_name == "perf_stat.branch-load-misses_MPKI":
        if status == "too low":
            return (
                "Add one or two data-dependent indirect branches or conditional branches whose "
                "outcome depends on randomly-loaded memory values. The branch predictor must not "
                "be able to learn the pattern.",
                "Keep the control-flow expansion local and avoid a large fan-out.",
            )
        return (
            "Add one or two additional hot branch targets or sites so the branch-target footprint grows modestly.",
            "Keep the control-flow expansion local and avoid a large fan-out.",
        )
    if feature_name == "perf_stat.LLC-load-misses_MPKI":
        if status == "too low":
            return (
                "PROVEN PATTERN: Add a separate cold pointer-chase chain of 128 MB (2M nodes x 64B, Fisher-Yates shuffled, exceeds LLC). "
                "Use PARALLEL independent cold loads (not serial chain) to exploit memory-level parallelism — serial loads stall for N×200cy, "
                "during which the L1i prefetcher runs freely and icache accuracy tanks. Parallel loads stall for ~200cy total regardless of count. "
                "Access via branchless scheduling every N iterations: `coldMask = -uint64_t(j % N == 0)`. "
                "CRITICAL: Allocate the cold chain on 1GB hugepages (MAP_HUGETLB | MAP_HUGE_1GB) so it consumes only 1 STLB entry instead of 32K entries on 4KB pages. "
                "Do NOT use `if (j % N == 0)` — that pollutes TAGE prediction.",
                "Use a TWO-TIER chain strategy: a 1 MB hot chain (16384 x 64B) chased every iteration for L1d misses, plus the 128 MB cold chain chased every N iterations for LLC misses. The two chains serve different features independently.",
            )
        return (
            "Reduce the cold-chain access frequency (increase N) or shrink the cold chain below LLC size.",
            "Only shrink the fraction of accesses that reach DRAM.",
        )
    if feature_name == "perf_stat.L1-icache-load-misses_MPKI":
        if status == "too low":
            return (
                "PROVEN PATTERN: Create 3 __attribute__((noinline)) functions each with a 256-case switch (N ALU ops per case). Rotate with j%3 in the hot loop. Switch index = payload & 0xFF from a pointer-chase load. L1i MPKI scales linearly with ops-per-case (8->~11, 10->~10, 12->~12, 14->~15 MPKI).",
                "Tune ops-per-case to reach target L1i MPKI. CAUTION: adding hot-chain walkers dilutes L1i MPKI by increasing instruction count without proportional icache misses.",
            )
        if status == "too high":
            return (
                "Reduce ops-per-case in the switch functions or reduce the number of switch functions to shrink the hot instruction footprint.",
                "Prefer removing one excess hot helper or collapsing one oversized hot path.",
            )
    if feature_name in {"perf_stat.L1-dcache-loads_LPKI", "perf_stat.L1-dcache-stores_SPKI"}:
        traffic_kind = "loads" if "loads" in feature_name else "stores"
        asm_example = 'asm volatile("movq (%0), %0" : "+r"(pos))' if traffic_kind == "loads" else 'asm volatile("movq %0, (%1)" : : "r"(val), "r"(addr) : "memory")'
        return (
            f"PROVEN PATTERN — INLINE ASM: Use `{asm_example}` in a loop. Each inline asm instruction is exactly 1 insn producing exactly 1 {traffic_kind[:-1]}. "
            f"N asm {traffic_kind} in a loop body = N {traffic_kind} per (N + overhead) instructions. CRITICAL: loop the asm (not unroll) to keep code footprint under ~2KB. "
            f"INSTRUCTION DILUTION TRAP: {traffic_kind.upper()[:-1]}PKI = {traffic_kind} / (total_insn / 1000). Adding non-{traffic_kind[:-1]} instructions dilutes the metric equally.",
            "Keep the access pattern local and preserve already-good branch/frontend behavior.",
        )
    if feature_name == "perf_stat.L1-dcache-load-misses_MPKI":
        if status == "too low":
            return (
                "PROVEN PATTERN: Use a 1 MB linked-list pointer-chase hot chain (16384 nodes x 64B, Fisher-Yates shuffled). Exceeds L1d (48 KB) but fits L2 (2 MB). For extra L1d pressure, add 2-3 independent hot-chain walkers that each advance one step per iteration. Do NOT use hash-based or linear array indexing — hardware prefetchers defeat those patterns.",
                "Tune hot-chain node count (working set size), number of walkers (2-3 optimal), and iteration count (MPKI dilution ratio). CAUTION: walkers dilute L1i and branch-miss MPKI.",
            )
        return (
            "Reduce the hot-chain size or remove walkers to improve L1d locality.",
            "Do not inflate code size just to move the cache-miss number.",
        )
    if feature_name == "perf_stat.iTLB-load-misses_MPKI":
        return (
            "STLB CAPACITY MODEL: iTLB misses are governed by the L2 STLB (2048 entries). If hot code + data pages < 2048, STLB absorbs all iTLB misses. "
            "To increase iTLB: (1) increase code page count via more noinline functions or PAD_SIZE padding between them; "
            "(2) allocate ALL large data buffers on 1GB hugepages (MAP_HUGETLB | MAP_HUGE_1GB) so they consume only 1 STLB entry each, "
            "reserving STLB capacity for instruction pages. Empirical: PAD_SIZE=380 with 8192 functions → ~8640 code pages on Intel Xeon Gold 5512U.",
            "Prefer one compact extra hot helper over many unrelated changes.",
        )
    if feature_name == "tma.TMA_Backend_Bound(%)":
        return (
            "Increase backend stall exposure with one dependent-latency mechanism in the steady-state loop.",
            "Preserve already-good frontend/branch shape while doing so.",
        )
    if feature_name == "tma.TMA_Frontend_Bound(%)":
        return (
            "Increase frontend pressure with one compact hot-path footprint or control-flow perturbation.",
            "Avoid large cold setup code or unrelated helper scaffolding.",
        )
    return (
        "Choose one minimal local transformation that targets the dominant subsystem cause without rewriting the benchmark.",
        "Preserve metrics already close to target.",
    )


def _task_priority_feature_names(
    *,
    top_feature_gaps: list[dict[str, Any]],
    primary_feature_name: str,
    max_features: int = 6,
    freeze_ipc: bool = False,
    tier_state: dict[str, Any] | None = None,
) -> list[str]:
    ordered: list[str] = []

    def add(feature_name: str) -> None:
        name = str(feature_name or "").strip()
        if name and name not in ordered:
            ordered.append(name)

    tier_state = tier_state or priority_tier_state(top_feature_gaps)
    for row in list(tier_state.get("active_focus_rows") or []):
        add(str(row.get("feature_name", "")).strip())
        if len(ordered) >= max_features:
            return ordered[:max_features]

    if primary_feature_name.startswith("perf_stat."):
        add(primary_feature_name)
    if not freeze_ipc:
        add("perf_stat.IPC")

    for row in _corrective_feature_rows(top_feature_gaps, freeze_ipc=freeze_ipc):
        feature_name = str(row.get("feature_name", "")).strip()
        if not feature_name.startswith("perf_stat."):
            continue
        add(feature_name)
        if len(ordered) >= max_features:
            break
    return ordered[:max_features]


def _ordered_feature_names_from_rows(
    rows: list[dict[str, Any]],
    *,
    max_features: int | None = None,
) -> list[str]:
    ordered: list[str] = []
    for row in rows:
        feature_name = str(row.get("feature_name", "")).strip()
        if not feature_name.startswith("perf_stat."):
            continue
        if feature_name in ordered:
            continue
        ordered.append(feature_name)
        if max_features is not None and len(ordered) >= max_features:
            break
    return ordered


def _task_protected_feature_names(
    *,
    top_feature_gaps: list[dict[str, Any]],
    focus_feature_names: list[str],
    max_features: int = 4,
    tier_state: dict[str, Any] | None = None,
) -> list[str]:
    tier_state = tier_state or priority_tier_state(top_feature_gaps)
    ordered: list[str] = []

    def add(feature_name: str) -> None:
        name = str(feature_name or "").strip()
        if name and name not in focus_feature_names and name not in ordered:
            ordered.append(name)

    protected_rows = list(tier_state.get("higher_priority_matched_rows") or []) + list(
        tier_state.get("active_protected_rows") or []
    )
    for feature_name in _ordered_feature_names_from_rows(protected_rows, max_features=max_features):
        add(feature_name)
        if len(ordered) >= max_features:
            break
    return ordered[:max_features]


def _task_deferred_feature_names(
    *,
    top_feature_gaps: list[dict[str, Any]],
    focus_feature_names: list[str],
    protected_feature_names: list[str],
    max_features: int = 4,
    tier_state: dict[str, Any] | None = None,
) -> list[str]:
    tier_state = tier_state or priority_tier_state(top_feature_gaps)
    ordered: list[str] = []

    def add(feature_name: str) -> None:
        name = str(feature_name or "").strip()
        if (
            name
            and name not in focus_feature_names
            and name not in protected_feature_names
            and name not in ordered
        ):
            ordered.append(name)

    for feature_name in _ordered_feature_names_from_rows(
        list(tier_state.get("deferred_rows") or []),
        max_features=max_features,
    ):
        add(feature_name)
        if len(ordered) >= max_features:
            break
    return ordered[:max_features]


def _task_protocol_rows(top_feature_gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in top_feature_gaps:
        feature_name = str(row.get("feature_name", "")).strip()
        if not feature_name.startswith("perf_stat."):
            continue
        rows.append(
            {
                "feature_name": feature_name,
                "target_value": row.get("target_value"),
                "candidate_value": row.get("baseline_value"),
                "tolerance_ratio": row.get("tolerance_ratio"),
                "within_tolerance": bool(row.get("within_tolerance")),
            }
        )
    return rows


def _ipc_should_be_guardrail(top_feature_gaps: list[dict[str, Any]]) -> bool:
    for row in top_feature_gaps:
        if str(row.get("feature_name", "")).strip() == "perf_stat.IPC":
            return bool(row.get("within_tolerance"))
    return False


_IPC_CACHE_BRANCH_COUPLED_FEATURES = (
    "perf_stat.L1-dcache-load-misses_MPKI",
    "perf_stat.LLC-load-misses_MPKI",
    "perf_stat.branch-misses_MPKI",
    "perf_stat.branch-load-misses_MPKI",
)
_IPC_CACHE_BRANCH_COUPLED_RATIO = 5.0
_IPC_CACHE_BRANCH_COUPLED_MIN_COUNT = 2


def _ipc_cache_branch_coupled(
    top_feature_gaps: list[dict[str, Any]],
    freeze_ipc: bool,
) -> bool:
    'Detect the IPC-cache/branch coupling scenario.'
    coupled_count = 0
    ipc_baseline: float | None = None
    ipc_target: float | None = None
    for gap in top_feature_gaps:
        fname = str(gap.get("feature_name", "")).strip()
        target = gap.get("target_value")
        baseline = gap.get("baseline_value")
        if fname == "perf_stat.IPC":
            if target is not None and baseline is not None:
                ipc_baseline = float(baseline)
                ipc_target = float(target)
            continue
        if fname not in _IPC_CACHE_BRANCH_COUPLED_FEATURES:
            continue
        if (
            target is not None
            and baseline is not None
            and float(target) > 0
            and float(baseline) > 0
            and float(target) / max(float(baseline), 1e-12) >= _IPC_CACHE_BRANCH_COUPLED_RATIO
        ):
            coupled_count += 1
    if coupled_count < _IPC_CACHE_BRANCH_COUPLED_MIN_COUNT:
        return False
    # Case A: IPC already near target but cache/branch counters expose
    # that generated code will be compute-bound.
    if freeze_ipc:
        return True
    # Case B: IPC baseline is well above target — same coupling, same fix.
    if (
        ipc_baseline is not None
        and ipc_target is not None
        and ipc_target > 0
        and ipc_baseline / ipc_target >= 1.3
    ):
        return True
    return False


def _compact_priority_lines(
    *,
    active_priority_rank: int | None,
    focus_feature_names: list[str],
    protected_feature_names: list[str],
    deferred_feature_names: list[str],
) -> list[str]:
    lines = ["1. buildability and benchmark validity"]
    next_index = 2

    if protected_feature_names:
        protected_labels = ", ".join(
            feature_display_name(feature_name) for feature_name in protected_feature_names
        )
        lines.append(
            f"{next_index}. Strictly conserve already-matched prioritized counters: {protected_labels}"
        )
        next_index += 1
    if focus_feature_names:
        focus_labels = ", ".join(
            feature_display_name(feature_name) for feature_name in focus_feature_names
        )
        if active_priority_rank is not None:
            lines.append(
                f"{next_index}. Close {priority_tier_label(active_priority_rank)} before touching lower tiers: {focus_labels}"
            )
        else:
            lines.append(f"{next_index}. Refine the remaining constrained counters: {focus_labels}")
        next_index += 1
    if deferred_feature_names:
        deferred_labels = ", ".join(
            feature_display_name(feature_name) for feature_name in deferred_feature_names
        )
        lines.append(f"{next_index}. Do not target lower tiers yet: {deferred_labels}")
    return lines


def _compact_baseline_lines(
    *,
    top_feature_gaps: list[dict[str, Any]],
    focus_feature_names: list[str],
    protected_feature_names: list[str],
    deferred_feature_names: list[str],
) -> list[str]:
    by_name = {
        str(row.get("feature_name", "")).strip(): row
        for row in top_feature_gaps
        if str(row.get("feature_name", "")).strip()
    }
    lines: list[str] = []
    if protected_feature_names:
        fragments: list[str] = []
        for feature_name in protected_feature_names:
            row = by_name.get(feature_name)
            if not row:
                continue
            fragments.append(
                f"{feature_display_name(feature_name)}={_format_metric_value(row.get('baseline_value'))}"
            )
        if fragments:
            lines.append("- Protected higher-priority counters: " + "; ".join(fragments))
    for feature_name in focus_feature_names:
        row = by_name.get(feature_name)
        if not row:
            continue
        label = feature_display_name(feature_name)
        baseline_value = _format_metric_value(row.get("baseline_value"))
        lines.append(f"- {label}: {baseline_value}")
    if deferred_feature_names:
        fragments: list[str] = []
        for feature_name in deferred_feature_names:
            row = by_name.get(feature_name)
            if not row:
                continue
            fragments.append(
                f"{feature_display_name(feature_name)}={_format_metric_value(row.get('baseline_value'))}"
            )
        if fragments:
            lines.append("- Deferred lower-tier counters: " + "; ".join(fragments))
    return lines


def _compact_target_lines(
    *,
    top_feature_gaps: list[dict[str, Any]],
    focus_feature_names: list[str],
    protected_feature_names: list[str],
    deferred_feature_names: list[str],
) -> list[str]:
    by_name = {
        str(row.get("feature_name", "")).strip(): row
        for row in top_feature_gaps
        if str(row.get("feature_name", "")).strip()
    }
    lines: list[str] = []
    if protected_feature_names:
        fragments: list[str] = []
        for feature_name in protected_feature_names:
            row = by_name.get(feature_name)
            if not row:
                continue
            target_value = row.get("target_value")
            tolerance_ratio = row.get("tolerance_ratio")
            if tolerance_ratio is not None:
                try:
                    acceptable_min, acceptable_max = tolerance_bounds(float(target_value), float(tolerance_ratio))
                    fragments.append(
                        f"{feature_display_name(feature_name)} target={_format_metric_value(target_value)} "
                        f"(acceptable {acceptable_min:.6g}..{acceptable_max:.6g})"
                    )
                    continue
                except (TypeError, ValueError):
                    pass
            fragments.append(
                f"{feature_display_name(feature_name)} target={_format_metric_value(target_value)}"
            )
        if fragments:
            lines.append("- Protected higher-priority targets: " + "; ".join(fragments))
    for feature_name in focus_feature_names:
        row = by_name.get(feature_name)
        if not row:
            continue
        label = feature_display_name(feature_name)
        target_value = row.get("target_value")
        tolerance_ratio = row.get("tolerance_ratio")
        if tolerance_ratio is not None:
            try:
                acceptable_min, acceptable_max = tolerance_bounds(float(target_value), float(tolerance_ratio))
                lines.append(
                    f"- {label}: target={_format_metric_value(target_value)}, "
                    f"acceptable={acceptable_min:.6g}..{acceptable_max:.6g}"
                )
                continue
            except (TypeError, ValueError):
                pass
        lines.append(
            f"- {label}: target={_format_metric_value(target_value)}"
        )
    if deferred_feature_names:
        fragments: list[str] = []
        for feature_name in deferred_feature_names:
            row = by_name.get(feature_name)
            if not row:
                continue
            target_value = row.get("target_value")
            tolerance_ratio = row.get("tolerance_ratio")
            if tolerance_ratio is not None:
                try:
                    acceptable_min, acceptable_max = tolerance_bounds(float(target_value), float(tolerance_ratio))
                    fragments.append(
                        f"{feature_display_name(feature_name)} target={_format_metric_value(target_value)} "
                        f"(acceptable {acceptable_min:.6g}..{acceptable_max:.6g})"
                    )
                    continue
                except (TypeError, ValueError):
                    pass
            fragments.append(
                f"{feature_display_name(feature_name)} target={_format_metric_value(target_value)}"
            )
        if fragments:
            lines.append("- Deferred lower-tier targets: " + "; ".join(fragments))
    return lines


def _compact_diagnosis_lines(
    *,
    top_feature_gaps: list[dict[str, Any]],
    active_priority_rank: int | None,
    focus_feature_names: list[str],
    protected_feature_names: list[str],
    deferred_feature_names: list[str],
    freeze_ipc: bool = False,
) -> list[str]:
    protocol_rows = _task_protocol_rows(top_feature_gaps)
    by_name = {str(row.get("feature_name", "")).strip(): row for row in protocol_rows}
    lines: list[str] = [
        "- Work tier by tier instead of trying to move all counters at once.",
        (
            f"- The first unsatisfied tier is {priority_tier_label(active_priority_rank)}. "
            "Do not spend effort on lower-tier counters until this tier is within range."
        )
        if active_priority_rank is not None
        else "- All prioritized tiers are currently within range; preserve them while making only minimal local corrections.",
        "- Classify each priority metric as too high, too low, or in range.",
        "- Treat counter gaps as symptoms, not direct edit targets.",
        "- Infer the likely subsystem cause before changing code.",
        "- Choose the smallest code transformation that targets that cause.",
    ]
    if protected_feature_names:
        labels = ", ".join(feature_display_name(feature_name) for feature_name in protected_feature_names)
        lines.append(f"- Strictly conserve already-matched prioritized counters: {labels}.")
    for feature_name in focus_feature_names:
        row = by_name.get(feature_name)
        if not row:
            continue
        label = feature_display_name(feature_name)
        status = _metric_status(row)
        lines.append(f"- Current status: {label} is {status}.")
    coupled = _ipc_cache_branch_coupled(top_feature_gaps, freeze_ipc)
    if coupled:
        lines.append(
            "- IPC-CACHE/BRANCH COUPLING: Cache and branch MPKIs are far below target and the code is "
            "compute-bound with much higher effective IPC than the target workload. "
            "Adding memory stalls and unpredictable branches is the primary mechanism to simultaneously "
            "lower IPC and raise cache/branch counters. Treat IPC as a consequence of fixing the "
            "memory/branch profile, not as a separate guardrail to preserve. "
            "Use a large permuted chase array (>64 KB) with dependent loads and data-dependent branches; "
            "simple if/else or hash-based indexing is defeated by modern hardware prefetchers."
        )
    elif freeze_ipc:
        lines.append("- IPC is already within range; treat it as a guardrail, not a primary objective in this attempt.")
    if deferred_feature_names:
        labels = ", ".join(feature_display_name(feature_name) for feature_name in deferred_feature_names)
        lines.append(f"- Defer lower-tier counters for later attempts: {labels}.")
    return lines


def _adapt_strategy_language(text: str, *, from_scratch: bool) -> str:
    rendered = str(text or "").strip()
    if not rendered or not from_scratch:
        return rendered
    replacements = [
        ("in the existing hot path", "in the new benchmark hot path"),
        ("existing hot path", "new benchmark hot path"),
        ("existing benchmark", "source-family benchmark"),
        ("source benchmark", "source benchmark family"),
        ("the benchmark", "the new benchmark"),
    ]
    for old, new in replacements:
        rendered = rendered.replace(old, new)
    return rendered


def _is_hash_map_benchmark(binary_name: str, target_file_name: str) -> bool:
    binary_lc = str(binary_name).lower()
    target_lc = Path(str(target_file_name)).name.lower()
    return "hash_maps" in binary_lc or target_lc == "hashmapsbench.cpp"


def _compact_transformation_policy_lines(
    *,
    top_feature_gaps: list[dict[str, Any]],
    active_priority_rank: int | None,
    primary_feature_name: str,
    focus_feature_names: list[str],
    protected_feature_names: list[str],
    deferred_feature_names: list[str],
    from_scratch: bool,
    freeze_ipc: bool = False,
    binary_name: str = "",
    target_file_name: str = "",
) -> list[str]:
    protocol_rows = _task_protocol_rows(top_feature_gaps)
    by_name = {str(row.get("feature_name", "")).strip(): row for row in protocol_rows}
    primary_row = by_name.get(primary_feature_name) or (protocol_rows[0] if protocol_rows else {})
    branch_rows = [
        row
        for row in [
            by_name.get("perf_stat.branch-misses_MPKI"),
            by_name.get("perf_stat.branch-load-misses_MPKI"),
        ]
        if row
    ]
    branch_overshoot = any(_metric_status(row) == "too high" for row in branch_rows)
    primary_fix, secondary_fix = _recommended_transformations(
        str(primary_row.get("feature_name", "")).strip(),
        status=_metric_status(primary_row) if primary_row else "out of range",
        by_name=by_name,
        branch_overshoot=branch_overshoot,
    )
    lines = ["- Use one primary mechanism only."]
    if active_priority_rank is not None:
        lines.append(
            f"- Work only the first unsatisfied tier, {priority_tier_label(active_priority_rank)}, until it is within tolerance."
        )
    if protected_feature_names:
        labels = ", ".join(feature_display_name(feature_name) for feature_name in protected_feature_names)
        lines.append(
            f"- Strictly conserve already-matched prioritized counters: {labels}. If a change threatens them, back it out and try a different lever."
        )
    coupled = _ipc_cache_branch_coupled(top_feature_gaps, freeze_ipc)
    lines.extend(
        [
            f"- {_adapt_strategy_language(primary_fix, from_scratch=from_scratch)}",
            "- The new benchmark must materially change the timed hot path. Do not submit a rename-only clone, registration-only alias, or a bare noinline wrapper around identical work.",
            "- Preserve the current benchmark's throughput-carrying structure.",
            "- Do not change loop trip counts, benchmark scale, or the main steady-state serialization/deserialization or operation volume unless the controller objective explicitly requires it.",
        ]
    )
    if coupled:
        lines.append(
            "- IPC-CACHE COUPLING: The seed benchmark is compute-bound with much higher effective IPC "
            "than the target. Adding memory stalls (pointer-chasing through permuted arrays exceeding "
            "L1D capacity, data-dependent branches gated on load values) will SIMULTANEOUSLY lower IPC "
            "toward the target AND raise cache-miss and branch-miss counters. "
            "This coupling is expected — do not avoid memory-bound or branch-unpredictable patterns out of "
            "concern for IPC. Ensure the inner loop executes 100K+ iterations to dominate instruction count."
        )
        lines.append(
            "- MANDATORY CODE PATTERN — you MUST use this exact pointer-chasing + branch pattern inside "
            "the timed hot path. Simple if/else branches or linear array walks are NOT sufficient because "
            "hardware prefetchers and branch predictors learn them instantly:\n"
            "```\n"
            "// Setup: build linked-list pointer-chase chains\n"
            "struct ChainNode { ChainNode* next; char pad[48]; }; // 64 bytes per node\n"
            "// Hot chain: 1 MB (16384 nodes) — exceeds L1d (48KB), fits L2 (2MB)\n"
            "static constexpr size_t kHotLen = 16384;\n"
            "static ChainNode hotNodes[kHotLen];\n"
            "// Cold chain: 128 MB (2M nodes) — exceeds LLC for DRAM misses\n"
            "static constexpr size_t kColdLen = 2u << 20; // 2097152\n"
            "static ChainNode* coldNodes = new ChainNode[kColdLen];\n"
            "// Fisher-Yates shuffle to create Hamiltonian cycles\n"
            "static void initChain(ChainNode* nodes, size_t len) {\n"
            "  std::vector<size_t> perm(len);\n"
            "  std::iota(perm.begin(), perm.end(), 0u);\n"
            "  std::mt19937_64 rng(42);\n"
            "  for (size_t i = len - 1; i > 0; --i)\n"
            "    std::swap(perm[i], perm[rng() %% (i + 1)]);\n"
            "  for (size_t i = 0; i < len; ++i)\n"
            "    nodes[perm[i]].next = &nodes[perm[(i + 1) %% len]];\n"
            "}\n"
            "// 3 noinline 256-case switch functions (N ALU ops per case)\n"
            "// — generates L1i pressure + branch misses via indirect misprediction\n"
            "__attribute__((noinline)) uint64_t switchA(uint64_t v, int c) {\n"
            "  switch(c) { case 0: v+=0x1234; v^=0x5678; /* ... 256 cases */ }\n"
            "  return v;\n"
            "}\n"
            "// Inside timed loop:\n"
            "ChainNode* hotPos = &hotNodes[0];\n"
            "ChainNode* coldPos = &coldNodes[0];\n"
            "uint64_t acc = 0;\n"
            "for (size_t j = 0; j < INNER_ITERS; ++j) {\n"
            "  hotPos = hotPos->next; // L1d miss (pointer-chase)\n"
            "  uint64_t payload = (uint64_t)(uintptr_t)hotPos;\n"
            "  // Rotate among 3 switch functions — L1i + branch misses\n"
            "  switch (j %% 3) {\n"
            "    case 0: acc = switchA(acc, payload & 0xFF); break;\n"
            "    case 1: acc = switchB(acc, payload & 0xFF); break;\n"
            "    case 2: acc = switchC(acc, payload & 0xFF); break;\n"
            "  }\n"
            "  // Branchless cold-chain access every N steps (LLC misses)\n"
            "  auto coldMask = -uint64_t(j %% 8 == 0);\n"
            "  coldPos = (ChainNode*)((uintptr_t)coldPos->next & coldMask\n"
            "            | (uintptr_t)coldPos & ~coldMask);\n"
            "}\n"
            "folly::doNotOptimizeAway(acc);\n"
            "```\n"
            "Adapt this into the benchmark body. CRITICAL SIZING:\n"
            "  - Hot chain: 1 MB (16384 x 64B) for L1d MPKI. Add 2-3 independent walkers for more L1d pressure.\n"
            "  - Cold chain: 128 MB (2M x 64B) for LLC MPKI. Access frequency N controls LLC miss rate.\n"
            "  - 3 switch functions x 256 cases x N ALU ops/case for L1i MPKI. Ops/case controls L1i linearly.\n"
            "  - Iteration count MUST be >= 500K to dominate the instruction mix over framework overhead.\n"
            "  - Use branchless cold-access scheduling to avoid TAGE predictor pollution.\n"
            "Keep the original benchmark's registration and structure but replace or augment the timed body."
        )
    elif freeze_ipc:
        lines.append("- Do not add new critical-path work that can materially reduce IPC.")
    if any(feature_name in FRONTEND_CONTROL_FEATURE_NAMES for feature_name in focus_feature_names):
        lines.append(
            "- Prefer compact hot-path control-flow or frontend changes first when IPC or branch/frontend counters are off."
        )
        if not coupled:
            lines.append(
                "- Introduce frontend pressure through compact control-flow or instruction-footprint changes that stay off the deepest dependency chain when possible."
            )
            lines.append(
                "- If a proposed frontend change risks lowering IPC, reduce its frequency or move it off the hottest path."
            )
    if secondary_fix and str(primary_row.get("feature_name", "")).strip() == "perf_stat.IPC":
        lines.append(f"- Only if IPC remains stuck after the first lever: {_adapt_strategy_language(secondary_fix, from_scratch=from_scratch)}")
    lines.extend(
        [
            "- Avoid broad rewrites that change the benchmark character. The proven 3-noinline-switch + pointer-chase-chain pattern is expected and should not be considered 'bulky' — it is the primary mechanism.",
            "- Preserve benchmark character and keep changes targeted to specific feature knobs.",
            "- ITERATION COUNT IS THE PRIMARY TUNING KNOB: When the benchmark framework controls total runtime, the inner loop iteration count determines what fraction of instructions come from tuned code vs framework overhead. Adjust iteration count to control the blend of all MPKI features simultaneously.",
        ]
    )
    if coupled:
        lines.append(
            "- IPC will naturally fall toward the target once memory and branch pressure are added. "
            "Do not try to preserve the seed's current IPC level — it reflects compute-bound execution "
            "that does not match the target workload's memory-bound character."
        )
    elif freeze_ipc:
        lines.append("- IPC is already within range; keep it stable while moving the dominant miss counters.")
    else:
        lines.append("- If goals conflict, preserve IPC before lower-tier counters.")
    if _is_hash_map_benchmark(binary_name, target_file_name):
        lines.append(
            "- For hash-map benchmarks, change executed branch entropy, memory-touch pattern, reuse distance, or cache footprint inside the timed loop itself rather than only wrapping the same map operation."
        )
        if any(feature_name in MEMORY_SIDE_FEATURE_NAMES for feature_name in focus_feature_names):
            lines.append(
                "- To move L1D counters, prefer changing probe sequence, touched state, reuse distance, or table footprint over only increasing code size."
            )
        if any(feature_name in FRONTEND_CONTROL_FEATURE_NAMES for feature_name in focus_feature_names):
            lines.append(
                "- To move branch or icache counters, add one compact alternate hot path or lightweight per-iteration state-driven decision that actually changes executed control flow."
            )
    if deferred_feature_names:
        labels = ", ".join(feature_display_name(feature_name) for feature_name in deferred_feature_names)
        lines.append(f"- Ignore lower-tier counters for now: {labels}. Only move to them in later attempts after the active tier is matched.")
    return lines


def _compact_feature_guidance_lines(
    feature_name: str,
    direction: str,
    *,
    max_lines: int = 3,
) -> list[str]:
    lines: list[str] = []
    for line in feature_prompt_guidance(feature_name, direction):
        cleaned = str(line).strip()
        if not cleaned or cleaned in lines:
            continue
        lines.append(cleaned)
        if len(lines) >= max_lines:
            break
    return lines


def _supplemental_feature_guidance_lines(
    *,
    binary_name: str,
    target_file_name: str,
    feature_name: str,
    freeze_ipc: bool,
) -> list[str]:
    lines: list[str] = []
    if _is_hash_map_benchmark(binary_name, target_file_name):
        lines.append(
            "For hash-map benchmarks, change the timed-loop branch entropy, memory-touch pattern, reuse distance, or hot footprint directly; a wrapper-only clone or noinline shell around identical work is too weak."
        )
        if feature_name in MEMORY_SIDE_FEATURE_NAMES:
            lines.append(
                "To move L1D counters, change probe order, touched state, reuse distance, or table footprint rather than only growing code size."
            )
        if feature_name in FRONTEND_CONTROL_FEATURE_NAMES:
            lines.append(
                "To move branch or icache counters, add compact alternate hot control flow that actually changes executed decisions in the steady-state loop."
            )
    if freeze_ipc:
        lines.append("IPC is already in range; keep it as a guardrail while you move the dominant miss counters.")
        lines.append("Preserve the current throughput-carrying structure. Do not change loop trip counts, benchmark scale, or the main steady-state work volume unless the controller objective explicitly requires it.")
    return lines


def _build_property_task(
    *,
    binary_name: str,
    source_microbenchmark: str,
    target_info: dict[str, Any],
    feature_name: str,
    target_value: float,
    baseline_value: float,
    delta_to_target: float,
    feature_scales: dict[str, float],
    top_feature_gaps: list[dict[str, Any]],
    from_scratch: bool = False,
    baseline_hw_accuracy_pct: float | None = None,
    baseline_hw_threshold_pct: float | None = None,
) -> dict[str, Any]:
    application_seed = is_application_seed_unit(source_microbenchmark)
    top_feature_gaps = _prompt_hardware_feature_rows(top_feature_gaps)
    ipc_guardrail_only = _ipc_should_be_guardrail(top_feature_gaps)
    scale = max(float(feature_scales.get(feature_name, 1.0)), 1e-9)
    direction = "increase" if delta_to_target >= 0 else "decrease"
    magnitude = _classify_magnitude(abs(delta_to_target), scale)
    manifest_row = _lookup_property_task(
        binary_name,
        source_microbenchmark,
        feature_name,
        direction,
        magnitude,
    )
    anchors = (
        [str(anchor) for anchor in manifest_row.get("style_anchors", []) if str(anchor).strip()]
        if manifest_row is not None
        else _style_anchors(binary_name, source_microbenchmark)
    )
    resolved_target_info = {
        "target_file": str((manifest_row or {}).get("target_file", target_info["target_file"])),
        "source_root": str((manifest_row or {}).get("source_root", target_info["source_root"])),
        "build_dir": "",
    }
    normalized_source_root = Path(resolved_target_info["source_root"])
    configured_build_dir = Path(str((manifest_row or {}).get("build_dir", target_info["build_dir"])))
    resolved_target_info["build_dir"] = str(
        _normalize_build_dir(configured_build_dir, normalized_source_root, binary_name)
    )
    lower_bound, upper_bound, target_center = _task_bounds(scale, magnitude)
    feature_label = feature_display_name(feature_name)
    constraint = feature_constraint(feature_name)
    target_file_path = normalized_source_root / resolved_target_info["target_file"]
    edit_location_clause = _edit_location_clause(target_file_path, source_microbenchmark)
    if from_scratch:
        edit_location_clause = _edit_location_clause(
            target_file_path,
            source_microbenchmark,
            from_scratch=True,
        )
    acceptable_min = None
    acceptable_max = None
    if constraint is not None:
        try:
            acceptable_min, acceptable_max = tolerance_bounds(target_value, constraint.tolerance_ratio)
        except (TypeError, ValueError):
            pass
    guidance_lines = _compact_feature_guidance_lines(feature_name, direction)
    for line in _supplemental_feature_guidance_lines(
        binary_name=binary_name,
        target_file_name=resolved_target_info["target_file"],
        feature_name=feature_name,
        freeze_ipc=ipc_guardrail_only,
    ):
        cleaned = str(line).strip()
        if cleaned and cleaned not in guidance_lines:
            guidance_lines.append(cleaned)
    strategy = (
        "in_place_application"
        if application_seed
        else ("from_scratch" if from_scratch else "derived_from_nearest")
    )
    registration_idiom_clause = _registration_idiom_clause(target_file_path)
    tier_state = priority_tier_state(top_feature_gaps)
    active_priority_rank = tier_state.get("active_priority_rank")
    focus_feature_names = _task_priority_feature_names(
        top_feature_gaps=top_feature_gaps,
        primary_feature_name=feature_name,
        freeze_ipc=ipc_guardrail_only,
        tier_state=tier_state,
    )
    protected_feature_names = _task_protected_feature_names(
        top_feature_gaps=top_feature_gaps,
        focus_feature_names=focus_feature_names,
        tier_state=tier_state,
    )
    deferred_feature_names = _task_deferred_feature_names(
        top_feature_gaps=top_feature_gaps,
        focus_feature_names=focus_feature_names,
        protected_feature_names=protected_feature_names,
        tier_state=tier_state,
    )
    priority_lines = _compact_priority_lines(
        active_priority_rank=active_priority_rank,
        focus_feature_names=focus_feature_names,
        protected_feature_names=protected_feature_names,
        deferred_feature_names=deferred_feature_names,
    )
    baseline_lines = _compact_baseline_lines(
        top_feature_gaps=top_feature_gaps,
        focus_feature_names=focus_feature_names,
        protected_feature_names=protected_feature_names,
        deferred_feature_names=deferred_feature_names,
    )
    target_lines = _compact_target_lines(
        top_feature_gaps=top_feature_gaps,
        focus_feature_names=focus_feature_names,
        protected_feature_names=protected_feature_names,
        deferred_feature_names=deferred_feature_names,
    )
    diagnosis_lines = _compact_diagnosis_lines(
        top_feature_gaps=top_feature_gaps,
        active_priority_rank=active_priority_rank,
        focus_feature_names=focus_feature_names,
        protected_feature_names=protected_feature_names,
        deferred_feature_names=deferred_feature_names,
        freeze_ipc=ipc_guardrail_only,
    )
    transformation_policy_lines = _compact_transformation_policy_lines(
        top_feature_gaps=top_feature_gaps,
        active_priority_rank=active_priority_rank,
        primary_feature_name=feature_name,
        focus_feature_names=focus_feature_names,
        protected_feature_names=protected_feature_names,
        deferred_feature_names=deferred_feature_names,
        from_scratch=from_scratch,
        freeze_ipc=ipc_guardrail_only,
        binary_name=binary_name,
        target_file_name=resolved_target_info["target_file"],
    )
    style_reference_clause = (
        f"Style references: {source_microbenchmark} and nearby variants such as {', '.join(anchors)}."
        if anchors
        else f"Style reference: {source_microbenchmark}."
    )
    coupled = _ipc_cache_branch_coupled(top_feature_gaps, ipc_guardrail_only)
    if coupled:
        objective_text = (
            "Objective:\n"
            "This is attempt 1. The seed benchmark is compute-bound (IPC far above target) and "
            "cache/branch miss counters are 10-100× below target. You MUST add a large permuted "
            "chase array (>64 KB) with pointer-chasing dependent loads and data-dependent branches "
            "in the hot loop. This is the ONLY mechanism that will move L1-dcache MPKI, LLC MPKI, "
            "and branch MPKI toward target. Do not use simple conditionals, linear scans, or hash "
            "functions — hardware prefetchers defeat them."
        )
    else:
        objective_text = (
            "Objective:\n"
            "This is attempt 1. Produce one nearby candidate that moves counters in the right direction. "
            "Do not try to perfectly solve every metric in one shot."
        )
    if application_seed:
        constraint_lines = [
            f"- Edit only the existing timed encoder path in {resolved_target_info['target_file']}; do not add a benchmark registration or a second application entry point.",
            "- Preserve the SVT-AV1 public API, command-line interface, frame count, preset, CRF, GOP, and single-LP execution semantics used by the seed command.",
            "- Keep the edit local and reversible. Do not add synthetic standalone benchmark loops, test-only code, or unrelated harness machinery.",
            "- Preserve encoder correctness and memory safety. Do not skip required stages or return fabricated bitstream results.",
            "- Change one concrete hot-path mechanism at a time so the next PMU measurement can attribute the effect.",
            "- Do not change any existing function signature, return type, required call, structure layout, ownership rule, array bound, or pointer lifetime.",
            "- Do not replace an existing function body or duplicate existing encoder logic. Make a surgical patch confined to one existing function and keep the original control-flow stages intact.",
            "- Avoid new division, modulo, unchecked indexing, pointer arithmetic, allocation, recursion, or writes through newly derived pointers. The candidate must complete the exact seed command without SIGFPE, SIGSEGV, abort, or assertion failure.",
            "- Keep the patch small: change at most 40 source lines. If the requested counter movement cannot be achieved safely within that limit, make a smaller directional change.",
        ]
        task_sections = [
            (
                "Task:\n"
                f"Refine the existing sustained SVT-AV1 encoder path in {Path(resolved_target_info['target_file']).name} in place for the {binary_name} application seed.\n"
                f"The exact seed command remains `{source_microbenchmark}`; it will be rebuilt and rerun after this edit.\n"
                "Do not create or register a new microbenchmark."
            ),
            "Hard constraints:\n" + "\n".join(constraint_lines),
            objective_text,
            "Priority:\n" + "\n".join(priority_lines),
            "Baseline:\n" + "\n".join(baseline_lines),
            "Targets:\n" + "\n".join(target_lines),
            "Diagnosis to follow internally:\n" + "\n".join(diagnosis_lines),
            "Transformation policy:\n" + "\n".join(transformation_policy_lines),
        ]
        task_text = "\n\n".join(section for section in task_sections if section.strip())
    elif from_scratch:
        experiment_line = ""
        if baseline_hw_accuracy_pct is not None and baseline_hw_threshold_pct is not None:
            experiment_line = (
                f"This is a from-scratch experiment because the nearest retrieved benchmark matches only "
                f"{baseline_hw_accuracy_pct:.1f}% average hardware accuracy, below the {baseline_hw_threshold_pct:.1f}% threshold."
            )
        constraint_lines = []
        if experiment_line:
            constraint_lines.append(f"- {experiment_line}")
        constraint_lines.extend(
            [
                f"- Do not clone or minimally edit the body of {source_microbenchmark}; use it only as a style and placement reference.",
                f"- {edit_location_clause}",
            ]
        )
        if registration_idiom_clause:
            constraint_lines.append(f"- {registration_idiom_clause}")
        constraint_lines.extend(
            [
                "- The added benchmark name must be unique in --bm_list. Do not reuse an existing benchmark name, including the original source benchmark name.",
                "- The new benchmark must materially change the timed hot path. Do not use a rename-only clone, registration-only alias, or a bare noinline wrapper around identical work as the whole change.",
                "- Preserve includes, namespace usage, benchmark harness structure, and main() unless a small local build fix requires otherwise.",
                "- The result must add exactly one extra microbenchmark in --bm_list and must not modify unrelated benchmark families.",
            ]
        )
        task_sections = [
            (
                "Task:\n"
                f"Add exactly one new {_binary_subject(binary_name)} variant to {Path(resolved_target_info['target_file']).name} for the {binary_name} binary.\n"
                f"Write one new compact C or C++ microbenchmark from scratch that stays behaviorally close to the {source_microbenchmark} family, but do not clone or modify the existing {source_microbenchmark} benchmark.\n"
                f"Keep the original {source_microbenchmark} benchmarks unchanged.\n"
                f"{style_reference_clause}"
            ),
            "Hard constraints:\n" + "\n".join(line for line in constraint_lines if line.strip()),
            objective_text,
            "Priority:\n" + "\n".join(priority_lines),
            "Baseline:\n" + "\n".join(baseline_lines),
            "Targets:\n" + "\n".join(target_lines),
            "Diagnosis to follow internally:\n" + "\n".join(diagnosis_lines),
            "Transformation policy:\n" + "\n".join(transformation_policy_lines),
        ]
        task_text = "\n\n".join(section for section in task_sections if section.strip())
    else:
        constraint_lines = [f"- {edit_location_clause}"]
        if registration_idiom_clause:
            constraint_lines.append(f"- {registration_idiom_clause}")
        constraint_lines.extend(
            [
                "- The added benchmark name must be unique in --bm_list. Do not reuse an existing benchmark name, including the original source benchmark name.",
                "- The new benchmark must materially change the timed hot path. Do not use a rename-only clone, registration-only alias, or a bare noinline wrapper around identical work as the whole change.",
                "- Preserve includes, namespace usage, benchmark harness structure, and main() unless a small local build fix requires otherwise.",
                "- The result must add exactly one extra microbenchmark in --bm_list and must not modify unrelated benchmark families.",
            ]
        )
        task_sections = [
            (
                "Task:\n"
                f"Add exactly one new {_binary_subject(binary_name)} variant to {Path(resolved_target_info['target_file']).name} for the {binary_name} binary.\n"
                f"Add one nearby local variant in the {source_microbenchmark} family without rewriting the broader file.\n"
                f"Keep the original {source_microbenchmark} benchmark unchanged.\n"
                f"{style_reference_clause}"
            ),
            "Hard constraints:\n" + "\n".join(line for line in constraint_lines if line.strip()),
            objective_text,
            "Priority:\n" + "\n".join(priority_lines),
            "Baseline:\n" + "\n".join(baseline_lines),
            "Targets:\n" + "\n".join(target_lines),
            "Diagnosis to follow internally:\n" + "\n".join(diagnosis_lines),
            "Transformation policy:\n" + "\n".join(transformation_policy_lines),
        ]
        task_text = "\n\n".join(section for section in task_sections if section.strip())
    return {
        "task_id": _slugify(
            f"{binary_name}_{source_microbenchmark}_{feature_name}_{direction}_{magnitude}_{strategy}"
        ),
        "task": task_text,
        "binary": binary_name,
        "target_file": resolved_target_info["target_file"],
        "source_root": resolved_target_info["source_root"],
        "build_dir": resolved_target_info["build_dir"],
        "source_microbenchmark": source_microbenchmark,
        "style_anchors": anchors,
        "feature_name": feature_name,
        "feature_display_name": feature_label,
        "direction": direction,
        "magnitude": magnitude,
        "generation_strategy": strategy,
        "application_seed": application_seed,
        "baseline_hw_accuracy_pct": baseline_hw_accuracy_pct,
        "baseline_hw_threshold_pct": baseline_hw_threshold_pct,
        "target_value": target_value,
        "baseline_value": baseline_value,
        "scale": scale,
        "target_delta_lower_bound": lower_bound,
        "target_delta_upper_bound": upper_bound,
        "target_delta_center": target_center,
        "priority_group": constraint.group if constraint is not None else None,
        "priority_rank": constraint.priority_rank if constraint is not None else None,
        "tolerance_ratio": constraint.tolerance_ratio if constraint is not None else None,
        "acceptable_min": acceptable_min,
        "acceptable_max": acceptable_max,
        "feature_specific_guidance": guidance_lines,
    }


def run_cloud_tuning_attempt(
    *,
    run_id: str,
    binary_name: str,
    nearest_microbenchmark: str,
    target_features: dict[str, Any],
    xlsx_path: Path = DEFAULT_XLSX,
    retrieval_top_k: int = 8,
    perf_events: str,
    amd: bool = False,
    max_wall_seconds: float | None = None,
    warm_start_file: str | None = None,
    warm_start_features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    amd = amd or os.environ.get("CLOUD_SKIP_INTEL_PT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    started_at = time.monotonic()
    from scripts.cloning_priority import get_priority_mode, priority_tier_label
    print(
        f"[cloud-tuning] priority mode: {get_priority_mode()} "
        f"(primary objective {priority_tier_label(1)})",
        flush=True,
    )
    deadline_monotonic = _deadline_from_wall_seconds(max_wall_seconds)
    df = _load_feature_table(xlsx_path)
    feature_scales = _feature_scales(df)
    canonical_target = canonicalize_feature_dict(target_features)

    baseline_rows = df[
        (df["binary"].astype(str) == str(binary_name))
        & (df["microbenchmark"].astype(str) == str(nearest_microbenchmark))
    ]
    if baseline_rows.empty:
        raise RuntimeError(f"Could not find baseline row for {binary_name}::{nearest_microbenchmark}")
    baseline_row = baseline_rows.iloc[0].to_dict()
    table_baseline = canonicalize_feature_dict(baseline_row)

    target_info = _resolve_binary_target(binary_name, nearest_microbenchmark)
    example_id = _slugify(f"{run_id}_{binary_name}_{nearest_microbenchmark}")
    run_dir = PROPERTY_TASKS_DIR / example_id
    run_dir.mkdir(parents=True, exist_ok=True)

    build_dir = Path(target_info["build_dir"]).resolve()
    binary_path = _resolve_binary_path(
        build_dir,
        binary_name,
        Path(str(target_info["source_root"])),
    )
    baseline_binary_microbenchmarks, baseline_has_bm_list = _capture_baseline_microbenchmarks(
        binary_path=binary_path,
        binary_name=binary_name,
        source_microbenchmark=nearest_microbenchmark,
        build_dir=build_dir,
        source_root=Path(str(target_info["source_root"])),
    )
    if is_application_seed_unit(nearest_microbenchmark):
        _rebuild_application_seed_binary(
            run_dir=run_dir,
            binary_name=binary_name,
            binary_path=binary_path,
        )
    print(
        f"[cloud-tuning] baseline benchmark inventory for {binary_name}: "
        f"{len(baseline_binary_microbenchmarks)} entries (has_bm_list={baseline_has_bm_list})",
        flush=True,
    )
    live_baseline, live_baseline_error = _collect_live_anchor_features(
        run_dir=run_dir,
        binary_path=binary_path,
        microbenchmark_name=nearest_microbenchmark,
        has_bm_list=baseline_has_bm_list,
        perf_events=perf_events,
        amd=amd,
    )
    application_seed = is_application_seed_unit(nearest_microbenchmark)
    if live_baseline is None and live_baseline_error and not application_seed:
        print(
            f"[cloud-tuning] live baseline measurement unavailable for {binary_name}::{nearest_microbenchmark}; "
            f"falling back to feature table baseline ({live_baseline_error})",
            flush=True,
        )
    canonical_baseline, baseline_feature_source = _select_baseline_features(
        table_baseline=table_baseline,
        live_baseline=live_baseline,
        live_baseline_error=live_baseline_error,
        application_seed=application_seed,
        binary_name=binary_name,
        microbenchmark_name=nearest_microbenchmark,
    )
    baseline_report = _hardware_counter_match_report(canonical_target, canonical_baseline, feature_scales)
    baseline_constraint_report = _hardware_counter_constraint_report(canonical_target, canonical_baseline)
    baseline_error = float(baseline_report["weighted_mean_normalized_abs_error"])
    baseline_hw_accuracy_pct = _baseline_hw_accuracy_pct(canonical_target, canonical_baseline)
    baseline_rank_key = candidate_rank_key(
        baseline_constraint_report,
        baseline_error,
    )
    objective_priority_rank = _locked_objective_priority_rank(baseline_constraint_report)
    baseline_low_hw = bool(
        FROM_SCRATCH_IF_BASELINE_LOW
        and baseline_hw_accuracy_pct is not None
        and baseline_hw_accuracy_pct < FROM_SCRATCH_BASELINE_HW_THRESHOLD
    )
    delayed_from_scratch_mode = bool(
        baseline_low_hw and SEED_BASELINE_SELECTION_MODE == "ipc_near_non_ipc_hw"
    )
    baseline_has_conflicting_features = _baseline_has_feature_coupling(
        baseline_constraint_report
    )
    from_scratch_generation = bool(
        baseline_low_hw and not delayed_from_scratch_mode and not baseline_has_conflicting_features
    )
    if baseline_low_hw and baseline_has_conflicting_features and not delayed_from_scratch_mode:
        print(
            f"[cloud-tuning] baseline HW accuracy {baseline_hw_accuracy_pct:.1f}% < "
            f"{FROM_SCRATCH_BASELINE_HW_THRESHOLD:.1f}% but top misses involve conflicting features; "
            f"keeping iterative refinement instead of from-scratch generation.",
            flush=True,
        )

    top_gaps = _top_feature_gaps(
        _hardware_counter_features(canonical_target),
        _hardware_counter_features(canonical_baseline),
        feature_scales,
    )
    if not top_gaps:
        raise RuntimeError(
            f"No overlapping canonical features were available for {binary_name}::{nearest_microbenchmark}"
        )
    primary_gap = top_gaps[0]
    task_payload = _build_property_task(
        binary_name=binary_name,
        source_microbenchmark=nearest_microbenchmark,
        target_info=target_info,
        feature_name=str(primary_gap["feature_name"]),
        target_value=float(primary_gap["target_value"]),
        baseline_value=float(primary_gap["baseline_value"]),
        delta_to_target=float(primary_gap["delta_to_target"]),
        feature_scales=feature_scales,
        top_feature_gaps=top_gaps,
        from_scratch=from_scratch_generation,
        baseline_hw_accuracy_pct=baseline_hw_accuracy_pct,
        baseline_hw_threshold_pct=(
            FROM_SCRATCH_BASELINE_HW_THRESHOLD if from_scratch_generation else None
        ),
    )
    target_info = {
        "source_root": Path(str(task_payload["source_root"])),
        "build_dir": Path(str(task_payload["build_dir"])),
        "target_file": str(task_payload["target_file"]),
        "current_file": Path(str(task_payload["source_root"])) / str(task_payload["target_file"]),
    }
    task_text = str(task_payload["task"])

    ipc_guardrail_only = _ipc_should_be_guardrail(top_gaps)
    ipc_coupled_detected = _ipc_cache_branch_coupled(top_gaps, ipc_guardrail_only)
    ipc_coupled = ipc_coupled_detected and not STRICT_IPC_GUARDRAIL
    if ipc_coupled:
        print(
            "[cloud-tuning] IPC-cache/branch coupling detected — IPC guardrail relaxed, "
            f"stagnation minimum raised to {_IPC_COUPLED_MIN_ATTEMPTS} attempts.",
            flush=True,
        )
    elif ipc_coupled_detected:
        print(
            "[cloud-tuning] IPC-cache/branch coupling detected, but strict IPC guardrail remains enforced.",
            flush=True,
        )
    forbidden_benchmark_names_path = run_dir / "baseline_benchmark_names.json"
    _json_write(
        forbidden_benchmark_names_path,
        {"benchmark_names": sorted(str(name) for name in baseline_binary_microbenchmarks)},
    )
    _json_write(run_dir / "task.json", task_payload)
    _jsonl_write(run_dir / "property_edit_tasks.jsonl", [task_payload])
    _json_write(
        run_dir / "request.json",
        {
            "run_id": run_id,
            "binary_name": binary_name,
            "nearest_microbenchmark": nearest_microbenchmark,
            "target_features": canonical_target,
            "baseline_report": baseline_report,
            "baseline_constraint_report": baseline_constraint_report,
            "baseline_hw_accuracy_pct": baseline_hw_accuracy_pct,
            "baseline_feature_source": baseline_feature_source,
            "baseline_feature_fallback_reason": live_baseline_error,
            "table_baseline": table_baseline,
            "top_feature_gaps": top_gaps,
            "selected_property_task": task_payload,
            "task": task_text,
            "selection_metric": "hardware_counters_only",
            "seed_selection_mode": SEED_BASELINE_SELECTION_MODE,
            "delayed_from_scratch_mode": delayed_from_scratch_mode,
            "max_wall_seconds": max_wall_seconds,
        },
    )

    retrieval_name = f"{example_id}_retrieval.json"
    retrieve_cmd = [
        sys.executable,
        str(FOLLY_EDIT_RAG_SCRIPTS / "03_retrieve_edit_context.py"),
        "--task",
        task_text,
        "--target-file",
        target_info["target_file"],
        "--binary-name",
        binary_name,
        "--top-k",
        str(max(1, int(retrieval_top_k))),
        "--output",
        retrieval_name,
    ]
    retrieve_proc = _run_logged(retrieve_cmd, run_dir / "01_retrieve.json", cwd=FOLLY_EDIT_RAG_SCRIPTS)
    if retrieve_proc.returncode != 0:
        return {"success": False, "stage": "retrieve", "run_dir": str(run_dir)}

    attempts: list[dict[str, Any]] = []
    cumulative_token_usage: dict[str, int] = {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "call_count": 0,
    }
    last_compile_record: dict[str, Any] | None = None
    last_evaluation: dict[str, Any] | None = None
    best_evaluation: dict[str, Any] | None = None
    best_compile_record: dict[str, Any] | None = None
    best_selection_key: tuple[Any, ...] | None = None
    best_candidate_error_so_far: float | None = None
    last_material_improvement_attempt = 0
    last_per_feature_improvement_attempt = 0
    per_feature_best: dict[str, float] = {}
    global_last_progress_attempt = 0
    global_per_feature_best: dict[str, float] = {}
    early_stop_reason = ""
    retry_feedback = ""
    feature_guidance = "\n".join(task_payload.get("feature_specific_guidance", []))
    variant_index = 1
    variant_history: list[dict[str, Any]] = []
    variant_lessons_text = ""
    variant_start_attempt = 1  # first attempt_index of current variant
    variant_best_evaluation: dict[str, Any] | None = None
    variant_best_error: float | None = None
    attempt_index = 0
    working_state: dict[str, Any] = {
        "reference_microbenchmark": nearest_microbenchmark,
        "active_source_microbenchmark": nearest_microbenchmark,
        "current_file": str(target_info["current_file"]),
        "current_features": canonical_baseline,
        "current_rank_key": list(_normalized_rank_key(baseline_rank_key)),
        "working_state_rank_key": list(
            _working_state_rank_key(canonical_target, canonical_baseline, baseline_rank_key)
        ),
        "origin_label": "baseline reference benchmark",
        "objective_priority_rank": objective_priority_rank,
        "initial_generation_mode": str(task_payload.get("generation_strategy", "derived_from_nearest")),
        "origin_attempt_index": None,
        "carry_forward_reason": "",
    }
    local_eligible_candidate_seen = False
    if warm_start_file and warm_start_features and Path(warm_start_file).is_file():
        ws_canonical = canonicalize_feature_dict(warm_start_features)
        ws_report = _hardware_counter_match_report(canonical_target, ws_canonical, feature_scales)
        ws_error = float(ws_report["weighted_mean_normalized_abs_error"])
        if ws_error + 1e-6 < baseline_error:
            import shutil
            ws_dest = run_dir / "warm_start_initial.cpp"
            shutil.copy2(warm_start_file, ws_dest)
            # Also copy to the expected source location so the build system sees it.
            ws_target_file = target_info["current_file"]
            shutil.copy2(warm_start_file, ws_target_file)
            working_state["current_file"] = str(ws_target_file)
            working_state["current_features"] = ws_canonical
            ws_rank_key = candidate_rank_key(
                _hardware_counter_constraint_report(canonical_target, ws_canonical),
                ws_error,
            )
            working_state["current_rank_key"] = list(_normalized_rank_key(ws_rank_key))
            working_state["working_state_rank_key"] = list(
                _working_state_rank_key(canonical_target, ws_canonical, ws_rank_key)
            )
            working_state["origin_label"] = "cross-segment warm start"
            working_state["carry_forward_reason"] = "warm start from best previous segment result"
            baseline_error = ws_error
            canonical_baseline = ws_canonical
            baseline_report = ws_report
            baseline_constraint_report = _hardware_counter_constraint_report(canonical_target, ws_canonical)
            baseline_rank_key = ws_rank_key
            baseline_hw_accuracy_pct = _baseline_hw_accuracy_pct(canonical_target, ws_canonical)
            objective_priority_rank = _locked_objective_priority_rank(baseline_constraint_report)
            working_state["objective_priority_rank"] = objective_priority_rank
            print(
                f"[cloud-tuning] warm-start from previous segment: error={ws_error:.4f} "
                f"(seed baseline was {float(ws_report['weighted_mean_normalized_abs_error']):.4f}) "
                f"file={warm_start_file}",
                flush=True,
            )
    # Solved-tier snapshot: feature relative-error floors for tiers we have
    # already solved.  Prevents lower-tier exploration from undoing T0/T1 work.
    solved_tier_snapshot: dict[str, float] = {}
    # Track last promotion attempt to detect tier-level stagnation for
    # adaptive tolerance relaxation (Fix 5).
    last_promotion_attempt = 0
    baseline_hw_text = (
        f"{baseline_hw_accuracy_pct:.1f}%"
        if baseline_hw_accuracy_pct is not None and math.isfinite(baseline_hw_accuracy_pct)
        else "n/a"
    )
    print(
        f"[cloud-tuning] nearest search candidate for {binary_name}: "
        f"{nearest_microbenchmark} -> editing {target_info['target_file']} "
        f"(baseline_hw_avg={baseline_hw_text})",
        flush=True,
    )
    if from_scratch_generation:
        print(
            f"[cloud-tuning] baseline HW average {baseline_hw_text} is below "
            f"{FROM_SCRATCH_BASELINE_HW_THRESHOLD:.1f}%: switching this run to from-scratch benchmark generation.",
            flush=True,
        )
    elif delayed_from_scratch_mode:
        print(
            f"[cloud-tuning] baseline HW average {baseline_hw_text} is below "
            f"{FROM_SCRATCH_BASELINE_HW_THRESHOLD:.1f}%, but seed mode {SEED_BASELINE_SELECTION_MODE or '(unset)'} "
            f"keeps the run in local-edit mode for the first {IPC_SAFE_LOCAL_EDIT_ATTEMPTS} attempt(s).",
            flush=True,
        )

    while True:
        estimated_cost_usd = _estimated_llm_cost_usd(cumulative_token_usage)
        if (
            MAX_LLM_COST_USD_PER_SEGMENT > 0
            and estimated_cost_usd >= MAX_LLM_COST_USD_PER_SEGMENT
        ):
            early_stop_reason = (
                f"Estimated LLM cost ${estimated_cost_usd:.2f} reached the "
                f"${MAX_LLM_COST_USD_PER_SEGMENT:.2f} per-segment limit."
            )
            print(f"[cloud-tuning] early stop: {early_stop_reason}", flush=True)
            break
        attempt_index += 1
        variant_attempt_count = attempt_index - variant_start_attempt + 1
        global_stagnation_window = max(
            SOURCE_SWITCH_MIN_ATTEMPTS,
            STAGNANT_ATTEMPT_WINDOW * 2,
        )
        if (
            STOP_ON_STAGNATION
            and attempt_index > global_stagnation_window
            and attempt_index - global_last_progress_attempt > global_stagnation_window
        ):
            early_stop_reason = (
                "converged after global PMU stagnation: no aggregate or per-feature "
                f"improvement in {global_stagnation_window} consecutive attempts, "
                "including generation and compile failures"
            )
            print(f"[cloud-tuning] {early_stop_reason}", flush=True)
            break

        if variant_attempt_count > MAX_ATTEMPTS_PER_VARIANT:
            stop_reason = (
                f"Variant {variant_index} exhausted {MAX_ATTEMPTS_PER_VARIANT} attempts "
                "without meeting all constraints."
            )
            _variant_gen_strategy = str(
                working_state.get(
                    "initial_generation_mode",
                    task_payload.get("generation_strategy", "derived_from_nearest"),
                )
            )
            lesson = _build_variant_lesson(
                variant_index=variant_index,
                variant_attempts=[a for a in attempts if a.get("attempt_index", 0) >= variant_start_attempt],
                best_variant_evaluation=variant_best_evaluation,
                best_variant_error=variant_best_error,
                canonical_target=canonical_target,
                generation_strategy=_variant_gen_strategy,
                stop_reason=stop_reason,
            )
            variant_history.append(lesson)
            variant_lessons_text = "\n\n".join(v["lesson_text"] for v in variant_history[-3:])
            print(
                f"[cloud-tuning] variant {variant_index} exhausted after "
                f"{MAX_ATTEMPTS_PER_VARIANT} attempts. Switching to variant {variant_index + 1}.",
                flush=True,
            )
            variant_index += 1
            variant_start_attempt = attempt_index
            variant_best_evaluation = None
            variant_best_error = None
            last_material_improvement_attempt = attempt_index - 1
            last_per_feature_improvement_attempt = attempt_index - 1
            per_feature_best = {}
            local_eligible_candidate_seen = False
            retry_feedback = ""
            # Reset working state: use best-so-far or fall back to original baseline.
            if best_evaluation is not None:
                best_so_far_candidate = dict((best_evaluation.get("best_candidate") or {}).get("features") or {})
                best_so_far_name = str((best_evaluation.get("best_candidate") or {}).get("microbenchmark") or nearest_microbenchmark)
                best_so_far_file = str(best_compile_record.get("final_file", target_info["current_file"])) if best_compile_record else str(target_info["current_file"])
                working_state = {
                    "reference_microbenchmark": nearest_microbenchmark,
                    "active_source_microbenchmark": best_so_far_name,
                    "current_file": best_so_far_file,
                    "current_features": best_so_far_candidate or canonical_baseline,
                    "current_rank_key": list(_normalized_rank_key(
                        best_evaluation.get("best_candidate_rank_key") or baseline_rank_key,
                    )),
                    "working_state_rank_key": list(_working_state_rank_key(
                        canonical_target,
                        best_so_far_candidate or canonical_baseline,
                        best_evaluation.get("best_candidate_rank_key") or baseline_rank_key,
                    )),
                    "origin_label": f"best of variant {variant_index - 1}",
                    "objective_priority_rank": objective_priority_rank,
                    "initial_generation_mode": "from_scratch" if _variant_gen_strategy == "derived_from_nearest" else "derived_from_nearest",
                    "origin_attempt_index": None,
                    "carry_forward_reason": f"Variant switch from variant {variant_index - 1}",
                }
            else:
                working_state = {
                    "reference_microbenchmark": nearest_microbenchmark,
                    "active_source_microbenchmark": nearest_microbenchmark,
                    "current_file": str(target_info["current_file"]),
                    "current_features": canonical_baseline,
                    "current_rank_key": list(_normalized_rank_key(baseline_rank_key)),
                    "working_state_rank_key": list(
                        _working_state_rank_key(canonical_target, canonical_baseline, baseline_rank_key)
                    ),
                    "origin_label": f"baseline (variant {variant_index} fresh start)",
                    "objective_priority_rank": objective_priority_rank,
                    "initial_generation_mode": "from_scratch",
                    "origin_attempt_index": None,
                    "carry_forward_reason": f"Variant switch from variant {variant_index - 1}",
                }
            continue

        remaining_before_attempt = _remaining_wall_seconds(deadline_monotonic)
        if remaining_before_attempt is not None and remaining_before_attempt <= (
            REQUEST_BUDGET_SLACK_SECONDS + MIN_GENERATE_BUDGET_SECONDS
        ):
            early_stop_reason = _wall_budget_stop_reason(
                "starting another tuning attempt",
                remaining_seconds=remaining_before_attempt,
                reserve_seconds=REQUEST_BUDGET_SLACK_SECONDS,
            )
            print(f"[cloud-tuning] early stop before attempt {attempt_index}: {early_stop_reason}", flush=True)
            break
        if (
            delayed_from_scratch_mode
            and not from_scratch_generation
            and variant_attempt_count > IPC_SAFE_LOCAL_EDIT_ATTEMPTS
            and not local_eligible_candidate_seen
        ):
            from_scratch_generation = True
            delayed_from_scratch_mode = False
            current_top_gaps = _top_feature_gaps(
                _hardware_counter_features(canonical_target),
                _hardware_counter_features(dict(working_state["current_features"])),
                feature_scales,
            )
            current_primary_gap = current_top_gaps[0] if current_top_gaps else primary_gap
            task_payload = _build_property_task(
                binary_name=binary_name,
                source_microbenchmark=nearest_microbenchmark,
                target_info=_resolve_binary_target(binary_name, nearest_microbenchmark),
                feature_name=str(current_primary_gap["feature_name"]),
                target_value=float(current_primary_gap["target_value"]),
                baseline_value=float(current_primary_gap["baseline_value"]),
                delta_to_target=float(current_primary_gap["delta_to_target"]),
                feature_scales=feature_scales,
                top_feature_gaps=current_top_gaps or top_gaps,
                from_scratch=True,
                baseline_hw_accuracy_pct=baseline_hw_accuracy_pct,
                baseline_hw_threshold_pct=FROM_SCRATCH_BASELINE_HW_THRESHOLD,
            )
            target_info = {
                "source_root": Path(str(task_payload["source_root"])),
                "build_dir": Path(str(task_payload["build_dir"])),
                "target_file": str(task_payload["target_file"]),
                "current_file": Path(str(task_payload["source_root"])) / str(task_payload["target_file"]),
            }
            task_text = str(task_payload["task"])
            feature_guidance = "\n".join(task_payload.get("feature_specific_guidance", []))
            working_state["initial_generation_mode"] = "from_scratch"
            retry_feedback = ""
            retrieve_proc = _run_logged(
                [
                    sys.executable,
                    str(FOLLY_EDIT_RAG_SCRIPTS / "03_retrieve_edit_context.py"),
                    "--task",
                    task_text,
                    "--target-file",
                    target_info["target_file"],
                    "--binary-name",
                    binary_name,
                    "--top-k",
                    str(max(1, int(retrieval_top_k))),
                    "--output",
                    retrieval_name,
                ],
                run_dir / f"01_retrieve_from_scratch_attempt_{attempt_index}.json",
                cwd=FOLLY_EDIT_RAG_SCRIPTS,
            )
            print(
                f"[cloud-tuning] local-edit mode did not produce an IPC-safe active-tier-improving candidate in "
                f"{IPC_SAFE_LOCAL_EDIT_ATTEMPTS} attempt(s): explicitly switching this run to from-scratch generation.",
                flush=True,
            )
            if retrieve_proc.returncode != 0:
                print(
                    "[cloud-tuning] warning: from-scratch retrieval refresh failed; keeping the previous retrieval context.",
                    flush=True,
                )

        attempt_id = f"{example_id}__attempt_{attempt_index}"
        print(
            f"[cloud-tuning] attempt {attempt_index} (variant {variant_index}, "
            f"variant-attempt {variant_attempt_count}/{MAX_ATTEMPTS_PER_VARIANT}): "
            f"source_anchor={working_state['active_source_microbenchmark']}",
            flush=True,
        )
        attempt_working_payload = _build_working_state_payload(
            reference_microbenchmark=str(working_state["reference_microbenchmark"]),
            active_source_microbenchmark=str(working_state["active_source_microbenchmark"]),
            current_file=Path(str(working_state["current_file"])),
            current_features=dict(working_state["current_features"]),
            current_rank_key=working_state.get("current_rank_key"),
            canonical_target=canonical_target,
            feature_scales=feature_scales,
            origin_label=str(working_state["origin_label"]),
            objective_priority_rank=(
                int(working_state["objective_priority_rank"])
                if working_state.get("objective_priority_rank") is not None
                else None
            ),
            initial_generation_mode=str(
                working_state.get(
                    "initial_generation_mode",
                    task_payload.get("generation_strategy", "derived_from_nearest"),
                )
            ),
            origin_attempt_index=(
                int(working_state["origin_attempt_index"])
                if working_state.get("origin_attempt_index") is not None
                else None
            ),
            carry_forward_reason=str(working_state.get("carry_forward_reason", "")),
        )
        attempt_record: dict[str, Any] = {
            "attempt_index": attempt_index,
            "variant_index": variant_index,
            "variant_attempt_count": variant_attempt_count,
            "success": False,
            "working_state_before": attempt_working_payload,
        }
        retry_feedback_path = run_dir / f"retry_feedback_attempt_{attempt_index}.txt"
        # Combine variant lessons with any existing retry feedback.
        full_retry_feedback = retry_feedback
        if variant_lessons_text:
            full_retry_feedback = (
                (variant_lessons_text + "\n\n" + retry_feedback) if retry_feedback
                else variant_lessons_text
            )
        if full_retry_feedback:
            retry_feedback_path.write_text(full_retry_feedback, encoding="utf-8")
        working_state_path = run_dir / f"working_state_attempt_{attempt_index}.txt"
        working_state_path.write_text(
            str(attempt_working_payload.get("summary_text", "")).strip() + "\n",
            encoding="utf-8",
        )
        _json_write(run_dir / f"working_state_attempt_{attempt_index}.json", attempt_working_payload)
        diagnosis_payload = build_diagnostician_report(
            target_features=canonical_target,
            current_features=dict(working_state["current_features"]),
            binary_name=binary_name,
            target_file_name=target_info["target_file"],
            generation_strategy=str(
                working_state.get(
                    "initial_generation_mode",
                    task_payload.get("generation_strategy", "derived_from_nearest"),
                )
            ),
            objective_priority_rank=(
                int(working_state["objective_priority_rank"])
                if working_state.get("objective_priority_rank") is not None
                else None
            ),
        )
        diagnosis_path = run_dir / f"diagnostician_attempt_{attempt_index}.json"
        diagnosis_text_path = run_dir / f"diagnostician_attempt_{attempt_index}.txt"
        _json_write(diagnosis_path, diagnosis_payload)
        diagnosis_text_path.write_text(
            str(diagnosis_payload.get("summary_text", "")).strip() + "\n",
            encoding="utf-8",
        )
        attempt_record["diagnosis"] = diagnosis_payload
        feature_guidance_lines = list(diagnosis_payload.get("generator_guidance") or [])
        feature_guidance = (
            "\n".join(str(line).strip() for line in feature_guidance_lines if str(line).strip())
            or "\n".join(task_payload.get("feature_specific_guidance", []))
        )

        remaining_before_generate = _remaining_wall_seconds(deadline_monotonic)
        if remaining_before_generate is not None and remaining_before_generate <= (
            REQUEST_BUDGET_SLACK_SECONDS + MIN_GENERATE_BUDGET_SECONDS
        ):
            early_stop_reason = _wall_budget_stop_reason(
                "patch generation",
                remaining_seconds=remaining_before_generate,
                reserve_seconds=REQUEST_BUDGET_SLACK_SECONDS,
            )
            attempt_record["stage"] = "deadline"
            attempt_record["reason"] = early_stop_reason
            attempts.append(attempt_record)
            print(f"[cloud-tuning] early stop after attempt {attempt_index} preflight: {early_stop_reason}", flush=True)
            break

        generated_name = f"{attempt_id}_candidate.patch"
        attempt_anchor_source_text = Path(str(working_state["current_file"])).read_text(
            encoding="utf-8",
            errors="ignore",
        )
        generate_cmd_base = [
            sys.executable,
            str(FOLLY_EDIT_RAG_SCRIPTS / "04_generate_patch.py"),
            "--task",
            task_text,
            "--target-file",
            target_info["target_file"],
            "--binary-name",
            binary_name,
            "--reference-microbenchmark",
            nearest_microbenchmark,
            "--source-microbenchmark",
            str(working_state["active_source_microbenchmark"]),
            "--current-file",
            str(working_state["current_file"]),
            "--retrieval-json",
            retrieval_name,
            "--output",
            generated_name,
            "--attempt-index",
            str(attempt_index),
            "--max-attempts",
            str(MAX_ATTEMPTS_PER_VARIANT),
            "--feature-guidance",
            feature_guidance,
            "--diagnosis-file",
            str(diagnosis_text_path),
            "--working-state-file",
            str(working_state_path),
            "--initial-generation-mode",
            str(task_payload.get("generation_strategy", "derived_from_nearest")),
            "--forbidden-benchmark-names-file",
            str(forbidden_benchmark_names_path),
        ]
        if bool(task_payload.get("application_seed")):
            generate_cmd_base.append("--application-seed")

        compile_cmd_base = [
            sys.executable,
            str(FOLLY_EDIT_RAG_SCRIPTS / "05_apply_compile_repair.py"),
            "--task",
            task_text,
            "--target-file",
            target_info["target_file"],
            "--binary-name",
            binary_name,
            "--current-file",
            str(working_state["current_file"]),
            "--build-dir",
            str(target_info["build_dir"]),
            "--source-root",
            str(target_info["source_root"]),
            "--reference-microbenchmark",
            nearest_microbenchmark,
            "--source-microbenchmark",
            str(working_state["active_source_microbenchmark"]),
            "--retrieval-json",
            retrieval_name,
            "--patch-file",
            generated_name,
            "--example-id",
            attempt_id,
            "--max-compile-repair-attempts",
            str(MAX_COMPILE_REPAIR_ATTEMPTS),
            "--no-save-success-example",
        ]
        if bool(task_payload.get("application_seed")):
            compile_cmd_base.append("--application-seed")

        _two_phase_env: dict[str, str] | None = None
        _tp_overrides = _two_phase_model_env(
            int(working_state["objective_priority_rank"])
            if working_state.get("objective_priority_rank") is not None
            else None
        )
        if _tp_overrides:
            _two_phase_env = dict(os.environ)
            _two_phase_env.update(_tp_overrides)
            print(
                f"[cloud-tuning] two-phase model: using {_tp_overrides.get('VERTEX_MODEL_NAME', '?')} "
                f"for tier objective {working_state.get('objective_priority_rank')}",
                flush=True,
            )

        if retry_feedback or variant_lessons_text:
            generate_cmd_base.extend(["--retry-feedback-file", str(retry_feedback_path)])
        _gen_ok = False
        for _gen_retry_idx in range(1 + GENERATION_QUICK_RETRIES):
            generate_proc = _run_logged(
                generate_cmd_base,
                run_dir / f"02_generate_attempt_{attempt_index}.json",
                cwd=FOLLY_EDIT_RAG_SCRIPTS,
                env=_two_phase_env,
            )
            attempt_record["generate"] = {"returncode": generate_proc.returncode}
            gen_token_path = _PATCHES_DIR / f"{generated_name}.token_usage.json"
            gen_token_usage = _read_token_usage_sidecar(gen_token_path)
            attempt_record["generate"]["token_usage"] = gen_token_usage
            _accumulate_token_usage(cumulative_token_usage, gen_token_usage)
            if generate_proc.returncode == 0:
                _gen_ok = True
                break
            if _gen_retry_idx < GENERATION_QUICK_RETRIES:
                print(
                    f"[cloud-tuning] attempt {attempt_index} generation failed "
                    f"(quick-retry {_gen_retry_idx + 1}/{GENERATION_QUICK_RETRIES})",
                    flush=True,
                )
        if not _gen_ok:
            attempt_record["stage"] = "generate"
            attempt_record["reason"] = "Generation step failed."
            attempts.append(attempt_record)
            retry_feedback = (
                f"Attempt {attempt_index} failed before patch generation completed. Produce a simpler, deterministic local patch."
            )
            continue

        compile_budget_s = _available_step_budget_seconds(
            deadline_monotonic,
            reserve_seconds=REQUEST_BUDGET_SLACK_SECONDS + MIN_EVALUATE_BUDGET_SECONDS,
        )
        if compile_budget_s is not None and compile_budget_s < MIN_COMPILE_BUDGET_SECONDS:
            early_stop_reason = _wall_budget_stop_reason(
                "compile/repair",
                remaining_seconds=_remaining_wall_seconds(deadline_monotonic),
                reserve_seconds=REQUEST_BUDGET_SLACK_SECONDS + MIN_EVALUATE_BUDGET_SECONDS,
            )
            attempt_record["stage"] = "deadline"
            attempt_record["reason"] = early_stop_reason
            attempts.append(attempt_record)
            print(f"[cloud-tuning] early stop after attempt {attempt_index}: {early_stop_reason}", flush=True)
            break

        if compile_budget_s is not None:
            compile_cmd_base.extend(["--wall-clock-budget-s", f"{compile_budget_s:.1f}"])
        compile_proc = _run_logged(
            compile_cmd_base,
            run_dir / f"03_compile_attempt_{attempt_index}.json",
            cwd=FOLLY_EDIT_RAG_SCRIPTS,
            env=_two_phase_env,
        )
        attempt_record["compile"] = {"returncode": compile_proc.returncode}
        compile_token_path = _COMPILE_LOGS_DIR / f"{attempt_id}.token_usage.json"
        compile_token_usage = _read_token_usage_sidecar(compile_token_path)
        attempt_record["compile"]["token_usage"] = compile_token_usage
        _accumulate_token_usage(cumulative_token_usage, compile_token_usage)
        compile_log = FOLLY_EDIT_RAG_SCRIPTS / "outputs" / "compile_logs" / f"{attempt_id}.json"
        try:
            compile_record = json.loads(compile_log.read_text(encoding="utf-8")) if compile_log.exists() else {}
        except (json.JSONDecodeError, OSError):
            compile_record = {}
        last_compile_record = compile_record
        attempt_record["compile_record"] = compile_record
        if not compile_record.get("success"):
            attempt_record["stage"] = "compile"
            attempt_record["reason"] = "Compile/repair step did not produce a successful build."
            failure_headline = _first_nonempty_line(
                "\n".join(
                    part
                    for part in (
                        str(compile_record.get("second_patch_apply_summary", "")),
                        str(compile_record.get("first_patch_apply_summary", "")),
                        str(compile_record.get("second_compile_summary", "")),
                        str(compile_record.get("first_compile_summary", "")),
                    )
                    if part.strip()
                )
            )
            if failure_headline:
                print(
                    f"[cloud-tuning] attempt {attempt_index} failed while adjusting "
                    f"{working_state['active_source_microbenchmark']}: {failure_headline}",
                    flush=True,
                )
            attempts.append(attempt_record)
            retry_feedback = _build_attempt_retry_feedback(
                task_payload,
                attempt_index,
                objective_priority_rank=(
                    int(working_state["objective_priority_rank"])
                    if working_state.get("objective_priority_rank") is not None
                    else None
                ),
                active_source_microbenchmark=str(working_state["active_source_microbenchmark"]),
                compile_record=compile_record,
                canonical_target=canonical_target,
            )
            continue

        remaining_before_evaluation = _remaining_wall_seconds(deadline_monotonic)
        if remaining_before_evaluation is not None and remaining_before_evaluation <= (
            REQUEST_BUDGET_SLACK_SECONDS + MIN_EVALUATE_BUDGET_SECONDS
        ):
            early_stop_reason = _wall_budget_stop_reason(
                "candidate evaluation",
                remaining_seconds=remaining_before_evaluation,
                reserve_seconds=REQUEST_BUDGET_SLACK_SECONDS,
            )
            attempt_record["stage"] = "deadline"
            attempt_record["reason"] = early_stop_reason
            attempts.append(attempt_record)
            print(f"[cloud-tuning] early stop after attempt {attempt_index} compile: {early_stop_reason}", flush=True)
            break

        evaluation = _evaluate_adjusted_binary(
            run_dir=run_dir,
            attempt_index=attempt_index,
            binary_path=binary_path,
            binary_name=binary_name,
            source_microbenchmark=str(working_state["active_source_microbenchmark"]),
            baseline_binary_microbenchmarks=baseline_binary_microbenchmarks,
            canonical_target=canonical_target,
            feature_scales=feature_scales,
            anchor_features=dict(working_state["current_features"]),
            anchor_constraint_report=_hardware_counter_constraint_report(
                canonical_target,
                dict(working_state["current_features"]),
            ),
            objective_priority_rank=(
                int(working_state["objective_priority_rank"])
                if working_state.get("objective_priority_rank") is not None
                else None
            ),
            perf_events=perf_events,
            amd=amd,
            deadline_monotonic=deadline_monotonic,
            compile_record=compile_record,
            anchor_source_text=attempt_anchor_source_text,
            ipc_coupled=ipc_coupled,
            tier_stagnation_attempts=max(0, attempt_index - last_promotion_attempt),
        )
        last_evaluation = evaluation
        attempt_record["evaluation"] = evaluation
        if evaluation.get("budget_exhausted"):
            early_stop_reason = str(evaluation.get("reason") or "").strip() or _wall_budget_stop_reason(
                "candidate evaluation",
                remaining_seconds=_remaining_wall_seconds(deadline_monotonic),
            )
            attempt_record["stage"] = "deadline"
            attempt_record["reason"] = early_stop_reason
            attempts.append(attempt_record)
            print(f"[cloud-tuning] early stop after attempt {attempt_index}: {early_stop_reason}", flush=True)
            break

        candidate = dict((evaluation.get("best_candidate") or {}).get("features") or {})
        candidate_error = evaluation.get("candidate_error")
        candidate_name = str((evaluation.get("best_candidate") or {}).get("microbenchmark") or "(unknown)")
        if candidate:
            _print_live_accuracy(
                f"[cloud-tuning] attempt {attempt_index} candidate {candidate_name}:",
                canonical_target,
                candidate,
                float(candidate_error) if candidate_error is not None else None,
            )
        print(
            f"[cloud-tuning] attempt {attempt_index} adjusted "
            f"{working_state['active_source_microbenchmark']} -> "
            f"{candidate_name if candidate_name else '(no new microbenchmark)'}: "
            f"{evaluation.get('reason', 'evaluated successfully')}",
            flush=True,
        )

        previous_best_error = best_candidate_error_so_far
        current_candidate_error = _selection_error(evaluation)
        if ipc_coupled:
            current_selection_key = _coupled_evaluation_selection_key(evaluation, canonical_target)
        else:
            current_selection_key = _evaluation_selection_key(evaluation)
        if current_candidate_error < float("inf"):
            if best_selection_key is None or current_selection_key < best_selection_key:
                best_selection_key = current_selection_key
                best_evaluation = evaluation
                best_compile_record = compile_record
                global_last_progress_attempt = attempt_index
            if best_candidate_error_so_far is None or current_candidate_error + 1e-12 < best_candidate_error_so_far:
                best_candidate_error_so_far = current_candidate_error
                if _material_accuracy_improvement(previous_best_error, current_candidate_error):
                    last_material_improvement_attempt = attempt_index
            # Per-variant best tracking.
            if variant_best_error is None or current_candidate_error + 1e-12 < variant_best_error:
                variant_best_error = current_candidate_error
                variant_best_evaluation = evaluation
        # Track per-feature Pareto progress.
        if candidate:
            if _update_per_feature_best(per_feature_best, candidate, canonical_target, ipc_coupled=ipc_coupled):
                last_per_feature_improvement_attempt = attempt_index
            if _update_per_feature_best(
                global_per_feature_best,
                candidate,
                canonical_target,
                ipc_coupled=ipc_coupled,
            ):
                global_last_progress_attempt = attempt_index
        if int(evaluation.get("eligible_candidate_count", 0) or 0) > 0:
            local_eligible_candidate_seen = True
        best_candidate = dict((best_evaluation or {}).get("best_candidate") or {})
        best_candidate_features = dict(best_candidate.get("features") or {})
        best_candidate_error = (best_evaluation or {}).get("candidate_error")
        if best_candidate_features:
            _print_live_accuracy(
                f"[cloud-tuning] best after attempt {attempt_index} {best_candidate.get('microbenchmark', '(unknown)')}:",
                canonical_target,
                best_candidate_features,
                float(best_candidate_error) if best_candidate_error is not None else None,
            )

        promoted_state, carry_forward_reason = _promote_working_state(
            current_state=working_state,
            evaluation=evaluation,
            compile_record=compile_record,
            canonical_target=canonical_target,
        )
        if promoted_state is not None and ipc_coupled:
            candidate_feats = dict((evaluation.get("best_candidate") or {}).get("features") or {})
            ws_feats = dict(working_state.get("current_features") or {})
            def _coupled_key(feats: dict, eval_or_none: dict | None = None) -> tuple[float, ...]:
                """Promotion gate key: weighted error primary, coupled gaps + IPC as tiebreakers."""
                # Use the actual weighted error so overall accuracy drives promotion.
                if eval_or_none is not None:
                    we = _selection_error(eval_or_none)
                else:
                    we_report = _hardware_counter_match_report(canonical_target, feats, feature_scales)
                    we = float(we_report.get("weighted_mean_normalized_abs_error", float("inf")))
                ipc_g = _feature_relative_gap(canonical_target, feats, "perf_stat.IPC")
                ipc_p = abs(float(ipc_g)) if ipc_g is not None and math.isfinite(float(ipc_g)) else float("inf")
                gaps = []
                for f in ("perf_stat.L1-dcache-load-misses_MPKI", "perf_stat.LLC-load-misses_MPKI",
                          "perf_stat.branch-misses_MPKI", "perf_stat.branch-load-misses_MPKI"):
                    g = _feature_relative_gap(canonical_target, feats, f)
                    if g is not None and math.isfinite(float(g)):
                        gaps.append(abs(float(g)))
                mean_g = sum(gaps) / len(gaps) if gaps else float("inf")
                return (we, mean_g, ipc_p)
            if _coupled_key(candidate_feats, evaluation) >= _coupled_key(ws_feats):
                promoted_state = None
                carry_forward_reason = ""
        if promoted_state is not None:
            # Check that the candidate doesn't regress features from tiers
            # we have already solved (prevents tier oscillation).
            candidate_feats_for_check = dict((evaluation.get("best_candidate") or {}).get("features") or {})
            regression_reason = _solved_tier_regression_check(
                solved_tier_snapshot, candidate_feats_for_check, canonical_target,
            )
            if regression_reason:
                print(
                    f"[cloud-tuning] attempt {attempt_index} promotion blocked: {regression_reason}",
                    flush=True,
                )
                promoted_state = None
                carry_forward_reason = ""
        if promoted_state is not None:
            # Detect tier advancement and snapshot the just-solved tier.
            old_tier = working_state.get("objective_priority_rank")
            new_tier = promoted_state.get("objective_priority_rank")
            if (
                old_tier is not None
                and new_tier is not None
                and int(new_tier) > int(old_tier)
            ):
                _update_solved_tier_snapshot(
                    solved_tier_snapshot,
                    dict(promoted_state.get("current_features") or {}),
                    canonical_target,
                    int(old_tier),
                )
                print(
                    f"[cloud-tuning] tier {priority_tier_label(int(old_tier))} solved; "
                    f"snapshotted {len(solved_tier_snapshot)} feature guardrails, advancing to "
                    f"{priority_tier_label(int(new_tier))}",
                    flush=True,
                )
            working_state = promoted_state
            last_promotion_attempt = attempt_index
            attempt_record["working_state_promoted"] = True
            attempt_record["working_state_promotion_reason"] = carry_forward_reason
        else:
            attempt_record["working_state_promoted"] = False

        attempts_since_promotion = attempt_index - last_promotion_attempt
        current_obj_rank = working_state.get("objective_priority_rank")
        if (
            TIER_ESCALATION_WINDOW > 0
            and attempts_since_promotion >= TIER_ESCALATION_WINDOW
            and current_obj_rank is not None
        ):
            try:
                from scripts.cloning_priority import PRIORITY_ORDER
                idx = PRIORITY_ORDER.index(int(current_obj_rank))
                if idx + 1 < len(PRIORITY_ORDER):
                    next_rank = PRIORITY_ORDER[idx + 1]
                    print(
                        f"[cloud-tuning] tier escalation: {priority_tier_label(int(current_obj_rank))} "
                        f"stagnant for {attempts_since_promotion} attempts, advancing to "
                        f"{priority_tier_label(next_rank)}",
                        flush=True,
                    )
                    _update_solved_tier_snapshot(
                        solved_tier_snapshot,
                        dict(working_state.get("current_features") or {}),
                        canonical_target,
                        int(current_obj_rank),
                    )
                    for _pad_fname in list(solved_tier_snapshot):
                        solved_tier_snapshot[_pad_fname] = min(
                            solved_tier_snapshot[_pad_fname] + 0.10, 1.0
                        )
                    working_state["objective_priority_rank"] = next_rank
                    last_promotion_attempt = attempt_index
            except (ValueError, IndexError):
                pass

        attempt_record["working_state_after"] = _build_working_state_payload(
            reference_microbenchmark=str(working_state["reference_microbenchmark"]),
            active_source_microbenchmark=str(working_state["active_source_microbenchmark"]),
            current_file=Path(str(working_state["current_file"])),
            current_features=dict(working_state["current_features"]),
            current_rank_key=working_state.get("current_rank_key"),
            canonical_target=canonical_target,
            feature_scales=feature_scales,
            origin_label=str(working_state["origin_label"]),
            objective_priority_rank=(
                int(working_state["objective_priority_rank"])
                if working_state.get("objective_priority_rank") is not None
                else None
            ),
            initial_generation_mode=str(
                working_state.get(
                    "initial_generation_mode",
                    task_payload.get("generation_strategy", "derived_from_nearest"),
                )
            ),
            origin_attempt_index=(
                int(working_state["origin_attempt_index"])
                if working_state.get("origin_attempt_index") is not None
                else None
            ),
            carry_forward_reason=str(working_state.get("carry_forward_reason", "")),
        )

        if evaluation.get("success"):
            attempt_record["success"] = True
            attempt_record["stage"] = "evaluate"
            attempts.append(attempt_record)
            break

        attempt_record["stage"] = "evaluate"
        attempt_record["reason"] = (
            evaluation.get("reason") or "Adjusted candidate missed the hardware-counter target profile."
        )
        attempts.append(attempt_record)
        stagnation_reason = _stagnation_stop_reason(
            attempt_index=variant_attempt_count,
            baseline_error=baseline_error,
            best_candidate_error=variant_best_error,
            last_material_improvement_attempt=last_material_improvement_attempt - variant_start_attempt + 1,
            last_per_feature_improvement_attempt=last_per_feature_improvement_attempt - variant_start_attempt + 1,
            baseline_constraint_report=baseline_constraint_report,
            ipc_coupled=ipc_coupled,
        )
        if stagnation_reason:
            true_convergence = (
                "No material accuracy improvement" in stagnation_reason
                or "no longer improving materially" in stagnation_reason
            )
            if STOP_ON_STAGNATION and true_convergence:
                early_stop_reason = f"converged after PMU stagnation: {stagnation_reason}"
                print(f"[cloud-tuning] {early_stop_reason}", flush=True)
                break
            _variant_gen_strategy = str(
                working_state.get(
                    "initial_generation_mode",
                    task_payload.get("generation_strategy", "derived_from_nearest"),
                )
            )
            lesson = _build_variant_lesson(
                variant_index=variant_index,
                variant_attempts=[a for a in attempts if a.get("variant_index") == variant_index],
                best_variant_evaluation=variant_best_evaluation,
                best_variant_error=variant_best_error,
                canonical_target=canonical_target,
                generation_strategy=_variant_gen_strategy,
                stop_reason=stagnation_reason,
            )
            variant_history.append(lesson)
            variant_lessons_text = "\n\n".join(v["lesson_text"] for v in variant_history[-3:])
            print(
                f"[cloud-tuning] variant {variant_index} stagnated after "
                f"{variant_attempt_count} attempts. Switching to variant {variant_index + 1}.",
                flush=True,
            )
            variant_index += 1
            variant_start_attempt = attempt_index + 1
            variant_best_evaluation = None
            variant_best_error = None
            last_material_improvement_attempt = attempt_index
            last_per_feature_improvement_attempt = attempt_index
            last_promotion_attempt = attempt_index
            per_feature_best = {}
            local_eligible_candidate_seen = False
            retry_feedback = ""
            # Reset working state: use global best or fall back to original baseline.
            if best_evaluation is not None:
                best_so_far_candidate = dict((best_evaluation.get("best_candidate") or {}).get("features") or {})
                best_so_far_name = str((best_evaluation.get("best_candidate") or {}).get("microbenchmark") or nearest_microbenchmark)
                best_so_far_file = str(best_compile_record.get("final_file", target_info["current_file"])) if best_compile_record else str(target_info["current_file"])
                working_state = {
                    "reference_microbenchmark": nearest_microbenchmark,
                    "active_source_microbenchmark": best_so_far_name,
                    "current_file": best_so_far_file,
                    "current_features": best_so_far_candidate or canonical_baseline,
                    "current_rank_key": list(_normalized_rank_key(
                        best_evaluation.get("best_candidate_rank_key") or baseline_rank_key,
                    )),
                    "working_state_rank_key": list(_working_state_rank_key(
                        canonical_target,
                        best_so_far_candidate or canonical_baseline,
                        best_evaluation.get("best_candidate_rank_key") or baseline_rank_key,
                    )),
                    "origin_label": f"best of variant {variant_index - 1}",
                    "objective_priority_rank": objective_priority_rank,
                    "initial_generation_mode": "from_scratch" if _variant_gen_strategy == "derived_from_nearest" else "derived_from_nearest",
                    "origin_attempt_index": None,
                    "carry_forward_reason": f"Stagnation-triggered variant switch from variant {variant_index - 1}",
                }
            else:
                working_state = {
                    "reference_microbenchmark": nearest_microbenchmark,
                    "active_source_microbenchmark": nearest_microbenchmark,
                    "current_file": str(target_info["current_file"]),
                    "current_features": canonical_baseline,
                    "current_rank_key": list(_normalized_rank_key(baseline_rank_key)),
                    "working_state_rank_key": list(
                        _working_state_rank_key(canonical_target, canonical_baseline, baseline_rank_key)
                    ),
                    "origin_label": f"baseline (variant {variant_index} fresh start)",
                    "objective_priority_rank": objective_priority_rank,
                    "initial_generation_mode": "from_scratch",
                    "origin_attempt_index": None,
                    "carry_forward_reason": f"Stagnation-triggered variant switch from variant {variant_index - 1}",
                }
            continue
        retry_feedback = _build_attempt_retry_feedback(
            task_payload,
            attempt_index,
            objective_priority_rank=(
                int(working_state["objective_priority_rank"])
                if working_state.get("objective_priority_rank") is not None
                else None
            ),
            active_source_microbenchmark=str(working_state["active_source_microbenchmark"]),
            compile_record=compile_record,
            evaluation=evaluation,
            canonical_target=canonical_target,
        )
        ipc_hint = evaluation.get("ipc_rejected_tier_hint") if isinstance(evaluation, dict) else None
        if ipc_hint is not None:
            hint_name = str(ipc_hint.get("microbenchmark", "")).strip()
            hint_feats = ipc_hint.get("features") or {}
            hint_ipc_report = ipc_hint.get("ipc_guardrail_report") or {}
            hint_ipc_reason = str(hint_ipc_report.get("reason", "")).strip()
            hint_lines: list[str] = []
            for fname in ("perf_stat.IPC", "perf_stat.L1-icache-load-misses_MPKI",
                          "perf_stat.branch-misses_MPKI", "perf_stat.LLC-load-misses_MPKI"):
                mv = hint_feats.get(fname)
                tv = (canonical_target or {}).get(fname)
                if mv is not None:
                    hint_lines.append(f"  {fname}: measured={mv}" + (f", target={tv}" if tv is not None else ""))
            retry_feedback += (
                f"\n\nIPC-REJECTED TIER HINT: Candidate {hint_name} improved the active tier but "
                f"was rejected by the IPC guardrail ({hint_ipc_reason}). "
                "Reuse the tier-improving code change from that candidate but add the smallest "
                "possible IPC-corrective edit (e.g. reduce inner loop count, lower contention "
                "level, or remove off-critical-path work that inflates IPC).\n"
                + "\n".join(hint_lines)
            )

    selected_evaluation = best_evaluation or {}
    selected_compile_record = best_compile_record or last_compile_record or {}
    candidate_error = (
        float(selected_evaluation["candidate_error"])
        if "candidate_error" in selected_evaluation
        else None
    )
    success = bool(selected_evaluation.get("success"))
    diagnostician_attempt = _select_diagnostician_attempt(attempts, selected_evaluation)
    diagnostician_report = (
        dict(diagnostician_attempt.get("diagnosis") or {})
        if isinstance(diagnostician_attempt, dict)
        else None
    )
    diagnostician_attempt_index = (
        diagnostician_attempt.get("attempt_index")
        if isinstance(diagnostician_attempt, dict)
        else None
    )

    result = {
        "success": success,
        "run_dir": str(run_dir),
        "task": task_text,
        "selected_property_task": task_payload,
        "target_file": target_info["target_file"],
        "current_file": str(target_info["current_file"]),
        "build_dir": str(target_info["build_dir"]),
        "binary_path": str(binary_path),
        "binary_name": binary_name,
        "nearest_microbenchmark": nearest_microbenchmark,
        "baseline_report": baseline_report,
        "baseline_constraint_report": baseline_constraint_report,
        "baseline_rank_key": list(baseline_rank_key),
        "baseline_error": baseline_error,
        "baseline_hw_accuracy_pct": baseline_hw_accuracy_pct,
        "selection_metric": "hardware_counters_only",
        "generation_strategy": task_payload.get("generation_strategy"),
        "candidate_error": candidate_error,
        "best_candidate": selected_evaluation.get("best_candidate"),
        "chosen_runtime_estimate": selected_evaluation.get("chosen_runtime_estimate"),
        "best_candidate_report": selected_evaluation.get("best_candidate_report"),
        "best_candidate_constraint_report": selected_evaluation.get("best_candidate_constraint_report"),
        "best_candidate_rank_key": selected_evaluation.get("best_candidate_rank_key"),
        "diagnostician_report": diagnostician_report,
        "diagnostician_attempt_index": diagnostician_attempt_index,
        "compile_record": selected_compile_record,
        "attempts": attempts,
        "final_working_state": working_state,
        "max_attempts_per_variant": MAX_ATTEMPTS_PER_VARIANT,
        "max_adjustment_versions": MAX_ADJUSTMENT_VERSIONS,
        "variant_count": variant_index,
        "variant_history": variant_history,
        "total_attempts": attempt_index,
        "max_wall_seconds": max_wall_seconds,
        "elapsed_wall_seconds": time.monotonic() - started_at,
        "early_stop_reason": early_stop_reason or None,
        "token_usage": cumulative_token_usage,
        "estimated_llm_cost_usd": _estimated_llm_cost_usd(cumulative_token_usage),
        "max_llm_cost_usd_per_segment": MAX_LLM_COST_USD_PER_SEGMENT or None,
    }
    if not success:
        final_reason_source = selected_evaluation or last_evaluation or {}
        result["stage"] = (
            str((attempts[-1] or {}).get("stage")) if attempts else ""
        ) or ("evaluate" if (selected_evaluation or last_evaluation) else "compile")
        result["reason"] = (
            early_stop_reason
            or str(final_reason_source.get("reason", "")).strip()
            or "No adjusted version met the hardware-counter target profile threshold "
            f"after {attempt_index} attempts across {variant_index} variant(s)."
        )
    _json_write(run_dir / "result.json", result)
    return result
