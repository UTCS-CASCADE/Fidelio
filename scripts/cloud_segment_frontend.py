#!/usr/bin/env python3
'Frontend driver for cloud multi-segment feature JSONs.'

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, Any

import requests
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.feature_schema import (
    CANONICAL_FEATURE_COLUMNS,
    absolute_match_report,
    canonicalize_feature_dict,
    dataframe_feature_scales,
    load_feature_table,
    resolve_feature_table_path,
)
from scripts.run_feature_extraction_end_to_end import measure_unit_runtime


SEGMENT_KEY_RE = re.compile(r"^segment_(\d+)$")
DEFAULT_XLSX = resolve_feature_table_path(REPO_ROOT / "search_engine" / "all_features.xlsx")
AXIS_FEATURES = {
    "overall": tuple(CANONICAL_FEATURE_COLUMNS),
    "hw_counters": tuple(name for name in CANONICAL_FEATURE_COLUMNS if name.startswith("perf_stat.")),
    "tma_metrics": tuple(name for name in CANONICAL_FEATURE_COLUMNS if name.startswith("tma.")),
    "sw_metrics": tuple(name for name in CANONICAL_FEATURE_COLUMNS if name.startswith("intel_pt.")),
    "ipc": ("perf_stat.IPC",),
    "branch_cache_tlb_mpki": tuple(
        name for name in CANONICAL_FEATURE_COLUMNS if name.startswith("perf_stat.") and name.endswith("_MPKI")
    ),
}
STOP_POLICY_AXIS = "hw_counters"
DEFAULT_SEARCH_TIMEOUT_SECONDS = 10800.0

AXIS_IMPORTANCE_WEIGHTS: Dict[str, float] = {
    "ipc":                   0.30,
    "tma_metrics":           0.25,
    "hw_counters":           0.25,
    "sw_metrics":            0.10,
    "branch_cache_tlb_mpki": 0.10,
}


# Vertex Gemini rates used by the tuning pipeline. Output includes billed
# reasoning tokens.
_VERTEX_GEMINI_PRICING = {
    "input_rate": 2.00,
    "output_rate": 12.00,
}

# Claude Sonnet 4.6 (Bedrock)
# Ref: https://aws.amazon.com/bedrock/pricing/
_CLAUDE_SONNET_PRICING = {
    "input_rate":  3.00,
    "output_rate": 15.00,
}

# Copilot proxy (cost is $0 for the user — included in subscription)
_COPILOT_PRICING = {
    "input_rate":  0.00,
    "output_rate": 0.00,
}

# Map provider tags to pricing dicts
_PRICING_TABLE: Dict[str, dict] = {
    "vertex":  _VERTEX_GEMINI_PRICING,
    "bedrock": _CLAUDE_SONNET_PRICING,
    "copilot": _COPILOT_PRICING,
}


def _detect_llm_provider() -> str:
    """Best-effort detection of the active LLM provider from env vars."""
    return os.environ.get("LLM_PROVIDER", "vertex").strip().lower()


def estimate_llm_cost(
    input_tokens: int,
    output_tokens: int,
    provider: str | None = None,
) -> float:
    'Estimate USD cost for a given token count.'
    if provider is None:
        provider = _detect_llm_provider()
    pricing = _PRICING_TABLE.get(provider, _VERTEX_GEMINI_PRICING)

    if provider == "vertex":
        in_rate = float(
            os.environ.get(
                "VERTEX_INPUT_USD_PER_MILLION_TOKENS",
                str(pricing["input_rate"]),
            )
        )
        out_rate = float(
            os.environ.get(
                "VERTEX_OUTPUT_USD_PER_MILLION_TOKENS",
                str(pricing["output_rate"]),
            )
        )
    else:
        in_rate = pricing["input_rate"]
        out_rate = pricing["output_rate"]

    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Cloud feature JSON must be a top-level object")
    return data


def numeric_features(d: Dict[str, Any]) -> Dict[str, float]:
    return canonicalize_feature_dict(d)


def average_feature_dict(feature_rows: list[Dict[str, Any]]) -> Dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in feature_rows:
        canonical = canonicalize_feature_dict(row)
        for name, value in canonical.items():
            sums[name] = sums.get(name, 0.0) + float(value)
            counts[name] = counts.get(name, 0) + 1
    return {name: sums[name] / counts[name] for name in sums.keys() if counts.get(name, 0) > 0}


def _features_from_report(report: dict | None, value_key: str) -> Dict[str, float]:
    if not isinstance(report, dict):
        return {}
    rows = report.get("feature_errors")
    if not isinstance(rows, list):
        return {}
    out: Dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        feature_name = row.get("feature_name")
        value = row.get(value_key)
        if feature_name is None or value is None:
            continue
        try:
            out[str(feature_name)] = float(value)
        except Exception:
            continue
    return canonicalize_feature_dict(out)


def accuracy_pct_from_report(report: dict) -> float | None:
    try:
        error = float(report.get("weighted_mean_normalized_abs_error", float("inf")))
    except Exception:
        return None
    if not math.isfinite(error):
        return None
    return max(0.0, 1.0 - error) * 100.0


def _float_or_none(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    out = float(value)
    if not math.isfinite(out):
        return None
    return out


def axis_accuracy_pct(axis_rows: Dict[str, dict], axis_name: str, value_key: str) -> float | None:
    axis_row = axis_rows.get(axis_name, {})
    if not isinstance(axis_row, dict):
        return None
    value = axis_row.get(value_key)
    if not isinstance(value, (int, float)):
        return None
    value_f = float(value)
    if not math.isfinite(value_f):
        return None
    return value_f


def build_segment_fill_plan(segment_seconds: Any, runtime_estimate: dict | None) -> dict:
    segment_s = _float_or_none(segment_seconds)
    elapsed_s = _float_or_none((runtime_estimate or {}).get("elapsed_seconds"))

    plan = {
        "segment_seconds": segment_s,
        "estimated_single_run_seconds": elapsed_s,
        "recommended_repeat_count": None,
        "estimated_clone_segment_seconds": None,
        "timing_gap_seconds": None,
        "runtime_estimate": runtime_estimate or None,
    }
    if segment_s is None or elapsed_s is None or elapsed_s <= 0:
        return plan

    repeat_count = max(1, int(math.ceil(segment_s / elapsed_s)))
    estimated_total = repeat_count * elapsed_s
    plan.update(
        {
            "recommended_repeat_count": repeat_count,
            "estimated_clone_segment_seconds": estimated_total,
            "timing_gap_seconds": estimated_total - segment_s,
        }
    )
    return plan


def summarize_clone_plan(segment_results: Dict[str, dict]) -> dict:
    rows: list[dict] = []
    for seg_key, segment_result in segment_results.items():
        if segment_result.get("status") != "ok":
            continue
        fill_plan = segment_result.get("segment_fill_plan")
        if not isinstance(fill_plan, dict):
            continue
        segment_s = _float_or_none(fill_plan.get("segment_seconds"))
        estimated_s = _float_or_none(fill_plan.get("estimated_clone_segment_seconds"))
        gap_s = _float_or_none(fill_plan.get("timing_gap_seconds"))
        rows.append(
            {
                "segment_key": seg_key,
                "segment_seconds": segment_s,
                "estimated_clone_segment_seconds": estimated_s,
                "timing_gap_seconds": gap_s,
            }
        )

    if not rows:
        return {
            "segments_with_runtime_estimate": 0,
            "target_total_seconds": None,
            "estimated_clone_total_seconds": None,
            "total_timing_gap_seconds": None,
            "mean_abs_timing_gap_seconds": None,
        }

    target_total = sum(row["segment_seconds"] for row in rows if row["segment_seconds"] is not None)
    estimated_total = sum(
        row["estimated_clone_segment_seconds"]
        for row in rows
        if row["estimated_clone_segment_seconds"] is not None
    )
    gap_values = [abs(row["timing_gap_seconds"]) for row in rows if row["timing_gap_seconds"] is not None]
    return {
        "segments_with_runtime_estimate": len(
            [row for row in rows if row["estimated_clone_segment_seconds"] is not None]
        ),
        "target_total_seconds": target_total,
        "estimated_clone_total_seconds": estimated_total,
        "total_timing_gap_seconds": (estimated_total - target_total) if rows else None,
        "mean_abs_timing_gap_seconds": (sum(gap_values) / len(gap_values)) if gap_values else None,
    }


_SCALE_TARGET_FLOOR_RATIO = float(os.environ.get("CLOUD_SCALE_TARGET_FLOOR_RATIO", "0.50"))


def _floored_scales(
    feature_scales: Dict[str, float],
    target_features: Dict[str, Any],
) -> Dict[str, float]:
    """Apply target-proportional floor so outlier-scale features don't dominate."""
    if _SCALE_TARGET_FLOOR_RATIO <= 0:
        return feature_scales
    out = dict(feature_scales)
    for name, scale in feature_scales.items():
        target_val = target_features.get(name)
        if target_val is not None:
            try:
                out[name] = max(scale, abs(float(target_val)) * _SCALE_TARGET_FLOOR_RATIO)
            except (TypeError, ValueError):
                pass
    return out


def axis_accuracy_reports(
    target_features: Dict[str, Any],
    chosen_features: Dict[str, Any],
    feature_scales: Dict[str, float],
) -> Dict[str, dict]:
    effective_scales = _floored_scales(feature_scales, target_features)
    reports: Dict[str, dict] = {}
    for axis_name, feature_names in AXIS_FEATURES.items():
        report = absolute_match_report(target_features, chosen_features, effective_scales, feature_names)
        reports[axis_name] = {
            "present_feature_count": int(report.get("present_feature_count", 0)),
            "feature_count": int(report.get("feature_count", 0)),
            "mean_normalized_abs_error": report.get("mean_normalized_abs_error"),
            "weighted_mean_normalized_abs_error": report.get("weighted_mean_normalized_abs_error"),
            "accuracy_pct": accuracy_pct_from_report(report),
        }
    return reports


def summarize_accuracy(segment_results: Dict[str, dict], feature_scales: Dict[str, float]) -> dict:
    per_segment: Dict[str, dict] = {}
    target_rows: list[Dict[str, Any]] = []
    chosen_rows: list[Dict[str, Any]] = []
    segment_weights: Dict[str, float] = {}

    for seg_key, segment_result in segment_results.items():
        if segment_result.get("status") != "ok":
            continue
        target_features = canonicalize_feature_dict(segment_result.get("target_features") or {})
        chosen_features = canonicalize_feature_dict(segment_result.get("chosen_features") or {})
        if not target_features or not chosen_features:
            continue
        cached_axes = segment_result.get("accuracy_axes")
        if isinstance(cached_axes, dict) and cached_axes:
            per_segment[seg_key] = cached_axes
        else:
            per_segment[seg_key] = axis_accuracy_reports(target_features, chosen_features, feature_scales)
        target_rows.append(target_features)
        chosen_rows.append(chosen_features)
        segment_seconds = _float_or_none(segment_result.get("segment_seconds"))
        segment_weights[seg_key] = segment_seconds if segment_seconds and segment_seconds > 0 else 1.0

    mean_target = average_feature_dict(target_rows)
    mean_chosen = average_feature_dict(chosen_rows)

    axes_summary: Dict[str, dict] = {}
    for axis_name, feature_names in AXIS_FEATURES.items():
        segment_errors: list[float] = []
        segment_accuracies: list[float] = []
        weighted_error_numer = 0.0
        weighted_acc_numer = 0.0
        total_weight = 0.0
        for seg_key, axis_report in per_segment.items():
            axis_row = axis_report.get(axis_name, {})
            present = int(axis_row.get("present_feature_count", 0) or 0)
            error = axis_row.get("weighted_mean_normalized_abs_error")
            accuracy = axis_row.get("accuracy_pct")
            weight = segment_weights.get(seg_key, 1.0)
            if present <= 0:
                continue
            if isinstance(error, (int, float)) and math.isfinite(float(error)):
                segment_errors.append(float(error))
                weighted_error_numer += float(error) * weight
            if isinstance(accuracy, (int, float)) and math.isfinite(float(accuracy)):
                segment_accuracies.append(float(accuracy))
                weighted_acc_numer += float(accuracy) * weight
            total_weight += weight

        mean_profile_report = absolute_match_report(
            mean_target, mean_chosen, _floored_scales(feature_scales, mean_target), feature_names
        )
        axes_summary[axis_name] = {
            "segments_scored": len(segment_accuracies),
            "mean_segment_accuracy_pct": (sum(segment_accuracies) / len(segment_accuracies)) if segment_accuracies else None,
            "mean_segment_normalized_abs_error": (sum(segment_errors) / len(segment_errors)) if segment_errors else None,
            "duration_weighted_accuracy_pct": (weighted_acc_numer / total_weight) if total_weight > 0 else None,
            "duration_weighted_normalized_abs_error": (weighted_error_numer / total_weight) if total_weight > 0 else None,
            "mean_profile_accuracy_pct": accuracy_pct_from_report(mean_profile_report),
            "mean_profile_normalized_abs_error": mean_profile_report.get("weighted_mean_normalized_abs_error"),
            "mean_profile_present_feature_count": mean_profile_report.get("present_feature_count"),
        }

    stop_policy_duration_weighted_accuracy_pct = axis_accuracy_pct(
        axes_summary,
        STOP_POLICY_AXIS,
        "duration_weighted_accuracy_pct",
    )
    stop_policy_mean_segment_accuracy_pct = axis_accuracy_pct(
        axes_summary,
        STOP_POLICY_AXIS,
        "mean_segment_accuracy_pct",
    )
    stop_policy_mean_profile_accuracy_pct = axis_accuracy_pct(
        axes_summary,
        STOP_POLICY_AXIS,
        "mean_profile_accuracy_pct",
    )

    weighted_axis_numer = 0.0
    weighted_axis_denom = 0.0
    for ax_name, ax_weight in AXIS_IMPORTANCE_WEIGHTS.items():
        ax_acc = axis_accuracy_pct(axes_summary, ax_name, "duration_weighted_accuracy_pct")
        if ax_acc is not None:
            weighted_axis_numer += ax_acc * ax_weight
            weighted_axis_denom += ax_weight
    weighted_axis_accuracy_pct = (
        weighted_axis_numer / weighted_axis_denom if weighted_axis_denom > 0 else None
    )

    return {
        "segments_scored": len(per_segment),
        "overall_mean_segment_accuracy_pct": axes_summary.get("overall", {}).get("mean_segment_accuracy_pct"),
        "overall_duration_weighted_accuracy_pct": axes_summary.get("overall", {}).get("duration_weighted_accuracy_pct"),
        "weighted_axis_accuracy_pct": weighted_axis_accuracy_pct,
        "stop_policy_axis": STOP_POLICY_AXIS,
        "stop_policy_mean_segment_accuracy_pct": stop_policy_mean_segment_accuracy_pct,
        "stop_policy_duration_weighted_accuracy_pct": stop_policy_duration_weighted_accuracy_pct,
        "stop_policy_duration_weighted_accuracy": (
            stop_policy_duration_weighted_accuracy_pct / 100.0
            if stop_policy_duration_weighted_accuracy_pct is not None
            else None
        ),
        "stop_policy_mean_profile_accuracy_pct": stop_policy_mean_profile_accuracy_pct,
        "axes": axes_summary,
        "per_segment": per_segment,
    }


def print_accuracy_report(report: dict) -> None:
    def fmt_value(value: Any, suffix: str = "") -> str:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return "n/a"
        return f"{float(value):.4f}{suffix}"

    print("\nAccuracy summary:")
    print(f"Scored segments: {report.get('segments_scored', 0)}")
    axes = report.get("axes", {})
    for axis_name in ("overall", "hw_counters", "tma_metrics", "sw_metrics", "ipc", "branch_cache_tlb_mpki"):
        axis = axes.get(axis_name, {})
        mean_seg_acc = axis.get("mean_segment_accuracy_pct")
        weighted_acc = axis.get("duration_weighted_accuracy_pct")
        mean_seg_err = axis.get("mean_segment_normalized_abs_error")
        mean_prof_acc = axis.get("mean_profile_accuracy_pct")
        mean_prof_err = axis.get("mean_profile_normalized_abs_error")
        print(
            f"  {axis_name}: "
            f"mean_segment_accuracy_pct={fmt_value(mean_seg_acc, '%')} "
            f"duration_weighted_accuracy_pct={fmt_value(weighted_acc, '%')} "
            f"mean_segment_normalized_abs_error={fmt_value(mean_seg_err)} "
            f"mean_profile_accuracy_pct={fmt_value(mean_prof_acc, '%')} "
            f"mean_profile_normalized_abs_error={fmt_value(mean_prof_err)}"
        )
    stop_policy_axis = report.get("stop_policy_axis", STOP_POLICY_AXIS)
    print(
        "  stop_policy: "
        f"accuracy={fmt_value(report.get('stop_policy_duration_weighted_accuracy'))} "
        f"accuracy_pct={fmt_value(report.get('stop_policy_duration_weighted_accuracy_pct'), '%')} "
        f"axis={stop_policy_axis}"
    )
    weighted_axis_acc = report.get("weighted_axis_accuracy_pct")
    print(
        f"  weighted_axis_aggregate: {fmt_value(weighted_axis_acc, '%')} "
        f"(weights: ipc=0.30, tma=0.25, hw_counters=0.25, sw_metrics=0.10, mpki=0.10)"
    )


_LLM_OUTPUTS_DIR = REPO_ROOT / "folly_edit_rag" / "scripts" / "outputs"


def _sidecar_token_usage(run_prefix: str, segment_key: str) -> dict[str, int]:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "call_count": 0,
    }
    if not run_prefix or not segment_key:
        return totals
    pattern = f"{run_prefix}_{segment_key}_*.token_usage.json"
    for directory in (
        _LLM_OUTPUTS_DIR / "patches",
        _LLM_OUTPUTS_DIR / "compile_logs",
    ):
        for path in directory.glob(pattern):
            try:
                usage = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for key in totals:
                totals[key] += int(usage.get(key) or 0)
    return totals


def _extract_segment_inference_stats(
    segment_result: dict,
    *,
    run_prefix: str = "",
    segment_key: str = "",
) -> dict:
    """Extract inference/timing stats from one segment result's adjustor response."""
    adjustor = (segment_result.get("search_response") or {}).get("adjustor_response") or {}
    attempts = adjustor.get("attempts") or []
    generation_calls = 0
    compile_repair_calls = 0
    adjustor_elapsed_s = _float_or_none(adjustor.get("elapsed_wall_seconds"))
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        gen = attempt.get("generate") or {}
        if isinstance(gen, dict) and gen.get("returncode") is not None:
            generation_calls += 1
        cr = attempt.get("compile_record") or {}
        if isinstance(cr, dict):
            compile_repair_calls += int(cr.get("compile_attempt_count") or 0)
    # Token usage (propagated from cloud_tuning.py → adjustor)
    token_usage = adjustor.get("token_usage") or {}
    if not int(token_usage.get("call_count") or 0):
        token_usage = _sidecar_token_usage(run_prefix, segment_key)
    return {
        "adjustor_elapsed_seconds": adjustor_elapsed_s,
        "adjustor_attempt_count": len(attempts),
        "generation_calls": generation_calls,
        "compile_repair_calls": compile_repair_calls,
        "total_llm_calls": generation_calls + compile_repair_calls,
        "input_tokens": int(token_usage.get("input_tokens") or 0),
        "output_tokens": int(token_usage.get("output_tokens") or 0),
        "total_tokens": int(token_usage.get("total_tokens") or 0),
        "llm_call_count": int(token_usage.get("call_count") or 0),
    }


def summarize_inference_stats(
    segment_results: Dict[str, dict],
    *,
    run_prefix: str = "",
) -> dict:
    """Aggregate inference stats across all segments."""
    total_generation_calls = 0
    total_compile_repair_calls = 0
    total_adjustor_elapsed_s = 0.0
    total_adjustor_attempts = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    total_llm_call_count = 0
    per_segment: Dict[str, dict] = {}
    for seg_key, segment_result in segment_results.items():
        stats = _extract_segment_inference_stats(
            segment_result,
            run_prefix=run_prefix,
            segment_key=seg_key,
        )
        per_segment[seg_key] = stats
        total_generation_calls += stats["generation_calls"]
        total_compile_repair_calls += stats["compile_repair_calls"]
        total_adjustor_attempts += stats["adjustor_attempt_count"]
        if stats["adjustor_elapsed_seconds"] is not None:
            total_adjustor_elapsed_s += stats["adjustor_elapsed_seconds"]
        total_input_tokens += stats["input_tokens"]
        total_output_tokens += stats["output_tokens"]
        total_tokens += stats["total_tokens"]
        total_llm_call_count += stats["llm_call_count"]
    total_llm_calls = total_generation_calls + total_compile_repair_calls

    provider = _detect_llm_provider()
    # In tournament mode, estimate cost for each provider separately
    tournament_providers_env = os.environ.get("CLOUD_TOURNAMENT_PROVIDERS", "")
    is_tournament = os.environ.get("CLOUD_TOURNAMENT_MODE", "0").strip().lower() not in {
        "", "0", "false", "no", "off",
    }
    if is_tournament and tournament_providers_env:
        providers_used = [
            p.strip().lower()
            for p in tournament_providers_env.split(",")
            if p.strip()
        ]
    else:
        providers_used = [provider]

    # Estimate per-provider cost (tokens split equally if multiple providers)
    cost_by_provider: Dict[str, float] = {}
    for prov in providers_used:
        # In tournament mode each provider handles roughly equal share
        share = 1.0 / len(providers_used) if len(providers_used) > 1 else 1.0
        cost_by_provider[prov] = estimate_llm_cost(
            int(total_input_tokens * share),
            int(total_output_tokens * share),
            provider=prov,
        )
    estimated_cost_usd = sum(cost_by_provider.values())

    return {
        "total_generation_calls": total_generation_calls,
        "total_compile_repair_calls": total_compile_repair_calls,
        "total_llm_calls": total_llm_calls,
        "total_adjustor_attempts": total_adjustor_attempts,
        "total_adjustor_elapsed_seconds": total_adjustor_elapsed_s,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_llm_call_count": total_llm_call_count,
        "estimated_cost_usd": estimated_cost_usd,
        "cost_by_provider": cost_by_provider,
        "providers_used": providers_used,
        "per_segment": per_segment,
    }


def _format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.1f}s"
    hours = int(minutes // 60)
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m {secs:.0f}s"


def _print_timing_and_inference_summary(
    segment_results: Dict[str, dict],
    pipeline_elapsed_s: float,
    inference_stats: dict,
) -> None:
    """Print timing and inference cost summary to stdout."""
    print("\nTiming & inference summary:")
    print(f"  Total pipeline elapsed time: {_format_duration(pipeline_elapsed_s)}")

    # Per-segment timing
    per_seg = inference_stats.get("per_segment") or {}
    for seg_key in sorted(segment_results.keys()):
        sr = segment_results[seg_key]
        seg_elapsed = sr.get("segment_elapsed_seconds")
        seg_stats = per_seg.get(seg_key) or {}
        elapsed_text = _format_duration(seg_elapsed) if seg_elapsed is not None else "n/a"
        adj_elapsed = seg_stats.get("adjustor_elapsed_seconds")
        adj_text = _format_duration(adj_elapsed) if adj_elapsed is not None else "n/a"
        attempts = seg_stats.get("adjustor_attempt_count", 0)
        gen_calls = seg_stats.get("generation_calls", 0)
        repair_calls = seg_stats.get("compile_repair_calls", 0)
        seg_tokens = seg_stats.get("total_tokens", 0)
        print(
            f"  [{seg_key}] elapsed={elapsed_text} adjustor={adj_text} "
            f"attempts={attempts} gen_calls={gen_calls} repair_calls={repair_calls} "
            f"tokens={seg_tokens:,}"
        )

    # Totals
    total_gen = inference_stats.get("total_generation_calls", 0)
    total_repair = inference_stats.get("total_compile_repair_calls", 0)
    total_llm = inference_stats.get("total_llm_calls", 0)
    total_attempts = inference_stats.get("total_adjustor_attempts", 0)
    total_adj_elapsed = inference_stats.get("total_adjustor_elapsed_seconds", 0.0)
    total_input_tok = inference_stats.get("total_input_tokens", 0)
    total_output_tok = inference_stats.get("total_output_tokens", 0)
    total_tok = inference_stats.get("total_tokens", 0)
    total_llm_count = inference_stats.get("total_llm_call_count", 0)
    print("\n  Totals:")
    print(f"    LLM generation calls:     {total_gen}")
    print(f"    LLM compile/repair calls: {total_repair}")
    print(f"    Total LLM calls:          {total_llm}")
    print(f"    Total LLM inferences:     {total_llm_count}")
    print(f"    Total adjustor attempts:  {total_attempts}")
    print(f"    Total adjustor time:      {_format_duration(total_adj_elapsed)}")
    print(f"    Total pipeline time:      {_format_duration(pipeline_elapsed_s)}")
    print(f"    Input tokens:             {total_input_tok:,}")
    print(f"    Output tokens:            {total_output_tok:,}")
    print(f"    Total tokens:             {total_tok:,}")

    estimated_cost = inference_stats.get("estimated_cost_usd", 0.0)
    cost_by_provider = inference_stats.get("cost_by_provider") or {}
    providers_used = inference_stats.get("providers_used") or []
    if estimated_cost > 0 or total_tok > 0:
        print("\n  Estimated LLM cost:")
        for prov in providers_used:
            prov_cost = cost_by_provider.get(prov, 0.0)
            pricing = _PRICING_TABLE.get(prov, {})
            in_rate = pricing.get("input_rate", 0)
            out_rate = pricing.get("output_rate", 0)
            print(f"    {prov}: ${prov_cost:.4f}  (in=${in_rate}/1M, out=${out_rate}/1M)")
        print(f"    Total estimated cost:     ${estimated_cost:.4f}")


def extract_segments(cloud_obj: dict) -> Dict[str, Dict[str, Any]]:
    segments: Dict[str, Dict[str, Any]] = {}
    sortable: list[Tuple[int, str]] = []

    for key in cloud_obj.keys():
        m = SEGMENT_KEY_RE.fullmatch(key)
        if not m:
            continue
        sortable.append((int(m.group(1)), key))

    for _, seg_key in sorted(sortable):
        seg_obj = cloud_obj.get(seg_key)
        if not isinstance(seg_obj, dict):
            continue

        raw_metrics = seg_obj.get("metrics", seg_obj)
        if not isinstance(raw_metrics, dict):
            continue

        feats = numeric_features(raw_metrics)
        if not feats:
            continue

        segments[seg_key] = {
            "segment_index": seg_obj.get("segment_index"),
            "segment_seconds": seg_obj.get("segment_seconds"),
            "features": feats,
        }

    return segments


def derive_replacement(search_result: dict) -> Tuple[str, str]:
    threshold_met = bool(search_result.get("threshold_met"))
    nearest_id = search_result.get("nearest_id")
    adjustor_resp = search_result.get("adjustor_response") or {}
    best_candidate = adjustor_resp.get("best_candidate") if isinstance(adjustor_resp, dict) else {}

    if threshold_met:
        return "found", str(nearest_id) if nearest_id is not None else ""

    if isinstance(best_candidate, dict):
        matched = best_candidate.get("microbenchmark")
        if matched:
            return "generated_or_adjusted", str(matched)

    if nearest_id is not None:
        return "generated_or_adjusted", str(nearest_id)
    return "generated_or_adjusted", ""


def process_segment(
    segment_key: str,
    segment: dict,
    search_url: str,
    run_prefix: str,
    timeout_s: float | None,
    k: int,
    max_wall_seconds: float | None = None,
    warm_start_file: str | None = None,
    warm_start_features: dict | None = None,
) -> dict:
    run_id = f"{run_prefix}_{segment_key}"
    payload = {
        "run_id": run_id,
        "query_json": segment["features"],
        "k": k,
    }
    if max_wall_seconds is not None:
        payload["max_wall_seconds"] = max_wall_seconds
    if warm_start_file is not None:
        payload["warm_start_file"] = warm_start_file
    if warm_start_features is not None:
        payload["warm_start_features"] = warm_start_features

    result = {
        "run_id": run_id,
        "segment_index": segment.get("segment_index"),
        "segment_seconds": segment.get("segment_seconds"),
        "feature_count": len(segment["features"]),
        "target_features": segment["features"],
        "status": "error",
    }

    try:
        resp = requests.post(search_url, json=payload, timeout=timeout_s)
        result["http_status"] = resp.status_code

        try:
            body = resp.json()
        except Exception:
            body = {"error": f"non-json response: {resp.text[:500]}"}

        result["search_response"] = body
        if resp.status_code != 200:
            result["error"] = body.get("error", f"search failed with status {resp.status_code}")
            return result

        replacement_type, chosen_microbenchmark = derive_replacement(body)
        adjustor_resp = body.get("adjustor_response") or {}
        adjustor_success = adjustor_resp.get("success") if isinstance(adjustor_resp, dict) else None
        if not body.get("threshold_met") and adjustor_success is not True:
            compile_record = adjustor_resp.get("compile_record") or {}
            best_candidate = adjustor_resp.get("best_candidate") or {}
            compiled_file = compile_record.get("final_file") or best_candidate.get("source_file")
            has_best_compiled_candidate = bool(
                compiled_file
                and Path(str(compiled_file)).is_file()
                and best_candidate.get("microbenchmark")
                and canonicalize_feature_dict(best_candidate.get("features") or {})
            )
            if not has_best_compiled_candidate:
                stage = adjustor_resp.get("stage", "adjustor") if isinstance(adjustor_resp, dict) else "adjustor"
                detail = (
                    adjustor_resp.get("error")
                    or adjustor_resp.get("reason")
                    or f"{stage} did not produce a compiled candidate"
                )
                result["error"] = str(detail)
                return result
            replacement_type = "best_effort_compiled"
            chosen_microbenchmark = str(best_candidate["microbenchmark"])
        chosen_features = {}
        runtime_estimate = None
        if body.get("threshold_met"):
            chosen_features = canonicalize_feature_dict(body.get("nearest_features") or {})
            if not chosen_features:
                chosen_features = _features_from_report(body.get("baseline_match_report"), "candidate_value")
            nearest_binary_path = body.get("nearest_binary_path")
            if nearest_binary_path and chosen_microbenchmark:
                runtime_estimate = measure_unit_runtime(Path(str(nearest_binary_path)), chosen_microbenchmark)
        elif isinstance(adjustor_resp, dict):
            best_candidate = adjustor_resp.get("best_candidate") or {}
            chosen_features = canonicalize_feature_dict(best_candidate.get("features") or {})
            if not chosen_features:
                chosen_features = _features_from_report(adjustor_resp.get("best_candidate_report"), "candidate_value")
            runtime_estimate = best_candidate.get("runtime_estimate") or adjustor_resp.get("chosen_runtime_estimate")

        segment_fill_plan = build_segment_fill_plan(segment.get("segment_seconds"), runtime_estimate)

        result.update(
            {
                "status": "ok",
                "nearest_id": body.get("nearest_id"),
                "nearest_binary": body.get("nearest_binary"),
                "nearest_binary_path": body.get("nearest_binary_path"),
                "retrieval_score": body.get("retrieval_score"),
                "seed_selection_mode": body.get("seed_selection_mode"),
                "baseline_error": body.get("baseline_error"),
                "threshold_met": body.get("threshold_met"),
                "replacement_type": replacement_type,
                "chosen_microbenchmark": chosen_microbenchmark,
                "adjustor_success": adjustor_success,
                "diagnostician_report": (
                    adjustor_resp.get("diagnostician_report") if isinstance(adjustor_resp, dict) else None
                ),
                "diagnostician_attempt_index": (
                    adjustor_resp.get("diagnostician_attempt_index") if isinstance(adjustor_resp, dict) else None
                ),
                "compile_record": adjustor_resp.get("compile_record") if isinstance(adjustor_resp, dict) else None,
                "candidate_error": adjustor_resp.get("candidate_error") if isinstance(adjustor_resp, dict) else None,
                "chosen_features": chosen_features,
                "chosen_runtime_estimate": runtime_estimate,
                "segment_fill_plan": segment_fill_plan,
                "target_file": adjustor_resp.get("target_file") if isinstance(adjustor_resp, dict) else None,
                "build_dir": adjustor_resp.get("build_dir") if isinstance(adjustor_resp, dict) else None,
                "binary_path": adjustor_resp.get("binary_path") if isinstance(adjustor_resp, dict) else body.get("nearest_binary_path"),
                "binary_name": adjustor_resp.get("binary_name") if isinstance(adjustor_resp, dict) else body.get("nearest_binary"),
                "pipeline": [
                    "input_to_search_delta.py",
                    "adjustor.py" if body.get("adjustor_response") is not None else "adjustor.py (skipped: baseline already acceptable)",
                    "folly_edit_rag scripts" if body.get("adjustor_response") is not None else "folly_edit_rag scripts (skipped)",
                ],
            }
        )
        return result
    except Exception as e:
        result["error"] = str(e)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run search/adjust/compile pipeline for each segment in a cloud feature JSON."
    )
    parser.add_argument(
        "--cloud-json",
        required=True,
        help="Path to cloud JSON produced by run_feature_extraction.py in cloud mode.",
    )
    parser.add_argument(
        "--search-url",
        default="http://127.0.0.1:5001/search",
        help="Search endpoint (input_to_search_delta.py service).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Top-k sent to search service (default: 1).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_SEARCH_TIMEOUT_SECONDS,
        help=f"Total pipeline wall-clock budget in seconds (default: {DEFAULT_SEARCH_TIMEOUT_SECONDS:.0f}); "
        "use 0 for no wall-clock limit. Time is divided fairly across remaining segments.",
    )
    parser.add_argument(
        "--run-id-prefix",
        default=None,
        help="Optional run_id prefix. Default uses cloud-json basename + UTC timestamp.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Path to write consolidated per-segment results JSON.",
    )
    parser.add_argument(
        "--stop-on-generated-accuracy",
        type=float,
        default=None,
        help=(
            "Optional early-stop threshold in [0,1]. Stop after the first segment whose newly generated "
            "candidate reaches the weighted HW/TMA/SW accuracy policy."
        ),
    )
    parser.add_argument(
        "--max-segment-wall-seconds",
        type=float,
        default=None,
        help=(
            "Optional per-segment wall-clock budget in seconds. Passed as max_wall_seconds "
            "to the adjustor so each segment gets a fair time share."
        ),
    )
    parser.add_argument(
        "--min-seed-ipc-accuracy",
        type=float,
        default=None,
        help=(
            "Only tune segments whose closest corpus IPC seed has at least this "
            "relative accuracy in [0,1]."
        ),
    )
    parser.add_argument(
        "--initial-warm-start-result-json",
        default=os.environ.get("CLOUD_INITIAL_WARM_START_RESULT_JSON"),
        help=(
            "Optional prior segment-replacements JSON. Its best successful candidate "
            "is supplied to the first segment as a warm start."
        ),
    )
    args = parser.parse_args()

    if args.stop_on_generated_accuracy is not None:
        if not math.isfinite(args.stop_on_generated_accuracy) or not (0.0 <= args.stop_on_generated_accuracy <= 1.0):
            print("Error: --stop-on-generated-accuracy must be a finite value in [0, 1]")
            return 1
    if args.min_seed_ipc_accuracy is not None:
        if not math.isfinite(args.min_seed_ipc_accuracy) or not (
            0.0 <= args.min_seed_ipc_accuracy <= 1.0
        ):
            print("Error: --min-seed-ipc-accuracy must be a finite value in [0, 1]")
            return 1

    cloud_json_path = Path(args.cloud_json).expanduser().resolve()
    if not cloud_json_path.exists():
        print(f"Error: cloud JSON not found: {cloud_json_path}")
        return 1

    cloud_obj = load_json(cloud_json_path)
    feature_table = load_feature_table(DEFAULT_XLSX)
    feature_scales = dataframe_feature_scales(feature_table)
    segments = extract_segments(cloud_obj)
    if not segments:
        print("Error: no segment_* entries with numeric metrics found in cloud JSON")
        return 1

    seed_filter_results: Dict[str, dict] = {}
    if args.min_seed_ipc_accuracy is not None:
        corpus_ipc = pd.to_numeric(feature_table["perf_stat.IPC"], errors="coerce").dropna()
        eligible_segments: Dict[str, Dict[str, Any]] = {}
        for seg_key, segment in segments.items():
            target_ipc = segment["features"].get("perf_stat.IPC")
            if target_ipc is None or abs(float(target_ipc)) <= 1e-9 or corpus_ipc.empty:
                seed_filter_results[seg_key] = {
                    "segment_key": seg_key,
                    "status": "skipped",
                    "reason": "no comparable target/corpus IPC",
                }
                continue
            nearest_index = (corpus_ipc - float(target_ipc)).abs().idxmin()
            nearest_ipc = float(corpus_ipc.loc[nearest_index])
            ipc_accuracy = max(
                0.0,
                1.0 - abs(nearest_ipc - float(target_ipc)) / abs(float(target_ipc)),
            )
            if ipc_accuracy < args.min_seed_ipc_accuracy:
                print(
                    f"[{seg_key}] skipping — closest corpus IPC {nearest_ipc:.4g} "
                    f"matches target {float(target_ipc):.4g} at {ipc_accuracy * 100:.1f}% "
                    f"(< {args.min_seed_ipc_accuracy * 100:.1f}%)",
                    flush=True,
                )
                seed_filter_results[seg_key] = {
                    "segment_key": seg_key,
                    "status": "skipped",
                    "reason": "closest corpus IPC below required accuracy",
                    "target_ipc": float(target_ipc),
                    "nearest_seed_ipc": nearest_ipc,
                    "seed_ipc_accuracy": ipc_accuracy,
                }
                continue
            print(
                f"[{seg_key}] eligible — closest corpus IPC {nearest_ipc:.4g} "
                f"matches target {float(target_ipc):.4g} at {ipc_accuracy * 100:.1f}%",
                flush=True,
            )
            segment["nearest_seed_ipc"] = nearest_ipc
            segment["seed_ipc_accuracy"] = ipc_accuracy
            eligible_segments[seg_key] = segment
        segments = eligible_segments
        if not segments:
            print("Error: no segments meet the minimum seed IPC accuracy")
            return 1

    planned_segments = cloud_obj.get("segments_planned")
    collected_segments = cloud_obj.get("segments_collected")
    collection_stop_reason = cloud_obj.get("collection_stop_reason")

    stamp = utc_compact()
    run_prefix = args.run_id_prefix or f"{cloud_json_path.stem}_{stamp}"
    out_path = Path(args.output_json).expanduser().resolve() if args.output_json else (
        Path.cwd() / f"{cloud_json_path.stem}_segment_replacements_{stamp}.json"
    )

    print(f"Processing {len(segments)} segments from {cloud_json_path}")
    if planned_segments is not None or collected_segments is not None:
        print(
            "Source segment summary: "
            f"planned={planned_segments if planned_segments is not None else 'unknown'} "
            f"collected={collected_segments if collected_segments is not None else 'unknown'} "
            f"processable={len(segments)}"
        )
    if collection_stop_reason:
        print(f"Source collection stop reason: {collection_stop_reason}")
    print(f"Search URL: {args.search_url}")
    print(f"Run prefix: {run_prefix}")

    segment_results: Dict[str, dict] = dict(seed_filter_results)
    ok_count = 0
    pipeline_start_monotonic = time.monotonic()
    stop_threshold_pct = (
        args.stop_on_generated_accuracy * 100.0 if args.stop_on_generated_accuracy is not None else None
    )
    early_stop = {
        "enabled": args.stop_on_generated_accuracy is not None,
        "requires_new_generation": True,
        "threshold_accuracy": args.stop_on_generated_accuracy,
        "threshold_accuracy_pct": stop_threshold_pct,
        "triggered": False,
        "segment_key": None,
        "policy_accuracy": None,
        "policy_accuracy_pct": None,
        "chosen_microbenchmark": None,
    }

    max_segment_wall_seconds = args.max_segment_wall_seconds
    pipeline_deadline = (
        None
        if args.timeout_seconds <= 0
        else pipeline_start_monotonic + args.timeout_seconds
    )
    num_segments = len(segments)
    # Cross-segment warm-start: track the best result seen so far.
    best_warm_start_file: str | None = None
    best_warm_start_features: dict | None = None
    best_warm_start_error: float = float("inf")
    if args.initial_warm_start_result_json:
        try:
            prior_payload = json.loads(
                Path(args.initial_warm_start_result_json).expanduser().resolve().read_text(
                    encoding="utf-8"
                )
            )
            prior_candidates: list[tuple[float, str, dict]] = []
            for prior_result in dict(prior_payload.get("segment_results") or {}).values():
                if prior_result.get("status") != "ok":
                    continue
                prior_features = canonicalize_feature_dict(
                    prior_result.get("chosen_features") or {}
                )
                prior_file = str(
                    dict(prior_result.get("compile_record") or {}).get("final_file") or ""
                )
                if not prior_features or not prior_file or not Path(prior_file).is_file():
                    continue
                prior_error = float(prior_result.get("candidate_error", float("inf")))
                prior_candidates.append((prior_error, prior_file, prior_features))
            if not prior_candidates:
                raise ValueError("no successful candidate with a readable final file was found")
            best_warm_start_error, best_warm_start_file, best_warm_start_features = min(
                prior_candidates, key=lambda item: item[0]
            )
            print(
                f"Initial warm start: error={best_warm_start_error:.4f} "
                f"file={best_warm_start_file}",
                flush=True,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(f"Error: could not load initial warm start: {exc}")
            return 1

    for seg_idx, (seg_key, seg) in enumerate(segments.items()):
        remaining_pipeline_s = (
            None
            if pipeline_deadline is None
            else pipeline_deadline - time.monotonic()
        )
        if remaining_pipeline_s is not None and remaining_pipeline_s <= 60:
            print(
                f"\n[{seg_key}] skipping — pipeline wall budget exhausted "
                f"({remaining_pipeline_s:.0f}s remaining)",
                flush=True,
            )
            segment_results[seg_key] = {
                "segment_key": seg_key,
                "status": "skipped",
                "reason": "pipeline wall budget exhausted",
            }
            continue
        # Per-segment budget: user-specified cap or fair share of remaining time
        remaining_segments = num_segments - seg_idx
        fair_share_s = (
            None
            if remaining_pipeline_s is None
            else remaining_pipeline_s / remaining_segments
        )
        effective_wall_seconds = fair_share_s
        if max_segment_wall_seconds is not None:
            effective_wall_seconds = (
                max_segment_wall_seconds
                if fair_share_s is None
                else min(max_segment_wall_seconds, fair_share_s)
            )
        effective_timeout_s = (
            None
            if remaining_pipeline_s is None
            else min(args.timeout_seconds, remaining_pipeline_s)
        )
        budget_text = (
            "unbounded"
            if effective_wall_seconds is None
            else f"{effective_wall_seconds:.0f}s"
        )

        print(
            f"\n[{seg_key}] feature_count={len(seg['features'])} -> pipeline request "
            f"(budget {budget_text}, {remaining_segments} segments remaining)",
        )
        seg_start_monotonic = time.monotonic()
        r = process_segment(
            segment_key=seg_key,
            segment=seg,
            search_url=args.search_url,
            run_prefix=run_prefix,
            timeout_s=effective_timeout_s,
            k=max(1, int(args.k)),
            max_wall_seconds=effective_wall_seconds,
            warm_start_file=best_warm_start_file,
            warm_start_features=best_warm_start_features,
        )
        seg_elapsed_s = time.monotonic() - seg_start_monotonic
        r["segment_elapsed_seconds"] = seg_elapsed_s
        if r.get("status") == "ok":
            accuracy_features = canonicalize_feature_dict(r.get("chosen_features") or {})
            accuracy_feature_source = "chosen_features" if accuracy_features else None
            if not accuracy_features:
                search_response = r.get("search_response") or {}
                accuracy_features = canonicalize_feature_dict(search_response.get("nearest_features") or {})
                if accuracy_features:
                    accuracy_feature_source = "search_response.nearest_features"
            if not accuracy_features:
                search_response = r.get("search_response") or {}
                accuracy_features = _features_from_report(
                    search_response.get("baseline_match_report"),
                    "candidate_value",
                )
                if accuracy_features:
                    accuracy_feature_source = "baseline_match_report.candidate_value"
            accuracy_axes = axis_accuracy_reports(
                r.get("target_features") or {},
                accuracy_features,
                feature_scales,
            )
            policy_accuracy_pct = axis_accuracy_pct(
                accuracy_axes,
                STOP_POLICY_AXIS,
                "accuracy_pct",
            )
            r["accuracy_axes"] = accuracy_axes
            r["accuracy_feature_source"] = accuracy_feature_source
            r["policy_accuracy_pct"] = policy_accuracy_pct
            r["policy_accuracy"] = (
                policy_accuracy_pct / 100.0 if policy_accuracy_pct is not None else None
            )
            r["is_new_generation"] = bool(r.get("adjustor_success"))
        segment_results[seg_key] = r
        # Cross-segment warm-start: update best-so-far candidate for future segments.
        if r.get("status") == "ok":
            seg_candidate_error = r.get("candidate_error")
            seg_compile_file = None
            seg_sr = r.get("search_response") or {}
            seg_adj = seg_sr.get("adjustor_response") or {}
            if isinstance(seg_adj, dict):
                seg_bc = seg_adj.get("best_candidate") or {}
                seg_compile = seg_adj.get("compile_record") or {}
                seg_compile_file = seg_compile.get("final_file") or seg_bc.get("source_file")
            if (
                seg_candidate_error is not None
                and float(seg_candidate_error) < best_warm_start_error
                and seg_compile_file
                and r.get("chosen_features")
            ):
                best_warm_start_error = float(seg_candidate_error)
                best_warm_start_file = str(seg_compile_file)
                best_warm_start_features = r["chosen_features"]
                print(
                    f"[{seg_key}] warm-start updated: error={best_warm_start_error:.4f} "
                    f"file={best_warm_start_file}",
                    flush=True,
                )
        if r.get("status") == "ok":
            ok_count += 1
            policy_accuracy_pct = r.get("policy_accuracy_pct")
            policy_text = (
                f" policy_accuracy_pct={float(policy_accuracy_pct):.4f}%"
                if isinstance(policy_accuracy_pct, (int, float)) and math.isfinite(float(policy_accuracy_pct))
                else ""
            )
            print(
                f"[{seg_key}] ok: baseline_error={r.get('baseline_error')} "
                f"replacement={r.get('chosen_microbenchmark')} type={r.get('replacement_type')}"
                f"{policy_text}"
            )
            if (
                stop_threshold_pct is not None
                and bool(r.get("is_new_generation"))
                and isinstance(policy_accuracy_pct, (int, float))
                and math.isfinite(float(policy_accuracy_pct))
                and float(policy_accuracy_pct) >= stop_threshold_pct
            ):
                early_stop.update(
                    {
                        "triggered": True,
                        "segment_key": seg_key,
                        "policy_accuracy": r.get("policy_accuracy"),
                        "policy_accuracy_pct": policy_accuracy_pct,
                        "chosen_microbenchmark": r.get("chosen_microbenchmark"),
                    }
                )
                print(
                    f"[{seg_key}] early stop: new generation reached "
                    f"{float(policy_accuracy_pct):.4f}% >= {stop_threshold_pct:.4f}%"
                )
                break
        else:
            print(f"[{seg_key}] error: {r.get('error')}")

    stitching_plan = []
    for seg_key, segment_result in segment_results.items():
        if segment_result.get("status") != "ok":
            continue
        stitching_plan.append(
            {
                "segment_key": seg_key,
                "segment_index": segment_result.get("segment_index"),
                "segment_seconds": segment_result.get("segment_seconds"),
                "replacement_type": segment_result.get("replacement_type"),
                "chosen_microbenchmark": segment_result.get("chosen_microbenchmark"),
                "nearest_id": segment_result.get("nearest_id"),
                "target_file": segment_result.get("target_file"),
                "build_dir": segment_result.get("build_dir"),
                "candidate_error": segment_result.get("candidate_error"),
                "segment_fill_plan": segment_result.get("segment_fill_plan"),
            }
        )

    accuracy_report = summarize_accuracy(segment_results, feature_scales)
    pipeline_elapsed_s = time.monotonic() - pipeline_start_monotonic
    inference_stats = summarize_inference_stats(segment_results, run_prefix=run_prefix)
    processed_count = len(segment_results)
    failed_count = sum(1 for segment_result in segment_results.values() if segment_result.get("status") != "ok")
    summary = {
        "source_cloud_json": str(cloud_json_path),
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "search_url": args.search_url,
        "source_segments_planned": planned_segments,
        "source_segments_collected": collected_segments,
        "source_collection_stop_reason": collection_stop_reason,
        "segments_total": len(segments),
        "segments_processed": processed_count,
        "segments_succeeded": ok_count,
        "segments_failed": failed_count,
        "segments_unprocessed": len(segments) - processed_count,
        "pipeline_elapsed_seconds": pipeline_elapsed_s,
        "inference_stats": inference_stats,
        "segment_results": segment_results,
        "stitching_plan": stitching_plan,
        "clone_plan_summary": summarize_clone_plan(segment_results),
        "accuracy_report": accuracy_report,
        "final_aggregate_accuracy_pct": accuracy_report.get("weighted_axis_accuracy_pct"),
        "overall_duration_weighted_accuracy_pct": accuracy_report.get("overall_duration_weighted_accuracy_pct"),
        "stop_policy_accuracy_pct": accuracy_report.get("stop_policy_duration_weighted_accuracy_pct"),
        "stop_policy_accuracy": accuracy_report.get("stop_policy_duration_weighted_accuracy"),
        "estimated_cost_usd": inference_stats.get("estimated_cost_usd"),
        "cost_by_provider": inference_stats.get("cost_by_provider"),
        "early_stop": early_stop,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Wrote: {out_path}")
    print(f"Succeeded: {ok_count}/{processed_count}")
    if early_stop.get("enabled"):
        if early_stop.get("triggered"):
            print(
                "Early stop: triggered "
                f"at {early_stop.get('segment_key')} with "
                f"{early_stop.get('policy_accuracy_pct'):.4f}%"
            )
        else:
            print(
                "Early stop: not triggered "
                f"(threshold={early_stop.get('threshold_accuracy_pct'):.4f}% on new generations only)"
            )
    print_accuracy_report(summary["accuracy_report"])
    _print_timing_and_inference_summary(segment_results, pipeline_elapsed_s, inference_stats)

    if early_stop.get("triggered"):
        return 0
    return 0 if failed_count == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
