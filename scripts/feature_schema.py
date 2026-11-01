from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from cloning_priority import feature_constraint


def resolve_feature_table_path(default_path: Path) -> Path:
    configured = os.environ.get("FEATURE_TABLE_XLSX")
    return Path(configured).expanduser().resolve() if configured else default_path.resolve()

CANONICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "intel_pt.block_size_P50",
    "intel_pt.family::arith_P50",
    "intel_pt.family::branch_P50",
    "intel_pt.family::load_P50",
    "intel_pt.family::logic_P50",
    "intel_pt.family::mem_P50",
    "intel_pt.family::move_P50",
    "intel_pt.family::other_P50",
    "intel_pt.family::store_P50",
    "intel_pt.jumps_P50",
    "intel_pt.not_taken_runs_P50",
    "intel_pt.raw_distances_P50",
    "intel_pt.taken_runs_P50",
    "perf_stat.IPC",
    "perf_stat.L1-dcache-load-misses_MPKI",
    "perf_stat.L1-dcache-loads_LPKI",
    "perf_stat.L1-dcache-stores_SPKI",
    "perf_stat.L1-icache-load-misses_MPKI",
    "perf_stat.L2-icache-load-misses_MPKI",
    "perf_stat.LLC-load-misses_MPKI",
    "perf_stat.branch-load-misses_MPKI",
    "perf_stat.branch-misses_MPKI",
    "perf_stat.dTLB-load-misses_MPKI",
    "perf_stat.iTLB-load-misses_MPKI",
    "tma.TMA_Backend_Bound(%)",
    "tma.TMA_Bad_Speculation(%)",
    "tma.TMA_Frontend_Bound(%)",
    "tma.TMA_Retiring(%)",
    "tma.TMA_Memory_Bound(%)",
    "tma.TMA_Core_Bound(%)",
    "tma.TMA_DRAM_Bound(%)",
    "tma.TMA_L1_Bound(%)",
    "tma.TMA_L2_Bound(%)",
    "tma.TMA_L3_Bound(%)",
)

ID_COLUMNS = ("binary", "binary_path", "command", "microbenchmark")
STATUS_COLUMNS = ("status.perf_stat_ok", "status.tma_ok", "status.intel_pt_ok")

_ALIAS_MAP = {
    "block_size_P50": "intel_pt.block_size_P50",
    "family::arith_P50": "intel_pt.family::arith_P50",
    "family::branch_P50": "intel_pt.family::branch_P50",
    "family::load_P50": "intel_pt.family::load_P50",
    "family::logic_P50": "intel_pt.family::logic_P50",
    "family::mem_P50": "intel_pt.family::mem_P50",
    "family::move_P50": "intel_pt.family::move_P50",
    "family::other_P50": "intel_pt.family::other_P50",
    "family::store_P50": "intel_pt.family::store_P50",
    "jumps_P50": "intel_pt.jumps_P50",
    "not_taken_runs_P50": "intel_pt.not_taken_runs_P50",
    "raw_distances_P50": "intel_pt.raw_distances_P50",
    "taken_runs_P50": "intel_pt.taken_runs_P50",
    "IPC": "perf_stat.IPC",
    "L1-dcache-load-misses_MPKI": "perf_stat.L1-dcache-load-misses_MPKI",
    "perf_stat.L1-dcache-load-misses_MPKI": "perf_stat.L1-dcache-load-misses_MPKI",
    "L1-dcache-loads_MPKI": "perf_stat.L1-dcache-loads_LPKI",
    "L1-dcache-stores_MPKI": "perf_stat.L1-dcache-stores_SPKI",
    "perf_stat.L1-dcache-loads_MPKI": "perf_stat.L1-dcache-loads_LPKI",
    "perf_stat.L1-dcache-stores_MPKI": "perf_stat.L1-dcache-stores_SPKI",
    "L1-dcache-loads_LPKI": "perf_stat.L1-dcache-loads_LPKI",
    "L1-dcache-stores_SPKI": "perf_stat.L1-dcache-stores_SPKI",
    "perf_stat.L1-dcache-loads_LPKI": "perf_stat.L1-dcache-loads_LPKI",
    "perf_stat.L1-dcache-stores_SPKI": "perf_stat.L1-dcache-stores_SPKI",
    "L1-icache-load-misses_MPKI": "perf_stat.L1-icache-load-misses_MPKI",
    "L2-icache-load-misses_MPKI": "perf_stat.L2-icache-load-misses_MPKI",
    "LLC-load-misses_MPKI": "perf_stat.LLC-load-misses_MPKI",
    "branch-load-misses_MPKI": "perf_stat.branch-load-misses_MPKI",
    "branch-misses_MPKI": "perf_stat.branch-misses_MPKI",
    "dTLB-load-misses_MPKI": "perf_stat.dTLB-load-misses_MPKI",
    "perf_stat.dTLB-load-misses_MPKI": "perf_stat.dTLB-load-misses_MPKI",
    "iTLB-load-misses_MPKI": "perf_stat.iTLB-load-misses_MPKI",
    "TMA_Backend_Bound(%)": "tma.TMA_Backend_Bound(%)",
    "TMA_Bad_Speculation(%)": "tma.TMA_Bad_Speculation(%)",
    "TMA_Frontend_Bound(%)": "tma.TMA_Frontend_Bound(%)",
    "TMA_Retiring(%)": "tma.TMA_Retiring(%)",
    "TMA_Memory_Bound(%)": "tma.TMA_Memory_Bound(%)",
    "TMA_Core_Bound(%)": "tma.TMA_Core_Bound(%)",
    "TMA_DRAM_Bound(%)": "tma.TMA_DRAM_Bound(%)",
    "TMA_L1_Bound(%)": "tma.TMA_L1_Bound(%)",
    "TMA_L2_Bound(%)": "tma.TMA_L2_Bound(%)",
    "TMA_L3_Bound(%)": "tma.TMA_L3_Bound(%)",
    "perfspect.TMA_Backend_Bound(%)": "tma.TMA_Backend_Bound(%)",
    "perfspect.TMA_Bad_Speculation(%)": "tma.TMA_Bad_Speculation(%)",
    "perfspect.TMA_Frontend_Bound(%)": "tma.TMA_Frontend_Bound(%)",
    "perfspect.TMA_Retiring(%)": "tma.TMA_Retiring(%)",
    "perfspect.TMA_Memory_Bound(%)": "tma.TMA_Memory_Bound(%)",
    "perfspect.TMA_Core_Bound(%)": "tma.TMA_Core_Bound(%)",
    "perfspect.TMA_DRAM_Bound(%)": "tma.TMA_DRAM_Bound(%)",
    "perfspect.TMA_L1_Bound(%)": "tma.TMA_L1_Bound(%)",
    "perfspect.TMA_L2_Bound(%)": "tma.TMA_L2_Bound(%)",
    "perfspect.TMA_L3_Bound(%)": "tma.TMA_L3_Bound(%)",
}


def canonicalize_feature_name(name: str) -> str:
    return _ALIAS_MAP.get(name, name)


def canonicalize_feature_dict(features: dict[str, object]) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw_name, raw_value in features.items():
        name = canonicalize_feature_name(str(raw_name))
        if name not in CANONICAL_FEATURE_COLUMNS:
            continue
        try:
            value = float(raw_value)
        except Exception:
            continue
        if math.isnan(value) or math.isinf(value):
            continue
        out[name] = value
    return out


def feature_display_name(feature_name: str) -> str:
    canonical = canonicalize_feature_name(feature_name)
    for prefix in ("intel_pt.", "perf_stat.", "tma."):
        if canonical.startswith(prefix):
            return canonical[len(prefix) :]
    return canonical


def canonicalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    exact_canonical: set[str] = set()
    for raw_column in df.columns:
        canonical = canonicalize_feature_name(str(raw_column))
        values = df[raw_column]
        is_exact = raw_column == canonical
        if canonical not in columns:
            columns[canonical] = values
        elif is_exact and canonical not in exact_canonical:
            columns[canonical] = values.combine_first(columns[canonical])
        else:
            columns[canonical] = columns[canonical].combine_first(values)
        if is_exact:
            exact_canonical.add(canonical)

    renamed = pd.DataFrame(columns, index=df.index)
    for column in CANONICAL_FEATURE_COLUMNS:
        if column not in renamed.columns:
            renamed[column] = np.nan
    for column in CANONICAL_FEATURE_COLUMNS:
        renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    return renamed


def canonical_feature_columns(df: pd.DataFrame) -> list[str]:
    canonical_df = canonicalize_dataframe(df)
    selected: list[str] = []
    for column in CANONICAL_FEATURE_COLUMNS:
        if column in canonical_df.columns and canonical_df[column].notna().any():
            selected.append(column)
    return selected


def feature_scale(values: Iterable[object]) -> float:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return 1.0
    if arr.size == 1:
        return max(abs(float(arr[0])) * 0.10, 1e-6)
    q10, q25, q50, q75, q90 = np.quantile(arr, [0.10, 0.25, 0.50, 0.75, 0.90])
    return max(
        float(q90 - q10),
        float(q75 - q25),
        float(np.std(arr)),
        abs(float(q50)) * 0.10,
        1e-6,
    )


def dataframe_feature_scales(df: pd.DataFrame) -> dict[str, float]:
    canonical_df = canonicalize_dataframe(df)
    return {column: feature_scale(canonical_df[column]) for column in canonical_feature_columns(canonical_df)}


def feature_vector(
    features: dict[str, object],
    feature_names: Iterable[str] | None = None,
) -> np.ndarray:
    names = tuple(feature_names or CANONICAL_FEATURE_COLUMNS)
    canonical = canonicalize_feature_dict({str(k): v for k, v in features.items()})
    return np.array([float(canonical.get(name, 0.0)) for name in names], dtype=np.float32)


def absolute_match_report(
    target_features: dict[str, object],
    candidate_features: dict[str, object],
    feature_scales: dict[str, float],
    feature_names: Iterable[str] | None = None,
) -> dict[str, object]:
    names = tuple(feature_names or CANONICAL_FEATURE_COLUMNS)
    target = canonicalize_feature_dict({str(k): v for k, v in target_features.items()})
    candidate = canonicalize_feature_dict({str(k): v for k, v in candidate_features.items()})

    rows: list[dict[str, object]] = []
    present = 0
    normalized_error_sum = 0.0
    weighted_error_sum = 0.0
    weights_sum = 0.0
    for name in names:
        if name not in target or name not in candidate:
            continue
        scale = max(float(feature_scales.get(name, 1.0)), 1e-9)
        delta = float(candidate[name]) - float(target[name])
        normalized_error = abs(delta) / scale
        present += 1
        normalized_error_sum += normalized_error
        constraint = feature_constraint(name)
        weight = constraint.importance_weight if constraint is not None else 1.0
        weighted_error_sum += normalized_error * weight
        weights_sum += weight
        rows.append(
            {
                "feature_name": name,
                "target_value": float(target[name]),
                "candidate_value": float(candidate[name]),
                "delta": delta,
                "abs_delta": abs(delta),
                "scale": scale,
                "normalized_abs_delta": normalized_error,
            }
        )

    rows.sort(key=lambda row: float(row["normalized_abs_delta"]), reverse=True)
    mean_error = normalized_error_sum / present if present else float("inf")
    weighted_mean_error = weighted_error_sum / weights_sum if weights_sum else float("inf")
    return {
        "present_feature_count": present,
        "feature_count": len(names),
        "mean_normalized_abs_error": mean_error,
        "weighted_mean_normalized_abs_error": weighted_mean_error,
        "feature_errors": rows,
    }


def row_from_feature_json(
    binary_name: str,
    binary_path: str,
    microbenchmark_name: str,
    command: str,
    feature_payload: dict[str, object],
    perf_stat_ok: bool = True,
    tma_ok: bool = True,
    intel_pt_ok: bool = True,
) -> dict[str, object]:
    row = {
        "binary": binary_name,
        "binary_path": binary_path,
        "microbenchmark": microbenchmark_name,
        "command": command,
        "status.perf_stat_ok": perf_stat_ok,
        "status.tma_ok": tma_ok,
        "status.intel_pt_ok": intel_pt_ok,
    }
    row.update(canonicalize_feature_dict(feature_payload))
    return row


def load_feature_table(xlsx_path: Path) -> pd.DataFrame:
    return canonicalize_dataframe(pd.read_excel(xlsx_path))
