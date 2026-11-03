from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import scann
from flask import Flask, jsonify, request

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.cloning_priority import evaluate_priority_constraints
from scripts.benchmark_sources import parse_benchmark_binary_sources
from scripts.feature_schema import (
    absolute_match_report,
    canonicalize_feature_dict,
    load_feature_table,
    resolve_feature_table_path,
)

app = Flask(__name__)

INDEX_DIR = REPO_ROOT / "search_engine" / "scann_index"
OUT_DIR = REPO_ROOT / "search_engine" / "search_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ADJUSTOR_URL = os.environ.get("ADJUSTOR_URL", "http://127.0.0.1:5000/adjustor")
_ADJUSTOR_TIMEOUT_SECONDS = float(
    os.environ.get("ADJUSTOR_TIMEOUT_SECONDS", os.environ.get("CLOUD_PIPELINE_TIMEOUT_SECONDS", "7200"))
)
ADJUSTOR_TIMEOUT_SECONDS = (
    None if _ADJUSTOR_TIMEOUT_SECONDS <= 0 else _ADJUSTOR_TIMEOUT_SECONDS
)
ADJUSTOR_TIMEOUT_SLACK_SECONDS = float(os.environ.get("ADJUSTOR_TIMEOUT_SLACK_SECONDS", "120"))
MIN_ADJUSTOR_WALL_SECONDS = float(os.environ.get("MIN_ADJUSTOR_WALL_SECONDS", "60"))
DEFAULT_XLSX = resolve_feature_table_path(REPO_ROOT / "search_engine" / "all_features.xlsx")
MATCH_ERROR_THRESHOLD = float(os.environ.get("MATCH_ERROR_THRESHOLD", "0.75"))
ADJUSTOR_RETRIEVAL_TOP_K = int(os.environ.get("ADJUSTOR_RETRIEVAL_TOP_K", "8"))
SEED_BASELINE_SELECTION_MODE = os.environ.get("SEED_BASELINE_SELECTION_MODE", "ipc_only").strip().lower()
FORCE_SEED_ID = os.environ.get("FORCE_SEED_ID", "").strip()
FORCE_SEED_BINARY = os.environ.get("FORCE_SEED_BINARY", "").strip()
IPC_FEATURE_NAME = "perf_stat.IPC"
IPC_NEAR_HW_FILTER_SIZE = int(os.environ.get("SEED_IPC_NEAR_HW_FILTER_SIZE", "32"))


def utc_compact() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(float(n), 1e-12)


def preprocess_query(q_raw: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    q = (q_raw - mu.reshape(-1)) / sigma.reshape(-1)
    return l2_normalize(q.astype(np.float32, copy=False))


searcher = scann.scann_ops_pybind.load_searcher(str(INDEX_DIR))
ids = np.load(INDEX_DIR / "ids.npy", allow_pickle=True)
feature_names = tuple(np.load(INDEX_DIR / "feature_names.npy", allow_pickle=True).tolist())
feature_scales = np.load(INDEX_DIR / "feature_scales.npy").reshape(-1)
mu = np.load(INDEX_DIR / "mu.npy")
sigma = np.load(INDEX_DIR / "sigma.npy")
X_raw = np.load(INDEX_DIR / "raw_features.npy")
binary_names = np.load(INDEX_DIR / "binary_names.npy", allow_pickle=True)
binary_paths = np.load(INDEX_DIR / "binary_paths.npy", allow_pickle=True)
supported_binary_names = frozenset(parse_benchmark_binary_sources())
supported_candidate_indices = np.array(
    [
        idx
        for idx, binary_name in enumerate(binary_names)
        if str(binary_name) in supported_binary_names
    ],
    dtype=np.int64,
)
if supported_candidate_indices.size == 0:
    raise RuntimeError("Search index contains no benchmarks with editable source mappings")
feature_df = load_feature_table(DEFAULT_XLSX)
feature_lookup = {
    (str(row["binary"]), str(row["microbenchmark"])): row.to_dict()
    for _, row in feature_df.iterrows()
}
scale_lookup = {name: float(feature_scales[idx]) for idx, name in enumerate(feature_names)}
NON_IPC_HW_FEATURE_NAMES = tuple(
    name for name in feature_names if str(name).startswith("perf_stat.") and name != IPC_FEATURE_NAME
)
ALL_HW_FEATURE_NAMES = tuple(
    name for name in feature_names if str(name).startswith("perf_stat.")
)


def row_feature_dict(raw_row: np.ndarray) -> dict[str, float]:
    return {feature_names[i]: float(raw_row[i]) for i in range(len(feature_names))}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _full_feature_similarity(q_raw: np.ndarray, candidate_idx: int) -> float:
    q = preprocess_query(q_raw, mu, sigma)
    candidate_vector = preprocess_query(X_raw[candidate_idx].astype(np.float32, copy=False), mu, sigma)
    return float(np.dot(candidate_vector, q))


def _best_supported_similarity(q_raw: np.ndarray) -> tuple[int, float]:
    q = preprocess_query(q_raw, mu, sigma)
    best_idx = -1
    best_score = float("-inf")
    for candidate_idx in supported_candidate_indices.tolist():
        candidate_vector = preprocess_query(
            X_raw[candidate_idx].astype(np.float32, copy=False),
            mu,
            sigma,
        )
        score = float(np.dot(candidate_vector, q))
        if score > best_score:
            best_idx = int(candidate_idx)
            best_score = score
    if best_idx < 0:
        raise ValueError("No editable benchmark candidates are available")
    return best_idx, best_score


def _non_ipc_hardware_error(
    canonical_target: dict[str, float],
    candidate_features: dict[str, float],
) -> float:
    if not NON_IPC_HW_FEATURE_NAMES:
        return float("inf")
    report = absolute_match_report(
        canonical_target,
        candidate_features,
        scale_lookup,
        feature_names=NON_IPC_HW_FEATURE_NAMES,
    )
    return float(report["weighted_mean_normalized_abs_error"])


def _choose_seed_baseline(
    canonical_target: dict[str, float],
    *,
    top_k: int,
) -> tuple[int, float, str]:
    q_raw = np.array([float(canonical_target.get(name, 0.0)) for name in feature_names], dtype=np.float32)
    if SEED_BASELINE_SELECTION_MODE == "ipc_only":
        if IPC_FEATURE_NAME not in feature_names or IPC_FEATURE_NAME not in canonical_target:
            chosen_idx, chosen_score = _best_supported_similarity(q_raw)
            return chosen_idx, chosen_score, "full_feature_supported_fallback_missing_ipc"
        ipc_idx = feature_names.index(IPC_FEATURE_NAME)
        target_ipc = float(canonical_target[IPC_FEATURE_NAME])
        ipc_values = X_raw[:, ipc_idx].astype(np.float32, copy=False)
        ipc_abs_delta = np.abs(ipc_values - target_ipc)
        supported_deltas = ipc_abs_delta[supported_candidate_indices]
        min_abs_delta = float(np.min(supported_deltas))
        candidate_idxs = supported_candidate_indices[
            np.isclose(supported_deltas, min_abs_delta, rtol=0.0, atol=1e-12)
        ]
        if candidate_idxs.size == 1:
            chosen_idx = int(candidate_idxs[0])
        else:
            tie_break_q = preprocess_query(q_raw, mu, sigma)
            candidate_vectors = X_raw[candidate_idxs].astype(np.float32, copy=False)
            candidate_scores = np.matmul(candidate_vectors, tie_break_q)
            chosen_idx = int(candidate_idxs[int(np.argmax(candidate_scores))])
        return chosen_idx, -min_abs_delta, "ipc_only"
    if SEED_BASELINE_SELECTION_MODE in {"ipc_near_non_ipc_hw", "ipc_filter_non_ipc_hw", "ipc_filter_hw"}:
        if IPC_FEATURE_NAME not in feature_names or IPC_FEATURE_NAME not in canonical_target:
            chosen_idx, chosen_score = _best_supported_similarity(q_raw)
            return chosen_idx, chosen_score, "full_feature_supported_fallback_missing_ipc"
        if not NON_IPC_HW_FEATURE_NAMES:
            chosen_idx, chosen_score = _best_supported_similarity(q_raw)
            return chosen_idx, chosen_score, "full_feature_supported_fallback_missing_non_ipc_hw"

        ipc_idx = feature_names.index(IPC_FEATURE_NAME)
        target_ipc = float(canonical_target[IPC_FEATURE_NAME])
        ipc_values = X_raw[:, ipc_idx].astype(np.float32, copy=False)
        ipc_abs_delta = np.abs(ipc_values - target_ipc)
        candidate_order = supported_candidate_indices[
            np.argsort(ipc_abs_delta[supported_candidate_indices], kind="stable")
        ]
        filter_size = min(len(candidate_order), max(top_k, IPC_NEAR_HW_FILTER_SIZE))
        candidate_idxs = candidate_order[:filter_size]

        best_idx: int | None = None
        best_key: tuple[float, float, float, int] | None = None
        best_error = float("inf")
        for raw_idx in candidate_idxs.tolist():
            candidate_idx = int(raw_idx)
            candidate_features = row_feature_dict(X_raw[candidate_idx].astype(np.float32, copy=False))
            non_ipc_hw_error = _non_ipc_hardware_error(canonical_target, candidate_features)
            ipc_delta = float(ipc_abs_delta[candidate_idx])
            similarity = _full_feature_similarity(q_raw, candidate_idx)
            selection_key = (non_ipc_hw_error, ipc_delta, -similarity, candidate_idx)
            if best_key is None or selection_key < best_key:
                best_key = selection_key
                best_idx = candidate_idx
                best_error = non_ipc_hw_error
        if best_idx is None:
            raise ValueError("IPC-filtered seed selection returned no candidates")
        return best_idx, -best_error, "ipc_near_non_ipc_hw"

    if SEED_BASELINE_SELECTION_MODE == "weighted_hw_nearest":
        if not ALL_HW_FEATURE_NAMES:
            chosen_idx, chosen_score = _best_supported_similarity(q_raw)
            return chosen_idx, chosen_score, "full_feature_supported_fallback_no_hw"
        _MALLEABILITY_WEIGHTS: dict[str, float] = {
            "perf_stat.LLC-load-misses_MPKI": 2.5,
            "perf_stat.L1-dcache-load-misses_MPKI": 1.5,
            "perf_stat.L1-dcache-loads_LPKI": 1.2,
            "perf_stat.L1-dcache-stores_SPKI": 1.2,
            "perf_stat.iTLB-load-misses_MPKI": 3.0,
            "perf_stat.L1-icache-load-misses_MPKI": 1.5,
            "perf_stat.branch-misses_MPKI": 0.7,
            "perf_stat.branch-load-misses_MPKI": 0.6,
            "perf_stat.IPC": 0.5,
        }
        scale_floor_ratio = 0.50
        floored_scales = {}
        for name in ALL_HW_FEATURE_NAMES:
            db_scale = scale_lookup.get(name, 1e-6)
            target_val = float(canonical_target.get(name, 0.0))
            floored_scales[name] = max(db_scale, abs(target_val) * scale_floor_ratio)
        best_idx: int | None = None
        best_error = float("inf")
        for candidate_idx in supported_candidate_indices.tolist():
            candidate_features = row_feature_dict(X_raw[candidate_idx].astype(np.float32, copy=False))
            w_err_sum = 0.0
            w_sum = 0.0
            for name in ALL_HW_FEATURE_NAMES:
                if name not in canonical_target or name not in candidate_features:
                    continue
                scale = max(floored_scales.get(name, 1e-6), 1e-9)
                delta = abs(float(candidate_features[name]) - float(canonical_target[name]))
                w = _MALLEABILITY_WEIGHTS.get(name, 1.0)
                w_err_sum += (delta / scale) * w
                w_sum += w
            hw_error = w_err_sum / w_sum if w_sum > 0 else float("inf")
            if hw_error < best_error:
                best_error = hw_error
                best_idx = candidate_idx
        if best_idx is None:
            raise ValueError("weighted_hw_nearest: no candidates found")
        return best_idx, -best_error, "weighted_hw_nearest"

    chosen_idx, chosen_score = _best_supported_similarity(q_raw)
    return chosen_idx, chosen_score, "full_feature_supported"


@app.get("/healthz")
def healthz():
    return jsonify({
        "status": "ok",
        "seed_baseline_selection_mode": SEED_BASELINE_SELECTION_MODE,
    })


@app.post("/search")
def search():
    request_started_at = time.monotonic()
    data = request.get_json(force=True)
    run_id = data.get("run_id", "unknown")
    q_in = data.get("query_json")
    k = max(1, int(data.get("k", 1)))
    retrieval_top_k = max(1, int(data.get("retrieval_top_k", ADJUSTOR_RETRIEVAL_TOP_K)))
    allow_adjust = bool(data.get("allow_adjust", True))
    top_k = max(k, 1)

    if q_in is None:
        return jsonify({"error": "missing 'query_json'"}), 400
    if isinstance(q_in, dict):
        query_features = q_in
    else:
        try:
            query_features = json.loads(q_in)
        except Exception as exc:
            return jsonify({"error": f"query_json is not valid JSON: {exc}"}), 400

    canonical_target = canonicalize_feature_dict(query_features)
    try:
        chosen_idx, chosen_score, seed_selection_mode = _choose_seed_baseline(
            canonical_target,
            top_k=top_k,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if FORCE_SEED_ID:
        forced_indices = [
            i
            for i in range(len(ids))
            if str(ids[i]) == FORCE_SEED_ID
            and (not FORCE_SEED_BINARY or str(binary_names[i]) == FORCE_SEED_BINARY)
            and str(binary_names[i]) in supported_binary_names
        ]
        if forced_indices:
            chosen_idx = forced_indices[0]
            q_raw_force = np.array(
                [float(canonical_target.get(name, 0.0)) for name in feature_names],
                dtype=np.float32,
            )
            chosen_score = _full_feature_similarity(q_raw_force, chosen_idx)
            seed_selection_mode = f"forced:{FORCE_SEED_ID}"

    chosen_id = str(ids[chosen_idx])
    chosen_binary = str(binary_names[chosen_idx])
    chosen_binary_path = str(binary_paths[chosen_idx])
    nearest_features = row_feature_dict(X_raw[chosen_idx].astype(np.float32, copy=False))

    baseline_report = absolute_match_report(canonical_target, nearest_features, scale_lookup)
    baseline_error = float(baseline_report["weighted_mean_normalized_abs_error"])
    baseline_constraint_report = evaluate_priority_constraints(canonical_target, nearest_features)
    baseline_ok = (
        bool(baseline_constraint_report.get("all_within_tolerance"))
        if int(baseline_constraint_report.get("checked_feature_count", 0)) > 0
        else baseline_error <= MATCH_ERROR_THRESHOLD
    )

    stamp = utc_compact()
    result_path = OUT_DIR / f"{run_id}_{stamp}_match.json"
    result_payload = {
        "run_id": run_id,
        "nearest_id": chosen_id,
        "nearest_binary": chosen_binary,
        "nearest_binary_path": chosen_binary_path,
        "nearest_features": nearest_features,
        "retrieval_score": chosen_score,
        "seed_selection_mode": seed_selection_mode,
        "baseline_error": baseline_error,
        "baseline_success": baseline_ok,
        "baseline_match_report": baseline_report,
        "baseline_constraint_report": baseline_constraint_report,
        "threshold_met": baseline_ok,
    }
    write_json(result_path, result_payload)

    adjustor_response = None
    if allow_adjust and not baseline_ok:
        elapsed_s = max(0.0, time.monotonic() - request_started_at)
        # Honor per-segment wall budget from client if provided; otherwise
        # fall back to the global ADJUSTOR_TIMEOUT_SECONDS.
        client_max_wall_s = data.get("max_wall_seconds")
        effective_timeout_s = ADJUSTOR_TIMEOUT_SECONDS
        if client_max_wall_s is not None:
            try:
                client_max_wall_s = float(client_max_wall_s)
                if client_max_wall_s > 0:
                    effective_timeout_s = (
                        client_max_wall_s
                        if effective_timeout_s is None
                        else min(effective_timeout_s, client_max_wall_s)
                    )
            except (TypeError, ValueError):
                pass
        request_budget_remaining_s = (
            None
            if effective_timeout_s is None
            else max(0.0, effective_timeout_s - elapsed_s)
        )
        adjustor_wall_seconds = (
            None
            if request_budget_remaining_s is None
            else max(0.0, request_budget_remaining_s - ADJUSTOR_TIMEOUT_SLACK_SECONDS)
        )
        payload = {
            "run_id": run_id,
            "nearest_id": chosen_id,
            "binary_name": chosen_binary,
            "binary_path": chosen_binary_path,
            "query_json": canonical_target,
            "baseline_error": baseline_error,
            "baseline_match_report": baseline_report,
            "baseline_constraint_report": baseline_constraint_report,
            "top_k": top_k,
            "retrieval_top_k": retrieval_top_k,
        }
        # Forward cross-segment warm-start hints to the adjustor.
        if data.get("warm_start_file"):
            payload["warm_start_file"] = data["warm_start_file"]
        if data.get("warm_start_features"):
            payload["warm_start_features"] = data["warm_start_features"]
        if (
            adjustor_wall_seconds is not None
            and adjustor_wall_seconds < MIN_ADJUSTOR_WALL_SECONDS
        ):
            adjustor_response = {
                "success": False,
                "stage": "deadline",
                "reason": (
                    "Skipped adjustor: insufficient request budget remained after baseline matching "
                    f"({request_budget_remaining_s:.1f}s left, {ADJUSTOR_TIMEOUT_SLACK_SECONDS:.1f}s reserved as slack)."
                ),
                "max_wall_seconds": 0.0,
                "request_budget_remaining_seconds": request_budget_remaining_s,
                "timeout_reserved_seconds": ADJUSTOR_TIMEOUT_SLACK_SECONDS,
            }
        else:
            if adjustor_wall_seconds is not None:
                payload["max_wall_seconds"] = adjustor_wall_seconds
            try:
                resp = requests.post(
                    ADJUSTOR_URL,
                    json=payload,
                    timeout=(
                        None
                        if request_budget_remaining_s is None
                        else max(1.0, request_budget_remaining_s)
                    ),
                )
                resp.raise_for_status()
                adjustor_response = resp.json()
            except Exception as exc:
                # Preserve the nearest-match result even when the slower adjustor stage misses
                # its budget. Downstream code can still score against nearest_features.
                adjustor_response = {
                    "success": False,
                    "stage": "adjustor_request",
                    "error": f"failed POST to adjustor: {exc}",
                    "max_wall_seconds": adjustor_wall_seconds,
                    "request_budget_remaining_seconds": request_budget_remaining_s,
                    "timeout_reserved_seconds": ADJUSTOR_TIMEOUT_SLACK_SECONDS,
                }

    return jsonify(
        {
            **result_payload,
            "status": "ok",
            "match_path": str(result_path),
            "adjustor_response": adjustor_response,
        }
    )


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5001"))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    print(f"Seed baseline selection mode: {SEED_BASELINE_SELECTION_MODE}", flush=True)
    app.run(host=host, port=port, debug=debug, use_reloader=False)
