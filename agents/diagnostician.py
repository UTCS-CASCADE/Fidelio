from __future__ import annotations

import os
from typing import Any

from cloning_priority import (
    evaluate_priority_constraints,
    first_unsatisfied_priority,
    priority_tier_label,
    priority_tier_state,
    tolerance_bounds,
)
from feature_prompt_guidance import feature_prompt_guidance
from feature_schema import canonicalize_feature_dict, feature_display_name


FRONTEND_PLAYBOOK_FEATURES: tuple[str, ...] = (
    "perf_stat.L1-icache-load-misses_MPKI",
    "perf_stat.branch-misses_MPKI",
    "perf_stat.branch-load-misses_MPKI",
    "perf_stat.iTLB-load-misses_MPKI",
)
MEMORY_PLAYBOOK_FEATURES: tuple[str, ...] = (
    "perf_stat.L1-dcache-load-misses_MPKI",
    "perf_stat.L1-dcache-loads_LPKI",
    "perf_stat.L1-dcache-stores_SPKI",
)
TOPDOWN_FRONTEND_FEATURES: tuple[str, ...] = (
    "tma.TMA_Frontend_Bound(%)",
    "tma.TMA_Bad_Speculation(%)",
)
TOPDOWN_BACKEND_FEATURES: tuple[str, ...] = ("tma.TMA_Backend_Bound(%)",)
STRUCTURAL_EDIT_OVERFLOW_THRESHOLD = 0.35


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _priority_view(
    feature_rows: list[dict[str, Any]],
    *,
    locked_priority_rank: int | None = None,
) -> dict[str, Any]:
    if locked_priority_rank is None:
        tier_state = priority_tier_state(feature_rows)
        tier_state["guardrail_rows"] = list(tier_state.get("higher_priority_matched_rows") or [])
        tier_state["guardrail_violated_rows"] = []
        return tier_state

    active_priority_rank = int(locked_priority_rank)
    relevant_rows = [
        row
        for row in feature_rows
        if int(row.get("priority_rank", 999) or 999) < 999
    ]
    guardrail_rows = _sort_rows(
        [
            row
            for row in relevant_rows
            if int(row.get("priority_rank", 999) or 999) < active_priority_rank
        ]
    )
    active_rows = _sort_rows(
        [
            row
            for row in relevant_rows
            if int(row.get("priority_rank", 999) or 999) == active_priority_rank
        ]
    )
    deferred_rows = _sort_rows(
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
        "guardrail_violated_rows": [row for row in guardrail_rows if not bool(row.get("within_tolerance"))],
        "higher_priority_matched_rows": [row for row in guardrail_rows if bool(row.get("within_tolerance"))],
        "active_rows": active_rows,
        "active_focus_rows": [row for row in active_rows if not bool(row.get("within_tolerance"))],
        "active_protected_rows": [row for row in active_rows if bool(row.get("within_tolerance"))],
        "deferred_rows": deferred_rows,
    }


def _resolve_active_priority_rank(
    feature_rows: list[dict[str, Any]],
    objective_priority_rank: int | None,
) -> int | None:
    if objective_priority_rank is not None:
        locked_rows = [
            row
            for row in feature_rows
            if int(row.get("priority_rank", 999) or 999) == int(objective_priority_rank)
        ]
        if any(not bool(row.get("within_tolerance")) for row in locked_rows):
            return int(objective_priority_rank)
    return first_unsatisfied_priority(feature_rows)


def _format_metric_value(value: Any) -> str:
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if numeric == 0.0:
        return "0"
    magnitude = abs(numeric)
    if magnitude >= 1000 or magnitude < 0.001:
        return f"{numeric:.6g}"
    return f"{numeric:.4f}".rstrip("0").rstrip(".")


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


def _acceptable_range(row: dict[str, Any]) -> str:
    target_value = row.get("target_value")
    tolerance_ratio = row.get("tolerance_ratio")
    if tolerance_ratio is None:
        return f"target={_format_metric_value(target_value)}"
    try:
        acceptable_min, acceptable_max = tolerance_bounds(
            float(target_value),
            float(tolerance_ratio),
        )
    except Exception:
        return f"target={_format_metric_value(target_value)}"
    return (
        f"target={_format_metric_value(target_value)}, "
        f"acceptable={_format_metric_value(acceptable_min)}..{_format_metric_value(acceptable_max)}"
    )


def _row_score(row: dict[str, Any]) -> float:
    return float(row.get("overflow_ratio", 0.0) or 0.0) * float(
        row.get("importance_weight", 1.0) or 1.0
    )


def _rows_by_name(feature_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("feature_name", "")).strip(): row
        for row in feature_rows
        if str(row.get("feature_name", "")).strip()
    }


def _best_candidate_row(
    rows_by_name: dict[str, dict[str, Any]],
    feature_names: tuple[str, ...],
    *,
    desired_status: str,
) -> dict[str, Any] | None:
    candidates = [
        row
        for feature_name in feature_names
        for row in [rows_by_name.get(feature_name)]
        if row and _metric_status(row) == desired_status
    ]
    if not candidates:
        return None
    return max(candidates, key=_row_score)


def _ipc_decomposition(
    ipc_row: dict[str, Any],
    *,
    rows_by_name: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    status = _metric_status(ipc_row)
    if status == "within range":
        return ipc_row, "IPC is already within range; keep it as a guardrail instead of making it the primary edit target."

    if status == "too high":
        desired_status = "too low"
        frontend_clause = "frontend-bound"
        backend_clause = "backend-bound"
    else:
        desired_status = "too high"
        frontend_clause = "frontend-stall"
        backend_clause = "backend-stall"

    frontend_row = _best_candidate_row(rows_by_name, FRONTEND_PLAYBOOK_FEATURES, desired_status=desired_status)
    backend_row = _best_candidate_row(rows_by_name, MEMORY_PLAYBOOK_FEATURES, desired_status=desired_status)
    topdown_front = _best_candidate_row(rows_by_name, TOPDOWN_FRONTEND_FEATURES, desired_status=desired_status)
    topdown_back = _best_candidate_row(rows_by_name, TOPDOWN_BACKEND_FEATURES, desired_status=desired_status)

    if topdown_front and frontend_row:
        return (
            frontend_row,
            f"The IPC gap looks {frontend_clause}, so delegate this attempt to the {feature_display_name(str(frontend_row.get('feature_name', '')))} playbook.",
        )
    if topdown_back and backend_row:
        return (
            backend_row,
            f"The IPC gap looks {backend_clause}, so delegate this attempt to the {feature_display_name(str(backend_row.get('feature_name', '')))} playbook.",
        )
    if frontend_row and backend_row:
        if _row_score(frontend_row) >= _row_score(backend_row):
            return (
                frontend_row,
                f"IPC is the visible symptom, but the stronger remaining mismatch is on the frontend side, so delegate to {feature_display_name(str(frontend_row.get('feature_name', '')))} first.",
            )
        return (
            backend_row,
            f"IPC is the visible symptom, but the stronger remaining mismatch is on the backend/data side, so delegate to {feature_display_name(str(backend_row.get('feature_name', '')))} first.",
        )
    if frontend_row:
        return (
            frontend_row,
            f"IPC is the visible symptom; the cleanest lever is the {feature_display_name(str(frontend_row.get('feature_name', '')))} playbook.",
        )
    if backend_row:
        return (
            backend_row,
            f"IPC is the visible symptom; the cleanest lever is the {feature_display_name(str(backend_row.get('feature_name', '')))} playbook.",
        )
    return (
        ipc_row,
        "IPC is still the dominant mismatch, but no stronger component counter is available; use the smallest local IPC correction and preserve the current cache/branch shape.",
    )


def _adapt_for_generation_mode(text: str, generation_strategy: str) -> str:
    rendered = str(text or "").strip()
    if generation_strategy != "from_scratch" or not rendered:
        return rendered
    replacements = [
        ("existing hot path", "new benchmark hot path"),
        ("the hot path", "the new benchmark hot path"),
        ("existing benchmark", "source-family benchmark"),
        ("current benchmark", "new benchmark"),
    ]
    for old, new in replacements:
        rendered = rendered.replace(old, new)
    return rendered


def _playbook_name(feature_name: str) -> str:
    mapping = {
        "perf_stat.L1-icache-load-misses_MPKI": "I-cache counter-matching playbook",
        "perf_stat.branch-misses_MPKI": "branch-predictor entropy playbook",
        "perf_stat.branch-load-misses_MPKI": "branch-target footprint playbook",
        "perf_stat.L1-dcache-load-misses_MPKI": "D-cache locality playbook",
        "perf_stat.L1-dcache-loads_LPKI": "D-cache load-traffic playbook",
        "perf_stat.L1-dcache-stores_SPKI": "D-cache store-traffic playbook",
        "perf_stat.iTLB-load-misses_MPKI": "iTLB footprint playbook",
        "perf_stat.IPC": "IPC correction playbook",
    }
    return mapping.get(feature_name, f"{feature_display_name(feature_name)} playbook")


# Short, human-facing metric labels used when phrasing an optimization's
# expected-effect directive (e.g. "This reduces L1i MPKI").
_EFFECT_METRIC_LABEL: dict[str, str] = {
    "perf_stat.L1-icache-load-misses_MPKI": "L1i MPKI",
    "perf_stat.branch-misses_MPKI": "branch-miss MPKI",
    "perf_stat.branch-load-misses_MPKI": "branch-target MPKI",
    "perf_stat.L1-dcache-load-misses_MPKI": "L1d-miss MPKI",
    "perf_stat.L1-dcache-loads_LPKI": "L1d-load LPKI",
    "perf_stat.L1-dcache-stores_SPKI": "L1d-store SPKI",
    "perf_stat.iTLB-load-misses_MPKI": "iTLB MPKI",
    "perf_stat.LLC-load-misses_MPKI": "LLC-miss MPKI",
    "perf_stat.IPC": "IPC",
}

# Directive verb -> the metric movement the deciding script must confirm.
EFFECT_VERB_DIRECTION: dict[str, str] = {
    "reduces": "decrease",
    "lowers": "decrease",
    "raises": "increase",
    "increases": "increase",
    "holds": "hold",
}


def _effect_metric_label(feature_name: str) -> str:
    label = _EFFECT_METRIC_LABEL.get(feature_name)
    if label:
        return label
    return feature_display_name(feature_name)


def expected_effect_directive(feature_name: str, status: str) -> str:
    """Return the spec directive describing what the prescribed optimization
    is expected to do to the diagnosed counter.

    ``status`` is the diagnosed metric status ("too low" / "too high" /
    "within range"). A "too low" counter is raised, a "too high" counter is
    reduced. The phrasing is deliberately terse so it can be embedded directly
    in the spec/prompt (e.g. "This reduces L1i MPKI") and later parsed by the
    deciding script via :data:`EFFECT_VERB_DIRECTION`.
    """
    label = _effect_metric_label(feature_name)
    if status == "too low":
        return f"This raises {label}."
    if status == "too high":
        return f"This reduces {label}."
    return f"This holds {label} steady."


def expected_effect_direction(directive: str) -> str:
    """Parse a spec directive (e.g. "This reduces L1i MPKI") into the metric
    movement the deciding script should observe: "decrease", "increase",
    "hold", or "unknown"."""
    text = str(directive or "").lower()
    for verb, direction in EFFECT_VERB_DIRECTION.items():
        if verb in text:
            return direction
    return "unknown"


def _observed_direction(before: float, after: float, *, rel_eps: float = 0.02) -> str:
    """Classify the measured movement of a counter across an optimization.

    A change smaller than ``rel_eps`` (relative to the before magnitude, with a
    tiny absolute floor) is treated as "hold" so measurement noise is not
    mistaken for a real effect."""
    delta = after - before
    floor = max(abs(before) * rel_eps, 1e-9)
    if delta > floor:
        return "increase"
    if delta < -floor:
        return "decrease"
    return "hold"


def validate_playbook_effects(
    records: list[dict[str, Any]],
    *,
    rel_eps: float = 0.02,
) -> dict[str, Any]:
    """Confirm that prescribed optimizations actually moved their counters the
    way the spec directive claimed, across a fleet of measured benchmarks.

    Each record describes one applied optimization on one benchmark and must
    provide:

    - ``benchmark``: fleet member / segment name (str).
    - ``feature_name``: the diagnosed counter the optimization targeted.
    - ``before_value`` / ``after_value``: the measured counter value before and
      after applying the optimization.
    - one of ``directive`` (the spec text) or ``status`` (the diagnosed
      status, used to regenerate the directive).

    Returns a report with one entry per record plus fleet-wide aggregates. An
    optimization "confirms" when the observed counter movement matches the
    direction claimed by its directive (a directive that claims no movement,
    "hold", confirms when the counter stays within ``rel_eps``).
    """
    results: list[dict[str, Any]] = []
    confirmed = 0
    evaluable = 0
    for record in records:
        feature_name = str(record.get("feature_name", "")).strip()
        directive = record.get("directive")
        status = str(record.get("status", "")).strip()
        if not directive and feature_name:
            directive = expected_effect_directive(feature_name, status)
        expected = expected_effect_direction(str(directive or ""))
        entry: dict[str, Any] = {
            "benchmark": str(record.get("benchmark", "")).strip(),
            "feature_name": feature_name,
            "feature_label": _effect_metric_label(feature_name) if feature_name else "",
            "directive": str(directive or "").strip(),
            "expected_direction": expected,
        }
        try:
            before = float(record.get("before_value"))
            after = float(record.get("after_value"))
        except Exception:
            entry.update(
                {
                    "before_value": record.get("before_value"),
                    "after_value": record.get("after_value"),
                    "observed_direction": "unknown",
                    "confirmed": False,
                    "evaluable": False,
                    "note": "before/after values not numeric",
                }
            )
            results.append(entry)
            continue
        observed = _observed_direction(before, after, rel_eps=rel_eps)
        is_evaluable = expected in ("increase", "decrease", "hold")
        is_confirmed = is_evaluable and observed == expected
        if is_evaluable:
            evaluable += 1
        if is_confirmed:
            confirmed += 1
        entry.update(
            {
                "before_value": before,
                "after_value": after,
                "delta": after - before,
                "observed_direction": observed,
                "confirmed": is_confirmed,
                "evaluable": is_evaluable,
                "note": "" if is_evaluable else "directive has no parseable effect direction",
            }
        )
        results.append(entry)

    by_feature: dict[str, dict[str, int]] = {}
    for entry in results:
        if not entry.get("evaluable"):
            continue
        feat = entry["feature_name"]
        bucket = by_feature.setdefault(feat, {"confirmed": 0, "total": 0})
        bucket["total"] += 1
        if entry["confirmed"]:
            bucket["confirmed"] += 1

    return {
        "records": results,
        "total": len(results),
        "evaluable": evaluable,
        "confirmed": confirmed,
        "violations": [e for e in results if e.get("evaluable") and not e.get("confirmed")],
        "fleet_confirm_rate": (confirmed / evaluable) if evaluable else 0.0,
        "all_confirmed": evaluable > 0 and confirmed == evaluable,
        "by_feature": by_feature,
    }


def _playbook_payload(
    *,
    feature_name: str,
    status: str,
) -> dict[str, Any]:
    if feature_name == "perf_stat.L1-icache-load-misses_MPKI":
        if status == "too low":
            return {
                "root_cause": (
                    "insufficient hot code footprint or excessive temporal reuse is keeping the executed instruction working set too compact"
                ),
                "structural_actions": [
                    "PROVEN PATTERN: Create 3 __attribute__((noinline)) functions each containing a 256-case switch with N ALU ops per case (add/xor/mul on a local accumulator). Rotate among them with j%3 in the hot loop. The switch index should be payload & 0xFF from a pointer-chase load.",
                    "L1i MPKI scales approximately linearly with ALU ops per switch case: 8 ops/case ~ 11 MPKI, 10 ~ 10 MPKI, 12 ~ 12 MPKI, 14 ~ 15 MPKI. Adjust ops-per-case to reach target L1i MPKI.",
                    "Each switch case must use unique literal constants to prevent compiler deduplication. The 3-function rotation forces icache thrashing across a massive code footprint.",
                ],
                "parameter_actions": [
                    "Tune ALU ops-per-case (the primary L1i knob) and iteration count (controls what fraction of instructions come from the switch loop vs framework overhead).",
                ],
            }
        return {
            "root_cause": "the hot instruction footprint is too fragmented and execution fans out across too many code bodies",
            "structural_actions": [
                "Consolidate helper bodies and concentrate execution on fewer hot paths.",
                "Inline or merge the hottest helper sequences where feasible.",
                "Shrink oversized hot basic blocks and remove excess alternate paths.",
            ],
            "parameter_actions": [
                "Tune helper count and block size downward before rewriting the benchmark again.",
            ],
        }

    if feature_name == "perf_stat.branch-misses_MPKI":
        if status == "too low":
            return {
                "root_cause": "branch entropy is too low and the predictor is learning the hot path too easily",
                "structural_actions": [
                    "PROVEN PATTERN: Use 3 noinline 256-case switch functions rotated with j%3. The indirect branch for each switch is unpredictable because the switch index depends on a cache-missing pointer-chase load value (payload & 0xFF). This defeats TAGE prediction.",
                    "COMPACT ALTERNATIVE — COLLATZ BRANCHES: For branch misses without expanding code footprint, use Collatz iteration: `if (x & 1) { x = 3*x + 1; } else { x >>= 1; }`. This generates ~35% misprediction rate in a tiny code footprint (~2 instructions per branch). Insert N_COLLATZ iterations per loop iteration to control branch-miss MPKI. Seed with a pointer-chase load value for unpredictability. Proven: 75 Collatz iterations per loop iteration produces reliable branch misses with minimal icache impact.",
                    "The same 3-switch structure that generates L1i pressure also generates branch misses via indirect-branch misprediction. If branch misses need to increase independently of L1i, increase the hot-chain size or ensure the chase cycle length exceeds 32K to prevent TAGE learning.",
                    "AVOID using simple if/else branches driven by register-computed state; the CPU resolves them speculatively before they mispreddict. Only branches gated on high-latency load values reliably cause mispredictions.",
                ],
                "parameter_actions": [
                    "Tune N_COLLATZ (Collatz iterations per loop) or iteration count and hot-chain size to control the ratio of branch misses per kilo-instruction.",
                ],
            }
        return {
            "root_cause": "the hot path has too much branch entropy or too many unpredictable decisions",
            "structural_actions": [
                "Regularize the hottest branch outcomes and bias a single common path.",
                "Convert the most unpredictable hot branch to a conditional move or arithmetic select when practical.",
                "Remove unnecessary branchy side paths from the timed loop.",
            ],
            "parameter_actions": [
                "Tune the remaining branch frequency or bias instead of adding another structural rewrite.",
            ],
        }

    if feature_name == "perf_stat.branch-load-misses_MPKI":
        if status == "too low":
            return {
                "root_cause": "the hot branch-target footprint is still too compact and the BTB/front-end target structures are under-stressed",
                "structural_actions": [
                    "Add one or two additional hot branch targets or dispatch alternatives that execute repeatedly in steady state.",
                    "Spread hot branch sites across a slightly broader target set without building a large dispatcher tree.",
                    "Use lightweight state-driven target selection so the branch-target working set actually expands.",
                ],
                "parameter_actions": [
                    "Tune target count or dispatch frequency after the branch-target gap becomes small.",
                ],
            }
        return {
            "root_cause": "the hot branch-target working set is too large or too chaotic",
            "structural_actions": [
                "Collapse the hot branch target set onto fewer sites or call targets.",
                "Inline or merge excess dispatch paths so execution bounces among fewer targets.",
                "Remove one layer of hot dispatch indirection if it is no longer needed.",
            ],
            "parameter_actions": [
                "Tune target count downward before attempting another structural rewrite.",
            ],
        }

    if feature_name == "perf_stat.L1-dcache-load-misses_MPKI":
        if status == "too low":
            return {
                "root_cause": "the hot data footprint is too small or reuse distance is too short relative to L1D capacity",
                "structural_actions": [
                    "PROVEN PATTERN: Use a linked-list pointer-chase through a 1 MB hot chain (16384 nodes x 64 bytes, Fisher-Yates shuffled). This exceeds L1d (48 KB) but fits in L2 (2 MB), generating reliable L1d misses without LLC misses. struct ChainNode { ChainNode* next; char pad[48]; }.",
                    "WORKING SET vs CACHE SIZE MODEL: L1d miss rate is controlled by working-set-to-L1d ratio. Chain that fits L1d (≤48KB) → 0 MPKI. Chain at 1MB (21× L1d) → most accesses miss L1d but hit L2. A 'medium' chain at ~128KB (2.7× L1d, fits L2) gives moderate L1d misses. Use multiple independent medium-chain walkers to multiply miss rate without growing working set beyond L2.",
                    "For additional L1d pressure without serializing the main loop, add 2-3 independent hot-chain walkers that each maintain their own position and advance one step per iteration. Each walker multiplies L1d miss rate. CAUTION: walkers dilute L1i and branch-miss MPKI.",
                    "Do NOT use hash-based or LCG-based random array indexing; hardware prefetchers learn those patterns. Only pointer-chasing (address of each load depends on VALUE of previous load) defeats the prefetcher.",
                ],
                "parameter_actions": [
                    "Tune hot-chain node count (controls working set size relative to L1d/L2), number of independent walkers (2-3 proven optimal), and iteration count (controls MPKI dilution ratio).",
                ],
            }
        return {
            "root_cause": "locality is too weak and the hot working set is too scattered for the target",
            "structural_actions": [
                "Shrink the hot data footprint so it fits more comfortably in L1D.",
                "Shorten stride and collapse one level of indirection or scattering.",
                "Concentrate repeated accesses onto fewer cache lines in the timed loop.",
            ],
            "parameter_actions": [
                "Tune stride and footprint downward instead of applying another broad locality rewrite.",
            ],
        }

    if feature_name == "perf_stat.L1-dcache-loads_LPKI":
        if status == "too low":
            return {
                "root_cause": "the hot path is issuing too few memory reads per unit work",
                "structural_actions": [
                    "PROVEN PATTERN — INLINE ASM LOADS: Use `asm volatile(\"movq (%0), %0\" : \"+r\"(pos))` in a loop. Each inline asm load is exactly 1 instruction producing exactly 1 L1-dcache-load, giving a precise 1:1 load-per-instruction ratio. N chase loads in a loop body = N loads per (N + overhead) instructions. CRITICAL: loop the asm (not unroll) to keep code footprint under ~2KB.",
                    "INSTRUCTION DILUTION TRAP: LPKI = loads / (total_insn / 1000). Adding ANY non-load instructions dilutes LPKI. To increase LPKI, add more load instructions per iteration rather than removing other instructions — removing non-loads risks destabilizing other counters.",
                    "Add compact extra hot-path reads from memory-resident state instead of reusing everything in registers.",
                    "Keep the added read traffic inside the timed steady-state loop.",
                ],
                "parameter_actions": [
                    "Tune number of inline asm chase loads per loop iteration (N_CHASE_PER_LOOP) — this is the primary LPKI knob. Also tune iteration count to control dilution from framework overhead.",
                ],
            }
        return {
            "root_cause": "the hot path is performing too many reads and not reusing loaded values enough",
            "structural_actions": [
                "Hoist or reuse repeatedly loaded values so fewer memory reads occur per unit work.",
                "Collapse redundant reads onto a smaller hot state footprint.",
                "Keep more intermediate values live in registers before reloading memory.",
            ],
            "parameter_actions": [
                "Tune reload frequency downward without changing the broader benchmark shape again.",
            ],
        }

    if feature_name == "perf_stat.L1-dcache-stores_SPKI":
        if status == "too low":
            return {
                "root_cause": "the hot path is materializing too little state back to memory",
                "structural_actions": [
                    "PROVEN PATTERN — INLINE ASM STORES: Use `asm volatile(\"movq %0, (%1)\" : : \"r\"(val), \"r\"(addr) : \"memory\")` in a loop. Each inline asm store is exactly 1 instruction producing exactly 1 L1-dcache-store. Pair with a 64-byte aligned scratch buffer to keep stores in L1d. N stores in a loop body gives precise SPKI control.",
                    "INSTRUCTION DILUTION TRAP: SPKI = stores / (total_insn / 1000). Adding non-store instructions dilutes SPKI. To increase SPKI, add more store instructions per iteration. Be aware that each store also adds 1 instruction, so net SPKI gain is (new_stores - store_fraction × new_insns) / total_insns × 1000.",
                    "Materialize one or two temporaries into a memory-resident structure during steady-state execution.",
                    "Keep the extra stores inside the timed loop rather than in one-time setup.",
                ],
                "parameter_actions": [
                    "Tune number of inline asm stores per loop iteration (N_STORE_PER_LOOP) — this is the primary SPKI knob. Also tune iteration count to control dilution from framework overhead.",
                ],
            }
        return {
            "root_cause": "the hot path is writing back too often, likely dominated by framework overhead (e.g. Folly fibers promise/queue management, fbthrift protocol serialization)",
            "structural_actions": [
                "INTRACTABILITY WARNING: Stores SPKI is often dominated by framework overhead that generates ~3x more stores than user code can offset. If the benchmark uses a heavy framework (Folly fibers, fbthrift) and the target is much lower than baseline, Stores SPKI may be intractable. Accept the discrepancy and focus on other features.",
                "If still attempting: batch updates in registers before storing them back to memory.",
                "Collapse repeated writes onto fewer store sites or fewer touched fields.",
            ],
            "parameter_actions": [
                "Tune store batching or write frequency downward, but expect diminishing returns if framework stores dominate.",
            ],
        }

    if feature_name == "perf_stat.iTLB-load-misses_MPKI":
        if status == "too low":
            return {
                "root_cause": "the hot executed code spans too few instruction pages, or data allocations on 4KB pages are consuming STLB entries that should map instruction pages",
                "structural_actions": [
                    "PROVEN PATTERN: The 3-noinline-function x 256-case-switch pattern naturally creates iTLB pressure because each switch body spans many 4 KB instruction pages. If the 3-switch pattern is already in place, increase ALU ops-per-case to make each switch body span more pages.",
                    "STLB CAPACITY MODEL: iTLB misses are governed by the L2 STLB (2048 entries on Intel Xeon). If hot code + data together touch fewer than 2048 pages, most iTLB misses are absorbed by the STLB. Formula: iTLB_misses_per_iter ≈ 1.73 × N_funcs × (total_pages - 2048) / total_pages. You need total touched pages > 2048 to generate meaningful iTLB misses.",
                    "1GB HUGEPAGES FOR DATA: Large data allocations (cold chains, hot chains) on regular 4KB pages consume thousands of STLB entries, crowding out instruction page mappings. Allocate all large data buffers with 1GB hugepages (mmap with MAP_HUGETLB | MAP_HUGE_1GB) so each buffer uses only 1 STLB entry regardless of size. This reserves almost all 2048 STLB entries for instruction pages.",
                    "PAD-BASED CODE PAGE CONTROL: Each noinline function's size controls how many 4KB pages it spans. With N functions of PAD_SIZE bytes padding: total_code_pages ≈ N × (code_bytes + PAD_SIZE) / 4096. Empirical: PAD_SIZE=380 with 8192 functions → ~8640 code pages → reliable iTLB misses on Intel Xeon Gold 5512U.",
                    "Separate those helpers enough that the front-end must touch more instruction pages in steady state.",
                    "Cycle through the distinct helpers repeatedly instead of adding cold one-time setup code.",
                ],
                "parameter_actions": [
                    "Tune PAD_SIZE to control code page count (primary iTLB knob), and use 1GB hugepages for all data allocations to maximize STLB availability for instruction pages.",
                ],
            }
        return {
            "root_cause": "the hot code is spread across too many instruction pages",
            "structural_actions": [
                "Concentrate hot execution on fewer helper bodies and fewer code pages.",
                "Inline or merge the hottest instruction-page crossings where possible.",
                "Remove excess page-spanning helper fan-out from the timed path.",
            ],
            "parameter_actions": [
                "Tune helper count and spacing downward once the major iTLB overshoot is corrected.",
            ],
        }

    if feature_name == "perf_stat.IPC":
        if status == "too high":
            return {
                "root_cause": "execution remains too efficient overall and needs one additional stall source in the dominant subsystem",
                "structural_actions": [
                    "PROVEN APPROACH: IPC is controlled by the balance between ALU work and memory stall cycles. The 3-switch + pointer-chase pattern produces IPC 1.5-2.0 depending on iteration count. If IPC is too high, increase iteration count so the switch-loop (with its cache-missing pointer chases) dominates more of the total runtime.",
                    "ITERATION COUNT AS PRIMARY IPC KNOB: When the benchmark framework controls runtime, iteration count determines what fraction of instructions come from tuned code vs framework overhead. Higher iterations = lower IPC. Empirical: 50M iters -> IPC 1.96, 100M -> 1.70, 130M -> 1.62, 160M -> 1.63.",
                ],
                "parameter_actions": [
                    "Tune iteration count first (the cheapest knob), then adjust ops-per-case or chain configuration if needed.",
                ],
            }
        return {
            "root_cause": "the hot path has too much stall time, dependency depth, or extra work",
            "structural_actions": [
                "Remove only the smallest avoidable source of stall time without changing benchmark character.",
            ],
            "parameter_actions": [
                "Tune the remaining stall source downward while keeping the current shape intact.",
            ],
        }

    return {
        "root_cause": "the dominant hot-path subsystem is still mismatched relative to target",
        "structural_actions": [
            "Apply one minimal local structural change that directly targets the dominant subsystem cause.",
        ],
        "parameter_actions": [
            "Tune the size or frequency of that mechanism after the main gap shrinks.",
        ],
    }


def _edit_regime(row: dict[str, Any]) -> str:
    overflow_ratio = float(row.get("overflow_ratio", 0.0) or 0.0)
    if overflow_ratio > STRUCTURAL_EDIT_OVERFLOW_THRESHOLD:
        return "structural_edit"
    return "parameter_tuning"


def _unique_lines(lines: list[str]) -> list[str]:
    rendered: list[str] = []
    seen: set[str] = set()
    for line in lines:
        cleaned = " ".join(str(line).split()).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        rendered.append(cleaned)
    return rendered


def _sentence(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    return cleaned.rstrip(".") + "."


def _detect_cache_hierarchy() -> str | None:
    """Read CPU cache sizes from sysfs and return a one-line summary, or None."""
    sizes: dict[str, str] = {}
    cpu0_cache = "/sys/devices/system/cpu/cpu0/cache"
    try:
        if not os.path.isdir(cpu0_cache):
            return None
        for entry in sorted(os.listdir(cpu0_cache)):
            idx_path = os.path.join(cpu0_cache, entry)
            if not os.path.isdir(idx_path):
                continue
            try:
                level = open(os.path.join(idx_path, "level")).read().strip()
                ctype = open(os.path.join(idx_path, "type")).read().strip().lower()
                size = open(os.path.join(idx_path, "size")).read().strip()
            except OSError:
                continue
            if level == "1" and "data" in ctype:
                sizes["L1d"] = size
            elif level == "1" and "instruction" in ctype:
                sizes["L1i"] = size
            elif level == "2":
                sizes["L2"] = size
            elif level == "3":
                sizes["L3"] = size
        # Detect core count for shared L3 estimate
        core_count = os.cpu_count() or 1
        parts = []
        if "L1d" in sizes:
            parts.append(f"L1d={sizes['L1d']}")
        if "L1i" in sizes:
            parts.append(f"L1i={sizes['L1i']}")
        if "L2" in sizes:
            parts.append(f"L2={sizes['L2']}")
        if "L3" in sizes:
            parts.append(f"L3={sizes['L3']} shared ({core_count} cores)")
        return ", ".join(parts) if parts else None
    except Exception:
        return None


# ── Feature coupling constants ────────────────────────────────────────────
# Pairs of features that have conflicting data-size requirements.
_CONFLICTING_FEATURE_PAIRS: tuple[tuple[str, str, str], ...] = (
    (
        "perf_stat.L1-dcache-load-misses_MPKI",
        "perf_stat.LLC-load-misses_MPKI",
        "L1-d misses need a working set that fits in L2 but not L1, while LLC misses need a working set that exceeds the LLC. Use a two-tier design: a small hot chain (fits L2) for L1-d pressure plus a separate cold chain (exceeds LLC) accessed at controlled frequency for LLC misses.",
    ),
    (
        "perf_stat.branch-misses_MPKI",
        "perf_stat.IPC",
        "Branch misses and IPC have opposing requirements: high branch misses cause pipeline flushes that lower IPC, but IPC targets may require higher throughput. Tune branch mispredict rate to a precise level rather than maximizing it.",
    ),
    (
        "perf_stat.L1-dcache-load-misses_MPKI",
        "perf_stat.IPC",
        "L1-d cache misses stall the pipeline and reduce IPC. If IPC must stay high while L1-d misses increase, use a working set that hits in L2 (4-10 cycle latency) rather than spilling to L3 or DRAM.",
    ),
    (
        "perf_stat.L1-dcache-loads_LPKI",
        "perf_stat.L1-icache-load-misses_MPKI",
        "INSTRUCTION DILUTION TRAP: Adding inline asm loads to increase LPKI also increases total instruction count, which dilutes ALL MPKI metrics (icache, iTLB, LLC, branch, L1d misses) equally. If LPKI and any MPKI metric are both out of tolerance, prioritize the MPKI metrics first, then fine-tune LPKI last. Each added load instruction dilutes every MPKI by ~1/(total_insn+1).",
    ),
    (
        "perf_stat.L1-dcache-stores_SPKI",
        "perf_stat.L1-icache-load-misses_MPKI",
        "INSTRUCTION DILUTION TRAP: Adding inline asm stores to increase SPKI also increases total instruction count, which dilutes ALL MPKI metrics equally. If SPKI and any MPKI metric are both out of tolerance, prioritize the MPKI metrics first, then fine-tune SPKI last.",
    ),
    (
        "perf_stat.LLC-load-misses_MPKI",
        "perf_stat.iTLB-load-misses_MPKI",
        "STLB COMPETITION: Large data allocations (cold chains for LLC misses) on 4KB pages consume thousands of STLB entries, crowding out instruction page mappings and destroying iTLB accuracy. Allocate cold chain buffers on 1GB hugepages (MAP_HUGETLB | MAP_HUGE_1GB) so each buffer uses only 1 STLB entry, leaving STLB capacity for instruction pages.",
    ),
)


def _feature_coupling_warnings(
    feature_rows: list[dict[str, Any]],
) -> list[str]:
    """Return coupling warnings for features that are both out-of-tolerance."""
    violated = {
        str(row.get("feature_name", "")).strip()
        for row in feature_rows
        if not bool(row.get("within_tolerance"))
    }
    warnings: list[str] = []
    for feat_a, feat_b, warning in _CONFLICTING_FEATURE_PAIRS:
        if feat_a in violated and feat_b in violated:
            warnings.append(f"FEATURE COUPLING: {warning}")
    return warnings


def build_diagnostician_report(
    *,
    target_features: dict[str, Any],
    current_features: dict[str, Any],
    binary_name: str = "",
    target_file_name: str = "",
    generation_strategy: str = "derived_from_nearest",
    objective_priority_rank: int | None = None,
) -> dict[str, Any]:
    canonical_target = canonicalize_feature_dict(target_features)
    canonical_current = canonicalize_feature_dict(current_features)
    try:
        constraint_report = evaluate_priority_constraints(canonical_target, canonical_current)
    except Exception:
        constraint_report = {}
    feature_rows = list(constraint_report.get("feature_rows") or [])

    if not feature_rows:
        summary_text = (
            "- Diagnostician could not compare the current benchmark features against the target.\n"
            "- Fall back to one minimal local edit and re-measure before attempting a broader rewrite."
        )
        return {
            "active_priority_rank": None,
            "active_tier_label": priority_tier_label(None),
            "playbook_name": "fallback",
            "edit_regime": "parameter_tuning",
            "summary_text": summary_text,
            "prompt_text": summary_text,
            "generator_guidance": [
                "Make one minimal local edit only and re-measure before trying a broader rewrite.",
            ],
            "constraint_report": constraint_report,
        }

    active_priority_rank = _resolve_active_priority_rank(feature_rows, objective_priority_rank)
    tier_view = _priority_view(feature_rows, locked_priority_rank=active_priority_rank)
    rows_by_name = _rows_by_name(feature_rows)
    focus_rows = list(tier_view.get("active_focus_rows") or [])
    protected_rows = list(tier_view.get("guardrail_rows") or []) + list(
        tier_view.get("active_protected_rows") or []
    )
    deferred_rows = list(tier_view.get("deferred_rows") or [])

    primary_row = focus_rows[0] if focus_rows else None
    if primary_row is None:
        primary_row = next((row for row in feature_rows if not bool(row.get("within_tolerance"))), feature_rows[0])

    delegated_from_ipc = False
    delegation_reason = ""
    diagnosed_row = primary_row
    if str(primary_row.get("feature_name", "")).strip() == "perf_stat.IPC":
        delegated_from_ipc = True
        diagnosed_row, delegation_reason = _ipc_decomposition(primary_row, rows_by_name=rows_by_name)

    diagnosed_feature_name = str(diagnosed_row.get("feature_name", "")).strip()
    diagnosed_status = _metric_status(diagnosed_row)
    edit_regime = _edit_regime(diagnosed_row)
    playbook = _playbook_payload(feature_name=diagnosed_feature_name, status=diagnosed_status)
    playbook_name = _playbook_name(diagnosed_feature_name)
    expected_effect = expected_effect_directive(diagnosed_feature_name, diagnosed_status)
    overflow_ratio = float(diagnosed_row.get("overflow_ratio", 0.0) or 0.0)

    if diagnosed_status == "too low":
        direction = "increase"
    elif diagnosed_status == "too high":
        direction = "decrease"
    else:
        direction = ""

    try:
        active_tier_label = priority_tier_label(active_priority_rank)
    except Exception:
        active_tier_label = f"Tier {active_priority_rank}" if active_priority_rank is not None else "unknown"
    excluded_label_names = {diagnosed_feature_name}
    protected_labels = _unique_lines(
        [
            feature_display_name(feature_name)
            for row in protected_rows
            for feature_name in [str(row.get("feature_name", "")).strip()]
            if feature_name and feature_name not in excluded_label_names
        ]
    )
    deferred_labels = _unique_lines(
        [
            feature_display_name(feature_name)
            for row in deferred_rows
            for feature_name in [str(row.get("feature_name", "")).strip()]
            if feature_name and feature_name not in excluded_label_names
        ]
    )
    current_value_text = _format_metric_value(diagnosed_row.get("candidate_value"))
    target_range_text = _acceptable_range(diagnosed_row)

    summary_lines = [
        f"- Active priority tier: {active_tier_label}.",
    ]
    if delegated_from_ipc:
        ipc_row = rows_by_name.get("perf_stat.IPC") or primary_row
        summary_lines.append(
            f"- IPC is {_metric_status(ipc_row)} (current={_format_metric_value(ipc_row.get('candidate_value'))}, {_acceptable_range(ipc_row)})."
        )
        if delegation_reason:
            summary_lines.append(f"- {delegation_reason}")
    summary_lines.extend(
        [
            f"- Playbook: {playbook_name}.",
            (
                f"- Primary mismatch: {feature_display_name(diagnosed_feature_name)} is {diagnosed_status} "
                f"(current={current_value_text}, {target_range_text})."
            ),
            f"- Root cause: {playbook['root_cause']}.",
            f"- Expected effect: {expected_effect} The deciding script must confirm this movement across the benchmark fleet.",
            (
                f"- Edit regime: {'structural edit' if edit_regime == 'structural_edit' else 'parameter tuning'} "
                f"because the residual overflow ratio is {overflow_ratio:.2f}."
            ),
        ]
    )

    active_actions = list(playbook["structural_actions"] if edit_regime == "structural_edit" else playbook["parameter_actions"])
    followup_actions = list(playbook["parameter_actions"])
    if active_actions:
        summary_lines.append(
            f"- Prescribe now: {_sentence(_adapt_for_generation_mode(active_actions[0], generation_strategy))}"
        )
    if edit_regime == "structural_edit" and len(active_actions) > 1:
        summary_lines.append(
            f"- Structural follow-through: {_sentence(_adapt_for_generation_mode(active_actions[1], generation_strategy))}"
        )
    if followup_actions:
        summary_lines.append(
            f"- After the main gap shrinks, tune locally: {_sentence(_adapt_for_generation_mode(followup_actions[0], generation_strategy))}"
        )
    if protected_labels:
        summary_lines.append(f"- Protect: {', '.join(protected_labels)}.")
    if deferred_labels:
        summary_lines.append(f"- Defer until later attempts: {', '.join(deferred_labels)}.")

    generator_guidance = []
    generator_guidance.append(
        f"{expected_effect} Make the edit so this effect is measurable; the deciding "
        "script verifies it materializes across the benchmark fleet."
    )
    if delegated_from_ipc and delegation_reason:
        generator_guidance.append(delegation_reason)
    for line in active_actions:
        generator_guidance.append(_adapt_for_generation_mode(line, generation_strategy))
    if edit_regime == "structural_edit":
        for line in followup_actions[:1]:
            generator_guidance.append(
                _adapt_for_generation_mode(
                    f"Once the structural gap is smaller, switch to parameter tuning: {line}",
                    generation_strategy,
                )
            )
    try:
        guidance_lines = feature_prompt_guidance(diagnosed_feature_name, direction)[:2]
    except Exception:
        guidance_lines = []
    for line in guidance_lines:
        generator_guidance.append(_adapt_for_generation_mode(line, generation_strategy))
    if protected_labels:
        generator_guidance.append(
            f"Protect already-matched prioritized counters: {', '.join(protected_labels)}."
        )
    if deferred_labels:
        generator_guidance.append(
            f"Ignore lower-tier counters for this attempt: {', '.join(deferred_labels)}."
        )
    # Inject cache hierarchy info so the LLM knows exact cache sizes.
    cache_info = _detect_cache_hierarchy()
    if cache_info:
        generator_guidance.append(
            f"TARGET CPU CACHE HIERARCHY: {cache_info}. Size data structures relative to these capacities."
        )
    # Inject coupling warnings for conflicting out-of-tolerance features.
    for warning in _feature_coupling_warnings(feature_rows):
        generator_guidance.append(warning)
    generator_guidance = _unique_lines(generator_guidance)

    summary_text = "\n".join(summary_lines)
    return {
        "binary_name": binary_name,
        "target_file_name": target_file_name,
        "generation_strategy": generation_strategy,
        "active_priority_rank": active_priority_rank,
        "active_tier_label": active_tier_label,
        "primary_feature_name": str(primary_row.get("feature_name", "")).strip(),
        "diagnosed_feature_name": diagnosed_feature_name,
        "diagnosed_status": diagnosed_status,
        "delegated_from_ipc": delegated_from_ipc,
        "playbook_name": playbook_name,
        "expected_effect": expected_effect,
        "expected_effect_direction": expected_effect_direction(expected_effect),
        "edit_regime": edit_regime,
        "overflow_ratio": overflow_ratio,
        "root_cause": playbook["root_cause"],
        "summary_text": summary_text,
        "prompt_text": summary_text,
        "generator_guidance": generator_guidance,
        "constraint_report": constraint_report,
    }
