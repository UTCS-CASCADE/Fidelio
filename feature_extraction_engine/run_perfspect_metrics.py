#!/usr/bin/env python3
"""
Compatibility runner for TMA metrics.

The old implementation shellled out to `perfspect`. The new implementation
uses `perf stat --topdown --td-level 1` and exposes the same public function
name so existing callers do not need to change at the callsite.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, Optional


DEFAULT_TOPDOWN_CPU = int(os.environ.get("TOPDOWN_CPU", "2"))
TOPDOWN_METRICS = {
    "TMA_Frontend_Bound(%)": ("topdown-fe-bound", "frontend bound"),
    "TMA_Backend_Bound(%)": ("topdown-be-bound", "backend bound"),
    "TMA_Bad_Speculation(%)": ("topdown-bad-spec", "bad speculation"),
    "TMA_Retiring(%)": ("topdown-retiring", "retiring"),
}


def _resolve_feature_window(feature_warmup_s: Optional[float], feature_window_s: Optional[float]) -> tuple[float, float]:
    warmup_val = feature_warmup_s
    window_val = feature_window_s

    if warmup_val is None:
        try:
            warmup_val = float(os.environ.get("FEATURE_WARMUP_S", "0") or 0)
        except ValueError:
            warmup_val = 0.0
    if window_val is None:
        try:
            window_val = float(os.environ.get("FEATURE_WINDOW_S", "0") or 0)
        except ValueError:
            window_val = 0.0

    return max(0.0, float(warmup_val)), max(0.0, float(window_val))


def safe_name(s: str) -> str:
    return re.sub(r'[/\\:*?"<>|]', "_", s)


def _normalize_cmd_list(cmd_list: Iterable[str] | str) -> list[str]:
    if isinstance(cmd_list, str):
        return shlex.split(cmd_list)
    tokens = [str(tok) for tok in cmd_list]
    if len(tokens) == 1:
        single = tokens[0].strip()
        if not single:
            return []
        try:
            split_tokens = shlex.split(single)
        except ValueError:
            split_tokens = [single]
        if len(split_tokens) > 1:
            return split_tokens
    return tokens


def _prepend_taskset(cmd_list: list[str], cpu: int) -> list[str]:
    if cmd_list[:2] == ["taskset", "-c"]:
        return cmd_list
    return ["taskset", "-c", str(cpu)] + cmd_list


def _build_topdown_command(
    cmd_list: list[str],
    *,
    cpu: int,
    warmup_s: float,
    window_s: float,
    target_pid: Optional[int],
) -> list[str]:
    if target_pid is not None:
        sample_s = window_s if window_s > 0 else 3.0
        hot_pid = __import__("_hot_pid_resolver").resolve_hot_pid(target_pid)
        base = [
            "taskset", "-c", str(cpu),
            "sudo", "perf", "stat",
            "-p", str(hot_pid),
            "-C", str(cpu),
            "--topdown", "--td-level", "1",
        ]
        return base + ["--", "sleep", f"{sample_s:g}"]
    base = [
        "taskset", "-c", str(cpu),
        "sudo", "perf", "stat",
        "-a", "-C", str(cpu),
        "--topdown", "--td-level", "1",
    ]

    payload = _prepend_taskset(cmd_list, cpu)
    if warmup_s <= 0 and window_s <= 0:
        return base + ["--"] + payload

    quoted_payload = " ".join(shlex.quote(tok) for tok in payload)
    if window_s > 0:
        quoted_payload = f"timeout {window_s:g} {quoted_payload}"
    if warmup_s > 0:
        quoted_payload = f"sleep {warmup_s:g}; exec {quoted_payload}"
    else:
        quoted_payload = f"exec {quoted_payload}"
    return base + ["--", "bash", "-lc", quoted_payload]


def _extract_metric_value(line: str) -> Optional[float]:
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", line.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None


def _extract_percent_values(line: str) -> list[float]:
    values: list[float] = []
    for match in re.findall(r"[-+]?\d+(?:\.\d+)?(?=%)", line.replace(",", "")):
        try:
            values.append(float(match))
        except ValueError:
            continue
    return values


def _parse_tabular_topdown_output(lines: list[str]) -> Dict[str, float]:
    table_columns = [
        ("retiring", "TMA_Retiring(%)"),
        ("bad speculation", "TMA_Bad_Speculation(%)"),
        ("frontend bound", "TMA_Frontend_Bound(%)"),
        ("backend bound", "TMA_Backend_Bound(%)"),
    ]

    for idx, raw_line in enumerate(lines):
        normalized = " ".join(raw_line.lower().split())
        if not all(label in normalized for label, _ in table_columns):
            continue

        for value_line in lines[idx + 1:]:
            stripped = value_line.strip()
            if not stripped:
                continue
            values = _extract_percent_values(stripped)
            if len(values) < len(table_columns):
                continue
            return {
                metric_name: value
                for (_, metric_name), value in zip(table_columns, values)
            }
    return {}


def parse_topdown_output(text: str) -> Dict[str, float]:
    lines = text.splitlines()
    metrics = _parse_tabular_topdown_output(lines)
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        value = _extract_metric_value(line)
        if value is None:
            continue
        for metric_name, aliases in TOPDOWN_METRICS.items():
            if any(alias in lowered for alias in aliases):
                metrics[metric_name] = value
                break
    return metrics


def _write_summary_csv(output_dir: Path, metrics: Dict[str, float]) -> Path:
    summary_path = output_dir / "tma_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for metric_name, value in metrics.items():
            writer.writerow({"metric": metric_name, "value": value})
    return summary_path


# ---------------------------------------------------------------------------
# Backend-bound TMA drill-down (Level-2/3) collection.
#
# The Level-1 `--topdown` pass only yields Frontend/Backend/BadSpec/Retiring.
# The backend-bound prioritization profile additionally needs the memory/core
# split and its cache drill-down. These come from perf metric groups
# (`perf stat -M tma_*`) which require a TMA-aware perf build. We auto-detect a
# capable perf (configurable via TMA_PERF_BIN) and collect these additively so
# the Level-1 path is never disturbed. Disable with COLLECT_BACKEND_TMA=0.
# ---------------------------------------------------------------------------
BACKEND_TMA_METRICS = {
    "tma_memory_bound": "TMA_Memory_Bound(%)",
    "tma_core_bound": "TMA_Core_Bound(%)",
    "tma_dram_bound": "TMA_DRAM_Bound(%)",
    "tma_l1_bound": "TMA_L1_Bound(%)",
    "tma_l2_bound": "TMA_L2_Bound(%)",
    "tma_l3_bound": "TMA_L3_Bound(%)",
}

_TMA_PERF_BIN_CACHE: list[Optional[str]] = []


def _resolve_tma_perf() -> Optional[str]:
    """Return a perf binary that exposes `-M tma_*` metric groups, or None."""
    if _TMA_PERF_BIN_CACHE:
        return _TMA_PERF_BIN_CACHE[0]
    candidates = []
    env_bin = os.environ.get("TMA_PERF_BIN")
    if env_bin:
        candidates.append(env_bin)
    candidates.append("/myd/perf-build/linux-6.12/tools/perf/perf")
    candidates.append("perf")
    resolved: Optional[str] = None
    for cand in candidates:
        if not cand:
            continue
        try:
            out = subprocess.run(
                [cand, "list", "metric"], capture_output=True, text=True, timeout=30
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if "tma_" in (out.stdout or ""):
            resolved = cand
            break
    _TMA_PERF_BIN_CACHE.append(resolved)
    return resolved


def _build_backend_tma_command(
    cmd_list: list[str],
    *,
    perf_bin: str,
    cpu: int,
    warmup_s: float,
    window_s: float,
    target_pid: Optional[int],
) -> list[str]:
    base = [
        "taskset",
        "-c",
        str(cpu),
        "sudo",
        perf_bin,
        "stat",
        "-a",
        "-C",
        str(cpu),
        "-M",
        ",".join(BACKEND_TMA_METRICS.keys()),
    ]
    if target_pid is not None:
        sample_s = window_s if window_s > 0 else 3.0
        return base + ["--", "sleep", f"{sample_s:g}"]

    payload = _prepend_taskset(cmd_list, cpu)
    if warmup_s <= 0 and window_s <= 0:
        return base + ["--"] + payload

    quoted_payload = " ".join(shlex.quote(tok) for tok in payload)
    if window_s > 0:
        quoted_payload = f"timeout {window_s:g} {quoted_payload}"
    if warmup_s > 0:
        quoted_payload = f"sleep {warmup_s:g}; exec {quoted_payload}"
    else:
        quoted_payload = f"exec {quoted_payload}"
    return base + ["--", "bash", "-lc", quoted_payload]


def parse_backend_tma_output(text: str) -> Dict[str, float]:
    """Parse `# 12.3 %  tma_memory_bound` style metric annotations."""
    metrics: Dict[str, float] = {}
    for match in re.finditer(
        r"#\s*([-+]?\d+(?:\.\d+)?)\s*%\s*(tma_[a-z0-9_]+)", text
    ):
        key = match.group(2).lower()
        mapped = BACKEND_TMA_METRICS.get(key)
        if mapped is None:
            continue
        try:
            metrics[mapped] = float(match.group(1))
        except ValueError:
            continue
    return metrics


def collect_backend_tma_metrics(
    cmd_tokens: list[str],
    *,
    work_dir: Path,
    cpu: int,
    warmup_s: float,
    window_s: float,
    target_pid: Optional[int],
) -> Dict[str, float]:
    if os.environ.get("COLLECT_BACKEND_TMA", "1").strip().lower() in {"0", "false", "no"}:
        return {}
    perf_bin = _resolve_tma_perf()
    if not perf_bin:
        print("[topdown] No TMA-capable perf found; skipping backend drill-down "
              "(set TMA_PERF_BIN to a perf with `-M tma_*` support)")
        return {}

    perf_cmd = _build_backend_tma_command(
        cmd_tokens,
        perf_bin=perf_bin,
        cpu=cpu,
        warmup_s=warmup_s,
        window_s=window_s,
        target_pid=target_pid,
    )
    print(f"[topdown] Collecting backend TMA drill-down via {perf_bin}")
    try:
        result = subprocess.run(perf_cmd, capture_output=True, text=True)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[topdown] Backend TMA collection failed: {exc}")
        return {}

    (work_dir / "tma_backend_raw.log").write_text(
        "\n".join(
            [
                f"Command: {' '.join(perf_cmd)}",
                f"Returncode: {result.returncode}",
                "",
                "=== STDOUT ===",
                result.stdout or "",
                "",
                "=== STDERR ===",
                result.stderr or "",
            ]
        ),
        encoding="utf-8",
    )
    return parse_backend_tma_output((result.stdout or "") + "\n" + (result.stderr or ""))



def run_single_cmd_perfspect(
    cmd_list,
    perfspect_bin: str = "perf",
    extra_args: Optional[list[str]] = None,
    output_dir: Optional[str] = None,
    feature_warmup_s: Optional[float] = None,
    feature_window_s: Optional[float] = None,
    target_pid: Optional[int] = None,
    cpu: int = DEFAULT_TOPDOWN_CPU,
):
    del perfspect_bin
    del extra_args

    cmd_tokens = _normalize_cmd_list(cmd_list)
    warmup_s, window_s = _resolve_feature_window(feature_warmup_s, feature_window_s)
    sample_s = window_s if window_s > 0 else 3.0

    if output_dir is None:
        work_dir = Path.cwd() / "topdown_metrics"
    else:
        work_dir = Path(output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if target_pid is not None and warmup_s > 0:
        time.sleep(warmup_s)

    perf_cmd = _build_topdown_command(
        cmd_tokens,
        cpu=cpu,
        warmup_s=warmup_s,
        window_s=window_s,
        target_pid=target_pid,
    )

    print(f"\n{'='*70}")
    print("[topdown] Collecting level-1 TMA metrics")
    if target_pid is None:
        print(f"[topdown] Payload command: {' '.join(cmd_tokens)}")
    else:
        print(f"[topdown] Sampling system-wide on CPU {cpu} for external workload for {sample_s:g}s")
    print(f"[topdown] Command: {' '.join(perf_cmd)}")
    print(f"{'='*70}")

    start = time.time()
    result = subprocess.run(perf_cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    raw_log = work_dir / "tma_raw.log"
    raw_log.write_text(
        "\n".join(
            [
                f"Payload command: {' '.join(cmd_tokens)}",
                f"Elapsed: {elapsed:.3f}s",
                f"Returncode: {result.returncode}",
                "",
                "=== STDOUT ===",
                result.stdout or "",
                "",
                "=== STDERR ===",
                result.stderr or "",
            ]
        ),
        encoding="utf-8",
    )

    metrics = parse_topdown_output((result.stdout or "") + "\n" + (result.stderr or ""))
    backend_metrics = collect_backend_tma_metrics(
        cmd_tokens,
        work_dir=work_dir,
        cpu=cpu,
        warmup_s=warmup_s,
        window_s=window_s,
        target_pid=target_pid,
    )
    metrics.update(backend_metrics)
    _write_summary_csv(work_dir, metrics)
    return metrics


def _discover_binaries(bench_bin_dir: Path, binary_pattern: str) -> list[Path]:
    return sorted(
        p.resolve()
        for p in bench_bin_dir.glob(binary_pattern)
        if p.is_file() and os.access(p, os.X_OK)
    )


def _list_microbenchmarks(binary_path: Path) -> tuple[list[str], bool]:
    cmd = [str(binary_path), "--bm_list"]
    try:
        res = subprocess.run(
            cmd,
            cwd=str(binary_path.parent),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if res.returncode == 0 and (res.stdout or "").strip():
            units = [line.strip() for line in res.stdout.splitlines() if line.strip()]
            if units:
                return units, True
    except Exception:
        pass
    return [binary_path.name], False


def _choose_cmd(binary_path: Path, unit_name: str, has_bm_list: bool) -> list[str]:
    if not has_bm_list:
        return [str(binary_path)]
    rx = f"^{re.escape(unit_name)}$"
    candidates = [
        [str(binary_path), "--benchmark", "--bm_regex", rx],
        [str(binary_path), "--bm_regex", rx],
    ]
    for cmd in candidates:
        probe = ["timeout", "3"] + cmd
        try:
            res = subprocess.run(
                probe,
                cwd=str(binary_path.parent),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if res.returncode in (0, 124):
                return cmd
        except Exception:
            continue
    return candidates[0]


def _write_unit_summary(output_dir: Path, unit_name: str, metrics: Dict[str, float]) -> None:
    unit_csv = output_dir / f"{safe_name(unit_name)}_summary.csv"
    with unit_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for metric_name, value in metrics.items():
            writer.writerow({"metric": metric_name, "value": value})


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect level-1 topdown metrics for benchmark binaries")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bench-bin-dir", required=True)
    parser.add_argument("--binary-pattern", default="*")
    parser.add_argument("--bm-regex", default="")
    parser.add_argument("--feature-warmup-seconds", type=float, default=0.0)
    parser.add_argument("--feature-window-seconds", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    bench_bin_dir = Path(args.bench_bin_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    binaries = _discover_binaries(bench_bin_dir, args.binary_pattern)
    if not binaries:
        raise SystemExit(f"No executable binaries found in {bench_bin_dir} matching {args.binary_pattern}")

    unit_filter = re.compile(args.bm_regex) if args.bm_regex else None
    any_metrics = False
    for binary_path in binaries:
        units, has_bm_list = _list_microbenchmarks(binary_path)
        for unit_name in units:
            if unit_filter and not unit_filter.search(unit_name):
                continue
            cmd = _choose_cmd(binary_path, unit_name, has_bm_list)
            unit_dir = output_dir / safe_name(binary_path.name) / safe_name(unit_name)
            unit_dir.mkdir(parents=True, exist_ok=True)
            metrics = run_single_cmd_perfspect(
                cmd,
                output_dir=str(unit_dir),
                feature_warmup_s=args.feature_warmup_seconds,
                feature_window_s=args.feature_window_seconds,
                cpu=DEFAULT_TOPDOWN_CPU,
            )
            _write_unit_summary(unit_dir.parent, unit_name, metrics)
            any_metrics = any_metrics or bool(metrics)

    if not any_metrics:
        raise SystemExit("No topdown metrics were collected")


if __name__ == "__main__":
    main()
