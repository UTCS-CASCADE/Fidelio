#!/usr/bin/env python3
"""
End-to-end feature extraction.

Modes:
  1. Binary batch mode:
     input: directory of binaries
     output: single XLSX with one row per (binary, microbenchmark)

  2. Cloud workload mode:
     input: workload command
     output: single JSON with one feature vector per collected segment
"""

import argparse
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openpyxl import Workbook

from feature_extraction_engine.run_detailed_perf_benchmarks import run_single_cmd_perf_stat
from feature_extraction_engine.run_intel_pt_record import run_single_cmd_intel_pt
from feature_extraction_engine.run_perfspect_metrics import run_single_cmd_perfspect
from feature_schema import CANONICAL_FEATURE_COLUMNS, canonicalize_feature_dict

PERF_STAT_FEATURE_EVENTS = {
    "perf_stat.IPC": "cycles,instructions",
    "perf_stat.L1-dcache-load-misses_MPKI": "instructions,L1-dcache-load-misses",
    "perf_stat.L1-dcache-loads_LPKI": "instructions,L1-dcache-loads",
    "perf_stat.L1-dcache-stores_SPKI": "instructions,L1-dcache-stores",
    "perf_stat.L1-icache-load-misses_MPKI": "instructions,L1-icache-load-misses",
    "perf_stat.L2-icache-load-misses_MPKI": "instructions,r2424",
    "perf_stat.LLC-load-misses_MPKI": "instructions,LLC-load-misses",
    "perf_stat.branch-load-misses_MPKI": "instructions,branch-load-misses",
    "perf_stat.branch-misses_MPKI": "instructions,branch-misses",
    "perf_stat.iTLB-load-misses_MPKI": "instructions,iTLB-load-misses",
}

REPO_ROOT = Path(__file__).resolve().parent
FEATURE_EXTRACTION_ROOT = REPO_ROOT / "feature_extraction_engine"
if str(FEATURE_EXTRACTION_ROOT) not in sys.path:
    sys.path.insert(0, str(FEATURE_EXTRACTION_ROOT))
WORKLOAD_EXTRACTOR = REPO_ROOT / "feature_extraction_engine" / "run_feature_extraction.py"


def safe_name(s: str) -> str:
    return re.sub(r'[/\\:*?"<>|]', "_", s)


def _coerce_subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return value


def is_executable_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_mode & 0o111 != 0


def discover_binaries(binaries_dir: Path, binary_pattern: str) -> List[Path]:
    bins = [p.resolve() for p in sorted(binaries_dir.glob(binary_pattern)) if is_executable_file(p)]
    return bins


def load_command_manifest(path: Path) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(entry, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        command = entry.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(token, str) for token in command):
            raise ValueError(f"{path}:{line_number}: command must be a non-empty string list")
        binary_path = Path(str(entry.get("binary_path") or command[0])).expanduser().resolve()
        if not is_executable_file(binary_path):
            raise ValueError(f"{path}:{line_number}: binary is not executable: {binary_path}")
        command_path = Path(command[0]).expanduser()
        if command_path.exists():
            command = [str(command_path.resolve()), *command[1:]]
        normalized = {
            "binary": str(entry.get("binary") or binary_path.name),
            "binary_path": binary_path,
            "microbenchmark": str(entry.get("microbenchmark") or f"command_{line_number}"),
            "command": command,
        }
        for key in ("benchmark_suite", "source_root", "source_commit", "license"):
            if entry.get(key) is not None:
                normalized[key] = str(entry[key])
        entries.append(normalized)
    if not entries:
        raise ValueError(f"{path}: no command entries found")
    return entries


def list_microbenchmarks(binary_path: Path, timeout_s: int = 30) -> Tuple[List[str], bool]:
    binary_text = str(binary_path).lower()
    if "benchmarkreplication-codec-seeds/ffmpeg/" in binary_text:
        res = subprocess.run(
            [str(binary_path), "--list-tests", "1"],
            cwd=str(binary_path.parent),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        units = [f"ffmpeg::{line.strip()}" for line in res.stdout.splitlines() if line.strip()]
        return units, bool(units)
    if "benchmarkreplication-codec-seeds/dav1d/" in binary_text:
        res = subprocess.run(
            [str(binary_path), "--list-tests", "1"],
            cwd=str(binary_path.parent),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        units = [f"dav1d::{line.strip()}" for line in res.stdout.splitlines() if line.strip()]
        return units, bool(units)
    if "benchmarkreplication-codec-seeds/x264/" in binary_text:
        bit_depth = "10bit" if binary_path.name.endswith("10") else "8bit"
        return [f"x264::{bit_depth}::{unit}" for unit in ("pixel", "dct", "mc", "intra", "deblock", "quant", "cabac")], True

    # folly Benchmark binaries enumerate via --bm_list; Google Benchmark binaries
    # (e.g. fbthrift's ProtocolBench) use --benchmark_list_tests=true. Try both so
    # candidates from either harness reach evaluation.
    list_cmds = [
        [str(binary_path), "--bm_list"],
        [str(binary_path), "--benchmark_list_tests=true"],
    ]
    for cmd in list_cmds:
        try:
            res = subprocess.run(
                cmd,
                cwd=str(binary_path.parent),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            if res.returncode == 0 and (res.stdout or "").strip():
                units = [line.strip() for line in res.stdout.splitlines() if line.strip()]
                if units:
                    return units, True
        except Exception:
            pass
    return [binary_path.name], False


def choose_unit_cmd(binary_path: Path, unit_name: str, has_bm_list: bool) -> List[str]:
    if unit_name.startswith("ffmpeg::"):
        unit = unit_name.split("::", 1)[1]
        return [
            "taskset",
            "-c",
            os.environ.get("CODEC_SEED_CPU", "2"),
            str(binary_path),
            f"--test={unit}",
            "--bench",
            f"--duration={os.environ.get('CODEC_SEED_BENCH_DURATION_US', '500')}",
            "1",
        ]
    if unit_name.startswith("dav1d::"):
        unit = unit_name.split("::", 1)[1]
        return [
            "taskset",
            "-c",
            os.environ.get("CODEC_SEED_CPU", "2"),
            str(binary_path),
            f"--test={unit}",
            "--bench",
            f"--duration={os.environ.get('CODEC_SEED_BENCH_DURATION_US', '500')}",
            "1",
        ]
    if unit_name.startswith("x264::"):
        unit = unit_name.rsplit("::", 1)[1]
        return [
            "taskset",
            "-c",
            os.environ.get("CODEC_SEED_CPU", "2"),
            str(binary_path),
            f"--bench={unit}",
            "1",
        ]
    if unit_name.startswith("svt_av1::"):
        match = re.fullmatch(
            r"svt_av1::preset(?P<preset>\d+)::crf(?P<crf>\d+(?:\.\d+)?)::"
            r"elfuente_(?P<start>\d+)_(?P<end>\d+)::(?P<frames>\d+)frames"
            r"(?:::peers(?P<peers>\d+))?",
            unit_name,
        )
        if match is None:
            raise ValueError(f"Unsupported sustained SVT-AV1 seed identifier: {unit_name}")
        start = match.group("start")
        end = match.group("end")
        clip_root = Path(
            os.environ.get(
                "SVT_AV1_CLIP_ROOT",
                "/myd/DCPerf/benchmarks/video_transcode_bench/datasets/cuts",
            )
        )
        clip = (
            clip_root
            / f"NETFLIX_ElFuente_1920x1080_14296frames_2997fps_{start}_{end}.y4m"
        )
        if not clip.is_file():
            raise FileNotFoundError(f"SVT-AV1 seed input clip not found: {clip}")
        keyint = int(end) - int(start) + 2
        bitstream = Path("/dev/shm") / f"svt_eval_{os.getpid()}_{uuid.uuid4().hex}.ivf"
        prefix = (
            []
            if match.group("peers")
            else ["taskset", "-c", os.environ.get("CODEC_SEED_CPU", "2")]
        )
        return prefix + [
            str(binary_path),
            "-i",
            str(clip),
            "-b",
            str(bitstream),
            "--frames",
            match.group("frames"),
            "--preset",
            match.group("preset"),
            "--crf",
            match.group("crf"),
            "--keyint",
            str(keyint),
            "--lp",
            "1",
            "--no-progress",
            "1",
        ]

    if not has_bm_list:
        return [str(binary_path)]

    rx = f"^{re.escape(unit_name)}$"
    # Optional iteration floor: under co-located noise the per-process counters
    # only reflect contention when the benchmark runs long enough to overlap the
    # noise during the perf window. CLONE_BM_MIN_ITERS forces that.
    _min_iters = os.environ.get("CLONE_BM_MIN_ITERS", "").strip()
    _it = [f"--bm_min_iters={_min_iters}"] if _min_iters else []
    candidates = [
        [str(binary_path), "--benchmark", "--bm_regex", rx] + _it,
        [str(binary_path), "--bm_regex", rx] + _it,
        # Google Benchmark single-unit selection.
        [str(binary_path), f"--benchmark_filter={rx}"],
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


def measure_unit_runtime(
    binary_path: Path,
    unit_name: str,
    has_bm_list: Optional[bool] = None,
    timeout_s: float = 240.0,
) -> Dict[str, object]:
    """Measure a microbenchmark's rough wall-clock runtime with a plain invocation.

    This is intentionally lightweight and is used only to estimate how many
    repeated runs would roughly cover a target cloud segment.
    """
    if has_bm_list is None:
        _, has_bm_list = list_microbenchmarks(binary_path)

    cmd = choose_unit_cmd(binary_path, unit_name, bool(has_bm_list))
    start_time = time.perf_counter()
    timed_out = False
    returncode: Optional[int] = None
    stdout = ""
    stderr = ""

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(binary_path.parent),
            capture_output=True,
            text=True,
            timeout=max(float(timeout_s), 0.1),
        )
        elapsed = time.perf_counter() - start_time
        returncode = int(proc.returncode)
        stdout = _coerce_subprocess_text(proc.stdout)
        stderr = _coerce_subprocess_text(proc.stderr)
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start_time
        timed_out = True
        stdout = _coerce_subprocess_text(exc.stdout)
        stderr = _coerce_subprocess_text(exc.stderr)
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        stderr = str(exc)
    finally:
        _remove_application_seed_output(cmd)

    return {
        "command": cmd,
        "elapsed_seconds": float(elapsed),
        "timeout_seconds": float(timeout_s),
        "timed_out": timed_out,
        "returncode": returncode,
        "stdout_preview": stdout[:500],
        "stderr_preview": stderr[:500],
    }


def is_application_seed_unit(unit_name: str) -> bool:
    return str(unit_name).startswith("svt_av1::")


def application_seed_peer_count(unit_name: str) -> int:
    match = re.search(r"::peers(?P<peers>\d+)$", str(unit_name))
    return int(match.group("peers")) if match else 0


def _remove_application_seed_output(cmd: List[str]) -> None:
    if "--no-progress" not in cmd or "-b" not in cmd:
        return
    try:
        output_path = Path(cmd[cmd.index("-b") + 1])
    except (ValueError, IndexError):
        return
    if output_path.parent == Path("/dev/shm") and output_path.name.startswith("svt_eval_"):
        output_path.unlink(missing_ok=True)


def _application_seed_peer_cmd(cmd: List[str]) -> List[str]:
    peer_cmd = list(cmd)
    peer_cmd[peer_cmd.index("-b") + 1] = "/dev/null"
    return peer_cmd


def _stop_application_process(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)


def _application_returncode_description(returncode: int) -> str:
    if returncode >= 0:
        return f"exit status {returncode}"
    signal_number = -returncode
    try:
        signal_name = signal.Signals(signal_number).name
    except ValueError:
        signal_name = f"signal {signal_number}"
    return f"{signal_name} ({returncode})"


def collect_application_seed_features(
    *,
    binary_path: Path,
    unit_name: str,
    unit_output_dir: Path,
    perf_events: str,
) -> Dict[str, object]:
    cmd = choose_unit_cmd(binary_path, unit_name, False)
    unit_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = unit_output_dir / "application.log"
    status_path = unit_output_dir / "application_status.json"
    warmup_s = max(float(os.environ.get("APPLICATION_SEED_WARMUP_S", "0")), 0.0)
    window_s = max(float(os.environ.get("APPLICATION_SEED_WINDOW_S", "3")), 0.1)
    proc: subprocess.Popen[str] | None = None
    peer_procs: List[subprocess.Popen[str]] = []
    perf_metrics: Dict[str, object] = {}
    status: Dict[str, object] = {
        "command": cmd,
        "warmup_seconds": warmup_s,
        "window_seconds": window_s,
        "peer_count": application_seed_peer_count(unit_name),
        "returncode_before_collection": None,
        "returncode_after_collection": None,
    }
    try:
        peer_count = int(status["peer_count"])
        if peer_count:
            peer_cmd = _application_seed_peer_cmd(cmd)
            for _ in range(peer_count):
                peer_procs.append(
                    subprocess.Popen(
                        peer_cmd,
                        cwd=str(binary_path.parent),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        start_new_session=True,
                    )
                )
            time.sleep(
                max(float(os.environ.get("APPLICATION_SEED_PEER_STARTUP_S", "1")), 0.0)
            )
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                cmd,
                cwd=str(binary_path.parent),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            if warmup_s > 0:
                time.sleep(warmup_s)
            status["returncode_before_collection"] = proc.poll()
            if status["returncode_before_collection"] is not None:
                raise RuntimeError(
                    "Application seed exited before PMU collection with "
                    f"rc={status['returncode_before_collection']}; "
                    f"see {log_path}"
                )
            perf_metrics = run_single_cmd_perf_stat(
                [],
                perf_events=perf_events,
                output_dir=str(unit_output_dir / "perf_stat"),
                feature_window_s=window_s,
                target_pid=proc.pid,
            )
            status["returncode_after_collection"] = proc.poll()
            returncode_after_collection = status["returncode_after_collection"]
            if (
                isinstance(returncode_after_collection, int)
                and returncode_after_collection != 0
            ):
                raise RuntimeError(
                    "Application seed failed during PMU collection with "
                    f"{_application_returncode_description(returncode_after_collection)}; "
                    f"see {log_path}"
                )
    finally:
        status["returncode_before_cleanup"] = proc.poll() if proc is not None else None
        _stop_application_process(proc)
        status["returncode_after_cleanup"] = proc.poll() if proc is not None else None
        for peer_proc in peer_procs:
            _stop_application_process(peer_proc)
        _remove_application_seed_output(cmd)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    if not perf_metrics:
        raise RuntimeError(f"Application seed PMU collection returned no metrics; see {log_path}")

    row: Dict[str, object] = {
        "binary": binary_path.name,
        "binary_path": str(binary_path),
        "microbenchmark": unit_name,
        "command": " ".join(shlex.quote(x) for x in cmd),
        "status.perf_stat_ok": bool(perf_metrics),
        "status.tma_ok": False,
        "status.intel_pt_ok": False,
    }
    row.update({f"perf_stat.{k}": v for k, v in _sanitize_metrics(perf_metrics).items()})
    canonical_row = {k: v for k, v in row.items() if k not in CANONICAL_FEATURE_COLUMNS}
    canonical_row.update(canonicalize_feature_dict(row))
    return canonical_row


def _sanitize_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in metrics.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue
        out[k] = v
    return out


def _first_diagnostic_line(text: str) -> str:
    patterns = [
        r"sudo:.*",
        r"Permission denied.*",
        r"No permission to enable.*",
        r"perf_event_paranoid.*",
        r"not supported.*",
        r"Error:.*",
        r"failed.*",
        r"cannot.*",
    ]
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for pat in patterns:
            if re.search(pat, stripped, flags=re.IGNORECASE):
                return stripped
    return ""


def _warn_if_empty_metrics(
    tool_name: str,
    metrics: Dict[str, object],
    unit_output_dir: Path,
) -> None:
    if metrics:
        return

    if tool_name == "perf_stat":
        raw_log = unit_output_dir / "perf_stat" / "perf_stat_raw.txt"
        if raw_log.exists():
            reason = _first_diagnostic_line(raw_log.read_text(errors="ignore"))
            if reason:
                print(f"[warn] {tool_name}: no parsed metrics. likely cause: {reason}")
            else:
                print(f"[warn] {tool_name}: no parsed metrics; see {raw_log}")
        else:
            print(f"[warn] {tool_name}: no parsed metrics and no raw log found")
        return

    if tool_name == "tma":
        raw_log = unit_output_dir / "tma" / "tma_raw.log"
        if raw_log.exists():
            reason = _first_diagnostic_line(raw_log.read_text(errors="ignore"))
            if reason:
                print(f"[warn] {tool_name}: no parsed metrics. likely cause: {reason}")
            else:
                print(f"[warn] {tool_name}: no parsed metrics; see {raw_log}")
        else:
            print(f"[warn] {tool_name}: no parsed metrics and no raw log found")


def _status_flags_for_feature(requested_feature: Optional[str]) -> Dict[str, bool]:
    if not requested_feature:
        return {
            "status.perf_stat_ok": False,
            "status.tma_ok": False,
            "status.intel_pt_ok": False,
        }
    if requested_feature.startswith("perf_stat."):
        return {
            "status.perf_stat_ok": False,
            "status.tma_ok": True,
            "status.intel_pt_ok": True,
        }
    if requested_feature.startswith("tma."):
        return {
            "status.perf_stat_ok": True,
            "status.tma_ok": False,
            "status.intel_pt_ok": True,
        }
    if requested_feature.startswith("intel_pt."):
        return {
            "status.perf_stat_ok": True,
            "status.tma_ok": True,
            "status.intel_pt_ok": False,
        }
    return {
        "status.perf_stat_ok": False,
        "status.tma_ok": False,
        "status.intel_pt_ok": False,
    }


def collect_unit_features(
    binary_path: Path,
    unit_name: str,
    cmd: List[str],
    unit_output_dir: Path,
    perfspect_bin: str,
    perf_events: str,
    skip_intel_pt: bool,
    requested_feature: Optional[str] = None,
) -> Dict[str, object]:
    unit_output_dir.mkdir(parents=True, exist_ok=True)

    perf_metrics: Dict[str, object] = {}
    tma_metrics: Dict[str, object] = {}
    pt_metrics: Dict[str, object] = {}

    if requested_feature is None or requested_feature.startswith("perf_stat."):
        perf_metrics = run_single_cmd_perf_stat(
            cmd,
            perf_events=PERF_STAT_FEATURE_EVENTS.get(requested_feature or "", perf_events),
            output_dir=str(unit_output_dir / "perf_stat"),
        )
        _warn_if_empty_metrics("perf_stat", perf_metrics, unit_output_dir)

    if requested_feature is None or requested_feature.startswith("tma."):
        tma_metrics = run_single_cmd_perfspect(
            cmd,
            extra_args=[],
            output_dir=str(unit_output_dir / "tma"),
        )
        _warn_if_empty_metrics("tma", tma_metrics, unit_output_dir)

    if (requested_feature is None or requested_feature.startswith("intel_pt.")) and not skip_intel_pt:
        pt_metrics = run_single_cmd_intel_pt(
            cmd,
            output_dir=str(unit_output_dir / "intel_pt"),
        )

    status_flags = _status_flags_for_feature(requested_feature)
    if requested_feature is None or requested_feature.startswith("perf_stat."):
        status_flags["status.perf_stat_ok"] = bool(perf_metrics)
    if requested_feature is None or requested_feature.startswith("tma."):
        status_flags["status.tma_ok"] = bool(tma_metrics)
    if requested_feature is None or requested_feature.startswith("intel_pt."):
        status_flags["status.intel_pt_ok"] = bool(pt_metrics)

    row: Dict[str, object] = {
        "binary": binary_path.name,
        "binary_path": str(binary_path),
        "microbenchmark": unit_name,
        "command": " ".join(shlex.quote(x) for x in cmd),
    }
    row.update(status_flags)
    row.update({f"perf_stat.{k}": v for k, v in _sanitize_metrics(perf_metrics).items()})
    row.update({f"tma.{k}": v for k, v in _sanitize_metrics(tma_metrics).items()})
    row.update({f"intel_pt.{k}": v for k, v in _sanitize_metrics(pt_metrics).items()})
    canonical_row = {k: v for k, v in row.items() if k not in CANONICAL_FEATURE_COLUMNS}
    canonical_row.update(canonicalize_feature_dict(row))
    if requested_feature:
        keep = {
            "binary",
            "binary_path",
            "microbenchmark",
            "command",
            "status.perf_stat_ok",
            "status.tma_ok",
            "status.intel_pt_ok",
            requested_feature,
        }
        canonical_row = {k: v for k, v in canonical_row.items() if k in keep}
    return canonical_row


def write_xlsx(rows: List[Dict[str, object]], output_xlsx: Path) -> None:
    if not rows:
        raise RuntimeError("No rows to write")

    headers = sorted({k for row in rows for k in row.keys()})

    wb = Workbook()
    ws = wb.active
    ws.title = "features"
    ws.append(headers)

    for row in rows:
        ws.append([row.get(h, None) for h in headers])

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_xlsx)


def build_p50_signature(row: Dict[str, object]) -> str:
    keys = sorted(k for k in row if k.startswith("intel_pt.") and k.endswith("_P50"))
    sig = {k: row.get(k) for k in keys}
    return json.dumps(sig, sort_keys=True)


def run_cloud_workload_mode(args: argparse.Namespace) -> None:
    if not args.cmd:
        raise SystemExit("Error: --cmd is required for cloud workload mode")

    cmd_list = list(args.cmd)
    if cmd_list and cmd_list[0] == "--":
        cmd_list = cmd_list[1:]
    if not cmd_list:
        raise SystemExit("Error: --cmd requires a command to run")

    output_dir = Path(args.output_dir or "./feature_extraction_outputs").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_name = args.benchmark_name
    if not benchmark_name:
        benchmark_name = safe_name(Path(cmd_list[0]).name)

    delegate_cmd = [
        "python3",
        str(WORKLOAD_EXTRACTOR),
        "--cloud-mode",
        "--output-dir",
        str(output_dir),
        "--benchmark-name",
        benchmark_name,
        "--cloud-total-seconds",
        str(args.cloud_total_seconds),
        "--cloud-workload-warmup-seconds",
        str(args.cloud_workload_warmup_seconds),
        "--cloud-segment-seconds",
        str(args.cloud_segment_seconds),
        "--events",
        args.perf_events,
    ]
    if args.amd:
        delegate_cmd.append("--skip-intel-pt")
    delegate_cmd.extend(["--cmd", *cmd_list])

    print(f"[cloud] delegating segmented workload extraction to {WORKLOAD_EXTRACTOR}")
    print(f"[cloud] output dir: {output_dir}")
    print(f"[cloud] benchmark name: {benchmark_name}")
    subprocess.run(delegate_cmd, check=True, cwd=str(REPO_ROOT))


def main() -> None:
    raw_argv = sys.argv[1:]
    passthrough_cmd: Optional[List[str]] = None
    if "--cmd" not in raw_argv and "--" in raw_argv:
        split_idx = raw_argv.index("--")
        passthrough_cmd = raw_argv[split_idx + 1 :]
        raw_argv = raw_argv[:split_idx]

    ap = argparse.ArgumentParser(description="End-to-end feature extraction")
    ap.add_argument("--binaries-dir", help="Directory containing benchmark binaries")
    ap.add_argument("--output-xlsx", help="Output XLSX path")
    ap.add_argument(
        "--commands-manifest",
        help="JSONL manifest of arbitrary microbenchmark commands to profile",
    )
    ap.add_argument(
        "--output-dir",
        help="Output directory for cloud workload mode JSONs",
    )
    ap.add_argument(
        "--benchmark-name",
        help="Benchmark/workload name for cloud workload mode",
    )
    ap.add_argument(
        "--work-dir",
        default="./feature_extraction_outputs",
        help="Directory for intermediate outputs",
    )
    ap.add_argument(
        "--binary-pattern",
        default="*",
        help="Glob for binaries inside --binaries-dir (default: *)",
    )
    ap.add_argument(
        "--perfspect-bin",
        default="perf",
        help="Deprecated compatibility flag; ignored because TMA is collected with perf stat.",
    )
    ap.add_argument(
        "--perf-events",
        default=(
            "cycles,instructions,L1-icache-load-misses,iTLB-load-misses,"
            "L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,"
            "LLC-load-misses,branch-load-misses,branch-misses,r2424"
        ),
        help="perf stat events",
    )
    ap.add_argument(
        "--amd",
        action="store_true",
        help="Skip Intel PT processing (useful on AMD systems)",
    )
    ap.add_argument(
        "--cloud-total-seconds",
        type=float,
        default=10.0,
        help="Total segmented collection horizon in cloud workload mode",
    )
    ap.add_argument(
        "--cloud-workload-warmup-seconds",
        type=float,
        default=30.0,
        help="Warmup time before segmented collection in cloud workload mode",
    )
    ap.add_argument(
        "--cloud-segment-seconds",
        type=float,
        default=3.0,
        help="Per-segment collection duration in cloud workload mode",
    )
    ap.add_argument(
        "--cmd",
        nargs=argparse.REMAINDER,
        default=None,
        help="Workload command for cloud workload mode. Everything after --cmd is the command.",
    )
    ap.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep per-unit raw outputs/logs under --work-dir for debugging",
    )
    args = ap.parse_args(raw_argv)
    if passthrough_cmd is not None:
        args.cmd = passthrough_cmd

    if args.cmd:
        run_cloud_workload_mode(args)
        return

    if not args.output_xlsx or (not args.binaries_dir and not args.commands_manifest):
        raise SystemExit(
            "Error: --output-xlsx and either --binaries-dir or --commands-manifest are required "
            "unless using --cmd cloud workload mode"
        )

    output_xlsx = Path(args.output_xlsx).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()

    if args.commands_manifest:
        manifest_path = Path(args.commands_manifest).expanduser().resolve()
        entries = load_command_manifest(manifest_path)
        rows: List[Dict[str, object]] = []
        for entry in entries:
            binary_path = Path(entry["binary_path"])
            unit = str(entry["microbenchmark"])
            cmd = list(entry["command"])
            unit_dir = work_dir / safe_name(str(entry["binary"])) / safe_name(unit)
            print(f"[run] {entry['binary']} :: {unit}")
            if is_application_seed_unit(unit):
                row = collect_application_seed_features(
                    binary_path=binary_path,
                    unit_name=unit,
                    unit_output_dir=unit_dir,
                    perf_events=args.perf_events,
                )
            else:
                row = collect_unit_features(
                    binary_path=binary_path,
                    unit_name=unit,
                    cmd=cmd,
                    unit_output_dir=unit_dir,
                    perfspect_bin=args.perfspect_bin,
                    perf_events=args.perf_events,
                    skip_intel_pt=args.amd,
                )
            row["binary"] = str(entry["binary"])
            for key in ("benchmark_suite", "source_root", "source_commit", "license"):
                if entry.get(key) is not None:
                    row[key] = str(entry[key])
            rows.append(row)
            if unit_dir.exists() and not args.keep_work_dir:
                shutil.rmtree(unit_dir, ignore_errors=True)
        write_xlsx(rows, output_xlsx)
        print(f"[ok] wrote {len(rows)} rows to {output_xlsx}")
        return

    binaries_dir = Path(args.binaries_dir).expanduser().resolve()
    if not binaries_dir.is_dir():
        raise SystemExit(f"binaries dir not found: {binaries_dir}")

    binaries = discover_binaries(binaries_dir, args.binary_pattern)
    if not binaries:
        raise SystemExit(f"no executable binaries found in {binaries_dir} with pattern {args.binary_pattern}")

    rows: List[Dict[str, object]] = []
    p50_by_binary: Dict[str, set] = {}

    print(f"[info] discovered {len(binaries)} binaries")
    for binary in binaries:
        binary_work_dir = work_dir / safe_name(binary.name)
        units, has_bm_list = list_microbenchmarks(binary)
        print(f"[info] {binary.name}: {len(units)} microbenchmark unit(s)")
        p50_by_binary.setdefault(binary.name, set())

        for unit in units:
            cmd = choose_unit_cmd(binary, unit, has_bm_list)
            unit_dir = binary_work_dir / safe_name(unit)
            print(f"[run] {binary.name} :: {unit}")
            row = collect_unit_features(
                binary_path=binary,
                unit_name=unit,
                cmd=cmd,
                unit_output_dir=unit_dir,
                perfspect_bin=args.perfspect_bin,
                perf_events=args.perf_events,
                skip_intel_pt=args.amd,
            )
            if not args.amd:
                row["intel_pt.p50_signature"] = build_p50_signature(row)
                p50_by_binary[binary.name].add(row["intel_pt.p50_signature"])
            rows.append(row)

        # Intermediate outputs are not needed after row-level features are collected.
        if binary_work_dir.exists() and not args.keep_work_dir:
            shutil.rmtree(binary_work_dir, ignore_errors=True)
            print(f"[cleanup] removed intermediate directory for {binary.name}")

    write_xlsx(rows, output_xlsx)
    print(f"[ok] wrote {len(rows)} rows to {output_xlsx}")

    if not args.amd:
        all_sigs = {sig for sigs in p50_by_binary.values() for sig in sigs}
        if len(all_sigs) < len(p50_by_binary):
            print(
                "[warn] at least two binaries produced identical Intel-PT P50 signatures. "
                "This can happen if workloads or traces are effectively the same."
            )


if __name__ == "__main__":
    main()
