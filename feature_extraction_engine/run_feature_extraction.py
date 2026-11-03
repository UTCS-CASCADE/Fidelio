#!/usr/bin/env python3
'Feature Extraction'

import atexit
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import inspect
from pathlib import Path
from collections import OrderedDict

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from run_detailed_perf_benchmarks import run_single_cmd_perf_stat
from run_perfspect_metrics import run_single_cmd_perfspect
from run_intel_pt_record import run_single_cmd_intel_pt


def cleanup_dcperf_workload() -> None:
    'Kill all DCPerf worker processes (ffmpeg, benchpress_cli, etc.). Called on script exit to ensure thorough cleanup.'
    my_pid = os.getpid()
    # Collect ancestor PIDs to protect the entire parent chain
    protected = {my_pid}
    try:
        ppid = os.getppid()
        while ppid > 1:
            protected.add(ppid)
            # Read parent of ppid from /proc
            stat = Path(f"/proc/{ppid}/stat").read_text()
            ppid = int(stat.split()[3])
    except Exception:
        pass

    patterns = ["ffmpeg", "SvtAv1EncApp", "benchpress_cli.py", "video_transcode"]
    for pattern in patterns:
        try:
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            pids = [
                int(p) for p in result.stdout.split()
                if p.strip() and int(p) not in protected
            ]
            for pid in pids:
                subprocess.run(
                    ["sudo", "kill", "-9", str(pid)],
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, Exception):
            pass


atexit.register(cleanup_dcperf_workload)


def _safe_unit_name(unit_name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', unit_name).strip('_') or "unit"

def _build_wdl_unit_regex(unit_name: str) -> str:
    """Build an anchored regex for a benchmark unit, tolerant to whitespace run differences."""
    unit = unit_name.strip()
    escaped = re.escape(unit)
    # Some bm_list outputs include variable spacing; allow any whitespace run.
    escaped = re.sub(r'(?:\\\s)+', r'\\s*', escaped)
    # Some benchmark listings include '%' in names, but matching behavior can vary.
    if unit.startswith('%'):
        escaped = escaped.replace(r"\%", r"%?")
    return f"^{escaped}$"


def _load_existing_unit_metrics(unit_json_path: Path):
    """Return parsed metrics dict for an existing unit output JSON, or None if invalid/missing."""
    if not unit_json_path.exists() or not unit_json_path.is_file():
        return None
    try:
        with open(unit_json_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def discover_wdl_units(binary_path: Path, unit_filter: str = None) -> list:
    """Discover microbenchmark units from a wdl_bench-style binary via --bm_list."""
    list_cmd = [str(binary_path), "--bm_list"]
    try:
        res = subprocess.run(
            list_cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=binary_path.parent,
        )
    except Exception as e:
        print(f"Error listing units from {binary_path}: {e}")
        return []

    if res.returncode != 0:
        stderr_preview = (res.stderr or "").strip().splitlines()
        if stderr_preview:
            print(f"--bm_list failed for {binary_path.name}: {stderr_preview[0]}")
        else:
            print(f"--bm_list failed for {binary_path.name} (return code {res.returncode})")
        return []

    units = []
    seen = set()
    for raw in (res.stdout or "").splitlines():
        unit = raw.strip()
        if not unit or unit in seen:
            continue
        seen.add(unit)
        units.append(unit)

    if unit_filter:
        try:
            regex = re.compile(unit_filter)
            units = [u for u in units if regex.search(u)]
        except re.error as e:
            print(f"Invalid --wdl-unit-filter regex '{unit_filter}': {e}")
            return []

    return units


def extract_benchmark_name(cmd_list):
    'Extract a short benchmark name from the command list.'
    if not cmd_list:
        return "unknown"
    exe = Path(cmd_list[0]).name
    # Remove common extensions
    for ext in ['.py', '.sh', '.bin', '.exe']:
        if exe.endswith(ext):
            exe = exe[:-len(ext)]
    return exe


def _set_feature_window_env(warmup_s: float, window_s: float):
    """
    Propagate window settings to downstream scripts. This is the lowest-friction way
    to thread the policy through without breaking existing function signatures.
    """
    os.environ["FEATURE_WARMUP_S"] = str(float(warmup_s))
    os.environ["FEATURE_WINDOW_S"] = str(float(window_s))


def _maybe_call_with_window(fn, *args, warmup_s: float, window_s: float, **kwargs):
    'Call downstream runner functions in a backward-compatible way: - Prefer passing warmup/window kwargs if the callee accepts them. - Otherwise fall back to calling without them (env vars still set).'
    call_kwargs = dict(kwargs)
    call_kwargs["feature_warmup_s"] = warmup_s
    call_kwargs["feature_window_s"] = window_s

    sig = inspect.signature(fn)
    has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_var_kwargs:
        return fn(*args, **call_kwargs)

    accepted = {}
    for k, v in call_kwargs.items():
        if k in sig.parameters:
            accepted[k] = v
    return fn(*args, **accepted)


def _is_alive(proc: subprocess.Popen) -> bool:
    return proc is not None and proc.poll() is None


def _stop_workload(proc: subprocess.Popen):
    if not _is_alive(proc):
        return
    pgid = None
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pass
    # First try SIGTERM on the process group
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    # Wait up to 5 seconds for clean exit
    try:
        proc.wait(timeout=5)
        return
    except Exception:
        pass
    # Escalate to SIGKILL; use sudo to handle root-owned DCPerf processes
    if pgid is not None:
        try:
            subprocess.run(
                ["sudo", "kill", "-KILL", "--", f"-{pgid}"],
                capture_output=True,
                timeout=5,
                check=False,
            )
        except Exception:
            pass
        try:
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass
    # Wait for the process to actually exit after SIGKILL to prevent FD leaks
    try:
        proc.wait(timeout=10)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Run all feature extraction tools on a single benchmark command",
        usage="%(prog)s [options] (--cmd <benchmark command...> | --wdl-binary <path>)"
    )
    parser.add_argument(
        '--output-dir',
        default='./feature_extraction',
        help='Base directory for intermediate files and final JSON output (default: ./feature_extraction)'
    )
    parser.add_argument(
        '--benchmark-name',
        default=None,
        help='Override the auto-detected benchmark name used in the output filename'
    )
    parser.add_argument(
        '--tma-bin',
        default='perf',
        help='Deprecated compatibility flag; TMA collection now uses perf stat topdown metrics.'
    )
    parser.add_argument(
        '--events',
        default='cycles,instructions,L1-icache-load-misses,iTLB-load-misses,'
                'L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,'
                'LLC-load-misses,branch-load-misses,branch-misses,r2424',
        help='Comma-separated PMU events for perf stat'
    )
    parser.add_argument('--skip-perf-stat', action='store_true', help='Skip perf stat collection')
    parser.add_argument('--skip-intel-pt', action='store_true', help='Skip Intel PT collection')
    parser.add_argument('--skip-tma', action='store_true', help='Skip tma metrics collection')

    parser.add_argument(
        '--feature-warmup-seconds',
        type=float,
        default=0.0,
        help='Legacy warmup flag. In cloud mode this warms the workload process before collection starts (default: 0)'
    )
    parser.add_argument(
        '--feature-window-seconds',
        type=float,
        default=0.0,
        help='Total duration (seconds) to run feature extraction once started. 0 means full run / existing behavior (default: 0)'
    )
    parser.add_argument(
        '--cloud-mode',
        action='store_true',
        help='Cloud workload mode: keep workload alive and segment extraction into <=3s windows'
    )
    parser.add_argument(
        '--cloud-total-seconds',
        type=float,
        default=10.0,
        help='Total cloud feature extraction duration in seconds (default: 10)'
    )
    parser.add_argument(
        '--cloud-workload-warmup-seconds',
        type=float,
        default=None,
        help='Warm workload process before segmented collection (cloud mode only). '
             'Overrides --feature-warmup-seconds in cloud mode. Explicit 0 disables warmup.'
    )
    parser.add_argument(
        '--cloud-segment-seconds',
        type=float,
        default=3.0,
        help='Max per-segment collection duration in cloud mode (default: 3, capped at 3)'
    )

    parser.add_argument(
        '--cmd',
        nargs=argparse.REMAINDER,
        default=None,
        help='The benchmark command to run. Everything after --cmd is the command. '
             'Example: --cmd sysbench cpu --cpu-max-prime=20000 --threads=8 --time=8 run'
    )
    parser.add_argument(
        '--wdl-binary',
        default=None,
        help='Path to a single wdl_bench binary. Uses --bm_list and runs all discovered units.'
    )
    parser.add_argument(
        '--wdl-unit-filter',
        default=None,
        help='Optional regex filter for unit names discovered via --bm_list'
    )
    parser.add_argument(
        '--wdl-add-benchmark-flag',
        action='store_true',
        help='Also pass --benchmark before --bm_regex when running each discovered unit'
    )

    args = parser.parse_args()

    # Parse command mode
    cmd_list = args.cmd
    if args.wdl_binary and cmd_list:
        print("Error: use either --cmd or --wdl-binary, not both")
        sys.exit(1)
    if not args.wdl_binary and not cmd_list:
        parser.print_help()
        print("\nError: provide either --cmd or --wdl-binary.")
        sys.exit(1)
    if cmd_list:
        # Strip leading '--' if present (argparse REMAINDER quirk)
        if cmd_list and cmd_list[0] == '--':
            cmd_list = cmd_list[1:]
        if not cmd_list:
            print("Error: --cmd requires a command to run")
            sys.exit(1)

    # Resolve timing parameters.
    warmup_s = float(args.feature_warmup_seconds)
    window_s = float(args.feature_window_seconds)
    cloud_total_s = float(args.cloud_total_seconds)
    cloud_segment_s = float(args.cloud_segment_seconds)
    cloud_workload_warmup_s = args.cloud_workload_warmup_seconds

    if args.cloud_mode:
        if cloud_workload_warmup_s is not None:
            warmup_s = float(cloud_workload_warmup_s)
        if window_s <= 0:
            window_s = cloud_total_s
        if cloud_segment_s <= 0:
            print("Error: --cloud-segment-seconds must be > 0")
            sys.exit(1)
        if cloud_segment_s > 3.0:
            print("[cloud-mode] --cloud-segment-seconds capped to 3.0s")
            cloud_segment_s = 3.0

    if warmup_s < 0 or window_s < 0 or cloud_total_s < 0:
        print("Error: timing arguments must be >= 0")
        sys.exit(1)

    if args.cloud_mode and window_s <= 0:
        print("Error: cloud mode requires a positive total duration "
              "(use --feature-window-seconds or --cloud-total-seconds)")
        sys.exit(1)
    if args.cloud_mode and args.wdl_binary:
        print("Error: --cloud-mode is not supported with --wdl-binary.")
        sys.exit(1)

    # Publish to env for downstream tools (even if they don't take kwargs)
    _set_feature_window_env(warmup_s, window_s)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine benchmark name and mode-specific inputs
    wdl_binary_path = None
    wdl_units = []
    if args.wdl_binary:
        wdl_binary_path = Path(args.wdl_binary).expanduser().resolve()
        if not wdl_binary_path.exists():
            print(f"Error: --wdl-binary does not exist: {wdl_binary_path}")
            sys.exit(1)
        if not wdl_binary_path.is_file():
            print(f"Error: --wdl-binary is not a file: {wdl_binary_path}")
            sys.exit(1)
        if not os.access(wdl_binary_path, os.X_OK):
            print(f"Error: --wdl-binary is not executable: {wdl_binary_path}")
            sys.exit(1)
        wdl_units = discover_wdl_units(wdl_binary_path, args.wdl_unit_filter)
        if not wdl_units:
            print(f"Error: no units discovered from {wdl_binary_path} with --bm_list")
            sys.exit(1)
        benchmark_name = args.benchmark_name or wdl_binary_path.name
    else:
        benchmark_name = args.benchmark_name or extract_benchmark_name(cmd_list)

    print(f"{'='*70}")
    print("Feature Extraction Pipeline")
    print(f"{'='*70}")
    print(f"Benchmark name: {benchmark_name}")
    if args.wdl_binary:
        print(f"WDL binary: {wdl_binary_path}")
        print(f"Discovered units: {len(wdl_units)}")
        if args.wdl_unit_filter:
            print(f"Unit filter: {args.wdl_unit_filter}")
    else:
        print(f"Command: {' '.join(cmd_list)}")
    print(f"Output directory: {output_dir}")

    if args.cloud_mode:
        full_segments = int(window_s // cloud_segment_s)
        print(f"Feature extraction timing (cloud): warmup={warmup_s:.1f}s, total={window_s:.1f}s, "
              f"segment_max={cloud_segment_s:.1f}s, full_segments={full_segments}, "
              f"ignored_tail={max(0.0, window_s - (full_segments * cloud_segment_s)):.1f}s")
        print(f"  (exported: FEATURE_WARMUP_S={os.environ.get('FEATURE_WARMUP_S')}, "
              f"FEATURE_WINDOW_S={os.environ.get('FEATURE_WINDOW_S')})")
    elif window_s > 0 or warmup_s > 0:
        print(f"Feature extraction timing: warmup={warmup_s:.1f}s, window={window_s:.1f}s")
        print(f"  (exported: FEATURE_WARMUP_S={os.environ.get('FEATURE_WARMUP_S')}, "
              f"FEATURE_WINDOW_S={os.environ.get('FEATURE_WINDOW_S')})")
    else:
        print("Feature extraction timing: (default) full run / existing behavior")
    print(f"{'='*70}\n")

    def collect_once(pass_cmd_list, pass_output_dir: Path, pass_warmup_s: float, pass_window_s: float, target_pid=None):
        pass_metrics = OrderedDict()

        if not args.skip_perf_stat:
            print("\n" + "="*70)
            print("STEP 1/3: Collecting perf stat hardware counters")
            print("="*70)
            try:
                perf_stat_dir = str(pass_output_dir / "perf_stat")
                perf_metrics = _maybe_call_with_window(
                    run_single_cmd_perf_stat,
                    pass_cmd_list,
                    args.events,
                    perf_stat_dir,
                    warmup_s=pass_warmup_s,
                    window_s=pass_window_s,
                    target_pid=target_pid,
                )
                pass_metrics.update(perf_metrics)
                print(f"\n[Step 1] Collected {len(perf_metrics)} perf stat metrics")
            except Exception as e:
                print(f"\n[Step 1] ERROR: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n[Step 1] Skipped perf stat collection")

        if not args.skip_tma:
            print("\n" + "="*70)
            print("STEP 2/3: Collecting perf-stat topdown TMA metrics")
            print("="*70)
            try:
                tma_dir = str(pass_output_dir / "tma")
                tma_metrics = _maybe_call_with_window(
                    run_single_cmd_perfspect,
                    pass_cmd_list,
                    args.tma_bin,
                    [],
                    tma_dir,
                    warmup_s=pass_warmup_s,
                    window_s=pass_window_s,
                    target_pid=target_pid,
                    cpu=1,
                )
                pass_metrics.update(tma_metrics)
                print(f"\n[Step 2] Collected {len(tma_metrics)} TMA metrics")
            except Exception as e:
                print(f"\n[Step 2] ERROR: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n[Step 2] Skipped topdown TMA collection")

        if not args.skip_intel_pt:
            print("\n" + "="*70)
            print("STEP 3/3: Collecting Intel PT instruction trace metrics")
            print("="*70)
            try:
                pt_dir = str(pass_output_dir / "intel_pt")
                pt_metrics = _maybe_call_with_window(
                    run_single_cmd_intel_pt,
                    pass_cmd_list,
                    pt_dir,
                    warmup_s=pass_warmup_s,
                    window_s=pass_window_s,
                    target_pid=target_pid,
                )
                pass_metrics.update(pt_metrics)
                print(f"\n[Step 3] Collected {len(pt_metrics)} trace metrics")
            except Exception as e:
                print(f"\n[Step 3] ERROR: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n[Step 3] Skipped Intel PT collection")

        return pass_metrics

    def run_pipeline_for_cmd(run_cmd_list, run_benchmark_name: str, run_output_dir: Path):
        all_metrics = OrderedDict()
        cloud_segment_metrics = OrderedDict()
        total_start = time.time()
        planned_segments = 0
        ignored_tail = 0.0
        collection_stop_reason = None

        if args.cloud_mode:
            workload_log_path = run_output_dir / "cloud_workload.log"
            print(f"[cloud-mode] Starting workload process once: {' '.join(run_cmd_list)}")
            with open(workload_log_path, "w") as workload_log:
                workload_proc = subprocess.Popen(
                    run_cmd_list,
                    stdout=workload_log,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,
                    text=True,
                )
                workload_pid_path = run_output_dir / "cloud_workload.pid"
                try:
                    workload_pid_path.write_text(str(workload_proc.pid))
                except Exception:
                    pass

                try:
                    if warmup_s > 0:
                        print(f"[cloud-mode] Warmup for {warmup_s:.1f}s")
                        time.sleep(warmup_s)

                    if not _is_alive(workload_proc):
                        print("[cloud-mode] Workload exited before segmented collection started")
                        collection_stop_reason = "workload_exited_before_collection"
                    else:
                        planned_segments = int(window_s // cloud_segment_s)
                        tail_s = max(0.0, window_s - (planned_segments * cloud_segment_s))
                        ignored_tail = 0.0

                        segment_idx = 0
                        while segment_idx < planned_segments and _is_alive(workload_proc):
                            segment_idx += 1
                            segment_s = cloud_segment_s
                            segment_dir = run_output_dir / f"segment_{segment_idx:02d}"
                            segment_dir.mkdir(parents=True, exist_ok=True)
                            print(f"\n[cloud-mode] Segment {segment_idx}: collecting for {segment_s:.3f}s")
                            _set_feature_window_env(0.0, segment_s)
                            segment_metrics = collect_once(
                                run_cmd_list,
                                segment_dir,
                                pass_warmup_s=0.0,
                                pass_window_s=segment_s,
                                target_pid=workload_proc.pid,
                            )
                            segment_key = f"segment_{segment_idx:02d}"
                            cloud_segment_metrics[segment_key] = OrderedDict([
                                ("segment_index", segment_idx),
                                ("segment_seconds", segment_s),
                                ("metrics", segment_metrics),
                            ])

                        # Collect tail remainder as a shorter segment if long enough
                        if tail_s >= 0.5 and _is_alive(workload_proc):
                            segment_idx += 1
                            segment_dir = run_output_dir / f"segment_{segment_idx:02d}"
                            segment_dir.mkdir(parents=True, exist_ok=True)
                            print(f"\n[cloud-mode] Tail segment {segment_idx}: collecting for {tail_s:.3f}s")
                            _set_feature_window_env(0.0, tail_s)
                            segment_metrics = collect_once(
                                run_cmd_list,
                                segment_dir,
                                pass_warmup_s=0.0,
                                pass_window_s=tail_s,
                                target_pid=workload_proc.pid,
                            )
                            segment_key = f"segment_{segment_idx:02d}"
                            cloud_segment_metrics[segment_key] = OrderedDict([
                                ("segment_index", segment_idx),
                                ("segment_seconds", tail_s),
                                ("metrics", segment_metrics),
                            ])
                        elif tail_s > 1e-9:
                            ignored_tail = tail_s
                            print(f"[cloud-mode] Tail {tail_s:.3f}s too short to collect (< 0.5s)")

                        if segment_idx < planned_segments and not _is_alive(workload_proc):
                            print("[cloud-mode] Workload exited before all segments finished")
                            collection_stop_reason = "workload_exited_early"
                        elif segment_idx >= planned_segments:
                            collection_stop_reason = "completed_planned_segments"
                finally:
                    _stop_workload(workload_proc)
                    print(f"[cloud-mode] Workload stopped. Log: {workload_log_path}")

            all_metrics = OrderedDict([
                ("mode", "cloud"),
                ("total_requested_seconds", window_s),
                ("segment_seconds", cloud_segment_s),
                ("segments_planned", planned_segments),
                ("segments_collected", len(cloud_segment_metrics)),
                ("ignored_tail_seconds", ignored_tail),
                ("collection_stop_reason", collection_stop_reason),
            ])
            all_metrics.update(cloud_segment_metrics)
        else:
            all_metrics = collect_once(run_cmd_list, run_output_dir, pass_warmup_s=warmup_s, pass_window_s=window_s)

        total_elapsed = time.time() - total_start
        output_json = run_output_dir / f"{run_benchmark_name}_feature_extraction.json"
        with open(output_json, 'w') as f:
            json.dump(all_metrics, f, indent=4)

        print(f"\n{'='*70}")
        print("FEATURE EXTRACTION COMPLETE")
        print(f"{'='*70}")
        if args.cloud_mode:
            print(f"Segments planned:   {all_metrics.get('segments_planned', 0)}")
            print(f"Segments collected: {all_metrics.get('segments_collected', 0)}")
            print(f"Collection reason:  {all_metrics.get('collection_stop_reason')}")
            first_segment_key = next((k for k in all_metrics.keys() if re.fullmatch(r"segment_\d+", k)), None)
            if first_segment_key and isinstance(all_metrics.get(first_segment_key), dict):
                first_segment_metrics = all_metrics[first_segment_key].get("metrics", {})
                print(f"Metrics per segment: {len(first_segment_metrics)}")
        else:
            print(f"Total metrics collected: {len(all_metrics)}")
        print(f"Total elapsed time: {total_elapsed:.1f}s")
        print(f"Output file: {output_json}")
        print(f"{'='*70}\n")

        if args.cloud_mode:
            print("Preview of first segment metrics:")
            first_segment_key = next((k for k in all_metrics.keys() if re.fullmatch(r"segment_\d+", k)), None)
            if first_segment_key and isinstance(all_metrics.get(first_segment_key), dict):
                first_metrics = all_metrics[first_segment_key].get("metrics", {})
                for i, (k, v) in enumerate(first_metrics.items()):
                    if i >= 10:
                        print(f"  ... and {len(first_metrics) - 10} more")
                        break
                    print(f"  {k}: {v}")
            else:
                print("  (no segments collected)")
        else:
            print("Preview of collected metrics:")
            for i, (k, v) in enumerate(all_metrics.items()):
                if i >= 10:
                    print(f"  ... and {len(all_metrics) - 10} more")
                    break
                print(f"  {k}: {v}")

        return all_metrics, output_json

    if args.wdl_binary:
        aggregate_start = time.time()
        aggregate = OrderedDict([
            ("mode", "wdl_binary"),
            ("binary", str(wdl_binary_path)),
            ("unit_count", len(wdl_units)),
            ("units", OrderedDict()),
        ])
        resumed_units = 0
        executed_units = 0

        for idx, unit_name in enumerate(wdl_units, start=1):
            safe_unit = _safe_unit_name(unit_name)
            unit_regex = _build_wdl_unit_regex(unit_name)
            run_cmd = [str(wdl_binary_path)]
            if args.wdl_add_benchmark_flag:
                run_cmd.append("--benchmark")
            run_cmd += ["--bm_regex", unit_regex]

            unit_output_dir = output_dir / safe_unit
            unit_output_dir.mkdir(parents=True, exist_ok=True)
            unit_benchmark_name = f"{benchmark_name}_{safe_unit}"
            unit_json = unit_output_dir / f"{unit_benchmark_name}_feature_extraction.json"

            existing_metrics = _load_existing_unit_metrics(unit_json)
            if existing_metrics is not None:
                resumed_units += 1
                print(f"[WDL] Skipping completed unit {idx}/{len(wdl_units)}: {unit_name}")
                print(f"[WDL] Existing output: {unit_json}")
                aggregate["units"][unit_name] = OrderedDict([
                    ("status", "resumed_existing"),
                    ("command", run_cmd),
                    ("output_json", str(unit_json)),
                    ("metric_count", len(existing_metrics)),
                    ("metrics", existing_metrics),
                ])
                continue

            print(f"\n{'='*70}")
            print(f"[WDL] Unit {idx}/{len(wdl_units)}: {unit_name}")
            print(f"[WDL] Command: {' '.join(run_cmd)}")
            print(f"{'='*70}\n")

            unit_metrics, unit_json = run_pipeline_for_cmd(run_cmd, unit_benchmark_name, unit_output_dir)
            executed_units += 1
            aggregate["units"][unit_name] = OrderedDict([
                ("status", "executed"),
                ("command", run_cmd),
                ("output_json", str(unit_json)),
                ("metric_count", len(unit_metrics)),
                ("metrics", unit_metrics),
            ])

        aggregate["elapsed_seconds"] = round(time.time() - aggregate_start, 3)
        aggregate["units_executed"] = executed_units
        aggregate["units_resumed_existing"] = resumed_units
        aggregate_json = output_dir / f"{benchmark_name}_all_units_feature_extraction.json"
        with open(aggregate_json, "w") as f:
            json.dump(aggregate, f, indent=4)

        print(f"\n{'='*70}")
        print(f"{'='*70}")
        print(f"Units total: {len(wdl_units)}")
        print(f"Units executed this run: {executed_units}")
        print(f"Units resumed from existing output: {resumed_units}")
        print(f"Aggregate output: {aggregate_json}")
        print(f"{'='*70}")
    else:
        run_pipeline_for_cmd(cmd_list, benchmark_name, output_dir)


if __name__ == '__main__':
    main()
