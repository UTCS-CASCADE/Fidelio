#!/usr/bin/env python3

import json
import subprocess
import os
import sys
import re
import time
import shlex
from pathlib import Path
from typing import Dict, Optional, Tuple
import argparse
import tempfile


def _normalize_cmd_list(cmd_list):
    """Return a robust argv-style list from either a token list or a single shell string."""
    if isinstance(cmd_list, str):
        return shlex.split(cmd_list)

    tokens = [str(tok) for tok in cmd_list]
    if len(tokens) == 1:
        single = tokens[0].strip()
        if not single:
            return []
        # Some callers pass one string like "sudo ./benchpress_cli.py run ...".
        # Split it into argv so perf gets executable + args.
        try:
            split_tokens = shlex.split(single)
        except ValueError:
            split_tokens = [single]
        if len(split_tokens) > 1:
            return split_tokens
    return tokens


def _resolve_feature_window(
    feature_warmup_s: Optional[float],
    feature_window_s: Optional[float],
) -> Tuple[float, float]:
    """Resolve effective warmup/window from explicit args, then env vars, then defaults."""
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

    warmup_val = max(0.0, float(warmup_val))
    window_val = max(0.0, float(window_val))
    return warmup_val, window_val


def _build_windowed_cmd(cmd_list, warmup_s: float, window_s: float):
    """Wrap command with optional warmup sleep and timeout window."""
    cmd_tokens = _normalize_cmd_list(cmd_list)
    # perf is already launched via sudo in this runner; avoid nested sudo in payload.
    if cmd_tokens and cmd_tokens[0] == "sudo":
        cmd_tokens = cmd_tokens[1:]

    if warmup_s <= 0 and window_s <= 0:
        return cmd_tokens

    quoted_cmd = " ".join(shlex.quote(tok) for tok in cmd_tokens)
    run_part = f"timeout -k 2 {window_s:g} {quoted_cmd}" if window_s > 0 else quoted_cmd

    if warmup_s > 0:
        shell_cmd = f"sleep {warmup_s:g}; exec {run_part}"
    else:
        shell_cmd = f"exec {run_part}"

    return ["bash", "-lc", shell_cmd]


def _build_timed_cmd(cmd_list, duration_s: float):
    """Wrap command with timeout window only (no warmup sleep)."""
    cmd_tokens = _normalize_cmd_list(cmd_list)
    if cmd_tokens and cmd_tokens[0] == "sudo":
        cmd_tokens = cmd_tokens[1:]
    if duration_s <= 0:
        return cmd_tokens
    quoted_cmd = " ".join(shlex.quote(tok) for tok in cmd_tokens)
    return ["bash", "-lc", f"exec timeout -k 2 {duration_s:g} {quoted_cmd}"]


class FollyIntelPTRecorder:
    def __init__(self, folly_test_dir: str, output_dir: str):
        self.folly_test_dir = Path(folly_test_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.benchmarks = {}
        self.discover_benchmarks()

    def _unit_output_exists(self, benchmark_name: str, unit_name: str) -> bool:
        """Check if output file already exists for this benchmark/unit pair and is non-empty.
        
        Returns True only if the .data file exists AND has size > 0 bytes.
        """
        safe_unit_name = re.sub(r'[/\\:*?"<>|]', '_', unit_name)
        binary_dir = self.output_dir / benchmark_name
        data_file = binary_dir / f"{benchmark_name}_{safe_unit_name}_pt.data"
        if data_file.exists():
            try:
                file_size = data_file.stat().st_size
                return file_size > 0
            except OSError:
                return False
        return False

    def load_probe_times(self, benchmark_name: str) -> Dict[str, tuple]:
        """Load cached probe times and commands from disk if they exist.
        
        Returns dict of unit_name -> (elapsed_seconds, command_list), or empty dict if not found.
        """
        binary_dir = self.output_dir / benchmark_name
        probe_file = binary_dir / f"{benchmark_name}_probe_times.txt"
        if not probe_file.exists():
            return {}
        
        probe_times = {}
        try:
            with open(probe_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Format: unit_name elapsed_seconds [command]
                    # Unit names can contain spaces, so use greedy match to find the last float
                    # Pattern: everything up to the last space-delimited float
                    m = re.match(r'^(.+)\s+([\d.]+)(?:\s+(.*))?$', line)
                    if m:
                        unit_name = m.group(1).strip()
                        try:
                            elapsed = float(m.group(2))
                            cmd = m.group(3) if m.group(3) else None
                            probe_times[unit_name] = (elapsed, cmd)
                        except ValueError:
                            continue
            if probe_times:
                print(f"  Loaded cached probe times from {probe_file.name} ({len(probe_times)} units)")
        except Exception as e:
            print(f"  Error loading cached probe times: {e}")
        
        return probe_times

    def save_probe_times(self, benchmark_name: str, probe_times: Dict[str, tuple]) -> Path:
        """Save probe times and commands to disk as {bench_name}_probe_times.txt.
        
        probe_times should be dict of unit_name -> (elapsed_seconds, perfspect_command_str)
        
        Returns path to the saved file.
        """
        if not probe_times:
            return None
        
        binary_dir = self.output_dir / benchmark_name
        binary_dir.mkdir(parents=True, exist_ok=True)
        probe_file = binary_dir / f"{benchmark_name}_probe_times.txt"
        with open(probe_file, 'w') as f:
            for unit_name in sorted(probe_times.keys()):
                elapsed, cmd = probe_times[unit_name]
                if cmd:
                    f.write(f"{unit_name} {elapsed:.6f} {cmd}\n")
                else:
                    f.write(f"{unit_name} {elapsed:.6f}\n")
        
        return probe_file

    def discover_benchmarks(self):
        possible_paths = [
            Path("/myd/DCPerf/benchmarks/sw_bench_2")
        ]

        wdl_build_base = None
        for path in possible_paths:
            if path.exists():
                test_files = list(path.glob("memcpy_benchmark")) + list(path.glob("*benchmark")) + list(path.glob("*bench"))
                if test_files:
                    wdl_build_base = path
                    break

        if wdl_build_base is None:
            print("No benchmark directory found. Searched:")
            for p in possible_paths:
                print(f"  - {p}")
        else:
            print(f"Found benchmark base: {wdl_build_base}")
            print("Scanning for executables in:", wdl_build_base)
            for binary in sorted(wdl_build_base.glob("*")):
                if binary.is_file() and os.access(binary, os.X_OK):
                    name = binary.name
                    if any(p in name.lower() for p in ["benchmark", "bench", "perf"]):
                        self.benchmarks[name] = {"path": binary, "standalone": False}
                        print(f"Discovered benchmark binary: {name} -> {binary}")

        # Also search for standalone bench_bin_X binaries in current directory, output_dir, and wdl_bench
        search_paths = [
            Path.cwd(),
            self.output_dir,
            Path("/mydata/DCPerf/benchmarks/wdl_bench") if Path("/mydata/DCPerf/benchmarks/wdl_bench").exists() else None
        ]
        search_paths = [p for p in search_paths if p is not None]
        
        print("\nScanning for standalone bench_bin_* binaries in:")
        for search_path in search_paths:
            print(f"  - {search_path}")
            for binary in sorted(search_path.glob("bench_bin_*")):
                if binary.is_file() and os.access(binary, os.X_OK):
                    name = binary.name
                    # Check if it looks like bench_bin_<number>
                    if re.match(r'^bench_bin_\d+$', name):
                        self.benchmarks[name] = {"path": binary, "standalone": True}
                        print(f"Discovered standalone binary: {name} -> {binary}")

    def run_benchmark_get_units(self, benchmark_binary: Path, is_standalone: bool = False) -> Tuple[Dict[str, float], Dict[str, tuple]]:
        """Run the benchmark binary to extract unit names and throughput estimates.

        Returns (units_dict, probe_times_dict) where:
          - units_dict: mapping unit_name -> throughput (iters/sec)
          - probe_times_dict: mapping unit_name -> (elapsed_seconds, command_str)
        For standalone binaries, returns ({binary_name: 1e6}, {})
        """
        # For standalone binaries, treat the whole binary as a single unit
        if is_standalone:
            return {benchmark_binary.name: 1e6}, {}
        
        benchmark_name = benchmark_binary.name
        benchmark_dir = benchmark_binary.parent
        probe_times = {}  # Track elapsed time and command for each unit probe

        # Try to load cached probe times first
        cached_times = self.load_probe_times(benchmark_name)
        if cached_times:
            # Skip probing and use cached times
            # Still get unit list but don't re-probe
            list_cmd = [str(benchmark_binary), "--bm_list"]
            res = subprocess.run(list_cmd, capture_output=True, text=True, cwd=benchmark_dir)
            if res.returncode == 0 and res.stdout and res.stdout.strip():
                unit_names = [l.strip() for l in res.stdout.splitlines() if l.strip()]
                units = {}
                for unit in unit_names:
                    units[unit] = 1e6  # Default throughput estimate
                return units, cached_times

        # Primary path: many benchmarks support --bm_list to enumerate unit names
        list_cmd = [str(benchmark_binary), "--bm_list"]
        print(f"Running to list units: {' '.join(list_cmd)} (cwd={benchmark_dir})")
        res = subprocess.run(list_cmd, capture_output=True, text=True, cwd=benchmark_dir)
        print(f"List returncode: {res.returncode}; stdout_len={len(res.stdout or '')}")
        units = {}
        if res.returncode == 0 and res.stdout and res.stdout.strip():
            # Each non-empty line is a unit name
            unit_names = [l.strip() for l in res.stdout.splitlines() if l.strip()]
            print(f"Found {len(unit_names)} units via --bm_list for {benchmark_binary.name}")
            for unit in unit_names:
                # Probe each unit with --benchmark --bm_regex
                bm_cmd = [str(benchmark_binary), "--benchmark", "--bm_regex", f"^{re.escape(unit)}$"]
                print(f"  Probing unit '{unit}'")
                probe_start = time.time()
                r2 = subprocess.run(bm_cmd, capture_output=True, text=True, cwd=benchmark_dir)
                probe_elapsed = time.time() - probe_start
                
                # Verify the unit ran successfully
                if r2.returncode == 0:
                    # Build the perfspect command for this unit
                    perfspect_cmd_str = ' '.join(["perfspect", "metrics", "--"] + bm_cmd)
                    probe_times[unit] = (probe_elapsed, perfspect_cmd_str)
                    
                    parsed = self._extract_units_from_text(r2.stdout)
                    if unit in parsed:
                        units[unit] = parsed[unit]
                    elif parsed:
                        units[unit] = list(parsed.values())[0]
                    else:
                        units[unit] = 1e6
                else:
                    print(f"  Probe failed for unit '{unit}' (returncode {r2.returncode})")
            return units, probe_times
        # if --bm_list produced nothing, fall back to previous behavior
        print(f"--bm_list produced no output for {benchmark_binary.name}, falling back to --json/text discovery")

        # Fallback: try --json or plain run as before
        cmd = [str(benchmark_binary), "--json"]
        try:
            print(f"Running to discover units: {' '.join(cmd)} (cwd={benchmark_dir})")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=benchmark_dir)
            print(f"Discovery returncode: {result.returncode}; stdout_len={len(result.stdout or '')}; stderr_len={len(result.stderr or '')}")
            units = {}
            # Try JSON first
            try:
                data = json.loads(result.stdout)
                units = self._extract_units_from_json(data)
            except json.JSONDecodeError:
                units = self._extract_units_from_text(result.stdout)

            if not units:
                # helpful debug: show a truncated stdout/stderr to help diagnose format
                out_lines = (result.stdout or "").splitlines()
                err_lines = (result.stderr or "").splitlines()
                out_snip = "\n".join(out_lines[:80])
                err_snip = "\n".join(err_lines[:80])
                print(f"[discover units] No units parsed for {benchmark_binary.name}. stdout (truncated, first 80 lines):\n{out_snip}")
                if err_snip:
                    print(f"[discover units] stderr (truncated, first 80 lines):\n{err_snip}")

            return units, {}
        except Exception as e:
            print(f"Error running benchmark to discover units: {e}")
            return {}, {}

    def _extract_units_from_json(self, data: dict) -> Dict[str, float]:
        units = {}
        # Support dict or list outputs from various benchmark JSON formats
        if isinstance(data, dict):
            # Common Google benchmark style
            if 'benchmarks' in data and isinstance(data['benchmarks'], list):
                for bench in data['benchmarks']:
                    if 'name' in bench:
                        name = bench['name']
                        throughput = bench.get('items_per_second') or bench.get('iterations_per_second') or bench.get('real_time') or bench.get('time')
                        try:
                            units[name] = float(throughput) if throughput is not None else 1e6
                        except Exception:
                            units[name] = 1e6
            else:
                for key, value in data.items():
                    if not key.startswith('%') and isinstance(value, (int, float)):
                        units[key] = float(value)
        elif isinstance(data, list):
            for bench in data:
                if isinstance(bench, dict) and 'name' in bench:
                    name = bench['name']
                    throughput = bench.get('items_per_second') or bench.get('iterations_per_second') or bench.get('real_time') or bench.get('time')
                    try:
                        units[name] = float(throughput) if throughput is not None else 1e6
                    except Exception:
                        units[name] = 1e6
        return units

    def _extract_units_from_text(self, text: str) -> Dict[str, float]:
        units = {}
        # Common Google Benchmark / folly textual output often has columns like:
        # <Name> <time/iter> <iters/s>
        # Example: "BinaryProtocol_write_Empty                                  5.96ns   167.85M"
        # First try to match that specific two-/three-column layout.
        col_pattern = re.compile(r'^\s*(?P<name>\S+)\s+(?P<time>[\d\.]+)\s*(?P<time_unit>ns|us|ms|s)?\s+(?P<iters>[\d\.,]+)\s*(?P<mult>[KMGTP]?)\b', re.IGNORECASE)

        # Fallback flexible pattern (older heuristic)
        fallback_pattern = re.compile(r'^\s*([A-Za-z0-9_().:\-]+)\s+.*?(?:=|:)?\s*([\d.,]+)\s*([KMGT]?)(?:\s*(?:iters?/sec|items?/sec|ops?/s|ops/s|per second|it/s|it/sec))?', re.IGNORECASE)

        multipliers = {'K': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12, 'P': 1e15, '': 1}

        for line in text.splitlines():
            if not line.strip():
                continue
            m = col_pattern.match(line)
            if m:
                unit_name = m.group('name').strip()
                iters_str = m.group('iters').replace(',', '')
                mult = m.group('mult').upper() if m.group('mult') else ''
                try:
                    throughput_val = float(iters_str)
                except Exception:
                    continue
                throughput = throughput_val * multipliers.get(mult, 1)
                if unit_name and throughput > 0:
                    units[unit_name] = throughput
                continue

            # fallback
            m2 = fallback_pattern.match(line)
            if m2:
                unit_name = m2.group(1).strip()
                try:
                    throughput_val = float(m2.group(2).replace(',', ''))
                except Exception:
                    continue
                multiplier = m2.group(3).upper() if m2.group(3) else ''
                throughput = throughput_val * multipliers.get(multiplier, 1)
                if unit_name and throughput > 0:
                    units[unit_name] = throughput
        return units

    def calculate_measurement_duration(self, throughput: float, target_iterations: int = 50000) -> float:
        if throughput <= 0:
            return 0.5
        duration = target_iterations / throughput
        duration = max(0.05, min(60.0, duration))
        return duration

    def run_unit_record(self, benchmark_binary: Path, unit_name: str, throughput: float,
                        dry_run: bool = False, is_standalone: bool = False) -> Tuple[bool, str, float]:
        """Run perf record with intel_pt while executing the benchmark unit.

        Returns (success, perf_data_path, elapsed_time)
        """
        benchmark_name = benchmark_binary.name
        benchmark_dir = benchmark_binary.parent
        safe_unit_name = re.sub(r'[/\\:*?"<>|]', '_', unit_name)
        
        # Create binary-specific output directory
        binary_out_dir = self.output_dir / benchmark_name
        binary_out_dir.mkdir(parents=True, exist_ok=True)

        perf_data_file = binary_out_dir / f"{benchmark_name}_{safe_unit_name}_pt.data"
        bench_out_file = binary_out_dir / f"{benchmark_name}_{safe_unit_name}_bench.txt"
        
        # Check if already completed (output file exists)
        if perf_data_file.exists():
            print(f"Skipping already completed: {benchmark_name} :: {unit_name}")
            return (True, "", 0.0)

        duration = self.calculate_measurement_duration(throughput, target_iterations=50000)
        # Add small padding
        perf_sleep = max(1, int(duration + 1)) if duration > 1 else round(duration + 0.2, 2)
        # Cap total perf record time to at most 10 seconds per microbenchmark
        MAX_RECORD_SECS = 10.0
        if perf_sleep > MAX_RECORD_SECS:
            perf_sleep = MAX_RECORD_SECS

        perf_cmd = [
            "sudo", "perf", "record",
            "-e", "intel_pt//u",
            "-a",
            "-o", str(perf_data_file),
            "sleep", str(perf_sleep)
        ]

        # For standalone binaries, just run the binary directly; for regular benchmarks, use benchmark flags
        if is_standalone:
            benchmark_cmd = [str(benchmark_binary)]
        else:
            benchmark_cmd = [str(benchmark_binary), "--benchmark", "--bm_regex", f"^{re.escape(unit_name)}$"]

        print(f"Recording: {benchmark_name} :: {unit_name} for ~{duration:.3f}s -> perf sleep {perf_sleep}")
        print(f"Perf cmd: {' '.join(perf_cmd)}")
        print(f"Benchmark cmd: {' '.join(benchmark_cmd)} (cwd={benchmark_dir})")

        if dry_run:
            print("DRY-RUN: ", " ".join(perf_cmd))
            print("DRY-RUN: ", " ".join(benchmark_cmd))
            return (True, str(perf_data_file), 0.0)

        start = time.time()

        try:
            # Start perf record first

            perf_proc = subprocess.Popen(perf_cmd, cwd=benchmark_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Started perf PIDs: {getattr(perf_proc, 'pid', 'unknown')}")

            # Small delay to ensure perf record is up
            time.sleep(0.05)

            # Run benchmark; don't wait too long for it
            print("Launching benchmark process...")
            bench_proc = subprocess.run(
                benchmark_cmd, capture_output=True, text=True, cwd=benchmark_dir
            )
            print(f"Benchmark finished: returncode={bench_proc.returncode}; stdout_len={len(bench_proc.stdout or '')}; stderr_len={len(bench_proc.stderr or '')}")

            # Wait for perf to finish (it will finish when sleep finishes)
            print("Waiting for perf to finish...")
            perf_stdout, perf_stderr = perf_proc.communicate()
            print(f"Perf finished: returncode={perf_proc.returncode}; stdout_len={len(perf_stdout or '')}; stderr_len={len(perf_stderr or '')}")

            elapsed = time.time() - start

            # Print perf stderr for debugging (often contains warnings/errors)
            if perf_stderr:
                print(f"[perf stderr] ({benchmark_name}::{unit_name}):\n{perf_stderr[:500]}")

            # Do not write per-unit bench_out_file on disk as requested; keep short debug output instead
            if bench_proc.stdout:
                print(f"[bench stdout] ({benchmark_name}::{unit_name}) {len(bench_proc.stdout)} bytes")
            if bench_proc.stderr:
                print(f"[bench stderr] ({benchmark_name}::{unit_name}):\n{bench_proc.stderr[:500]}")

            # Check if perf had non-zero return code
            if perf_proc.returncode != 0:
                print(f"WARNING: perf record exited with code {perf_proc.returncode}")
                return (False, "", elapsed)

            return (True, str(perf_data_file), elapsed)

        except PermissionError as pe:
            print(f"Permission denied: {pe}. Need sudo to run perf record")
            return (False, "", 0.0)
        except Exception as e:
            print(f"Error recording unit: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return (False, "", 0.0)

    def run_all(self, benchmark_filter: Optional[str] = None, dry_run: bool = False):
        if not self.benchmarks:
            print("No benchmarks discovered.")
            return

        selected = self.benchmarks
        if benchmark_filter:
            selected = {k: v for k, v in self.benchmarks.items() if benchmark_filter.lower() in k.lower()}

        total_work = 0
        skipped_count = 0
        failed_count = 0
        
        for name in sorted(selected.keys()):
            bench_info = selected[name]
            binary = bench_info["path"]
            is_standalone = bench_info["standalone"]
            
            # Create binary-specific output directory
            binary_out_dir = self.output_dir / name
            binary_out_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if we have cached probe times - if so, use them and skip probing
            probe_cache_file = binary_out_dir / f"{name}_probe_times.txt"
            units = {}
            cached_probe_times = {}
            
            if probe_cache_file.exists():
                # Load cached probe times using the proper parser
                cached_times = self.load_probe_times(name)
                if cached_times:
                    # Extract units from cached times
                    units = {unit: 1e6 for unit in cached_times.keys()}
                    cached_probe_times = {unit: elapsed for unit, (elapsed, cmd) in cached_times.items()}
                    
                    print(f"\n{name}: Loaded {len(units)} cached units from probe times")
                    
                    # Check if all units already have output files (complete)
                    completed_count = 0
                    for unit in units.keys():
                        if self._unit_output_exists(name, unit):
                            completed_count += 1
                    
                    if completed_count == len(units):
                        print(f"  All {len(units)} units already completed (output files exist) - skipping entirely")
                        skipped_count += len(units)
                        continue
                    else:
                        print(f"  {len(units)} cached units, {completed_count} already completed, {len(units) - completed_count} need recording")
            else:
                # No cache - probe normally
                units, probe_times = self.run_benchmark_get_units(binary, is_standalone=is_standalone)
                # Save probe times from discovery
                if probe_times:
                    probe_file = self.save_probe_times(name, probe_times)
                    print(f"  Saved probe times to {probe_file.name}")
            
            if not units:
                print(f"No units for {name}")
                continue
            
            # Check if all units have output files
            completed_count = sum(1 for unit in units.keys() if self._unit_output_exists(name, unit))
            if completed_count == len(units):
                print(f"\n{name}: All {len(units)} units already completed - skipping")
                skipped_count += len(units)
                continue
            
            # Dict to accumulate probe times for this benchmark (for elapsed time tracking)
            probe_times_dict = {}
            
            for unit, thr in sorted(units.items()):
                total_work += 1
                if self._unit_output_exists(name, unit):
                    skipped_count += 1
                    continue
                
                # Use cached duration if available, otherwise use calculated throughput
                if unit in cached_probe_times:
                    unit_duration = cached_probe_times[unit]
                else:
                    unit_duration = thr
                
                ok, perf_path, elapsed = self.run_unit_record(binary, unit, unit_duration, dry_run=dry_run, is_standalone=is_standalone)
                
                # Record the actual elapsed time for future caching
                # Build the perfspect command string for this unit
                if elapsed > 0:
                    if is_standalone:
                        bm_cmd_parts = [str(binary)]
                    else:
                        bm_cmd_parts = [str(binary), "--benchmark", "--bm_regex", f"^{re.escape(unit)}$"]
                    perfspect_cmd_str = ' '.join(["perfspect", "metrics", "--"] + bm_cmd_parts)
                    probe_times_dict[unit] = (elapsed, perfspect_cmd_str)
                
                if not ok:
                    print(f"Failed: {name} :: {unit}")
                    failed_count += 1
            
            # Save probe times for this benchmark if we ran any units
            if probe_times_dict:
                probe_file = self.save_probe_times(name, probe_times_dict)
                print(f"  Saved probe times to {probe_file.name}")
        
        print(f"\n=== Summary ===")
        print(f"Total work items: {total_work}")
        print(f"Already completed: {skipped_count}")
        print(f"Failed in this run: {failed_count}")
        print(f"Completed in this run: {total_work - skipped_count - failed_count}")


def run_single_cmd_intel_pt(
    cmd_list,
    output_dir=None,
    feature_warmup_s: Optional[float] = None,
    feature_window_s: Optional[float] = None,
    target_pid: Optional[int] = None,
):
    """Run Intel PT recording on a single command, disassemble, and compute trace metrics.
    
    This function:
      1. Records an Intel PT trace while running the given command
      2. Disassembles the trace using perf script
      3. Computes instruction trace statistics (block sizes, jumps, branch runs,
         RAW dependency distances, instruction family distributions)
      4. Returns a flat dict of percentile metrics suitable for the feature extraction JSON
    
    Args:
        cmd_list: List of strings representing the command to run.
        output_dir: Optional directory to save intermediate files.
    
    Returns:
        Dict of metric_name -> value, e.g.:
          {"block_size_P50": 5, "family::arith_P50": 1, ...}
    """
    import numpy as np
    
    # Import trace analysis functions from frequencies.py  
    script_dir = Path(__file__).parent
    import importlib.util
    freq_spec = importlib.util.spec_from_file_location("frequencies", script_dir / "frequencies.py")
    freq_mod = importlib.util.module_from_spec(freq_spec)
    freq_spec.loader.exec_module(freq_mod)
    
    cmd_list = _normalize_cmd_list(cmd_list)
    warmup_s, window_s = _resolve_feature_window(feature_warmup_s, feature_window_s)
    timed_cmd = _build_timed_cmd(cmd_list, window_s)
    sample_s = window_s if window_s > 0 else 3.0

    if output_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="intel_pt_"))
    else:
        work_dir = Path(output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    perf_data_file = work_dir / "pt_trace.data"
    dis_file = work_dir / "pt_trace_dis.txt"
    
    print(f"\n{'='*70}")
    if target_pid is None:
        print(f"[Intel PT] Recording: {' '.join(cmd_list)}")
    else:
        print(f"[Intel PT] Recording system-wide for external workload for {sample_s:g}s")
    if warmup_s > 0 or window_s > 0:
        print(f"[Intel PT] Window mode: warmup={warmup_s:.1f}s, window={window_s:.1f}s")
    print(f"{'='*70}")

    # Optional warmup phase.
    if warmup_s > 0:
        if target_pid is None:
            # For command-based mode, warmup runs workload without trace.
            warmup_cmd = _build_timed_cmd(cmd_list, warmup_s)
            print(f"[Intel PT] Warmup (no trace): {' '.join(warmup_cmd)}")
            try:
                warmup_timeout_s = int(max(30, warmup_s + 20))
                warmup_res = subprocess.run(
                    warmup_cmd,
                    capture_output=True,
                    text=True,
                    timeout=warmup_timeout_s,
                )
                # timeout(1) returns 124 when it stops the command as intended.
                if warmup_res.returncode not in (0, 124):
                    print(f"[Intel PT] Warmup failed (code {warmup_res.returncode}); aborting trace collection")
                    if warmup_res.stderr:
                        print(f"[Intel PT] warmup stderr: {warmup_res.stderr[:500]}")
                    return {}
            except subprocess.TimeoutExpired:
                print("[Intel PT] Warmup timed out unexpectedly; aborting trace collection")
                return {}
            except Exception as e:
                print(f"[Intel PT] Warmup error: {e}")
                return {}
        else:
            time.sleep(warmup_s)
    
    # Step 1: Record Intel PT trace
    # Start perf record in background, then run the benchmark command, then stop perf
    if target_pid is None:
        perf_cmd = [
            "sudo", "perf", "record",
            "-e", "intel_pt//u",
            "-a",
            "-o", str(perf_data_file),
            "--"
        ] + timed_cmd
    else:
        perf_cmd = [
            "sudo", "perf", "record",
            "-e", "intel_pt//u",
            "-p", str(__import__("_hot_pid_resolver").resolve_hot_pid(target_pid)),
            "-o", str(perf_data_file),
            "--",
            "sleep",
            f"{sample_s:g}",
        ]
    
    print(f"[Intel PT] Command: {' '.join(perf_cmd)}")
    
    try:
        start = time.time()
        # Safety timeout: recording duration + generous flush budget.
        # For system-wide (-a) the data file can be very large so give 90 s of
        # post-recording flush time; for -p (target_pid) 60 s is ample.
        flush_budget = 60 if target_pid is not None else 90
        perf_timeout_s = int(max(120, warmup_s + window_s + flush_budget))
        result = subprocess.run(
            perf_cmd,
            capture_output=True,
            text=True,
            timeout=perf_timeout_s,
        )
        elapsed = time.time() - start
        
        if result.returncode not in (0, 124):
            print(f"[Intel PT] perf record failed (code {result.returncode})")
            if result.stderr:
                print(f"[Intel PT] stderr: {result.stderr[:500]}")
            if not perf_data_file.exists() or perf_data_file.stat().st_size == 0:
                return {}
        
        print(f"[Intel PT] Recording complete in {elapsed:.2f}s")
        
        if not perf_data_file.exists() or perf_data_file.stat().st_size == 0:
            print("[Intel PT] No trace data recorded")
            return {}
        
    except subprocess.TimeoutExpired:
        print("[Intel PT] perf record timed out and was terminated")
        return {}
    except Exception as e:
        print(f"[Intel PT] Recording error: {e}")
        return {}
    
    # Step 2: Disassemble the trace.
    # Write output directly to file to avoid pipe-buffer deadlocks with large traces.
    DISASSEMBLE_MAX_S = 5
    print(f"[Intel PT] Disassembling trace (max {DISASSEMBLE_MAX_S}s)...")
    try:
        dis_cmd = [
            "sudo", "timeout", "--signal=TERM", "--kill-after=2",
            f"{DISASSEMBLE_MAX_S}",
            "perf", "script",
            "--insn-trace",
            "--xed",
            "-i", str(perf_data_file),
        ]
        with open(dis_file, "w", encoding="utf-8") as out_f:
            proc = subprocess.run(
                dis_cmd,
                stdout=out_f,
                stderr=subprocess.PIPE,
                text=True,
                timeout=DISASSEMBLE_MAX_S + 20,
            )
        
        # returncode 124 = `timeout` killed perf; partial output is still usable
        if proc.returncode not in (0, 124):
            # Check if we still got some output despite the error code
            if not dis_file.exists() or dis_file.stat().st_size == 0:
                print(f"[Intel PT] Disassembly failed (code {proc.returncode})")
                if proc.stderr:
                    print(f"[Intel PT] stderr: {proc.stderr[:500]}")
                return {}
        
        # Count lines in the output file
        line_count = 0
        if dis_file.exists():
            with open(dis_file, 'r') as f:
                for _ in f:
                    line_count += 1
        
        if line_count == 0:
            print("[Intel PT] No instruction lines produced")
            return {}
        
        print(f"[Intel PT] Disassembled {line_count} instruction lines")
        
    except subprocess.TimeoutExpired:
        # Python 30s safety net fired; still try to use whatever was written
        if dis_file.exists() and dis_file.stat().st_size > 0:
            print("[Intel PT] Disassembly hit Python safety timeout – using partial output")
        else:
            print("[Intel PT] Disassembly timed out with no output")
            return {}
    except Exception as e:
        print(f"[Intel PT] Disassembly error: {e}")
        return {}
    
    # Step 3: Parse trace and compute statistics
    print(f"[Intel PT] Computing trace statistics...")
    try:
        instrs, _fam_counts = freq_mod.parse_trace(str(dis_file))
        
        if not instrs:
            print("[Intel PT] No instructions parsed from trace")
            return {}
        
        blocks = freq_mod.group_blocks(instrs)
        per_block_counts, family_lists = freq_mod.family_counts_per_block(instrs, blocks)
        jumps, blens, pairs = freq_mod.compute_jump_sizes_with_blocklens(blocks)
        branches = freq_mod.classify_branches(instrs)
        taken_runs, not_taken_runs = freq_mod.compute_runs(branches)
        raw_dists = freq_mod.raw_dependency_distances(instrs)
        
        print(f"[Intel PT] {len(instrs)} instrs, {len(blocks)} blocks, "
              f"{len(jumps)} jumps, {len(raw_dists)} RAW deps")
        
    except Exception as e:
        print(f"[Intel PT] Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    # Step 4: Build percentile metrics dict (P50 only)
    percentile_values = [50]
    percentile_labels = ["P50"]
    
    datasets = {
        "block_size": [b[2] for b in blocks],
        "jumps": jumps if jumps else [0],
        "taken_runs": taken_runs if taken_runs else [0],
        "not_taken_runs": not_taken_runs if not_taken_runs else [0],
        "raw_distances": raw_dists if raw_dists else [0],
    }
    
    # Add instruction family distributions
    for fam, counts in family_lists.items():
        datasets[f"family::{fam}"] = counts
    
    # Compute percentiles and flatten into output dict
    metrics = {}
    for metric_name, values in datasets.items():
        if not values:
            values = [0]
        arr = np.asarray(values)
        try:
            pcts = np.percentile(arr, percentile_values, method="nearest")
        except TypeError:
            pcts = np.percentile(arr, percentile_values, interpolation="nearest")
        for label, val in zip(percentile_labels, pcts):
            metrics[f"{metric_name}_{label}"] = int(round(float(val)))
    
    print(f"[Intel PT] Computed {len(metrics)} percentile metrics")
    
    # The parsed trace can be hundreds of megabytes per candidate. Keep only
    # the derived metrics unless explicitly requested for debugging.
    if os.environ.get("KEEP_INTEL_PT_ARTIFACTS", "").strip().lower() not in {"1", "true", "yes"}:
        for trace_file in (perf_data_file, dis_file):
            try:
                if trace_file.exists():
                    trace_file.unlink()
            except OSError as exc:
                print(f"[Intel PT] Warning: could not remove {trace_file}: {exc}")
        print("[Intel PT] Cleaned up trace artifacts")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Record Intel PT traces for folly microbenchmark units")
    parser.add_argument('--folly-test-dir', default='/myd/DCPerf/benchmarks/sw_bench/wdl_sources/folly/folly/test')
    parser.add_argument('--output-dir', default='./pt_records')
    parser.add_argument('--benchmark', help='Filter to a specific benchmark (partial match)')
    parser.add_argument('--dry-run', action='store_true', help='Print commands instead of running')
    parser.add_argument(
        '--cmd',
        nargs=argparse.REMAINDER,
        default=None,
        help='Record Intel PT for a single command. Everything after --cmd is treated as the command. '
             'Example: --cmd sysbench cpu --cpu-max-prime=20000 --threads=8 --time=8 run'
    )

    args = parser.parse_args()

    outdir = Path(args.output_dir)
    if not outdir.is_absolute():
        outdir = Path.cwd() / outdir
    
    # Single-command mode
    if args.cmd:
        cmd_list = args.cmd
        # Strip leading '--' if present (argparse REMAINDER quirk)
        if cmd_list and cmd_list[0] == '--':
            cmd_list = cmd_list[1:]
        if not cmd_list:
            print("Error: --cmd requires a command to run")
            sys.exit(1)
        
        metrics = run_single_cmd_intel_pt(cmd_list, str(outdir))
        
        # Save metrics to JSON
        outdir.mkdir(parents=True, exist_ok=True)
        json_file = outdir / "intel_pt_metrics.json"
        with open(json_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"[Intel PT] Metrics saved to {json_file}")
        return
    
    # Original batch mode
    recorder = FollyIntelPTRecorder(args.folly_test_dir, str(outdir))
    recorder.run_all(args.benchmark, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
