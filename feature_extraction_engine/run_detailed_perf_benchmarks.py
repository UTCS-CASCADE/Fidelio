#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Advanced script to run folly microbenchmark units with perf stat
# Automatically discovers benchmarks and measures each unit for its entire duration

import json
import subprocess
import os
import sys
import re
import time
import shlex
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import argparse
import tempfile

class FollyBenchmarkPerfRunner:
    """Run folly benchmark units and generic binaries with perf stat measurements."""
    
    def __init__(self, folly_test_dir: str, output_dir: str, perf_events: str, 
                 bench_bin_dir: Optional[str] = None, additional_paths: Optional[List[str]] = None,
                 binary_pattern: str = 'bench_bin_*'):
        self.folly_test_dir = Path(folly_test_dir)
        self.output_dir = Path(output_dir)
        self.perf_events = perf_events
        self.bench_bin_dir = Path(bench_bin_dir) if bench_bin_dir else None
        self.additional_paths = [Path(p) for p in (additional_paths or [])]
        self.binary_pattern = binary_pattern
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.benchmarks = {}  # benchmark_name -> binary_path
        self.benchmark_configs = {}  # benchmark_name -> config_string
        self.benchmark_type = {}  # benchmark_name -> 'folly' or 'generic'
        self.discover_benchmarks()
    
    def _unit_output_exists(self, benchmark_name: str, unit_name: str) -> bool:
        """Check if output file already exists for this benchmark/unit pair."""
        safe_unit_name = re.sub(r'[/\\:*?"<>|]', '_', unit_name)
        binary_dir = self.output_dir / benchmark_name
        perf_output_file = binary_dir / f"{benchmark_name}_{safe_unit_name}_perf.txt"
        return perf_output_file.exists()

    def load_probe_times(self, benchmark_name: str) -> Dict[str, float]:
        """Load cached probe times (unit durations) from disk if they exist.
        
        Returns dict of unit_name -> duration_in_seconds, or empty dict if not found.
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
                    parts = line.rsplit(maxsplit=1)
                    if len(parts) == 2:
                        unit_name = parts[0]
                        try:
                            duration = float(parts[1])
                            probe_times[unit_name] = duration
                        except ValueError:
                            continue
            if probe_times:
                print(f"  Loaded cached probe times from {probe_file.name} ({len(probe_times)} units)")
        except Exception as e:
            print(f"  Error loading cached probe times: {e}")
        
        return probe_times

    def save_probe_times(self, benchmark_name: str, probe_times: Dict[str, float]) -> Path:
        """Save probe times (unit durations) to disk as {bench_name}_probe_times.txt.
        
        probe_times should be dict of unit_name -> duration_in_seconds
        
        Returns path to the saved file.
        """
        if not probe_times:
            return None
        
        binary_dir = self.output_dir / benchmark_name
        binary_dir.mkdir(parents=True, exist_ok=True)
        probe_file = binary_dir / f"{benchmark_name}_probe_times.txt"
        with open(probe_file, 'w') as f:
            for unit_name in sorted(probe_times.keys()):
                duration = probe_times[unit_name]
                f.write(f"{unit_name} {duration:.6f}\n")
        
        return probe_file

    def discover_benchmarks(self):
        """Discover all benchmark binaries in configured directories and patterns."""
        print(f"Scanning for benchmark binaries...")

        # 1. Discover folly/wdl_bench benchmarks
        possible_paths = [
            Path("/myd/DCPerf/benchmarks/pmu_bench"),
        ]

        wdl_build_base = None
        for path in possible_paths:
            if path.exists():
                test_files = list(path.glob("memcpy_benchmark")) + list(path.glob("*benchmark"))
                if test_files:
                    wdl_build_base = path
                    break

        if wdl_build_base is None:
            print("No folly benchmark directory found. Searched:")
            for p in possible_paths:
                print(f"  - {p}")
        else:
            print(f"Found folly benchmark base: {wdl_build_base}")
            print("Scanning for executables in:", wdl_build_base)

            for binary in sorted(wdl_build_base.glob("*")):
                if binary.is_file() and os.access(binary, os.X_OK):
                    name = binary.name
                    if any(p in name.lower() for p in ["benchmark", "bench", "perf"]):
                        self.benchmarks[name] = binary
                        self.benchmark_type[name] = 'folly'
                        print(f"Discovered folly benchmark: {name} -> {binary}")
        
        # 2. Discover generic binaries from configured paths
        search_paths = [self.bench_bin_dir] + self.additional_paths
        search_paths = [p for p in search_paths if p is not None and p.exists()]
        
        if search_paths:
            print(f"\nScanning for binaries matching pattern '{self.binary_pattern}' in:")
            for search_path in search_paths:
                print(f"  - {search_path}")
            
            for search_path in search_paths:
                for binary in sorted(search_path.glob(self.binary_pattern)):
                    if binary.is_file() and os.access(binary, os.X_OK):
                        name = binary.name
                        # Skip if already discovered as folly benchmark
                        if name not in self.benchmarks:
                            self.benchmarks[name] = binary
                            self.benchmark_type[name] = 'generic'
                            print(f"Discovered generic benchmark: {name} -> {binary}")
        else:
            print(f"\nNo additional benchmark paths configured.")
    
    def run_benchmark_get_units(self, benchmark_name: str) -> Dict[str, float]:
        """Run benchmark and extract unit names with their estimated throughput.
        
        For folly benchmarks: uses --bm_list or --json flags
        For generic binaries: treats the entire binary as a single unit
        
        Returns dict mapping unit_name -> throughput (iters/sec)
        """
        
        if benchmark_name not in self.benchmarks:
            return {}

        benchmark_binary = self.benchmarks[benchmark_name]
        benchmark_type = self.benchmark_type.get(benchmark_name, 'folly')
        benchmark_dir = benchmark_binary.parent

        print(f"\n{'='*70}")
        print(f"Extracting units from: {benchmark_name} ({benchmark_type})")
        print(f"{'='*70}")

        # For generic binaries, just treat as a single unit
        if benchmark_type == 'generic':
            return {benchmark_name: 1e6}  # Default throughput estimate
        
        # For folly benchmarks, try to extract units
        # 1) Prefer --bm_list if available, then probe each unit with --benchmark --bm_regex '^unit$'
        try:
            list_cmd = [str(benchmark_binary), "--bm_list"]
            list_res = subprocess.run(
                list_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=benchmark_dir
            )

            stdout = (list_res.stdout or "").strip()
            if stdout:
                unit_lines = [l.strip() for l in stdout.splitlines() if l.strip()]
                if unit_lines:
                    print(f"  Found {len(unit_lines)} units via --bm_list")
                    units: Dict[str, float] = {}
                    for unit in unit_lines:
                        print(f"  Probing unit: {unit}")
                        probe_cmd = [str(benchmark_binary), "--benchmark", "--bm_regex", f'^{re.escape(unit)}$']
                        try:
                            p = subprocess.run(
                                probe_cmd,
                                capture_output=True,
                                text=True,
                                timeout=30,
                                cwd=benchmark_dir
                            )

                            # Try to parse textual output for throughput
                            parsed = self._extract_units_from_text(p.stdout)
                            if parsed:
                                # Prefer a parsed key that matches the unit name
                                matched = None
                                for k in parsed.keys():
                                    if unit in k or k in unit:
                                        matched = k
                                        break
                                if matched is None:
                                    matched = next(iter(parsed.keys()))
                                units[unit] = parsed.get(matched, 1e6)
                            else:
                                # Fallback estimate
                                units[unit] = 1e6
                        except subprocess.TimeoutExpired:
                            print(f"    probe timed out for unit: {unit}")
                            units[unit] = 1e6
                        except Exception as e:
                            print(f"    probe error for unit {unit}: {e}")
                            units[unit] = 1e6

                    print(f"  Probed {len(units)} units via --bm_list")
                    return units

        except subprocess.TimeoutExpired:
            print("  --bm_list timed out; falling back")
        except Exception as e:
            print(f"  --bm_list failed: {e}; falling back")

        # 2) Fallback: try --json (older behavior)
        try:
            cmd = [str(benchmark_binary), "--json"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=benchmark_dir
            )

            units = {}
            try:
                data = json.loads(result.stdout)
                units = self._extract_units_from_json(data)
            except json.JSONDecodeError:
                units = self._extract_units_from_text(result.stdout)

            print(f"Found {len(units)} benchmark units (fallback)")
            return units

        except subprocess.TimeoutExpired:
            print(f"  Warning: Benchmark timed out")
            return {}
        except Exception as e:
            print(f"  Error: {e}")
            return {}
    
    def _extract_units_from_json(self, data: dict) -> Dict[str, float]:
        """Extract unit names and throughput from JSON data."""
        
        units = {}
        
        if isinstance(data, dict):
            if 'benchmarks' in data:
                # Google Benchmark format
                for bench in data['benchmarks']:
                    if 'name' in bench:
                        name = bench['name']
                        # Try to extract throughput (items_per_second or similar)
                        throughput = bench.get('items_per_second', bench.get('iterations_per_second', 1e6))
                        units[name] = float(throughput) if throughput else 1e6
            else:
                # Direct key-value format (folly-style)
                for key, value in data.items():
                    if not key.startswith('%') and isinstance(value, (int, float)):
                        units[key] = float(value)
        
        return units
    
    def _extract_units_from_text(self, text: str) -> Dict[str, float]:
        """Extract unit names from text output using regex."""
        
        units = {}
        
        # Pattern to match benchmark results like:
        # bench(0_to_1024_COLD_folly)                                     2.21ns   452.91M
        pattern = r'([a-zA-Z0-9_\-().]+)\s+[\d.]+\s+[a-z]+\s+([\d.]+)([KMGT]?)'
        
        for match in re.finditer(pattern, text, re.MULTILINE):
            unit_name = match.group(1)
            throughput_val = float(match.group(2))
            multiplier = match.group(3)
            
            # Convert to per-second
            multipliers = {'K': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12}
            throughput = throughput_val * multipliers.get(multiplier, 1)
            
            if unit_name and throughput > 0:
                units[unit_name] = throughput
        
        return units
    
    def calculate_measurement_duration(self, throughput: float, target_iterations: int = 100000) -> float:
        """Calculate measurement duration based on throughput.
        
        Ensures we capture at least target_iterations worth of data.
        throughput: iterations/second
        """
        
        if throughput <= 0:
            return 0.5  # Default 500ms
        
        # Duration = (target iterations / throughput)
        duration = target_iterations / throughput
        
        # Clamp to reasonable range (50ms - 60 seconds)
        duration = max(0.05, min(60.0, duration))
        
        return duration
    
    def run_unit_with_perf(self, benchmark_binary: Path, unit_name: str,
                          throughput: float = 0) -> Tuple[bool, str, float]:
        """Run a single benchmark unit with perf stat for appropriate duration.
        
        For folly benchmarks: runs with --bm_pattern flag
        For generic binaries: runs the binary directly
        
        Returns (success, output_file, elapsed_time)
        """
        
        benchmark_name = benchmark_binary.name
        benchmark_type = self.benchmark_type.get(benchmark_name, 'folly')
        benchmark_dir = benchmark_binary.parent
        safe_unit_name = re.sub(r'[/\\:*?"<>|]', '_', unit_name)
        
        # Create binary-specific output directory
        binary_out_dir = self.output_dir / benchmark_name
        binary_out_dir.mkdir(parents=True, exist_ok=True)
        
        perf_output_file = binary_out_dir / f"{benchmark_name}_{safe_unit_name}_perf.txt"
        perf_json_file = binary_out_dir / f"{benchmark_name}_{safe_unit_name}_perf.json"
        
        # Check if already completed (output file exists)
        if perf_output_file.exists():
            print(f"  {unit_name:<65} [SKIPPED - already completed]", flush=True)
            return (True, "", 0.0)
        
        # Build benchmark command based on type
        if benchmark_type == 'generic':
            # For generic binaries, just run the binary directly
            benchmark_cmd = [str(benchmark_binary)]
        else:
            # For folly benchmarks, use --bm_pattern to run specific unit
            benchmark_cmd = [str(benchmark_binary), "--bm_regex", f"^{re.escape(unit_name)}$"]
        
        # Build perf stat command
        perf_cmd = [
            "sudo", "perf", "stat",
            "-a",  # All CPUs
            "-e", self.perf_events,
        ]
        
        # Estimate measurement duration
        duration = self.calculate_measurement_duration(throughput, target_iterations=50000)
        timeout = max(5, duration * 3)  # 3x safety margin
        
        # Print progress
        throughput_str = f"{throughput/1e6:.1f}M" if throughput >= 1e6 else f"{throughput/1e3:.1f}K"
        duration_str = f"{duration*1000:.0f}ms" if duration < 1 else f"{duration:.2f}s"
        
        print(f"  {unit_name:<65} [{throughput_str:>8} iter/s, ~{duration_str:>6}]", end=" ", flush=True)
        
        # Print the complete command that will be executed
        full_cmd = perf_cmd + benchmark_cmd
        print(f"\n    Command: {' '.join(full_cmd)}", flush=True)
        
        start_time = time.time()
        
        try:
            # Run with perf stat from the benchmark directory
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                cwd=benchmark_dir,  # Run from benchmark directory
                timeout=timeout
            )
            
            elapsed = time.time() - start_time
            
            # Save text output (perf stat output goes to stderr)
            perf_output = result.stderr if result.stderr else result.stdout
            
            with open(perf_output_file, 'w') as f:
                f.write(f"Benchmark: {benchmark_name}\n")
                f.write(f"Type: {benchmark_type}\n")
                f.write(f"Unit: {unit_name}\n")
                f.write(f"Throughput estimate: {throughput:.2e} iters/s\n")
                f.write(f"Target duration: {duration:.3f}s\n")
                f.write(f"Actual elapsed: {elapsed:.3f}s\n")
                f.write(f"\nCommand: {' '.join(benchmark_cmd)}\n\n")
                f.write(f"Perf output:\n{perf_output}\n")
                if result.stdout:
                    f.write(f"\nBenchmark output:\n{result.stdout}\n")
            
            print(f"✓ ({elapsed:.2f}s)")
            return (True, str(perf_output_file), elapsed)
        
        except PermissionError:
            print("✗ (need sudo)")
            return (False, "", 0)
        except Exception as e:
            print(f"✗ ({e})")
            return (False, "", 0)
    
    def run_all_benchmarks(self, benchmark_filter: Optional[str] = None):
        """Run all discovered benchmarks with perf measurements."""
        
        if not self.benchmarks:
            print("No benchmarks found!")
            return
        
        # Filter benchmarks if requested
        benchmarks_to_run = self.benchmarks
        if benchmark_filter:
            benchmarks_to_run = {
                k: v for k, v in self.benchmarks.items()
                if benchmark_filter.lower() in k.lower()
            }
        
        total_start = time.time()
        summary = {}
        
        for benchmark_name in sorted(benchmarks_to_run.keys()):
            benchmark_binary = benchmarks_to_run[benchmark_name]
            
            # Create binary-specific output directory
            binary_out_dir = self.output_dir / benchmark_name
            binary_out_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if we have cached probe times - if so, use them and skip probing
            probe_cache_file = binary_out_dir / f"{benchmark_name}_probe_times.txt"
            units = {}
            cached_probe_times = {}
            
            if probe_cache_file.exists():
                # Load cached probe times and extract units from it
                # Format: unit_name duration_in_seconds
                num_cached_units = 0
                with open(probe_cache_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.rsplit(maxsplit=1)
                        if len(parts) == 2:
                            unit_name = parts[0]
                            try:
                                duration = float(parts[1])
                                units[unit_name] = 1e6  # Default throughput estimate
                                cached_probe_times[unit_name] = duration
                                num_cached_units += 1
                            except ValueError:
                                continue
                
                print(f"\n{benchmark_name}: Loaded {num_cached_units} cached units from probe times")
                
                # Check if all units already have output files (complete)
                completed_count = 0
                for unit in units.keys():
                    if self._unit_output_exists(benchmark_name, unit):
                        completed_count += 1
                
                if completed_count == len(units):
                    print(f"  All {len(units)} units already completed (output files exist) - skipping entirely")
                    benchmark_results = {
                        'total_units': len(units),
                        'successful': 0,
                        'failed': 0,
                        'skipped': len(units),
                        'total_time': 0
                    }
                    summary[benchmark_name] = benchmark_results
                    continue
                else:
                    print(f"  {len(units)} cached units, {completed_count} already completed, {len(units) - completed_count} need measuring")
            else:
                # No cache - probe normally
                units = self.run_benchmark_get_units(benchmark_name)
            
            if not units:
                print(f"  (no units found or benchmark failed)")
                continue
            
            # Check if all units have output files
            completed_count = sum(1 for unit in units.keys() if self._unit_output_exists(benchmark_name, unit))
            if completed_count == len(units):
                print(f"\n{benchmark_name}: All {len(units)} units already completed - skipping")
                benchmark_results = {
                    'total_units': len(units),
                    'successful': 0,
                    'failed': 0,
                    'skipped': len(units),
                    'total_time': 0
                }
                summary[benchmark_name] = benchmark_results
                continue
            
            benchmark_results = {
                'total_units': len(units),
                'successful': 0,
                'failed': 0,
                'skipped': 0,
                'total_time': 0
            }
            
            # Dict to accumulate probe times for this benchmark
            probe_times_dict = {}
            
            # Run each unit with perf
            for unit_name in sorted(units.keys()):
                # Use cached duration if available, otherwise use calculated
                if unit_name in cached_probe_times:
                    unit_duration = cached_probe_times[unit_name]
                else:
                    unit_duration = units[unit_name]
                
                success, output_file, elapsed = self.run_unit_with_perf(
                    benchmark_binary,
                    unit_name,
                    unit_duration
                )
                
                # Record the actual elapsed time for future caching
                if elapsed > 0:
                    probe_times_dict[unit_name] = elapsed
                
                benchmark_results['total_time'] += elapsed
                if output_file == "":
                    # Skipped (already completed)
                    benchmark_results['skipped'] += 1
                elif success:
                    benchmark_results['successful'] += 1
                else:
                    benchmark_results['failed'] += 1
            
            # Save probe times for this benchmark if we ran any units
            if probe_times_dict:
                probe_file = self.save_probe_times(benchmark_name, probe_times_dict)
                print(f"  Saved probe times to {probe_file.name}")
            
            summary[benchmark_name] = benchmark_results
        
        # Print final summary
        total_elapsed = time.time() - total_start
        
        print(f"\n{'='*70}")
        print("MEASUREMENT SUMMARY")
        print(f"{'='*70}\n")
        
        total_units = 0
        total_success = 0
        total_failed = 0
        total_skipped = 0
        
        for benchmark_name, results in summary.items():
            print(f"{benchmark_name}:")
            print(f"  Total units: {results['total_units']}")
            if results['skipped']:
                print(f"  Already completed: {results['skipped']}")
            print(f"  Successful: {results['successful']}")
            if results['failed']:
                print(f"  Failed: {results['failed']}")
            print(f"  Time: {results['total_time']:.1f}s")
            print()
            
            total_units += results['total_units']
            total_success += results['successful']
            total_failed += results['failed']
            total_skipped += results['skipped']
        
        print(f"{'='*70}")
        print(f"Total units measured: {total_success}/{total_units}")
        if total_skipped:
            print(f"Skipped (already done): {total_skipped}")
        if total_failed:
            print(f"Failed: {total_failed}")
        print(f"Total elapsed time: {total_elapsed/60:.1f} minutes ({total_elapsed:.0f}s)")
        print(f"Results directory: {self.output_dir}")
        print(f"{'='*70}\n")

def parse_perf_stat_output(text, elapsed=None):
    """Parse perf stat stderr output into a dict of metrics.
    
    Args:
        text: The stderr output from perf stat.
        elapsed: Optional wall-clock elapsed time (seconds). If not provided,
                 parsed from perf output.
    
    Returns:
        Dict of metric_name -> numeric_value.
    """
    metrics = {}
    event_counts = {}
    
    # Mapping raw event codes to human-readable names
    event_name_map = {"r2424": "L2-icache-load-misses"}
    
    ipc_found = False
    for line in text.splitlines():
        # Pattern: <count> <event-name> [optional annotations like # 1.22 insn per cycle (58.41%)]
        m = re.match(r'\s*([\d,]+)\s+(\S+)', line)
        if m:
            count_str = m.group(1).replace(',', '')
            event_name = m.group(2)
            try:
                count = int(count_str)
                output_name = event_name_map.get(event_name, event_name)
                event_counts[output_name] = count
            except ValueError:
                pass
        
        # Parse IPC from the annotation: # 1.22 insn per cycle
        ipc_match = re.search(r'#\s+([\d.]+)\s+insn per cycle', line)
        if ipc_match and not ipc_found:
            metrics["IPC"] = float(ipc_match.group(1))
            ipc_found = True
    
    # Fall back to count-based IPC if annotation is unavailable.
    if "IPC" not in metrics:
        cycles = event_counts.get("cycles", 0)
        instructions = event_counts.get("instructions", 0)
        if cycles > 0 and instructions > 0:
            metrics["IPC"] = round(float(instructions) / float(cycles), 6)

    # Convert all hardware counter totals into MPKI and drop raw nominal counts.
    instructions = event_counts.get("instructions", 0)
    if instructions > 0:
        for event_name, count in event_counts.items():
            if event_name in {"cycles", "instructions"}:
                continue
            metrics[f"{event_name}_MPKI"] = round((float(count) * 1000.0) / float(instructions), 6)
    
    return metrics


def _resolve_feature_window(feature_warmup_s: Optional[float], feature_window_s: Optional[float]) -> Tuple[float, float]:
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


def _build_windowed_cmd(cmd_list: List[str], warmup_s: float, window_s: float) -> List[str]:
    """Wrap command with optional warmup sleep and timeout window."""
    if warmup_s <= 0 and window_s <= 0:
        return cmd_list

    quoted_cmd = " ".join(shlex.quote(tok) for tok in cmd_list)
    run_part = f"timeout {window_s:g} {quoted_cmd}" if window_s > 0 else quoted_cmd

    if warmup_s > 0:
        shell_cmd = f"sleep {warmup_s:g}; exec {run_part}"
    else:
        shell_cmd = f"exec {run_part}"

    return ["bash", "-lc", shell_cmd]


def run_single_cmd_perf_stat(
    cmd_list,
    perf_events=None,
    output_dir=None,
    feature_warmup_s: Optional[float] = None,
    feature_window_s: Optional[float] = None,
    target_pid: Optional[int] = None,
):
    """Run perf stat on a single arbitrary command and return parsed metrics dict.
    
    Args:
        cmd_list: List of strings representing the command to run.
        perf_events: Comma-separated string of PMU events. Uses default set if None.
        output_dir: Optional directory to save raw perf output.
    
    Returns:
        Dict of metric_name -> numeric_value (IPC + *_MPKI metrics).
    """
    if perf_events is None:
        perf_events = ('cycles,instructions,L1-icache-load-misses,iTLB-load-misses,'
                       'dTLB-load-misses,'
                       'L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,'
                       'LLC-load-misses,branch-load-misses,branch-misses,r2424')
    
    warmup_s, window_s = _resolve_feature_window(feature_warmup_s, feature_window_s)
    sample_s = window_s if window_s > 0 else 3.0
    if target_pid is None:
        effective_cmd = _build_windowed_cmd(cmd_list, warmup_s, window_s)
        _perproc = __import__("os").environ.get("SEG08_PERPROC") == "1"
        _base = ["sudo", "perf", "stat"] + ([] if _perproc else ["-a"])
        perf_cmd = _base + ["-e", perf_events, "--"] + effective_cmd
    else:
        if warmup_s > 0:
            time.sleep(warmup_s)
        perf_cmd = ["sudo", "perf", "stat", "-p", str(__import__("_hot_pid_resolver").resolve_hot_pid(target_pid)), "-e", perf_events, "--", "sleep", f"{sample_s:g}"]
    
    print(f"\n{'='*70}")
    if target_pid is None:
        print(f"[perf stat] Running: {' '.join(cmd_list)}")
    else:
        print(f"[perf stat] Sampling system-wide for external workload for {sample_s:g}s")
    if warmup_s > 0 or window_s > 0:
        print(f"[perf stat] Window mode: warmup={warmup_s:.1f}s, window={window_s:.1f}s")
    print(f"[perf stat] Full command: {' '.join(perf_cmd)}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(perf_cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        # perf stat output goes to stderr
        perf_output = result.stderr if result.stderr else ""
        
        # Save raw output if output_dir specified
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            raw_file = out_dir / "perf_stat_raw.txt"
            with open(raw_file, 'w') as f:
                f.write(f"Command: {' '.join(cmd_list)}\n")
                f.write(f"Elapsed: {elapsed:.3f}s\n\n")
                f.write(perf_output)
                if result.stdout:
                    f.write(f"\n\nBenchmark stdout:\n{result.stdout}")
            print(f"[perf stat] Raw output saved to {raw_file}")
        
        metrics = parse_perf_stat_output(perf_output, elapsed)
        print(f"[perf stat] Collected {len(metrics)} metrics in {elapsed:.2f}s")
        return metrics
    
    except Exception as e:
        print(f"[perf stat] Error: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Run folly benchmark units and generic binaries with perf stat PMU measurements"
    )
    parser.add_argument(
        '--folly-test-dir',
        default='/users/alanuiuc/DCPerf/benchmarks/wdl_bench/wdl_sources/folly/folly/test',
        help='Path to folly test directory'
    )
    parser.add_argument(
        '--output-dir',
        default='./pmu_results',
        help='Output directory for perf results'
    )
    parser.add_argument(
        '--bench-bin-dir',
        default=None,
        help='Directory containing binaries to run'
    )
    parser.add_argument(
        '--additional-paths',
        nargs='*',
        default=[],
        help='Additional directories to scan for binaries (space-separated)'
    )
    parser.add_argument(
        '--binary-pattern',
        default='bench_bin_*',
        help='Pattern to match binary names (e.g., "bench_bin_*", "*_bench", "test_*")'
    )
    parser.add_argument(
        '--benchmark',
        help='Specific benchmark to run (can be partial match)'
    )
    parser.add_argument(
        '--events',
        default='cycles,instructions,L1-icache-load-misses,iTLB-load-misses,dTLB-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,LLC-load-misses,branch-load-misses,branch-misses,r2424',
        help='PMU events to measure'
    )
    parser.add_argument(
        '--cmd',
        nargs=argparse.REMAINDER,
        default=None,
        help='Run perf stat on a single command. Everything after --cmd is treated as the command. '
             'Example: --cmd sysbench cpu --cpu-max-prime=20000 --threads=8 --time=8 run'
    )
    parser.add_argument('--feature-warmup-seconds', type=float, default=0.0,
                        help='Delay before starting command in single-command mode (default: 0)')
    parser.add_argument('--feature-window-seconds', type=float, default=0.0,
                        help='Limit command runtime to N seconds in single-command mode (default: 0)')
    parser.add_argument('--cloud-mode', action='store_true',
                        help='Convenience mode for cloud workloads: warmup=30s, window=10s')
    
    args = parser.parse_args()
    
    # Single-command mode
    if args.cmd:
        cmd_list = args.cmd
        # Strip leading '--' if present (argparse REMAINDER quirk)
        if cmd_list and cmd_list[0] == '--':
            cmd_list = cmd_list[1:]
        if not cmd_list:
            print("Error: --cmd requires a command to run")
            sys.exit(1)
        
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir

        warmup_s = float(args.feature_warmup_seconds)
        window_s = float(args.feature_window_seconds)
        if args.cloud_mode:
            warmup_s = 30.0
            window_s = 10.0
        if warmup_s < 0 or window_s < 0:
            print("Error: --feature-warmup-seconds and --feature-window-seconds must be >= 0")
            sys.exit(1)

        os.environ["FEATURE_WARMUP_S"] = str(warmup_s)
        os.environ["FEATURE_WINDOW_S"] = str(window_s)
        
        metrics = run_single_cmd_perf_stat(
            cmd_list,
            args.events,
            str(output_dir),
            feature_warmup_s=warmup_s,
            feature_window_s=window_s,
        )
        
        # Save metrics to JSON
        output_dir.mkdir(parents=True, exist_ok=True)
        json_file = output_dir / "perf_stat_metrics.json"
        with open(json_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"[perf stat] Metrics saved to {json_file}")
        return
    
    # Original batch mode
    # Make output directory absolute if it's relative
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    
    runner = FollyBenchmarkPerfRunner(
        args.folly_test_dir,
        str(output_dir),
        args.events,
        args.bench_bin_dir,
        args.additional_paths,
        args.binary_pattern
    )
    
    runner.run_all_benchmarks(args.benchmark)

if __name__ == '__main__':
    main()
