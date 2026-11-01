#!/usr/bin/env python3
"""Resolve the busiest process launched by a cloud workload wrapper."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _Process:
    pid: int
    ppid: int
    session: int
    cpu_ticks: int
    name: str


def _read_process(pid: int) -> _Process | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        closing_paren = stat.rfind(")")
        if closing_paren < 0:
            return None
        name = stat[stat.find("(") + 1:closing_paren]
        fields = stat[closing_paren + 2:].split()
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            errors="replace"
        ).strip()
        return _Process(
            pid=pid,
            ppid=int(fields[1]),
            session=int(fields[3]),
            cpu_ticks=int(fields[11]) + int(fields[12]),
            name=cmdline or name,
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, OSError):
        return None


def _snapshot() -> dict[int, _Process]:
    processes: dict[int, _Process] = {}
    try:
        proc_entries = os.scandir("/proc")
    except OSError:
        return processes
    with proc_entries:
        for entry in proc_entries:
            if not entry.name.isdigit():
                continue
            process = _read_process(int(entry.name))
            if process is not None:
                processes[process.pid] = process
    return processes


def _candidate_pids(root_pid: int, processes: dict[int, _Process]) -> set[int]:
    root = processes.get(root_pid)
    if root is None:
        return set()

    candidates = {root_pid}
    changed = True
    while changed:
        changed = False
        for process in processes.values():
            if process.ppid in candidates and process.pid not in candidates:
                candidates.add(process.pid)
                changed = True

    # Cloud workloads are started with setsid(). Include reparented members that
    # remain in the workload session, such as daemonized service workers.
    candidates.update(
        process.pid
        for process in processes.values()
        if process.session == root.session
    )
    return candidates


def resolve_hot_pid(root_pid: int) -> int:
    """Return the workload process with the largest CPU-time increase."""
    root_pid = int(root_pid)
    first = _snapshot()
    candidates = _candidate_pids(root_pid, first)
    if not candidates:
        raise ProcessLookupError(f"workload PID {root_pid} is not running")

    sample_seconds = float(os.environ.get("HOT_PID_SAMPLE_SECONDS", "0.25"))
    time.sleep(max(0.01, sample_seconds))

    second = _snapshot()
    candidates.update(_candidate_pids(root_pid, second))
    live = [second[pid] for pid in candidates if pid in second]
    if not live:
        raise ProcessLookupError(f"workload PID {root_pid} exited during PID resolution")

    exclude_pattern = os.environ.get(
        "HOT_PID_EXCLUDE_REGEX",
        r"(^|/)(sudo|setsid|taskset|timeout|sleep|bash|sh)(\s|$)",
    )
    try:
        exclude_re = re.compile(exclude_pattern, re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"invalid HOT_PID_EXCLUDE_REGEX: {exc}") from exc

    eligible = [process for process in live if not exclude_re.search(process.name)]
    if not eligible:
        eligible = live

    prefer_pattern = os.environ.get("HOT_PID_PREFER_REGEX", "").strip()
    try:
        prefer_re = re.compile(prefer_pattern, re.IGNORECASE) if prefer_pattern else None
    except re.error as exc:
        raise ValueError(f"invalid HOT_PID_PREFER_REGEX: {exc}") from exc

    def rank(process: _Process) -> tuple[int, int, int, int]:
        previous_ticks = first.get(process.pid, process).cpu_ticks
        cpu_delta = max(0, process.cpu_ticks - previous_ticks)
        preferred = int(prefer_re is not None and bool(prefer_re.search(process.name)))
        return preferred, cpu_delta, process.cpu_ticks, -process.pid

    selected = max(eligible, key=rank)
    print(f"[pid-resolver] Selected PID {selected.pid}: {selected.name}", flush=True)
    return selected.pid
