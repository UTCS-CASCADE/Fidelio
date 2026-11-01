from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

DC_PERF_ROOT = Path(os.environ.get("DC_PERF_ROOT", "/myd/DCPerf"))
WDL_BENCH_ROOT = DC_PERF_ROOT / "benchmarks" / "wdl_bench"
FOLLY_SRC_ROOT = WDL_BENCH_ROOT / "wdl_sources" / "folly"
FOLLY_CMAKE = FOLLY_SRC_ROOT / "CMakeLists.txt"
FBTHRIFT_SRC_ROOT = WDL_BENCH_ROOT / "wdl_sources" / "fbthrift"
BUILD_ROOT = WDL_BENCH_ROOT / "wdl_build" / "build"
CODEC_SEED_ROOT = Path(
    os.environ.get("CODEC_SEED_ROOT", "/myd/BenchmarkReplication-codec-seeds")
)
DEFAULT_BUILD_TIMEOUT_SEC = int(os.environ.get("RUN_TIMEOUT_SEC", "1800"))

_BINARY_BUILD_TIMEOUT_OVERRIDES = {
    "ProtocolBench": int(os.environ.get("PROTOCOLBENCH_RUN_TIMEOUT_SEC", "3600")),
}


def resolve_build_timeout_sec(binary_name: str, default_timeout_sec: int | None = None) -> int:
    baseline = int(default_timeout_sec if default_timeout_sec is not None else DEFAULT_BUILD_TIMEOUT_SEC)
    override = _BINARY_BUILD_TIMEOUT_OVERRIDES.get(str(binary_name), baseline)
    return max(int(override), baseline)


def _cmake_cache_home_directory(build_dir: Path) -> Path | None:
    cache_path = build_dir / "CMakeCache.txt"
    if not cache_path.exists():
        return None
    try:
        for line in cache_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
                raw = line.split("=", 1)[1].strip()
                if raw:
                    return Path(raw).resolve()
    except Exception:
        return None
    return None


def build_dir_matches_source(build_dir: Path, source_root: Path) -> bool:
    home_dir = _cmake_cache_home_directory(build_dir)
    if home_dir is None:
        return False
    try:
        return home_dir == source_root.resolve()
    except Exception:
        return False


def _build_dir_supports_binary_target(build_dir: Path, source_root: Path, binary_name: str) -> bool:
    try:
        binary_path = (
            build_dir / "bin" / binary_name
            if "fbthrift" in build_dir.name.lower() or "fbthrift" in str(source_root).lower()
            else build_dir / binary_name
        )
        if binary_path.exists():
            return True
        makefile = build_dir / "Makefile"
        if makefile.exists():
            target_prefix = f"{binary_name}:"
            for line in makefile.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith(target_prefix):
                    return True
        build_ninja = build_dir / "build.ninja"
        if build_ninja.exists():
            needle = f" {binary_name}:"
            for line in build_ninja.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith(f"build {binary_name}:") or needle in line:
                    return True
    except Exception:
        return False
    return False


def resolve_preferred_build_dir(build_dir: Path, source_root: Path, binary_name: str | None = None) -> Path:
    build_dir = Path(build_dir)
    source_root = Path(source_root).resolve()

    if build_dir.name.endswith("_rag"):
        candidates = [build_dir, build_dir.with_name(build_dir.name[:-4])]
    else:
        candidates = [build_dir.with_name(f"{build_dir.name}_rag"), build_dir]

    sibling_dir = build_dir.resolve().parent
    name_prefixes = {candidate.name for candidate in candidates}
    if sibling_dir.exists():
        for entry in sorted(sibling_dir.iterdir()):
            if not entry.is_dir():
                continue
            if any(entry.name == prefix or entry.name.startswith(f"{prefix}_") for prefix in name_prefixes):
                candidates.append(entry)

    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)

    if binary_name:
        for candidate in ordered:
            if (
                candidate.exists()
                and build_dir_matches_source(candidate, source_root)
                and _build_dir_supports_binary_target(candidate, source_root, binary_name)
            ):
                return candidate

    for candidate in ordered:
        if candidate.exists() and build_dir_matches_source(candidate, source_root):
            return candidate

    if binary_name:
        for candidate in ordered:
            if candidate.exists() and _build_dir_supports_binary_target(candidate, source_root, binary_name):
                return candidate

    for candidate in ordered:
        if candidate.exists():
            return candidate

    return ordered[0]


FOLLY_BUILD_DIR = resolve_preferred_build_dir(BUILD_ROOT / "folly", FOLLY_SRC_ROOT)
FBTHRIFT_BUILD_DIR = resolve_preferred_build_dir(BUILD_ROOT / "fbthrift", FBTHRIFT_SRC_ROOT)


def _parse_sources_token_line(text: str) -> list[str]:
    out: list[str] = []
    for token in text.split():
        if token.endswith((".cpp", ".cc", ".cxx", ".c")):
            out.append(token)
    return out


def parse_benchmark_binary_sources(cmake_path: Path = FOLLY_CMAKE) -> dict[str, dict[str, Any]]:
    lines = cmake_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    current_dir = Path("folly")
    result: dict[str, dict[str, Any]] = {}
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        dir_match = re.match(r"^DIRECTORY\s+(.+?)/?\s*$", stripped)
        if dir_match:
            current_dir = Path("folly") / Path(dir_match.group(1).strip())
            i += 1
            continue

        bench_match = re.match(r"^BENCHMARK\s+(\S+)(.*)$", stripped)
        if not bench_match:
            i += 1
            continue

        binary_name = bench_match.group(1)
        stanza: list[str] = [stripped]
        j = i + 1
        while j < len(lines):
            next_stripped = lines[j].strip()
            if re.match(r"^(DIRECTORY|TEST|BENCHMARK)\b", next_stripped):
                break
            stanza.append(next_stripped)
            j += 1

        sources: list[str] = []
        capture_sources = False
        for stanza_line in stanza:
            if not stanza_line or stanza_line.startswith("#"):
                continue
            if "SOURCES" in stanza_line:
                after = stanza_line.split("SOURCES", 1)[1].strip()
                if after:
                    sources.extend(_parse_sources_token_line(after))
                    capture_sources = False
                else:
                    capture_sources = True
                continue
            if capture_sources:
                if re.match(
                    r"^(HEADERS|DEPENDS|EXTRA_ARGS|CONTENT_DIR|WINDOWS_DISABLED|APPLE_DISABLED|BROKEN|SLOW|HANGING)\b",
                    stanza_line,
                ):
                    capture_sources = False
                    continue
                sources.extend(_parse_sources_token_line(stanza_line))

        if sources:
            result[binary_name] = {
                "source_root": FOLLY_SRC_ROOT,
                "build_dir": FOLLY_BUILD_DIR,
                "source_rel_paths": [current_dir / src for src in sources],
                "build_timeout_sec": resolve_build_timeout_sec(binary_name),
            }
        i = j

    result.update(
        {
            "VarintUtilsBench": {
                "source_root": FBTHRIFT_SRC_ROOT,
                "build_dir": FBTHRIFT_BUILD_DIR,
                "source_rel_paths": [Path("thrift/lib/cpp/util/test/VarintUtilsBench.cpp")],
                "build_timeout_sec": resolve_build_timeout_sec("VarintUtilsBench"),
            },
            "ProtocolBench": {
                "source_root": FBTHRIFT_SRC_ROOT,
                "build_dir": FBTHRIFT_BUILD_DIR,
                "source_rel_paths": [Path("thrift/lib/cpp2/test/ProtocolBench.cpp")],
                "build_timeout_sec": resolve_build_timeout_sec("ProtocolBench"),
            },
        }
    )
    ffmpeg_root = CODEC_SEED_ROOT / "ffmpeg"
    ffmpeg_build = ffmpeg_root / "build"
    ffmpeg_units = (
        "blockdsp",
        "fdctdsp",
        "h264chroma",
        "h264dsp",
        "h264pred",
        "h264qpel",
        "hevc_deblock",
        "hevc_dequant",
        "hevc_idct",
        "hevc_pel",
        "hevc_pred",
        "hevc_sao",
        "idctdsp",
        "motion",
        "mpegvideo_unquantize",
        "mpegvideoencdsp",
        "vp9dsp",
    )
    if (ffmpeg_build / "tests/checkasm/checkasm").exists():
        result["ffmpeg_checkasm"] = {
            "source_root": ffmpeg_root,
            "build_dir": ffmpeg_build,
            "source_rel_paths": [Path("tests/checkasm") / f"{unit}.c" for unit in ffmpeg_units],
            "binary_path": ffmpeg_build / "tests/checkasm/checkasm",
            "build_system": "ffmpeg_make",
            "build_timeout_sec": resolve_build_timeout_sec("ffmpeg_checkasm"),
        }

    dav1d_root = CODEC_SEED_ROOT / "dav1d"
    dav1d_build = dav1d_root / "build"
    dav1d_units = (
        "msac",
        "pal",
        "refmvs",
        "cdef",
        "ipred",
        "itx",
        "loopfilter",
        "looprestoration",
        "mc",
    )
    if (dav1d_build / "tests/checkasm").exists():
        result["dav1d_checkasm"] = {
            "source_root": dav1d_root,
            "build_dir": dav1d_build,
            "source_rel_paths": [Path("tests/checkasm") / f"{unit}.c" for unit in dav1d_units],
            "binary_path": dav1d_build / "tests/checkasm",
            "build_system": "ninja",
            "build_timeout_sec": resolve_build_timeout_sec("dav1d_checkasm"),
        }

    x264_root = CODEC_SEED_ROOT / "x264"
    x264_build = x264_root / "build"
    for bit_depth in (8, 10):
        binary_name = f"x264_checkasm_{bit_depth}bit"
        binary_path = x264_build / f"checkasm{bit_depth}"
        if binary_path.exists():
            result[binary_name] = {
                "source_root": x264_root,
                "build_dir": x264_build,
                "source_rel_paths": [Path("tools/checkasm.c")],
                "binary_path": binary_path,
                "build_system": "x264_make",
                "build_target": f"checkasm{bit_depth}",
                "build_timeout_sec": resolve_build_timeout_sec(binary_name),
            }
    svt_root = CODEC_SEED_ROOT / "SVT-AV1"
    svt_build = svt_root / "build"
    svt_binary = svt_root / "Bin" / "Release" / "SvtAv1EncApp"
    if (svt_root / "CMakeLists.txt").exists() and svt_build.exists():
        result["svt_av1_encoder"] = {
            "source_root": svt_root,
            "build_dir": svt_build,
            "source_rel_paths": [
                Path("Source/Lib/Codec/motion_estimation.c"),
                Path("Source/Lib/Codec/md_process.c"),
                Path("Source/Lib/Codec/coding_loop.c"),
                Path("Source/Lib/Codec/full_loop.c"),
                Path("Source/Lib/Codec/rc_crf_cqp.c"),
                Path("Source/Lib/Codec/ec_process.c"),
                Path("Source/Lib/Codec/temporal_filtering.c"),
                Path("Source/Lib/Codec/src_ops_process.c"),
            ],
            "binary_path": svt_binary,
            "build_system": "svt_ninja",
            "build_target": "SvtAv1EncApp",
            "seed_kind": "application",
            "evaluation_mode": "pid_attached",
            "build_timeout_sec": resolve_build_timeout_sec("svt_av1_encoder"),
        }
    return result
