#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./run_full_successful_edit_database.sh [options]

Batch-runs the full property_edit_tasks manifest through retrieval, rewrite,
compile/repair, and feature evaluation to populate successful_edits.

Options:
  --manifest PATH           Manifest to run
  --refresh-manifest        Regenerate the manifest before running so the latest prompt guidance is used
  --binary NAME             Restrict regenerated manifest to this binary (repeatable)
  --exclude-binary NAME     Exclude this binary from regenerated manifest (repeatable)
  --folly-build-dir PATH    Replacement build dir for Folly tasks
  --fbthrift-build-dir PATH Replacement build dir for fbthrift tasks
  --batch-size N            Grouped jobs per batch (default: 25)
  --start-index N           Start offset within matched grouped jobs (default: 0)
  --max-batches N           Stop after N batches (default: unlimited)
  --top-k N                 Retrieval top-k (default: 8)
  --max-child-attempts N    Batched generation attempts per child target (default: 5)
  --sample-large-binaries   Generate a diverse sampled manifest before running
  --small-binary-threshold N Keep all microbenchmarks at or below this per-binary count
  --large-binary-sample-cap N Maximum sampled microbenchmarks for large binaries
  --large-binary-sample-scale F sqrt-based scale factor for large-binary sample budgets
  --amd                     Skip Intel PT during evaluation
  --no-skip-existing        Re-run tasks even if evaluation.json exists
  --build-tuning-dataset    Export tuning/train.jsonl after batches finish
  -h, --help                Show this help

Environment:
  PYTHON
  LLM_PROVIDER
  VERTEX_PROJECT_ID
  VERTEX_LOCATION
  AWS_REGION / AWS_DEFAULT_REGION
  AWS_PROFILE or AWS credentials
  CMAKE_BUILD_DIR
  DC_PERF_ROOT
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"
MANIFEST="${REPO_ROOT}/folly_edit_rag/data/property_edit_tasks/property_edit_tasks.jsonl"
DC_PERF_ROOT_DIR="${DC_PERF_ROOT:-/myd/DCPerf}"
FOLLY_BUILD_DIR="${DC_PERF_ROOT_DIR}/benchmarks/wdl_bench/wdl_build/build/folly_rag"
FBTHRIFT_BUILD_DIR="${DC_PERF_ROOT_DIR}/benchmarks/wdl_bench/wdl_build/build/fbthrift_rag"
DEFAULT_FOLLY_BUILD_DIR="${DC_PERF_ROOT_DIR}/benchmarks/wdl_bench/wdl_build/build/folly"
DEFAULT_FBTHRIFT_BUILD_DIR="${DC_PERF_ROOT_DIR}/benchmarks/wdl_bench/wdl_build/build/fbthrift"
BATCH_SIZE=25
START_INDEX=0
MAX_BATCHES=0
TOP_K=8
MAX_CHILD_ATTEMPTS=5
SAMPLE_LARGE_BINARIES=0
REFRESH_MANIFEST=0
BINARIES=()
EXCLUDE_BINARIES=()
SMALL_BINARY_THRESHOLD=64
LARGE_BINARY_SAMPLE_CAP=36
LARGE_BINARY_SAMPLE_SCALE=1.25
AMD=0
SKIP_EXISTING=1
BUILD_TUNING_DATASET=0
LOG_DIR="${REPO_ROOT}/folly_edit_rag/scripts/outputs/full_property_db_runs"
FOLLY_SOURCE_DIR="${DC_PERF_ROOT_DIR}/benchmarks/wdl_bench/wdl_sources/folly"
FBTHRIFT_SOURCE_DIR="${DC_PERF_ROOT_DIR}/benchmarks/wdl_bench/wdl_sources/fbthrift"
INSTALLED_DIR="${DC_PERF_ROOT_DIR}/benchmarks/wdl_bench/wdl_build/installed"
SYSTEM_LIB_DIR="/usr/lib/x86_64-linux-gnu"
SYSTEM_INCLUDE_DIR="/usr/include"
TEMP_ROOT="${DC_PERF_ROOT_DIR}/tmp/benchmark_replication"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --folly-build-dir)
      FOLLY_BUILD_DIR="$2"
      shift 2
      ;;
    --fbthrift-build-dir)
      FBTHRIFT_BUILD_DIR="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --start-index)
      START_INDEX="$2"
      shift 2
      ;;
    --max-batches)
      MAX_BATCHES="$2"
      shift 2
      ;;
    --top-k)
      TOP_K="$2"
      shift 2
      ;;
    --max-child-attempts)
      MAX_CHILD_ATTEMPTS="$2"
      shift 2
      ;;
    --sample-large-binaries)
      SAMPLE_LARGE_BINARIES=1
      shift
      ;;
    --refresh-manifest)
      REFRESH_MANIFEST=1
      shift
      ;;
    --binary)
      BINARIES+=("$2")
      shift 2
      ;;
    --exclude-binary)
      EXCLUDE_BINARIES+=("$2")
      shift 2
      ;;
    --small-binary-threshold)
      SMALL_BINARY_THRESHOLD="$2"
      shift 2
      ;;
    --large-binary-sample-cap)
      LARGE_BINARY_SAMPLE_CAP="$2"
      shift 2
      ;;
    --large-binary-sample-scale)
      LARGE_BINARY_SAMPLE_SCALE="$2"
      shift 2
      ;;
    --amd)
      AMD=1
      shift
      ;;
    --no-skip-existing)
      SKIP_EXISTING=0
      shift
      ;;
    --build-tuning-dataset)
      BUILD_TUNING_DATASET=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

LLM_PROVIDER="${LLM_PROVIDER:-vertex}"
if [[ "${LLM_PROVIDER}" == "vertex" && -z "${VERTEX_PROJECT_ID:-}" ]]; then
  echo "VERTEX_PROJECT_ID is not set." >&2
  exit 1
elif [[ "${LLM_PROVIDER}" == "bedrock" && -z "${AWS_REGION:-${AWS_DEFAULT_REGION:-}}" ]]; then
  echo "AWS_REGION or AWS_DEFAULT_REGION is not set." >&2
  exit 1
elif [[ "${LLM_PROVIDER}" == "copilot" ]]; then
  COPILOT_API_URL="${COPILOT_API_URL:-http://127.0.0.1:4141}"
  if ! curl -fsS --max-time 3 "${COPILOT_API_URL}/v1/models" >/dev/null 2>&1; then
    echo "Copilot API proxy is not reachable at ${COPILOT_API_URL}." >&2
    exit 1
  fi
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "${TEMP_ROOT}"
export TMPDIR="${TEMP_ROOT}"
export TMP="${TEMP_ROOT}"
export TEMP="${TEMP_ROOT}"
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_DIR}"
if [[ "${SAMPLE_LARGE_BINARIES}" -eq 1 || "${REFRESH_MANIFEST}" -eq 1 ]]; then
  if [[ "${SAMPLE_LARGE_BINARIES}" -eq 1 ]]; then
    echo "Generating diverse sampled manifest at ${MANIFEST}"
  else
    echo "Regenerating manifest at ${MANIFEST} to pick up latest prompt guidance"
  fi

  GENERATE_CMD=(
    "${PYTHON_BIN}"
    "${REPO_ROOT}/folly_edit_rag/scripts/07_generate_property_edit_tasks.py"
    generate
    --output "${MANIFEST}"
  )
  if [[ "${SAMPLE_LARGE_BINARIES}" -eq 1 ]]; then
    GENERATE_CMD+=(
      --sample-large-binaries
      --small-binary-threshold "${SMALL_BINARY_THRESHOLD}"
      --large-binary-sample-cap "${LARGE_BINARY_SAMPLE_CAP}"
      --large-binary-sample-scale "${LARGE_BINARY_SAMPLE_SCALE}"
    )
  fi
  for binary_name in "${BINARIES[@]}"; do
    GENERATE_CMD+=(--binary "${binary_name}")
  done
  if [[ "${#EXCLUDE_BINARIES[@]}" -gt 0 ]]; then
    EXCLUDE_ARGS="$("${PYTHON_BIN}" - <<'PY' "${FEATURE_TABLE_XLSX:-${REPO_ROOT}/search_engine/all_features.xlsx}" "${EXCLUDE_BINARIES[@]}"
import sys
from pathlib import Path
import pandas as pd

xlsx = Path(sys.argv[1])
excluded = set(sys.argv[2:])
df = pd.read_excel(xlsx)
all_bins = sorted(set(df["binary"].dropna().astype(str)))
kept = [b for b in all_bins if b not in excluded]
for name in kept:
    print(name)
PY
)"
    while IFS= read -r binary_name; do
      [[ -n "${binary_name}" ]] || continue
      GENERATE_CMD+=(--binary "${binary_name}")
    done <<< "${EXCLUDE_ARGS}"
  fi
  "${GENERATE_CMD[@]}"
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

TOTAL_TASKS="$("${PYTHON_BIN}" - <<'PY' "${MANIFEST}"
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
groups = set()
with manifest.open() as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        groups.add((
            str(row["target_file"]),
            str(row["binary"]),
            str(row["source_microbenchmark"]),
            str(row["feature_name"]),
        ))
print(len(groups))
PY
)"

find_cmake_dir() {
  local name="$1"
  local direct
  direct="$(find "${INSTALLED_DIR}" -maxdepth 4 -type d -path "*/lib/cmake/${name}" | sort | head -n 1)"
  if [[ -n "${direct}" ]]; then
    printf '%s\n' "${direct}"
    return 0
  fi
  find "${INSTALLED_DIR}" -maxdepth 5 -type d -path "*/${name}-*/lib/cmake/${name}" | sort | head -n 1
}

find_prefix_dir() {
  local pattern="$1"
  find "${INSTALLED_DIR}" -maxdepth 1 -mindepth 1 -type d -name "${pattern}" | sort | head -n 1
}

FMT_DIR="$(find_cmake_dir fmt)"
FOLLY_DIR="$(find_cmake_dir folly)"
FIZZ_DIR="$(find_cmake_dir fizz)"
WANGLE_DIR="$(find_cmake_dir wangle)"
MVFST_DIR="$(find_cmake_dir mvfst)"
LIBOQS_DIR="$(find_cmake_dir liboqs)"
GTEST_DIR="$(
  find "${INSTALLED_DIR}" -maxdepth 5 -type d -path "*/lib/cmake/GTest" |
    sort |
    head -n 1
)"
GTEST_PREFIX=""
GMOCK_INCLUDE_DIR=""
GMOCK_LIBRARY=""
GMOCK_MAIN_LIBRARY=""
if [[ -n "${GTEST_DIR}" ]]; then
  GTEST_PREFIX="$(dirname "$(dirname "$(dirname "${GTEST_DIR}")")")"
  GMOCK_INCLUDE_DIR="${GTEST_PREFIX}/include"
  GMOCK_LIBRARY="${GTEST_PREFIX}/lib/libgmock.a"
  GMOCK_MAIN_LIBRARY="${GTEST_PREFIX}/lib/libgmock_main.a"
fi
GLOG_PREFIX="$(find_prefix_dir 'glog-*')"
FAST_FLOAT_PREFIX=""
for candidate in "${INSTALLED_DIR}"/fast_float-*; do
  if [[ -f "${candidate}/include/fast_float/fast_float.h" ]] &&
     grep -Rqs "allow_leading_plus" "${candidate}/include/fast_float"; then
    FAST_FLOAT_PREFIX="${candidate}"
    break
  fi
done
INSTALLED_PREFIX_PATH="$(
  find "${INSTALLED_DIR}" -maxdepth 1 -mindepth 1 -type d -print |
    sort |
    paste -sd ';' -
)"
GLOG_DIR=""
GLOG_INCLUDE_DIR=""
GLOG_LIBRARY_DIR=""
if [[ -n "${GLOG_PREFIX}" ]]; then
  GLOG_DIR="${GLOG_PREFIX}/lib/cmake/glog"
  GLOG_INCLUDE_DIR="${GLOG_PREFIX}/include"
  GLOG_LIBRARY_DIR="${GLOG_PREFIX}/lib"
fi

if [[ -z "${FMT_DIR}" || -z "${FOLLY_DIR}" || -z "${FIZZ_DIR}" || -z "${WANGLE_DIR}" || -z "${MVFST_DIR}" || -z "${LIBOQS_DIR}" || -z "${GTEST_DIR}" || ! -f "${GMOCK_LIBRARY}" || ! -f "${GMOCK_MAIN_LIBRARY}" || -z "${GLOG_DIR}" || -z "${FAST_FLOAT_PREFIX}" || -z "${INSTALLED_PREFIX_PATH}" ]]; then
  echo "Missing one or more required installed CMake package directories under ${INSTALLED_DIR}" >&2
  exit 1
fi

echo "Configuring Folly build in ${FOLLY_BUILD_DIR}"
cmake \
  -S "${FOLLY_SOURCE_DIR}" \
  -B "${FOLLY_BUILD_DIR}" \
  -DBUILD_BENCHMARKS=ON \
  -DFASTFLOAT_INCLUDE_DIR="${FAST_FLOAT_PREFIX}/include" \
  -DCMAKE_PREFIX_PATH="${INSTALLED_PREFIX_PATH}"

echo "Resetting fbthrift cache in ${FBTHRIFT_BUILD_DIR}"
rm -f "${FBTHRIFT_BUILD_DIR}/CMakeCache.txt"
rm -rf "${FBTHRIFT_BUILD_DIR}/CMakeFiles"

echo "Configuring fbthrift build in ${FBTHRIFT_BUILD_DIR}"
cmake \
  -S "${FBTHRIFT_SOURCE_DIR}" \
  -B "${FBTHRIFT_BUILD_DIR}" \
  -DBUILD_SHARED_LIBS=OFF \
  -DBoost_NO_BOOST_CMAKE=ON \
  -DBoost_NO_SYSTEM_PATHS=OFF \
  -DBoost_DIR= \
  -DBoost_INCLUDE_DIR="${SYSTEM_INCLUDE_DIR}" \
  -DBoost_LIBRARY_DIR_RELEASE="${SYSTEM_LIB_DIR}" \
  -DBoost_CONTEXT_LIBRARY_RELEASE="${SYSTEM_LIB_DIR}/libboost_context.so" \
  -DBoost_FILESYSTEM_LIBRARY_RELEASE="${SYSTEM_LIB_DIR}/libboost_filesystem.so" \
  -DBoost_PROGRAM_OPTIONS_LIBRARY_RELEASE="${SYSTEM_LIB_DIR}/libboost_program_options.so" \
  -DBoost_REGEX_LIBRARY_RELEASE="${SYSTEM_LIB_DIR}/libboost_regex.so" \
  -DBoost_THREAD_LIBRARY_RELEASE="${SYSTEM_LIB_DIR}/libboost_thread.so" \
  -DCMAKE_IGNORE_PREFIX_PATH=/usr/local \
  -Dgflags_DIR= \
  -DLIBGFLAGS_INCLUDE_DIR="${SYSTEM_INCLUDE_DIR}" \
  -DLIBGFLAGS_LIBRARY="${SYSTEM_LIB_DIR}/libgflags.so" \
  -DGFLAGS_INCLUDE_DIR="${SYSTEM_INCLUDE_DIR}" \
  -DGFLAGS_LIBRARY="${SYSTEM_LIB_DIR}/libgflags.so" \
  -Denable_tests=ON \
  -Dfmt_DIR="${FMT_DIR}" \
  -Dfolly_DIR="${FOLLY_DIR}" \
  -DFolly_DIR="${FOLLY_DIR}" \
  -Dfizz_DIR="${FIZZ_DIR}" \
  -DFizz_DIR="${FIZZ_DIR}" \
  -Dwangle_DIR="${WANGLE_DIR}" \
  -DWangle_DIR="${WANGLE_DIR}" \
  -Dmvfst_DIR="${MVFST_DIR}" \
  -Dliboqs_DIR="${LIBOQS_DIR}" \
  -DGTest_DIR="${GTEST_DIR}" \
  -DGMOCK_INCLUDE_DIRS="${GMOCK_INCLUDE_DIR}" \
  -DGMOCK_LIBRARIES="${GMOCK_LIBRARY}" \
  -DGMOCK_MAIN_LIBRARIES="${GMOCK_MAIN_LIBRARY}" \
  -Dglog_DIR="${GLOG_DIR}" \
  -DGLOG_INCLUDEDIR="${GLOG_INCLUDE_DIR}" \
  -DGLOG_LIBRARYDIR="${GLOG_LIBRARY_DIR}" \
  -DGLOG_INCLUDE_DIR="${GLOG_INCLUDE_DIR}" \
  -DGLOG_LIBRARY="${GLOG_LIBRARY_DIR}/libglog.so" \
  -DGLOG_LIBRARY_RELEASE="${GLOG_LIBRARY_DIR}/libglog.so" \
  -DCMAKE_PREFIX_PATH="${FMT_DIR};${FOLLY_DIR};${FIZZ_DIR};${WANGLE_DIR};${MVFST_DIR};${LIBOQS_DIR};${GLOG_DIR}"

mkdir -p "${FBTHRIFT_BUILD_DIR}/thrift/conformance/if"

echo "Validating representative benchmark targets"
cmake --build "${FOLLY_BUILD_DIR}" --parallel 16 --target hash_hash_benchmark
cmake --build "${FBTHRIFT_BUILD_DIR}" --parallel 16 \
  --target protocol-cpp2-target protocol_detail-cpp2-target field_mask-cpp2-target
cmake --build "${FBTHRIFT_BUILD_DIR}" --parallel 16 --target ProtocolBench

batch_num=0
current_start="${START_INDEX}"

while [[ "${current_start}" -lt "${TOTAL_TASKS}" ]]; do
  if [[ "${MAX_BATCHES}" -gt 0 && "${batch_num}" -ge "${MAX_BATCHES}" ]]; then
    break
  fi

  batch_num=$((batch_num + 1))
  batch_end=$((current_start + BATCH_SIZE - 1))
  if [[ "${batch_end}" -ge "${TOTAL_TASKS}" ]]; then
    batch_end=$((TOTAL_TASKS - 1))
  fi

  log_path="${LOG_DIR}/batch_${current_start}_to_${batch_end}.log"
  echo
  echo "============================================================"
  echo "Batch ${batch_num}: grouped jobs ${current_start}-${batch_end} of $((TOTAL_TASKS - 1))"
  echo "Manifest: ${MANIFEST}"
  echo "Folly build dir: ${FOLLY_BUILD_DIR}"
  echo "fbthrift build dir: ${FBTHRIFT_BUILD_DIR}"
  echo "Temp dir: ${TMPDIR}"
  echo "Max child attempts: ${MAX_CHILD_ATTEMPTS}"
  echo "Log: ${log_path}"
  echo "============================================================"

  cmd=(
    "${PYTHON_BIN}"
    "${REPO_ROOT}/folly_edit_rag/scripts/07_generate_property_edit_tasks.py"
    run
    --manifest "${MANIFEST}"
    --start-index "${current_start}"
    --limit "${BATCH_SIZE}"
    --top-k "${TOP_K}"
    --max-child-attempts "${MAX_CHILD_ATTEMPTS}"
    --build-dir-map "${DEFAULT_FOLLY_BUILD_DIR}=${FOLLY_BUILD_DIR}"
    --build-dir-map "${DEFAULT_FBTHRIFT_BUILD_DIR}=${FBTHRIFT_BUILD_DIR}"
  )
  if [[ "${AMD}" -eq 1 ]]; then
    cmd+=(--amd)
  fi
  if [[ "${SKIP_EXISTING}" -eq 1 ]]; then
    cmd+=(--skip-existing)
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee "${log_path}"
  rc=${PIPESTATUS[0]}
  set -e

  echo "Batch ${batch_num} exit code: ${rc}"
  current_start=$((current_start + BATCH_SIZE))
done

if [[ "${BUILD_TUNING_DATASET}" -eq 1 ]]; then
  echo
  echo "Building tuning dataset from successful edits"
  "${PYTHON_BIN}" "${REPO_ROOT}/folly_edit_rag/scripts/06_build_tuning_dataset.py" --output train.jsonl
fi

echo
echo "Finished batch run."
echo "Grouped jobs: ${TOTAL_TASKS}"
echo "Next start index: ${current_start}"
echo "Logs: ${LOG_DIR}"
