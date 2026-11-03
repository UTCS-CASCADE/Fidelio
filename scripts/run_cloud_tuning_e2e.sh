#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/run_cloud_tuning_e2e.sh [options] -- <cloud workload command...>

Starts local adjustor/search services, runs segmented cloud feature extraction,
then writes the final segment replacement plan JSON.

Options:
  --output-dir DIR            Output directory (default: ./outputs/cloud_e2e)
  --benchmark-name NAME       Output basename for the run
  --cloud-total-seconds N     Total collection horizon in seconds (default: 10)
  --cloud-workload-warmup-seconds N
                              Warm the cloud workload before segmented collection (default: 0)
  --cloud-segment-seconds N   Per-segment duration in seconds (default: 3)
  --input-cloud-json FILE     Reuse an existing segmented feature JSON instead of profiling
  --k N                       Top-k passed to cloud_segment_frontend.py (default: 1)
  -h, --help                  Show this help

Required environment:
  For Vertex mode (default):
    VERTEX_PROJECT_ID
  For Bedrock mode:
    AWS_REGION or AWS_DEFAULT_REGION
    and either AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE

Optional environment:
  VERTEX_LOCATION
  LLM_PROVIDER
  BEDROCK_MODEL_NAME
  DC_PERF_ROOT
  CMAKE_BUILD_DIR
  CLOUD_PIPELINE_TIMEOUT_SECONDS
      Set to 0 for no tuning wall-clock limit.
  CLOUD_MAX_LLM_COST_USD_PER_SEGMENT
  CLOUD_INITIAL_WARM_START_RESULT_JSON
  CLOUD_STOP_ON_STAGNATION
  PYTHON

PMU collection for the workload, seed baseline, and generated candidates is
always process-scoped so their feature vectors remain comparable.

Example:
  ./scripts/run_cloud_tuning_e2e.sh \
    --output-dir ./outputs/cloud_demo \
    --benchmark-name bench_ilp_cloud \
    --cloud-total-seconds 10 \
    --cloud-workload-warmup-seconds 2 \
    --cloud-segment-seconds 3 \
    -- ./bench_ilp
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
OUTPUT_DIR="${REPO_ROOT}/outputs/cloud_e2e"
BENCHMARK_NAME=""
CLOUD_TOTAL_SECONDS="10"
CLOUD_WORKLOAD_WARMUP_SECONDS="0"
CLOUD_SEGMENT_SECONDS="3"
INPUT_CLOUD_JSON=""
TOP_K="1"
CLOUD_PIPELINE_TIMEOUT_SECONDS="${CLOUD_PIPELINE_TIMEOUT_SECONDS:-10800}"
LLM_PROVIDER="${LLM_PROVIDER:-vertex}"
export SEED_BASELINE_SELECTION_MODE="${SEED_BASELINE_SELECTION_MODE:-ipc_only}"
export SEG08_PERPROC="1"


ADJUSTOR_PORT="5510"
SEARCH_PORT="5511"
ADJUSTOR_URL="http://127.0.0.1:${ADJUSTOR_PORT}/adjustor"
SEARCH_URL="http://127.0.0.1:${SEARCH_PORT}/search"
ADJUSTOR_HEALTH_URL="http://127.0.0.1:${ADJUSTOR_PORT}/healthz"
SEARCH_HEALTH_URL="http://127.0.0.1:${SEARCH_PORT}/healthz"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --benchmark-name)
      BENCHMARK_NAME="$2"
      shift 2
      ;;
    --cloud-total-seconds)
      CLOUD_TOTAL_SECONDS="$2"
      shift 2
      ;;
    --cloud-workload-warmup-seconds)
      CLOUD_WORKLOAD_WARMUP_SECONDS="$2"
      shift 2
      ;;
    --cloud-segment-seconds)
      CLOUD_SEGMENT_SECONDS="$2"
      shift 2
      ;;
    --input-cloud-json)
      INPUT_CLOUD_JSON="$2"
      shift 2
      ;;
    --k)
      TOP_K="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ $# -eq 0 && -z "${INPUT_CLOUD_JSON}" ]]; then
  echo "Missing cloud workload command. Pass it after --." >&2
  usage >&2
  exit 1
fi
if [[ -n "${INPUT_CLOUD_JSON}" && ! -f "${INPUT_CLOUD_JSON}" ]]; then
  echo "Input cloud JSON not found: ${INPUT_CLOUD_JSON}" >&2
  exit 1
fi

# Fail before starting services or modifying inputs when setup is incomplete.
if [[ ! -f "${REPO_ROOT}/folly_edit_rag/scripts/local_patch_utils.py" ]]; then
  echo "Missing folly_edit_rag. Run git submodule update --init --recursive first (see README.md)." >&2
  exit 1
fi
if [[ ! -s "${FEATURE_TABLE_XLSX:-${REPO_ROOT}/search_engine/all_features.xlsx}" ]]; then
  echo "Missing seed feature table. Profile seed binaries and set FEATURE_TABLE_XLSX (see README.md)." >&2
  exit 1
fi
if [[ ! -s "${REPO_ROOT}/folly_edit_rag/data/property_edit_tasks/property_edit_tasks.jsonl" ]]; then
  echo "Missing property-edit manifest. Generate it with folly_edit_rag (see README.md)." >&2
  exit 1
fi

if [[ "${LLM_PROVIDER}" == "bedrock" ]]; then
  if [[ -z "${AWS_REGION:-${AWS_DEFAULT_REGION:-}}" ]]; then
    echo "AWS_REGION or AWS_DEFAULT_REGION must be set for Bedrock mode." >&2
    exit 1
  fi
  if [[ -z "${AWS_BEARER_TOKEN_BEDROCK:-}" && -z "${BEDROCK_API_KEY:-}" && -z "${AWS_PROFILE:-}" && ( -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" ) ]]; then
    echo "Bedrock mode requires AWS_BEARER_TOKEN_BEDROCK or BEDROCK_API_KEY, or AWS_PROFILE, or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY." >&2
    exit 1
  fi
elif [[ "${LLM_PROVIDER}" == "copilot" ]]; then
  COPILOT_API_BASE="${COPILOT_API_BASE:-http://localhost:4141/v1}"
  if ! curl -sS -o /dev/null --max-time 2 "${COPILOT_API_BASE}/models" >/dev/null 2>&1; then
    echo "Copilot API proxy not reachable at ${COPILOT_API_BASE}. Start it with: npx copilot-api start" >&2
    exit 1
  fi
  echo "Copilot API proxy reachable at ${COPILOT_API_BASE}"
else
  if [[ -z "${VERTEX_PROJECT_ID:-}" ]]; then
    echo "VERTEX_PROJECT_ID is not set." >&2
    exit 1
  fi
fi

if [[ "${LLM_PROVIDER}" == "bedrock" ]]; then
  export LLM_PROVIDER="bedrock"
  export BEDROCK_MODEL_NAME="${BEDROCK_MODEL_NAME:-us.anthropic.claude-sonnet-4-6}"
  export BEDROCK_GENERATION_MODEL_NAME="${BEDROCK_GENERATION_MODEL_NAME:-${BEDROCK_MODEL_NAME}}"
  export BEDROCK_REPAIR_MODEL_NAME="${BEDROCK_REPAIR_MODEL_NAME:-${BEDROCK_MODEL_NAME}}"
elif [[ "${LLM_PROVIDER}" == "copilot" ]]; then
  export LLM_PROVIDER="copilot"
  export COPILOT_API_BASE="${COPILOT_API_BASE:-http://localhost:4141/v1}"
  export COPILOT_MODEL_NAME="${COPILOT_MODEL_NAME:-claude-opus-4.6}"
  export COPILOT_GENERATION_MODEL_NAME="${COPILOT_GENERATION_MODEL_NAME:-${COPILOT_MODEL_NAME}}"
  export COPILOT_REPAIR_MODEL_NAME="${COPILOT_REPAIR_MODEL_NAME:-${COPILOT_MODEL_NAME}}"
else
  export LLM_PROVIDER="vertex"
  export VERTEX_LOCATION="${VERTEX_LOCATION:-us-south1}"
  export VERTEX_MODEL_NAME="${VERTEX_MODEL_NAME:-gemini-2.5-pro}"
  export VERTEX_GENERATION_MODEL_NAME="${VERTEX_GENERATION_MODEL_NAME:-gemini-2.5-pro}"
  export VERTEX_REPAIR_MODEL_NAME="${VERTEX_REPAIR_MODEL_NAME:-gemini-2.5-pro}"
fi
export CLOUD_MAX_ADJUSTMENT_VERSIONS="${CLOUD_MAX_ADJUSTMENT_VERSIONS:-10}"
export CLOUD_MAX_COMPILE_REPAIR_ATTEMPTS="${CLOUD_MAX_COMPILE_REPAIR_ATTEMPTS:-20}"
export RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-1800}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required but not installed." >&2
  exit 1
fi

if ! sudo -n true >/dev/null 2>&1; then
  echo "sudo credentials are not currently cached. Run 'sudo -v' first." >&2
  exit 1
fi

WORKLOAD_CMD=("$@")

if [[ -z "${BENCHMARK_NAME}" ]]; then
  if [[ -n "${INPUT_CLOUD_JSON}" ]]; then
    BENCHMARK_NAME="$(basename "${INPUT_CLOUD_JSON}" _feature_extraction.json)"
  else
    BENCHMARK_NAME="$(basename "${WORKLOAD_CMD[0]}")"
  fi
fi

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
LOG_DIR="${OUTPUT_DIR}/service_logs"
mkdir -p "${LOG_DIR}"

CLOUD_JSON="${OUTPUT_DIR}/${BENCHMARK_NAME}_feature_extraction.json"
REPLACEMENTS_JSON="${OUTPUT_DIR}/${BENCHMARK_NAME}_segment_replacements.json"
ADJUSTOR_LOG="${LOG_DIR}/adjustor.log"
SEARCH_LOG="${LOG_DIR}/search.log"
INPUT_SNAPSHOT_DIR="${OUTPUT_DIR}/input_snapshot"

snapshot_pipeline_inputs() {
  "${PYTHON_BIN}" - "${REPO_ROOT}" "${INPUT_SNAPSHOT_DIR}" "${FEATURE_TABLE_XLSX:-}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

repo_root = Path(sys.argv[1])
snapshot_dir = Path(sys.argv[2])
feature_table = Path(sys.argv[3]) if sys.argv[3] else repo_root / "search_engine" / "all_features.xlsx"
manifest = repo_root / "folly_edit_rag" / "data" / "property_edit_tasks" / "property_edit_tasks.jsonl"

paths = set()
for line in manifest.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    path = Path(str(row["source_root"])) / str(row["target_file"])
    if path.is_file():
        paths.add(path.resolve())

frame = pd.read_excel(feature_table, usecols=["binary_path"])
for raw_path in frame["binary_path"].dropna().astype(str).unique():
    path = Path(raw_path)
    if path.is_file():
        paths.add(path.resolve())
        if path.name == "SvtAv1EncApp":
            paths.update(
                companion.resolve()
                for companion in path.parent.glob("libSvtAv1Enc.so*")
                if companion.is_file()
            )
            svt_source = path.parents[2] / "Source" / "Lib" / "Codec" / "motion_estimation.c"
            if not svt_source.is_file():
                raise FileNotFoundError(
                    f"SVT application source is missing for snapshot: {svt_source}"
                )
            paths.add(svt_source.resolve())

snapshot_dir.mkdir(parents=True, exist_ok=True)
entries = []
for index, path in enumerate(sorted(paths, key=str)):
    backup = snapshot_dir / f"{index:04d}.backup"
    shutil.copy2(path, backup)
    entries.append({"path": str(path), "backup": str(backup)})

(snapshot_dir / "state.json").write_text(json.dumps(entries), encoding="utf-8")
print(f"Snapshotted {len(entries)} editable sources and benchmark binaries")
PY
}

restore_pipeline_inputs() {
  "${PYTHON_BIN}" - "${INPUT_SNAPSHOT_DIR}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

snapshot_dir = Path(sys.argv[1])
state_path = snapshot_dir / "state.json"
if not state_path.is_file():
    raise SystemExit

for entry in json.loads(state_path.read_text(encoding="utf-8")):
    shutil.copy2(entry["backup"], entry["path"])
print("Restored snapshotted editable sources and benchmark binaries")
PY
}

clear_stale_mediawiki_ports() {
  ((${#WORKLOAD_CMD[@]} > 0)) || return 0
  [[ "$(basename "${WORKLOAD_CMD[0]}")" == "run_mediawiki_workload.sh" ]] || return 0

  local pids
  pids="$(
    {
      sudo ss -ltnp 2>/dev/null \
        | awk '$4 ~ /:(8090|8091|11888)$/ {print}' \
        | grep -o 'pid=[0-9]\+' \
        | cut -d= -f2
      pgrep -f '/tmp/hhvm-nginx' 2>/dev/null || true
    } | sort -u || true
  )"
  [[ -n "${pids}" ]] || return 0

  local confirmed_dcperf=0
  local pid
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    if sudo tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null | grep -q '/tmp/hhvm-nginx'; then
      confirmed_dcperf=1
      break
    fi
  done <<< "${pids}"

  if [[ "${confirmed_dcperf}" -ne 1 ]]; then
    echo "MediaWiki ports are occupied by processes not recognized as stale DCPerf services: ${pids}" >&2
    return 1
  fi

  echo "Clearing stale DCPerf MediaWiki port owners: ${pids//$'\n'/ }"
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    sudo kill -KILL "${pid}" >/dev/null 2>&1 || true
  done <<< "${pids}"
  sleep 1
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  set +e
  if [[ -n "${SEARCH_PID:-}" ]]; then
    kill "${SEARCH_PID}" >/dev/null 2>&1 || true
    wait "${SEARCH_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${ADJUSTOR_PID:-}" ]]; then
    kill "${ADJUSTOR_PID}" >/dev/null 2>&1 || true
    wait "${ADJUSTOR_PID}" >/dev/null 2>&1 || true
  fi
  local workload_pid_file="${OUTPUT_DIR}/cloud_workload.pid"
  if [[ -f "${workload_pid_file}" ]]; then
    local workload_pgid
    workload_pgid="$(cat "${workload_pid_file}" 2>/dev/null || true)"
    if [[ "${workload_pgid}" =~ ^[0-9]+$ ]]; then
      sudo kill -KILL -- "-${workload_pgid}" >/dev/null 2>&1 || true
      local workload_session_pids
      workload_session_pids="$(
        ps -eo pid=,sid= 2>/dev/null |
          awk -v sid="${workload_pgid}" '$2 == sid {print $1}'
      )"
      while IFS= read -r pid; do
        [[ "${pid}" =~ ^[0-9]+$ ]] || continue
        sudo kill -KILL "${pid}" >/dev/null 2>&1 || true
      done <<< "${workload_session_pids}"
    fi
  fi
  clear_stale_mediawiki_ports || true
  restore_pipeline_inputs || true
  # Kill any remaining workload processes by name, but never ourselves.
  if ((${#WORKLOAD_CMD[@]} > 0)); then
    local workload_name
    workload_name="$(basename "${WORKLOAD_CMD[0]}")"
    local my_pid=$$
    local my_pgid
    my_pgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"
    local pids
    pids=$(pgrep -f "${workload_name}" 2>/dev/null || true)
    if [[ -n "${pids}" ]]; then
      while IFS= read -r pid; do
        [[ -z "${pid}" ]] && continue
        [[ "${pid}" == "${my_pid}" ]] && continue
        local pid_pgid
        pid_pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
        [[ "${pid_pgid}" == "${my_pgid}" ]] && continue
        kill -9 "${pid}" >/dev/null 2>&1 || true
      done <<< "${pids}"
    fi
  fi
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM
snapshot_pipeline_inputs

wait_for_http() {
  local name="$1"
  local url="$2"
  local pid="$3"
  local log_path="$4"
  local i

  for ((i = 0; i < 60; i++)); do
    if curl -sS -o /dev/null --max-time 1 "${url}" >/dev/null 2>&1; then
      echo "${name} ready at ${url}"
      return 0
    fi
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "${name} exited before becoming ready. Log: ${log_path}" >&2
      tail -n 100 "${log_path}" >&2 || true
      return 1
    fi
    sleep 1
  done

  echo "Timed out waiting for ${name} at ${url}. Log: ${log_path}" >&2
  tail -n 100 "${log_path}" >&2 || true
  return 1
}

ensure_ports_free() {
  local ports=("${ADJUSTOR_PORT}" "${SEARCH_PORT}")
  local port
  
  for port in "${ports[@]}"; do
    local pids
    pids=$(ss -ltnp 2>/dev/null | awk -v port="$port" '$4 ~ ":" port "$" {print $NF}' | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u)
    
    if [[ -n "$pids" ]]; then
      echo "[warning] Port $port already in use by PIDs: $pids (likely stale services)"
      while IFS= read -r pid; do
        [[ -z "$pid" ]] && continue
        echo "  Killing stale process: $pid"
        kill -9 "$pid" 2>/dev/null || true
      done <<< "$pids"
      sleep 1
    fi
  done
  
  # Final verification
  for port in "${ports[@]}"; do
    if ss -ltnp 2>/dev/null | awk -v port="$port" '$4 ~ ":" port "$"' | grep -q .; then
      echo "ERROR: Port $port still in use after cleanup attempt" >&2
      return 1
    fi
  done
  
  return 0
}

echo "Building canonical ScaNN index"
"${PYTHON_BIN}" "${REPO_ROOT}/search_engine/build_scann.py"

echo "LLM provider: ${LLM_PROVIDER}"
echo "PMU measurement scope: process (SEG08_PERPROC=${SEG08_PERPROC})"
if [[ -n "${CLOUD_MAX_LLM_COST_USD_PER_SEGMENT:-}" ]]; then
  echo "LLM cost ceiling: \$${CLOUD_MAX_LLM_COST_USD_PER_SEGMENT} per segment"
fi
if [[ "${LLM_PROVIDER}" == "bedrock" ]]; then
  echo "Bedrock model: ${BEDROCK_MODEL_NAME}"
fi

echo "Ensuring ports ${ADJUSTOR_PORT} and ${SEARCH_PORT} are free..."
ensure_ports_free || exit 1

echo "Starting adjustor service on port ${ADJUSTOR_PORT}"
HOST="127.0.0.1" PORT="${ADJUSTOR_PORT}" FLASK_DEBUG="0" \
  "${PYTHON_BIN}" "${REPO_ROOT}/agents/adjustor.py" >"${ADJUSTOR_LOG}" 2>&1 &
ADJUSTOR_PID=$!
wait_for_http "adjustor" "${ADJUSTOR_HEALTH_URL}" "${ADJUSTOR_PID}" "${ADJUSTOR_LOG}"

echo "Starting search service on port ${SEARCH_PORT}"
HOST="127.0.0.1" PORT="${SEARCH_PORT}" FLASK_DEBUG="0" ADJUSTOR_URL="${ADJUSTOR_URL}" \
  CLOUD_PIPELINE_TIMEOUT_SECONDS="${CLOUD_PIPELINE_TIMEOUT_SECONDS}" \
  "${PYTHON_BIN}" "${REPO_ROOT}/search_engine/input_to_search_delta.py" >"${SEARCH_LOG}" 2>&1 &
SEARCH_PID=$!
wait_for_http "search" "${SEARCH_HEALTH_URL}" "${SEARCH_PID}" "${SEARCH_LOG}"

if [[ -n "${INPUT_CLOUD_JSON}" ]]; then
  echo "Reusing segmented cloud features: ${INPUT_CLOUD_JSON}"
  cp "${INPUT_CLOUD_JSON}" "${CLOUD_JSON}"
else
  echo "Collecting segmented cloud features"
  clear_stale_mediawiki_ports
  EXTRACTION_ARGS=()
  if [[ "${CLOUD_SKIP_TMA:-0}" =~ ^(1|true|yes|on)$ ]]; then
    EXTRACTION_ARGS+=(--skip-tma)
  fi
  if [[ "${CLOUD_SKIP_INTEL_PT:-0}" =~ ^(1|true|yes|on)$ ]]; then
    EXTRACTION_ARGS+=(--skip-intel-pt)
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/feature_extraction_engine/run_feature_extraction.py" \
    --cloud-mode \
    --cloud-total-seconds "${CLOUD_TOTAL_SECONDS}" \
    --cloud-workload-warmup-seconds "${CLOUD_WORKLOAD_WARMUP_SECONDS}" \
    --cloud-segment-seconds "${CLOUD_SEGMENT_SECONDS}" \
    --output-dir "${OUTPUT_DIR}" \
    --benchmark-name "${BENCHMARK_NAME}" \
    "${EXTRACTION_ARGS[@]}" \
    --cmd "${WORKLOAD_CMD[@]}"
fi

# ── Kill DCPerf workload after feature extraction ───────────────────────────
# Feature extraction already calls _stop_workload internally, but DCPerf
# spawns root-owned processes in nested sessions that require sudo SIGKILL.
# This block ensures no idle dcperf threads survive before the LLM pipeline.
#
# IMPORTANT: never use `pkill -f "benchpress_cli"` — our own bash process has
# the workload command in its argv and would be self-killed.  Instead we match
# on "python.*benchpress_cli\.py" which only matches the actual Python process,
# then kill its whole process group to take down all ffmpeg/SVT workers.
if [[ -z "${INPUT_CLOUD_JSON}" ]]; then
echo "Killing DCPerf workload after feature extraction"
_kill_dcperf_after_extraction() {
  local pid_file="${OUTPUT_DIR}/cloud_workload.pid"
  set +e

  # 1. Kill the outer process group by PGID (the group our Popen created:
  #    sudo → bash -c → benchpress launcher).  benchpress_cli.py calls setsid
  #    internally so it survives this; handled below.
  if [[ -f "${pid_file}" ]]; then
    local pgid
    pgid="$(cat "${pid_file}" 2>/dev/null || true)"
    if [[ "${pgid}" =~ ^[0-9]+$ ]]; then
      echo "  Killing outer process group -${pgid}"
      sudo kill -KILL -- "-${pgid}" >/dev/null 2>&1 || true
      local session_pids
      session_pids="$(
        ps -eo pid=,sid= 2>/dev/null |
          awk -v sid="${pgid}" '$2 == sid {print $1}'
      )"
      while IFS= read -r pid; do
        [[ "${pid}" =~ ^[0-9]+$ ]] || continue
        sudo kill -KILL "${pid}" >/dev/null 2>&1 || true
      done <<< "${session_pids}"
    fi
  fi

  # 2. Find the inner benchpress_cli Python process (it moved to its own
  #    session via setsid) and kill its entire process group.  Using the
  #    "python.*benchpress_cli\.py" regex avoids matching our own bash shell,
  #    which has "benchpress_cli.py" in its argument vector but is not python.
  #    We also explicitly exclude our own PID and parent PID to prevent
  #    self-kill when the regex matches our argv (e.g. the "-- python3 ...
  #    benchpress_cli.py ..." tail of the e2e invocation).
  local bp_pids
  bp_pids="$(
    ps -eo pid=,comm=,args= 2>/dev/null |
      awk '$2 ~ /^python/ && index($0, "benchpress_cli.py") {print $1}' |
      grep -v -w -e "$$" -e "${PPID:-0}" -e "${BASHPID:-0}" || true
  )"
  if [[ -n "${bp_pids}" ]]; then
    while IFS= read -r pid; do
      [[ -z "${pid}" ]] && continue
      local bp_pgid
      bp_pgid=$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ' || true)
      if [[ "${bp_pgid}" =~ ^[0-9]+$ && "${bp_pgid}" != "0" ]]; then
        echo "  Killing benchpress_cli process group -${bp_pgid} (pid ${pid})"
        sudo kill -KILL -- "-${bp_pgid}" >/dev/null 2>&1 || true
      fi
      # Belt-and-suspenders: kill the process directly too
      sudo kill -KILL "${pid}" >/dev/null 2>&1 || true
    done <<< "${bp_pids}"
  fi

  # 3. Kill any surviving ffmpeg / SVT encoder workers by process name only
  #    (pgrep without -f matches the comm field, never our bash argv).
  for procname in "ffmpeg" "SvtAv1EncApp"; do
    if pgrep "${procname}" >/dev/null 2>&1; then
      echo "  Killing remaining '${procname}' processes"
      sudo pkill -KILL "${procname}" >/dev/null 2>&1 || true
    fi
  done

  # Brief settle then confirm nothing remains
  sleep 1

  local still_running=0
  local remaining_benchpress
  remaining_benchpress="$(
    ps -eo pid=,comm=,args= 2>/dev/null |
      awk '$2 ~ /^python/ && index($0, "benchpress_cli.py") {print}' || true
  )"
  if [[ -n "${remaining_benchpress}" ]]; then
    echo "WARNING: Benchpress Python processes still running after kill:" >&2
    head -10 <<< "${remaining_benchpress}" >&2
    still_running=1
  fi
  for procname in "ffmpeg" "SvtAv1EncApp"; do
    if pgrep "${procname}" >/dev/null 2>&1; then
      echo "WARNING: DCPerf process '${procname}' still running after kill:" >&2
      pgrep -a "${procname}" 2>/dev/null | head -10 >&2 || true
      still_running=1
    fi
  done

  if [[ "${still_running}" -eq 0 ]]; then
    echo "DCPerf workload terminated — no idle threads remain."
  fi
  set -e
}
_kill_dcperf_after_extraction
fi
# ────────────────────────────────────────────────────────────────────────────

echo "Running segment replacement pipeline"
set +e
FRONTEND_ARGS=(
  --cloud-json "${CLOUD_JSON}"
  --search-url "${SEARCH_URL}"
  --k "${TOP_K}"
  --timeout-seconds "${CLOUD_PIPELINE_TIMEOUT_SECONDS}"
  --output-json "${REPLACEMENTS_JSON}"
)
if [[ -n "${CLOUD_MAX_SEGMENT_WALL_SECONDS:-}" ]]; then
  FRONTEND_ARGS+=(--max-segment-wall-seconds "${CLOUD_MAX_SEGMENT_WALL_SECONDS}")
fi
if [[ -n "${CLOUD_MIN_SEED_IPC_ACCURACY:-}" ]]; then
  FRONTEND_ARGS+=(--min-seed-ipc-accuracy "${CLOUD_MIN_SEED_IPC_ACCURACY}")
fi
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/cloud_segment_frontend.py" "${FRONTEND_ARGS[@]}"
FRONTEND_RC=$?
set -e


echo
echo "Cloud feature JSON: ${CLOUD_JSON}"
echo "Replacement plan:   ${REPLACEMENTS_JSON}"
echo "Accuracy report:    embedded under accuracy_report in ${REPLACEMENTS_JSON}"
echo "Adjustor log:       ${ADJUSTOR_LOG}"
echo "Search log:         ${SEARCH_LOG}"

exit "${FRONTEND_RC}"
