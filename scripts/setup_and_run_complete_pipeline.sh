#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./setup_and_run_complete_pipeline.sh [options] -- <cloud workload command...>

Sets up and runs the complete BenchmarkReplication pipeline:
  1. system, Python, perf, and Intel XED prerequisites
  2. folly_edit_rag restoration
  3. DCPerf WDL Folly/fbthrift preparation
  4. ScaNN index and property-edit manifest generation
  5. successful-edits database population and tuning-data export
  6. segmented cloud extraction, replacement, evaluation, and stitching

Options:
  --provider NAME          vertex, bedrock, or copilot (default: vertex)
  --vertex-project-id ID   Vertex project (default: VERTEX_PROJECT_ID or dcperf)
  --dc-perf-root PATH      DCPerf checkout (auto-detected by default)
  --batch-size N           Successful-edits batch size (default: 25)
  --max-batches N          Limit successful-edits batches; 0 means all (default: 0)
  --top-k N                Retrieval top-k (default: 8)
  --max-child-attempts N   Generation attempts per child target (default: 5)
  --amd                    Skip Intel PT evaluation
  --setup-only             Stop after dependencies, WDL, index, and manifest setup
  --skip-system-setup      Do not install apt, Python, perf, or XED prerequisites
  --skip-wdl-install       Reuse an existing WDL installation
  --skip-successful-edits  Do not run the successful-edits database pipeline
  --skip-cloud             Do not run the cloud pipeline
  --refresh-manifest       Regenerate the complete property-edit manifest
  --output-dir PATH        Cloud output directory
  --benchmark-name NAME    Cloud run name
  --cloud-total-seconds N  Cloud profiling horizon (default: 10)
  --cloud-warmup-seconds N Cloud workload warmup (default: 0)
  --cloud-segment-seconds N Segment duration (default: 3)
  -h, --help               Show this help

The cloud workload command is required unless --setup-only or --skip-cloud is used.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
PROVIDER="${LLM_PROVIDER:-vertex}"
export SEED_BASELINE_SELECTION_MODE="ipc_only"
export SEG08_PERPROC="1"

VERTEX_PROJECT="${VERTEX_PROJECT_ID:-dcperf}"
DC_PERF_ROOT_DIR="${DC_PERF_ROOT:-}"
BATCH_SIZE=25
MAX_BATCHES=0
TOP_K=8
MAX_CHILD_ATTEMPTS=5
AMD=0
SETUP_ONLY=0
SKIP_SYSTEM_SETUP=0
SKIP_WDL_INSTALL=0
SKIP_SUCCESSFUL_EDITS=0
SKIP_CLOUD=0
REFRESH_MANIFEST=0
OUTPUT_DIR="${REPO_ROOT}/outputs/complete_pipeline"
BENCHMARK_NAME=""
CLOUD_TOTAL_SECONDS=10
CLOUD_WARMUP_SECONDS=0
CLOUD_SEGMENT_SECONDS=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --provider) PROVIDER="$2"; shift 2 ;;
    --vertex-project-id) VERTEX_PROJECT="$2"; shift 2 ;;
    --dc-perf-root) DC_PERF_ROOT_DIR="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --max-batches) MAX_BATCHES="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --max-child-attempts) MAX_CHILD_ATTEMPTS="$2"; shift 2 ;;
    --amd) AMD=1; shift ;;
    --setup-only) SETUP_ONLY=1; shift ;;
    --skip-system-setup) SKIP_SYSTEM_SETUP=1; shift ;;
    --skip-wdl-install) SKIP_WDL_INSTALL=1; shift ;;
    --skip-successful-edits) SKIP_SUCCESSFUL_EDITS=1; shift ;;
    --skip-cloud) SKIP_CLOUD=1; shift ;;
    --refresh-manifest) REFRESH_MANIFEST=1; shift ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --benchmark-name) BENCHMARK_NAME="$2"; shift 2 ;;
    --cloud-total-seconds) CLOUD_TOTAL_SECONDS="$2"; shift 2 ;;
    --cloud-warmup-seconds) CLOUD_WARMUP_SECONDS="$2"; shift 2 ;;
    --cloud-segment-seconds) CLOUD_SEGMENT_SECONDS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

WORKLOAD_CMD=("$@")

case "${PROVIDER}" in
  vertex|bedrock|copilot) ;;
  *) echo "Unsupported provider: ${PROVIDER}" >&2; exit 1 ;;
esac

if [[ -z "${DC_PERF_ROOT_DIR}" ]]; then
  if [[ -d /users/alanuiuc/DCPerf ]]; then
    DC_PERF_ROOT_DIR=/users/alanuiuc/DCPerf
  else
    DC_PERF_ROOT_DIR=/myd/DCPerf
  fi
fi
DC_PERF_ROOT_DIR="$(readlink -f "${DC_PERF_ROOT_DIR}")"

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHON="${PYTHON_BIN}"
export LLM_PROVIDER="${PROVIDER}"
export VERTEX_PROJECT_ID="${VERTEX_PROJECT}"
export VERTEX_LOCATION="${VERTEX_LOCATION:-us-south1}"
export DC_PERF_ROOT="${DC_PERF_ROOT_DIR}"
export CMAKE_BUILD_DIR="${DC_PERF_ROOT_DIR}/benchmarks/wdl_bench/wdl_build/build/folly_rag"

WDL_ROOT="${DC_PERF_ROOT_DIR}/benchmarks/wdl_bench"
WDL_BUILD="${WDL_ROOT}/wdl_build"
WDL_INSTALLED="${WDL_BUILD}/installed"
FOLLY_RAG_ROOT="${REPO_ROOT}/folly_edit_rag"
MANIFEST="${FOLLY_RAG_ROOT}/data/property_edit_tasks/property_edit_tasks.jsonl"

require_sudo() {
  if ! sudo -n true >/dev/null 2>&1; then
    echo "sudo credentials are not cached. Run 'sudo -v' and retry." >&2
    exit 1
  fi
}

restore_folly_edit_rag() {
  "${REPO_ROOT}/bootstrap_folly_edit_rag.sh"
}

install_system_setup() {
  require_sudo
  sudo apt-get update -qq
  sudo apt-get install -y -qq \
    autoconf automake bison build-essential clang cmake curl flex git \
    libaio-dev libboost-all-dev libcap-dev libdouble-conversion-dev \
    libdwarf-dev libelf-dev libevent-dev libfmt-dev libgflags-dev \
    libgoogle-glog-dev libiberty-dev liblz4-dev libnuma-dev libsnappy-dev \
    libssl-dev libunwind-dev libzstd-dev linux-cloud-tools-generic \
    linux-tools-common linux-tools-generic make nasm ninja-build openssl \
    patch perl pkg-config python3-dev python3-venv tar unzip wget

  sudo apt-get install -y -qq \
    "linux-tools-$(uname -r)" "linux-cloud-tools-$(uname -r)"

  "${PYTHON_BIN}" -m pip install --user --quiet \
    -r "${FOLLY_RAG_ROOT}/requirements.txt" \
    -r "${REPO_ROOT}/requirements.txt"

  if [[ "${PROVIDER}" == "vertex" ]] && ! command -v gcloud >/dev/null 2>&1; then
    local gcloud_version="${GCLOUD_VERSION:-576.0.0}"
    local gcloud_archive="google-cloud-cli-${gcloud_version}-linux-x86_64.tar.gz"
    curl -fsSLo "/tmp/${gcloud_archive}" \
      "https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/${gcloud_archive}"
    rm -rf "${HOME}/.local/google-cloud-sdk"
    tar -xzf "/tmp/${gcloud_archive}" -C "${HOME}/.local"
    "${HOME}/.local/google-cloud-sdk/install.sh" \
      --quiet --path-update=false --command-completion=false --usage-reporting=false
    ln -sfn "${HOME}/.local/google-cloud-sdk/bin/gcloud" "${HOME}/.local/bin/gcloud"
  fi

  if ! command -v xed >/dev/null 2>&1; then
    local tools_root="${HOME}/.local/src"
    mkdir -p "${tools_root}"
    [[ -d "${tools_root}/mbuild/.git" ]] ||
      git clone https://github.com/intelxed/mbuild.git "${tools_root}/mbuild"
    [[ -d "${tools_root}/xed/.git" ]] ||
      git clone https://github.com/intelxed/xed.git "${tools_root}/xed"
    (
      cd "${tools_root}/xed"
      ./mfile.py --share
      ./mfile.py examples
      sudo ./mfile.py --prefix=/usr/local install
      sudo cp obj/wkit/examples/obj/xed /usr/local/bin/xed
      sudo ldconfig
    )
  fi
}


required_wdl_packages_ready() {
  local name
  for name in folly fizz wangle mvfst liboqs; do
    if ! find "${WDL_INSTALLED}" -maxdepth 5 -type d -path "*/lib/cmake/${name}" -print -quit 2>/dev/null | grep -q .; then
      return 1
    fi
  done
  return 0
}

repair_fbthrift_dependency_build() {
  local fbthrift_source="${WDL_ROOT}/wdl_sources/fbthrift"
  local archived_suffix="wdl-source-$(date +%Y%m%d%H%M%S)"
  local archived_build="${WDL_BUILD}/build/folly-${archived_suffix}"
  local archived_install="${WDL_INSTALLED}/folly-${archived_suffix}"
  local hashes_dir="${fbthrift_source}/build/deps/github_hashes"

  [[ -f "${fbthrift_source}/build/fbcode_builder/getdeps.py" ]] || return 1

  # The 2024.12 fbthrift release omits public pins for itself and Fizz, so
  # getdeps otherwise combines the release with incompatible current heads.
  mkdir -p "${hashes_dir}/facebook" "${hashes_dir}/facebookincubator"
  printf '%s\n' \
    "Subproject commit f63b3d9cc4834b94361133aa84ce6ca3120b1fe6" \
    > "${hashes_dir}/facebook/fbthrift-rev.txt"
  printf '%s\n' \
    "Subproject commit 6448c39b70e6006cea7061094cdc3598097d4f46" \
    > "${hashes_dir}/facebookincubator/fizz-rev.txt"

  # The DCPerf installer builds its editable Folly source and fbthrift's pinned
  # Folly dependency into the same getdeps paths. Preserve the first build and
  # give fbthrift clean paths so Fizz sees the matching Folly CMake targets.
  if [[ -d "${WDL_BUILD}/build/folly" ]]; then
    mv "${WDL_BUILD}/build/folly" "${archived_build}"
  fi
  if [[ -d "${WDL_INSTALLED}/folly" ]]; then
    mv "${WDL_INSTALLED}/folly" "${archived_install}"
  fi
  rm -rf \
    "${WDL_BUILD}/repos/github.com-facebook-fbthrift.git" \
    "${WDL_BUILD}/repos/github.com-facebookincubator-fizz.git" \
    "${WDL_BUILD}/build/fbthrift" \
    "${WDL_BUILD}/build/fizz" \
    "${WDL_INSTALLED}/fbthrift" \
    "${WDL_INSTALLED}/fizz"

  (
    cd "${fbthrift_source}"
    "${PYTHON_BIN}" ./build/fbcode_builder/getdeps.py \
      --allow-system-packages build fbthrift --scratch-path "${WDL_BUILD}"
  )

  [[ ! -d "${archived_build}" ]] || rm -rf "${archived_build}"
  [[ ! -d "${archived_install}" ]] || rm -rf "${archived_install}"
}

install_wdl() {
  if required_wdl_packages_ready; then
    echo "WDL dependencies already prepared."
    return
  fi

  require_sudo
  if [[ ! -x "${DC_PERF_ROOT_DIR}/benchpress_cli.py" ]]; then
    echo "DCPerf checkout not found at ${DC_PERF_ROOT_DIR}" >&2
    exit 1
  fi

  set +e
  (
    cd "${DC_PERF_ROOT_DIR}"
    ./benchpress_cli.py -b wdl install folly_single_core
  )
  local install_rc=$?
  set -e

  sudo chown -R "$(id -u):$(id -g)" "${WDL_ROOT}"

  if [[ "${install_rc}" -ne 0 ]] && ! required_wdl_packages_ready; then
    echo "DCPerf's combined WDL installer hit its Folly/fbthrift getdeps path collision; repairing it."
    repair_fbthrift_dependency_build
  fi

  if ! required_wdl_packages_ready; then
    echo "WDL setup did not produce all required CMake packages." >&2
    exit 1
  fi
}

validate_provider() {
  case "${PROVIDER}" in
    vertex)
      "${PYTHON_BIN}" - <<'PY'
import google.auth
from google.auth.transport.requests import Request
credentials, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
credentials.refresh(Request())
PY
      ;;
    bedrock)
      if [[ -z "${AWS_REGION:-${AWS_DEFAULT_REGION:-}}" ]]; then
        echo "AWS_REGION or AWS_DEFAULT_REGION is required for Bedrock." >&2
        exit 1
      fi
      ;;
    copilot)
      if ! curl -fsS --max-time 3 http://127.0.0.1:4141/v1/models >/dev/null; then
        echo "Copilot API is not reachable at http://127.0.0.1:4141." >&2
        exit 1
      fi
      ;;
  esac
}

prepare_indexes_and_manifest() {
  "${PYTHON_BIN}" "${FOLLY_RAG_ROOT}/scripts/01_prepare_folly_rag.py"
  local feature_table="${FEATURE_TABLE_XLSX:-${REPO_ROOT}/search_engine/all_features.xlsx}"
  if [[ ! -s "${feature_table}" ]]; then
    echo "Profile the seed binaries with run_feature_extraction_end_to_end.py and set FEATURE_TABLE_XLSX (see README.md)." >&2
    return 1
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/search_engine/build_scann.py"

  if [[ "${REFRESH_MANIFEST}" -eq 1 || ! -s "${MANIFEST}" ]]; then
    "${PYTHON_BIN}" "${FOLLY_RAG_ROOT}/scripts/07_generate_property_edit_tasks.py" \
      generate --output "${MANIFEST}"
  fi
}

run_successful_edits() {
  local args=(
    --manifest "${MANIFEST}"
    --batch-size "${BATCH_SIZE}"
    --max-batches "${MAX_BATCHES}"
    --top-k "${TOP_K}"
    --max-child-attempts "${MAX_CHILD_ATTEMPTS}"
    --build-tuning-dataset
  )
  [[ "${AMD}" -eq 1 ]] && args+=(--amd)
  "${REPO_ROOT}/run_full_successful_edit_database.sh" "${args[@]}"
}

run_cloud_pipeline() {
  if [[ "${#WORKLOAD_CMD[@]}" -eq 0 ]]; then
    echo "A cloud workload command is required after --." >&2
    exit 1
  fi

  local args=(
    --output-dir "${OUTPUT_DIR}"
    --cloud-total-seconds "${CLOUD_TOTAL_SECONDS}"
    --cloud-workload-warmup-seconds "${CLOUD_WARMUP_SECONDS}"
    --cloud-segment-seconds "${CLOUD_SEGMENT_SECONDS}"
    --k "${TOP_K}"
  )
  [[ -n "${BENCHMARK_NAME}" ]] && args+=(--benchmark-name "${BENCHMARK_NAME}")
  args+=(-- "${WORKLOAD_CMD[@]}")

  "${REPO_ROOT}/run_cloud_tuning_e2e.sh" "${args[@]}"
}

restore_folly_edit_rag
if [[ "${SKIP_SYSTEM_SETUP}" -eq 0 ]]; then
  install_system_setup
fi
validate_provider
if [[ "${SKIP_WDL_INSTALL}" -eq 0 ]]; then
  install_wdl
fi
prepare_indexes_and_manifest

if [[ "${SETUP_ONLY}" -eq 1 ]]; then
  echo "Complete pipeline setup finished."
  exit 0
fi

if [[ "${SKIP_SUCCESSFUL_EDITS}" -eq 0 ]]; then
  run_successful_edits
fi
if [[ "${SKIP_CLOUD}" -eq 0 ]]; then
  run_cloud_pipeline
fi

echo "Complete BenchmarkReplication pipeline finished."
