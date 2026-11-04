# Fidelio

Counter matching implementation from "Fidelio: Agentic AI Framework for
Synthesizing Microarchitecture-Faithful Datacenter Benchmarks" (MICRO 2026). The workflow
profiles a workload, selects an editable microbenchmark by IPC, retrieves editing
context, generates and compiles changes, and measures candidates against the
target counters.

The retained path is `scripts/run_cloud_tuning_e2e.sh` → workload profiling →
`scripts/cloud_segment_frontend.py` → search → adjustor → `cloud_tuning.py`.
The tuning loop uses the diagnostician, priority and schema helpers, source
resolution, candidate measurement, and the `folly_edit_rag` dependency. Seed
measurement and ScaNN index construction support this path. Intel PT software
features and Top-Down metrics guide counter-matching edits.

Offline edit-corpus population, system installation wrappers, and post-tuning
segment stitching are outside this checkout's scope. Regression tests remain
alongside the counter-matching implementation.

## dependencies


```bash
git submodule update --init --recursive
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r folly_edit_rag/requirements.txt
export PYTHON=python3
export PYTHONPATH="$PWD:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export SEG08_PERPROC=1
export DC_PERF_ROOT=/path/to/DCPerf
export CMAKE_BUILD_DIR="$DC_PERF_ROOT/benchmarks/wdl_bench/wdl_build/build/folly_rag"
```

DCPerf/WDL supplies editable Folly/fbthrift sources, build dependencies and seed
binaries; it is external to this extraction. Linux perf, C/C++ build tools and
Intel XED (for Intel PT decoding) are also external prerequisites. Choose a perf
build supporting your CPU's events; `TMA_PERF_BIN` can override the TMA collector.
Install these prerequisites before running the workflow.

## Prepare seed measurements and editing context

With WDL and the Python dependencies installed:

```bash
python3 folly_edit_rag/scripts/01_prepare_folly_rag.py
sudo -v
python3 scripts/run_feature_extraction_end_to_end.py \
  --binaries-dir "$CMAKE_BUILD_DIR" \
  --output-xlsx "$PWD/search_engine/all_features.xlsx"
export FEATURE_TABLE_XLSX="$PWD/search_engine/all_features.xlsx"
python3 search_engine/build_scann.py
python3 folly_edit_rag/scripts/07_generate_property_edit_tasks.py \
  generate --output "$PWD/folly_edit_rag/data/property_edit_tasks/property_edit_tasks.jsonl"
```

The seed table must refer to binaries and editable sources on this machine.
The extractor also accepts `--commands-manifest` for command-based seeds.
`--amd` skips Intel PT in seed measurement; use `CLOUD_SKIP_INTEL_PT=1` for
workload collection when Intel PT is unavailable.

## Generate a benchmark

Configure one supported LLM provider and its credentials (Vertex, Bedrock, or
the local Copilot API proxy). For example, with Vertex credentials configured:

```bash
export LLM_PROVIDER=vertex
export VERTEX_PROJECT_ID=your-project
sudo -v
./scripts/run_cloud_tuning_e2e.sh \
  --output-dir "$PWD/outputs/my_workload" \
  --benchmark-name my_workload \
  --cloud-total-seconds 10 --cloud-segment-seconds 3 \
  -- /path/to/workload arg1
```

Use `--input-cloud-json /path/to/workload_feature_extraction.json` to reuse a
recording instead of providing a command. Results include the feature JSON,
segment replacement plan with accuracy report and candidate source paths,
generated candidates under `folly_edit_rag/outputs/cloud_tuning/`, and service logs. IPC is the primary seed-selection signal;
`CLONE_PRIORITY_MODE=backend` selects the backend counter priority profile.

For recording alone:

```bash
python3 feature_extraction_engine/run_feature_extraction.py \
  --cloud-mode --cloud-total-seconds 10 --cloud-segment-seconds 3 \
  --output-dir outputs/profile --benchmark-name workload \
  --cmd /path/to/workload
```

Run `bash scripts/run_cloud_tuning_e2e.sh --help` for runner options. Build the
search index only after measuring the seed table. Fidelio's search indexes and
outputs stay local to this directory.

Iteration execution uses one configured provider; the experimental multi-provider
tournament has been removed. Prompts include the latest three variant lessons;
full history remains in the result JSON. Intel PT collection still runs for valid
candidates unless the existing AMD/skip settings disable it. Trace analysis uses
bounded instruction caches and avoids a redundant scan of the decoded file.
Seed tables are cached until their modification time or size changes, and CPU
cache metadata is read once per process.

## Validation

```bash
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s feature_extraction_engine -p 'test_*.py' -v
```

Tuning tests require the folly_edit_rag dependency. Full generation
also requires PMU access, a built seed corpus, and provider credentials.
