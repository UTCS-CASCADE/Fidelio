from __future__ import annotations
import os
import sys
import traceback
from pathlib import Path

from flask import Flask, jsonify, request

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cloud_tuning import run_cloud_tuning_attempt

app = Flask(__name__)
app.config["PROPAGATE_EXCEPTIONS"] = True

DEFAULT_PERF_EVENTS = (
    "cycles,instructions,L1-icache-load-misses,iTLB-load-misses,"
    "L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,"
    "LLC-load-misses,branch-load-misses,branch-misses,r2424"
)


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.post("/adjustor")
def adjustor():
    data = request.get_json(force=True)
    run_id = str(data.get("run_id", "unknown"))
    nearest_id = str(data.get("nearest_id", "")).strip()
    binary_name = str(data.get("binary_name", "")).strip()
    query_json = data.get("query_json")
    max_wall_seconds = data.get("max_wall_seconds")

    if not nearest_id:
        return jsonify({"error": "missing 'nearest_id'"}), 400
    if not binary_name:
        return jsonify({"error": "missing 'binary_name'"}), 400
    if not isinstance(query_json, dict):
        return jsonify({"error": "missing or invalid 'query_json'"}), 400

    try:
        result = run_cloud_tuning_attempt(
            run_id=run_id,
            binary_name=binary_name,
            nearest_microbenchmark=nearest_id,
            target_features=query_json,
            retrieval_top_k=int(data.get("retrieval_top_k", 8)),
            perf_events=str(data.get("perf_events", DEFAULT_PERF_EVENTS)),
            amd=bool(data.get("amd", False)),
            max_wall_seconds=(
                float(max_wall_seconds)
                if max_wall_seconds is not None and str(max_wall_seconds).strip() != ""
                else None
            ),
            warm_start_file=data.get("warm_start_file"),
            warm_start_features=data.get("warm_start_features"),
        )
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug, use_reloader=False)
