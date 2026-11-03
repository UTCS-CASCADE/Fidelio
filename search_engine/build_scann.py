from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scann

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.feature_schema import (
    CANONICAL_FEATURE_COLUMNS,
    canonicalize_dataframe,
    dataframe_feature_scales,
    load_feature_table,
    resolve_feature_table_path,
)

XLSX_PATH = resolve_feature_table_path(Path(__file__).resolve().parent / "all_features.xlsx")
AUGMENTED_CORPUS_PATH = Path(__file__).resolve().parent / "augmented_corpus.jsonl"
INDEX_DIR = REPO_ROOT / "search_engine" / "scann_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def _feedback_enabled() -> bool:
    return os.environ.get("CLOUD_CORPUS_FEEDBACK", "0").strip().lower() in ("1", "true", "yes", "on")


def maybe_merge_augmented(df: pd.DataFrame) -> pd.DataFrame:
    """Opt-in feedback loop: fold harvested, agent-generated benchmark variants
    into the corpus. Disabled unless CLOUD_CORPUS_FEEDBACK is truthy, so default
    behaviour (and any in-flight experiment) is byte-for-byte unchanged."""
    if not _feedback_enabled():
        return df
    if not AUGMENTED_CORPUS_PATH.exists():
        print("[feedback] CLOUD_CORPUS_FEEDBACK=1 but no augmented_corpus.jsonl found; using base corpus only.")
        return df
    rows = []
    for line in AUGMENTED_CORPUS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        rec.pop("_provenance", None)
        rows.append(rec)
    if not rows:
        return df
    aug = canonicalize_dataframe(pd.DataFrame(rows))
    # De-duplicate against base + within augmented set by microbenchmark id.
    if "microbenchmark" in aug.columns:
        existing = set(df["microbenchmark"].astype(str)) if "microbenchmark" in df.columns else set()
        aug = aug[~aug["microbenchmark"].astype(str).isin(existing)]
        aug = aug.drop_duplicates(subset=["microbenchmark"], keep="last")
    merged = pd.concat([df, aug], ignore_index=True)
    print(f"[feedback] Merged {len(aug)} augmented variant(s) into corpus (base N={len(df)}).")
    return merged


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def main() -> None:
    df = load_feature_table(XLSX_PATH)
    df = maybe_merge_augmented(df)
    all_nan = [c for c in CANONICAL_FEATURE_COLUMNS if c in df.columns and df[c].isna().all()]
    if all_nan:
        print(f"Warning: {len(all_nan)} canonical columns have no data: {all_nan[:5]}")

    ids = df["microbenchmark"].astype(str).to_numpy()
    binary_names = df["binary"].fillna("").astype(str).to_numpy()
    binary_paths = (
        df["binary_path"].fillna("").astype(str).to_numpy()
        if "binary_path" in df.columns
        else np.array([""] * len(df), dtype=object)
    )

    feature_names = [
        c
        for c in CANONICAL_FEATURE_COLUMNS
        if c in df.columns and not df[c].isna().all()
    ]
    X_nan = df[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    mu = np.nanmean(X_nan, axis=0, keepdims=True)
    sigma = np.nanstd(X_nan, axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    X_raw = np.where(np.isnan(X_nan), mu, X_nan).astype(np.float32)
    X = (X_raw - mu) / sigma
    X = l2_normalize_rows(X)

    feature_scales_map = dataframe_feature_scales(df)
    feature_scales = np.array([feature_scales_map.get(name, 1.0) for name in feature_names], dtype=np.float32)

    n = X.shape[0]
    num_leaves = min(256, max(16, int(np.sqrt(max(n, 1)) * 8)))
    num_leaves = min(num_leaves, max(n, 1))
    num_leaves_to_search = min(50, max(1, num_leaves // 10))
    reorder_k = min(100, max(n, 1))

    searcher = scann.scann_ops_pybind.builder(
        X,
        num_neighbors=10,
        distance_measure="dot_product",
    ).tree(
        num_leaves=num_leaves,
        num_leaves_to_search=num_leaves_to_search,
    ).score_ah(
        2,
        anisotropic_quantization_threshold=0.2,
    ).reorder(
        reorder_k,
    ).build()

    searcher.serialize(str(INDEX_DIR))
    np.save(INDEX_DIR / "ids.npy", ids)
    np.save(INDEX_DIR / "feature_names.npy", np.array(feature_names, dtype=object))
    np.save(INDEX_DIR / "feature_scales.npy", feature_scales)
    np.save(INDEX_DIR / "mu.npy", mu.astype(np.float32))
    np.save(INDEX_DIR / "sigma.npy", sigma.astype(np.float32))
    np.save(INDEX_DIR / "raw_features.npy", X_raw)
    np.save(INDEX_DIR / "binary_names.npy", binary_names)
    np.save(INDEX_DIR / "binary_paths.npy", binary_paths)

    print(f"Built canonical ScaNN index: N={n}, D={X.shape[1]}")


if __name__ == "__main__":
    main()
