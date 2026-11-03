import unittest
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import cloud_tuning


class StrictIpcGuardrailTest(unittest.TestCase):
    def test_coupling_does_not_relax_strict_guardrail(self) -> None:
        with (
            patch.object(cloud_tuning, "STRICT_IPC_GUARDRAIL", True),
            patch.object(cloud_tuning, "MIN_CANDIDATE_IPC_ACCURACY", 0.9),
        ):
            report = cloud_tuning._ipc_guardrail_report(
                canonical_target={"perf_stat.IPC": 1.0},
                anchor_features={"perf_stat.IPC": 1.0},
                candidate_features={"perf_stat.IPC": 0.8},
                ipc_coupled=True,
            )

        self.assertFalse(report["passed"])
        self.assertAlmostEqual(report["max_allowed_relative_gap"], 0.1)

    def test_candidate_above_absolute_floor_passes(self) -> None:
        with (
            patch.object(cloud_tuning, "STRICT_IPC_GUARDRAIL", True),
            patch.object(cloud_tuning, "MIN_CANDIDATE_IPC_ACCURACY", 0.9),
            patch.object(cloud_tuning, "IPC_GUARDRAIL_MAX_WORSENING", 0.06),
        ):
            report = cloud_tuning._ipc_guardrail_report(
                canonical_target={"perf_stat.IPC": 1.0},
                anchor_features={"perf_stat.IPC": 0.94},
                candidate_features={"perf_stat.IPC": 0.92},
                ipc_coupled=True,
            )

        self.assertTrue(report["passed"])


class SubprocessImportsTest(unittest.TestCase):
    def test_rag_helpers_import_from_an_unrelated_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = cloud_tuning._run_logged(
                [sys.executable, "-c",
                 "import benchmark_sources, feature_schema, run_feature_extraction_end_to_end"],
                Path(tmp) / "imports.json",
                cwd=Path(tmp),
                env={**os.environ, "PYTHONPATH": ""},
            )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
