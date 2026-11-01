import unittest
import sys
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cloud_tuning
import cloud_segment_frontend
from cloud_tuning import _estimated_llm_cost_usd

sys.path.insert(0, str(Path(__file__).parent / "folly_edit_rag" / "scripts"))
from llm_provider import _vertex_token_usage


class EstimatedLlmCostTest(unittest.TestCase):
    def test_uses_vertex_token_rates(self) -> None:
        self.assertAlmostEqual(
            _estimated_llm_cost_usd(
                {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
            ),
            14.0,
        )
        self.assertAlmostEqual(
            cloud_segment_frontend.estimate_llm_cost(
                1_000_000,
                1_000_000,
                provider="vertex",
            ),
            14.0,
        )

    def test_reads_provider_sidecars_from_scripts_outputs(self) -> None:
        expected = (
            Path(__file__).parent
            / "folly_edit_rag"
            / "scripts"
            / "outputs"
            / "patches"
        ).resolve()
        self.assertEqual(cloud_tuning._PATCHES_DIR.resolve(), expected)


class FrontendTokenFallbackTest(unittest.TestCase):
    def test_aggregates_sidecars_for_timed_out_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp)
            for directory in ("patches", "compile_logs"):
                (outputs / directory).mkdir()
            (outputs / "patches" / "run_segment_03_attempt.token_usage.json").write_text(
                json.dumps(
                    {
                        "input_tokens": 100,
                        "output_tokens": 200,
                        "total_tokens": 300,
                        "call_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            (outputs / "compile_logs" / "run_segment_03_attempt.token_usage.json").write_text(
                json.dumps(
                    {
                        "input_tokens": 50,
                        "output_tokens": 75,
                        "total_tokens": 125,
                        "call_count": 2,
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(cloud_segment_frontend, "_LLM_OUTPUTS_DIR", outputs):
                usage = cloud_segment_frontend._sidecar_token_usage("run", "segment_03")

        self.assertEqual(
            usage,
            {
                "input_tokens": 150,
                "output_tokens": 275,
                "total_tokens": 425,
                "call_count": 3,
            },
        )


class VertexTokenUsageTest(unittest.TestCase):
    def test_includes_billed_reasoning_tokens(self) -> None:
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=200,
                thoughts_token_count=300,
            )
        )

        self.assertEqual(_vertex_token_usage(response), (100, 500))


if __name__ == "__main__":
    unittest.main()
