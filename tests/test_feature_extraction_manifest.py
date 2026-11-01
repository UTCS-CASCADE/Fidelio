import json
import tempfile
import unittest
from pathlib import Path

from run_feature_extraction_end_to_end import load_command_manifest


class CommandManifestTest(unittest.TestCase):
    def test_loads_arbitrary_benchmark_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "bench"
            binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            binary.chmod(0o755)
            manifest = root / "commands.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "binary": "codec_checkasm",
                        "binary_path": str(binary),
                        "microbenchmark": "motion_compensation",
                        "command": [str(binary), "--bench=mc"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            entries = load_command_manifest(manifest)

        self.assertEqual(entries[0]["binary"], "codec_checkasm")
        self.assertEqual(entries[0]["microbenchmark"], "motion_compensation")
        self.assertEqual(entries[0]["command"][1], "--bench=mc")


if __name__ == "__main__":
    unittest.main()
