import os
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from feature_extraction_engine._hot_pid_resolver import resolve_hot_pid


class HotPidResolverTest(unittest.TestCase):
    def test_selects_busy_descendant(self):
        root = subprocess.Popen(
            [
                "bash",
                "-c",
                f"{sys.executable} -c 'while True: pass' & wait",
            ],
            start_new_session=True,
        )
        try:
            time.sleep(0.1)
            selected = resolve_hot_pid(root.pid)
            self.assertNotEqual(selected, root.pid)
            with open(f"/proc/{selected}/comm", encoding="utf-8") as comm_file:
                self.assertTrue(comm_file.read().strip().startswith("python"))
        finally:
            os.killpg(root.pid, signal.SIGTERM)
            try:
                root.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(root.pid, signal.SIGKILL)
                root.wait(timeout=2)

    def test_rejects_missing_root(self):
        with self.assertRaises(ProcessLookupError):
            resolve_hot_pid(999_999_999)

    def test_prefers_matching_descendant(self):
        root = subprocess.Popen(
            [
                "bash",
                "-c",
                (
                    f"{sys.executable} -c 'while True: pass' busy_worker & "
                    f"{sys.executable} -c 'import time; time.sleep(10)' svt_encoder & wait"
                ),
            ],
            start_new_session=True,
        )
        try:
            time.sleep(0.1)
            with patch.dict(os.environ, {"HOT_PID_PREFER_REGEX": "svt_encoder"}):
                selected = resolve_hot_pid(root.pid)
            with open(f"/proc/{selected}/cmdline", "rb") as cmdline_file:
                self.assertIn(b"svt_encoder", cmdline_file.read())
        finally:
            os.killpg(root.pid, signal.SIGTERM)
            try:
                root.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(root.pid, signal.SIGKILL)
                root.wait(timeout=2)


if __name__ == "__main__":
    unittest.main()
