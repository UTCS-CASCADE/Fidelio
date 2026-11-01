import unittest

import pandas as pd

from feature_schema import canonicalize_dataframe


class CanonicalizeDataframeTest(unittest.TestCase):
    def test_coalesces_alias_with_canonical_column(self) -> None:
        frame = pd.DataFrame(
            {
                "perf_stat.L1-dcache-loads_LPKI": [10.0, None],
                "perf_stat.L1-dcache-loads_MPKI": [99.0, 20.0],
            }
        )

        canonical = canonicalize_dataframe(frame)

        self.assertEqual(
            canonical["perf_stat.L1-dcache-loads_LPKI"].tolist(),
            [10.0, 20.0],
        )
        self.assertFalse(canonical.columns.duplicated().any())


if __name__ == "__main__":
    unittest.main()
