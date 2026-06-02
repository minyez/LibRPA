import unittest

import cmp_float


class TestCmpFloat(unittest.TestCase):

    def _compare(self, values1, values2, **kwargs):
        compare = cmp_float.abs_diff(1e-4, **kwargs)
        return compare({"librpa.out": values1}, {"librpa.out": values2})

    def test_values_within_tolerance_pass(self):
        passed, _ = self._compare(["1.00001"], ["1.00002"])
        self.assertTrue(passed)

    def test_matching_nan_values_pass(self):
        passed, _ = self._compare(["nan"], ["NaN"])
        self.assertTrue(passed)

    def test_nan_in_test_only_fails(self):
        passed, msg = self._compare(["nan"], ["1.0"])
        self.assertFalse(passed)
        self.assertIn("nan mismatch", msg)
        self.assertIn("value 1", msg)

    def test_nan_in_reference_only_fails(self):
        passed, msg = self._compare(["1.0"], ["nan"])
        self.assertFalse(passed)
        self.assertIn("nan mismatch", msg)
        self.assertIn("value 1", msg)

    def test_missing_extracted_value_fails(self):
        passed, msg = self._compare([], ["-1.234"])
        self.assertFalse(passed)
        self.assertIn("value count mismatch", msg)
        self.assertIn("0 != 1", msg)

    def test_no_extracted_values_fails(self):
        passed, msg = self._compare([], [])
        self.assertFalse(passed)
        self.assertIn("no scalar values", msg)


if __name__ == "__main__":
    unittest.main()
