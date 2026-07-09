import unittest

import cmp_table


class TestCmpTableNaN(unittest.TestCase):

    def _compare(self, table1, table2, **kwargs):
        compare = cmp_table.abs_diff(1e-4, **kwargs)
        return compare({"table.dat": table1}, {"table.dat": table2})

    def test_no_extracted_files_fails(self):
        compare = cmp_table.abs_diff(1e-4)
        passed, msg = compare({}, {})
        self.assertFalse(passed)
        self.assertIn("no files found", msg)

    def test_no_extracted_tables_fails(self):
        compare = cmp_table.abs_diff(1e-4)
        passed, msg = compare({"table.dat": []}, {"table.dat": []})
        self.assertFalse(passed)
        self.assertIn("no tables found", msg)

    def test_zero_row_table_fails(self):
        passed, msg = self._compare("", "")
        self.assertFalse(passed)
        self.assertIn("no table rows", msg)

    def test_no_selected_columns_fails(self):
        passed, msg = self._compare("1.0 2.0\n", "1.0 2.0\n", columns="")
        self.assertFalse(passed)
        self.assertIn("no table cells compared", msg)

    def test_matching_nan_cells_pass(self):
        passed, _ = self._compare("1.0 nan\n2.0 3.0\n", "1.0 NaN\n2.0 3.0\n")
        self.assertTrue(passed)

    def test_nan_in_test_only_fails(self):
        passed, msg = self._compare("1.0 nan\n", "1.0 2.0\n")
        self.assertFalse(passed)
        self.assertIn("nan mismatch", msg)
        self.assertIn("column 2", msg)

    def test_nan_in_reference_only_fails(self):
        passed, msg = self._compare("1.0 2.0\n", "1.0 nan\n")
        self.assertFalse(passed)
        self.assertIn("nan mismatch", msg)
        self.assertIn("column 2", msg)

    def test_nan_in_unselected_column_is_ignored(self):
        passed, _ = self._compare("1.0 nan\n", "1.0 2.0\n", columns="1")
        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
