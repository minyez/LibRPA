import pathlib
import shlex
import sys
import tarfile
import tempfile
import unittest

from regression_tests.backend.driver import (
    TestDriver,
    _prepare_librpa_workspace,
    _validate_scope_filter,
)


class TestPrepareLibrpaWorkspace(unittest.TestCase):

    def _write_archive(self, root: pathlib.Path, name: str, topdir: str):
        data_dir = root / topdir
        data_dir.mkdir()
        (data_dir / "marker.txt").write_text("data\n")
        with tarfile.open(root / name, "w:gz") as tar:
            tar.add(data_dir, arcname=topdir)

    def test_prepares_standard_dataset_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = pathlib.Path(tmp) / "src"
            dst = pathlib.Path(tmp) / "dst"
            (src / "librpa").mkdir(parents=True)
            (src / "librpa" / "librpa.in").write_text("input_dir = ../dataset\n")
            self._write_archive(src, "dataset.tar.gz", "dataset")

            run_dir = _prepare_librpa_workspace(src, dst)

            self.assertEqual(run_dir, dst / "librpa")
            self.assertEqual((dst / "librpa" / "librpa.in").read_text(),
                             "input_dir = ../dataset\n")
            self.assertEqual((dst / "dataset" / "marker.txt").read_text(), "data\n")

    def test_rejects_legacy_input_librpa_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = pathlib.Path(tmp) / "src"
            dst = pathlib.Path(tmp) / "dst"
            src.mkdir()
            (src / "librpa.in").write_text("input_dir = ./input_librpa\n")
            self._write_archive(src, "input_librpa.tar.gz", "input_librpa")

            with self.assertRaises(FileNotFoundError):
                _prepare_librpa_workspace(src, dst)


class TestDriverRunFailure(unittest.TestCase):

    def test_run_failure_is_reported_by_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            src = root / "testcases" / "case"
            refs = root / "refs"
            workspace = root / "workspace"
            (src / "librpa").mkdir(parents=True)
            (src / "librpa" / "librpa.in").write_text("input_dir = ../dataset\n")
            (refs / "case").mkdir(parents=True)
            TestPrepareLibrpaWorkspace()._write_archive(src, "dataset.tar.gz", "dataset")

            tc = {
                "directory": "case",
                "name": "failing case",
                "build": {"require_libri": False},
                "run": {
                    "ntasks_disable": [],
                    "nthreads_disable": [],
                    "ntasks_enable": [],
                    "nthreads_enable": [],
                },
                "labels": {},
                "validates": [],
            }
            driver = TestDriver(root / "testcases", refs, workspace, {"group": [tc]})
            driver.initialize(1, 1, False)
            mpiexec = "{} -c {}".format(
                shlex.quote(sys.executable),
                shlex.quote("import sys; sys.exit(3)"),
            )

            driver.run(sys.executable, mpiexec, force=True)

            self.assertEqual(
                (workspace / "testcases" / "case" / "librpa" / "librpa.exitcode").read_text(),
                "3\n",
            )
            self.assertEqual(driver.analyze(), 1)
            self.assertIn("exit code 3", tc["run_failure"])


class TestScopeFilter(unittest.TestCase):

    def test_accepts_path_like_testcase_values(self):
        testcases = [{"directory": "case-a"}, {"directory": "case-b"}]

        selected = _validate_scope_filter(
            testcases, "--only", ["testcases/case-a", pathlib.Path("/tmp/case-b/")]
        )

        self.assertEqual(selected, {"case-a", "case-b"})

    def test_only_enables_disabled_testcase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            testcases = root / "testcases"
            refs = root / "refs"
            testcases.mkdir()
            refs.mkdir()
            tc = {
                "directory": "case",
                "name": "disabled case",
                "build": {"require_libri": False},
                "run": {
                    "ntasks_disable": [],
                    "nthreads_disable": [],
                    "ntasks_enable": [],
                    "nthreads_enable": [],
                },
                "labels": {"disable": "experimental"},
                "validates": [],
            }
            driver = TestDriver(testcases, refs, root / "workspace", {"group": [tc]})

            driver.initialize(1, 1, False, only=["testcases/case"])

            self.assertEqual(driver._testcases_filtered, [tc])
            self.assertFalse(tc["labels"]["disable"])


if __name__ == "__main__":
    unittest.main()
