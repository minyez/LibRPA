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
from regression_tests.backend import validate as validate_module
from regression_tests.backend.xmlparser import XMLParser


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

    def test_copies_extra_librpa_files_and_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = pathlib.Path(tmp) / "src"
            dst = pathlib.Path(tmp) / "dst"
            (src / "librpa" / "nested").mkdir(parents=True)
            (src / "librpa" / "extra_dir" / "sub").mkdir(parents=True)
            (src / "librpa" / "librpa.in").write_text("input_dir = ../dataset\n")
            (src / "librpa" / "extra.in").write_text("extra\n")
            (src / "librpa" / "nested" / "more.in").write_text("more\n")
            (src / "librpa" / "extra_dir" / "sub" / "file.in").write_text("dir\n")
            (dst / "librpa" / "extra_dir" / "sub").mkdir(parents=True)
            (dst / "librpa" / "extra_dir" / "stale.in").write_text("stale\n")
            self._write_archive(src, "dataset.tar.gz", "dataset")

            _prepare_librpa_workspace(src, dst, ("extra.in", "nested/more.in", "extra_dir"))

            self.assertEqual((dst / "librpa" / "extra.in").read_text(), "extra\n")
            self.assertEqual((dst / "librpa" / "nested" / "more.in").read_text(), "more\n")
            self.assertEqual((dst / "librpa" / "extra_dir" / "sub" / "file.in").read_text(), "dir\n")
            self.assertFalse((dst / "librpa" / "extra_dir" / "stale.in").exists())

    def test_rejects_extra_file_outside_librpa(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = pathlib.Path(tmp) / "src"
            dst = pathlib.Path(tmp) / "dst"
            (src / "librpa").mkdir(parents=True)
            (src / "librpa" / "librpa.in").write_text("input_dir = ../dataset\n")
            self._write_archive(src, "dataset.tar.gz", "dataset")

            with self.assertRaises(ValueError):
                _prepare_librpa_workspace(src, dst, ("../outside.in",))

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
            self.assertEqual(driver.result_counts(), (1, 0, 1))


class TestValidateFileOverrides(unittest.TestCase):

    def test_analyze_compares_test_and_reference_override_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "testcases").mkdir()
            (root / "refs" / "case").mkdir(parents=True)
            (root / "workspace" / "testcases" / "case").mkdir(parents=True)
            (root / "workspace" / "testcases" / "case" / "actual.out").write_text("same\n")
            (root / "refs" / "case" / "reference.out").write_text("same\n")

            xml = pathlib.Path(tmp) / "testsuite.xml"
            xml.write_text(
                '<testsuite><group name="g"><testcase name="n" directory="case">'
                '<validate name="v" file="common.out" file_test="actual.out" '
                'file_refr="reference.out" comparison="dummy" />'
                '</testcase></group></testsuite>'
            )
            groups = XMLParser(xml).groups

            old_import = validate_module._import_comparison
            validate_module._import_comparison = lambda _: lambda lhs, rhs: (lhs == rhs, "ok")
            try:
                driver = TestDriver(root / "testcases", root / "refs",
                                    root / "workspace", groups)
                driver.initialize(1, 1, False)
                self.assertEqual(driver.analyze(), 0)
                self.assertEqual(driver.result_counts(), (1, 1, 0))
            finally:
                validate_module._import_comparison = old_import

            self.assertEqual(groups["g"][0]["results"], [[True, "ok"]])


class TestRunCopyFiles(unittest.TestCase):

    def test_xml_parser_reads_copy_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml = pathlib.Path(tmp) / "testsuite.xml"
            xml.write_text(
                '<testsuite><group name="g"><testcase name="n" directory="case">'
                '<run copy_files="extra.in nested/more.in" />'
                '</testcase></group></testsuite>'
            )

            run = XMLParser(xml).groups["g"][0]["run"]

            self.assertEqual(run["copy_files"], ("extra.in", "nested/more.in"))


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
