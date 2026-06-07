import pathlib
import tarfile
import tempfile
import unittest

from regression_tests.backend.driver import _prepare_librpa_workspace


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


if __name__ == "__main__":
    unittest.main()
