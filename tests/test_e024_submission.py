import json
import unittest
from pathlib import Path
import zipfile

REPO = Path(__file__).resolve().parents[1]


class TestE024Submission(unittest.TestCase):
    def test_report_and_archive(self):
        report_path = REPO / "artifacts" / "e024_submission" / "report.json"
        self.assertTrue(report_path.is_file(), "report.json missing")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertTrue(report["strict_hypothesis_accepted"])
        for check, passed in report["checks"].items():
            self.assertTrue(passed, f"check failed: {check}")

        archive_path = REPO / "artifacts" / "e024_submission" / "e024_candidate.zip"
        self.assertTrue(archive_path.is_file(), "e024_candidate.zip missing")
        self.assertLess(archive_path.stat().st_size, 100 * 1024 * 1024)

        with zipfile.ZipFile(archive_path) as zf:
            names = set(zf.namelist())
            self.assertIn("submission.py", names)
            self.assertIn("model.pth", names)
            self.assertIn("load_baseline.py", names)


if __name__ == "__main__":
    unittest.main()
