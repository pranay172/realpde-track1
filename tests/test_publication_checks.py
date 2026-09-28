"""Synthetic regression tests for the public-release checks."""
import importlib.util
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "snapshot_audit", Path(__file__).resolve().parents[1] / "scripts/publication_audit.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)

ARCHIVE_SPEC = importlib.util.spec_from_file_location(
    "snapshot_archive", Path(__file__).resolve().parents[1] / "scripts/check_archive.py")
ARCHIVE = importlib.util.module_from_spec(ARCHIVE_SPEC)
with mock.patch.dict(sys.modules, {"publication_audit": AUDIT}):
    ARCHIVE_SPEC.loader.exec_module(ARCHIVE)


class PublicationChecks(unittest.TestCase):
    def test_reader_links_include_html_and_nested_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "docs/hero.svg").write_text("<svg/>\n")
            (root / "docs/guide.md").write_text("[Back](../README.md)\n")
            (root / "README.md").write_text(
                '[Guide](docs/guide.md#setup)\n<img src="docs/hero.svg">\n'
                "<a href='docs/guide.md'>Guide</a>\n"
                '<a href="https://example.com">External</a>\n'
                '<a href="#section">Anchor</a>\n')
            self.assertEqual(ARCHIVE.check_reader_links(root, ("README.md", "docs/guide.md")), 4)

    def test_missing_html_reader_assets_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for markup in ('<img src="absent.svg">', "<a href='absent.md'>Missing</a>"):
                with self.subTest(markup=markup):
                    (root / "README.md").write_text(markup)
                    with self.assertRaisesRegex(ValueError, "reader-doc link"):
                        ARCHIVE.check_reader_links(root, ("README.md",))

    def test_reader_links_cannot_depend_on_files_outside_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "snapshot"
            root.mkdir()
            (root.parent / "outside.md").write_text("Not published\n")
            (root / "README.md").write_text('[Outside](../outside.md)\n')
            with self.assertRaisesRegex(ValueError, "out-of-repository"):
                ARCHIVE.check_reader_links(root, ("README.md",))

    def test_source_only_folder_passes_without_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("Research snapshot\n")
            report = AUDIT.audit(root)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["history"]["state"], "not_initialized")

    def test_secret_report_is_redacted(self):
        token = b"ghp_" + b"x" * 36
        report = AUDIT.scan_blob("example.py", token)
        self.assertTrue(any(x["category"] == "github_token" for x in report))
        self.assertNotIn(token.decode(), str(report))

    def test_private_path_and_email_are_flagged(self):
        value = (b"/" + b"home" + b"/person/project/ " + b"private" + b"@example.com")
        categories = {x["category"] for x in AUDIT.scan_blob("note.md", value)}
        self.assertIn("personal_path", categories)
        self.assertIn("unreviewed_email", categories)

    def test_organizer_and_artifact_payloads_are_forbidden(self):
        for name in ("realpde_t1_starting_kit_v9/scoring.py",
                     "realpde_t2_starting_kit_v6/local_eval.py",
                     "competition_docs/Terms.md", "model.pth", "sample.h5"):
            self.assertTrue(AUDIT.forbidden_path(name), name)

    def test_data_configuration_is_not_a_dataset(self):
        self.assertFalse(AUDIT.forbidden_path("configs/data/release_v1.json"))

    def test_fresh_git_and_reachable_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def git(*args):
                return subprocess.run(["git", "-C", str(root), *args], check=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            git("init", "-b", "main")
            git("config", "user.name", "Snapshot test fixture")
            git("config", "user.email", "pranayvandanapu2001@gmail.com")
            git("config", "commit.gpgsign", "false")
            git("config", "core.hooksPath", str(root / "no-hooks"))
            self.assertEqual(AUDIT.audit(root)["history"]["state"], "fresh_uncommitted")
            (root / "README.md").write_text("Synthetic fixture\n")
            git("add", "README.md")
            git("commit", "-m", "Synthetic safe snapshot")
            self.assertEqual(AUDIT.audit(root)["status"], "passed")
            (root / "removed.txt").write_text("ghp_" + "x" * 36)
            git("add", "removed.txt")
            git("commit", "-m", "Synthetic credential fixture")
            (root / "removed.txt").unlink()
            git("add", "-u")
            git("commit", "-m", "Remove synthetic fixture")
            report = AUDIT.audit(root)
            self.assertTrue(any(x["category"] == "github_token" and "object" in x
                                for x in report["findings"]))
            git("config", "user.email", "unapproved" + "@example.com")
            git("commit", "--allow-empty", "-m", "Synthetic unapproved identity")
            report = AUDIT.audit(root)
            self.assertTrue(any(x["category"] == "unreviewed_commit_email"
                                for x in report["findings"]))

    def test_public_email_requires_noreply_or_exact_approval(self):
        self.assertTrue(AUDIT.public_email(b"123+researcher@users.noreply.github.com"))
        self.assertTrue(AUDIT.public_email(b"pranayvandanapu2001@gmail.com"))
        self.assertEqual(AUDIT.scan_blob("approved.txt", b"pranayvandanapu2001@gmail.com"), [])
        self.assertFalse(AUDIT.public_email(b"another.person" + b"@gmail.com"))
        self.assertFalse(AUDIT.public_email(b"person" + b"@example.com"))

    def test_symlink_is_flagged_without_reading_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "link").symlink_to(root / "absent")
            report = AUDIT.audit(root)
            self.assertTrue(any(x["category"] == "symbolic_link" for x in report["findings"]))


if __name__ == "__main__":
    unittest.main()
