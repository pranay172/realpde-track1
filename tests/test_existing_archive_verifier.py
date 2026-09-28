import sys
from pathlib import Path
import tempfile
import unittest
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from verify_existing_submission import inspect_zip


class ArchiveInspectionTests(unittest.TestCase):
    def test_accepts_root_entrypoint(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"ok.zip"
            with zipfile.ZipFile(p,"w") as z:
                z.writestr("submission.py","pass")
            inspect_zip(p)

    def test_rejects_path_traversal_and_symlinks(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"bad.zip"
            for name in ("../escape.py","/absolute.py","nested\\bad.py"):
                with zipfile.ZipFile(p,"w") as z:
                    z.writestr("submission.py","pass")
                    z.writestr(name,"pass")
                with self.assertRaises(ValueError):
                    inspect_zip(p)
            with zipfile.ZipFile(p,"w") as z:
                z.writestr("submission.py","pass")
                link=zipfile.ZipInfo("link")
                link.external_attr=0o120777 << 16
                z.writestr(link,"/elsewhere")
            with self.assertRaises(ValueError):
                inspect_zip(p)
