import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

SCRIPT = Path(__file__).with_name("verify_mentor_zip.py")

spec = importlib.util.spec_from_file_location("verify_mentor_zip", SCRIPT)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class TestVerifyMentorZip(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.zip_path = self.root / "test.zip"

    def create_zip(self, entries):
        with zipfile.ZipFile(self.zip_path, "w") as archive:
            for name, content in entries.items():
                archive.writestr(name, content)

    def run_verifier(self):
        original_path = verifier.ZIP_PATH
        verifier.ZIP_PATH = self.zip_path
        output = StringIO()
        try:
            with redirect_stdout(output):
                result = verifier.main()
        finally:
            verifier.ZIP_PATH = original_path
        return result, output.getvalue()

    def test_valid_zip_passes(self):
        self.create_zip({
            "final_application/app.py": "print('hello')",
            "final_application/README.md": "# OCR",
            "final_application/requirements.txt": "streamlit",
        })
        result, output = self.run_verifier()
        self.assertEqual(result, 0, output)

    def test_missing_required_file_fails(self):
        self.create_zip({
            "final_application/app.py": "print('hello')",
            "final_application/README.md": "# OCR",
        })
        result, output = self.run_verifier()
        self.assertEqual(result, 1)
        self.assertIn("missing", output.lower())

    def test_env_file_is_flagged(self):
        self.create_zip({
            "final_application/app.py": "print('hello')",
            "final_application/README.md": "# OCR",
            "final_application/requirements.txt": "streamlit",
            ".env": "API_KEY=example",
        })
        result, output = self.run_verifier()
        self.assertEqual(result, 1)
        self.assertIn(".env", output)

    def test_corrupt_zip_fails(self):
        self.zip_path.write_bytes(b"this is not a zip file")
        result, output = self.run_verifier()
        self.assertEqual(result, 1)
        self.assertIn("FAIL", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
