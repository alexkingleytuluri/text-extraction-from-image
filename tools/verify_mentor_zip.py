import sys
import zipfile
from pathlib import Path

ZIP_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("OCR_Project_Mentor.zip")

REQUIRED_FILES = {
    "final_application/app.py",
    "final_application/README.md",
    "final_application/requirements.txt",
}

FORBIDDEN_NAMES = {".env", ".env.local", ".env.production"}
FORBIDDEN_SUFFIXES = {".joblib", ".pth", ".safetensors", ".pyc"}
FORBIDDEN_PARTS = {".git", "__pycache__"}


def main():
    if not ZIP_PATH.is_file():
        print(f"FAIL: ZIP file not found: {ZIP_PATH}")
        return 1

    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as archive:
            bad_member = archive.testzip()
            if bad_member:
                print(f"FAIL: Corrupt archive member: {bad_member}")
                return 1

            names = set(archive.namelist())
    except (OSError, zipfile.BadZipFile) as exc:
        print(f"FAIL: Cannot read ZIP: {exc}")
        return 1

    normalized_names = {name.replace("\\", "/").lstrip("./") for name in names}

    missing = sorted(REQUIRED_FILES - normalized_names)
    if missing:
        print("FAIL: Required files are missing:")
        for name in missing:
            print(f"  - {name}")
        return 1

    flagged = []
    for name in sorted(normalized_names):
        path = Path(name)
        parts = set(path.parts)
        filename = path.name.lower()

        if filename in FORBIDDEN_NAMES:
            flagged.append(name)
        elif path.suffix.lower() in FORBIDDEN_SUFFIXES:
            flagged.append(name)
        elif parts & FORBIDDEN_PARTS:
            flagged.append(name)

    if flagged:
        print("FAIL: Review these potentially sensitive or unwanted entries:")
        for name in flagged:
            print(f"  - {name}")
        return 1

    print(f"PASS: ZIP is readable ({len(normalized_names)} entries checked).")
    print("PASS: All required application files are present.")
    print("PASS: No configured sensitive or unwanted file patterns found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
