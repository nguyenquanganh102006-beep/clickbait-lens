"""Create a shareable ZIP while excluding local state and credentials."""

from __future__ import annotations

import re
import sys
import tomllib
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
ARCHIVE_PATH = ROOT / "dist" / f"KHDL_Auto_Labeler_v{VERSION}.zip"
ROOT_FILES = (
    ".env.example",
    ".gitignore",
    "app.py",
    "pyproject.toml",
    "README.md",
    "USER_GUIDE.md",
    "DEVELOPER_GUIDE.md",
    "requirements.txt",
    "requirements-lock.txt",
    "requirements-dev.txt",
    "CHAY_UNG_DUNG.bat",
    "KHDL_Auto_Labeler_Implementation_Plan.md",
)
PACKAGE_DIRS = ("src", "config", "prompts", "scripts", "tests", "docs")
PLACEHOLDER_DIRS = (
    "data/input",
    "data/output",
    "data/samples",
    "state",
)
SKIP_PARTS = {".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist", ".git"}
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)


def _is_shareable(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if relative.as_posix() in {"docs/verification/clean-release.json", "docs/screenshots/clean-start.png"}:
        # Verification receipts bind the ZIP hash; including them in that ZIP
        # would change the very artifact they attest to.
        return False
    if any(part in SKIP_PARTS for part in relative.parts):
        return False
    if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
        return False
    if relative.parts[0] == "data":
        return path.name == ".gitkeep"
    return path.is_file()


def _collect_files() -> list[Path]:
    candidates = [ROOT / name for name in ROOT_FILES]
    for directory in PACKAGE_DIRS:
        candidates.extend((ROOT / directory).rglob("*"))
    for directory in PLACEHOLDER_DIRS:
        candidates.extend((ROOT / directory).glob(".gitkeep"))
    return sorted({path for path in candidates if path.is_file() and _is_shareable(path)})


def _check_for_credentials(paths: list[Path]) -> None:
    for path in paths:
        if path.suffix.casefold() not in {".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".bat", ".sh", ".ps1"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            raise RuntimeError(f"Có chuỗi giống credential trong {path.relative_to(ROOT)}; chưa tạo ZIP.")


def create_archive(destination: Path = ARCHIVE_PATH) -> Path:
    paths = _collect_files()
    _check_for_credentials(paths)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
        manifest = {"version": VERSION, "files": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
        archive.writestr("release-manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return destination


if __name__ == "__main__":
    try:
        archive_path = create_archive()
    except (OSError, RuntimeError) as exc:
        print(f"Could not create release archive: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Created release archive: {archive_path}")
