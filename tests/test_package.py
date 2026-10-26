from zipfile import ZipFile
import hashlib
import json

from scripts.package_release import create_archive


def test_release_archive_contains_app_and_excludes_local_data_and_runtime(tmp_path):
    archive_path = create_archive(tmp_path / "release.zip")

    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("release-manifest.json"))
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        assert manifest["version"] == "0.2.0"

    assert "app.py" in names
    assert "USER_GUIDE.md" in names
    assert "docs/screenshots/app.png" in names
    assert "src/jobs/runner.py" in names
    assert {"prompts/detection_v2.md", "prompts/type_v2.md", "prompts/consistency_v2.md", "docs/LABELING_V2.md"} <= names
    assert "data/input/.gitkeep" in names
    assert not any(name.startswith(".venv/") or "/__pycache__/" in name for name in names)
    assert not any(name.startswith("state/") and not name.endswith(".gitkeep") for name in names)
    assert not any(name.startswith("data/input/") and not name.endswith(".gitkeep") for name in names)
    assert not any(name == ".env" or name.startswith(".env.") and name != ".env.example" for name in names)
