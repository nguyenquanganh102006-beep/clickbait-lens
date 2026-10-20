"""Verify a source ZIP in a fresh Windows virtual environment and browser."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen
from zipfile import ZipFile
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


def main():
    archive_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist" / "KHDL_Auto_Labeler_v0.2.0.zip"
    report = {"archive": archive_path.name, "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(), "checks": []}
    with tempfile.TemporaryDirectory(prefix="khdl-release-") as temporary:
        extracted = Path(temporary).resolve()
        with ZipFile(archive_path) as archive:
            for name in archive.namelist():
                if not (extracted / name).resolve().is_relative_to(extracted):
                    raise ValueError("Archive path escapes temporary workspace")
            archive.extractall(extracted)
        manifest = json.loads((extracted / "release-manifest.json").read_text(encoding="utf-8"))
        for name, expected_hash in manifest["files"].items():
            assert hashlib.sha256((extracted / name).read_bytes()).hexdigest() == expected_hash, name
        report["checks"].append("release manifest matches every extracted file")
        subprocess.run([sys.executable, "-m", "venv", str(extracted / ".venv")], check=True)
        python = extracted / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        print("Installing pinned dependencies into clean environment…", flush=True)
        installed = subprocess.run([str(python), "-m", "pip", "install", "-r", "requirements-lock.txt"], cwd=extracted, capture_output=True, text=True)
        if installed.returncode:
            raise RuntimeError(installed.stderr[-3000:])
        report["checks"].append("fresh virtual environment and pinned install")
        for args, label in [(["-m", "pip", "check"], "dependency consistency"),
                            (["-m", "compileall", "-q", "app.py", "src", "scripts", "tests"], "bytecode compilation"),
                            (["-m", "pytest", "-q"], "full test suite from extracted ZIP")]:
            print("Checking " + label, flush=True)
            result = subprocess.run([str(python), *args], cwd=extracted, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if result.returncode:
                raise RuntimeError(label + "\n" + result.stdout[-5000:] + result.stderr[-1000:])
            report["checks"].append(label)
            if "pytest" in args:
                report["pytest"] = result.stdout.strip().splitlines()[-1]
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with (extracted / "server-check.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([str(python), "-m", "streamlit", "run", "app.py", "--server.address", "127.0.0.1",
                "--server.port", str(port), "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                cwd=extracted, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            try:
                base = f"http://127.0.0.1:{port}"
                for _ in range(100):
                    try:
                        if urlopen(base + "/_stcore/health", timeout=1).read() == b"ok":
                            break
                    except OSError:
                        time.sleep(0.1)
                else:
                    raise RuntimeError("Clean release did not start")
                with sync_playwright() as playwright:
                    executable = os.environ.get("KHDL_TEST_BROWSER")
                    if not executable and os.name == "nt":
                        executable = next(str(path) for path in [Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
                            Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")] if path.exists())
                    browser = playwright.chromium.launch(headless=True, **({"executable_path": executable} if executable else {}))
                    page = browser.new_page(viewport={"width": 1440, "height": 1000})
                    page.goto(base)
                    expect(page.get_by_role("heading", name="Cấu hình nhà cung cấp", exact=True)).to_be_visible(timeout=30000)
                    expect(page.get_by_text("v0.2.0", exact=False)).to_be_visible()
                    expect(page.get_by_role("button", name="Tự kiểm tra toàn bộ model và lưu kết quả", exact=True)).to_be_disabled()
                    expect(page.locator('[data-testid="stException"]')).to_have_count(0)
                    page.screenshot(path=str(ROOT / "docs" / "screenshots" / "clean-start.png"), full_page=True)
                    browser.close()
                report["checks"].append("clean release starts in Chrome without private runtime data")
            finally:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
                else:
                    process.terminate()
                process.wait(timeout=15)
    report["passed"] = True
    target = ROOT / "docs" / "verification" / "clean-release.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
