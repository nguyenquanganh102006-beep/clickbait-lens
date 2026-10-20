"""Isolated browser acceptance test. Uses synthetic data and offline provider only."""
from __future__ import annotations

from io import BytesIO
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

import pandas as pd
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    with tempfile.TemporaryDirectory(prefix="khdl-browser-") as temporary:
        workspace = Path(temporary)
        for key, value in {
            "KHDL_STATE_DIR": workspace / "state", "KHDL_INPUT_DIR": workspace / "input",
            "KHDL_OUTPUT_DIR": workspace / "output",
        }.items():
            os.environ[key] = str(value)
            value.mkdir(parents=True)
        from src.ui import local_memory
        local_memory.save_provider({"provider": "mock", "api_key": "", "model": "mock-v1",
                                    "timeout_seconds": 10, "requests_per_minute": 10,
                                    "concurrency": 1, "max_retries": 0})
        local_memory.save_workflow(["detection", "type", "consistency"], "independent")
        sample = pd.DataFrame({"id": ["demo-1", "demo-2", "demo-3"],
            "title": ["Thư viện mở cửa trở lại", "Tin này khiến bạn bất ngờ?", "Thông báo lịch hoạt động"],
            "content": ["Thư viện mở cửa từ 8 giờ đến 17 giờ mỗi ngày và phục vụ bạn đọc trong toàn thành phố. " * 3] * 3,
            "label": [0, 1, 0], "source": ["Dữ liệu minh họa"] * 3})
        sample_path = workspace / "sample.xlsx"
        sample.to_excel(sample_path, index=False)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        log_path = workspace / "server.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py",
                "--server.address", "127.0.0.1", "--server.port", str(port),
                "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            try:
                base_url = f"http://127.0.0.1:{port}"
                for _ in range(100):
                    try:
                        if urlopen(base_url + "/_stcore/health", timeout=1).read() == b"ok":
                            break
                    except OSError:
                        time.sleep(0.1)
                else:
                    raise RuntimeError("Isolated server did not become ready")
                screenshots = ROOT / "docs" / "screenshots"
                screenshots.mkdir(parents=True, exist_ok=True)
                with sync_playwright() as playwright:
                    executable = os.environ.get("KHDL_TEST_BROWSER")
                    if not executable and os.name == "nt":
                        for candidate in [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                                          r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]:
                            if Path(candidate).exists():
                                executable = candidate
                                break
                    browser = playwright.chromium.launch(headless=True, **({"executable_path": executable} if executable else {}))
                    page = browser.new_page(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
                    page.set_default_timeout(30000)
                    page.goto(base_url)

                    def go(step):
                        page.get_by_text(step, exact=True).click()
                        expect(page.get_by_role("radio", name=step, exact=True)).to_be_checked()
                        headings = {"2. Nạp dữ liệu": "Nạp dữ liệu", "3. Chọn tác vụ": "Cấu hình tác vụ và prompt",
                                    "4. Chạy gán nhãn": "Chạy, tạm dừng và tiếp tục job", "5. Xem & tải kết quả": "Rà soát và xuất kết quả"}
                        expect(page.get_by_role("heading", name=headings[step], exact=True)).to_be_visible()
                        expect(page.locator('[data-testid="stException"]')).to_have_count(0)

                    expect(page.get_by_text("Cấu hình nhà cung cấp", exact=True)).to_be_visible()
                    page.screenshot(path=str(screenshots / "connection.png"), full_page=True)
                    go("2. Nạp dữ liệu")
                    page.locator('input[type="file"]').set_input_files(sample_path)
                    page.get_by_role("button", name="Đọc tệp", exact=True).click()
                    expect(page.get_by_text("Ánh xạ cột chuẩn", exact=True)).to_be_visible()
                    page.get_by_role("button", name="Lưu dữ liệu và ánh xạ trên máy", exact=True).click()
                    expect(page.get_by_text("Đã nhớ bộ dữ liệu. Lần mở sau không cần tải tệp lên lại.", exact=True)).to_be_visible()
                    page.screenshot(path=str(screenshots / "app.png"), full_page=True)
                    go("3. Chọn tác vụ")
                    expect(page.get_by_text("Đã chọn tác vụ. Tiếp theo: chọn bước 4 để chạy thử.", exact=True)).to_be_visible()
                    page.get_by_text("Gộp trong một yêu cầu", exact=True).click()
                    expect(page.get_by_text("Chế độ đang áp dụng: Gộp", exact=True)).to_be_visible()
                    page.screenshot(path=str(screenshots / "tasks.png"), full_page=True)
                    go("4. Chạy gán nhãn")
                    page.get_by_role("button", name="Kiểm tra model đang chọn ngay bây giờ", exact=True).click()
                    expect(page.get_by_text("Model vừa gọi thử thành công với tác vụ hiện tại.", exact=True)).to_be_visible()
                    page.get_by_role("button", name="Tạo và chạy thử", exact=True).click()
                    expect(page.get_by_text("Lượt chạy đã kết thúc. Chọn bước 5 để rà soát và tải kết quả.", exact=True)).to_be_visible()
                    page.screenshot(path=str(screenshots / "jobs.png"), full_page=True)
                    go("5. Xem & tải kết quả")
                    with page.expect_download() as download_info:
                        page.get_by_role("button", name="Tải toàn bộ kết quả (.zip)", exact=True).click()
                    archive_path = workspace / "results.zip"
                    download_info.value.save_as(archive_path)
                    with ZipFile(archive_path) as archive:
                        assert len(archive.namelist()) == 5
                        labels = pd.read_csv(BytesIO(archive.read("labeled_dataset.csv")))
                        assert labels["label"].tolist() == [0, 1, 0]
                        summary = json.loads(archive.read("summary.json"))
                        assert summary["records_exported"] == 3
                        assert summary["error_count"] == 0
                    page.get_by_role("button", name="Lưu toàn bộ tệp vào data/output", exact=True).click()
                    expect(page.get_by_text("Đã lưu 5 tệp tại", exact=False)).to_be_visible()
                    page.get_by_role("textbox", name="Tìm trong tiêu đề hoặc URL", exact=True).fill("does-not-exist-xyz")
                    page.get_by_role("textbox", name="Tìm trong tiêu đề hoặc URL", exact=True).press("Enter")
                    expect(page.get_by_text("Hiển thị 0 / 9 dòng tác vụ.", exact=True)).to_be_visible()
                    page.get_by_role("textbox", name="Tìm trong tiêu đề hoặc URL", exact=True).fill("")
                    page.get_by_role("textbox", name="Tìm trong tiêu đề hoặc URL", exact=True).press("Enter")
                    page.screenshot(path=str(screenshots / "results.png"), full_page=True)
                    go("3. Chọn tác vụ")
                    page.get_by_text("Độc lập (khuyến nghị)", exact=True).click()
                    expect(page.get_by_text("Chế độ đang áp dụng: Độc lập", exact=True)).to_be_visible()
                    go("4. Chạy gán nhãn")
                    page.get_by_role("button", name="Tạo và chạy toàn bộ", exact=True).click()
                    expect(page.get_by_text("Chạy toàn bộ · mock / mock-v1 · independent", exact=False)).to_be_visible()
                    expect(page.get_by_text("Lượt chạy đã kết thúc. Chọn bước 5 để rà soát và tải kết quả.", exact=True)).to_be_visible()
                    go("5. Xem & tải kết quả")
                    expect(page.get_by_text("Mode: independent", exact=False)).to_be_visible()
                    with page.expect_download() as full_download_info:
                        page.get_by_role("button", name="Tải toàn bộ kết quả (.zip)", exact=True).click()
                    full_archive = workspace / "full-results.zip"
                    full_download_info.value.save_as(full_archive)
                    with ZipFile(full_archive) as archive:
                        full_summary = json.loads(archive.read("summary.json"))
                        assert full_summary["job"]["mode"] == "independent"
                        assert full_summary["job"]["status"] == "completed"
                        assert full_summary["settings"]["job_kind"] == "batch"
                    page.reload()
                    expect(page.get_by_text("Cấu hình nhà cung cấp", exact=True)).to_be_visible()
                    go("2. Nạp dữ liệu")
                    expect(page.get_by_text("Ánh xạ cột chuẩn", exact=True)).to_be_visible()
                    go("3. Chọn tác vụ")
                    expect(page.get_by_role("radio", name="Độc lập (khuyến nghị)", exact=True)).to_be_checked()
                    expect(page.locator('[data-testid="stException"]')).to_have_count(0)
                    browser.close()
                report = {"passed": True, "provider": "mock", "external_api_calls": 0,
                          "checks": ["five pages", "xlsx upload", "mapping", "remember dataset",
                                     "three tasks", "combined mode", "fresh model check", "dry-run job",
                                     "ZIP download", "gold labels preserved", "save five files", "search filter", "full batch job",
                                     "fresh session restores provider, dataset and workflow"],
                          "records": 3, "task_results": 9}
                destination = ROOT / "docs" / "verification" / "browser-smoke.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps(report, ensure_ascii=True))
            finally:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
                else:
                    process.terminate()
                process.wait(timeout=15)
                if process.returncode not in (0, 1):
                    print("Isolated test server stopped")


if __name__ == "__main__":
    main()
