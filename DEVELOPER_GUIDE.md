# Hướng dẫn phát triển — 0.2.0

## Thiết lập môi trường

Yêu cầu Python 3.11+. Tạo môi trường bằng `scripts\setup_windows.bat` trên Windows hoặc `sh scripts/setup_unix.sh` trên macOS/Linux. Chạy kiểm tra:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m compileall -q app.py src tests
.venv\Scripts\python.exe -m pip check
```

## Luồng module

- `src/data/loader.py` đọc CSV/XLSX/JSONL; `mapping.py` gợi ý map và chuẩn hóa record nhưng giữ bản sao cột nguồn; `exporter.py` tạo CSV/JSONL/summary/review/errors.
- `src/providers/base.py` định nghĩa adapter chung, `ProviderResponse` và lỗi đã làm sạch; các adapter Gemini/OpenAI dùng structured output; `MockProvider` không gọi mạng.
- `src/labeling/prompts.py` đọc prompt/taxonomy theo phiên bản. `schemas.py` kiểm tra cấu trúc Pydantic; `validation.py` kiểm tra taxonomy và ràng buộc input; `engine.py` thực hiện độc lập hoặc gộp.
- `src/jobs/storage.py` quản lý SQLite và prediction cache; `checkpoint.py` xây cache key; `runner.py` thực hiện RPM/concurrency/retry/budget/pause/resume.
- `src/ui/` chứa năm bước Streamlit; các worker chạy thread riêng và không gọi Streamlit API. `model_scan.py` quét toàn bộ model trong nền và lưu checkpoint; `model_checks.py` kiểm tra đúng cấu hình tác vụ; `local_memory.py` đọc/ghi nguyên tử và mã hóa key trên Windows.
- `src/labeling/probe.py` chứa bài mẫu chung. Model chỉ đạt nếu mọi tác vụ được chọn có output hợp lệ và `status=success`, không chỉ JSON hợp schema.
- `src/runtime.py` đồng bộ chạy job và quét model. `src/paths.py` tách thư mục kiểm thử bằng `KHDL_STATE_DIR`, `KHDL_INPUT_DIR`, `KHDL_OUTPUT_DIR`.

Job lưu canonical input và `_original` riêng để có thể resume và xuất lại mà không sửa nhãn nguồn. Job settings loại key/secret/token/password trước khi ghi DB. Key ở trong session/closure của worker; bản nhớ tùy chọn trên đĩa được mã hóa bằng Windows DPAPI. Không lưu key trong checkpoint model, job hoặc tệp xuất.

## SQLite, resume và cache

- `jobs`: provider/model, mode, task list, versions, settings an toàn, input fingerprint và trạng thái job.
- `job_items`: trạng thái cho từng `(job_id, record_id, task)`, output, số lần thử, usage và chi phí ước lượng.
- `prediction_cache`: output có cache key chứa input hash, provider/model, prompt/taxonomy/schema version, mode/task và các thiết lập ảnh hưởng đến nội dung gửi.

Independent cache key chỉ phụ thuộc tác vụ đó; combined cache key còn phụ thuộc tập task được gửi cùng request. Lỗi không được cache để Retry Failed có thể gọi model lại. Khi resume sau process restart, các item `running` được đưa về `pending`; kết quả checkpoint hoàn tất được giữ. Request có thể bị lặp nếu process tắt sau khi provider trả lời nhưng trước khi SQLite ghi checkpoint, vì API và SQLite không có giao dịch phân tán chung.

Thay đổi schema DB cần migration tường minh và test database cũ; `CREATE TABLE IF NOT EXISTS` chỉ khởi tạo cấu trúc hiện tại, không tự nâng cấp cột. Thay đổi output/schema/taxonomy/prompt phải tăng version để tách cache và giữ provenance.

## Thêm provider

1. Kế thừa `LLMProvider` và triển khai `validate_key`, `list_models`, `generate_structured`.
2. Parse/validate kết quả thành `output_schema`; không chấp nhận text tự do khi schema không hợp lệ.
3. Chuẩn hóa token usage khi provider trả về. Làm sạch thông báo lỗi; không bao giờ lưu key hoặc raw response chứa dữ liệu không cần thiết.
4. Đăng ký adapter trong `src/providers/__init__.py` và `src/ui/provider.py`.
5. Thêm test bằng fake SDK hoặc mock response; không cần API key thật trong unit tests.

API liệt kê model không xác nhận quyền chạy. Giao diện gọi `probe_labeling` theo tasks/mode bằng bài mẫu. Gemini dùng `response_json_schema` để bảo toàn enum số 0/1. Kiểm tra API thật tiêu thụ quota và có thể phát sinh chi phí.

## Thay prompt, schema hoặc taxonomy

- Lưu prompt thành file version mới, ví dụ `prompts/detection_v2.md`; thêm version trong `src/labeling/prompts.py` thay vì sửa prompt đang gắn với job cũ.
- Khi đổi output fields hoặc enum, cập nhật `schemas.py`, validation, prompt schema description và `SCHEMA_VERSION`.
- Khi taxonomy đổi, cập nhật `config/taxonomy.yaml`, nâng `version`, kiểm tra nhãn output và lưu taxonomy version theo job.
- Thêm regression test cho thiếu dữ liệu, nhãn ngoài taxonomy, truncation, cache key và export.

## Chạy và kiểm thử

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Unit tests phải dùng SQLite tạm trong `tmp_path`, mock provider và dữ liệu tổng hợp. Không commit tệp dữ liệu nghiên cứu, DB, output, API key, `.env` hoặc `.venv`.

Smoke test giao diện có thể dùng Streamlit `AppTest`; kiểm thử end-to-end API thật cần nêu rõ model/tài khoản và chi phí trước khi chạy.

### Kiểm tra bổ sung của bản 0.2.0

- Cache format 2 xét toàn bộ input/tập tác vụ của request gộp; retry một phần vẫn giữ nguyên ngữ cảnh request gốc. Thành công đã checkpoint không bị ghi đè.
- Rate limiter giãn các lệnh gọi thay vì gọi dồn đầu phút. 429 tôn trọng retry delay; job hết quota, key sai hoặc lỗi model được tạm dừng, phần chưa xử lý giữ pending. Chế độ mock không chờ tốc độ mạng.
- Checkpoint model chứa kết quả và metadata lượt quét trong một JSON nguyên tử theo context hash provider/key/tasks/mode/prompt/schema; đọc được định dạng dictionary cũ. Khi bị gián đoạn, tiếp tục từ phần đã lưu.
- Windows có khóa đọc/ghi và thử thay tệp lại khi bị khóa tạm thời. Lỗi lưu kéo dài được hiển thị; worker vẫn giữ tiến độ trong bộ nhớ và giao diện cho thử lưu lại.
- CSV xuất lại không ghi đè cột `pseudo_*` nguồn: tự dùng `pseudo2_*` hoặc số tiếp theo. Mock được đánh dấu `label_source=mock`.

Kiểm thử Chrome/Edge thật với dữ liệu giả và mock, không dùng key thật:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe scripts\browser_smoke.py
```

Script tạo server/thư mục riêng, thử năm trang, upload Excel, ba tác vụ, hai mode, dry-run, batch, tải ZIP, bộ lọc, lưu file và khôi phục phiên. Báo cáo: `docs/verification/browser-smoke.json`; ảnh: `docs/screenshots/`. Dùng `KHDL_TEST_BROWSER` để chỉ định trình duyệt; trên hệ khác cần cài Chromium của Playwright.

Kiểm thử API thật được gọi tường minh bằng một bài mẫu, ba tác vụ, hai mode:

```powershell
.venv\Scripts\python.exe scripts\live_smoke.py
```

Lệnh này dùng key đã lưu, có thể tốn quota/phí; báo cáo riêng trong `state/live_smoke_report.json` bị loại khỏi ZIP. Test suite mặc định không gọi mạng. `requirements-lock.txt` ghi môi trường Windows/Python 3.13 đã kiểm tra; dùng file này để tái lập đúng môi trường, còn Python khác dùng `requirements.txt` và kiểm tra lại tương thích.

## Gói phát hành

Tạo ZIP bằng `.venv\Scripts\python.exe scripts\package_release.py` trên Windows hoặc `.venv/bin/python scripts/package_release.py` trên macOS/Linux; bản hiện tại là `dist/KHDL_Auto_Labeler_v0.2.0.zip` (version lấy từ `pyproject.toml`). Script chỉ thêm mã nguồn, prompts, config, scripts, tests, tài liệu và `.gitkeep`; loại môi trường, dữ liệu riêng, output, SQLite, bản nhớ key và artifacts build; đồng thời dừng nếu phát hiện chuỗi giống API key.
