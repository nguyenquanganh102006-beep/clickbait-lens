# Nghiệm thu KHDL Auto Labeler 0.2.0

Ngày kiểm tra: 20/10/2026. Môi trường thực tế: Windows, Python 3.13, Chrome chạy headless bằng Playwright. Không đổi định nghĩa ba tác vụ hoặc taxonomy/prompt/schema v1.

## Phạm vi và bằng chứng

| Luồng | Kết quả | Bằng chứng |
|---|---|---|
| Khởi chạy, năm bước và trạng thái phiên | Qua trên trình duyệt thật | `browser-smoke.json`, ảnh `../screenshots/` |
| CSV tiếng Việt, Excel nhiều sheet, JSONL | Qua | `tests/test_loader.py`, `tests/test_end_to_end.py` |
| Ánh xạ, thiếu tiêu đề/toàn văn, ID/URL trùng, URL hỏng, header số | Qua | `tests/test_mapping.py`, `tests/test_ui_workflow.py` |
| Bảo toàn nhãn nguồn, cột `pseudo_*` cũ và Unicode | Qua | `tests/test_exporter.py`, `tests/test_end_to_end.py` |
| Đọc và khôi phục dữ liệu nhớ thực tế | 28.000 dòng, 15 cột, 0 thiếu tiêu đề, 0 ID trùng; khoảng 1,33 giây | Kiểm tra chỉ đọc trên máy người dùng; không đưa nội dung dữ liệu vào gói phát hành |
| Nhớ key, model, dữ liệu và cấu hình tác vụ | Qua, kể cả phiên mới | `tests/test_local_memory.py`, `tests/test_ui_workflow.py`, `browser-smoke.json` |
| Key tách theo nhà cung cấp | Qua; chuyển nhà cung cấp không gửi lại key cũ | `test_provider_switch_does_not_send_previous_providers_key` |
| Quét toàn bộ, lưu sau từng model, chờ và thử lại 429 | Qua | `tests/test_model_scan.py` |
| Tiếp tục lượt quét bị gián đoạn | Qua; bỏ qua model đã xong, chỉ thử phần còn lại | `test_resume_scan_skips_finished_models_and_retries_only_pending_429` |
| Lỗi 503/504 tạm thời và lỗi quyền truy cập | Qua; retry có giới hạn, auth dừng rõ nguyên nhân | `tests/test_model_scan.py`, `tests/test_provider.py` |
| Khóa tệp Windows | Qua với lỗi mô phỏng và khóa file thật bằng Win32 API; bản cũ được giữ | `tests/test_local_memory.py` |
| Nút kiểm tra model trực tiếp ở bước 4 | Qua; lỗi mới vô hiệu hóa kết quả đạt cũ | `test_saved_model_verdict_restores_and_step4_fresh_check_invalidates` |
| Ba tác vụ, độc lập và gộp | Qua trong engine, runner, export và API Gemini thật | `tests/test_end_to_end.py`, `live-smoke-summary.json` |
| Dry-run và chạy toàn bộ | Qua trên Chrome | `browser-smoke.json` |
| Tạm dừng, dừng, tiếp tục, thử lại lỗi, checkpoint | Qua; giữ thành công đã lưu, không chạy lại toàn bộ | `tests/test_resume.py` |
| Giới hạn tốc độ và dừng khi quota/model/key từ chối | Qua; không làm lỗi hàng loạt phần chưa xử lý | `tests/test_rate_limit.py`, `tests/test_resume.py` |
| Cache độc lập/gộp và cache format 2 | Qua; chế độ gộp xét toàn bộ input và giữ ngữ cảnh khi retry một phần | `tests/test_resume.py` |
| Ngưỡng ngân sách ở engine | Qua trong kiểm thử backend; không tự điền đơn giá API | `test_budget_guard_checkpoints_then_resumes_after_user_raises_limit` |
| Lịch sử và kết quả chọn cùng job | Qua | `test_job_selection_is_shared_between_history_and_results` |
| Bộ lọc, làm mới, năm tệp xuất và ZIP | Qua trên Chrome; bộ lọc không cắt dữ liệu tải xuống | `browser-smoke.json`, `tests/test_exporter.py` |
| Mô phỏng không gọi API | Qua | Test suite, browser smoke có `external_api_calls=0` |
| Gói ZIP không chứa key/dữ liệu riêng | Qua kiểm thử danh sách file và quét credential; có manifest SHA-256 | `tests/test_package.py` |
| Giải nén, cài sạch, pytest, compile, pip check, mở Chrome | Qua | Receipt `clean-release.json` bên ngoài ZIP để tránh vòng lặp hash |

## Kiểm thử API thật

Provider: Gemini. Model: `gemini-3.5-flash-lite`. Một bài mẫu do ứng dụng cung cấp, ba tác vụ, hai chế độ; mỗi chế độ có ba kết quả thành công và không có lỗi. Chi tiết riêng giữ trong `state/live_smoke_report.json`, không chia sẻ trong gói phát hành. Bản tổng hợp không có key hoặc dữ liệu người dùng: `live-smoke-summary.json`.

OpenAI được kiểm tra bằng SDK giả lập về cấu trúc request, strict JSON schema, parse, usage và xử lý lỗi. Chưa gọi API OpenAI thật vì máy hiện không có key OpenAI được cung cấp. Không coi kiểm thử giả lập là bằng chứng tài khoản OpenAI có quota/quyền chạy.

## Giới hạn thực tế

- Kết quả model đã lưu phản ánh lần kiểm tra gần nhất. Quota/quyền truy cập và dịch vụ ngoài có thể đổi; dùng nút bước 4 để kiểm tra lại trước khi chạy dữ liệu.
- Không thể bảo đảm dịch vụ không trả 400/404/429/503/504; ứng dụng kiểm soát retry, chờ, dừng và checkpoint thay vì che giấu lỗi.
- Nếu process dừng sau khi API trả lời nhưng trước khi checkpoint được ghi, request đang dở có thể cần gọi lại; tác vụ đã checkpoint được giữ.
- Nhớ key được mã hóa hiện dùng Windows DPAPI. Dữ liệu nhớ trên máy không được mã hóa, và bị loại khỏi gói chia sẻ.

## Lệnh tái kiểm tra

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m compileall -q app.py src scripts tests
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe scripts\browser_smoke.py
.venv\Scripts\python.exe scripts\package_release.py
.venv\Scripts\python.exe scripts\verify_release.py
```

`live_smoke.py` và `probe_saved_models.py` là lệnh gọi API thật, chỉ chạy tường minh vì có tiêu thụ quota/phí. Test suite mặc định và browser smoke không gọi API ngoài.
