# KHDL Auto Labeler 4.1.0

Công cụ tạo pseudo-label cho bài báo tiếng Việt từ tiêu đề, toàn văn và ảnh JPEG. Phiên bản chính trên nhánh `main` chỉ dùng quy trình v4.1: hai lượt đánh giá mù, phân xử tối đa một lượt khi cần, bằng chứng tham chiếu đoạn và đồng thuận theo từng nhãn.

## Cài đặt và chạy

Yêu cầu Python 3.11 trở lên.

```powershell
git clone https://github.com/nguyenquanganh102006-beep/clickbait-lens.git
cd clickbait-lens
.\scripts\setup_windows.bat
.\scripts\run_windows.bat
```

Có thể mở nhanh bằng `CHAY_UNG_DUNG.bat`. Trên macOS/Linux: `sh scripts/setup_unix.sh`, sau đó `sh scripts/run_unix.sh`. Mở <http://127.0.0.1:8501>.

1. **Kết nối AI**: nhập Gemini/OpenAI key và model hỗ trợ ảnh + structured output, lưu cấu hình. Lưu không gọi API. Có thể nhớ kết nối trên Windows bằng DPAPI.
2. **Gán nhãn đa phương thức**: nhập đường dẫn CSV, thư mục gốc giải `image_path`, số bài và chế độ ngoại tuyến/API. Ví dụ `images/a_1.jpg` tương ứng gốc `clean_final`.
3. Tải CSV kết quả, kiểm tra flags và bằng chứng. Mô phỏng không tạo nhãn train.

CSV đầu vào cần `article_id`, `title`, `content`, `image_path`; các trường như `lead`, `image_caption` được dùng khi có. Giữ dữ liệu nguồn và ảnh bên ngoài Git. Đường dẫn mặc định trong giao diện trỏ đến `../data_pr/clean_final`; có thể thay bằng đường dẫn trên máy khác.

## Nhãn và review

- Detection: 0/1/Unknown.
- Năm Type: curiosity_gap, exaggeration, emotional_bait, question_based, misleading_presentation; ba subtype misleading chỉ là metadata.
- TC/TI/IC: Consistent, Inconsistent, Insufficient Evidence. Not Available mô tả thiếu modality, không phải lớp train.
- `label_mask_json` xác định target có ít nhất hai phiếu hợp lệ đồng thuận. Chỉ dùng target có mask=true; Unknown không bị ép thành 0.
- Lưu evidence, phản hồi từng lượt, model, usage, timestamp và checksum. Toàn bộ human review vẫn pending cho đến khi người xác nhận.

Xem [hướng dẫn nhãn và phục hồi](docs/ANNOTATION.md). Các lượt mù dùng cùng model qua các yêu cầu riêng, có thể cùng sai; cần gold độc lập để đánh giá accuracy/F1. Mỗi bài dùng 2–3 phân tích, mỗi yêu cầu retry tạm thời tối đa hai lần, tôn trọng nhịp API đã cấu hình.

## CLI pilot

CLI giới hạn 1–10 bài mỗi lần để kiểm chứng có kiểm soát:

```powershell
.\.venv\Scripts\python.exe scripts/annotate.py --csv ..\data_pr\clean_final\sample_1000.csv --image-root ..\data_pr\clean_final --limit 10 --live --output data/output/pilot.csv
```

Mặc định dùng kết nối mã hóa đã lưu. Thêm `--key-stdin --provider gemini --model gemini-3.1-flash-lite` để nhập key tạm qua ô ẩn trong terminal; không truyền key trong command line. Thêm `--resume` để tiếp tục; thêm `--retry-incomplete` để chạy lại riêng dòng review lỗi API. Cache phải khớp model/prompt/schema/input/ảnh/checksum. Không ghi đè CSV nguồn.

Kiểm tra kết quả đã lưu không gọi API:

```powershell
.\.venv\Scripts\python.exe scripts/validate_quality_pilot.py input.csv output.csv receipt.json
```

## Phát triển

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/package_release.py
```

Mã chính ở `src/labeling/quality.py`; tiện ích input/evidence ở `src/labeling/multimodal.py`; schema API ở `config/annotation_response.schema.json`; prompt ở `prompts/`. Gói phát hành loại trừ dataset, key và state cục bộ. [Danh mục dọn dẹp](docs/CLEANUP.json) ghi các file đã xóa khi chọn bản chính; lịch sử code cũ vẫn có trong Git.
