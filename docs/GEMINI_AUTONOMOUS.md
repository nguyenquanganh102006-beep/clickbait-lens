# Chạy Gemini tự động cho dữ liệu Tuấn

Chạy từ thư mục repository:

```powershell
.\scripts\run_gemini_autonomous.bat
```

Nhập API key tại lời nhắc ẩn. Key chỉ ở bộ nhớ, không ghi vào tệp hoặc tham số tiến trình.

Luồng tự đọc toàn bộ bài của Nguyễn Minh Tuấn trong `../data_pr/clean_final/data.csv`, dùng ảnh thật và schema hiện hành, tiếp tục checkpoint cũ và bỏ qua bài đã có đủ nhãn trong CSV xuất bản. Không cần tạo packet, chia khoảng bài hoặc giao batch Gemini. Mặc định Gemini 3.5 Flash Lite, một lượt/bài, 4 luồng, giới hạn chung 10 lượt/phút.

CSV được cập nhật tự động tại `../data_pr/clean_final/data_tuan_labeled.csv`. Các nhãn Luna nộp thêm và kết quả review của leader cũng được nhập vào CSV. Bài thiếu nhãn hoặc có vấn đề cần review được giữ riêng, không điền nhãn giả cho đủ cột.

Khi API báo hết quota ngày, luồng ngừng nộp yêu cầu mới, đợi các yêu cầu đang chạy và ghi kết quả còn lại, xuất CSV lần cuối rồi lưu báo cáo. Lần chạy tiếp theo tự bỏ qua checkpoint đã có. Lỗi xác thực/cấu hình hoặc lỗi hạ tầng lặp lại cũng có thể dừng luồng; xem trạng thái thực tế trong báo cáo. Không tự đổi model hay khẳng định biết quota còn lại.

Các tệp trong `data/output/tuan_run/`:

- `autonomous_status.json`: tiến độ worker và số bài đã xuất bản.
- `autonomous_events.jsonl`: nhật ký tiến độ.
- `autonomous_final_report.json`: lý do dừng và kết quả cuối đợt.
- `autonomous_review_sample.json`: mẫu khoảng 5% **chờ** leader review, không tự nhận là đã review.
- `fast_run/checkpoint.jsonl`: kết quả từng bài để tiếp tục an toàn.

Không chạy thêm một publisher hoặc worker thủ công cùng lúc với luồng này.

Khi chủ động chọn model khác, dùng `--model TEN_MODEL`. Model mặc định giữ checkpoint cũ trong `fast_run`; các model khác dùng thư mục riêng dưới `model_runs`. Publisher gộp các checkpoint này vào cùng CSV và tiếp tục bỏ qua bài đã đủ nhãn, không thay tên model hoặc ghi đè checkpoint của model trước. Luồng không tự chuyển model khi quota hết.
