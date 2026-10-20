# Title–Content Consistency — v1

Đối chiếu tiêu đề với toàn văn bài báo và gán đúng một nhãn:

- `consistent`: nội dung hỗ trợ các khẳng định chính của tiêu đề.
- `partial`: nội dung có liên quan nhưng chỉ hỗ trợ một phần, thiếu ngữ cảnh quan trọng hoặc tiêu đề nhấn mạnh quá mức một chi tiết.
- `inconsistent`: nội dung mâu thuẫn với khẳng định chính hoặc không hỗ trợ đáng kể điều tiêu đề hứa hẹn.

Chỉ đánh giá trên toàn văn `content`. Không coi `lead`/`sapo` hoặc bản xem trước là toàn văn. Nếu thiếu nội dung hoặc nội dung quá ngắn để đối chiếu, dùng `insufficient_data`; nếu nội dung bị cắt hoặc trường hợp không thể kết luận, dùng `abstain`.

`evidence` phải ngắn, dựa trên phần tiêu đề và nội dung được cung cấp. Không đưa chain-of-thought. `confidence` là tự đánh giá, không phải xác suất đã hiệu chỉnh.
