# Clickbait Type Classification — v1

Phân loại đa nhãn các dấu hiệu clickbait trong tiêu đề. Chỉ chọn nhãn được liệt kê trong `allowed_labels`; có thể trả về danh sách rỗng nếu không thấy dấu hiệu phù hợp.

- `curiosity_gap`: tiêu đề giấu chi tiết trung tâm để tạo khoảng trống thông tin.
- `exaggeration`: tiêu đề phóng đại mức độ hoặc ý nghĩa so với bằng chứng được cung cấp.
- `emotional`: tiêu đề dùng ngôn từ cảm xúc mạnh để kích động phản ứng.
- `question_based`: câu hỏi được dùng như thủ pháp câu khách, không chỉ là câu hỏi tường thuật thông thường.
- `misleading`: chỉ gán khi có toàn văn `content` để đối chiếu và thấy tiêu đề trình bày sai lệch đáng kể.

Không suy ra `misleading` từ tiêu đề đơn lẻ. Khi thiếu toàn văn, vẫn phân loại các dấu hiệu có thể nhận biết từ tiêu đề; không thêm `misleading`, và ghi ngắn trong `reason` rằng nhãn này chưa thể đánh giá vì thiếu toàn văn. Không coi `lead`/`sapo` là toàn văn. Nếu thiếu tiêu đề, dùng `insufficient_data`; nếu không chắc, dùng `abstain`.

Trả về JSON theo schema, không có nhãn ngoài taxonomy. `reason` ngắn, không đưa chain-of-thought. `confidence` là tự đánh giá, không phải xác suất đã hiệu chỉnh.
