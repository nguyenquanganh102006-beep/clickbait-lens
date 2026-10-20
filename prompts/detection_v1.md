# Clickbait Detection — v1

Gán nhãn tiêu đề tin tức theo yêu cầu Clickbait Detection.

- `1` (Clickbait): tiêu đề dùng cách trình bày câu khách rõ rệt, chẳng hạn cố ý giấu thông tin then chốt để kích thích tò mò, phóng đại đáng kể hoặc gây cảm xúc mạnh nhằm kéo người đọc.
- `0` (Non-clickbait): tiêu đề mô tả nội dung tin theo cách hấp dẫn thông thường nhưng không có dấu hiệu câu khách rõ rệt.
- Không suy luận chỉ từ việc tiêu đề có câu hỏi, dấu chấm than hoặc chủ đề gây chú ý.
- Nếu không đủ dữ liệu để đánh giá, dùng trạng thái `insufficient_data`; nếu trường hợp thực sự mơ hồ, dùng `abstain`.
- `reason` phải ngắn và trích dẫn dấu hiệu trong tiêu đề; không đưa chain-of-thought.

Trả về đúng schema được yêu cầu. `confidence` là tự đánh giá, không phải xác suất đã hiệu chỉnh.
