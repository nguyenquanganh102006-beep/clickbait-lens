# Clickbait Type — v2

Phân loại các THỦ PHÁP CÂU KHÁCH của tiêu đề, không phân loại mọi cách viết hấp dẫn. Áp dụng cùng ngưỡng thận trọng với Detection: phải có dấu hiệu rõ vượt cách đưa tin, hỏi đáp, chân dung, hướng dẫn thông thường. Không suy ra có clickbait chỉ vì mẫu được đưa vào bước Type. Nếu không có thủ pháp đủ rõ, success với labels=[]; nếu ranh giới không phân xử được, abstain với labels=[]. Không tự tạo nhãn để khớp một quyết định Detection trước đó.

Chỉ dùng allowed_labels:
- curiosity_gap: giữ lại thông tin trung tâm mà câu hứa tiết lộ, bằng cách đóng khung mơ hồ/kích tò mò; thiếu tên người hoặc địa điểm một mình không đủ. “Động thái kỳ lạ…” giữ lại động thái là ví dụ rõ hơn một tít tin chỉ thiếu tên riêng.
- exaggeration: mức độ, quy mô, độ chắc chắn được thổi phồng rõ, không phải cứ có số lớn, thành tích hiếm, ẩn dụ hoặc danh hiệu là phóng đại. Khi viện dẫn so sánh với nội dung phải chỉ ra khẳng định cụ thể không được hỗ trợ.
- emotional: ngôn từ cảm xúc mạnh được dùng để kích động, không phải chỉ vì chủ đề buồn, nguy hiểm hoặc từ tiếc nuối/xúc động trong tin chân dung.
- question_based: câu hỏi là thủ pháp câu khách, KHÔNG phải mọi câu hỏi tư vấn/kiến thức/phỏng vấn. “Tốc độ chạy của con người giới hạn ở đâu?” không tự động là clickbait.
- misleading: chỉ khi có body content để đối chiếu và thấy một khẳng định chính của tiêu đề sai lệch đáng kể. Nêu đúng mâu thuẫn, không suy từ điều chưa xuất hiện trong một đoạn. Nếu có dấu hiệu content bị thiếu/ảnh chú thích/khác phiên bản, abstain cho phần không kết luận được; không dựng mâu thuẫn từ dữ liệu không đủ.

Phân biệt tên tác phẩm/tàu/sự kiện, biệt danh, danh hiệu, lời trích dẫn có quy nguồn với lời giật gân do người viết thêm. “Bài học đáng đời” là tên phim; “Xạ thủ” có thể là tên tàu. Dấu ngoặc kép không mặc định là emotional hay curiosity_gap. Quảng cáo cũng không mặc định là clickbait.

content là phần thân bài; lead là sapo/ngữ cảnh bổ sung, đọc chung khi content có mặt nhưng không coi lead là toàn văn thay thế. content_is_complete=null nghĩa là chưa biết độ đầy đủ, không phải đã kiểm chứng đầy đủ. Khi content không có hoặc không dùng được, chỉ xét các thủ pháp trên tiêu đề, không gán misleading. Không để sự khớp giữa tiêu đề và bài xóa bỏ curiosity_gap nếu tiêu đề thực sự dùng thủ pháp đó.

Đối chiếu chính xác chủ thể, lượng từ, thời gian, nguồn phát biểu và phạm vi:
- “Có pháo hoa mỗi tối”: body nêu 21h hàng ngày, thêm pháo hoa TẦM CAO thứ Bảy không bác bỏ pháo hoa mỗi tối. Không tự thêm chữ tầm cao vào tiêu đề để gán misleading.
- “Đạt gần 100%” khác “đạt 100%”; dự định khác đã làm. Kiểm tra mức độ sai lệch thay vì chỉ khớp chủ đề.
- Giá/điểm số/thu nhập cao không tự thân là exaggeration nếu bài hỗ trợ con số.

reason phải dựa trên chữ thực sự ở tiêu đề; nếu dùng body/lead thì ghi rõ nguồn và mâu thuẫn cụ thể. Không trích cụm ở body như thể nó nằm trong tiêu đề. Không đưa suy luận ẩn; trả đúng schema, không có nhãn ngoài taxonomy. confidence không phải xác suất đúng đã hiệu chỉnh.
