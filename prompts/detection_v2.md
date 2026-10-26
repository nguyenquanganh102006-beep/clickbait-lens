# Clickbait Detection — v2

Chỉ xét cách trình bày TIÊU ĐỀ, gán 0/1 theo ngưỡng thận trọng. Mọi tin tức đều muốn thu hút người đọc; hấp dẫn không đồng nghĩa clickbait.

- `1`: chỉ khi chỉ ra được một thủ pháp câu khách rõ: hứa tiết lộ nhưng giữ lại chi tiết trung tâm (động thái nào, bước ngoặt gì); phóng đại cực đoan; kích động cảm xúc mạnh vượt lối đưa tin thông thường. Khoảng trống thông tin phải nằm ở điều tiêu đề hứa giải đáp, không đơn thuần thiếu tên/ngày/địa điểm do giới hạn độ dài.
- `0`: thông báo sự việc, trích dẫn, phân tích, chân dung, hỏi đáp hoặc hướng dẫn thông thường mà chưa có thủ pháp đủ rõ.
- `abstain`, label=null: ca ranh giới thực sự không phân xử được. Không dùng confidence cao để che sự mơ hồ. Thiếu tiêu đề: insufficient_data.

Không dùng một từ khóa làm luật gán 1. Các yếu tố sau KHÔNG tự chúng là clickbait: số tiền lớn, thành tích hiếm, người nổi tiếng, tin tội phạm/bệnh tật, quảng cáo, dấu hỏi/ngoặc kép/ba chấm, tiêu đề ngắn, các từ nóng, hỏa tốc, nguy hiểm, hé lộ, bất ngờ. Cần căn cứ về chức năng câu khách trong toàn câu.

Nhận diện tên sách/phim/tàu/sự kiện, danh hiệu, biệt danh và lời nói được quy cho nhân vật trước khi đánh giá. Đừng đọc tên tác phẩm như từ giật gân của người viết. Một ẩn dụ thể thao/kinh tế hay một lời trích dẫn mạnh không tự động chứng minh exaggeration/emotional. Đồng thời trích dẫn vẫn có thể được dùng làm mồi câu khách nếu cách đóng khung rõ rệt; không miễn trừ mọi câu trích dẫn.

Ví dụ phản chứng:
- `0`: “Giáo viên Anh chán nghề” (tít chủ đề ngắn); “Ca sĩ chi 2 tỷ, lỗ 200 triệu khi làm show” (số liệu); “Hỏa tốc ứng phó mưa lớn” (công điện); “Nhật Bản chờ 'Xạ thủ' đáp xuống mặt trăng” (tên tàu); “'Địa chấn': Tập phóng sự mới của nhà báo…” (tên sách).
- `0`: “Hiệu trưởng nói gì về vụ bạo lực ở trường?” hoặc “Làm sao kiểm tra giao dịch an toàn?” nếu chỉ là câu hỏi thông tin, không có mồi câu khách bổ sung.
- Có cơ sở `1/curiosity_gap`: “Động thái kỳ lạ của quan chức trong vụ án…” không nói động thái gì; “Bước ngoặt không ai ngờ của người lái tàu…” giữ lại chính kết cục hứa tiết lộ.
- Không gán 1 cho mọi “Danh tính…”, “Nơi…”, “Sự thật…”: phân biệt tít tin thông thường với câu đóng khung mơ hồ để kéo nhấp.

reason ngắn, nêu chi tiết thật sự có trong tiêu đề và vì sao đủ/chưa đủ ngưỡng; không suy đoán sai sự thật từ một con số khi chưa đọc bài. Không đưa suy luận ẩn. Trả đúng schema; confidence là tự đánh giá, không phải xác suất đã hiệu chỉnh.
