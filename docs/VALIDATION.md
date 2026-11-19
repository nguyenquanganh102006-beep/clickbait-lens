# Kiểm chứng bản chính 4.1.0

Ngày 19/11/2026, sau khi bỏ các luồng cũ:

- 53 test của mã hiện còn dùng đều qua, gồm evidence/reference, vận chuyển ảnh Gemini/OpenAI, đồng thuận từng target, Unknown, placeholder, mâu thuẫn số liệu, retry/quota, resume/checksum, ghi file khi Windows khóa tạm, cấu hình và điều hướng UI, đóng gói.
- Các test dành riêng cho luồng đã xóa được bỏ cùng mã tương ứng; con số này không so trực tiếp với 146 test trước dọn dẹp.
- Gói `KHDL_Auto_Labeler_v4.1.0.zip` tạo thành công; test kiểm tra manifest và loại trừ dữ liệu/secret/state.
- Chạy lại validator trên pilot API đã lưu: đủ 10 ID, title/content/image_path không đổi, checksum đúng, nhãn/mask khớp đồng thuận tái tính từ wire response. 8 bài đủ target; 2 bài cần review do placeholder và body tự mâu thuẫn. Có 77/90 target dùng được theo mask; human review vẫn pending.
- Dọn dẹp không gọi lại API. CSV đầu vào/kết quả/báo cáo/receipt của pilot cuối giữ trên máy trong `data/output/`, không đưa dữ liệu bài báo lên Git. Bộ sạch và ảnh ở `../data_pr/clean_final` giữ nguyên.

Đây là kiểm tra kỹ thuật và pilot có giới hạn, không phải accuracy/F1 hoặc gold label độc lập. Các lượt đánh giá dùng cùng model, có thể cùng sai. Báo cáo chi tiết riêng của pilot còn ở `data/output/quality_v41_review_20261119.md` trên máy thực hiện.
