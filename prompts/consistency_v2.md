# Title–Content Consistency — v2

Đánh giá sự hỗ trợ cho CÁC KHẲNG ĐỊNH CHÍNH của tiêu đề, không yêu cầu tiêu đề bao quát mọi nội dung. Không đánh giá mức câu khách ở tác vụ này: một tiêu đề giấu thông tin vẫn có thể consistent.

Đọc content là thân bài và lead là sapo bổ sung có nhãn nguồn riêng. lead không thay thế content khi thiếu thân bài. content_is_complete=null nghĩa là chưa xác minh độ đầy đủ; độ dài không chứng minh đã có toàn văn. Không coi một chi tiết chưa thấy trong phần trích là bằng chứng điều đó sai.

Kiểm tra ngắn gọn: chủ thể/hành động/đối tượng, con số và lượng từ, thời gian-hiện tại/tương lai, mức chắc chắn (được cho là, dự kiến, đã), nguồn lời nói và phạm vi. Không đưa suy luận ẩn.

- consistent: content cùng ngữ cảnh lead hỗ trợ các khẳng định chính; có thêm các mục khác không làm tiêu đề sai.
- partial: có hỗ trợ phần chính nhưng thiếu căn cứ cho một phần quan trọng, hoặc tiêu đề nâng mức khẳng định quá mức. Chỉ nêu một chênh lệch thực chất, không chỉ “bài còn nói thêm…”.
- inconsistent: có mâu thuẫn trực tiếp đáng kể hoặc nội dung đủ để đánh giá nhưng không hỗ trợ đáng kể điều tiêu đề hứa hẹn. Nếu chỉ thiếu một nhân vật trong bài giải thích vẫn hỗ trợ chủ đề chính, cân nhắc partial thay vì inconsistent.
- insufficient_data: không có body hoặc quá ít thông tin để đối chiếu.
- abstain: nghi body thu thập thiếu, chỉ còn chú thích ảnh/giới thiệu chung, bản xem trước trận nhưng tiêu đề đã cập nhật kết quả, dữ liệu các phần không đồng bộ hoặc không thể phân xử. Nêu rõ hạn chế thay vì kết luận bài gốc sai. Không dùng lead một mình để biến thiếu toàn văn thành success.

Ví dụ:
- “3 thương hiệu lớn nhất”: body nêu rõ vị trí 1,2,3 và nhắc thêm thương hiệu thứ 4 => không vì có mục thứ 4 mà gán partial.
- “Clip A nổi bật tuần qua”: bài tổng hợp có mục A và nhiều clip khác => các mục bổ sung không bác bỏ tiêu đề.
- “Hổ SẼ thoát cũi”: body nêu kế hoạch bán tự nhiên trong tương lai, hiện vẫn ở cũi => hiện trạng không bác bỏ khẳng định tương lai; đánh giá căn cứ kế hoạch, không xóa chữ SẼ.
- “Pháo hoa mỗi tối”: có lịch hàng ngày và lịch tầm cao thứ Bảy => không đánh đồng hai loại lịch.
- “Chính xác 100%”: body chỉ nói đang hướng tới gần 100% => kiểm tra hai chênh lệch mức độ và thời điểm, không cho consistent chỉ vì cùng nói vũ khí.
- Tiêu đề nói kết quả trận đấu nhưng content ghi NHẬN ĐỊNH TRƯỚC TRẬN => có dấu hiệu phiên bản thiếu/khác, ưu tiên abstain thay vì khẳng định báo đưa tin sai.

reason nêu khẳng định cần kiểm tra và sự hỗ trợ/chênh lệch cụ thể. evidence là trích đoạn ngắn có thật hoặc diễn giải sát nghĩa, chỉ rõ lead/content nếu cần; không đổi quan hệ nhân vật hoặc gán phát biểu sang người khác. Khi success phải có evidence. Nếu nguồn không đủ thì confidence cao không bù được: dùng trạng thái phù hợp, không mặc định 1,0. confidence không phải xác suất đã hiệu chỉnh. Trả đúng schema.
