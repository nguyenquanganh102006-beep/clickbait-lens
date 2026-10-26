# Cài và dùng cơ chế gán nhãn v2

## Cài đặt

Yêu cầu Python 3.11 trở lên và Git. Từ PowerShell:

```powershell
git clone --branch feat/auto-labeler-v0.2.0 --single-branch https://github.com/nguyenquanganh102006-beep/clickbait-lens.git
cd clickbait-lens
.\scripts\setup_windows.bat
.\scripts\run_windows.bat
```

Nếu đã có checkout trên branch này, cập nhật bằng `git pull --ff-only`, rồi chạy lại setup để cài các dependency cần thiết. Trên macOS/Linux, dùng `sh scripts/setup_unix.sh` và `sh scripts/run_unix.sh` thay cho hai lệnh Windows.

Mở http://127.0.0.1:8501. Dòng phiên bản prompt ở bước 3 phải hiện **v2**. Các file rubric được đóng gói trong `prompts/`; không cần dữ liệu hoặc cấu hình riêng của người phát triển để nạp cơ chế này.

## Thiết lập một lượt chạy

1. Nhập API key của bạn, chọn model và kiểm tra model bằng đúng các tác vụ/chế độ muốn sử dụng. Mô phỏng ngoại tuyến chỉ kiểm tra giao diện/luồng dữ liệu, không đánh giá độ đúng của nhãn.
2. Nạp file dữ liệu. Ánh xạ tiêu đề vào `title`, thân bài vào `content`, sapo vào `lead`. Không ánh xạ sapo thành toàn văn.
3. Chọn các tác vụ. Khi mới triển khai, chạy một tập nhỏ, đọc lý do/bằng chứng và hàng chờ rà soát trước khi tăng số lượng.
4. Tạo **job mới** để dùng v2. Job cũ tiếp tục theo phiên bản prompt đã lưu. Sau khi cập nhật, kiểm tra lại model vì trạng thái kiểm tra gắn với phiên bản prompt.
5. Xuất kết quả và kiểm tra `review_queue.csv`; không chỉ lọc confidence cao rồi mặc định là đúng.

## Quy tắc của ba tác vụ

### Detection

Chỉ xét tiêu đề. Nhãn 1 cần một thủ pháp câu khách rõ: giữ lại chi tiết trung tâm được hứa tiết lộ, phóng đại rõ hoặc kích động cảm xúc mạnh. Nhãn 0 dành cho cách đưa tin/phân tích/hỏi đáp thông thường. Ca ranh giới có thể trả `abstain`.

Không dùng một từ khóa làm luật: số tiền lớn, tên người nổi tiếng, câu hỏi, ngoặc kép, tiêu đề ngắn, lời cảnh báo hoặc ẩn dụ đều chưa đủ để gán 1. Phải phân biệt tên tác phẩm/sự kiện, danh hiệu và lời trích dẫn được quy nguồn với cách người viết đóng khung câu khách.

### Type

Cùng một yêu cầu API được yêu cầu đánh giá hai phần:

- Thủ pháp của tiêu đề: curiosity_gap, exaggeration, emotional, question_based.
- Đối chiếu nội dung: bổ sung misleading hoặc exaggeration khi có chênh lệch cụ thể với body/lead.

Chương trình hợp các nhãn hai phần. Việc thân bài trả lời điều tiêu đề giấu không xóa nhãn curiosity_gap đã nhận diện. Không ép Type có nhãn chỉ vì Detection trước đó cho 1. Thiếu body không được dùng để khẳng định misleading; trường hợp chưa phân xử được phải được giữ để rà soát.

### Consistency

Đối chiếu các khẳng định chính: chủ thể/hành động, con số, thời gian, mức độ chắc chắn và phạm vi. Bài có nội dung bổ sung không tự làm tiêu đề thành partial. Ví dụ tiêu đề về top 3 vẫn có thể khớp bài liệt kê thêm vị trí thứ 4; kế hoạch sẽ làm trong tương lai không mâu thuẫn chỉ vì hiện tại chưa làm.

Lead được gửi riêng như ngữ cảnh bổ sung khi có thân bài. Độ dài và sự tồn tại của content không chứng minh đã thu thập đủ toàn văn. Model phải đánh giá độ đầy đủ của chính thân bài, kèm một đoạn bằng chứng nguyên văn. Đoạn đó được kiểm tra có thật trong content trước khi chấp nhận kết quả khẳng định.

Nếu thân bài chỉ có định nghĩa chung, có dấu hiệu thiếu phần chính hoặc không đồng bộ với tiêu đề (chẳng hạn nhận định trước trận và kết quả sau trận), đầu ra chuyển về `abstain`, không giữ một nhãn khẳng định kèm theo. Thiếu body hoặc quá ngắn vẫn là `insufficient_data`; sapo không thay thế body.

## Hàng chờ rà soát

Khi có đủ kết quả các tác vụ trên cùng mẫu trong lượt xuất, hệ thống gắn cờ:

- `detection_positive_type_empty`: Detection=1 nhưng Type rỗng.
- `detection_negative_with_type_labels`: Detection=0 nhưng Type có nhãn thủ pháp/phóng đại. Nhãn misleading đơn lẻ có thể phụ thuộc body nên không tự động tạo cờ này.
- `misleading_but_consistent`: Type có misleading nhưng Consistency=consistent.
- Ngoài ra còn có abstain, insufficient_data, confidence thấp/thiếu và đầu vào bị cắt.

Các cờ là yêu cầu kiểm tra, không phải kết luận model chắc chắn sai. Hệ thống không tự ghi đè nhãn để làm các tác vụ khớp nhau.

## Phiên bản, cache và kiểm chứng

- Prompt mặc định là v2; schema đầu ra công khai vẫn v1. Các job v1 tiếp tục đọc rubric v1.
- Cache phân biệt phiên bản prompt; Type/Consistency v2 tính cả lead và cờ độ đầy đủ của content. Thay ngữ cảnh sẽ không tái sử dụng nhãn của ngữ cảnh cũ.
- Cả chế độ độc lập và gộp đều dùng các kiểm tra đầu vào/đầu ra. Phân tích hai phần không có nghĩa nhân đôi số yêu cầu API cho Type.
- Bộ kiểm tra có thể chạy không cần API key:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Các kiểm tra bảo đảm hợp đồng dữ liệu, cache, bằng chứng và hàng chờ rà soát; chúng không chứng minh accuracy của LLM. Thử API thực tế trong quá trình phát triển đã xử lý được nhiều ca lỗi cụ thể, nhưng vẫn còn ca ranh giới và biến thiên của model. Muốn đo precision/recall/F1 cần tập nhãn tham chiếu độc lập, gồm cả mẫu dương và âm. Không coi các pseudo-label hoặc confidence là gold label.

Mỗi người dùng tự cung cấp API key và dữ liệu. Repository không chứa dữ liệu làm việc, key đã nhớ, cơ sở dữ liệu job hoặc kết quả chạy riêng trên máy phát triển.
