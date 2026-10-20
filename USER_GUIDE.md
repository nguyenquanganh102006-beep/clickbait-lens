# Hướng dẫn sử dụng KHDL Auto Labeler 0.2.0

## 1. Chuẩn bị và khởi chạy

Ứng dụng cần Python 3.11 trở lên và kết nối Internet nếu dùng Gemini/OpenAI.

Trên Windows, chạy `scripts\setup_windows.bat` một lần, sau đó chạy `scripts\run_windows.bat` và mở <http://localhost:8501>. Trên macOS/Linux, chạy `sh scripts/setup_unix.sh` rồi `sh scripts/run_unix.sh`.

Ứng dụng mở trên máy local. Nếu terminal đang mở từ trước chưa nhận đường dẫn Python mới, script Windows sẽ thử các thư mục cài Python theo user; nếu vẫn chưa thấy runtime, mở terminal mới. Không chia sẻ API key, tệp dữ liệu, thư mục `state/` hoặc `data/output/` lên nơi công khai.

## 2. Cấu hình nhà cung cấp

Chọn **1. Kết nối AI** ở thanh bên:

1. Chọn Gemini, OpenAI hoặc **Mô phỏng ngoại tuyến**.
2. Với Gemini/OpenAI, nhập API key vào ô mật khẩu rồi nhấn Enter hoặc bấm ra ngoài ô. Danh sách đã kiểm tra trên máy sẽ được khôi phục; nếu chưa có, ứng dụng tải danh sách từ API.
3. Bấm **Tự kiểm tra toàn bộ model và lưu kết quả**. Ứng dụng gọi thử từng model bằng bài mẫu có sẵn theo đúng tác vụ và chế độ đã chọn. Chỉ model trả kết quả hợp lệ và thành công cho mọi tác vụ mới xuất hiện ở ô chọn để chạy.
4. Chỉnh timeout, số request/phút, mức đồng thời và số lần retry nếu cần.
5. Chọn model trong danh sách đạt, bật/tắt **Nhớ kết nối AI trên máy cho những lần sau**, rồi bấm **Lưu cấu hình**.

Kết quả kiểm tra được nhớ theo key, tác vụ, chế độ chạy và phiên bản prompt/schema; lần mở sau không cần gọi lại API để kiểm tra. Đổi tác vụ ở bước 3 có thể cần kiểm tra model lại. Quota và quyền truy cập có thể thay đổi sau lần kiểm tra; dùng nút kiểm tra trực tiếp ở bước 4 khi cần.

Lượt quét chạy nền, cách các model 6 giây. Model gặp 429 được chờ ít nhất 60 giây hoặc thời gian API yêu cầu rồi thử lại một lần. Có thể chuyển bước để xem dữ liệu, nhưng đợi quét xong trước khi chạy job. Nếu đóng ứng dụng giữa chừng, mở lại bước 1 và bấm **Tiếp tục kiểm tra model còn lại**; model đã kiểm tra xong không bị gọi lại. Lỗi key/quyền truy cập sẽ dừng lượt quét.

API key được mã hóa bằng Windows DPAPI, gắn với tài khoản Windows hiện tại khi chọn nhớ kết nối. Bỏ chọn nhớ rồi lưu chỉ giữ key trong phiên và xóa bản nhớ trên máy. Nút xóa bản nhớ không xóa kết quả job hoặc kết nối đang mở. Trên macOS/Linux, dùng key trong phiên; lưu key được mã hóa hiện hỗ trợ Windows.

Mock chạy ngoại tuyến và chỉ tạo output minh họa hợp schema. Không dùng pseudo-label mock làm dữ liệu nghiên cứu.

## 3. Nạp dữ liệu và ánh xạ cột

Bước **2. Nạp dữ liệu** nhận CSV UTF-8/UTF-8-SIG, XLSX và JSONL. Có thể tải file qua giao diện hoặc đặt trong `data/input/`. Với XLSX, chọn sheet rồi bấm **Đọc tệp**. Tệp lớn có thể cần vài phút; khung trạng thái hiển thị việc đang làm, kết quả hoặc lỗi.

Ứng dụng gợi ý ánh xạ cho các trường:

| Trường chuẩn | Ví dụ tên cột nguồn |
|---|---|
| `record_id` | `id`, `article_id`, `news_id` |
| `title` | `headline`, `news_title`, `tieu_de` |
| `content` | `body_text`, `full_text`, `article_text` |
| `lead` | `sapo`, `summary`, `lead_paragraph` |
| `source` | `publisher`, `outlet`, `newspaper` |
| `url` | `link`, `article_url` |
| `published_at` | `date`, `publish_datetime` |

Kiểm tra preview, tiêu đề thiếu, ID trùng, URL trùng và số mẫu đủ dữ liệu theo từng task. Nếu thiếu ID, ứng dụng sinh ID ổn định từ URL hoặc nội dung chuẩn hóa; bản ghi trùng nội dung không có ID nguồn được thêm hậu tố để tránh trùng khóa. Cột gốc vẫn được giữ.

`lead`/`sapo` không thay cho `content`. Consistency yêu cầu toàn văn; nội dung quá ngắn hoặc quá dài sẽ được đánh dấu thiếu dữ liệu/abstain tùy trường hợp.

Sau khi kiểm tra ánh xạ, bấm **Lưu dữ liệu và ánh xạ trên máy** để lần mở sau tự khôi phục. Mỗi lần lưu thay bộ dữ liệu đã nhớ trước đó. **Xóa dữ liệu đã nhớ trên máy** chỉ xóa bản nhớ; tệp nguồn, dữ liệu đang mở và kết quả job không bị xóa. Nếu ID nguồn trùng, sửa cột ID hoặc bỏ ánh xạ ID để ứng dụng tạo ID duy nhất trước khi chạy.

## 4. Chọn tác vụ và xem prompt

Trong bước **3. Chọn tác vụ**, chọn một hoặc nhiều tác vụ và mode. Lựa chọn được tự lưu để khôi phục lần sau:

- **Độc lập:** gửi prompt riêng cho từng task; phù hợp để rà soát rubric và benchmark.
- **Gộp:** gửi một request cho các task đã chọn; kết quả các tác vụ được lưu cùng checkpoint.

Mở **Xem nội dung prompt và dữ liệu mẫu** để xem prompt sau khi ghép với bản ghi đầu tiên. Việc xem preview không gọi API.

Detection và Type cần tiêu đề. Type có thể phân loại dấu hiệu nhìn từ tiêu đề khi thiếu content nhưng không được gán `misleading`. Consistency cần tiêu đề và toàn văn đủ dài.

## 5. Chạy thử và chạy toàn bộ

Bước **4. Chạy gán nhãn** cho phép chạy thử và xử lý toàn bộ dữ liệu. Nút **Kiểm tra model đang chọn ngay bây giờ** gọi API bằng bài mẫu và đúng cấu hình tác vụ hiện tại. Nếu kiểm tra thất bại, model bị bỏ khỏi danh sách đạt và nút tạo job bị khóa cho đến khi kiểm tra thành công hoặc đổi model.

Chạy thử tạo job riêng với tối đa 10 bản ghi (mặc định 5) để xem output. Với Gemini/OpenAI, ứng dụng gửi dữ liệu mẫu đến API đã chọn. Với chế độ mô phỏng, không có yêu cầu nào được gửi ra ngoài.

**Tạo và chạy toàn bộ** xếp mọi bản ghi vào job, kể cả những bản ghi thiếu trường cần thiết; các task này được ghi trạng thái `insufficient_data` để kết quả không bị âm thầm bỏ qua.

Trong khi chạy có thể:

- **Tạm dừng:** job không nhận thêm request; request đang thực hiện có thể hoàn thành và được checkpoint.
- **Tiếp tục phần còn lại:** nạp lại task pending/running và dùng cache cho dự đoán đã có.
- **Dừng job:** không tạo request mới sau khi các request đang thực hiện kết thúc; phần còn lại có thể tiếp tục sau.
- **Retry các mẫu lỗi:** chỉ thử lại task trạng thái lỗi, giới hạn retry nội bộ vẫn áp dụng.

Job và kết quả được lưu local trong `state/jobs.sqlite3`. Tắt ứng dụng không xóa checkpoint. Nếu ứng dụng dừng bất thường, mở lại, khôi phục đúng provider/model, chọn job và tiếp tục. Tác vụ đang chạy dở sẽ về chờ; tác vụ đã lưu thành công được giữ. Nếu đổi model, tạo job mới. Khi thử lại tác vụ lỗi của job cũ, tác vụ/chế độ kiểm tra phải phù hợp với chính job đó.

Tiến độ và trạng thái cập nhật mỗi 3 giây. Khi hết quota, key sai hoặc model từ chối yêu cầu 400/404, job tạm dừng để tránh tạo hàng loạt lỗi cho phần dữ liệu chưa xử lý. Phần **Xem lỗi và cách xử lý** ghi nguyên nhân; **Hoạt động gần đây** ở thanh bên lưu các thao tác của phiên.

## 6. Rà soát và xuất dữ liệu

Bước **5. Xem & tải kết quả** cho phép lọc theo task, trạng thái, nguồn, confidence, nhãn, tiêu đề hoặc URL. Confidence là tự đánh giá của model, không phải xác suất đúng đã hiệu chỉnh. Nhãn gốc (ví dụ `label`, `final_label`) được hiển thị để đối chiếu, không bị ghi đè và ứng dụng không tự kết luận accuracy. Nút **Làm mới kết quả** lấy checkpoint mới nhất nếu job chưa hoàn tất.

Chọn **Tải toàn bộ kết quả (.zip)**, tải từng tệp, hoặc chọn **Lưu toàn bộ tệp vào data/output** để ghi vào thư mục job. Các bộ lọc chỉ thay đổi bảng xem; tệp tải xuống vẫn chứa toàn bộ kết quả của job:

- `labeled_dataset.csv`: dữ liệu nguồn và cột pseudo-label có tiền tố `pseudo_`.
- `predictions.jsonl`: input, output có cấu trúc, metadata phiên bản, usage và trạng thái.
- `review_queue.csv`: lỗi, confidence thấp, thiếu dữ liệu, abstain và trường hợp input bị cắt.
- `summary.json`: thống kê trạng thái/nhãn và cấu hình job đã loại bỏ thông tin nhạy cảm.
- `errors.csv`: record ID, task, loại lỗi, thông báo an toàn và số lần thử.

Các file CSV xuất UTF-8-SIG để thuận tiện mở bằng Excel. Tệp JSONL/JSON dùng UTF-8.

Nếu dữ liệu nguồn đã có cột `pseudo_*`, lần gán nhãn mới dùng tiền tố `pseudo2_*` (hoặc số tiếp theo) để bảo toàn các cột cũ. Kết quả mô phỏng có `label_source=mock` để phân biệt với API thật.

## 7. Trạng thái thường gặp

| Trạng thái | Ý nghĩa |
|---|---|
| `success` | Có output theo schema. |
| `insufficient_data` | Thiếu tiêu đề/toàn văn hoặc toàn văn quá ngắn. |
| `abstain` | Model/engine không kết luận đáng tin cậy, gồm trường hợp consistency bị cắt. |
| `not_applicable` | Tác vụ không được chọn trong mode gộp. |
| `error` | Lỗi provider, validation hoặc xử lý sau số lần thử giới hạn. |
| `pending` / `running` | Chưa bắt đầu hoặc đang chờ kết quả. |

Lỗi xác thực dừng job và yêu cầu kiểm tra key/quyền truy cập. Lỗi mạng/quota tạm thời được thử lại với backoff có giới hạn. Lỗi 429 có thời gian chờ từ API được tôn trọng; quota ngày hoặc quota bằng 0 cần xử lý tài khoản/project, không tự hết khi bấm chạy lại.

| Lỗi | Cách xử lý |
|---|---|
| 400 | Model không chấp nhận yêu cầu/schema hiện tại; chọn model đã qua kiểm tra đúng tác vụ. |
| 401/403 | Kiểm tra API key và quyền project. |
| 404 | Model có thể được liệt kê nhưng tài khoản không được chạy; chọn model đã gọi thử thành công. |
| 429 | Chờ thời gian API yêu cầu; kiểm tra quota/billing nếu vẫn bị chặn. |
| 503/504 | Dịch vụ lỗi tạm thời hoặc quá thời gian chờ; thử lại có giới hạn. |
| WinError 5 | Tệp lưu bị khóa hoặc thiếu quyền ghi. Ứng dụng thử lưu lại; nếu vẫn lỗi, giữ ứng dụng mở và bấm **Thử lưu lại kết quả hiện có** sau khi đóng phần mềm đang mở tệp. |

## 8. Bảo vệ dữ liệu

Khi dùng provider API, title/content đã ánh xạ rời máy để gửi đến nhà cung cấp. Chỉ xử lý dữ liệu mà nhóm được phép chia sẻ. Không gửi API key trong log hoặc tài liệu. Thư mục dữ liệu, kết quả và SQLite bị loại khỏi gói ZIP phát hành mặc định.
