# Luồng gán nhãn kiểm tra kỹ v4.1

## Dùng trong giao diện

1. Lưu kết nối Gemini/OpenAI ở bước 1. Model cần hỗ trợ ảnh và structured output.
2. Chọn **2. Gán nhãn đa phương thức**. Nhập CSV và thư mục gốc có thư mục `images/`.
3. Rà soát mù luôn bật. Mỗi bài dùng hai lượt đánh giá; chỉ thêm tối đa một lượt phân xử khi có vấn đề có thể giải quyết. Lỗi API tạm thời được retry có giới hạn; không retry quota đã cạn.
4. Chạy pilot nhỏ, tải CSV và kiểm tra hàng chờ review. Mô phỏng chỉ kiểm tra luồng, không tạo nhãn huấn luyện.

Ứng dụng chỉ cung cấp quy trình v4.1. Các luồng văn bản cũ và chế độ API một lượt đã được loại bỏ. Tiêu đề v3 trong nội dung rubric được giữ nguyên để không thay đổi prompt/hash của pilot đã kiểm chứng; đó là cùng rubric được sử dụng trong v4.1, không phải một chế độ chạy khác.

## Quy chuẩn và chất lượng

- Giữ taxonomy 5 Type: curiosity_gap, exaggeration, emotional_bait, question_based, misleading_presentation. Subtype của nhãn cuối chỉ là metadata.
- Lượt thứ hai không nhận nhãn/lý do lượt đầu. Lượt phân xử nhận các đề xuất được ghi rõ là không đáng tin và đối chiếu input gốc. Cả ba có thể cùng model: đây không phải ba annotator độc lập hoặc gold.
- Một target chỉ nhận nhãn khi có ít nhất hai đánh giá hợp lệ cùng kết luận. Các Type và ba quan hệ được xét riêng; không gộp confidence hoặc biến Unknown thành 0.
- Detection phải phù hợp với các Type đã đủ căn cứ. Một Type dương đủ để hỗ trợ Detection dương; không chấp nhận Detection âm nếu còn Type chưa được loại trừ.
- Schema API kiểm tra kỹ chỉ cho chọn `text_reference`, `content_paragraph_id` và mô tả ảnh. Không có trường quote tự do. Tool tự trích chính xác dữ liệu gốc, kiểm tra reference hợp lệ; việc chọn đúng ID vẫn chưa chứng minh đoạn đó hỗ trợ kết luận, nên cần audit ý nghĩa.
- Quan hệ TC/IC khẳng định phải có bằng chứng body; TI/IC phải có quan sát ảnh. Hình giữ chỗ/ảnh không dùng được theo kiểm tra đầu vào hoặc đồng thuận ảnh dẫn tới TI/IC Not Available; trạng thái này không phải lớp thứ tư để train.
- Ca bất đồng, evidence sai, ứng viên teaser chưa phân xử và mâu thuẫn số liệu nội bộ vẫn chuyển review. Các cờ heuristic không tự gán clickbait dương hay sửa dữ liệu.

## Mask theo từng nhãn

Không lọc toàn bộ một hàng chỉ theo `annotation_status`. Một hàng có thể cần review ảnh nhưng TC vẫn có nhãn dùng được. Đọc `label_mask_json`:

- `detection` tương ứng `clickbait_label`.
- Tên Type không tiền tố tương ứng cột `type_<tên>`.
- `consistency_tc`, `consistency_ti`, `consistency_ic` tương ứng cùng tên cột.
- Chỉ tính loss trên target mask=true. Mask=false phải bỏ qua, kể cả trong JSON tổng hợp có giá trị mô tả trạng thái. `Not Available` luôn mask=false. `Insufficient Evidence` là một lớp hợp lệ khi đủ input và có đồng thuận.
- Các cột chưa có căn cứ để trống/null. `clickbait_types=null` khi vector Type chưa hoàn chỉnh; dùng từng cột Type và mask để giữ phần đã có căn cứ. `[]` chỉ khi đã loại trừ đủ năm loại.

`human_review_status` vẫn pending. Đồng thuận AI và việc qua validator chưa phải nhãn con người xác nhận.

## Bằng chứng và provenance

- `annotation_passes_json`: phản hồi thực của từng lượt, gồm `wire_output`, output chuẩn hóa, model, usage, attempts và hash prompt.
- `annotation_raw_json` trong luồng này là bản tổng hợp đồng thuận, được đánh dấu `annotation_raw_is_consensus=True`; không giả là raw một model. Quan hệ chưa phân xử có label=null.
- `annotation_grounded_json`: bản tổng hợp với trích dẫn do tool lấy từ input, giữ null cho nhãn chưa phân xử.
- `annotation_quality_mode`, model yêu cầu/thực tế, prompt/schema version, input hash và result hash dùng cho audit/resume.
- `annotation_usage_incomplete=True` khi có retry/lỗi không có đầy đủ usage; tổng token chỉ cộng phần provider báo, không đoán phí.

## CLI, tiếp tục và an toàn dữ liệu

```powershell
.\.venv\Scripts\python.exe scripts/annotate.py --csv ..\data_pr\clean_final\sample_1000.csv --limit 10 --live --output data/output/pilot.csv
```

Mặc định dùng cấu hình kết nối đã lưu. `--key-stdin --provider gemini --model <model>` nhập key tạm bằng ô ẩn khi có terminal; không lưu key, không đưa key vào đối số command line. Không chia sẻ key qua log.

```powershell
# Tiếp tục các bài chưa hoàn tất, không gọi lại dòng đã có kết quả khớp
.\.venv\Scripts\python.exe scripts/annotate.py --csv ..\data_pr\clean_final\sample_1000.csv --limit 10 --live --resume --output data/output/pilot.csv
# Thử lại riêng các dòng có review lỗi API, không tự gọi lại các ca mơ hồ thực sự
.\.venv\Scripts\python.exe scripts/annotate.py --csv ..\data_pr\clean_final\sample_1000.csv --limit 10 --live --resume --retry-incomplete --output data/output/pilot.csv
```

Resume yêu cầu cùng model/prompt/schema/input/ảnh và CSV không bị sửa/hỏng. Cache khác phiên bản phải tạo output mới. Resume theo dòng hoàn tất, không bảo đảm phục hồi giữa các lượt của cùng một bài. CSV được ghi atomic, có retry file lock Windows và không được ghi đè input. Khi review chưa hoàn tất do API, CLI trả mã 2; khi lỗi chính làm dừng job trả mã 1; các dòng đã xong được giữ.

## Trước khi mở rộng

Chốt gold bởi người gán nhãn độc lập, đánh giá trên holdout chưa dùng để chỉnh prompt, kiểm tra cả nhãn dương/âm và ca đã qua validator. Không dùng confidence hoặc tỷ lệ đồng thuận làm accuracy. Nhóm trùng/gần trùng trước chia train/validation/test. Ảnh chỉ nối bằng ID, ảnh minh họa, bài thiếu nội dung và dữ liệu nguồn tự mâu thuẫn vẫn cần rà soát.

Nhịp gọi robust tuân theo requests_per_minute trong cấu hình (CLI key tạm mặc định 10/phút). Các lượt đánh giá mù hiện dùng cùng model trong các yêu cầu riêng; đồng thuận không loại bỏ thiên lệch chung của model.
