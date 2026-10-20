# KHDL Auto Labeler — Kế hoạch thiết kế và triển khai

**Phiên bản:** 1.0  
**Trạng thái:** Kế hoạch kỹ thuật, chưa triển khai code  
**Mục đích:** Làm tài liệu đầu vào cho Codex xây dựng công cụ gán nhãn giả (LLM Pseudo-labeling) chạy trên máy cá nhân và đóng gói để chia sẻ trong nhóm KHDL.

## 1. Mục tiêu

Xây dựng ứng dụng local có giao diện web để thành viên nhóm có thể:

1. Nhập API key Google Gemini hoặc OpenAI GPT và chọn model khả dụng.
2. Đưa dữ liệu CSV, XLSX hoặc JSONL vào giao diện hoặc thư mục `data/input/`.
3. Ánh xạ tên cột tùy ý sang schema chuẩn.
4. Chọn một hoặc nhiều task: Clickbait Detection, Clickbait Type Classification, Title–Content Consistency.
5. Xem trước 5–10 ví dụ, kiểm tra prompt, chi phí ước lượng rồi chạy batch.
6. Tạm dừng, khôi phục và retry các mẫu lỗi mà không làm mất kết quả đã chạy.
7. Xem thống kê, lọc các trường hợp cần kiểm tra thủ công và xuất CSV/JSONL.
8. Chia sẻ repo/ZIP để người khác cài và sử dụng mà không sửa mã nguồn.

**Không nằm trong phạm vi V1:** triển khai cloud, đăng nhập nhiều người dùng, phân tán nhiều máy, tự động chứng minh độ đúng của nhãn giả, huấn luyện model, bắt buộc hỗ trợ local LLM. Chuẩn bị giao diện mở rộng Ollama/Qwen cho V2.

## 2. Bối cảnh dữ liệu KHDL

Đề tài: *Phát hiện và phân tích Clickbait trong tin tức trực tuyến bằng Machine Learning và NLP*.

Ba tác vụ:

| Task | Input tối thiểu | Output chuẩn |
|---|---|---|
| Clickbait Detection | `title` | `0` Non-clickbait / `1` Clickbait |
| Clickbait Type Classification | `title`; cần `content` khi xét `misleading` | Danh sách nhãn (multi-label) |
| Title–Content Consistency | `title` + toàn văn `content` | `consistent` / `partial` / `inconsistent` |

Dữ liệu nhóm hiện gồm dữ liệu crawl từ các thành viên và dữ liệu công khai như ViClickbait-2025 (3.414 mẫu theo tài liệu nhóm). Một số tập chỉ có `title` và `lead/sapo`; **không được tự động coi sapo là toàn văn**. Cần đặt tên task `Title–Lead Consistency` riêng nếu muốn nghiên cứu tiêu đề so với sapo.

Các tên cột có thể xuất hiện: `id`, `record_id`, `title`, `lead`, `sapo`, `lead_paragraph`, `content`, `body_text`, `body_preview`, `source`, `category`, `published_at`, `publish_datetime`, `url`, `label`, `final_label`.

## 3. Kiến trúc và công nghệ

- **Python:** 3.11+.
- **UI:** Streamlit, chạy trên `http://localhost:8501`.
- **LLM:** SDK chính thức Google GenAI và OpenAI; adapter interface chung.
- **Schema:** Pydantic + validation định danh nhãn.
- **Data:** pandas; đọc CSV UTF-8/UTF-8-SIG, XLSX nhiều sheet, JSONL.
- **State:** SQLite local lưu job, từng kết quả và trạng thái.
- **Cấu hình:** YAML, `.env.example`; API key nhập qua UI và chỉ lưu trong phiên hoặc qua cơ chế lưu cục bộ an toàn có đồng ý của người dùng.
- **Testing:** pytest, mock provider; không cần gọi API thật để chạy unit tests.
- **Phân phối:** GitHub/ZIP + script Windows và Unix. Không yêu cầu build EXE trong V1.

Luồng:

```text
CSV/XLSX/JSONL + Public Dataset
            |
            v
 Import -> Column Mapping -> Validation -> Preview
            |
            v
 Select Tasks + Provider + Model + Prompt Version
            |
            v
      Dry Run + Cost Estimate
            |
            v
 Job Queue -> LLM API -> Structured JSON -> Pydantic
            |                         |
            |                         +--> Error/Retry/Abstain
            v
    SQLite Checkpoints + Provenance
            |
            v
   Summary + Human Review + Export
```

## 4. Giao diện gồm 5 tab

### Tab 1 — Provider Settings

- Chọn `Gemini` hoặc `OpenAI`.
- Nhập API key dưới dạng password field; không hiển thị đầy đủ key, không log, không xuất theo job.
- Nút **Test Connection** kiểm tra key, quota/thông báo lỗi ở mức có thể.
- Hiển thị/chọn model từ danh sách model thực tế có thể dùng; có tùy chọn nhập model ID thủ công và validate.
- Thiết lập timeout, requests-per-minute, concurrency, max retries, giới hạn ngân sách dự kiến.
- Hiển thị giải thích rằng gửi bài báo đến API đồng nghĩa nội dung được chuyển cho nhà cung cấp; người dùng phải kiểm tra điều kiện sử dụng dữ liệu.

### Tab 2 — Dataset Import

- Upload CSV, JSONL, XLSX; hoặc chọn file tại `data/input/`.
- XLSX: chọn sheet; preview 10 dòng; báo số mẫu, cột trống, lỗi đọc, trùng ID/URL.
- Giao diện map trường gốc sang `record_id`, `title`, `content`, `lead`, `source`, `url`, `published_at`.
- Không cần xóa/đổi tên cột dữ liệu gốc; giữ nguyên khi export.
- Sinh `record_id` ổn định nếu thiếu (ưu tiên URL chuẩn hóa hoặc hash dữ liệu chuẩn hóa).
- Mẫu thiếu title sẽ bỏ qua có ghi lý do; mẫu thiếu content chỉ bỏ qua task Full Article Consistency, không bỏ qua các task khác.
- Xem trước số lượng bản ghi hợp lệ **theo từng task** trước khi gọi API.

### Tab 3 — Task & Prompt Configuration

- Chọn độc lập Detection, Type, Consistency hoặc cả ba.
- `Independent` mode: prompt riêng cho từng task; ưu tiên để benchmarking/debug.
- `Combined` mode: xử lý cùng một request; chỉ dùng sau khi đã kiểm tra chất lượng và chi phí.
- Chọn prompt version; cho xem trước nội dung prompt; không sửa prompt giữa một job đang chạy.
- Taxonomy đặt trong `config/taxonomy.yaml`; phiên bản taxonomy được lưu trong metadata job.
- Nếu `misleading` cần so sánh title-content mà không có content, trả `insufficient_data`, không tự đoán.

### Tab 4 — Job Runner

- Dry-run 5–10 dòng; hiển thị nhãn dự đoán và JSON lỗi (nếu có).
- Bắt đầu, Pause, Resume, Stop, Retry Failed.
- Progress thực theo `completed / total`, trạng thái từng task, số request thành công/thất bại, tokens thực dùng (nếu API trả về), chi phí ước tính.
- Chống request trùng thông qua cache key gồm input hash, provider, model, prompt version, taxonomy/schema version và task.
- Retry lỗi tạm thời 429/5xx/timeouts với exponential backoff + jitter; không retry vô hạn; lỗi xác thực 401/403 phải dừng hoặc yêu cầu sửa key.
- Ghi checkpoint khi từng request hoàn thành; việc pause/stop không làm hỏng dữ liệu.
- Nếu đạt ngưỡng ngân sách do người dùng đặt, tạm dừng job và hiển thị lý do.

### Tab 5 — Results & Export

- Bảng preview kết quả, lọc theo nhãn/task/source/confidence/trạng thái lỗi.
- Thống kê số dòng xử lý thành công, số dòng bị bỏ qua, số dòng cần review, phân bố label.
- So sánh với nhãn gốc (nếu có), **không ghi đè nhãn gốc**.
- Export:
  - `labeled_dataset.csv`: dữ liệu gốc + cột nhãn giả đã được flatten.
  - `predictions.jsonl`: kết quả cấu trúc đầy đủ với nguồn gốc từng prediction.
  - `review_queue.csv`: mẫu lỗi/không chắc chắn/thiếu dữ liệu/bất đồng (nếu áp dụng).
  - `summary.json`: thống kê job, cấu hình và kết quả, không chứa API key.
  - `errors.csv`: record ID, task, loại lỗi, số lần retry; không chứa bí mật.

## 5. Schema và quy tắc gán nhãn

Schema ví dụ (minh họa, không phải nhãn thật):

```json
{
  "record_id": "news_001",
  "labels": {
    "detection": {"status": "success", "label": 1, "confidence": 0.88, "reason": "Tạo tò mò"},
    "type": {"status": "success", "labels": ["curiosity_gap", "exaggeration"], "confidence": 0.82, "reason": "Tiêu đề giấu thông tin và phóng đại"},
    "consistency": {"status": "success", "label": "partial", "confidence": 0.76, "evidence": "Thông tin có liên quan nhưng bị phóng đại"}
  },
  "metadata": {
    "provider": "gemini",
    "model": "selected-model-id",
    "prompt_version": "v1",
    "taxonomy_version": "v1",
    "schema_version": "v1",
    "label_source": "llm",
    "human_verified": false,
    "review_status": "pending"
  }
}
```

- `status` thuộc `success`, `insufficient_data`, `not_applicable`, `error`, `abstain`.
- Không dùng `confidence` làm xác suất đúng đã hiệu chỉnh; đây chỉ là tự đánh giá từ model, cần kiểm định.
- **Detection:** nhãn 0/1, phân biệt các dấu hiệu clickbait với tiêu đề hấp dẫn thông thường.
- **Type:** cho phép multi-label. Taxonomy mặc định `curiosity_gap`, `exaggeration`, `emotional`, `question_based`, `misleading`. Cho phép cấu hình mở rộng `listicle`, `formatting`, `bait_and_switch`, v.v. sau khi thống nhất rubric.
- **Consistency:** `consistent`, `partial`, `inconsistent`. Phải đọc title + content, và có cơ chế abstain khi content thiếu/quá ngắn.
- `reason/evidence` phải ngắn, dựa trên văn bản đưa vào; không đòi chain-of-thought.
- Không ép một bài phải có tất cả nhãn; thiếu dữ liệu là trạng thái hợp lệ.
- Dữ liệu gốc, nhãn thật, nhãn giả và nhãn đã human review là **các trường tách biệt**.

## 6. LLM Adapter & Prompt

Giao diện logic:

```python
class LLMProvider:
    def validate_key(self): ...
    def list_models(self): ...
    def generate_structured(self, prompt, output_schema): ...
    def estimate_tokens(self, text): ...
```

- Viết adapter `GeminiProvider` và `OpenAIProvider`.
- Dùng structured output khi model hỗ trợ, parse JSON/Pydantic rồi validate enum và kiểu dữ liệu.
- Có fallback/nhắc lỗi rõ nếu model không hỗ trợ schema, không lặng lẽ coi text bất kỳ là nhãn hợp lệ.
- Prompts lưu tại `prompts/detection_v1.md`, `prompts/type_v1.md`, `prompts/consistency_v1.md`; có rubric, ví dụ khó và hướng dẫn abstain.
- Chống prompt injection: bài báo là **dữ liệu không tin cậy**, không được phép thay đổi hướng dẫn gán nhãn, gọi công cụ hay làm lộ key.
- Chuẩn hóa việc cắt/ngắt content dài, ghi nhận truncation, không đánh giá consistency như thể đã đọc đủ toàn văn.
- Ghi chính xác provider/model/prompt version theo từng prediction để tái lập thí nghiệm.

## 7. Batch, chi phí, an toàn

- Mặc định concurrency thấp và giới hạn RPM có thể cấu hình.
- Ước lượng tokens/chi phí **chỉ khi có bảng giá do người dùng cấu hình hoặc bảng giá cập nhật được xác minh**; phân biệt estimate với actual usage.
- Tự động tạm dừng khi đạt budget guardrail; người dùng xác nhận mới chạy tiếp.
- Kết quả API không hợp schema -> retry có giới hạn hoặc đưa vào review/error, không crash cả job.
- Tạo checkpoint theo từng `(job_id, record_id, task)` và đảm bảo export lặp lại không tạo duplicate.
- API key không lưu trong `.env` được chia sẻ; `.env.example` chỉ chứa placeholder; thêm `.env`, `data/input/*`, `data/output/*`, `state/*` vào `.gitignore` theo cách giữ thư mục bằng `.gitkeep`.
- Không đưa toàn văn bài báo hoặc API response chứa thông tin nhạy cảm vào log lỗi không cần thiết.
- Kiểm tra quyền sử dụng dữ liệu và chính sách của API trước khi gửi dữ liệu cho bên thứ ba.

## 8. Cấu trúc repo

```text
khdl-auto-labeler/
├── app.py
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
├── USER_GUIDE.md
├── DEVELOPER_GUIDE.md
├── src/
│   ├── providers/
│   │   ├── base.py
│   │   ├── gemini.py
│   │   └── openai.py
│   ├── labeling/
│   │   ├── engine.py
│   │   ├── prompts.py
│   │   ├── validation.py
│   │   └── schemas.py
│   ├── data/
│   │   ├── loader.py
│   │   ├── mapping.py
│   │   └── exporter.py
│   ├── jobs/
│   │   ├── runner.py
│   │   ├── checkpoint.py
│   │   └── storage.py
│   └── ui/
│       ├── provider.py
│       ├── dataset.py
│       ├── tasks.py
│       ├── jobs.py
│       └── results.py
├── config/
│   ├── settings.yaml
│   └── taxonomy.yaml
├── prompts/
│   ├── detection_v1.md
│   ├── type_v1.md
│   └── consistency_v1.md
├── data/
│   ├── input/.gitkeep
│   ├── output/.gitkeep
│   └── samples/
├── state/.gitkeep
├── scripts/
│   ├── setup_windows.bat
│   ├── run_windows.bat
│   └── run_unix.sh
└── tests/
    ├── test_loader.py
    ├── test_schema.py
    ├── test_provider.py
    └── test_resume.py
```

## 9. Lộ trình giao Codex (theo từng phase)

| Phase | Công việc | Điều kiện hoàn thành |
|---|---|---|
| 1. Scaffold | Repo, venv, dependencies, app skeleton, scripts | Khởi động Streamlit local từ bản clone sạch |
| 2. Import | CSV/XLSX/JSONL, preview, mapping, validation | Import được dữ liệu mẫu với cột khác tên, không mất data gốc |
| 3. Providers | Gemini/OpenAI, key test, model selection, structured JSON | Gán thử được 1 bài qua mỗi provider khi có key hợp lệ; mock test không cần key |
| 4. Labeling | 3 prompts, taxonomy, Pydantic, independent/combined, abstain | Output hợp schema và xử lý thiếu content đúng |
| 5. Jobs | SQLite, resume, retry, rate limit, budget, cache | Ngắt giữa chừng, chạy lại không mất hoặc nhân đôi nhãn |
| 6. Review/Export | Dashboard, filter, review queue, CSV/JSONL | Xuất lại được kết quả thống nhất, bảo toàn cột gốc |
| 7. QA/Packaging | Unit/integration tests, Windows/Unix scripts, 3 Markdown docs | Một thành viên khác cài bằng tài liệu mà không sửa mã |

**Quy tắc cho Codex:** Triển khai lần lượt, mỗi phase có test và demo. Không thay đổi schema/taxonomy ngầm. Giữ source code module hóa, không viết toàn bộ logic vào một file Streamlit. Không báo hoàn thành nếu chưa chạy test tương ứng.

## 10. Nội dung bắt buộc của các tài liệu bàn giao

### `README.md`
- Giới thiệu công cụ, 3 task, các định dạng input/output.
- Quick start; các screenshot giao diện sau khi hoàn thiện.
- Cấu trúc dự án và cách mở issue/báo lỗi.

### `USER_GUIDE.md` (tiếng Việt, dành cho thành viên)
1. Yêu cầu máy, cài Python 3.11+.
2. Giải nén repo; chạy `setup_windows.bat` (lần đầu).
3. Chạy `run_windows.bat` và truy cập địa chỉ local.
4. Cách lấy/nhập Gemini hoặc OpenAI API key, tránh chia sẻ key.
5. Cách đặt file trong `data/input/` hoặc upload; mapping cột.
6. Cách chọn task và mode; trường hợp nào cần full content.
7. Dry-run, giới hạn mẫu và ngân sách; Start/Pause/Resume.
8. Vị trí file output, ý nghĩa mỗi cột và trạng thái.
9. Cách xử lý lỗi API, rate limit, dữ liệu thiếu, JSON không hợp lệ.
10. FAQ và cách gửi log đã loại bỏ bí mật khi cần trợ giúp.

### `DEVELOPER_GUIDE.md`
- Cấu trúc module và luồng dữ liệu.
- Cách thêm LLM provider/model hoặc taxonomy.
- Quản lý prompt version, schema version, migration SQLite.
- Hướng dẫn mock tests, integration tests, release ZIP/GitHub.

## 11. Kịch bản kiểm thử tối thiểu

1. Import CSV UTF-8 tiếng Việt và Excel có nhiều sheet.
2. Map `body_text` sang `content`, vẫn giữ `body_text` ở export.
3. Chỉ có title: Detection/Type chạy; Consistency `insufficient_data`.
4. LLM trả label ngoài enum, JSON hỏng, timeouts và 429: xử lý có kiểm soát.
5. Mất mạng hoặc tắt ứng dụng giữa job: resume chỉ chạy phần chưa hoàn tất.
6. Run lại cùng job/config không sinh duplicate; đổi prompt version thì được phép chạy lại.
7. File dữ liệu có URL trùng, ID trùng, Unicode đặc biệt, title rỗng.
8. API key không xuất hiện trong logs, CSV, JSONL và ZIP phát hành.
9. So sánh output Independent/Combined trên bộ mẫu để kiểm tra độ khác biệt.
10. Export + re-import không làm sai record ID, timestamp, dấu tiếng Việt.

## 12. Checklist nghiệm thu

- [ ] Chạy trên Windows sau khi clone/giải nén sạch.
- [ ] Gemini và OpenAI hoạt động với key người dùng nhập.
- [ ] Chọn riêng hoặc phối hợp 3 task.
- [ ] Import CSV/XLSX/JSONL; column mapping; preview.
- [ ] Prompt, taxonomy, schema đều có version.
- [ ] Pydantic validation và các trạng thái lỗi/abstain.
- [ ] Dry-run; batch; pause/resume; retry; checkpoint SQLite.
- [ ] Rate-limit và ngưỡng ngân sách.
- [ ] Bảo toàn dữ liệu gốc và nhãn thật.
- [ ] Export labeled CSV, predictions JSONL, review queue, summary, errors.
- [ ] Test mock chạy không tốn API.
- [ ] Có README, USER_GUIDE, DEVELOPER_GUIDE tiếng Việt rõ ràng.
- [ ] Đóng gói GitHub/ZIP không chứa API key và dữ liệu riêng.

## 13. Chỉ dẫn triển khai tổng quát cho Codex

> Xây dựng repository `khdl-auto-labeler` theo đúng kế hoạch này. Bắt đầu từ Phase 1; triển khai lần lượt và chạy test ở mỗi phase. Ưu tiên Windows local, Streamlit, Python 3.11+, Gemini/OpenAI API, SQLite checkpoint, Pydantic structured outputs, input CSV/XLSX/JSONL. Không tự ý thay đổi định nghĩa 3 task. Phân biệt title-content và title-lead; giữ nguồn nhãn. Cuối cùng tạo README.md, USER_GUIDE.md và DEVELOPER_GUIDE.md, chạy nghiệm thu, đóng gói ZIP chia sẻ không chứa key/dữ liệu cá nhân. Khi chưa có API key, phải cung cấp mock provider để chạy demo/test offline.

---

**Tiêu chí thành công:** Thành viên trong nhóm có thể dùng công cụ trên máy riêng, nhập API key, nạp dữ liệu chưa nhãn, chạy pseudo-labeling đáng kiểm soát, tiếp tục sau lỗi và xuất kết quả có provenance. Pseudo-label **không thay thế** gold labels/human audit trong đánh giá mô hình.
