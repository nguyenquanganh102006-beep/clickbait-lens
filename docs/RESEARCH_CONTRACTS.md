# Hợp đồng tích hợp nghiên cứu v1.0

## Input và nhãn

`src.common.dataset.ClickbaitDataset` là lớp đọc dùng chung, kế thừa logic dataset_v1.0. Input: title, content, một ảnh RGB; lead/caption tùy chọn. ID chỉ để định danh. Preprocessing theo model ghi trong config, giữ nguyên split và nhãn gốc.

| Target key | Shape | Dtype | Ý nghĩa |
|---|---|---|---|
| detection | [B] | float32 | 0/1, BCE với logits |
| types | [B,5] | float32 | 5 nhãn độc lập, BCE với logits |
| consistency | [B,3] | int64 | TC/TI/IC: 0 Consistent, 1 Inconsistent, 2 Insufficient Evidence |

Type theo thứ tự: curiosity_gap, exaggeration, emotional_bait, question_based, misleading_presentation. Subtype không là head độc lập. Not Available mã -100, mask=0; không là lớp thứ tư. Insufficient Evidence vẫn là lớp thật. Loss tính trên nhãn hợp lệ; batch không có nhãn hợp lệ trả zero có gradient.

Encoder/fusion xuất vector [B,D] vào `MultiTaskHeads`. Logits trả dict detection [B], types [B,5], consistency [B,3,3]. Không sigmoid/softmax trong head; chuyển xác suất khi evaluation. Single-task chọn key tương ứng bằng config. Prediction Type và Detection có thể mâu thuẫn: exporter ghi detection_type_conflict, không tự sửa score từng task. Consistency dự đoán độc lập.

## Experiment và logging

Tên run: `<task>__<model-slug>__v1-0__s20261124__<UTC YYYYMMDDTHHMMSSZ>__<config-hash8>`.

Task: detection, type_classification, consistency, multitask. Model slug chữ thường/số/gạch nối. Tạo run mới, không ghi đè run cũ. UTC cho logging trên nhiều máy.

- `experiments/<run_id>/config.yaml`: config thực thi đóng băng.
- `run.json`: format version, seed, Git commit/dirty flag, versions, dataset version, SHA manifest/split/schema, nguồn nhãn pseudo/human_gold/synthetic.
- `events.jsonl`: timestamp ISO UTC, event, step, scalar loss/metrics. Không log toàn văn hoặc secrets.
- `results/<run_id>/metrics.json`: split, label_origin, số mẫu, metrics, official_benchmark.
- `results/<run_id>/predictions.jsonl`: một object/bài; đủ ID theo yêu cầu, không thiếu/trùng.
- `checkpoints/<run_id>/last.pt`: checkpoint gần nhất. Trainer chính thức có thể thêm best.pt, chọn bằng validation metric ghi trong config. Không chọn bằng test.

Config encoder phải bổ sung model ID/revision, tokenizer, max length/chunking, image resize/normalize, fusion, optimizer, loss weights, batch/precision/device, threshold và tiêu chí chọn checkpoint. Template smoke chỉ thử tích hợp, không quyết định thay nhóm những lựa chọn nghiên cứu này.

## Checkpoint .pt

Format version 1.0 gồm model_state_dict, optimizer_state_dict, step, config, metadata, RNG PyTorch/NumPy/Python/CUDA. Không pickle cả model class; lưu atomic qua .tmp rồi replace.

Metadata bắt buộc: run_id, dataset_version, dataset_manifest_sha256, split_assignments_sha256, label_schema_sha256, git_commit. Load bằng `torch.load(map_location="cpu", weights_only=True)`, đối chiếu split/schema trước khi load_state_dict(strict=True). Architecture/config phải khớp. Không load checkpoint không rõ nguồn. Tham khảo [serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html) và [torch.load](https://docs.pytorch.org/docs/stable/generated/torch.load.html) của PyTorch.

`restore_rng=True` phục hồi RNG. Trainer muốn resume chính xác giữa epoch cần thêm epoch/batch cursor, sampler/generator, scheduler và AMP scaler; skeleton chưa tuyên bố resume giữa batch.

## Prediction JSONL

Trường chung: format_version, article_id, run_id, dataset_version, split, task, label_origin, outputs.

```json
{
  "format_version": "1.0",
  "article_id": "example-id",
  "run_id": "example-run",
  "dataset_version": "1.0",
  "split": "validation",
  "task": "detection",
  "label_origin": "pseudo",
  "outputs": {"detection": {"probability": 0.72, "label": 1, "threshold": 0.5}}
}
```

Type trả map đủ 5 tên, mỗi tên có probability/label/threshold. Consistency trả tc/ti/ic; mỗi quan hệ có probabilities theo [Consistent, Inconsistent, Insufficient Evidence] và chuỗi label. Chỉ xuất head của task đang chạy; không điền nhãn giả cho task không hỗ trợ. Không NaN/Infinity.

Threshold Type có thể khác nhau, được chọn trên validation và lưu cùng run. Smoke dùng 0.5. Khi đổi threshold, truyền cùng detection_threshold/type_thresholds vào compute_metrics và prediction_record để metrics và file dự đoán khớp nhau.

Metric: Detection macro-F1 và precision/recall từng lớp; Type positive micro-F1/macro-F1, từng nhãn; Consistency macro-F1 đủ 3 lớp cho từng quan hệ. Accuracy chỉ bổ sung. Zero-support class vẫn trong macro. Không gộp Insufficient Evidence thành Inconsistent.

## Artifact và dữ liệu

Git giữ code/config, manifest/schema/split và báo cáo tóm tắt đã duyệt. Bài báo, ảnh, checkpoint, prediction từng bài và logs nằm ngoài Git, chia sẻ qua kho dữ liệu nhóm. Smoke không tự upload artifact.

Tệp dưới data/manifests giữ nguyên byte bằng .gitattributes để Windows/Linux có cùng checksum. Manifest mô tả snapshot ngoài Git; data.csv/guidelines trong manifest không có nghĩa đã upload vào repo. Validator đối chiếu local snapshot với bản chung, không tạo split mới.
