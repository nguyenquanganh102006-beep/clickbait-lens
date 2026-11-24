# Kiểm tra bộ khung — 2026-11-24

Môi trường đã chạy: Windows, Python 3.12.14, PyTorch 2.6.0+cpu, NumPy 2.3.5, Pillow 12.3.0, PyYAML 6.0.3. Không tải pretrained model hoặc gọi API gán nhãn.

- Research: 10 tests đạt, gồm shape 3 head, masked loss/backward, zero-mask metrics, threshold chung, macro-F1 cố định lớp, checkpoint roundtrip/guard, coverage JSONL, fingerprint/group/split, schema/config và smoke end-to-end.
- Labeler kế thừa từ nhánh gốc: 76 tests đạt trong môi trường dependency labeler. Không bao gồm thay đổi chưa commit của checkout labeler cũ bên ngoài thư mục này.
- Real-data smoke: 8 bài train và 8 bài validation của dataset_v1.0; loss hữu hạn, checkpoint nạp lại cho logits giống hệt, xuất đủ 8 dòng prediction. Không dùng test split.
- PyTorch DataLoader thật: batch 4 bài, Type shape [4,5], Consistency int64.
- Ba cấu hình single-task detection/type_classification/consistency đều có smoke tổng hợp để kiểm tra đường đi riêng. Mọi artifact smoke gắn official_benchmark=false.
- Black check đạt. Kiểm tra staged blob xác nhận 8 tệp metadata/split có checksum trùng bản phát hành, kể cả CSV CRLF gốc. Không stage ảnh, toàn văn, checkpoint hay run output.

Lệnh research: `python -m pytest tests/research -q` và `python scripts/smoke_research.py --synthetic`. Smoke thật truyền `--dataset-root` tới snapshot. Kết quả runtime nằm trong experiments/results/checkpoints ngoài Git.

CI research được cấu hình cho Windows/Linux. Các kiểm tra bên trên là kết quả local Windows; không suy rằng đã chạy local trên Linux. Gold test vẫn chờ hai con người xác minh độc lập, không là phần đã hoàn thành của bộ khung.
