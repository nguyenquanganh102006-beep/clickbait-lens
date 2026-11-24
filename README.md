# Multimodal Clickbait Detection

Repository chung cho Clickbait Detection, Clickbait Type Classification và Title–Content–Image Consistency. Tên GitHub hiện tại vẫn là `clickbait-lens`; bộ khung nghiên cứu trên nhánh **`feat/shared-research-scaffold`**. Tool gán nhãn hiện có được giữ lại, xem [hướng dẫn labeler](docs/LABELER.md).

## Cấu trúc

```text
configs/                    # YAML từng task và smoke test
data/manifests/dataset_v1.0/ # Manifest, ID và split chính thức; không có toàn văn/ảnh
src/
  common/                   # Dataset, batch, loss, metrics, checkpoint, logging
  detection/                # Binary head và model detection
  type_classification/      # Multilabel head, đúng 5 Type
  consistency/              # TC/TI/IC, mỗi quan hệ 3 lớp
scripts/                    # Validate dataset và smoke test offline
experiments/                # Config đóng băng, run.json, events.jsonl; ngoài Git
checkpoints/                # .pt theo run_id; ngoài Git
results/                    # predictions.jsonl, metrics.json theo run_id; ngoài Git
tests/research/             # Không cần dữ liệu thật
```

`src/labeling`, `src/providers`, `src/ui` và các script annotation thuộc tool gán nhãn cũ; không là đầu vào của model nghiên cứu.

## Framework và cài đặt

Chốt **Python 3.12, PyTorch 2.6.0, NumPy, Pillow, YAML** cho core nghiên cứu. Encoder PhoBERT/XLM-R/CLIP/ViT do từng thành viên bổ sung; khóa model ID, revision và dependency trong config/requirements của thí nghiệm. Bộ khung chưa tải pretrained model hoặc chạy thí nghiệm chính thức.

```powershell
git clone --branch feat/shared-research-scaffold https://github.com/nguyenquanganh102006-beep/clickbait-lens.git multimodal-clickbait-detection
cd multimodal-clickbait-detection
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-research-dev.txt
python -m pytest tests/research -q
python scripts/smoke_research.py --synthetic
```

macOS/Linux: `python3.12 -m venv .venv` và `source .venv/bin/activate`. Với GPU, cài wheel PyTorch cùng phiên bản phù hợp máy theo [hướng dẫn PyTorch](https://pytorch.org/get-started/locally/), rồi cài requirements; ghi CUDA/driver trong run metadata. Khác phần cứng có thể cho kết quả khác dù cùng seed.

`requirements-research.txt` là core gọn. `requirements.txt` gồm cả core và labeler. `requirements-lock.txt` kế thừa khóa dependency của **labeler**, không phải lock của research. Các package nghiên cứu trực tiếp được pin; phiên bản thực tế mỗi run được ghi trong `run.json`.

## Dataset dùng chung

Dataset **v1.0**: **28.042** bài, **19.629 train / 4.206 validation / 4.207 test**, seed **20261124**, 27.729 leakage groups. **Không tự random split lại**. Mỗi bài, nhóm leakage và ảnh trùng byte chỉ nằm trong một tập.

Lấy snapshot `dataset_v1.0` và ảnh qua kênh dữ liệu của nhóm. Git chỉ giữ manifest/checksum và ID split. Trên workspace KHDL, từ checkout tại `KHDL/github/multimodal-clickbait-detection`:

```powershell
python scripts/validate_research_dataset.py --dataset-root "../../data_pr/dataset_v1.0"
python scripts/smoke_research.py --dataset-root "../../data_pr/dataset_v1.0"
```

Máy khác truyền đường dẫn snapshot thực tế; không hardcode ổ C hoặc copy ảnh vào Git.

```python
from src.common.dataset import ClickbaitDataset, make_dataloader
train = ClickbaitDataset("../../data_pr/dataset_v1.0", "train")
loader = make_dataloader(train, batch_size=16, num_workers=0)
```

Batch giữ text/PIL image dạng list để mỗi model dùng tokenizer/transform riêng; targets/masks đã stack. Chỉ fit preprocessing trên train, augmentation chỉ trên train, chọn threshold/model bằng validation. Không đưa nguồn/người gán nhãn/model AI/ghi chú annotation vào feature mặc định. Chi tiết: [hợp đồng tích hợp](docs/RESEARCH_CONTRACTS.md).

**Test chưa là human gold.** `ClickbaitDataset(..., "test")` mặc định từ chối nếu chưa cung cấp gold receipt. `allow_pseudo_test=True` chỉ để diagnostic. Hai người phải review độc lập và phân xử theo quy trình trong snapshot, rồi truyền `gold_csv`. Không random lại split sau khi biết nhãn gold. Manifest Git ghi trạng thái lúc phát hành; gold receipt là artifact riêng.

## Phát triển độc lập

- Detection trong `src/detection/`; Type trong `src/type_classification/`; Consistency trong `src/consistency/`.
- Hạ tầng chung ở `src/common/`; đổi API/schema cần nêu tác động trong PR và thêm kiểm tra.
- Model trả logits theo contract; dùng chung loss/metrics/exporter. Consistency độc lập với Detection/Type.
- Mỗi experiment có config, run ID, Git SHA, dataset/split/schema fingerprint và provenance nhãn. Xem [format artifact](docs/RESEARCH_CONTRACTS.md).
- Nhánh tích hợp hiện tại `feat/shared-research-scaffold`. Thành viên tạo `feat/<task>-<slug>` rồi PR về nhánh tích hợp. Xem [quy trình Git](CONTRIBUTING.md).

## Kiểm tra nhanh

```powershell
python -m pytest tests/research -q
python scripts/smoke_research.py --config configs/detection.yaml --synthetic
python scripts/smoke_research.py --config configs/type_classification.yaml --synthetic
python scripts/smoke_research.py --config configs/consistency.yaml --synthetic
```

Smoke runner huấn luyện MLP nhỏ trên đặc trưng độ dài/ảnh trung bình hoặc fixture tổng hợp để thử loss, checkpoint, dự đoán và logging. **Không phải baseline nghiên cứu**; mọi kết quả mang `official_benchmark=false`. Không sử dụng test split. CI chạy research tests + smoke offline trên Windows/Linux. Thay đổi tool labeler cần chạy thêm tests cũ trong môi trường dependency labeler.
