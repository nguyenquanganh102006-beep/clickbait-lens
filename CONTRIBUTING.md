# Quy trình Git của nhóm

Nhánh tích hợp hiện tại **develop**; main là bản ổn định. Không push trực tiếp/force-push main. Đưa lên main sau khi nhóm chấp nhận bản tích hợp. Tool labeler vẫn có nhánh feat/auto-labeler-v0.2.0.

## Nhận việc

Ba nhánh task dùng chung, đều tách từ develop:

- `feat/detection`: Clickbait Detection.
- `feat/type-classification`: Clickbait Type Classification.
- `feat/consistency`: Title–Content–Image Consistency.

Mỗi nhóm task tích hợp thay đổi vào nhánh task của mình rồi mở PR về develop.
Nhánh cá nhân có thể tách từ nhánh task và PR trở lại nhánh task trước khi tích hợp.
Không rebase hoặc force-push ba nhánh task dùng chung.

```bash
git fetch origin
git switch develop
git pull --ff-only origin develop
git switch -c feat/detection-phobert
```

Ví dụ: feat/type-xlm-r, feat/consistency-clip, fix/common-masked-loss. Làm task trong đúng thư mục; thay common/API/schema phải nêu ảnh hưởng trong PR. Không sửa split vì điểm model chưa tốt.

## Commit và PR

Commit dạng `feat(detection): add text encoder`, `fix(common): handle unavailable image`, `test(consistency): cover masks`, `docs: describe experiment`. Commit nhỏ, không gom output/code không liên quan.

Code research theo Black, 100 ký tự/dòng; giữ API có type/shape được mô tả trong docstring. Trước PR chạy Black trên các file đã sửa; CI chỉ format-check phần research, không ép format lại tool labeler cũ.

```bash
python -m pytest tests/research -q
python scripts/smoke_research.py --synthetic
git add src/detection configs tests/research
git commit -m "feat(detection): add text encoder"
git push -u origin feat/detection-phobert
```

PR về nhánh tích hợp: nêu behavior/model/config, kiểm tra, dataset version/preprocessing và tác động các task. Ít nhất một thành viên khác review, tác giả không tự duyệt. Merge khi CI đạt và review đã xử lý; mặc định squash merge. Không merge PR người khác hoặc đổi repo settings khi chưa thống nhất.

Cập nhật từ nhánh tích hợp trước khi merge. Chỉ rebase nhánh cá nhân chưa chia sẻ; nhánh nhiều người thì merge để giữ lịch sử. Không force-push nhánh chung. Nếu cần sửa lịch sử nhánh cá nhân đã push, trao đổi reviewer và dùng --force-with-lease.

## Điều kiện review

- Cùng dataset/split/schema fingerprint, không chọn model/threshold bằng test.
- API/output không thay đổi bất ngờ, có kiểm tra tương thích.
- Không commit API key, .env, bài báo, ảnh, checkpoint, đường dẫn máy cá nhân.
- Ghi rõ pseudo hay gold; không gọi pseudo-test là gold.
- Thay labeler chạy thêm tests labeler; research chạy tests/research và smoke.

CI research trên Linux/Windows. Quy tắc reviewer/cấm push main ở đây là **quy trình nhóm**; chưa cấu hình branch protection/ruleset trên GitHub để cưỡng chế. Sau khi thống nhất nhánh tích hợp chính thức, chủ repo có thể bật required PR, required CI và một approval trong Settings. Tài liệu không khẳng định các settings đó đã bật.
