### Bảng 1. So sánh mô hình trên validation (threshold từng nhãn, chọn trên OOF của train)

| Mã | Mô hình | Macro-F1 [95% CI] | Micro-F1 | Hamming ↓ | Subset acc | Macro AUPRC | OOF Macro-F1 (train) |
|---|---|---|---:|---:|---:|---:|---:|
| T1.0 | `t1_0_prior` | **0,211** [0,203–0,220] | 0,217 | 0,878 | 0,001 | 0,122 | 0,211 |
| T1.0-diag | `t1_0_source_only` *(chẩn đoán)* | **0,269** [0,259–0,279] | 0,295 | 0,500 | 0,228 | 0,172 | 0,268 |
| T1.1 | `t1_1_tfidf_title_br` | **0,466** [0,448–0,483] | 0,497 | 0,148 | 0,518 | 0,421 | 0,470 |
| T1.1 | `t1_1_tfidf_title_content_br` | **0,482** [0,464–0,501] | 0,517 | 0,142 | 0,537 | 0,442 | 0,485 |
| T1.2 | `t1_2_tfidf_chains` | **0,476** [0,459–0,495] | 0,515 | 0,142 | 0,551 | 0,438 | 0,479 |
| T1.2 | `t1_2_tfidf_label_powerset` | **0,467** [0,449–0,485] | 0,503 | 0,150 | 0,494 | 0,423 | 0,465 |
| T1.2 | `t1_2_tfidf_ovr_svm` | **0,483** [0,465–0,502] | 0,520 | 0,137 | 0,547 | 0,443 | 0,490 |
| T1.3 | `t1_3_hand_catboost` | **0,448** [0,432–0,463] | 0,465 | 0,170 | 0,506 | 0,393 | 0,462 |
| T1.3 | `t1_3_hand_lr` | **0,443** [0,426–0,458] | 0,449 | 0,192 | 0,486 | 0,370 | 0,454 |
| T1.4 | `t1_4_semantic_hand_catboost` | **0,487** [0,471–0,503] | 0,498 | 0,166 | 0,502 | 0,461 | 0,502 |
| T1.4 | `t1_4_semantic_lr` | **0,435** [0,418–0,451] | 0,462 | 0,172 | 0,472 | 0,404 | 0,457 |
| T1.5 | `t1_5_combined_lr` | **0,444** [0,428–0,461] | 0,481 | 0,163 | 0,490 | 0,416 | 0,460 |
| T1.5 | `t1_5_stacking` | **0,506** [0,489–0,524] | 0,535 | 0,135 | 0,556 | 0,471 | 0,516 |
| T1.5 | `t1_5_stacking_own_label` | **0,505** [0,488–0,523] | 0,530 | 0,136 | 0,556 | 0,468 | 0,519 |

### Bảng 2. F1 từng nhãn (validation)

| Mô hình | Curiosity | Exaggeration | Emotional | Question | Misleading |
|---|---:|---:|---:|---:|---:|
| `t1_0_prior` | 0,340 | 0,303 | 0,217 | 0,106 | 0,090 |
| `t1_1_tfidf_title_br` | 0,547 | 0,503 | 0,501 | 0,597 | 0,181 |
| `t1_1_tfidf_title_content_br` | 0,563 | 0,516 | 0,529 | 0,589 | 0,211 |
| `t1_2_tfidf_chains` | 0,568 | 0,512 | 0,517 | 0,583 | 0,202 |
| `t1_2_tfidf_label_powerset` | 0,552 | 0,517 | 0,519 | 0,556 | 0,193 |
| `t1_2_tfidf_ovr_svm` | 0,565 | 0,516 | 0,529 | 0,602 | 0,204 |
| `t1_3_hand_catboost` | 0,513 | 0,492 | 0,434 | 0,651 | 0,147 |
| `t1_3_hand_lr` | 0,502 | 0,489 | 0,422 | 0,650 | 0,150 |
| `t1_4_semantic_hand_catboost` | 0,545 | 0,537 | 0,536 | 0,649 | 0,168 |
| `t1_4_semantic_lr` | 0,526 | 0,505 | 0,489 | 0,521 | 0,135 |
| `t1_5_combined_lr` | 0,538 | 0,525 | 0,498 | 0,511 | 0,149 |
| `t1_5_stacking` | 0,569 | 0,551 | 0,544 | 0,646 | 0,220 |
| `t1_5_stacking_own_label` | 0,566 | 0,543 | 0,553 | 0,647 | 0,218 |

### Bảng 3. Ảnh hưởng của chiến lược threshold (Macro-F1 validation)

| Mô hình | 0,5 cố định | 1 ngưỡng chung | Ngưỡng từng nhãn |
|---|---:|---:|---:|
| `t1_0_prior` | 0,000 | 0,211 | 0,211 |
| `t1_1_tfidf_title_br` | 0,466 | 0,461 | 0,466 |
| `t1_1_tfidf_title_content_br` | 0,479 | 0,480 | 0,482 |
| `t1_2_tfidf_chains` | 0,472 | 0,472 | 0,476 |
| `t1_2_tfidf_label_powerset` | 0,411 | 0,451 | 0,467 |
| `t1_2_tfidf_ovr_svm` | 0,337 | 0,459 | 0,483 |
| `t1_3_hand_catboost` | 0,440 | 0,445 | 0,448 |
| `t1_3_hand_lr` | 0,430 | 0,436 | 0,443 |
| `t1_4_semantic_hand_catboost` | 0,463 | 0,468 | 0,487 |
| `t1_4_semantic_lr` | 0,429 | 0,432 | 0,435 |
| `t1_5_combined_lr` | 0,442 | 0,442 | 0,444 |
| `t1_5_stacking` | 0,465 | 0,498 | 0,506 |
| `t1_5_stacking_own_label` | 0,470 | 0,501 | 0,505 |

### Bảng 4. Kiểm định bootstrap ghép cặp (theo nhóm leakage, 2.000 lần)

| So sánh (A − B) | ΔMacro-F1 | 95% CI | p (A không tốt hơn) |
|---|---:|---|---:|
| `t1_5_stacking` − `t1_1_tfidf_title_br` | 0,040 | [0,028; 0,051] | 0,000 |
| `t1_1_tfidf_title_content_br` − `t1_1_tfidf_title_br` | 0,016 | [0,007; 0,024] | 0,001 |
| `t1_2_tfidf_chains` − `t1_1_tfidf_title_content_br` | -0,005 | [-0,014; 0,003] | 0,887 |
| `t1_5_combined_lr` − `t1_1_tfidf_title_content_br` | -0,038 | [-0,054; -0,022] | 1,000 |
| `t1_5_stacking` − `t1_5_combined_lr` | 0,062 | [0,048; 0,075] | 0,000 |
| `t1_5_stacking` − `t1_5_stacking_own_label` | 0,001 | [-0,006; 0,008] | 0,428 |
| `t1_3_hand_catboost` − `t1_0_prior` | 0,236 | [0,222; 0,251] | 0,000 |

### Bảng 4b. Ablation theo từng nhãn (ΔF1, bootstrap ghép cặp theo nhóm leakage)

| Ablation (A − B) | Curiosity | Exaggeration | Emotional | Question | Misleading |
|---|---:|---:|---:|---:|---:|
| Thêm nội dung (`t1_1_tfidf_title_content_br` − `t1_1_tfidf_title_br`) | +0,016* | +0,013 | +0,028* | −0,009 | +0,030* |
| Logit mọi nhãn vs cùng nhãn (`t1_5_stacking` − `t1_5_stacking_own_label`) | +0,003 | +0,008 | −0,010 | 0,000 | +0,002 |

\* khoảng tin cậy 95% không chứa 0.

### Bảng 5. Mô hình tốt nhất theo từng nhãn

| Nhãn | Mô hình tốt nhất | F1 | Mô hình chung tốt nhất (`t1_5_stacking`) |
|---|---|---:|---:|
| Curiosity | `t1_5_stacking` | 0,569 | 0,569 |
| Exaggeration | `t1_5_stacking` | 0,551 | 0,551 |
| Emotional | `t1_5_stacking_own_label` | 0,553 | 0,544 |
| Question | `t1_3_hand_catboost` | 0,651 | 0,646 |
| Misleading | `t1_5_stacking` | 0,220 | 0,220 |

Ràng buộc Detection–Type của `t1_5_stacking`: dự đoán có Type trên bài gold non-clickbait 0,231; không dự đoán Type nào trên bài gold clickbait 0,223; F1 của Detection suy ra từ Type 0,726.
