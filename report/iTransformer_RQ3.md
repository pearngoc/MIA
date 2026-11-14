# Model iTransformer & Phân tích RQ3

**Họ tên:** Cao Anh Tuấn  
**MSHV:** 25C01025  
**Phụ trách:** iTransformer (+ RMIA, Temporal Feature)  
**Model:** iTransformer (Liu et al., ICLR 2024)  
**Phụ trách thêm:** RMIA (Zarifzadeh et al., 2024), Temporal Feature (Koren et al., 2025)  
**Câu hỏi:** RQ3 — Cùng đầu vào X, classification và forecasting khác nhau bao nhiêu về TPR @ low FPR?

---

## 1. Bối cảnh và đóng góp

iTransformer đảo chiều Transformer cổ điển: mỗi **biến (lead ECG)** là một token, chuỗi thời gian của lead được nhúng thành vector, rồi self-attention học quan hệ **giữa các biến**. Đây là kiến trúc time series hiện đại trong nhóm so sánh với LSTM/TCN (TV1), PatchTST (TV2) và foundation model (TV4).

Đóng góp liên quan iTransformer trong đề tài:

1. Huấn luyện iTransformer từ đầu trên PTB-XL cho **classification** (5 superclass) và **forecasting** (L = 512, H = 96).
2. Đánh giá MIA theo protocol nhóm: Loss / Confidence / Entropy, LiRA, RMIA, Temporal Feature; metric chính TPR @ 0,1% và 1% FPR kèm bootstrap CI 95%.
3. Trả lời **RQ3** trên cùng tập bản ghi X (member / non-member cố định).

---

## 2. Thiết lập thí nghiệm

### 2.1. Dữ liệu và split (dùng chung)

| Mục | Giá trị |
|---|---|
| Dataset | PTB-XL 100 Hz, 12 lead, chỉ bản ghi đúng 1 superclass |
| Split | `ptbxl_record_s42.json`, seed 42, record-level, stratified |
| Kích thước | 8136 member / 500 val / 7636 non-member |
| Input | Cls: L = 512. Fcst: L = 512, H = 96 (nội suy từ 1000 → 608 điểm) |
| Chuẩn hóa | Z-score theo lead trên từng bản ghi |

**Hạn chế:** non-member ~7,6k (&lt; mục tiêu 10k). TPR @ 0,1% FPR dựa trên khoảng 7–8 mẫu false positive — khoảng tin cậy rộng; cả nhóm dùng chung con số này.

### 2.2. Kiến trúc và huấn luyện

| Siêu tham số | Giá trị |
|---|---|
| Backbone | iTransformer, train từ đầu |
| d_model / n_heads / e_layers / d_ff | 128 / 8 / 3 / 256 |
| Batch / Optimizer / Schedule | 16 / AdamW / OneCycleLR |
| max_lr / Epochs | 1e-3 / 50 |
| Loss | Cls: CrossEntropy. Fcst: MSE |
| Eval MIA | Epoch 5, 10, 25, 50 (10%/20%/50%/100%); số chính ở epoch 50 |
| Shadow (LiRA/RMIA) | K = 8, mask dùng chung `shadow_masks_record_K8_s42.npz` |
| GPU | NVIDIA RTX 2060 6 GB (k3s node `tuan-cao`) |

Không chọn hyperparameter theo điểm MIA; chỉ theo utility trên `val` (cls: accuracy, fcst: MSE). Early stopping không dùng.

### 2.3. Utility (epoch cuối)

| Task | Metric train | Metric test (non-member) | Gap |
|---|---|---|---|
| Classification | Acc = 0,9999 | Acc = 0,5195 | **0,480** |
| Forecasting | MSE = 0,351 | MSE = 0,828 | **0,477** |

Hai task đều **overfit mạnh** với gap xấp xỉ nhau (~0,48). Điều này thuận lợi cho RQ3: so sánh rò rỉ membership khi generalization gap tương đương, trên cùng X.

---

## 3. Kết quả MIA — Classification

Số chính báo cáo: `loss`, `conf`, `entropy`, `lira` (online), `rmia` (offline), `temporal`.  
Bootstrap: 1000 lần, 95% CI, seed 42.

| Attack | TPR @ 0,1% FPR | 95% CI | TPR @ 1% FPR | 95% CI | AUC |
|---|---|---|---|---|---|
| Loss (−CE) | 0,000 | [0,000; 0,000] | 0,000 | [0,000; 0,000] | 0,897 |
| Confidence | 0,000 | [0,000; 0,000] | 0,000 | [0,000; 0,000] | 0,857 |
| Entropy | 0,006 | [0,001; 0,014] | 0,092 | [0,061; 0,113] | 0,895 |
| LiRA online | 0,008 | [0,003; 0,014] | 0,060 | [0,046; 0,077] | 0,584 |
| RMIA offline | 0,013 | [0,007; 0,023] | 0,096 | [0,077; 0,121] | 0,650 |
| **Temporal** | **0,077** | [0,026; 0,108] | **0,271** | [0,251; 0,307] | **0,945** |

**Ghi chú polarity:** Bản thô LiRA/RMIA cls có `mean(member) < mean(non-member)` (AUC &lt; 0,5), trái quy ước protocol *điểm cao = member*. Đã **đảo dấu score** (`negate`) rồi tính lại TPR/AUC/CI; chi tiết nằm trong `extra.score_polarity_fix` của từng JSON.

**Nhận xét cls:** Attack đơn giản (Loss/Conf) gần như không bắt được member ở FPR rất thấp dù AUC trung bình cao — điển hình khi đuôi phân phối non-member vẫn chồng lên member. Sau khi căn chiều score, LiRA/RMIA đạt AUC 0,58–0,65 và TPR@1% khoảng 6–10%. Temporal vẫn mạnh nhất rõ rệt.

Ablation Temporal cls (TPR @ 1% FPR): trajectory HGB 0,271 &gt; trajectory LR 0,174 &gt; final-epoch clf 0,090 &gt; loss-only ≈ 0 — động học huấn luyện đóng góp chính.

---

## 4. Kết quả MIA — Forecasting

| Attack | TPR @ 0,1% FPR | 95% CI | TPR @ 1% FPR | 95% CI | AUC |
|---|---|---|---|---|---|
| Loss (−MSE) | 0,030 | [0,022; 0,060] | 0,164 | [0,139; 0,196] | 0,911 |
| LiRA online | 0,011 | [0,007; 0,015] | 0,065 | [0,053; 0,079] | 0,806 |
| RMIA offline | 0,012 | [0,006; 0,025] | 0,094 | [0,067; 0,110] | 0,872 |
| **Temporal** | **0,140** | [0,094; 0,194] | **0,459** | [0,411; 0,512] | **0,963** |

**Nhận xét fcst:** Loss đã lộ rõ hơn cls ở low FPR. Temporal (DFT 8 dải + trend + drift + per-lead log MSE trên residual) lại là attack mạnh nhất, TPR @ 1% FPR ≈ 46%.

Ablation Temporal fcst (TPR @ 1% FPR): full features 0,459 &gt; LR trên cùng features 0,420 ≫ loss-only clf 0,160 ≈ raw −MSE 0,164 — đặc trưng phổ/trend của residual mang tín hiệu membership vượt xa scalar loss.

---

## 5. Phân tích RQ3: Classification vs Forecasting (cùng X)

### 5.1. Bảng so sánh trực tiếp

Cùng split, cùng seed, cùng recipe huấn luyện (50 epoch, max_lr 1e-3), gap gần bằng nhau.

| Attack (số chính) | Cls TPR@0,1% | Fcst TPR@0,1% | Cls TPR@1% | Fcst TPR@1% | Cls AUC | Fcst AUC |
|---|---|---|---|---|---|---|
| Loss | 0,000 | **0,030** | 0,000 | **0,164** | 0,897 | 0,911 |
| LiRA | 0,008 | **0,011** | 0,060 | **0,065** | 0,584 | **0,806** |
| RMIA | **0,013** | 0,012 | **0,096** | 0,094 | 0,650 | **0,872** |
| Temporal | 0,077 | **0,140** | 0,271 | **0,459** | 0,945 | **0,963** |

### 5.2. Kết luận RQ3

1. **Forecasting lộ membership mạnh hơn classification** trên iTransformer ở các attack chính **Loss** và **Temporal** (metric low-FPR), và có AUC LiRA/RMIA cao hơn rõ trên fcst.
2. Với **Temporal** (attack mạnh nhất mỗi task): fcst cao hơn cls khoảng **+0,063** ở 0,1% FPR và **+0,188** ở 1% FPR (tương đối: TPR@1% fcst ≈ 1,7× cls).
3. Với **Loss**: cls gần như không tấn công được ở low FPR; fcst đạt TPR@1% ≈ 16% — cùng X và gap tương đương, **mục tiêu dự báo tạo bề mặt tấn công dễ hơn mục tiêu phân loại**.
4. Gap generalization gần bằng nhau (~0,48) nên khác biệt không giải thích đơn thuần bằng “fcst overfit hơn”; nhiều khả năng đến từ **cấu trúc tín hiệu** (residual đa chiều / phổ thời gian) so với scalar CE trên nhãn rời rạc.
5. Sau khi căn polarity, LiRA/RMIA cls đạt AUC &gt; 0,5; TPR@1% của LiRA/RMIA gần nhau giữa cls và fcst — tín hiệu RQ3 **rõ nhất ở Temporal và Loss**, không nên chỉ dựa LiRA/RMIA.

### 5.3. Ý nghĩa thực tiễn

Nếu hệ thống ECG vừa phân loại bệnh vừa forecast đoạn tín hiệu tiếp theo từ cùng cửa sổ, **nhánh forecasting cần ưu tiên biện pháp chống MIA** (regularization, privacy accounting, hạn chế lộ residual/checkpoint) vì rò rỉ quan sát được cao hơn ở low FPR — đúng tinh thần đánh giá theo chuẩn privacy thực tế (Carlini et al., low-FPR).

---

## 6. RMIA và Temporal Feature (phần dùng chung)

Đã cài và chạy đủ trên iTransformer:

- **RMIA:** offline (số chính) và online; γ = 2,0, a = 0,3; τ fcst = median MSE của shadow OUT.
- **Temporal:**  
  - Cls: quỹ đạo CE / entropy / margin qua epoch 5–10–25–50 + độ dốc; HistGradientBoosting (max_iter 200, lr 0,05), 5-fold cross-fitting.  
  - Fcst: DFT 8 dải + trend + drift + per-lead log MSE (Koren-style).

Code nằm trong `experiment/itransformer/itransformer/` (`train_lira_rmia_*.py`, `temporal_*.py`) và có thể ghép vào attack lib của TV2 theo interface chung `results/<model>/<task>/<attack>.json`.

**Chưa làm:** DLinear (tuỳ chọn, bỏ qua vì đã đủ số cho RQ3).

---

## 7. Hạn chế

1. Non-member &lt; 10k → TPR @ 0,1% FPR ước lượng thô.  
2. RTX 2060 / fp32 (không bf16) — khác A100 của một số TV khác về precision, nhưng recipe và split giống nhau.  
3. Một cấu hình HP mặc định (chưa quét đủ 6 cấu hình trên val) — utility test cls ~52% (gần majority/NORM); kết luận “ít/nhiều rò” phải gắn với gap đã báo cáo.  
4. LiRA/RMIA cls từng lệch polarity ở bản thô; đã sửa bằng đảo dấu score (ghi trong `extra.score_polarity_fix`). Nên đối chiếu lại với attack lib TV2 khi ghép số liên-model.

---

## 8. Tóm tắt cho báo cáo nhóm

- iTransformer đã có đủ kết quả MIA cls + fcst theo schema `results/itransformer/`; LiRA/RMIA cls đã căn polarity theo protocol.  
- **RQ3 (iTransformer):** cùng X và gap ≈ 0,48, **forecasting rò rỉ membership nhiều hơn classification**, rõ nhất ở Temporal (TPR@1%: 0,459 vs 0,271) và Loss (0,164 vs 0,000).  
- Temporal là attack mạnh nhất cả hai task; RMIA/Temporal đã sẵn sàng đưa vào attack lib chung.

**File số liệu:** `results/itransformer/cls/*.json`, `results/itransformer/fcst/*.json`.

---

## Tài liệu tham chiếu

1. Liu et al. (2024). *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.* ICLR.  
2. Zarifzadeh et al. (2024). *Low-Cost High-Power Membership Inference Attacks.* ICML (RMIA).  
3. Carlini et al. (2022). *Membership Inference Attacks From First Principles.* IEEE S&P (LiRA, low-FPR).  
4. Koren et al. (2025). Temporal feature / time-series MIA (ACML 2024 / bản nhóm dùng).  
5. Watson et al. (2022). *On the Importance of Difficulty Calibration…* ICLR.  
6. Wagner et al. PTB-XL (PhysioNet).
