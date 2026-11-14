# Config chung để các model so sánh được với nhau

Mọi thành viên (LSTM/TCN, PatchTST, iTransformer, MOMENT) phải theo file này. Số liệu chỉ so sánh được khi mục 1, 2, 4, 5 giống nhau và mục 3 tuân đúng quy trình chọn. Mục nào khác thì ghi vào `extra` của JSON kết quả và nêu trong báo cáo.

Quy ước: **[BẮT BUỘC]** = đã cố định bởi code dùng chung, không được đổi.

## 1. Dữ liệu và split [BẮT BUỘC]

| Mục | Giá trị | Nguồn |
|---|---|---|
| Dataset | PTB-XL, 100 Hz, chỉ bản ghi có đúng 1 superclass (5 lớp: CD, HYP, MI, NORM, STTC) | `make_split.py` |
| Split | `splits/ptbxl_record_s42.json`: `train` = member, `test` = non-member, `val` = 500 bản ghi lấy từ non-member (không dùng cho attack) | `make_split.py` |
| Level | chỉ `record` (stratified theo nhãn) | `make_split.py` |
| Seed | 42 | |
| Kích thước | 8136 member / 500 val / 7636 non-member | `.meta.json` |
| Toàn vẹn | In `sha256` của file split khi chạy và so với `.meta.json`. Khác thì dừng | |

Không tự tạo split riêng. Nếu thiếu file thì chạy `make_split.py --level record --seed 42`, đừng đổi seed hay `--n-val`.

**Lưu ý:** non-member chỉ ~7.6k, dưới mục tiêu 10k trong `../Research_v2.md`. TPR @ 0,1% FPR khi đó dựa trên khoảng 7 mẫu. Cả nhóm phải dùng chung số này và nêu hạn chế đó. Đừng ai tự tăng non-member riêng cho model của mình.

## 2. Tiền xử lý và input [BẮT BUỘC]

Một cấu hình duy nhất cho mọi model. Lấy theo MOMENT vì L = 512 là cố định theo kiến trúc của nó, nên các model khác (LSTM, TCN, PatchTST, iTransformer) phải theo, không tự đổi.

| Mục | Giá trị |
|---|---|
| Số lead | 12 |
| Độ dài input (lookback) L | 512 |
| Horizon fcst H | 96 |
| Resample | cả bản ghi 1000 điểm được nội suy về L + H = 608 điểm, rồi cắt: 512 đầu là input, 96 cuối là target. cls chỉ dùng 512 điểm đầu |
| Chuẩn hóa | z-score theo từng lead trên từng bản ghi (cả cửa sổ L + H), giống nhau cho member và non-member |
| Augmentation | không |

## 3. Huấn luyện [BẮT BUỘC cho quy trình, được chọn riêng epoch và lr]

Generalization gap ảnh hưởng trực tiếp tới MIA (RQ1 hỏi "ở cùng gap"). Recipe fine-tune của MOMENT (10 epoch, lr 1e-4) có thể làm model train từ đầu bị underfit, khiến gap nhỏ và MIA yếu giả tạo. Vì vậy điều cần giống nhau là **quy trình chọn**, không phải con số epoch và lr.

**Cố định chung:**

| Mục | Giá trị |
|---|---|
| Tập train | toàn bộ `train` của split (member), không thêm gì |
| Batch | 16 |
| Optimizer | AdamW |
| LR schedule | OneCycleLR |
| Grad clip | 1.0 |
| Loss | cls: CrossEntropy. fcst: MSE |
| Precision | bf16 autocast (A100/L4), fp32 nếu GPU cũ |
| Early stopping | không. Lấy epoch cuối |

**Chọn riêng cho từng model (epoch tổng, max_lr):**

| Mục | MOMENT (fine-tune) | Model khác (train từ đầu) |
|---|---|---|
| Epoch tổng | 10 | tự chọn |
| max_lr | 1e-4 (lr đầu = max/25) | tự chọn |

Quy tắc chọn:
- Chỉ tune theo **utility trên `val`** (cls: accuracy, fcst: MSE). Tuyệt đối không nhìn điểm MIA, không dùng non-member của `test`.
- Cùng ngân sách tune cho mọi model: tối đa 6 cấu hình (epoch, max_lr) mỗi model và mỗi task, ghi lại cả 6 vào `extra`.
- Phải báo cáo: utility, gap, epoch và max_lr cuối cùng. Khi so kiến trúc (RQ1) so ở gap tương đương và nêu rõ gap của từng model.
- Nếu model vẫn underfit sau khi tune thì ghi thẳng vào báo cáo, đừng kết luận "ít rò hơn" chỉ từ TPR.

**Epoch đánh giá MIA** (cho RQ2 và Temporal cls) lưu theo vị trí tương đối, làm tròn lên: 10%, 20%, 50% và 100% tổng epoch. Với 10 epoch là 1, 2, 5, 10 (đúng MOMENT). Ví dụ 50 epoch là 5, 10, 25, 50. Ghi danh sách epoch thật vào `extra.eval_epochs`. Số chính báo cáo ở epoch cuối.

**Shadow:** train bằng đúng recipe cuối cùng của target (epoch tổng, max_lr đã chọn), không tune lại.

Quy tắc chung: **không chọn checkpoint hay hyperparameter dựa trên điểm MIA.** Không early stopping bằng test/non-member.

## 4. Attack và score [BẮT BUỘC]

Một bộ attack và một bộ tham số duy nhất cho mọi model. Quy ước chung: **điểm càng cao càng có khả năng là member.**

Phải nộp đủ attack của task mình chạy: cls có tám (`loss`, `conf`, `entropy`, `lira`, `lira_offline`, `rmia`, `rmia_online`, `temporal`), fcst có sáu (bỏ `conf` và `entropy`). **Số chính trong báo cáo:** `loss`, `conf`, `entropy` (chỉ cls), `lira` (online), `rmia` (offline), `temporal`. **Chỉ để đối chiếu:** `lira_offline`, `rmia_online` (dùng chung 8 shadow nên không tốn thêm huấn luyện). Khi so model với nhau chỉ dùng số chính.

| Attack (tên trong `attack`) | Score | Task |
|---|---|---|
| Loss (`loss`) | cls: `-CE`. fcst: `-MSE` (trung bình trên lead và bước) | cls, fcst |
| Confidence (`conf`) | `max softmax` | cls |
| Entropy (`entropy`) | `-H(p)` | cls |
| LiRA online (`lira`) | log-likelihood ratio Gaussian, std chung toàn cục. cls: φ = `log(p/(1-p))` của lớp đúng. fcst: φ = `-log(MSE)` | cls, fcst |
| LiRA offline (`lira_offline`) | z-score của φ so với phân phối OUT | cls, fcst |
| RMIA offline (`rmia`) | cls: p(x) = softmax của lớp đúng. fcst: p(x) = `exp(-MSE/τ)`, τ = median MSE của shadow OUT (một giá trị chung, ×1.0). p̄ = ½((1+a)·p_OUT + (1−a)), γ = 2.0, a = 0.3 | cls, fcst |
| RMIA online (`rmia_online`) | như trên nhưng p̄ = ½(p_IN + p_OUT) | cls, fcst |
| Temporal (`temporal`) | fcst: DFT 8 dải + trend + drift + log MSE theo lead của residual (Koren). cls: quỹ đạo CE, entropy, margin qua bốn epoch đánh giá (mục 3) và độ dốc của chúng. Cả hai: gray-box, HistGradientBoosting (max_iter 200, lr 0.05), cross-fitting 5-fold | cls, fcst |

Mọi model phải lưu logit (cls) hoặc dự đoán (fcst) của member và non-member ở bốn epoch đánh giá của mục 3 để chạy Temporal cls và RQ2.

Shadow model cho LiRA/RMIA:
- Dùng mask chung `splits/shadow_masks_record_K8_s42.npz` (K = 8, mỗi bản ghi nằm trong đúng 4 shadow). Tập tham chiếu `ref_ids` (5000 bản ghi) cũng lấy từ file này.
- Shadow huấn luyện **cùng recipe với target** (mục 3), chỉ khác tập train.
- Không dùng `ref_ids` hay shadow để chọn tham số attack. τ, γ, a cố định như trên. Bảng độ nhạy τ (×0.5, ×1, ×2) chỉ để báo cáo, file kết quả luôn dùng ×1.
- Temporal là gray-box: attacker có một ít nhãn member/non-member, mỗi bản ghi được chấm bởi bộ phân loại chưa thấy nó.

## 5. Metric [BẮT BUỘC]

- **Chính:** TPR @ 0,1% FPR và TPR @ 1% FPR, tính trên toàn bộ member vs non-member. Với FPR ≤ ngưỡng thì lấy TPR lớn nhất.
- **Bootstrap CI:** 1000 lần, 95%, resample member và non-member độc lập (percentile), seed 42.
- **AUC:** chỉ để đối chiếu.
- **ROC:** log–log, trục 1e-4 đến 1.
- Utility: cls báo `train_acc`, `test_acc`, `gap`. fcst báo `train_mse`, `test_mse`, `gap` (tên trường này do TV4 đặt, các TV khác cần dùng cùng tên nếu làm fcst).

## 6. Định dạng kết quả

Theo `../results/README.md` và `results/_sample.json`: `results/<model>/<task>/<attack>.json`, điểm lưu đủ cho cả member và non-member, tỉ lệ 0–1.

Mỗi file phải ghi vào `extra` để đối chiếu: `epochs`, `eval_epoch`, `batch`, `max_lr`, `backbone`, và với fcst thêm `lookback`, `horizon`. Thêm `eval_epochs` và `tuning` (các cấu hình epoch, max_lr đã thử cùng utility trên `val`). Nhờ đó kiểm tra được mọi model theo cùng quy trình.

## 7. Checklist trước khi nộp kết quả

- [ ] `sha256` split trùng `.meta.json`
- [ ] Số member / non-member khớp mục 1
- [ ] Recipe mục 3 đã điền đủ và nằm trong `extra`
- [ ] Score đúng chiều (cao = member). Kiểm tra bằng AUC > 0,5 ở attack Loss
- [ ] Có bootstrap CI cho cả hai mức FPR
- [ ] Ghi rõ mọi khác biệt so với bảng mục 2 và 3
- [ ] Không có bước chọn hyperparameter nào nhìn vào điểm MIA

## 8. Chạy MOMENT (tham chiếu)

Thứ tự notebook (đều chạy trên Colab, GPU cho bước có chữ "GPU"):

| Bước | Notebook | Ghi vào |
|---|---|---|
| 1 | `../experiment/moment_colab.ipynb` (GPU): cls, Loss/Conf/Entropy | `ckpt/foundation_cls`, `../results/foundation/cls` |
| 2 | `../experiment/moment_lira_rmia_colab.ipynb` (GPU): cls, LiRA/RMIA, 8 shadow | cùng trên |
| 3 | `../experiment/moment_temporal_cls_colab.ipynb` (CPU): cls, Temporal | cùng trên |
| 4 | `../experiment/moment_fcst_colab.ipynb` (GPU): fcst, Loss | `ckpt/foundation_fcst`, `../results/foundation/fcst` |
| 5 | `../experiment/moment_temporal_colab.ipynb` (CPU): fcst, Temporal | cùng trên |
| 6 | `../experiment/moment_lira_rmia_fcst_colab.ipynb` (GPU): fcst, LiRA/RMIA, 8 shadow | cùng trên |

Đổi epoch, lr, batch, horizon hoặc split thì checkpoint cũ tự bị bỏ qua và notebook train lại.
