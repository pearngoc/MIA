# Config chung để các model so sánh được với nhau

Mọi thành viên (LSTM/TCN, PatchTST, iTransformer, MOMENT) phải theo file này. Số liệu chỉ so sánh được khi mọi thứ ở mục 1–3 giống nhau. Mục nào khác thì ghi vào `extra` của JSON kết quả và nêu trong báo cáo.

Quy ước: **[BẮT BUỘC]** = đã cố định bởi code dùng chung, không được đổi. **[THỐNG NHẤT]** = cả nhóm phải chọn cùng một giá trị, hiện chưa chốt. **[MOMENT]** = giá trị MOMENT đang dùng, các model khác điền cột của mình.

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

## 3. Huấn luyện [THỐNG NHẤT]

Generalization gap ảnh hưởng trực tiếp tới MIA (RQ1 hỏi "ở cùng gap"). Vì vậy phải báo cáo gap, và chỉ so sánh kiến trúc khi recipe công bằng.

| Mục | [MOMENT] | Model của bạn |
|---|---|---|
| Tập train | toàn bộ `train` của split (member), không thêm gì | |
| Epoch | 10 | |
| Batch | 16 | |
| Optimizer | AdamW | |
| LR | OneCycleLR, max_lr 1e-4 (lr đầu = max/25) | |
| Grad clip | 1.0 | |
| Loss | cls: CrossEntropy. fcst: MSE | |
| Precision | bf16 autocast (A100/L4), fp32 nếu GPU cũ | |
| Early stopping / chọn checkpoint | không. Lấy epoch cuối đã eval | |
| Epoch đánh giá MIA | 1, 2, 5, 10 (cho RQ2); số chính báo cáo ở epoch cuối | |

Quy tắc: **không chọn checkpoint hay hyperparameter dựa trên điểm MIA**. Chỉ được tune theo utility trên `val`. Không early stopping bằng test/non-member.

## 4. Attack và score [BẮT BUỘC cho định dạng, THỐNG NHẤT cho tham số]

Quy ước chung: **điểm càng cao càng có khả năng là member.**

| Attack | Score | Task |
|---|---|---|
| Loss | cls: `-CE`. fcst: `-MSE` (trung bình trên lead và bước) | cls, fcst |
| Confidence | `max softmax` | cls |
| Entropy | `-H(p)` | cls |
| LiRA online | log-likelihood ratio Gaussian. cls: φ = logit-scaled p đúng. fcst: φ = `-log(MSE)` | cả hai |
| LiRA offline | z-score so với phân phối OUT | cả hai |
| RMIA | cls: p(x) = p đúng. fcst: p(x) = `exp(-MSE/τ)`, τ = median MSE của shadow OUT. γ = 2.0, a = 0.3 (từ `.meta.json`), online và offline | cả hai |
| Temporal | fcst: DFT + trend của residual (Koren). cls: quỹ đạo logit qua epoch 1, 2, 5, 10 (đặc thù MOMENT, model khác nếu không lưu checkpoint trung gian thì ghi `n/a`) | |

Shadow model cho LiRA/RMIA:
- Dùng mask chung `splits/shadow_masks_record_K8_s42.npz` (K = 8, mỗi bản ghi nằm trong đúng 4 shadow). Tập tham chiếu `ref_ids` (5000 bản ghi) cũng lấy từ file này.
- Shadow huấn luyện **cùng recipe với target** (mục 3), chỉ khác tập train.
- Không dùng `ref_ids` hay shadow để chọn tham số attack. RMIA τ chọn trước (median OUT), bảng độ nhạy chỉ để báo cáo.
- Temporal là gray-box: attacker có một ít nhãn member/non-member. Dùng cross-fitting 5-fold, classifier cố định từ trước (HistGradientBoosting, max_iter 200, lr 0.05).

## 5. Metric [BẮT BUỘC]

- **Chính:** TPR @ 0,1% FPR và TPR @ 1% FPR, tính trên toàn bộ member vs non-member. Với FPR ≤ ngưỡng thì lấy TPR lớn nhất.
- **Bootstrap CI:** 1000 lần, 95%, resample member và non-member độc lập (percentile), seed 42.
- **AUC:** chỉ để đối chiếu.
- **ROC:** log–log, trục 1e-4 đến 1.
- Utility: cls báo `train_acc`, `test_acc`, `gap`. fcst báo `train_mse`, `test_mse`, `gap` (tên trường này do TV4 đặt, các TV khác cần dùng cùng tên nếu làm fcst).

## 6. Định dạng kết quả

Theo `../results/README.md` và `results/_sample.json`: `results/<model>/<task>/<attack>.json`, điểm lưu đủ cho cả member và non-member, tỉ lệ 0–1.

Mỗi file phải ghi vào `extra` để đối chiếu: `epochs`, `eval_epoch`, `batch`, `max_lr`, `backbone`, và với fcst thêm `lookback`, `horizon`. Nhờ đó có thể kiểm tra mọi model chạy cùng recipe.

**Tên file:** notebook MOMENT hiện ghi `<attack>_full.json`, còn quy ước repo là `<attack>.json`. Cần thống nhất một kiểu rồi đổi tên khi copy vào repo.

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
