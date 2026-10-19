# Membership Inference trên mô hình chuỗi thời gian hiện đại (bản đề xuất v2)

Rò rỉ thành viên ở kiến trúc mới và foundation model, đánh giá theo chuẩn low-FPR.
Bản này thay thế hướng TUH-EEG + LSTM/TCN trong `Research.md`, giữ nguyên file cũ để đối chiếu.

## 1. Thay đổi so với bản cũ

| Vấn đề cũ | Xử lý |
|---|---|
| TUH-EEG cần xin quyền | Dùng **PTB-XL** (PhysioNet, CC BY 4.0, mở). Phụ: UCI HAR / ETTh1 |
| Model cũ (LSTM/TCN) | Giữ làm baseline; thêm PatchTST, iTransformer, fine-tune MOMENT/Chronos-Bolt |
| 2.000 non-member không đủ cho 0,1% FPR | ≥10.000 non-member; báo cáo thêm TPR @ 1% FPR và bootstrap CI |
| Chỉ user-level, lẫn distribution shift | Dùng **record-level** split (chia ngẫu nhiên theo bản ghi, stratified) |
| Nhãn classification chưa định nghĩa | PTB-XL diagnostic superclass (5 lớp) |
| A5 nhận là "đề xuất" | Gọi đúng tên: Temporal Feature của Koren et al. (2025). Chạy ở chế độ gray-box, ghi rõ trong threat model |
| LiRA K=16 yếu | Thêm RMIA (Zarifzadeh et al., 2024) vì ổn khi ít shadow |

## 2. Dataset

- **Chính: PTB-XL.** ~21,8k bản ghi ECG, ~18,8k bệnh nhân, 10 giây, 100 Hz, có `patient_id`. Dùng 12 lead, cửa sổ input L = 512 (nội suy từ 1000 điểm). Forecast H = 96 trên cùng X, tức cả bản ghi nội suy về L + H = 608 điểm.
- **Phụ:** UCI HAR (classification, theo subject), ETTh1 (forecast, kiểm tra khái quát).
- Phải xác nhận lại điều khoản và định dạng trên PhysioNet khi tải.

## 3. Model

| Họ | Model | Ghi chú |
|---|---|---|
| Cũ (baseline) | LSTM, TCN | Đối chiếu Koren/Johansson |
| Hiện đại | PatchTST, iTransformer, (DLinear) | Train từ đầu |
| Foundation | MOMENT hoặc Chronos-Bolt | Fine-tune (full vs LoRA); chọn 1 nếu thiếu GPU |

## 4. Attack và metric

- Loss (Yeom), Confidence/Entropy (Salem, Song & Mittal) cho classification.
- LiRA offline (Carlini) và RMIA (Zarifzadeh).
- Temporal Feature (Koren): DFT + trend, gray-box.
- Metric chính: TPR @ 0,1% và 1% FPR, ROC log–log, kèm bootstrap CI. AUC chỉ đối chiếu.

## 5. Khoảng trống (cần kiểm chứng bằng tìm kiếm tài liệu 2025–2026)

- **G1.** Chưa có đánh giá MIA có hệ thống trên kiến trúc time series hiện đại và foundation model.
- **G2.** Chưa có so sánh classify vs forecast trên cùng dữ liệu theo chuẩn low-FPR.

## 6. Câu hỏi nghiên cứu

- **RQ1.** Kiến trúc hiện đại có lộ membership nhiều hay ít hơn LSTM/TCN, ở cùng generalization gap?
- **RQ2.** Fine-tune foundation model có rò rỉ hơn train từ đầu không? Phụ thuộc thế nào vào số bước và full vs LoRA?
- **RQ3.** Cùng X, classification và forecasting khác nhau bao nhiêu về TPR @ low FPR?

## 7. Phân công 4 thành viên

Chia theo model: mỗi người sở hữu trọn pipeline (train, tấn công, phân tích) cho một model, cả classification và forecasting.

| TV | Model chính | Paper của model | Phần dùng chung (phụ) |
|---|---|---|---|
| 1 | LSTM, TCN (baseline) | Koren 2025, Johansson 2025, Shokri 2017 | Data & protocol: pipeline PTB-XL, split record-level, ≥10k non-member, bootstrap |
| 2 | PatchTST | Nie 2023, Carlini 2022 (LiRA), Ye 2022 | Attack lib: Loss/Conf/Entropy, LiRA, module TPR@FPR và ROC log–log |
| 3 | iTransformer (+DLinear nếu kịp) | Liu 2024, Zarifzadeh 2024 (RMIA), Watson 2022 | RMIA + Temporal Feature (đưa vào attack lib của TV2) |
| 4 | Foundation: MOMENT hoặc Chronos-Bolt (full vs LoRA) | Goswami 2024, Ansari 2024, Duan 2024 | Literature & report: kiểm chứng G1–G2 với tài liệu 2025–2026, trích dẫn, ghép báo cáo và slide |

### Output của mỗi người

Schema chung: `results/<model>/<task>/<attack>.json` (xem `results/README.md`): scores member/non-member, TPR @ 0,1% và 1% FPR, AUC, bootstrap CI.

| TV | Output |
|---|---|
| 1 | Split cố định và `protocol.md` (seed, kích thước, bootstrap); bảng utility và gap LSTM/TCN; kết quả MIA LSTM/TCN |
| 2 | Attack lib có test, interface cố định; kết quả MIA PatchTST; ROC log–log gộp cả nhóm; phân tích **RQ1** (ghép số TV1, TV3) |
| 3 | Cài RMIA và Temporal Feature; kết quả MIA iTransformer; phân tích **RQ3** |
| 4 | Fine-tune foundation model, bảng số bước và full vs LoRA; kết quả MIA và phân tích **RQ2**; bảng gap G1–G2 đã kiểm chứng, bibliography; báo cáo và slide hoàn chỉnh |

Mỗi người viết một section "Model X" (2–3 trang) trong báo cáo, TV4 ghép. Mốc W1: TV1 giao split và protocol, TV2 giao interface attack; các TV còn lại train baseline của mình.

## 8. Kế hoạch 5 tuần

| Tuần | Mục tiêu |
|---|---|
| W1 | Đọc paper, chốt gap và RQ; TV1 xong split PTB-XL; TV2 train LSTM-cls; TV3 skeleton đánh giá |
| W2 | Loss/Conf/Entropy trên baseline; LSTM-fcst; PatchTST, iTransformer; tìm tài liệu mới |
| W3 | LiRA/RMIA, Temporal Feature; fine-tune foundation model; RQ1, RQ3 |
| W4 | RQ2; dataset phụ nếu còn GPU; viết chính |
| W5 | Freeze số, vẽ hình, slide, rehearsal |

Rủi ro: thiếu GPU thì chạy LiRA/RMIA chỉ trên 1 model mỗi họ, và bỏ dataset phụ.

## 9. Tài liệu (đã tải vào `papers/`)

Venue lấy từ trí nhớ và từ header PDF, cần xác nhận lại khi trích dẫn.

- Shokri et al. (2017), IEEE S&P: `shokri2017_MIA.pdf`
- Song & Mittal (2021), USENIX Security: `song2021_systematic.pdf`
- Carlini et al. (2022), IEEE S&P: `carlini2022_LiRA.pdf`
- Ye et al. (2022), ACM CCS: `ye2022_enhanced_MIA.pdf`
- Watson et al. (2022), ICLR: `watson2022_difficulty_calibration.pdf`
- Zarifzadeh et al. (2024), ICML: `zarifzadeh2024_RMIA.pdf`
- Duan et al. (2024), COLM: `duan2024_MIA_LLM.pdf`
- Koren et al. (2025), ACML 2024: `koren2025_ts_MIA.pdf`
- Johansson et al. (2025), arXiv: `johansson2025_ts_forecasting_MIA.pdf`
- Nie et al. (2023), PatchTST, ICLR: `nie2023_PatchTST.pdf`
- Liu et al. (2024), iTransformer, ICLR: `liu2024_iTransformer.pdf`
- Ansari et al. (2024), Chronos, TMLR: `ansari2024_Chronos.pdf`
- Goswami et al. (2024), MOMENT, ICML: `goswami2024_MOMENT.pdf`
