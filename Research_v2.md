# Membership Inference trên mô hình chuỗi thời gian hiện đại (bản đề xuất v2)

Rò rỉ thành viên ở kiến trúc mới và foundation model, đánh giá theo chuẩn low-FPR.
Bản này thay thế hướng TUH-EEG + LSTM/TCN trong `Research.md`, giữ nguyên file cũ để đối chiếu.

## 1. Thay đổi so với bản cũ

| Vấn đề cũ | Xử lý |
|---|---|
| TUH-EEG cần xin quyền | Dùng **PTB-XL** (PhysioNet, CC BY 4.0, mở). Phụ: UCI HAR / ETTh1 |
| Model cũ (LSTM/TCN) | Giữ làm baseline; thêm PatchTST, iTransformer, fine-tune MOMENT/Chronos-Bolt |
| 2.000 non-member không đủ cho 0,1% FPR | ≥10.000 non-member; báo cáo thêm TPR @ 1% FPR và bootstrap CI |
| Chỉ user-level, lẫn distribution shift | Tách **record-level** và **user-level** (RQ4) |
| Nhãn classification chưa định nghĩa | PTB-XL diagnostic superclass (5 lớp) |
| A5 nhận là "đề xuất" | Gọi đúng tên: Temporal Feature của Koren et al. (2025). Chạy ở chế độ gray-box, ghi rõ trong threat model |
| LiRA K=16 yếu | Thêm RMIA (Zarifzadeh et al., 2024) vì ổn khi ít shadow |

## 2. Dataset

- **Chính: PTB-XL.** ~21,8k bản ghi ECG, ~18,8k bệnh nhân, 10 giây, 100 Hz, có `patient_id`. Chọn 3 lead, cửa sổ L = 100. Forecast H = 20 trên cùng X.
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
- **G3.** Chưa tách record-level / user-level cho classification.

## 6. Câu hỏi nghiên cứu

- **RQ1.** Kiến trúc hiện đại có lộ membership nhiều hay ít hơn LSTM/TCN, ở cùng generalization gap?
- **RQ2.** Fine-tune foundation model có rò rỉ hơn train từ đầu không? Phụ thuộc thế nào vào số bước và full vs LoRA?
- **RQ3.** Cùng X, classification và forecasting khác nhau bao nhiêu về TPR @ low FPR?
- **RQ4.** Record-level và user-level khác nhau thế nào, và khoảng cách có do distribution shift giữa bệnh nhân không?

## 7. Phân công 4 thành viên

| TV | Vai trò | Sản phẩm |
|---|---|---|
| 1 | Data & Protocol | Pipeline PTB-XL/HAR, split record/user-level, protocol thống kê (≥10k non-member, bootstrap) |
| 2 | Models | Train LSTM/TCN, PatchTST, iTransformer; fine-tune foundation model; bảng utility và gap (RQ1, RQ2) |
| 3 | Attacks | Loss/Conf/Entropy, LiRA, RMIA, Temporal Feature; module TPR@FPR và ROC log–log |
| 4 | Literature & Writing | Cập nhật gap, trích dẫn, phân tích RQ3, báo cáo, slide |

Cả nhóm đọc paper nền ở tuần 1. TV1 và TV3 phối hợp cho RQ4; TV2 và TV3 phối hợp cho RQ1–RQ2.

## 8. Kế hoạch 5 tuần

| Tuần | Mục tiêu |
|---|---|
| W1 | Đọc paper, chốt gap và RQ; TV1 xong split PTB-XL; TV2 train LSTM-cls; TV3 skeleton đánh giá |
| W2 | Loss/Conf/Entropy trên baseline; LSTM-fcst; PatchTST, iTransformer; tìm tài liệu mới |
| W3 | LiRA/RMIA, Temporal Feature; fine-tune foundation model; RQ1, RQ3 |
| W4 | RQ2, RQ4; dataset phụ nếu còn GPU; viết chính |
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
