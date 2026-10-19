# MOMENT: Họ mô hình nền tảng chuỗi thời gian mã nguồn mở

*MOMENT: A Family of Open Time-series Foundation Models*

- **Tác giả:** Mononito Goswami, Konrad Szafer*, Arjun Choudhry*, Yifu Cai, Shuo Li, Artur Dubrawski (Auton Lab, Carnegie Mellon University; Shuo Li thuộc University of Pennsylvania). (*đóng góp ngang nhau)
- **Hội nghị/năm:** ICML 2024 (PMLR 235), Vienna; bản arXiv:2402.03885v3 (10/2024).
- **Mã/mô hình/dữ liệu:** `AutonLab/MOMENT-1-large` và `AutonLab/Timeseries-PILE` trên Hugging Face.

> *Bản tóm tắt chi tiết bằng tiếng Việt (do AI viết, không phải bản dịch nguyên văn).*

---

## Tóm tắt (Abstract)

Bài báo giới thiệu **MOMENT**, một họ mô hình nền tảng (*foundation model*) mã nguồn mở cho phân tích chuỗi thời gian đa nhiệm. Ba khó khăn chính khi tiền huấn luyện (*pre-training*) mô hình lớn cho chuỗi thời gian là: (1) thiếu kho dữ liệu công khai đủ lớn và thống nhất; (2) các chuỗi thời gian rất khác nhau về đặc tính nên khó huấn luyện chung nhiều dataset; (3) benchmark để đánh giá mô hình nền tảng trong điều kiện ít tài nguyên/ít nhãn còn sơ khai. Nhóm tác giả giải quyết bằng cách: gom **Time Series Pile** (tập hợp dữ liệu chuỗi thời gian công khai đa miền), xử lý có hệ thống các đặc thù của chuỗi thời gian để pre-train đa dataset, và xây dựng benchmark (dựa trên TimesNet) với các thiết lập *limited supervision*. Kết quả cho thấy MOMENT hiệu quả khi chỉ fine-tuning tối thiểu; ngoài ra bài báo nêu nhiều quan sát thực nghiệm về mô hình chuỗi thời gian cỡ lớn.

---

## 1. Giới thiệu (Introduction)

- Phân tích chuỗi thời gian (dự báo thời tiết, phát hiện nhịp tim bất thường từ ECG, phát hiện triển khai phần mềm bất thường...) có giá trị thực tiễn lớn nhưng thường cần nhiều chuyên môn miền, thời gian và thiết kế riêng cho từng tác vụ.
- Trong ngôn ngữ, thị giác, video, các mô hình tiền huấn luyện lớn làm tốt nhiều tác vụ với ít hoặc không cần giám sát. Bài báo muốn mang các khả năng này sang chuỗi thời gian, bằng họ MOMENT với ba tính chất: (1) là khối xây dựng cho nhiều tác vụ (forecasting, classification, anomaly detection, imputation); (2) dùng được ngay (*out-of-the-box*) với rất ít/không có ví dụ riêng cho tác vụ (zero-shot forecasting, few-shot classification); (3) tinh chỉnh được bằng dữ liệu đúng phân phối để cải thiện thêm.
- MOMENT là các transformer dung lượng lớn, tiền huấn luyện bằng bài toán **masked time series prediction** (dự đoán các đoạn bị che) trên dữ liệu đa miền.

**Ba đóng góp chính:**

- **C1 – Dữ liệu tiền huấn luyện:** thiếu kho dữ liệu công khai thống nhất là rào cản chính. Tác giả xây **Time Series Pile** từ các kho công khai thuộc nhiều lĩnh vực (y tế, kỹ thuật, tài chính...).
- **C2 – Pre-train đa dataset:** khác văn bản/ảnh, chuỗi thời gian khác nhau về tần số lấy mẫu, số kênh, độ dài, biên độ, và có thể thiếu giá trị. Vì vậy đa số phương pháp trước đó chỉ huấn luyện trên một dataset rồi chuyển sang dataset khác, kết quả khiêm tốn. MOMENT xử lý các khác biệt này để huấn luyện chung quy mô lớn.
- **C3 – Đánh giá:** mở rộng benchmark của Wu et al. (2023, TimesNet) theo nhiều chiều: 5 tác vụ (short/long-horizon forecasting, classification, anomaly detection, imputation); so với cả baseline deep learning lẫn thống kê; nhiều dataset đặc thù tác vụ hơn; nhiều metric; và **chỉ** trong thiết lập giám sát hạn chế (zero-shot imputation, *linear probing* cho forecasting, unsupervised representation learning cho classification).

Cuối cùng bài báo khảo sát các tính chất của mô hình: MOMENT có nhận biết tần số/xu hướng hay không, ảnh hưởng của khởi tạo, của kích thước mô hình, và của chuyển giao đa phương thức (*cross-modal transfer*).

---

## 2. Công trình liên quan (Related Work)

- **Transformer và patching cho chuỗi thời gian.** Độ phức tạp self-attention tăng bậc hai theo số token nên nhiều công trình đầu tiên thiết kế lại attention. PatchTST (Nie et al., 2023) chỉ ra rằng coi các đoạn con liên tiếp (*patch*) là token thay vì từng điểm thời gian vừa đơn giản vừa hiệu quả; MOMENT kế thừa ý tưởng này với patch không chồng lấn.
- **Masked representation learning.** Che-và-tái-tạo là tác vụ tự giám sát phổ biến ở ngôn ngữ/ảnh. Với chuỗi thời gian, trước đây chủ yếu dùng *contrastive learning*, vốn phụ thuộc augmentation mang tính chủ quan và phụ thuộc dữ liệu. Một số công trình che bằng số 0 rồi tái tạo; MOMENT dùng một **embedding che [MASK] có thể học** thay vì số 0. Dạng bài toán này phù hợp tự nhiên với forecasting và imputation (đều là tái tạo phần bị che).
- **Chuyển giao đa phương thức bằng LLM.** Lu et al. (2022) cho thấy transformer tiền huấn luyện trên văn bản giải được các tác vụ chuỗi ở modality khác; ORCA (Shen et al., 2023) theo quy trình *align-then-refine*. Nhiều công trình "tái lập trình" LLM cho chuỗi thời gian (GPT4TS, Time-LLM, LLMTime...) nhưng một số mô hình có hàng tỷ tham số, rất tốn tài nguyên. MOMENT bổ sung ba quan sát (Mục 4.3): transformer huấn luyện trên chuỗi thời gian cũng mô hình hóa được chuỗi ở modality khác; khởi tạo ngẫu nhiên cho loss tiền huấn luyện thấp hơn khởi tạo từ trọng số ngôn ngữ; mô hình pre-train trên chuỗi thời gian vượt các mô hình dựa trên LLM ở nhiều tác vụ.
- **Hai câu hỏi chưa có lời đáp:** (i) lợi/hại của pre-train đa dataset quy mô lớn (mọi mô hình trước đó pre-train/fine-tune trên từng dataset riêng); (ii) mô hình hóa chuỗi thời gian trong thiết lập giám sát hạn chế (zero-shot forecasting, few-shot classification). Bài báo trả lời cả hai: mô hình đủ lớn, tiền huấn luyện trên kho dữ liệu không nhãn lớn, có thể dự đoán khá chính xác khi giám sát hạn chế.

---

## 3. Phương pháp (Methodology)

Quy trình hai bước: gom dữ liệu thành Time Series Pile, rồi pre-train một transformer bằng masked time series prediction.

### 3.1. Time Series Pile

- Khác NLP/CV (The Pile, ImageNet-1K), dữ liệu chuỗi thời gian công khai nhỏ, rải rác và thường gắn với một tác vụ cụ thể. Tác giả gộp dữ liệu từ **4 kho công khai theo tác vụ**:
  - **Informer long-horizon forecasting** (Zhou et al., 2021): 9 dataset (ETT giờ/phút, Electricity, Traffic, Weather, ILI, Exchange-rate).
  - **Monash forecasting archive** (Godahewa et al., 2021): 58 dataset dự báo ngắn hạn, >100K chuỗi, nhiều miền và độ phân giải thời gian.
  - **UCR/UEA classification archive** (Dau et al., 2018): 159 dataset phân loại, 7 loại (Image Outline, Sensor, Motion Capture, Spectrograph, ECG, Electric Devices, Simulated).
  - **TSB-UAD anomaly benchmark** (Paparrizos et al., 2022b): 1980 chuỗi đơn biến có nhãn bất thường từ 18 dataset, cả tổng hợp lẫn thực.
- (Phụ lục E: Pile có 13 miền, khoảng 20 GB, ~13 triệu chuỗi, ~1,23 tỷ mốc thời gian, và vẫn đang được mở rộng.)
- **Chống rò rỉ dữ liệu (*data contamination*):** mỗi dataset được chia thành train/val/test rời nhau.
  - Dùng cách chia do tác giả dataset định sẵn; nếu không có thì chọn ngẫu nhiên **60% train / 10% val / 30% test**.
  - Chuỗi dài (long-horizon forecasting, anomaly detection) được cắt "theo chiều ngang" (cắt theo thời gian) thành các đoạn train/val/test; chuỗi ngắn (short-horizon forecasting, classification) thì mỗi chuỗi thuộc trọn một split.
  - Dùng cùng một random seed = 13 xuyên suốt từ pre-training đến đánh giá, và **chỉ dùng split train để pre-train**.

*[Hình 2: sơ đồ chia dữ liệu train/val/test của Time Series Pile; chỉ phần train được dùng để pre-train.]*

### 3.2. Kiến trúc mô hình

**Luồng xử lý** (xem *[Hình 3: tổng quan MOMENT: masking → patching → encoding → reconstruction]*):

1. Đầu vào là một chuỗi đơn biến $\mathcal{T}\in\mathbb{R}^{1\times T}$ cùng mặt nạ $M\in\{0,1\}^{1\times T}$ ($0$ = không quan sát, $1$ = quan sát), với $T=512$.
2. Áp dụng **reversible instance normalization (RevIN)** lên phần quan sát để chuẩn hóa/tái chuẩn hóa mỗi chuỗi.
3. Chia thành $N$ patch không chồng lấn, mỗi patch dài $P$.
4. Mỗi patch được ánh xạ thành embedding $D$ chiều bằng phép chiếu tuyến tính khả huấn luyện nếu mọi mốc thời gian trong patch đều quan sát; ngược lại, thay bằng embedding che học được $[\text{MASK}]\in\mathbb{R}^{1\times D}$.
5. $N$ embedding đi qua transformer encoder (giữ nguyên kích thước $1\times D$ mỗi token).
6. Mỗi embedding đầu ra được đưa qua một **reconstruction head** nhẹ để tái tạo cả patch bị che lẫn không bị che.

**Chi tiết encoder:** dùng các sửa đổi kiểu T5 (Raffel et al., 2020): bỏ additive bias trong Layer Norm, đặt Layer Norm trước residual skip connection, dùng **relative positional embedding** (Shaw et al., 2018). Ngoài ra, MOMENT còn **cộng thêm absolute sinusoidal positional embedding** (ban đầu thêm vào do nhầm lẫn, nhưng thí nghiệm sau cho thấy kết hợp cả hai làm dự đoán tốt hơn).

**Các quyết định thiết kế chính:**

- **Xử lý đặc tính đa dạng của chuỗi thời gian:**
  - *Độ dài khác nhau:* cố định đầu vào $T=512$; chuỗi dài hơn thì lấy mẫu con (sub-sample), ngắn hơn thì đệm 0 bên trái. (Phần lớn dataset phân loại ngắn hơn 512; cửa sổ nhìn lại 512 đủ cho long-horizon forecasting.)
  - *Nhiều kênh:* xử lý chuỗi đa biến bằng cách **tách từng kênh độc lập** dọc theo chiều batch (*channel independence*), chiến lược được chứng minh hiệu quả ở các nghiên cứu trước.
  - *Biên độ/phân phối khác nhau:* dùng RevIN để chuẩn hóa/đưa về tâm.
  - *Độ phân giải thời gian:* không mô hình hóa tường minh vì thông tin này thường không có ngoài các dataset forecasting.
  - *Patching* giảm chi phí bộ nhớ/tính toán theo bậc hai và tăng tuyến tính độ dài chuỗi đầu vào xử lý được.
- **Encoder cố ý đơn giản:** bám sát thiết kế transformer của ngôn ngữ để tận dụng các cài đặt hiệu quả (gradient checkpointing, mixed precision).
- **Prediction head nhẹ:** thay vì decoder lớn bằng encoder, dùng head nhẹ để việc điều chỉnh theo tác vụ chỉ cần huấn luyện ít tham số, giữ nguyên phần lớn đặc trưng mức cao của encoder.

### 3.3. Pre-training bằng Masked Time Series Modeling

- Che ngẫu nhiên đồng đều một phần nhỏ patch bằng $[\text{MASK}]$; đưa chuỗi bị làm hỏng qua encoder; dùng reconstruction head để tái tạo chuỗi gốc.
- **Mục tiêu:** cực tiểu sai số tái tạo trên các patch bị che:

$$\mathcal{L}_{\text{MSE}}=\frac{1}{|\mathcal{M}|}\sum_{i\in\mathcal{M}}\left\|\hat{\mathbf{x}}_i-\mathbf{x}_i\right\|_2^2$$

  Trong đó $\mathcal{M}$ là tập chỉ số các patch bị che, $\mathbf{x}_i$ là patch thật thứ $i$ và $\hat{\mathbf{x}}_i$ là patch do mô hình tái tạo (bài báo mô tả bằng lời "MSE trung bình trên các patch bị che"; công thức ở đây là cách viết lại của người tóm tắt).
- **Thiết lập pre-training:**

| Mục | Small | Base | Large |
|---|---|---|---|
| Số lớp transformer | 6 | 12 | 24 |
| Hidden dim $D$ | 512 | 768 | 1024 |
| Số attention head | 8 | 12 | 16 |
| Kích thước FFN | 2048 | 3072 | 4096 |
| Số tham số (xấp xỉ) | 40M | 125M | 385M |

  - Cỡ tương ứng encoder của T5-Small/Base/Large; **mọi trọng số khởi tạo ngẫu nhiên**.
  - Đầu vào $T=512$, $N=64$ patch, độ dài patch $P=8$; che **30%** patch.
  - Tối ưu: AdamW ($\lambda=0.05$, $\beta_1=0.9$, $\beta_2=0.999$), clip gradient 5.0, batch size 2048, lịch cosine learning rate từ $10^{-4}$ xuống $10^{-5}$, gradient checkpointing, mixed precision (float32 cho phép toán không ổn định như layer norm, bfloat16 cho phần còn lại), huấn luyện 2 epoch.

### 3.4. Tinh chỉnh cho các tác vụ hạ nguồn (Downstream tasks)

- **Forecasting (horizon $H$):** thay reconstruction head bằng **forecasting head**: làm phẳng $N$ embedding $D$ chiều thành vector $N\times D$ rồi chiếu tuyến tính ra chuỗi $H$ chiều.
- **Các tác vụ còn lại:** giữ reconstruction head.
- **Các chế độ sử dụng:** fine-tuning toàn bộ (end-to-end); **linear probing** ($\text{MOMENT}_{LP}$: đóng băng mọi tham số trừ head tái tạo/dự báo); **zero-shot** ($\text{MOMENT}_0$: giữ nguyên reconstruction head, áp dụng cho anomaly detection, unsupervised representation learning, imputation).

---

## 4. Thiết lập thí nghiệm và kết quả (Experimental Setup and Results)

### Thiết kế benchmark

Benchmark mở rộng từ TimesNet (Wu et al., 2023), tập trung vào **tài nguyên tính toán và giám sát hạn chế** (mô phỏng tình huống không thể/không đáng huấn luyện cả mạng sâu): đánh giá zero-shot khi có thể, nếu không thì linear probing vài epoch.

| Tác vụ | Giám sát | Dataset | Metric | Baseline chính | Thiết lập |
|---|---|---|---|---|---|
| Long-horizon forecasting | Linear probing | ETT-h1/h2/m1/m2, Electricity, Traffic, Weather, Exchange, ILI | MSE, MAE | Time-LLM, GPT4TS, TimesNet, PatchTST, FEDformer, DLinear, N-BEATS, Stationary, LightTS | Look-back $L=512$; horizon $H\in\{24,60\}$ (ILI), $\{96,720\}$ (còn lại) |
| Short-horizon forecasting | Zero-shot | Tập con M3 và M4 | sMAPE | GPT4TS, TimesNet, N-BEATS, AutoARIMA/Theta/ETS, Seasonal Naive, Naive, Random Walk | Thống kê fit từng chuỗi; deep learning train trên dataset nguồn, test trên dataset đích cùng độ phân giải |
| Classification | Unsupervised representation learning | UCR (91 dataset, chuỗi <512 điểm) | Accuracy | GPT4TS, TimesNet, TS2Vec, T-Loss, TNC, TS-TCC, TST, CNN, Encoder, FCN, MCNN, MLP, ResNet, t-LeNet, TWIESN, DTW | Mọi mô hình khác MOMENT huấn luyện trên từng dataset; chất lượng biểu diễn đo bằng accuracy của SVM huấn luyện trên biểu diễn |
| Anomaly detection | Linear probing, zero-shot | UCR Anomaly Archive | Adjusted best F1, VUS-ROC | GPT4TS, TimesNet, Anomaly Transformer, DGHL, k-NN | Tái tạo theo cửa sổ 512; MSE giữa quan sát và dự đoán làm tiêu chí bất thường |
| Imputation | Linear probing, zero-shot | ETT-h1/h2/m1/m2, Electricity, Weather | MSE, MAE | GPT4TS, TimesNet, Linear, Naive, Cubic Spline, Nearest Neighbors | Che ngẫu nhiên các đoạn liên tiếp dài 8; tỷ lệ che {12.5%, 25%, 37.5%, 50%} |

**Điểm khác với TimesNet:**

- *Dataset:* forecasting và imputation dùng cùng dataset như TimesNet; còn classification và anomaly detection dùng tập lớn hơn, chọn có hệ thống từ UCR classification archive (91 dataset có chuỗi <512 điểm) và UCR anomaly archive (ưu tiên độ phủ miền/nguồn dữ liệu). UCR anomaly archive được đề xuất như bản cải thiện so với SMD, SMAP (dùng trong TimesNet).
- *Metric:* với anomaly detection, TimesNet/GPT4TS dùng F1 thông thường (bỏ qua tính tuần tự); bài báo dùng **adjusted best F1** và **VUS-ROC** (Paparrizos et al., 2022a).
- *Baseline:* so với cả phương pháp thống kê/non-transformer. Phát hiện quan trọng: ARIMA (short-horizon), N-BEATS (long-horizon), k-nearest neighbors (anomaly detection) thắng nhiều mô hình deep/transformer.
- *Hyper-parameter:* không tinh chỉnh siêu tham số. Mặc định fine-tune MOMENT-Large với batch 64, lịch one-cycle, peak learning rate trong khoảng $5\times10^{-5}$ đến $10^{-3}$; baseline dùng cấu hình khuyến nghị từ bài gốc.

**Ba câu hỏi nghiên cứu:**

- **RQ1 (hiệu quả):** MOMENT có hiệu quả cho nhiều tác vụ trong giám sát hạn chế không?
- **RQ2 (diễn giải):** MOMENT học được gì? Có nắm bắt tần số, xu hướng, biên độ không?
- **RQ3 (tính chất):** ảnh hưởng của kích thước mô hình; có chuyển giao đa phương thức như LLM không?

### 4.1. MOMENT giải được nhiều tác vụ trong giám sát hạn chế

**Long-horizon forecasting** (Bảng 2, MSE/MAE; trích một số dòng tiêu biểu, horizon ngắn nhất và dài nhất):

| Dataset (horizon) | $\text{MOMENT}_{LP}$ | GPT4TS | PatchTST | DLinear | TimesNet | N-BEATS |
|---|---|---|---|---|---|---|
| Weather (96) | 0.154 / 0.209 | 0.162 / 0.212 | **0.149 / 0.198** | 0.176 / 0.237 | 0.172 / 0.220 | 0.152 / 0.210 |
| ETTh1 (96) | 0.387 / 0.410 | **0.376 / 0.397** | 0.370 / 0.399 | 0.375 / 0.399 | 0.384 / 0.402 | 0.399 / 0.428 |
| ETTh1 (720) | 0.454 / 0.472 | 0.477 / 0.456 | **0.447 / 0.466** | 0.472 / 0.490 | 0.521 / 0.500 | 0.608 / 0.573 |
| ETTm2 (720) | 0.363 / 0.387 | 0.378 / 0.401 | **0.362 / 0.385** | 0.397 / 0.421 | 0.408 / 0.403 | 0.395 / 0.419 |
| ILI (24) | 2.728 / 1.114 | 2.063 / 0.881 | **1.319 / 0.754** | 2.215 / 1.081 | 2.317 / 0.934 | 4.539 / 1.528 |
| ECL (96) | 0.136 / 0.233 | 0.139 / 0.238 | **0.129 / 0.222** | 0.140 / 0.237 | 0.168 / 0.272 | 0.131 / 0.228 |
| Traffic (720) | 0.450 / 0.310 | 0.450 / 0.312 | **0.432 / 0.286** | 0.466 / 0.315 | 0.640 / 0.350 | 0.508 / 0.335 |

  - Linear probing MOMENT đạt gần state-of-the-art ở hầu hết dataset/horizon, chỉ đứng sau **PatchTST** (thường có MSE thấp nhất). Ở nhiều thiết lập, các mô hình dựa trên LLM (Time-LLM, GPT4TS) tệ hơn MOMENT. **N-BEATS** vượt nhiều phương pháp gần đây, nhấn mạnh nên so sánh cả ngoài transformer.
  - Điểm yếu rõ nhất: ILI (dataset nhỏ), nơi MOMENT kém xa PatchTST.

**Zero-shot short-horizon forecasting** (Bảng 3, sMAPE, thấp hơn tốt hơn; M4 = huấn luyện nguồn là M4, FR = FRED):

| Tập | $\text{MOMENT}_{LP}$ (M4 / FR) | GPT4TS (M4 / FR) | TimesNet (M4 / FR) | N-BEATS (M4 / FR) | ARIMA | Theta | ETS | Seasonal Naive |
|---|---|---|---|---|---|---|---|---|
| M3 Yearly | 16.74 / 16.97 | 18.39 / 17.40 | 27.48 / 16.21 | 16.82 / 15.92 | 17.90 | 16.70 | 16.47 | 17.54 |
| M3 Quarterly | 10.09 / 10.62 | 10.18 / 10.29 | 14.41 / 12.68 | 11.26 / 11.30 | 10.18 | 9.19 | 8.99 | 11.02 |
| M3 Monthly | 16.04 / 16.90 | 15.21 / 16.37 | 15.58 / 16.23 | 15.63 / 16.37 | 15.95 | 14.96 | 14.41 | 17.74 |
| M4 Yearly | - / 14.84 | - / 14.80 | - / 14.40 | - / 14.18 | 16.19 | 14.04 | 14.06 | 16.33 |
| M4 Quarterly | - / 12.02 | - / 11.77 | - / 13.21 | - / 12.25 | 10.86 | 10.21 | 10.24 | 12.55 |
| M4 Monthly | - / 15.80 | - / 15.36 | - / 15.67 | - / 15.24 | 13.68 | 13.19 | 13.58 | 16.00 |

  - Đây là tác vụ còn nhiều dư địa cải thiện nhất: các phương pháp thống kê (Theta, ETS) thắng mô hình sâu; chỉ ở một số dataset MOMENT đạt sMAPE thấp hơn ARIMA.
  - (Phụ lục) Chỉ báo cáo trên 40% M3/M4 chưa thấy khi pre-train và các tần số có nhiều dữ liệu (yearly, quarterly, monthly); daily/hourly/weekly quá ít dữ liệu nên không có zero-shot tốt cho mô hình sâu nào. Tác giả cho rằng ensembling góp phần vào kết quả zero-shot tốt của N-BEATS gốc.

**Classification** (Bảng 4, accuracy trên 91 dataset UCR; $\text{MOMENT}_0$ = biểu diễn không fine-tune + SVM):

| Chỉ số | $\text{MOMENT}_0$ | TS2Vec | T-Loss | TS-TCC | ResNet | FCN | DTW | TimesNet | GPT4TS |
|---|---|---|---|---|---|---|---|---|---|
| Mean accuracy | 0.794 | **0.851** | 0.833 | 0.793 | 0.825 | 0.809 | 0.764 | 0.572 | 0.566 |
| Median accuracy | 0.815 | **0.871** | 0.849 | 0.802 | 0.852 | 0.837 | 0.768 | 0.565 | 0.583 |
| Mean rank | 7.2 | **3.5** | 5.3 | 6.5 | 4.4 | 5.6 | 9.1 | 13.3 | 13.3 |

  - Không cần fine-tune theo từng dataset, SVM trên biểu diễn của MOMENT tốt hơn gần hết phương pháp chuyên cho phân loại chuỗi thời gian (chỉ 4 phương pháp tốt hơn về mean accuracy). GPT4TS và TimesNet kém dù được huấn luyện có nhãn trên từng dataset. Kết quả trên dataset đa biến UEA nằm ở phụ lục.
  - *[Hình 5: PCA/t-SNE biểu diễn MOMENT trên 3 dataset UCR lớn nhất cho thấy các lớp tách thành cụm riêng dù không fine-tune.]*

**Anomaly detection** (Bảng 7, trung bình trên 248 chuỗi UCR Anomaly Archive; bài chính nói 44 chuỗi, bảng phụ lục liệt kê một tập con ~45 chuỗi):

| Metric | $\text{MOMENT}_0$ | $\text{MOMENT}_{LP}$ | GPT4TS | TimesNet | Anomaly Transformer | DGHL | k-NN |
|---|---|---|---|---|---|---|---|
| Adj. F1 (mean) | 0.585 | **0.628** | 0.424 | 0.537 | 0.492 | 0.425 | 0.554 |
| Adj. F1 (mean rank) | 3.410 | **3.005** | 4.862 | 3.642 | 4.326 | 5.071 | 3.681 |
| VUS-ROC (mean) | 0.670 | 0.684 | 0.611 | 0.679 | 0.661 | 0.646 | **0.706** |
| VUS-ROC (mean rank) | 4.056 | 3.382 | 5.193 | 3.897 | 3.913 | 4.403 | **3.153** |

  - MOMENT (cả zero-shot lẫn linear probing) vượt TimesNet, GPT4TS và 2 mô hình SOTA chuyên cho anomaly detection; k-NN nhỉnh hơn một chút về VUS-ROC nhưng thấp hơn về adjusted best F1.

**Imputation** (Bảng 6, MSE/MAE trung bình trên 4 tỷ lệ che):

| Dataset | $\text{MOMENT}_0$ | $\text{MOMENT}_{LP}$ | GPT4TS | TimesNet | Naive | Linear | Nearest | Cubic |
|---|---|---|---|---|---|---|---|---|
| Weather | 0.082 / 0.130 | 0.035 / 0.075 | **0.031 / 0.071** | 0.036 / 0.098 | 0.119 / 0.108 | 0.065 / 0.067 | 0.083 / 0.078 | 0.601 / 0.153 |
| ETTh1 | 0.402 / 0.403 | **0.139 / 0.234** | 0.227 / 0.254 | 0.175 / 0.264 | 1.185 / 0.658 | 0.775 / 0.534 | 0.900 / 0.579 | 2.178 / 0.916 |
| ETTm2 | 0.078 / 0.184 | **0.031 / 0.108** | 0.052 / 0.133 | 0.112 / 0.220 | 0.113 / 0.191 | 0.062 / 0.138 | 0.079 / 0.152 | 0.534 / 0.356 |
| Electricity | 0.250 / 0.371 | 0.094 / 0.211 | **0.072 / 0.183** | 0.124 / 0.248 | 1.474 / 0.869 | 0.737 / 0.592 | 0.923 / 0.629 | 2.257 / 0.888 |

  - $\text{MOMENT}_{LP}$ đạt lỗi tái tạo thấp nhất trên mọi dataset ETT; zero-shot thì vượt mọi phương pháp nội suy thống kê trừ linear interpolation.
  - Lưu ý: tác giả che *đoạn liên tiếp* dài 8 (khó hơn) thay vì từng điểm rời như các nghiên cứu trước (dễ hơn đến mức phương pháp thống kê ngang transformer).

### 4.2. MOMENT học được gì?

- Dùng sóng sin tổng hợp thay đổi một yếu tố $c$ (xu hướng, biên độ, tần số, dịch nền, pha), lấy biểu diễn mức chuỗi rồi vẽ PCA/t-SNE. Kết quả: MOMENT phân biệt được thay đổi về **xu hướng, biên độ, tần số, pha**, nhưng **không phân biệt được dịch dọc (baseline shift)** do mỗi tín hiệu đã bị chuẩn hóa (RevIN) trước khi mô hình hóa.
- *[Hình 4: PCA của embedding các sóng sin tổng hợp cho thấy cấu trúc rõ theo trend, scale, frequency, phase.]*
- Trên nhiều dataset phân loại, MOMENT học biểu diễn tách biệt các lớp ngay cả ở zero-shot, không cần nhãn.
- (Phụ lục F.5) Mask embedding học được xấp xỉ giá trị lấy từ phân phối chuẩn tắc; MOMENT tái tạo tín hiệu tần số thấp tốt hơn, và có một đỉnh lỗi tái tạo bất thường quanh tần số $c=64$.

### 4.3. Các tính chất của mô hình chuỗi thời gian lớn

- **Tăng kích thước mô hình giảm training loss:** như LLM, mô hình lớn hơn có loss huấn luyện thấp hơn, thậm chí trước epoch đầu. (Phụ lục F.6: kích thước mô hình lớn hơn nhìn chung cải thiện hiệu năng zero-shot — MSE/sMAPE thấp hơn, VUS-ROC cao hơn; và các checkpoint trước khi hết epoch đầu gợi ý rằng đa dạng dữ liệu huấn luyện cũng có thể giúp zero-shot.)
- **Chuyển giao đa phương thức:** đóng băng self-attention và feed-forward, MOMENT mô hình hóa được các chuỗi ảnh/văn bản/nhị phân tương đương GPT-2 và Flan-T5 cùng cỡ:

| Mô hình | Bit Memory | MNIST | CIFAR-10 | IMDb |
|---|---|---|---|---|
| GPT-2 | 1.000 | 0.975 | **0.711** | 0.867 |
| Flan-T5 | 1.000 | **0.987** | 0.672 | 0.861 |
| MOMENT | 1.000 | 0.982 | 0.620 | **0.872** |

- **Khởi tạo ngẫu nhiên hội tụ đến loss thấp hơn:** khi có đủ dữ liệu, pre-train từ đầu cho loss thấp hơn tiếp tục pre-train từ trọng số ngôn ngữ (Flan-T5) cùng cỡ (MOMENT-Small ngẫu nhiên thắng cuối cùng). Điều này cũng cho thấy Time Series Pile đủ dữ liệu để pre-train mô hình nền tảng từ đầu.
- *[Hình 6: loss huấn luyện theo số patch xử lý: trái – mô hình lớn hơn có loss thấp hơn; phải – Small khởi tạo ngẫu nhiên cuối cùng vượt Small khởi tạo từ Flan-T5.]*

---

## 5. Kết luận và hướng phát triển (Conclusion and Future Work)

- Bài báo phát hành họ mô hình nền tảng chuỗi thời gian mã nguồn mở đầu tiên, với đóng góp ở mọi khâu: Time Series Pile; các chiến lược xử lý đặc thù chuỗi thời gian cho pre-train đa dataset; ba cỡ transformer; và benchmark đa tác vụ nhấn mạnh tài nguyên/giám sát hạn chế.
- MOMENT hiệu quả với fine-tuning tối thiểu; hiệu năng vượt trội ở anomaly detection và classification (thường có dataset nhỏ) được cho là nhờ pre-training. Đồng thời, nhiều phương pháp thống kê và mạng nông vẫn khá tốt trên nhiều tác vụ.
- Cam kết khoa học mở: công bố Time Series Pile, mã, trọng số, log huấn luyện.
- **Hướng tương lai:** ứng dụng thực tế; mô hình nền tảng đa phương thức chuỗi thời gian + văn bản; cải thiện forecasting bằng pre-train với causal attention và mục tiêu dự báo.

---

## Phụ lục (Appendix)

### A–D. Changelog, Related Work, Contemporary Work, Hướng tương lai

- **A. Changelog:** bản 10/2024 bổ sung liên kết benchmark nghiên cứu/suy luận, ghi chú về công trình đương thời (Timer, TOTEM), và nêu rõ khác biệt về positional embedding.
- **B. Related Work (mở rộng):** nội dung trùng ý với Mục 2, bổ sung danh sách transformer cho chuỗi thời gian (Informer, Autoformer, FEDformer, Pyraformer, Anomaly Transformer, TST, TS-TCC) và lập luận rằng contrastive learning khó áp dụng vì các phép biến đổi như lật, cắt có thể không phù hợp với chuỗi thời gian.
- **C. Contemporary Work (Bảng 9):** so MOMENT với Moirai, Lag-Llama, Chronos, Timer, TimesFM, TimeGPT-1, TinyTimeMixers, TOTEM. Điểm khác chính của MOMENT: kiến trúc encoder T5 (40M/125M/385M), tokenization bằng patch cố định, mục tiêu tái tạo MSE, context 512, đánh giá **nhiều tác vụ** (forecasting, classification, anomaly detection, imputation) trong khi hầu hết công trình khác chỉ đánh giá forecasting; có phân tích diễn giải, tác động khởi tạo; chưa có dự đoán phân phối (đang thực hiện post hoc). Tổng dữ liệu pre-train của MOMENT (~1,13 tỷ quan sát, 13 triệu chuỗi) nhỏ hơn nhiều so với Moirai (27,65B), Chronos (84B), Timer (28B), TimesFM (100B). Không công trình nào đương thời đánh giá cross-modal transfer, tác động môi trường hay mức minh bạch.
- **D. Hướng tương lai:** nghiên cứu ảnh hưởng của hàm loss (Huber, L1, L2), độ dài patch (4, 8), tỷ lệ che (0.3, 0.6); dùng augmentation/dữ liệu tổng hợp và tinh chỉnh tỷ lệ trộn dataset trong Pile.

### E. Time Series Pile

- 13 miền: Healthcare, Human Body, Nature, Audio, Power, Economics, Traffic, Weather, Facilities, Web Services, Synthetic, Sensors, Gait. Tổng ~20 GB, ~13M chuỗi, ~1,23 tỷ mốc thời gian.
- Thống kê theo nguồn (số quan sát / số chuỗi):

| Nguồn trong Pile | Số quan sát | Số chuỗi |
|---|---|---|
| UCR classification | ~634 triệu | 290.226 |
| Monash (short-horizon forecasting) | ~281 triệu | 559.102 |
| TSB-UAD (anomaly detection) | ~129,5 triệu | 1.970 |
| FRED (short-horizon forecasting) | ~56,8 triệu | 382.441 |
| Informer (long-horizon forecasting) | ~26 triệu | 1.247 |

- Một tập con của Pile được dùng cho thí nghiệm (ví dụ ETT/Electricity/Weather cho imputation; ECG5000, UWaveGestureLibraryX, OSULeaf, MedicalImages, Ham cho classification; M3 yearly/quarterly/monthly cho short-horizon).

### F. Thiết lập thực nghiệm và kết quả (chi tiết)

- **Hyper-parameter chung** (phụ lục): fine-tune MOMENT-Base với batch 16, lịch cosine, learning rate khởi đầu $10^{-5}$ (khác với bài chính dùng MOMENT-Large; tác giả ghi nhận ở hai nơi hơi khác nhau, nên tham khảo bảng hyper-parameter từng tác vụ).

**F.1. Forecasting**
- Định nghĩa: cho $\mathcal{T}=[x_1,\dots,x_L]$, dự báo $[x_{L+1},\dots,x_{L+H}]$. Hai cấu hình MOMENT: (1) **không cần huấn luyện** — thêm các patch bị che vào cuối và dự đoán bằng reconstruction head mặc định; (2) thay bằng forecasting head và fine-tune.
- *Long-horizon:* bỏ biến ngoại sinh khỏi dataset đa biến, chỉ dự báo chuỗi mục tiêu. Với Time-LLM không chạy được Weather, Electricity, Traffic (hạn chế thời gian/GPU). Bảng đầy đủ (Bảng 18) và phiên bản look-back 96 lấy từ iTransformer (Bảng 19) có trong phụ lục.
- *Zero-shot short-horizon:* mô hình sâu (MOMENT, N-BEATS, GPT4TS, TimesNet) fine-tune trên dataset nguồn (M4 hoặc FRED) rồi đánh giá trên đích (M3/M4) cùng tần số; thống kê thì fit từng chuỗi. Dùng định nghĩa sMAPE của cuộc thi M4.

**F.2. Classification**
- Hai cách dùng: (a) thay head bằng linear head ra $C$ logit; (b) hai giai đoạn kiểu unsupervised representation learning: lấy biểu diễn mức chuỗi không nhãn, rồi huấn luyện bộ phân loại (ví dụ SVM kernel RBF).
- Baseline gồm 5 phương pháp học biểu diễn không giám sát, 8 mạng sâu có giám sát, DTW, TimesNet, GPT4TS. Kết quả baseline thu thập từ Ismail Fawaz et al. (2019), Yue et al. (2022) và Dau et al. (2018).

**F.3. Anomaly detection**
- Bài toán nhị phân trên từng mốc thời gian; dùng reconstruction head tái tạo chuỗi, mốc nào lệch quá ngưỡng thì coi là bất thường. Việc chọn ngưỡng tốt nằm ngoài phạm vi (dùng *adjusted best F1* tức chọn ngưỡng tốt nhất).
- Cửa sổ cố định 512; chuỗi dài hơn 2560 được hạ mẫu 10 lần; không dùng tiêu chí chung (joint criterion) của Anomaly Transformer cho công bằng. Zero-shot được so với DGHL khởi tạo ngẫu nhiên (DGHL không phải baseline zero-shot tầm thường vì tối ưu latent lúc suy luận).
- Bảng 22 liệt kê điểm chi tiết trên từng chuỗi: ví dụ MOMENT nổi bật ở nhiều chuỗi ECG/Italianpowerdemand/weallwalk nhưng kém ở một số chuỗi như PowerDemand1, MesoplodonDensirostris so với TimesNet/Anomaly Transformer.

**F.4. Imputation**
- Một patch chỉ được coi là "quan sát" khi mọi mốc thời gian trong patch đều quan sát; các patch còn lại được thay bằng $[\text{MASK}]$ và dùng reconstruction head để điền giá trị.
- Dataset: 4 tập ETT, Weather, Electricity. Baseline: TimesNet, GPT4TS (FPT), DGHL + 3 phương pháp nội suy thống kê (Cubic Spline, Linear, 1-D Nearest Neighbor).
- Che đoạn dài 8 thay vì điểm rời làm bài toán khó hơn đáng kể; với che điểm rời (Bảng 28) các phương pháp thống kê gần ngang transformer.

**F.5. MOMENT học được gì** — xem Mục 4.2 ở trên (sóng sin tổng hợp, phân tích mask embedding, quan hệ tần số–lỗi tái tạo).

**F.6. Ảnh hưởng của kích thước mô hình** — xem Mục 4.3. (Các mục F.7 Training losses, F.8 Efficiency Analysis, G Transparency Index, H Results Sources trong bản PDF gần như chỉ có tiêu đề/hình/bảng; I Radar Plot mô tả cách vẽ Hình 1: chuẩn hóa min-max điểm của mỗi phương pháp trên 5 tác vụ, dùng 1−MSE/sMAPE cho forecasting & imputation (Weather và tập con M4), còn accuracy/VUS-ROC trung bình cho classification & anomaly detection.)

---

## Ghi chú

- **Mô hình nền tảng cho chuỗi thời gian là đối tượng tấn công mới:** MOMENT pre-train trên một kho dữ liệu công khai khổng lồ (Time Series Pile, ~13M chuỗi) rồi chuyển sang nhiều tác vụ; với MIA (*membership inference attack*, tấn công suy luận thành viên) đây là bối cảnh pre-train/fine-tune điển hình cần xem xét rò rỉ ở cả hai giai đoạn.
- **Tiêu chí mục tiêu tái tạo (masked reconstruction, MSE) tạo tín hiệu cho MIA:** loss tái tạo trên từng patch/chuỗi là một đặc trưng tự nhiên để suy luận thành viên (so với loss trên mẫu không thuộc tập huấn luyện), tương tự các MIA dựa trên loss.
- **Chia train/val/test tường minh và seed cố định (13):** bài báo công bố chính xác phần nào dữ liệu được dùng pre-train (chỉ split train), thuận lợi để xác lập nhãn member/non-member (ví dụ dùng split val/test làm non-member) khi đánh giá MIA trên MOMENT.
- **Thiết kế hạ nguồn cho MIA:** các chế độ zero-shot ($\text{MOMENT}_0$), linear probing ($\text{MOMENT}_{LP}$) và fine-tuning toàn bộ cho phép khảo sát MIA ở nhiều mức độ truy cập/độ tinh chỉnh; head nhẹ + encoder đóng băng nghĩa là rò rỉ có thể nằm chủ yếu ở encoder pre-train.
- **Cần thận trọng:** RevIN chuẩn hóa từng chuỗi (mất thông tin dịch nền), chuỗi đa biến được xử lý độc lập từng kênh, và chuỗi được cắt/đệm về 512 — những chi tiết tiền xử lý này ảnh hưởng đến việc xây dựng điểm số MIA và so khớp mẫu (một chuỗi có thể xuất hiện dưới dạng nhiều cửa sổ/kênh). Ngoài ra bài báo không đề cập đến quyền riêng tư; dữ liệu có thể gồm y tế (ECG, EEG) nên rủi ro rò rỉ là vấn đề đáng quan tâm.
