# Tấn công suy luận thành viên (MIA) nhắm vào các mô hình chuỗi thời gian

*Membership Inference Attacks Against Time-Series Models*

- **Tác giả:** Noam Koren, Abigail Goldsteen, Guy Amit, Ariel Farkash (IBM Research, Haifa, Israel)
- **Nơi công bố / năm:** ACML 2024 (Proceedings of Machine Learning Research, tập 260); file trong thư mục dự án ghi năm 2025
- **Từ khóa:** Privacy, Machine Learning, Time-Series, Membership Inference

> *Bản tóm tắt chi tiết bằng tiếng Việt (do AI viết, không phải bản dịch nguyên văn).*
> Thuật ngữ kỹ thuật, tên mô hình, tập dữ liệu và từ viết tắt được giữ nguyên tiếng Anh; lần đầu xuất hiện có chú giải tiếng Việt.

---

## Tóm tắt (Abstract)

Dữ liệu chuỗi thời gian chứa thông tin cá nhân (đặc biệt là dữ liệu y tế như EEG, ECG) đặt ra rủi ro riêng tư nghiêm trọng khi được dùng để huấn luyện mô hình chẩn đoán/chăm sóc. Để quyết định có đưa mô hình vào production hay chia sẻ cho bên thứ ba, cần đánh giá rủi ro riêng tư của nó; công cụ chính cho việc này là **Membership Inference Attack (MIA – tấn công suy luận thành viên)**. Tuy vậy, các mô hình *dự báo* chuỗi thời gian gần như chưa được nghiên cứu dưới góc độ MIA.

Bài báo:

- khảo sát các kỹ thuật MIA hiện có trên mô hình chuỗi thời gian;
- đề xuất hai nhóm đặc trưng tấn công mới dựa trên hai thành phần cơ bản của chuỗi thời gian: **seasonality** (tính mùa vụ/chu kỳ, ước lượng bằng biến đổi Fourier đa chiều) và **trend** (xu hướng, xấp xỉ bằng đa thức bậc thấp);
- thử nghiệm trên nhiều loại mô hình dự báo và hai tập dữ liệu y tế; các đặc trưng mới nâng hiệu quả MIA rõ rệt (AUC-ROC cải thiện 3%–26%, TPR ở FPR thấp tăng tới 12 lần).

---

## 1. Giới thiệu (Introduction)

- **Bối cảnh:** có mâu thuẫn giữa nhu cầu phân tích dữ liệu cá nhân để cải thiện quy trình và nhu cầu bảo vệ riêng tư. Trong y tế, mô hình ML được dùng để hỗ trợ bác sĩ, dự đoán sự cố thiết bị y tế, hỗ trợ chăm sóc liên tục (kể cả triển khai tại nhà bệnh nhân).
- **Đánh giá rủi ro riêng tư** thường được thực hiện bằng cách chạy MIA lên mô hình rồi đo tỉ lệ thành công của tấn công.
- **MIA là gì:** phân biệt mẫu *đã* nằm trong tập huấn luyện của mô hình mục tiêu (**members** – thành viên) với mẫu *chưa* (**non-members**), dựa trên đầu ra của mô hình. Nhiều tấn công huấn luyện một bộ phân loại nhị phân (**attack model**) theo hướng của Shokri et al. (2016). MIA đã được áp dụng cho mô hình phân loại, hồi quy, đồ thị (graph) và sinh (generative) — nhưng **chưa được nghiên cứu đầy đủ cho mô hình dự báo chuỗi thời gian**.
- **Đóng góp chính:** hai đặc trưng tấn công mới khai thác *trend* và *seasonality*:
  - trend: khớp đa thức bậc thấp;
  - seasonality: Discrete Fourier Transform (DFT, biến đổi Fourier rời rạc).
- **Động cơ giả thuyết:** chuỗi thời gian về bản chất gồm trend + seasonality; nhiều mô hình SOTA (NFT, TimesNet, ...) đưa hai thành phần này trực tiếp vào kiến trúc, nên mô hình nhiều khả năng ước lượng *chính xác hơn* trend/seasonality của chuỗi đã gặp khi huấn luyện → đây là tín hiệu để phân biệt member/non-member.
- **Thực nghiệm:** 6 mô hình dự báo × 2 tập dữ liệu y tế; nhiều horizon dự báo; cải thiện từ 3% đến 26% so với baseline. Đây là bước đầu cho việc đánh giá riêng tư mô hình chuỗi thời gian, vốn bị bỏ qua.

---

## 2. Kiến thức nền (Background)

### 2.1. Mô hình dự báo chuỗi thời gian (Time-Series Forecasting models)

- Tiến trình lịch sử: mô hình tuyến tính cổ điển (ARIMA, Exponential Smoothing) → mạng nơ-ron hồi quy (LSTM, GRU) → CNN và TCN (Temporal Convolutional Network) → Transformer (Autoformer, FEDformer).
- **DLinear** (Zeng et al., 2022) là mô hình tuyến tính đơn giản đặt nghi vấn về hiệu quả của Transformer; sau đó **TimesNet** và **PatchTST** cải tiến vượt DLinear; **NFT (Neural Fourier Transform)** nổi lên là một mô hình đa biến hàng đầu.
- **Biến đổi Fourier đa chiều (Multidimensional Fourier Transform, MFT):** Fourier Transform vốn được dùng để tìm chu kỳ trong dữ liệu bằng cách chuyển sang miền tần số. Nhiều mô hình dùng nó: Autoformer (autocorrelation qua FFT), FEDformer (chọn các tần số chính), Fourier Neural Operator (xấp xỉ toán tử PDE), TimesNet (phân tách đặc trưng để bắt chu kỳ). MFT mở rộng cho dữ liệu nhiều chiều; tác giả lấy cảm hứng từ NFT và dùng **2D-DFT** để trích seasonality.

### 2.2. Membership Inference Attacks

- **Mục tiêu:** với mô hình $M$ và mẫu $x$, kẻ tấn công muốn xác định $x \in D$ ($D$ là tập huấn luyện). Nếu thành công, có thể lộ thông tin nhạy cảm (tiền sử bệnh, tài chính, sở thích), hoặc xác định một người thuộc nhóm/cộng đồng nào đó → nguy cơ phân biệt đối xử, tổn hại.
- Kẻ tấn công thường phân tích đầu ra của $M$ và tạo các **đặc trưng số** (feature) như loss trên mẫu, log-probability, entropy của đầu ra.
- **Với mô hình dự báo chuỗi thời gian:** kẻ tấn công muốn biết một chuỗi cụ thể (ví dụ kết quả ECG của một bệnh nhân) có được dùng để huấn luyện hay không — rủi ro lớn ở y tế và tài chính.

---

## 3. Các nghiên cứu liên quan (Related work)

- **Hisamoto et al. (2020):** MIA trên mô hình seq2seq cho dịch máy; đầu ra là chuỗi các phân loại rời rạc → khác với dữ liệu y tế có đầu vào/đầu ra là số liên tục.
- **Pyrgelis et al. (2017):** nghiên cứu đầu tiên về MIA trên dữ liệu vị trí *tổng hợp* (aggregate location time series), coi bài toán là phân loại xem một người dùng có thuộc tập tổng hợp hay không; kết quả cho thấy MIA là mối đe dọa thật, phụ thuộc vào kiến thức nền của đối thủ, đặc tính dữ liệu, số người dùng và khung thời gian tổng hợp.
- **Voyez et al. (2022):** MIA trên chuỗi thời gian tổng hợp bằng quy hoạch tuyến tính (linear programming), khai thác tương quan giữa độ dài chuỗi công bố và kích thước dữ liệu tổng hợp.
- **Khoảng trống:** theo tác giả, chưa có công trình nào đánh giá rủi ro/MIA một cách thấu đáo cho *mô hình ML huấn luyện trên chuỗi thời gian số*. Các công trình trên nhắm vào dữ liệu tổng hợp công bố hoặc mô hình seq2seq rời rạc, không phải mô hình dự báo.

---

## 4. Phương pháp (Methodology)

### 4.1. Phát biểu bài toán (Problem Statement)

- Đối tượng: MIA trên mô hình **dự báo chuỗi thời gian đa biến** (multivariate forecasting).
- Giả định: kẻ tấn công có trong tay một *mẫu đầy đủ* (cả phần quá khứ lẫn phần tương lai thực) — mẫu này có thể đã hoặc chưa dùng để huấn luyện.
- Ký hiệu:
  - $y = [y_1, \dots, y_T] \in \mathbb{R}^{M\times T}$: dữ liệu đến thời điểm $T$ (**lookback** – cửa sổ nhìn lại); $M$ là số biến.
  - $Y = [y_{T+1}, \dots, y_{T+H}] \in \mathbb{R}^{M\times H}$: $H$ điểm tương lai thực (**horizon** – tầm dự báo).
  - Đầu vào mô hình: $X \in \mathbb{R}^{M\times t} = [y_{T-t+1}, \dots, y_T]$ (cửa sổ độ dài $t \le T$).
  - $\hat{Y}$: giá trị mô hình dự báo cho $Y$.
- **Nhiệm vụ:** quyết định $X \in D$ hay không bằng cách so sánh giá trị thực $Y$ với dự báo $\hat{Y}$.

### 4.2. Đặc trưng cho MIA (Features for MIA)

- **Attack features** là các thuộc tính mà attack model dùng để phân biệt member/non-member; chọn đặc trưng tốt là yếu tố quyết định mức độ thành công.
- Ý tưởng: các đặc trưng đề xuất **tách thành phần seasonality và trend** của dự báo rồi *đối chiếu với* thành phần tương ứng của dữ liệu thật (Hình 1).
- *[Hình 1: lưu đồ từ mô hình dự báo gốc đến attack model; các đặc trưng trích từ $Y$ và $\hat{Y}$ được đưa vào bộ phân loại tấn công.]*
- Lập luận: nhiều mô hình (NFT, TimesNet, FEDformer, Autoformer, FNO, N-BEATS, NeuralProphet, ARIMA) đều tích hợp tường minh trend/seasonality, nên giỏi mô phỏng chúng trên dữ liệu đã thấy; chênh lệch này là lợi thế cho kẻ tấn công.
- Công cụ: **Multidimensional Fourier Transform** cho seasonality (trích mẫu chu kỳ hiệu quả), **đa thức bậc thấp** cho trend (nắm hướng chính, lọc bớt dao động).

#### 4.2.1. Seasonality

2D-DFT được áp dụng lên ma trận $Y \in \mathbb{R}^{M\times H}$ (giống NFT): gồm hai DFT 1 chiều liên tiếp — theo chiều biến (cột, ma trận Fourier $F_M$) rồi theo chiều thời gian (hàng, ma trận Fourier $F_H$). Cách này bắt được cả các mẫu mùa vụ *phát sinh từ tương tác giữa các biến*, thứ không thấy được ở bối cảnh đơn biến.

$$C = F_M \, Y \, F_H^{\top}$$

- $C$: ma trận hệ số Fourier;
- $F_M$, $F_H$: ma trận Fourier (các hàng là cos và sin ở các tần số $0, \dots, M/2$ và $0, \dots, H/2$), kích thước tương ứng theo số biến và số bước thời gian.

**Ba đặc trưng đầu vào cho attack model (nhóm S):**

1. Hệ số Fourier của giá trị **thật**: $C = F_1^{\top} Y F_2$;
2. Hệ số Fourier của giá trị **dự báo**: $\hat{C} = F_1^{\top} \hat{Y} F_2$;
3. Chuẩn $L_2$ của sai khác: $\lVert C - \hat{C} \rVert_2$.

#### 4.2.2. Trend

Mỗi biến của chuỗi $Y$ ($H$ điểm, $M$ biến) được xấp xỉ bằng đa thức bậc $d$:

$$Y = P \, A, \qquad A = (P^{\top}P)^{-1}P^{\top} Y$$

- $P$: ma trận Vandermonde dựng từ vector thời gian chuẩn hóa $t_i = (i-1)/H$, $i=1..H$; chứa các lũy thừa của $t$ từ $0$ đến $d-1$ (hàng $j$: $t_1^{j}, \dots, t_H^{j}$);
- $A$: ma trận hệ số đa thức, tìm bằng bình phương tối thiểu (least squares);
- $d$: bậc đa thức (trong thực nghiệm dùng bậc 4).

**Ba đặc trưng (nhóm T):**

1. Hệ số đa thức của trend giá trị **thật**: $A$;
2. Hệ số đa thức của trend giá trị **dự báo**: $\hat{A} = (P^{\top}P)^{-1}P^{\top}\hat{Y}$;
3. Chuẩn $L_2$: $\lVert A - \hat{A} \rVert_2$.

#### 4.2.3. MASE và MSE

Hai đặc trưng dạng sai số dự báo cũng được dùng (như baseline và làm thành phần kết hợp):

$$\text{MASE} = \frac{\frac{1}{H}\sum_{i=1}^{H}\lvert y_{T+i}-\hat{y}_{T+i}\rvert}{\frac{1}{H-1}\sum_{i=2}^{H}\lvert y_{T+i}-y_{T+i-1}\rvert}, \qquad \text{MSE} = \frac{1}{H}\sum_{i=1}^{H}(y_{T+i}-\hat{y}_{T+i})^2$$

- **MASE** (Mean Absolute Scaled Error): sai số tuyệt đối trung bình của mô hình chia cho sai số của dự báo "ngây thơ" (naïve: lấy giá trị bước trước);
- **MSE** (Mean Squared Error): sai số bình phương trung bình;
- $y_{T+i}$, $\hat{y}_{T+i}$: giá trị thật và dự báo tại bước $i$ của horizon.
- Với dữ liệu đa biến: tính trên từng biến rồi lấy trung bình qua các biến.

---

## 5. Thiết lập thực nghiệm (Experimental Setup)

### 5.1. Mô hình đe dọa và cấu hình tấn công (Threat Model and MIA Attack Setup)

- **Threat model dạng gray-box:** theo các nghiên cứu đánh giá riêng tư trước đó (Shachor et al. 2023; Amit et al. 2024; Anderson et al. 2024), kẻ tấn công có một tập con dữ liệu huấn luyện *và* một tập mẫu non-member. Điều này cho phép ước lượng rủi ro riêng tư **trường hợp xấu nhất** trước khi triển khai mà **không cần dựng shadow model** (mô hình bóng, như trong Shokri et al. 2016).
- **Khung đánh giá:** dùng privacy risk assessment framework của Shachor et al. (2023), dựa trên các tiến bộ MIA gần đây (Carlini et al. 2022a; Shokri et al. 2016):
  - cách tiếp cận **ensemble**: tạo nhiều attack model chuyên biệt cho các tập con dữ liệu khác nhau;
  - dùng tập đặc trưng đa dạng từ đầu vào/đầu ra của mô hình mục tiêu;
  - **grid search** toàn diện qua kiến trúc attack model, siêu tham số và tiền xử lý để chọn cấu hình cho hiệu năng tấn công cao nhất.
- **Các đặc trưng được kết hợp:** (1) Seasonality **S**; (2) Trend **T** (đa thức bậc 4, như Koren & Radinsky 2024 và Oreshkin et al. 2019); (3) **MASE**; (4) **MSE**; (5) Predicted Values **PV** (các giá trị dự báo).
- **Quy trình lặp:** 5 *instance* tấn công, mỗi instance 3 *run*.
  - Instance = chạy toàn bộ quá trình tối ưu tấn công trên một mẫu dữ liệu ngẫu nhiên khác;
  - Run = các cách chia dữ liệu khác nhau để huấn luyện/suy luận attack model trong một instance;
  - Mỗi instance lấy ngẫu nhiên **450 member + 450 non-member**; kết quả được lấy trung bình qua mọi run và instance.

### 5.2. Tập dữ liệu (Datasets)

| Dataset | Mô tả | Số biến | Timesteps | Lookback | Prediction horizons |
|---|---|---|---|---|---|
| **EEG** | Cơ sở dữ liệu EEG 36 kênh (Temple University Hospital EEG corpus), >1000 bản ghi từ 2002 đến nay, tần số lấy mẫu 250 Hz; subset gồm 32 bệnh nhân, 3 kênh đầu mỗi bệnh nhân | 3 | 9.620.519 | 100 | 1, 5, 10, 15, 20 |
| **ECG** | Georgia 12-Lead ECG Challenge Database (Emory University, PhysioNet), >10.000 người, 500 Hz; subset gồm 600 người | 12 | 2.393.563 | 100 | 1, 5, 10, 15, 20, 25, 30 |

- **Tiền xử lý:** loại outlier bằng phương pháp IQR (Interquartile Range), điền giá trị thiếu bằng trung bình, chuẩn hóa (standardization).
- **Chia dữ liệu theo bệnh nhân:** 42,5% bệnh nhân để huấn luyện (members), 15% để validation (tune siêu tham số nhằm có mô hình mạnh để tấn công), 42,5% còn lại làm non-member cho attack. Sau đó cắt thành cặp lookback/horizon bằng **sliding window** (cửa sổ trượt).

### 5.3. Mô hình mục tiêu (Models)

Sáu kiến trúc dự báo SOTA, tham số chọn theo hiệu năng trên validation; huấn luyện bằng MSE loss.

| Mô hình | Cấu hình chính | Số tham số (xấp xỉ) |
|---|---|---|
| DLinear | dimension 16, dropout 0,1 | ~800 |
| TCN | channels [2, 2], kernel 2, dropout 0,2 | ~1.000 |
| LSTM | 2 lớp, hidden 50 | ~34.000 |
| NFT | Fourier granularity 8 cho khối seasonality, đa thức bậc 4 cho khối trend, 2 khối mỗi stack | ~130.000 |
| TimesNet | model dim 16, dropout 0,1 | ~160.000 |
| PatchTST | 1 encoder + 1 decoder, model dim 16, dropout 0,1 | ~3.000.000 |

- *[Hình 2: MSE của các mô hình trên tập test (non-member) theo từng horizon, cho EEG và ECG; sai số nói chung tăng theo horizon.]*

---

## 6. Kết quả (Results)

- **Baseline** (cùng bối cảnh gray-box): tấn công dựa trên loss/dự đoán như trong black-box — gồm MSE, MASE, MSE+MASE, và predicted values.
- **Hai thước đo:** AUC-ROC (Area Under ROC curve) và **TPR@FPR=1%** (True Positive Rate khi False Positive Rate cố định 1%).

### 6.1. Kết quả AUC-ROC

#### 6.1.1. So với các tấn công dựa trên loss

- *[Hình 3 (EEG) và Hình 4 (ECG): AUC-ROC theo horizon cho từng mô hình, so sánh các tổ hợp đặc trưng T/S/MASE/MSE; các tổ hợp có T hoặc S thường cao nhất.]*
- Tổ hợp chứa Trend hoặc Seasonality cho AUC-ROC cao nhất, vượt baseline tới **26%**.
- Cải thiện so với tấn công chỉ dùng MSE (tốt nhất trung bình theo horizon, từng mô hình):

| Dataset | Cải thiện AUC-ROC nhỏ nhất | Cải thiện lớn nhất | Ghi chú |
|---|---|---|---|
| EEG | 8,44% (std 0,007) | 26,41% (std 0,006) | TimesNet cải thiện nhiều nhất |
| ECG | 2,97% (std 0,008) | 24,55% (std 0,007) | TimesNet cải thiện nhiều nhất |

- MASE đơn lẻ cho kết quả tấn công thấp nhất, nhưng MASE+MSE thường tốt hơn chỉ MSE.
- Kết luận phần: S và T mạnh và ổn định qua nhiều mô hình, dataset, horizon.

**Phân tích theo horizon.** Tác giả tính tương quan giữa độ dài horizon $H$ và phần trăm cải thiện AUC-ROC (giữa tấn công tốt nhất và tấn công chỉ MSE):

| Mô hình | Tương quan (EEG) | Tương quan (ECG) |
|---|---|---|
| PatchTST | 0,859 | 0,942 |
| TimesNet | 0,796 | 0,732 |
| DLinear | 0,883 | 0,177 |
| LSTM | 0,482 | 0,376 |
| TCN | 0,780 | −0,150 |
| NFT | 0,089 | −0,931 |

- Hầu hết cho tương quan dương: horizon càng dài, lợi thế của đặc trưng S/T so với MSE càng lớn. Ngoại lệ: NFT và TCN trên ECG (âm) — tác giả phỏng đoán do cả hai dựa trên lớp tích chập, bắt trend/seasonality theo cách khác.

**Phân tích theo mô hình** (AUC-ROC trung bình qua các đặc trưng và horizon):

| Mô hình | DLinear | TCN | LSTM | NFT | TimesNet | PatchTST |
|---|---|---|---|---|---|---|
| Số tham số | ~8×10² | ~1×10³ | ~34×10³ | ~13×10⁴ | ~16×10⁴ | ~3×10⁶ |
| AUC-ROC trung bình EEG | 0,59 | 0,82 | 0,83 | 0,82 | 0,60 | 0,72 |
| AUC-ROC trung bình ECG | 0,58 | 0,56 | 0,56 | 0,56 | 0,72 | 0,83 |

- ECG: PatchTST (nhiều tham số nhất) dễ bị tấn công nhất; DLinear (ít tham số nhất) khó tấn công nhất. EEG: DLinear cũng bền nhất; mức dễ tổn thương của các mô hình còn lại thay đổi giữa hai dataset.

#### 6.1.2. So với các tấn công dựa trên Predicted Values (PV)

- PV được thêm vào từng tổ hợp đặc trưng và thử độc lập. *[Hình 5: AUC-ROC khi thêm PV; đường "Benchmark Limit" là cận trên AUC-ROC của các tổ hợp không có PV.]*
- Không có xu hướng nhất quán. EEG: thêm PV luôn kém hơn tổ hợp không PV. ECG: kết quả biến thiên mạnh; PV đôi khi vượt Benchmark Limit nhưng không tổ hợp nào vượt nhất quán trên mọi horizon.
- Kết luận: PV đôi khi giúp nhưng tác động phụ thuộc vào mô hình, dataset và horizon.

### 6.2. TPR ở FPR = 1% (horizon = 5)

Cải thiện nổi bật:

- **EEG:** S với PatchTST tăng TPR **9 lần** so với MSE; T với NFT, TimesNet, LSTM tăng 2,1–4 lần; TCN tăng 2,1 lần với T+S; DLinear cải thiện tương tự với S hoặc T+S.
- **ECG:** T và T+S với NFT tăng ~3 lần; S với TimesNet và PatchTST tăng tới **12 lần**.

**Bảng 4 – TPR@1% FPR, EEG, horizon 5 (rút gọn):**

| Mô hình | MSE | MASE | MSE+MASE | T | S | T+S | T,S,MASE,MSE,PV | PV |
|---|---|---|---|---|---|---|---|---|
| DLinear | 0,03 | 0,02 | 0,04 | 0,06 | 0,07 | 0,07 | 0,06 | 0,06 |
| TCN | 0,23 | 0,04 | 0,23 | 0,40 | 0,42 | 0,43 | 0,35 | 0,31 |
| LSTM | 0,18 | 0,10 | 0,20 | 0,44 | 0,39 | 0,38 | 0,39 | 0,27 |
| NFT | 0,20 | 0,06 | 0,21 | 0,42 | 0,37 | 0,41 | 0,38 | 0,30 |
| TimesNet | 0,02 | 0,01 | 0,03 | 0,08 | 0,06 | 0,05 | 0,06 | 0,06 |
| PatchTST | 0,07 | 0,02 | 0,06 | 0,28 | **0,63** | 0,39 | 0,35 | 0,13 |

**Bảng 5 – TPR@1% FPR, ECG, horizon 5 (rút gọn):**

| Mô hình | MSE | MASE | MSE+MASE | T | S | T+S | T,S,MASE,MSE,PV | PV |
|---|---|---|---|---|---|---|---|---|
| DLinear | 0,01 | 0,01 | 0,02 | 0,03 | 0,03 | 0,03 | 0,03 | 0,03 |
| TCN | 0,01 | 0,01 | 0,01 | 0,02 | 0,02 | 0,02 | 0,02 | 0,02 |
| LSTM | 0,02 | 0,02 | 0,01 | 0,02 | 0,02 | 0,01 | 0,02 | 0,04 |
| NFT | 0,01 | 0,02 | 0,02 | 0,03 | 0,02 | 0,03 | 0,02 | 0,02 |
| TimesNet | 0,03 | 0,05 | 0,03 | 0,14 | 0,36 | 0,35 | 0,36 | 0,03 |
| PatchTST | 0,34 | 0,03 | 0,32 | 0,52 | **0,99** | 0,51 | 0,66 | 0,03 |

Nhận xét thêm (từ số liệu): TPR@1%FPR trên ECG rất thấp ở DLinear/TCN/LSTM/NFT (≤ ~0,04) — nghĩa là các mô hình này khá an toàn ở chế độ FPR thấp, trong khi PatchTST gần như bị lộ hoàn toàn (S: 0,99).

---

## 7. Kết luận và hướng phát triển (Conclusion and Future Work)

- Đây (theo tác giả) là công trình đầu tiên khám phá MIA trên mô hình ML chuỗi thời gian *số*.
- Đóng góp chính: hai đặc trưng Trend (khớp đa thức) và Seasonality (Multivariate Fourier Transform) khai thác thuộc tính nội tại của chuỗi thời gian để nhận diện member tốt hơn.
- Kiểm chứng trên 6 mô hình và 2 dataset y tế: AUC-ROC tăng 3%–26%, TPR ở FPR thấp tăng tới 12× so với đặc trưng truyền thống.
- **Hướng tương lai:**
  - kịch bản mô hình *pre-trained* trên nhiều bệnh nhân rồi **fine-tuning** cho một bệnh nhân (giám sát tại nhà): liệu tấn công có làm lộ dữ liệu huấn luyện gốc?
  - **user-level attack**: khai thác việc nhiều mẫu huấn luyện thuộc cùng một người.

*(Phần Acknowledgment — dự án NEMECYS do EU/UKRI/SERI đồng tài trợ — và danh mục tài liệu tham khảo được lược bỏ.)*

---

## Ghi chú

- **Khoảng trống được lấp:** bài này là một trong những nghiên cứu MIA đầu tiên nhắm vào *mô hình dự báo* chuỗi thời gian số (không phải dữ liệu tổng hợp hay seq2seq rời rạc) — điểm xuất phát tham khảo hữu ích cho mảng MIA trên time-series models.
- **Ý tưởng đặc trưng dùng lại được:** thay vì chỉ dùng loss/MSE, so sánh *cấu trúc* (trend, seasonality) của dự báo với thực tế là tín hiệu membership giàu hơn; có thể mở rộng sang các phân rã khác (STL, wavelet) hoặc các mô hình khác.
- **Phát hiện đáng chú ý:** mức dễ tổn thương không đơn thuần tỉ lệ thuận với số tham số (PatchTST dễ lộ nhất trên ECG, nhưng DLinear ít tham số nhất lại bền nhất) và cải thiện thường tăng theo horizon — cần chú ý khi thiết kế thí nghiệm.
- **Hạn chế cần lưu ý:** threat model gray-box (kẻ tấn công biết một phần training data, không dùng shadow model/LiRA) là *worst-case* và chỉ mức mẫu (sample-level), chưa phải mức người dùng; chỉ 2 dataset y tế, 450+450 mẫu mỗi instance; bài báo không đánh giá biện pháp phòng thủ (DP...). Các hướng tương lai (fine-tuning, user-level) chính là chỗ có thể đóng góp thêm.
- **Chỉ số báo cáo:** nhấn mạnh cả AUC-ROC và TPR@low-FPR; thực tế TPR@1%FPR cho thấy khác biệt giữa các đặc trưng rõ hơn nhiều so với AUC — nên dùng cả hai khi so sánh với các nghiên cứu khác.
