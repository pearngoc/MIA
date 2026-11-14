# Chronos: Học "ngôn ngữ" của chuỗi thời gian

*Chronos: Learning the Language of Time Series*

**Tác giả:** Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, Yuyang Wang (AWS AI Labs, Amazon và các đại học đối tác).

**Nơi đăng / năm:** Transactions on Machine Learning Research (TMLR), 10/2024 (arXiv:2403.07815v3). Mã nguồn và mô hình huấn luyện sẵn: `amazon-science/chronos-forecasting`.

> *Bản tóm tắt chi tiết bằng tiếng Việt (do AI viết, không phải bản dịch nguyên văn).*

---

## Tóm tắt (Abstract)

- **Chronos** là một framework đơn giản cho các mô hình dự báo chuỗi thời gian xác suất (*probabilistic forecasting*) được huấn luyện sẵn (*pretrained*). Ý tưởng cốt lõi: **chuẩn hoá (scaling) + lượng tử hoá (quantization)** giá trị thực thành các token rời rạc trong một từ vựng cố định, rồi huấn luyện các kiến trúc *language model* (LM) dựa trên transformer có sẵn bằng hàm mất mát **cross-entropy**, không sửa kiến trúc.
- Các mô hình dựa trên họ **T5** (20M–710M tham số) được huấn luyện trên một tập lớn dữ liệu công khai, bổ sung thêm dữ liệu tổng hợp sinh bằng **Gaussian process (GP)** để cải thiện khả năng tổng quát hoá.
- Đánh giá trên **42 tập dữ liệu** (cả mô hình thống kê cổ điển và deep learning): (a) vượt trội rõ rệt trên các tập đã nằm trong dữ liệu huấn luyện (*in-domain*); (b) ở chế độ **zero-shot** trên tập mới, đạt kết quả tương đương và đôi khi tốt hơn các phương pháp được huấn luyện riêng cho từng tập.
- Kết luận: mô hình pretrained có thể đơn giản hoá đáng kể quy trình dự báo (chỉ cần suy luận, không cần huấn luyện cho từng tác vụ).

---

## 1. Giới thiệu (Introduction)

- **Bối cảnh:** dự báo chuỗi thời gian là nền tảng ra quyết định trong bán lẻ, năng lượng, tài chính, y tế, khí hậu... Trước đây do các mô hình thống kê (ARIMA, ETS) chi phối; sau đó deep learning nổi lên nhờ dữ liệu lớn. Tuy vậy, các deep forecaster vẫn theo mô hình "huấn luyện và dự đoán trên cùng một tập dữ liệu"; chưa có mô hình tổng quát (general-purpose) thống nhất.
- **Hai hướng dùng LLM cho chuỗi thời gian:** (i) đưa chuỗi vào LLM bằng prompt ngôn ngữ tự nhiên (LLMTime, PromptCast); (ii) fine-tune LLM cho từng tác vụ (GPT4TS, Time-LLM). Nhược điểm: phải thiết kế prompt/fine-tune cho từng tác vụ, hoặc dùng LLM rất lớn nên suy luận đắt đỏ. Một số công trình song song (ForecastPFN, TimesFM, Lag-Llama, Moirai) cũng pretrain transformer nhưng có nhiều thiết kế riêng cho chuỗi thời gian.
- **Câu hỏi nghiên cứu của bài:** khác biệt cơ bản giữa LM dự đoán token tiếp theo và mô hình dự báo dự đoán giá trị tiếp theo là gì? Liệu LM tốt có "tự chạy được" trên chuỗi thời gian không? Từ đó các tác giả thách thức giả định rằng cần kiến trúc/đặc trưng chuyên biệt cho chuỗi thời gian.
- **Chronos** chỉ thay đổi bước biểu diễn: scale + quantize thành bin → token, rồi huấn luyện LM nguyên bản (encoder-decoder hoặc decoder-only). Khi suy luận, lấy mẫu token tự hồi quy (*autoregressive*), ánh xạ ngược về số thực, lấy nhiều quỹ đạo để có phân phối dự báo.
- *[Hình 1: sơ đồ Chronos — trái: chuỗi được mean-scale rồi lượng tử hoá thành token; giữa: LM (encoder-decoder hoặc decoder-only) huấn luyện bằng cross-entropy; phải: lúc suy luận lấy mẫu token tự hồi quy rồi dequantize + unscale để có dự báo xác suất.]*
- **Vấn đề dữ liệu:** dữ liệu chuỗi thời gian công khai ít và kém đa dạng hơn nhiều so với văn bản. Hai kỹ thuật tăng cường dữ liệu được đề xuất:
  - **TSMixup:** trộn lồi (convex combination) của nhiều chuỗi lấy từ các tập khác nhau.
  - **KernelSynth:** sinh chuỗi tổng hợp bằng GP với kernel được ghép ngẫu nhiên.
- **Đóng góp chính:** (1) framework tokenization tối giản dùng được với mọi kiến trúc LM; (2) hai phương pháp tăng cường dữ liệu; (3) benchmark 42 tập dữ liệu cho cả in-domain và zero-shot; (4) phân tích các lựa chọn thiết kế và giới hạn.
- Do là LM trên từ vựng cố định, Chronos có thể tận dụng ngay các tiến bộ của cộng đồng LLM.

---

## 2. Kiến thức nền và các công trình liên quan (Background and Related Work)

- **Bài toán:** cho chuỗi đều đặn $x_{1:C}=[x_1,\dots,x_C]$, cần dự đoán phân phối đồng thời $p(x_{C+1:C+H}\mid x_{1:C})$. Bài chỉ xét **dự báo đơn biến** (*univariate*), $x_i\in\mathbb{R}$.
  - $C$: độ dài ngữ cảnh (*context length*); $H$: độ dài dự báo (*prediction horizon*).
- **Phân loại phương pháp dự báo:**
  - *Mô hình local* (cổ điển): ETS, ARIMA, Theta — khớp riêng từng chuỗi.
  - *Mô hình global* (deep learning): học chung trên nhiều chuỗi của một tập — DeepAR, DeepState, TFT, PatchTST... Khác nhau ở kiến trúc (RNN/transformer) và cách mô hình hoá đích (hàm mật độ hay hồi quy phân vị). Một số (Informer, DLinear) chỉ cho dự báo điểm.
- **Language model:** mô hình hoá $p(w_{k+1}\mid w_{1:k})$ trên từ vựng $\mathcal{V}$. Có hai họ kiến trúc: *encoder-decoder* (T5, BART) và *decoder-only* (GPT-3, Llama 2).
- **Dự báo dựa trên LLM:**
  - *PromptCast:* chuyển chuỗi thành cặp văn bản hỏi-đáp, cần template riêng cho từng tập.
  - *LLMTime:* mã hoá số thành chuỗi chữ số (cố định độ chính xác) và dùng GPT-3/Llama 2 zero-shot; tốn tính toán.
  - *GPT4TS (One-Fits-All):* dùng GPT-2 làm backbone, chỉ fine-tune positional embedding và layer norm; đưa patch embedding trực tiếp.
  - *Time-LLM:* căn chỉnh embedding patch với "prototype" văn bản và prompt LLM đóng băng.
  - Điểm khác của Chronos: **huấn luyện LM từ đầu** trên chuỗi đã token hoá, không prompt/fine-tune từng tác vụ.
- **Dự báo zero-shot:** các công trình đầu chỉ huấn luyện trên một tập rồi thử trên tập khác. *ForecastPFN* huấn luyện hoàn toàn trên dữ liệu tổng hợp (xu hướng + mùa vụ định sẵn) và chỉ cho dự báo điểm. Chronos cũng dùng dữ liệu tổng hợp (GP) nhưng **kết hợp với dữ liệu thật** và cho dự báo xác suất. Các công trình đồng thời khác (Lag-Llama, MOMENT, TimesFM, Moirai) làm việc trên giá trị thực với nhiều thiết kế riêng (time feature, lag, patching, distribution head). Chronos theo hướng tối giản: phân phối **categorical** trên token, tức *regression via classification*.
- **Các tác vụ khác:** các mô hình đa nhiệm (TimesNet, SimMTM) xử lý imputation/classification/anomaly detection. Tác giả giả thuyết biểu diễn của Chronos cũng có thể dùng cho các tác vụ này.

---

## 3. Chronos: framework mô hình hoá ngôn ngữ cho chuỗi thời gian

**Triết lý thiết kế:** thay đổi tối thiểu — chủ yếu ở tokenization — để dùng nguyên kiến trúc và quy trình huấn luyện của LM.

### 3.1 Token hoá chuỗi thời gian (Time Series Tokenization)

Xét $x_{1:C+H}$, trong đó $C$ bước đầu là ngữ cảnh lịch sử, $H$ bước sau là horizon. Ánh xạ $x_i\in\mathbb{R}$ sang tập token hữu hạn gồm hai bước.

**Bước 1 — Scaling (chuẩn hoá).**
- Thang đo giữa các chuỗi chênh lệch lớn, gây khó tối ưu; mục tiêu ở đây còn là đưa giá trị vào miền phù hợp cho lượng tử hoá.
- Dùng phép affine $\tilde{x}_i=(x_i-m)/s$. Chronos chọn **mean scaling**: $m=0$ và
$$s=\frac{1}{C}\sum_{i=1}^{C}|x_i|$$
  (trung bình trị tuyệt đối của phần ngữ cảnh). Ưu điểm: **giữ nguyên giá trị 0** (vốn có ý nghĩa, ví dụ doanh số bằng 0, điện mặt trời ban đêm).

**Bước 2 — Quantization (lượng tử hoá).**
- Chọn $B$ tâm bin $c_1<\dots<c_B$ và $B-1$ biên $b_i$ với $c_i<b_i<c_{i+1}$.
- Hàm lượng tử hoá $q:\mathbb{R}\to\{1,\dots,B\}$ và giải lượng tử $d(j)=c_j$ (Eq. 1):
$$q(x)=\begin{cases}1 & x<b_1\\ j & b_{j-1}\le x<b_j\\ B & x\ge b_{B-1}\end{cases}$$
- Hai cách đặt bin: *quantile binning* (phụ thuộc dữ liệu, mỗi bin có số điểm xấp xỉ bằng nhau) và *uniform binning* (tâm cách đều trong $[c_1,c_B]$, biên nằm giữa hai tâm: $b_i=(c_i+c_{i+1})/2$). Chronos dùng **uniform binning** vì phân phối giá trị của tập chưa thấy có thể khác phân phối huấn luyện.
- **Hạn chế:** miền dự báo bị chặn trong $[c_1,c_B]$, nên về lý thuyết khó mô hình hoá chuỗi có xu hướng mạnh (xem mục 5.7).

**Token đặc biệt:** thêm `PAD` (đệm cho các chuỗi dài khác nhau và thay giá trị thiếu) và `EOS` (đánh dấu kết thúc chuỗi — không thật sự cần thiết nhưng tiện dùng với các thư viện LM). Từ vựng chuỗi thời gian $\mathcal{V}_{ts}$ gồm $B$ token giá trị + 2 token đặc biệt.

**Các lựa chọn khác:**
- **Không** dùng thông tin thời gian/tần số (ngày trong tuần, tuần trong năm...): chuỗi được xem thuần tuý là một dãy.
- Kiến trúc chính là **T5** (encoder-decoder); có thử nghiệm bổ sung với **GPT-2** (decoder-only).
- Không đổi kiến trúc, chỉ đổi kích thước từ vựng thành $|\mathcal{V}_{ts}|$ bằng cách cắt/mở rộng lớp embedding đầu vào và đầu ra.

### 3.2 Hàm mục tiêu (Objective Function)

Phân phối đầu ra là phân phối categorical trên $\mathcal{V}_{ts}$. Mất mát cho một chuỗi đã token hoá $z_{1:C+H+1}$ (tính cả EOS) là cross-entropy (Eq. 2):
$$\ell(\theta)=-\sum_{h=1}^{H+1}\sum_{i=1}^{|\mathcal{V}_{ts}|}\mathbb{1}_{(z_{C+h+1}=i)}\log p_\theta(z_{C+h+1}=i\mid z_{1:C+h})$$
- $\theta$: tham số mô hình; $\mathbb{1}_{(\cdot)}$: hàm chỉ thị; $p_\theta$: xác suất mô hình gán cho token $i$ ở bước kế tiếp. Trong thực tế lấy trung bình theo batch.
- **Nhận xét quan trọng:**
  - Cross-entropy **không nhận biết khoảng cách** (bin $i$ gần bin $i+1$ hơn $i+2$ không được mã hoá rõ); mô hình tự học quan hệ lân cận từ dữ liệu. Đây là *regression via classification*, khác với các mô hình truyền thống dùng phân phối tham số (Gaussian, Student-t) hoặc hồi quy phân vị.
  - **Hai ưu điểm của đầu ra categorical:** (1) không cần sửa kiến trúc/mục tiêu nên dùng được thư viện LM sẵn có; (2) không áp đặt dạng phân phối đầu ra, học được cả phân phối **đa đỉnh** — hữu ích cho mô hình pretrain trên nhiều miền.
  - Về mặt lý thuyết, coi đầu ra là biến *thứ tự* (ordinal) sẽ hợp lý hơn; tác giả để ngỏ như hướng mở rộng.

### 3.3 Dự báo (Forecasting)

1. Lấy mẫu tự hồi quy từ $p_\theta(z_{C+h+1}\mid z_{1:C+h})$, $h=1,\dots,H$, nhiều lần để có nhiều quỹ đạo.
2. Mỗi quỹ đạo (chuỗi token ID) được ánh xạ về giá trị thực bằng hàm giải lượng tử $d$ (Eq. 1).
3. Hoàn tác scaling (với mean scaling: nhân với $s$).
4. Từ tập các quỹ đạo thu được phân phối dự báo (phân vị, trung vị, v.v.).

---

## 4. Tăng cường dữ liệu (Data Augmentation)

Dữ liệu chuỗi thời gian công khai nghèo nàn hơn nhiều so với NLP (WikiText-103, C4, The Pile), gây khó cho mô hình zero-shot. Giải pháp: tăng đa dạng bằng mixup từ dữ liệu thật và bổ sung dữ liệu tổng hợp.

### 4.1 TSMixup: Time Series Mixup

- Mở rộng Mixup (vốn dành cho phân loại ảnh, trộn cặp ảnh và nhãn) từ 2 lên **nhiều chuỗi**.
- Với mỗi mẫu tăng cường:
  1. Lấy $k\sim U\{1,K\}$ (số chuỗi được trộn) và độ dài $l\sim U\{l_{\min},l_{\max}\}$.
  2. Với mỗi $i=1..k$: chọn ngẫu nhiên một tập dữ liệu, lấy một chuỗi con độ dài $l$ từ đó.
  3. **Mean-scale** từng chuỗi trước khi trộn, để chuỗi giá trị lớn/nhỏ có vai trò ngang nhau.
  4. Lấy trọng số $[\lambda_1,\dots,\lambda_k]\sim\mathrm{Dir}(\alpha)$ (Dirichlet đối xứng).
  5. Trả về tổ hợp lồi (Eq. 3):
$$\tilde{x}^{\text{TSMixup}}_{1:l}=\sum_{i=1}^{k}\lambda_i\,\tilde{x}^{(i)}_{1:l}$$
- $\tilde{x}^{(i)}$: chuỗi thứ $i$ đã scale; $\lambda_i$: trọng số, $\sum_i\lambda_i=1$.
- Tham số dùng (Algorithm 1, Phụ lục A): $K=3$, $\alpha=1.5$, $l_{\min}=128$, $l_{\max}=2048$.
- Trực giác: kết hợp mẫu hình của nhiều chuỗi để tăng đa dạng pattern. *[Hình 2: ví dụ TSMixup với k=1,2,3 — chuỗi gốc và các tổ hợp có trọng số.]*

### 4.2 KernelSynth: dữ liệu tổng hợp bằng Gaussian process

- Lấy cảm hứng từ *Automatic Statistician* (tìm kiếm kernel để giải thích chuỗi); KernelSynth đảo ngược quy trình: **ghép kernel ngẫu nhiên để sinh chuỗi mới**.
- **GP:** phân phối trên hàm, xác định bởi hàm trung bình $m(t)$ và kernel $\kappa(t,t')$ (hiệp phương sai giữa hai điểm).
- **Kernel bank** $\mathcal{K}$ gồm các kernel cơ bản: *linear* (xu hướng), *RBF* (biến thiên cục bộ trơn), *periodic* (mùa vụ ở các tần số thường gặp), cùng constant, white noise, rational quadratic (Bảng 2).
- Thuật toán (Algorithm 2), với $J=5$, độ dài $l_{syn}=1024$:
  1. Lấy $j\sim U\{1,J\}$ kernel (có hoàn lại) từ $\mathcal{K}$.
  2. Ghép dần các kernel bằng toán tử nhị phân chọn ngẫu nhiên $+$ hoặc $\times$ để có $\tilde\kappa$.
  3. Lấy một mẫu độ dài $l_{syn}$ từ $\mathrm{GP}(0,\tilde\kappa)$.
- *[Hình 3: (a) quy trình lấy kernel từ bank và ghép; (b) ví dụ chuỗi tổng hợp với mẫu hình phức tạp.]*

---

## 5. Thực nghiệm (Experiments)

Cấu trúc: tổng quan dữ liệu, huấn luyện, baseline, chỉ số (5.1–5.4); kết quả chính (5.5); phân tích siêu tham số (5.6); phân tích định tính và giới hạn (5.7).

### 5.1 Dữ liệu (Datasets)

- Tổng cộng **55 tập dữ liệu** công khai (Monash Repository, các cuộc thi M, Kaggle...) thuộc nhiều lĩnh vực: năng lượng, giao thông, y tế, bán lẻ, web, thời tiết, tài chính; tần suất từ 5 phút đến hàng năm. Chia làm ba nhóm:

| Nhóm | Số tập | Số chuỗi | Mục đích | Baseline |
|---|---|---|---|---|
| Pretraining-only | 13 | 795,936 | chỉ huấn luyện | – |
| **Benchmark I** | 15 | 97,272 | huấn luyện + đánh giá **in-domain** | Naive, SeasonalNaive, AutoETS, AutoTheta, SCUM, AutoARIMA, DeepAR, TFT, PatchTST, DLinear, WaveNet, N-BEATS, N-HiTS, GPT4TS, Lag-Llama, Moirai-1.0-R |
| **Benchmark II** | 27 | 190,674 | chỉ đánh giá **zero-shot** | tất cả ở trên + LLMTime, ForecastPFN |

- Tổng cộng 28 tập dùng để huấn luyện: khoảng **890K chuỗi đơn biến, ~84 tỷ quan sát (token)**.
- Với cả hai benchmark, **$H$ quan sát cuối** của mỗi chuỗi làm tập kiểm tra giữ lại (held-out); không mô hình nào được thấy chúng khi huấn luyện. Một *task* = cặp (dataset, prediction length).
- Cân bằng: để lại càng nhiều tập càng tốt cho zero-shot nhưng vẫn đủ đa dạng miền/tần suất cho huấn luyện.

### 5.2 Kho huấn luyện và quy trình (Training Corpus and Protocols)

- **Kiến trúc:** T5 (nhiều kích cỡ). Huấn luyện bốn cỡ: **Mini (20M), Small (46M), Base (200M), Large (710M)**; thêm **GPT-2 base (90M)** để chứng minh áp dụng được cho decoder-only.
- **Dữ liệu huấn luyện:** 10M mẫu TSMixup (từ 28 tập, $K=3$) + 1M chuỗi KernelSynth; lấy mẫu theo tỉ lệ **9:1** (TSMixup : synthetic). Chuỗi gốc xuất hiện trong TSMixup với xác suất 1/3.
- **Siêu tham số chính:**
  - Batch hiệu dụng 256; cắt cửa sổ ngẫu nhiên, rồi scale + quantize.
  - Bin đều trong $[c_1=-15,\ c_B=+15]$; kích thước từ vựng $|\mathcal{V}_{ts}|=4096$ (gồm PAD, EOS).
  - Context length **512** (mặc định của T5), prediction length **64**.
  - 200K bước, AdamW, weight decay 0.01, learning rate giảm tuyến tính từ 0.001 về 0.
  - Phần cứng: 8 GPU A100 (40GB), dùng TF32 và biên dịch mô hình.
- **Chi phí huấn luyện (Bảng 6):**

| Mô hình | Thời gian (giờ) | Chi phí (USD) |
|---|---|---|
| Chronos-T5 Mini | 7.68 | 252 |
| Chronos-T5 Small | 7.73 | 253 |
| Chronos-T5 Base | 17.96 | 588 |
| Chronos-T5 Large | 63.05 | 2066 |

### 5.3 Baseline

- Ba nhóm: **local** (Naive, Seasonal Naive, AutoETS, AutoARIMA, AutoTheta, và ensemble mạnh **SCUM**); **task-specific** (huấn luyện/fine-tune riêng từng tác vụ: WaveNet, DeepAR, N-BEATS, TFT, DLinear, PatchTST, N-HiTS, GPT4TS); **pretrained** (không huấn luyện riêng: Lag-Llama, Moirai-1.0-R; trên Benchmark II còn có ForecastPFN và LLMTime).
- Chi tiết triển khai ở Phụ lục C.

### 5.4 Chỉ số đánh giá (Evaluation Metrics)

- **WQL (weighted quantile loss):** đánh giá dự báo xác suất, xấp xỉ CRPS; tính trên 9 mức phân vị $\{0.1,\dots,0.9\}$. Với mô hình lấy mẫu, ước lượng phân vị từ **20 quỹ đạo**.
- **MASE (mean absolute scaled error):** đánh giá dự báo điểm (dùng trung vị, tức phân vị 0.5, cho mô hình xác suất).
- GPT4TS và ForecastPFN chỉ cho dự báo điểm nên chỉ đánh giá bằng MASE.
- **Tổng hợp kết quả:** với mỗi tập, lấy điểm của mô hình chia cho điểm của **Seasonal Naive** (điểm tương đối), rồi lấy **trung bình nhân (geometric mean)** trên các tập. Lý do: trung bình cộng có thể gây kết luận sai với tỉ số; trung bình nhân không phụ thuộc vào baseline được chọn và giữ nguyên thứ hạng. Mô hình lỗi/quá thời gian trên một tập được gán điểm tương đối = 1. Các tác vụ có trọng số bằng nhau.

### 5.5 Kết quả chính (Main Results)

Tóm tắt: trên in-domain, Chronos vượt cả local, task-specific lẫn pretrained khác; trên zero-shot, Chronos vượt rõ local và pretrained khác, tương đương các mô hình deep learning tốt nhất được huấn luyện riêng; fine-tune nhẹ giúp Chronos-T5 Small đứng đầu.

#### 5.5.1 Benchmark I: in-domain

*[Hình 4: điểm tương đối tổng hợp (WQL, MASE; thấp hơn là tốt hơn) trên 15 tập in-domain; Chronos và task-specific lấy trung bình 3 seed.]*

- Các Chronos-T5 lớn (Base, Large) đạt điểm tổng hợp và thứ hạng trung bình tốt nhất; hơn cả local (AutoETS, AutoARIMA), task-specific (PatchTST, DeepAR) và pretrained khác (Lag-Llama, Moirai-1.0-R).
- Các mô hình nhỏ (Mini, Small) và Chronos-GPT2 vẫn hơn đa số baseline. Moirai-1.0-R tốt hơn Lag-Llama; tuy nhiên Moirai-1.0-R lớn nhất (311M) vẫn thua Chronos-T5 Mini (20M) dù dùng kho dữ liệu lớn hơn.
- Seasonal Naive cạnh tranh tốt với các local khác, gợi ý các tập này có tính mùa vụ mạnh (năng lượng, giao thông).
- Ý nghĩa: huấn luyện một lần trên nhiều tập tốt hơn huấn luyện riêng từng tác vụ.

**Điểm tương đối tổng hợp trên Benchmark I (Bảng 7, 8; thấp hơn = tốt hơn; Seasonal Naive = 1.000):**

| Mô hình | WQL | MASE |
|---|---|---|
| Chronos-T5 Large | **0.564** | **0.695** |
| Chronos-T5 Base | 0.580 | 0.706 |
| Chronos-T5 Small | 0.603 | 0.727 |
| Chronos-T5 Mini | 0.598 | 0.732 |
| Chronos-GPT2 | 0.623 | 0.741 |
| PatchTST | 0.601 | 0.740 |
| N-HiTS / N-BEATS | 0.656 / 0.664 | 0.854 / 0.861 |
| DeepAR | 0.676 | 0.821 |
| Moirai-1.0-R (Large / Base) | 0.670 / 0.691 | 0.806 / 0.857 |
| TFT | 0.734 | 0.939 |
| Lag-Llama | 0.937 | 1.141 |
| Seasonal Naive | 1.000 | 1.000 |

(Các mô hình local khác như AutoETS/AutoARIMA/AutoTheta đều nằm gần hoặc trên mức ~0.9–1.1 ở MASE; Naive thuộc nhóm kém nhất, ~1.43–1.48.)

#### 5.5.2 Benchmark II: zero-shot

*[Hình 5: điểm tương đối tổng hợp trên 27 tập chưa từng thấy khi huấn luyện.]*

- Benchmark khó hơn Benchmark I (cải thiện so với baseline nhỏ hơn).
- Chronos vẫn vượt các mô hình local độc lập. Ở dự báo xác suất (WQL), Chronos chiếm các hạng **2–4**, hơn đa số task-specific đã huấn luyện trên chính các tác vụ này. Ở dự báo điểm (MASE), Chronos-T5 Large xếp **thứ 2**, hơn đa số baseline kể cả SCUM.
- Chronos vượt mạnh các pretrained khác (Moirai-1.0-R, Lag-Llama, LLMTime, ForecastPFN) và cả GPT4TS (fine-tune GPT-2 cho từng tập). Moirai-1.0-R đứng sau Chronos, nhưng chú ý nhiều tập Benchmark II nằm trong dữ liệu pretrain của Moirai nên thiết lập có thể có lợi cho nó.
- Chú thích: để tránh rò rỉ thông tin một cách chặt chẽ, thời điểm bắt đầu các tập zero-shot phải sau mốc cuối của dữ liệu pretrain; tác giả cho rằng rủi ro thấp vì các tập không trùng nhau ngoài mức phân loại khái niệm.

**Điểm tương đối tổng hợp trên Benchmark II (Bảng 9, 10):**

| Mô hình | WQL | MASE |
|---|---|---|
| Chronos-T5 Large | 0.645 | 0.823 |
| Chronos-T5 Base | 0.662 | 0.832 |
| Chronos-T5 Small | 0.667 | 0.841 |
| Chronos-T5 Mini | 0.678 | 0.850 |
| Chronos-GPT2 | 0.687 | 0.852 |
| PatchTST (task-specific) | 0.684 | 0.810 |
| TFT (task-specific) | 0.639 | 0.847 |
| N-HiTS / N-BEATS | 0.672 / 0.681 | ~0.83 |
| Moirai-1.0-R (Base / Large) | 0.696 / 0.720 | 0.907 / 0.876 |
| LLMTime | 0.804 | 0.962 |
| Lag-Llama | 1.097 | 1.291 |
| ForecastPFN | – (chỉ điểm) | 2.450 |
| Seasonal Naive | 1.000 | 1.000 |

(Số liệu bảng được đọc lại từ các bảng phụ lục có định dạng khó trích xuất; thứ hạng tổng quát khớp với phần mô tả trong văn bản.)

**Fine-tuning (khảo sát sơ bộ).**
- Chọn Chronos-T5 Small (zero-shot tốt, chi phí thấp); fine-tune riêng cho từng tập Benchmark II với learning rate khởi đầu 0.001 giảm tuyến tính về 0 trong 1000 bước.
- *[Hình 6: sau fine-tune, Chronos-T5 Small cải thiện rõ so với zero-shot và trở thành mô hình tốt nhất trung bình trên Benchmark II.]*
- Dù Small không phải biến thể tốt nhất khi zero-shot, nó vượt cả các Chronos lớn hơn và task-specific tốt nhất ⇒ fine-tune các biến thể lớn hơn có thể còn tốt hơn.

### 5.6 Phân tích siêu tham số (Analysis of Hyperparameters)

Mỗi lần chỉ đổi một tham số, các tham số khác giữ như kết quả chính (trừ phần kích thước mô hình/khởi tạo, các thí nghiệm còn lại dùng Chronos-T5 Small).

| Yếu tố | Kết quả chính |
|---|---|
| **Kích thước mô hình** (20M→710M) | Training loss giảm và hiệu năng downstream (in-domain lẫn zero-shot) cải thiện theo kích thước (*Hình 7*); mô hình lớn hơn có thể tốt hơn nữa nhưng không thử vì suy luận chậm. |
| **Khởi tạo** (ngẫu nhiên vs trọng số T5 pretrain trên C4) | Khởi tạo ngẫu nhiên hội tụ đến loss thấp hơn; với mô hình lớn, trọng số LM giúp loss giảm nhanh lúc đầu nhưng cuối cùng loss cao hơn. Hiệu năng downstream bằng hoặc kém hơn một chút (*Hình 8, 9*). ⇒ Trọng số LM không có lợi cho dự báo; ưu tiên khởi tạo ngẫu nhiên. |
| **TSMixup** | In-domain tương đương, nhưng **zero-shot tốt hơn** khi dùng TSMixup ⇒ tăng đa dạng dữ liệu giúp tổng quát hoá (*Hình 10a*). |
| **Tỉ lệ dữ liệu tổng hợp** (0–100%) | Cải thiện đều nhất ở khoảng **~10%** synthetic; tăng thêm thì kém đi vì dữ liệu GP không đại diện đủ chuỗi thật (*Hình 10b*). Mô hình chỉ huấn luyện trên synthetic vẫn khá, vượt rõ ForecastPFN và vài baseline, dù chưa thấy dữ liệu thật (*Hình 20*). |
| **Số bước huấn luyện** (đến 1M) | Hiệu năng tiếp tục cải thiện theo thời gian; hứa hẹn cho mô hình lớn nếu huấn luyện lâu hơn (*Hình 11a*). |
| **Context length** | Cải thiện đến 1024, sau đó bão hoà hoặc xấu đi — có thể do tập đánh giá thiếu dữ liệu tần suất cao (≥15 phút), vốn cần ngữ cảnh dài để bắt mùa vụ dài hạn (*Hình 11b*). |
| **Kích thước từ vựng** | MASE cải thiện nhẹ khi từ vựng lớn; WQL lúc đầu cải thiện rồi xấu đi (*Hình 11c*). Giải thích: MASE bất biến theo thang đo giống loss huấn luyện; WQL phụ thuộc thang đo nên ít tương quan với loss. Về nguyên tắc có đánh đổi: từ vựng quá nhỏ gây sai số rời rạc hoá lớn, quá lớn thì mỗi bin ít điểm dữ liệu → sai số tổng quát hoá. |

### 5.7 Phân tích định tính và giới hạn (Qualitative Analysis and Limitations)

Phân tích chủ yếu trên chuỗi tổng hợp để kiểm soát từng loại mẫu hình (Chronos-T5 Base); ví dụ trên dữ liệu thật ở Phụ lục E (Hình 22–24).

- **Nhiễu i.i.d.:** với $\mathcal{N}(0,1)$ và $\mathcal{N}(100,10)$, dự báo hợp lý; khoảng dự đoán 80% khớp khoảng thật (*Hình 12a*).
- **Xu hướng và mùa vụ:**
  - Dự báo đúng xu hướng tuyến tính nhưng **gặp khó với xu hướng mũ**, có thể do dữ liệu huấn luyện ít mẫu này; gợi ý khắc phục: scale logarit trước khi đưa vào.
  - Khi **ngữ cảnh quá ngắn**, mô hình đoán đúng hình dạng nhưng **đánh giá thấp xu hướng**; với ngữ cảnh dài hơn thì bám đúng (*Hình 13*, ví dụ Air Passengers).
  - Nhận diện mùa vụ rất tốt (sinusoid đơn và tổng của ba tần số); kết hợp cộng/nhân xu hướng với mùa vụ vẫn dự báo chính xác (*Hình 12c, d*).
- **Quá trình tự hồi quy (AR):** sinh chuỗi từ AR bậc 1–4 rồi so Chronos với mô hình AR đúng, AR khớp đúng bậc và AutoARIMA. AR(1), AR(2): các mô hình đúng/khớp có MSE tốt hơn. AR(3), AR(4): Chronos vượt AutoARIMA và ngang mô hình AR khớp đúng bậc ⇒ Chronos nhận diện được cấu trúc cơ bản (*Hình 14, 21*).
- **Phân phối dự báo linh hoạt:** KDE trên token ID được lấy mẫu cho năm bước đầu của horizon (NN5, Traffic, Hospital) cho thấy phân phối liên tục trên các token lân cận, nhiều hình dạng (kể cả đa đỉnh) dù cross-entropy không nhận biết khoảng cách (*Hình 15*). Gợi ý hướng cải tiến: chèn thông tin topo vào loss (ví dụ label smoothing trên các bin lân cận), như công trình Farebrother et al. (2024) trong RL.
- **Tràn miền và mất độ chính xác (giới hạn của tokenization):**
  - Token biểu diễn giá trị gốc trong $[-15s,\,15s]$ với $s$ là mean absolute scale. Nếu $s$ rất nhỏ so với biên độ giá trị (ví dụ chuỗi thưa, có spike thỉnh thoảng), một số giá trị rơi ra ngoài miền biểu diễn (*Hình 16a*).
  - Nếu $s$ rất lớn so với độ biến thiên (ví dụ sóng sin dịch lên cao), các bin cách nhau $30s/(B-1)$ trong không gian gốc, nên các giá trị gần nhau bị gộp vào cùng token, mất độ chính xác (*Hình 16b*); ở thực nghiệm dùng $B=4094$ bin giá trị.
  - Giải pháp heuristic khi suy luận: dùng chuẩn hoá khác (ví dụ standardization). Cải tiến tokenization không cần heuristic là hướng tương lai, dù kết quả thực tế vẫn tốt.

---

## 6. Thảo luận (Discussion)

### 6.1 Vượt ra ngoài dự báo đơn biến zero-shot

- Kết quả có thể còn tốt hơn nếu fine-tune, nhất là với **PEFT** (parameter-efficient fine-tuning, ví dụ LoRA), hoặc hiệu chuẩn bằng **conformal prediction** — Chronos hấp dẫn cho conformal vì không cần tập huấn luyện nên toàn bộ dữ liệu dùng được cho calibration.
- Hạn chế phạm vi: bài chỉ làm đơn biến, chuỗi đều đặn. Các tình huống khác: biến ngoại sinh (covariate), dự báo đa biến, chuỗi lấy mẫu không đều. Hướng khả thi: huấn luyện adaptor cho từng tác vụ để đưa covariate vào mô hình pretrained, hoặc ensemble stacking giữa Chronos và mô hình nhẹ như LightGBM.
- Các tác vụ khác (classification, clustering, anomaly detection): giả thuyết rằng biểu diễn của encoder Chronos-T5 mang tính tổng quát; cần khảo sát thêm.

### 6.2 Suy luận (Inference)

- Mô hình lớn chậm hơn task-specific nhưng không đến mức không dùng được; tốc độ của Chronos lớn tương đương một số mô hình local thống kê (*Hình 17*). Ưu điểm: một mô hình phục vụ mọi history length/tần suất/horizon/context length, nên bớt hẳn bước huấn luyện theo tác vụ.
- Nhờ framework LM, có thể nhận ngay cải tiến từ NLP: kernel CUDA tối ưu, quantization mô hình, speculative/lookahead decoding, mô hình ngữ cảnh dài (cho tần suất cao), và các kỹ thuật lấy mẫu (temperature, beam search, Top-K, nucleus sampling) để cải thiện chất lượng/tốc độ, đặc biệt dự báo điểm hiện cần gộp nhiều mẫu.

### 6.3 Dữ liệu (Data)

- Dữ liệu chuỗi thời gian công khai chất lượng cao còn hạn chế so với NLP, tạo thế lưỡng nan: dùng nhiều tập để huấn luyện thì còn ít tập để đánh giá zero-shot. Cần thêm các bộ dữ liệu lớn (đã có vài nỗ lực theo miền và liên miền).
- Hướng khác: cải thiện **dữ liệu tổng hợp**. Bài đã chứng minh hiệu quả của dữ liệu GP; nghiên cứu tiếp có thể tìm hiểu chế độ thất bại của mô hình huấn luyện trên dữ liệu tổng hợp để thu hẹp khoảng cách thật–giả.

---

## 7. Kết luận (Conclusion)

- Tác giả tiếp cận bài toán với tư duy tối giản: dùng nguyên kiến trúc/quy trình huấn luyện LM cho dự báo, thách thức quan niệm cần đặc trưng/kiến trúc riêng cho chuỗi thời gian. Chronos "bất khả tri về thời gian" (không dùng thông tin thời gian), tương thích mọi kiến trúc LM, chỉ cần tokenization qua scaling + quantization.
- Các mô hình pretrained vượt local và task-specific trên in-domain; trên zero-shot đạt kết quả cạnh tranh với deep learning tốt nhất được huấn luyện riêng, và còn có bằng chứng cải thiện thêm nhờ fine-tune.
- Hai đóng góp: (1) kiến trúc LM sẵn có đủ sức dự báo mà không cần tuỳ biến cho chuỗi thời gian, mở đường cho tiến bộ từ LLM và chiến lược dữ liệu tốt hơn; (2) về thực tiễn, mô hình pretrained lớn (theo thang dự báo) có thể đơn giản hoá quy trình dự báo mà không mất độ chính xác — thay thế cách tiếp cận huấn luyện/tinh chỉnh từng tác vụ bằng chỉ suy luận.

---

## Phụ lục (Appendices)

### A. Thuật toán (Algorithms)

Hai thuật toán giả mã được đưa ra, đã mô tả ở mục 4:
- **Algorithm 1 (TSMixup)** — tham số $K=3$, $\alpha=1.5$, $(l_{\min},l_{\max})=(128,2048)$.
- **Algorithm 2 (KernelSynth)** — tham số $J=5$, $l_{syn}=1024$.

**Kernel bank (Bảng 2):**

| Kernel | Dạng | Siêu tham số |
|---|---|---|
| Constant | $\kappa(x,x')=C$ | $C=1$ |
| White Noise | $\sigma_n\cdot\mathbb{1}_{x=x'}$ | $\sigma_n\in\{0.1,1\}$ |
| Linear | tích trong tuyến tính, độ rộng $\sigma$ | $\sigma\in\{0,1,10\}$ |
| RBF | $\exp\!\big(-\|x-x'\|^2/2l^2\big)$ | $l\in\{0.1,1,10\}$ |
| Rational Quadratic | $\big(1+\|x-x'\|^2/2\alpha l^2\big)^{-\alpha}$ | $\alpha\in\{0.1,1,10\}$ |
| Periodic | $\exp\!\big(-2\sin^2(\pi\|x-x'\|/p)/l^2\big)$ | chu kỳ $p$ lấy từ nhiều giá trị tương ứng tần số thường gặp (24, 48, 96, 168, 336, 672, 7, 14, 30, 60, 365, 730, 4, 26, 52, 6, 12, 40, 10) |

### B. Dữ liệu (Datasets)

- 55 tập dữ liệu: 13 chỉ pretrain, 15 in-domain, 27 zero-shot (liệt kê đầy đủ ở Bảng 3 của bài). Mô tả theo lĩnh vực:
  - **Năng lượng:** Australian Electricity, Electricity (15 phút/giờ/tuần), ERCOT Load, ETT, London Smart Meters, Solar, Spanish Energy and Weather, Wind Farms.
  - **Tài chính/kinh tế:** CIF 2016, Exchange Rate, FRED-MD, NN5.
  - **Y tế:** Covid Deaths, Hospital.
  - **Thiên nhiên:** Brazilian Cities Temperature, KDD Cup 2018 (chất lượng không khí), Temperature-Rain, USHCN, Weather, WeatherBench.
  - **Bán lẻ:** Car Parts, Dominick.
  - **Di chuyển/giao thông:** Mexico City Bikes, Pedestrian Counts, Rideshare, Taxi, Tourism, Traffic, Uber TLC.
  - **Khác:** các cuộc thi M1, M3, M4, M5.
  - **Web:** Wiki Daily (100k).

### C. Baseline

- 17 baseline. Mô hình local dùng thư viện **StatsForecast**, chu kỳ mùa vụ suy ra từ tần suất (ví dụ ngày = 7, giờ = 24). Mô hình deep learning dùng **GluonTS** (DeepAR, PatchTST, TFT, DLinear, WaveNet) hoặc **NeuralForecast** (N-BEATS, N-HiTS); các mô hình không dựa trên LLM huấn luyện/đánh giá **3 lần** lấy trung bình. ForecastPFN, GPT4TS, LLMTime, Lag-Llama, Moirai-1.0-R dùng mã tham chiếu.
- Hyperparameter mặc định, không tuning riêng cho dữ liệu. GluonTS tối ưu với batch 128, giới hạn 4 giờ, early stopping patience 200 epoch. PatchTST và DLinear dùng phiên bản xác suất (loss là negative log-likelihood Student-t) vì cho kết quả tốt hơn.
- **GPT4TS:** context length = hệ số theo tần suất × prediction length; fine-tune bằng loss MASE.
- **LLMTime:** chỉ thử Llama-2 70B (chi phí API lớn), context 512 như Chronos, `STEP_MULTIPLIER=1.4` thay cho 1.2 và tăng prediction length thêm 1 để tránh việc giá trị dự đoán cuối bị giảm biên độ; suy luận rất chậm (~0.8 giây/quan sát; ước tính ~92 giờ cho riêng tập Traffic), nên bỏ qua một số tập lớn.
- Phần cứng suy luận: CPU cho mô hình local, N-HiTS, N-BEATS; GPU V100 cho phần còn lại; LLMTime chạy trên instance 8 GPU V100.

### D. Chỉ số đánh giá (Evaluation Metrics)

- **MASE** cho một chuỗi: sai số tuyệt đối trung bình của dự báo chia cho sai số tuyệt đối trung bình của mô hình seasonal naive trên lịch sử (độ trễ mùa vụ $S$):
$$\mathrm{MASE}(\hat{x}_i,x_i)=\frac{\tfrac{1}{H}\sum_{t=C+1}^{C+H}|\hat{x}_{i,t}-x_{i,t}|}{\tfrac{1}{C-S}\sum_{t=1}^{C-S}|x_{i,t}-x_{i,t+S}|}$$
  Mẫu số tỉ lệ với thang đo của chuỗi nên MASE độc lập thang đo; tổng hợp theo tập bằng trung bình trên các chuỗi.
- **Quantile loss** của phân vị dự đoán $q$ ở mức $\alpha$ cho quan sát $x$ (Eq. 4):
$$\mathrm{QL}_\alpha(q,x)=\begin{cases}\alpha\,(x-q) & x>q\\ (1-\alpha)(q-x) & \text{ngược lại}\end{cases}$$
- **WQL:** gộp quantile loss theo mọi chuỗi và thời điểm, chuẩn hoá bằng tổng $|x|$:
$$\mathrm{WQL}_\alpha=\frac{\sum_{i,t}\mathrm{QL}_\alpha(q^{(\alpha)}_{i,t},x_{i,t})}{\sum_{i,t}|x_{i,t}|},\qquad \mathrm{WQL}=\frac{1}{K}\sum_{j=1}^{K}\mathrm{WQL}_{\alpha_j}$$
  với $K=9$ mức $\alpha\in\{0.1,\dots,0.9\}$. WQL xấp xỉ CRPS (có trọng số). Khác MASE, **WQL là chỉ số phụ thuộc thang đo** (sai số tuyệt đối gộp nên chuỗi có giá trị lớn chi phối).

### E. Kết quả bổ sung (Additional Results)

- Bảng 6 (chi phí huấn luyện, đã nêu ở mục 5.2); Bảng 7–10 cho điểm WQL/MASE thô trên từng tập, điểm tương đối tổng hợp và thứ hạng trung bình của mọi mô hình trên Benchmark I, II. Thứ hạng trung bình trên Benchmark I: Chronos-T5 Large khoảng 3.4 (WQL) và 3.3 (MASE), tốt nhất; trên Benchmark II xấp xỉ 8.3 (WQL) và 8.5 (MASE) vì cạnh tranh gay gắt hơn.
- Hình 18, 19: thứ hạng trung bình trên hai benchmark. Hình 20: hiệu năng zero-shot của **Chronos-T5-Synth (Small)** (chỉ huấn luyện trên dữ liệu KernelSynth) so với các baseline — vượt rõ ForecastPFN.
- Hình 21–24: thêm ví dụ định tính (AR(2), AR(3) và dự báo trên dữ liệu thật).

---

## Ghi chú

- **Mục tiêu xem xét:** Chronos là mô hình nền tảng (foundation model) chuỗi thời gian công khai, kiến trúc T5 chuẩn — một đối tượng tự nhiên để nghiên cứu **membership inference attack (MIA)** trên mô hình chuỗi thời gian, vì dữ liệu pretrain là các tập công khai đã biết (danh sách Bảng 3), cho phép dựng tập "member" và "non-member" rõ ràng (Benchmark II chưa từng thấy lúc huấn luyện).
- **Đầu ra categorical trên token:** mô hình trả xác suất trên $\sim$4096 token, rất thuận lợi cho các tín hiệu MIA dựa trên loss/xác suất (ví dụ perplexity, log-likelihood từng bước, LiRA); loss huấn luyện là cross-entropy trên token nên cũng là điểm số tự nhiên để so sánh member/non-member.
- **Tokenization lượng tử hoá:** scaling + quantization làm thông tin bị rời rạc hoá; khi thiết kế tấn công cần chú ý rằng nhiễu lượng tử và mean-scaling có thể che giấu hoặc làm lộ thông tin tuỳ chuỗi (đặc biệt chuỗi tỉ lệ nhỏ/lớn, xem mục 5.7).
- **Dữ liệu augment (TSMixup, KernelSynth):** mẫu huấn luyện thật sự là tổ hợp lồi của các chuỗi gốc (xác suất xuất hiện chuỗi gốc nguyên bản là 1/3), nên "membership" của một chuỗi gốc không hoàn toàn tương ứng với một mẫu huấn luyện cụ thể — điều này làm bài toán MIA phức tạp hơn so với dữ liệu không augment.
- **Chế độ in-domain vs zero-shot:** sự khác biệt hiệu năng lớn giữa tập đã thấy (Benchmark I) và tập chưa thấy (Benchmark II) cung cấp một sự thiên lệch tiềm tàng có thể bị khai thác bởi MIA, đồng thời là cơ sở để đánh giá mức độ ghi nhớ.
- **Kích thước mô hình và số bước huấn luyện** (20M–710M, đến 1M bước) có hiệu năng cải thiện đều — hữu ích để nghiên cứu mối quan hệ giữa dung lượng mô hình, thời gian huấn luyện và mức rò rỉ thành viên.
