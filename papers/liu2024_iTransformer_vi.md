# iTransformer: Transformer đảo chiều hiệu quả cho dự báo chuỗi thời gian

*iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*

- **Tác giả:** Yong Liu\*, Tengge Hu\*, Haoran Zhang\*, Haixu Wu, Shiyu Wang, Lintao Ma, Mingsheng Long (Tsinghua University – School of Software, BNRist; Ant Group)
- **Hội nghị/năm:** ICLR 2024 (arXiv:2310.06625v4)
- **Mã nguồn:** https://github.com/thuml/iTransformer

> *Bản tóm tắt chi tiết bằng tiếng Việt (do AI viết, không phải bản dịch nguyên văn).*

---

## Tóm tắt (Abstract)

- Sự trỗi dậy của các mô hình tuyến tính (linear forecaster) làm dấy lên nghi ngờ về việc tiếp tục sửa đổi kiến trúc Transformer cho dự báo chuỗi thời gian. Các Transformer-based forecaster truyền thống gộp **tất cả biến (variate) tại cùng một timestamp** thành một *temporal token* rồi áp dụng attention giữa các token theo thời gian.
- Hai vấn đề chính: (i) hiệu năng giảm và chi phí tính toán tăng mạnh khi lookback window dài; (ii) một token trộn nhiều biến có ý nghĩa vật lý khác nhau, hoặc là các sự kiện bị trễ pha, nên khó học được biểu diễn "theo từng biến" và attention map trở nên vô nghĩa.
- Tác giả đề xuất **iTransformer**: *không sửa bất kỳ thành phần cơ bản nào* của Transformer, chỉ **đảo chiều** việc áp dụng chúng. Toàn bộ chuỗi thời gian của mỗi biến được nhúng thành một **variate token**; self-attention dùng để nắm bắt tương quan đa biến (multivariate correlation), còn feed-forward network (FFN) áp dụng trên từng variate token để học biểu diễn phi tuyến của chuỗi.
- Kết quả: đạt state-of-the-art trên nhiều bộ dữ liệu thực tế, đồng thời cải thiện các Transformer khác, khả năng tổng quát hoá sang biến chưa thấy (unseen variates) và khai thác tốt lookback window dài.

---

## 1. Giới thiệu (Introduction)

- Transformer rất thành công ở NLP và thị giác máy tính (tuân theo scaling law), nên được đưa vào dự báo chuỗi thời gian (Autoformer, PatchTST…). Tuy nhiên, gần đây các lớp tuyến tính đơn giản (DLinear, TiDE…) lại vượt Transformer phức tạp cả về độ chính xác lẫn hiệu quả. Đồng thời, việc giữ tính độc lập giữa các biến và tận dụng thông tin tương quan đa biến được nhấn mạnh (Crossformer, TSMixer), nhưng khó thực hiện nếu không phá cấu trúc Transformer gốc.
- **Phân tích của tác giả về cách dùng Transformer hiện tại** (Hình 2, nửa trên):
  - Các điểm dữ liệu cùng timestamp có thể mang ý nghĩa vật lý hoàn toàn khác nhau, đo bằng thiết bị không đồng nhất, nhưng bị nhúng vào một token → tương quan đa biến bị xoá mất.
  - Token tạo từ một bước thời gian có *receptive field* quá cục bộ, và các sự kiện không căn thẳng theo thời gian (do độ trễ hệ thống) làm token ít thông tin.
  - Attention vốn *permutation-invariant* lại được dùng trên chiều thời gian, nơi thứ tự rất quan trọng.
- **Ý tưởng đảo chiều:** nhúng *toàn bộ* chuỗi của từng biến thành một variate token (có thể xem là trường hợp cực đoan của *patching*, với patch = cả chuỗi). Token khi đó mang biểu diễn toàn cục của biến, attention được dùng để tương quan các biến, còn FFN học biểu diễn tổng quát cho từng biến từ lookback tuỳ ý và giải mã ra chuỗi tương lai.
- **Kết luận cốt lõi:** không phải Transformer kém hiệu quả cho dự báo chuỗi thời gian, mà là nó đã bị *dùng sai cách*.
- *[Hình 1: biểu đồ radar so sánh MSE trung bình của iTransformer với PatchTST, DLinear, TimesNet, FEDformer, Transformer trên các dataset; iTransformer thấp nhất ở hầu hết.]*
- *[Hình 2: so sánh Transformer (embed các giá trị cùng timestamp thành temporal token, attention trên các token thời gian, FFN trên biểu diễn đa biến trộn lẫn) với iTransformer (embed từng chuỗi thành variate token, attention trên token biến để nắm tương quan, FFN trên biểu diễn từng chuỗi).]*
- **Đóng góp (3 điểm):**
  1. Xem xét lại kiến trúc Transformer, chỉ ra năng lực của các thành phần gốc trên chuỗi đa biến chưa được khai thác đúng mức.
  2. Đề xuất iTransformer: coi mỗi chuỗi độc lập là một token; self-attention nắm tương quan đa biến; layer normalization và FFN học biểu diễn toàn cục của chuỗi.
  3. Thực nghiệm SOTA trên các benchmark thực tế, kèm phân tích sâu về các module đảo chiều và lựa chọn kiến trúc.

---

## 2. Công trình liên quan (Related Work)

Tác giả chia các Transformer-based forecaster thành **4 nhóm** theo việc có sửa *thành phần* (component) hay *kiến trúc* (architecture) hay không (Hình 3):

| Nhóm | Sửa component? | Sửa architecture? | Ví dụ | Ý chính |
|---|---|---|---|---|
| (I) | Có | Không | Autoformer, Informer, FEDformer | Sửa attention để mô hình hoá phụ thuộc thời gian và giảm độ phức tạp trên chuỗi dài; bị các mô hình tuyến tính thách thức |
| (II) | Không | Không (xử lý chuỗi tốt hơn) | PatchTST, Non-stationary Transformer | Stationarization, Channel Independence, Patching → cải thiện ổn định |
| (III) | Có | Có | Crossformer | Attention mới + kiến trúc mới để nắm phụ thuộc cross-time và cross-variate |
| (IV) | Không | Có | **iTransformer** | Giữ nguyên component, chỉ đảo chiều cách dùng |

- Theo tác giả, iTransformer là công trình duy nhất thuộc nhóm (IV): các component đã được kiểm chứng rộng rãi, cái sai là *kiến trúc* được áp dụng không phù hợp.
- *[Hình 3: sơ đồ phân loại 4 nhóm theo hai trục sửa component/sửa architecture.]*

---

## 3. iTransformer

### 3.0 Bài toán và ký hiệu

- Dự báo chuỗi đa biến: cho quan sát lịch sử $\mathbf{X}=\{\mathbf{x}_1,\dots,\mathbf{x}_T\}\in\mathbb{R}^{T\times N}$ ($T$ bước thời gian, $N$ biến), dự đoán $S$ bước tương lai $\mathbf{Y}=\{\mathbf{x}_{T+1},\dots,\mathbf{x}_{T+S}\}\in\mathbb{R}^{S\times N}$.
- $\mathbf{X}_{t,:}$ là các giá trị đo cùng lúc tại bước $t$; $\mathbf{X}_{:,n}$ là toàn bộ chuỗi của biến $n$.
- Lưu ý quan trọng: các phần tử trong $\mathbf{X}_{t,:}$ có thể *không* phản ánh cùng một sự kiện (do độ trễ hệ thống giữa các biến), và khác nhau về đại lượng vật lý lẫn phân phối thống kê; ngược lại, một biến $\mathbf{X}_{:,n}$ thường nhất quán về các tính chất đó.

### 3.1 Tổng quan cấu trúc (Structure Overview)

- iTransformer dùng kiến trúc **encoder-only** của Transformer gồm: embedding, các Transformer block, và projection (Hình 4).
- **Lý do chọn encoder-only:** dự báo không cần giải mã sinh token theo kiểu generative; việc tạo chuỗi dự đoán được giao cho lớp tuyến tính (đã được chứng minh đủ năng lực bởi TiDE). Mô hình tập trung vào học biểu diễn và tương quan thích nghi giữa các chuỗi.
- Quá trình dự báo cho biến $n$:

$$\mathbf{h}_n^0=\text{Embedding}(\mathbf{X}_{:,n}),\quad \mathbf{H}^{l+1}=\text{TrmBlock}(\mathbf{H}^{l}),\ l=0,\dots,L-1,\quad \hat{\mathbf{Y}}_{:,n}=\text{Projection}(\mathbf{h}_n^L)$$

  - $\mathbf{H}=\{\mathbf{h}_1,\dots,\mathbf{h}_N\}\in\mathbb{R}^{N\times D}$: $N$ variate token, mỗi token có chiều $D$; chỉ số trên là chỉ số tầng.
  - $\text{Embedding}:\mathbb{R}^T\to\mathbb{R}^D$ và $\text{Projection}:\mathbb{R}^D\to\mathbb{R}^S$ đều là MLP.
  - $L$: số Transformer block.
- Các variate token tương tác với nhau qua self-attention, và được xử lý độc lập bởi FFN dùng chung trong mỗi block. Vì thông tin thứ tự thời gian đã được lưu "ngầm" trong thứ tự nơ-ron của FFN/embedding, **không cần positional embedding**.
- **Họ iTransformers:** kiến trúc không đòi hỏi gì ngoài việc attention dùng được cho tương quan đa biến, nên có thể cắm các biến thể attention hiệu quả (Reformer, Informer, Flowformer, FlashAttention) để giảm chi phí khi số biến lớn. Do số token linh hoạt, số biến lúc huấn luyện và suy luận có thể khác nhau.
- *[Hình 4: sơ đồ tổng thể: (a) chuỗi thô từng biến được embed độc lập thành token; (b) self-attention trên variate token cho ra bản đồ tương quan đa biến dễ diễn giải; (c) FFN dùng chung trích biểu diễn chuỗi; (d) LayerNorm giảm khác biệt giữa các biến.]*

### 3.2 Các thành phần của iTransformer (Inverted Transformer Components)

Một chồng $L$ block gồm layer normalization, FFN và self-attention, nhưng vai trò của từng module trên chiều đảo được xem xét lại.

**Layer normalization**
- Trong Transformer thông thường, LayerNorm chuẩn hoá biểu diễn đa biến của cùng một timestamp, dần trộn các biến với nhau; nếu các điểm này không cùng sự kiện, sẽ gây nhiễu tương tác giữa các quá trình không nhân quả/bị trễ, và làm chuỗi bị "quá mượt" (oversmoothed).
- Trong iTransformer, LayerNorm được áp dụng cho biểu diễn của *từng biến* (token):

$$\text{LayerNorm}(\mathbf{H})=\left\{\frac{\mathbf{h}_n-\text{Mean}(\mathbf{h}_n)}{\sqrt{\text{Var}(\mathbf{h}_n)}}\ \Big|\ n=1,\dots,N\right\}$$

  ($\mathbf{h}_n$: token của biến $n$; Mean/Var: trung bình và phương sai trên chiều đặc trưng $D$.)
- Cách làm này gần với các kỹ thuật xử lý non-stationarity (RevIN, Stationarization) và giúp giảm chênh lệch do các phép đo không nhất quán, vì mọi chuỗi đều được đưa về phân phối chuẩn hoá.

**Feed-forward network (FFN)**
- Trong Transformer gốc, token của một timestamp quá cục bộ và có thể lệch pha nên FFN khó trích thông tin tốt. Khi đảo chiều, FFN được áp trên biểu diễn *toàn bộ chuỗi* của từng biến; nhờ universal approximation, nó có thể học các biểu diễn phức tạp. Qua nhiều tầng, FFN vừa mã hoá chuỗi quan sát vừa giải mã biểu diễn ra chuỗi tương lai bằng các kết nối dày phi tuyến (giống các mô hình toàn MLP như MLP-Mixer, TiDE).
- Phép tuyến tính giống nhau áp lên các chuỗi độc lập chính là sự kết hợp giữa **linear forecaster** và **Channel Independence**. Giả thuyết của tác giả: các nơ-ron MLP học được thuộc tính nội tại của *bất kỳ* chuỗi nào (biên độ, chu kỳ, thậm chí phổ tần – "nơ-ron như bộ lọc"), do đó là bộ học biểu diễn dự báo tốt hơn so với self-attention áp trên các điểm thời gian.
- Sự phân công này giải thích hai lợi ích thực nghiệm (Mục 4): hiệu năng tăng khi lookback dài, và khả năng tổng quát hoá sang biến chưa thấy.

**Self-attention**
- Mỗi chuỗi biến được coi là một quá trình độc lập. Từ $\mathbf{H}\in\mathbb{R}^{N\times D}$, các phép chiếu tuyến tính cho $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d_k}$.
- Điểm số trước Softmax: $A_{i,j}=(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})_{i,j}\propto \mathbf{q}_i^\top\mathbf{k}_j$, với $\mathbf{q}_i,\mathbf{k}_j\in\mathbb{R}^{d_k}$ là query/key của từng token. Vì các token đã được chuẩn hoá theo chiều đặc trưng, mỗi phần tử $A_{i,j}$ phần nào phản ánh tương quan giữa hai biến, và cả ma trận $\mathbf{A}\in\mathbb{R}^{N\times N}$ là **bản đồ tương quan đa biến**. Các biến tương quan cao được trọng số lớn hơn khi trộn giá trị $\mathbf{V}$.
- Cơ chế này tự nhiên và dễ diễn giải hơn cho dự báo đa biến (kiểm chứng bằng trực quan hoá ở Mục 4.3 và Phụ lục E.1).

---

## 4. Thực nghiệm (Experiments)

### Datasets

- 7 bộ dữ liệu thực tế: **ECL**, **ETT** (4 tập con), **Exchange**, **Traffic**, **Weather** (theo Autoformer), **Solar-Energy** (theo LSTNet) và **PEMS** (4 tập con, theo SCINet). Ngoài ra thêm **Market** (6 tập con, ở Phụ lục F.4): tải server theo phút của ứng dụng giao dịch trực tuyến Alipay, với hàng trăm biến.
- Chi tiết bộ dữ liệu xem Phụ lục A.1 (Bảng 4 tóm tắt bên dưới).

### 4.1 Kết quả dự báo (Forecasting Results)

**Baselines (10 mô hình):**
- Transformer-based: Autoformer, FEDformer, Stationary, Crossformer, PatchTST.
- Linear-based: DLinear, TiDE, RLinear.
- TCN-based: SCINet, TimesNet.

**Cài đặt:** lookback cố định $T=96$; độ dài dự đoán $S\in\{96,192,336,720\}$ (riêng PEMS dùng $\{12,24,48,96\}$ theo Phụ lục). Chỉ số: MSE/MAE (càng thấp càng tốt). Kết quả trung bình trên các độ dài dự đoán.

**Kết quả chính (Bảng 1, MSE / MAE trung bình):**

| Dataset | iTransformer | RLinear | PatchTST | Crossformer | TiDE | TimesNet | DLinear |
|---|---|---|---|---|---|---|---|
| ECL | **0.178 / 0.270** | 0.219 / 0.298 | 0.205 / 0.290 | 0.244 / 0.334 | 0.251 / 0.344 | 0.192 / 0.295 | 0.212 / 0.300 |
| ETT (Avg) | 0.383 / 0.399 | 0.380 / 0.392 | 0.381 / 0.397 | 0.685 / 0.578 | 0.482 / 0.470 | 0.391 / 0.404 | 0.442 / 0.444 |
| Exchange | 0.360 / 0.403 | 0.378 / 0.417 | 0.367 / 0.404 | 0.940 / 0.707 | 0.370 / 0.413 | 0.416 / 0.443 | 0.354 / 0.414 |
| Traffic | **0.428 / 0.282** | 0.626 / 0.378 | 0.481 / 0.304 | 0.550 / 0.304 | 0.760 / 0.473 | 0.620 / 0.336 | 0.625 / 0.383 |
| Weather | **0.258 / 0.278** | 0.272 / 0.291 | 0.259 / 0.281 | 0.259 / 0.315 | 0.271 / 0.320 | 0.259 / 0.287 | 0.265 / 0.317 |
| Solar-Energy | **0.233 / 0.262** | 0.369 / 0.356 | 0.270 / 0.307 | 0.641 / 0.639 | 0.347 / 0.417 | 0.301 / 0.319 | 0.330 / 0.401 |
| PEMS (Avg) | **0.119 / 0.218** | 0.514 / 0.482 | 0.217 / 0.305 | 0.220 / 0.304 | 0.375 / 0.440 | 0.148 / 0.246 | 0.320 / 0.394 |

(SCINet đạt PEMS Avg 0.121/0.222, FEDformer/Stationary/Autoformer thua rõ ở hầu hết dataset.)

**Nhận xét của tác giả:**
- iTransformer đặc biệt mạnh với chuỗi **nhiều biến** (Traffic 862 biến, Solar 137, ECL 321, PEMS).
- PatchTST (SOTA trước đó) thất bại ở nhiều trường hợp trên PEMS; lý do có thể là chuỗi biến động mạnh và patching mất tập trung vào cục bộ, trong khi iTransformer tổng hợp biến thiên toàn chuỗi.
- Crossformer – vốn cố tình mô hình hoá tương quan đa biến – vẫn kém hơn, gợi ý việc tương tác giữa các patch *không căn thời gian* của các biến khác nhau gây nhiễu không cần thiết.
- Kết luận: component gốc của Transformer đủ năng lực cho cả mô hình hoá thời gian lẫn tương quan đa biến, miễn là kiến trúc được đảo chiều.

### 4.2 Tính tổng quát của iTransformers (iTransformers Generality)

Áp dụng khung đảo chiều lên Transformer và các biến thể giảm độ phức tạp attention: Reformer, Informer, Flowformer và FlashAttention ("Flashformer").

**Cải thiện hiệu năng (Bảng 2):** giảm MSE tương đối trung bình ~**38.9%** (Transformer), **36.1%** (Reformer), **28.5%** (Informer), **16.8%** (Flowformer), **32.2%** (Flashformer). Do attention nay chạy trên chiều biến, các attention tuyến tính có thể giải quyết chi phí khi số biến lớn – điều vốn tốn kém với Channel Independence.

| Dataset | Mô hình | Gốc (MSE) | +Inverted (MSE) | Giảm |
|---|---|---|---|---|
| ECL | Transformer | 0.277 | 0.178 | 35.6% |
| ECL | Reformer | 0.338 | 0.208 | 38.4% |
| ECL | Informer | 0.311 | 0.216 | 30.5% |
| Traffic | Transformer | 0.665 | 0.428 | 35.6% |
| Traffic | Flowformer | 0.750 | 0.524 | 30.1% |
| Weather | Transformer | 0.657 | 0.258 | 60.2% |
| Weather | Reformer | 0.803 | 0.248 | 69.2% |

**Tổng quát hoá sang biến chưa thấy (Variate generalization):**
- Do số token linh hoạt và FFN dùng chung trên mọi variate token, mô hình có thể huấn luyện với số biến khác lúc suy luận. Giả thuyết: "nơ-ron là bộ lọc" học các quy luật nội tại có thể chuyển giao giữa các biến.
- Thiết lập: chia các biến của mỗi dataset thành 5 nhóm (folder), huấn luyện chỉ trên 20% số biến của một nhóm, rồi dự báo *toàn bộ* biến mà không fine-tuning; báo cáo trung bình 5 nhóm. So với chiến lược **CI-Transformer** (Channel Independence, backbone dùng chung, dự báo từng biến một).
- Kết quả: iTransformer suy luận một lần cho mọi biến (CI phải lặp từng biến, rất chậm), và nhìn chung tăng sai số ít hơn → FFN học được biểu diễn chuyển giao tốt. Hướng tiềm năng: xây foundation model trên iTransformer, huấn luyện chung các chuỗi đa biến có số biến khác nhau.
- *[Hình 5: cột MSE trên ECL, Traffic, Solar-Energy, PEMS so sánh CI-Transformers và iTransformers (Transformer, Informer, Reformer, Flowformer) khi chỉ train 20% biến; phiên bản iTransformer ít suy giảm hơn.]*

**Tăng lookback length:**
- Trước đây Transformer không nhất thiết tốt hơn khi lookback tăng (attention bị phân tán), trong khi mô hình tuyến tính thì có. Khi đảo chiều (MLP/FFN trên chiều thời gian), Transformer và các biến thể hưởng lợi rõ rệt từ lookback dài.
- *[Hình 6: MSE theo lookback $T\in\{48,96,192,336,720\}$ ($S=96$) trên ECL và Traffic; các phiên bản đảo chiều giảm lỗi khi $T$ tăng, bản gốc thì không ổn định.]*

### 4.3 Phân tích mô hình (Model Analysis)

**Ablation (Bảng 3, trung bình các độ dài dự đoán; MSE / MAE):**

| Thiết kế (Variate / Temporal) | ECL | Traffic | Weather | Solar |
|---|---|---|---|---|
| **iTransformer (Attention / FFN)** | **0.178 / 0.270** | **0.428 / 0.282** | 0.258 / 0.278 | **0.233 / 0.262** |
| Replace: Attention / Attention | 0.193 / 0.293 | 0.913 / 0.500 | 0.255 / 0.280 | 0.261 / 0.291 |
| Replace: FFN / Attention (≈ Transformer gốc) | 0.202 / 0.300 | 0.863 / 0.499 | 0.258 / 0.283 | 0.285 / 0.317 |
| Replace: FFN / FFN | 0.182 / 0.287 | 0.599 / 0.348 | 0.248 / 0.274 | 0.269 / 0.287 |
| w/o: Attention / bỏ temporal | 0.189 / 0.278 | 0.456 / 0.306 | 0.261 / 0.281 | 0.258 / 0.289 |
| w/o: bỏ variate / FFN | 0.193 / 0.276 | 0.461 / 0.294 | 0.265 / 0.283 | 0.261 / 0.283 |

- Thiết kế attention-trên-biến + FFN-trên-thời gian nhìn chung tốt nhất; kiến trúc Transformer gốc (hàng 3) là tệ nhất, cho thấy nguy cơ của cách dùng truyền thống (chi tiết ở Phụ lục E.3).

**Phân tích biểu diễn chuỗi (series representations):**
- Dùng **CKA (centered kernel alignment)** giữa đặc trưng đầu ra của block đầu và block cuối. Với dự báo (tác vụ sinh mức thấp), CKA cao thường đi cùng hiệu năng tốt hơn. Hình 7 (trái) cho thấy ranh giới rõ: iTransformers có CKA cao hơn và MSE thấp hơn các Transformer biến thể → biểu diễn phù hợp hơn sau khi đảo chiều.

**Phân tích tương quan đa biến:**
- Trên ví dụ Solar-Energy (tương quan giữa lookback và tương lai khác nhau rõ), bản đồ attention ở tầng nông giống tương quan của chuỗi lookback thô, còn ở tầng sâu dần giống tương quan của chuỗi *tương lai* cần dự đoán. Điều này gợi ý: attention tạo tương quan có thể diễn giải được, còn việc "mã hoá quá khứ và giải mã tương lai" chủ yếu diễn ra trong biểu diễn chuỗi ở FFN.
- *[Hình 7: trái – MSE và CKA của Transformer vs iTransformer; phải – ví dụ tương quan lookback/future và score map ở tầng 1 và tầng L.]*

**Chiến lược huấn luyện hiệu quả (efficient training strategy):**
- Với số biến rất lớn, attention bậc hai gây tốn kém. Tận dụng khả năng tổng quát sang biến chưa thấy: mỗi batch chỉ lấy ngẫu nhiên một phần các biến để huấn luyện, nhưng lúc suy luận vẫn dự đoán tất cả biến.
- Hiệu năng gần như không đổi so với huấn luyện full-variate trong khi bộ nhớ giảm đáng kể.
- *[Hình 8: MSE (trái) ổn định theo tỉ lệ biến lấy mẫu; bộ nhớ GPU (phải) giảm mạnh trên ECL, Traffic, Solar-Energy.]*

---

## 5. Kết luận và hướng tương lai (Conclusion and Future Work)

- iTransformer đảo chiều cấu trúc Transformer mà không đổi module gốc: chuỗi độc lập → variate token, attention cho tương quan đa biến, LayerNorm + FFN cho biểu diễn chuỗi. Đạt SOTA và thể hiện tính tổng quát khung cao.
- Hướng tương lai: pre-training quy mô lớn và mở rộng sang các tác vụ phân tích chuỗi thời gian khác.

*(Các mục Ethics Statement, Reproducibility Statement, Acknowledgments và References được lược bỏ. Ethics: tác giả cho rằng chỉ tập trung dự báo nên không có rủi ro đạo đức. Reproducibility: kiến trúc được hình thức hoá bằng công thức, chi tiết triển khai ở phụ lục.)*

---

## Phụ lục (Appendix)

### A. Chi tiết triển khai (Implementation Details)

#### A.1 Mô tả dataset

| Dataset | Số biến | Tần suất | Lĩnh vực | Độ dài dự đoán |
|---|---|---|---|---|
| ETTh1, ETTh2 | 7 | 1 giờ | Điện (máy biến áp) | 96/192/336/720 |
| ETTm1, ETTm2 | 7 | 15 phút | Điện | 96/192/336/720 |
| Exchange | 8 | Ngày | Kinh tế (tỉ giá 8 nước, 1990–2016) | 96/192/336/720 |
| Weather | 21 | 10 phút | Khí tượng (Max Planck, 2020) | 96/192/336/720 |
| ECL | 321 | 1 giờ | Điện tiêu thụ khách hàng | 96/192/336/720 |
| Traffic | 862 | 1 giờ | Độ chiếm dụng đường (vùng vịnh San Francisco) | 96/192/336/720 |
| Solar-Energy | 137 | 10 phút | Sản lượng điện mặt trời (2006) | 96/192/336/720 |
| PEMS03/04/07/08 | 358/307/883/170 | 5 phút | Giao thông California | 12/24/48/96 |
| Market (6 tập con: Merchant, Wealth, Finance, Terminal, Payment, Customer) | 285–759 | 10 phút (tập dữ liệu gốc minute-level) | Giao dịch Alipay | 12/24/72/144 |

- Chia train/validation/test **theo thứ tự thời gian** (giống TimesNet) để tránh rò rỉ dữ liệu.
- Lookback cố định 96 cho ETT, Weather, ECL, Solar-Energy, PEMS, Traffic; với Market lookback là 144 điểm (1 ngày).

#### A.2 Chi tiết cài đặt

- PyTorch, 1 GPU NVIDIA P100 16GB; optimizer **Adam**, learning rate khởi tạo $\in\{10^{-3},5\times10^{-4},10^{-4}\}$, loss L2; batch size 32; 10 epoch.
- Số block $L\in\{2,3,4\}$; chiều biểu diễn $D\in\{256,512\}$.
- Các baseline tái hiện theo repo benchmark của TimesNet với cấu hình gốc của từng mô hình.
- **Algorithm 1 (kiến trúc tổng thể), diễn giải từng bước:**
  1. Đầu vào $\mathbf{X}\in\mathbb{R}^{T\times N}$; chuyển vị thành $\mathbb{R}^{N\times T}$.
  2. Dùng MLP trên chiều cuối để nhúng chuỗi thành các variate token: $\mathbf{H}^0=\text{MLP}(\mathbf{X})\in\mathbb{R}^{N\times D}$.
  3. Lặp qua $L$ block, mỗi block: (a) $\mathbf{H}^{l-1}=\text{LayerNorm}(\mathbf{H}^{l-1}+\text{SelfAttn}(\mathbf{H}^{l-1}))$; (b) $\mathbf{H}^{l}=\text{LayerNorm}(\mathbf{H}^{l-1}+\text{FeedForward}(\mathbf{H}^{l-1}))$.
  4. Chiếu token về chuỗi dự báo: $\hat{\mathbf{Y}}=\text{MLP}(\mathbf{H}^L)\in\mathbb{R}^{N\times S}$.
  5. Chuyển vị lại thành $\mathbb{R}^{S\times N}$ và trả về.
- **Độ ổn định (Bảng 5):** chạy 5 seed ngẫu nhiên cho độ lệch chuẩn rất nhỏ (thường $\le 0.007$; lớn nhất là Exchange horizon 720: MSE $0.829\pm0.012$). Ví dụ ECL-96: $0.148\pm0.000$; Traffic-96: $0.395\pm0.001$; Weather-96: $0.174\pm0.000$.

### B. Nghiên cứu ablation đầy đủ

- Bảng 6 xác nhận kết quả của Bảng 3 theo từng độ dài dự đoán. Hai thiết kế dùng attention trên token thời gian (hàng 2 và 3) kém đáng kể, nhất là trên Traffic (MSE ~0.86–0.91 so với 0.428), do attention trên chuỗi bị trễ pha cho bản đồ vô nghĩa (xem E.3).
- Thiết kế FFN ở cả hai chiều vẫn khá ổn trên dữ liệu ít biến (Weather, 21 biến; thậm chí MSE 0.248 hơi tốt hơn), nhưng khi số biến tăng thì việc nắm tương quan đa biến trở nên quan trọng, nơi attention-trên-biến vượt trội.
- Kết luận: cả phụ thuộc thời gian và tương quan đa biến đều cần thiết; self-attention trên variate token cho phép tách các tương quan này mạnh và dễ diễn giải hơn FFN.

### C. Độ nhạy siêu tham số (Hyperparameter Sensitivity)

- Khảo sát learning rate, số block $L$, chiều ẩn $D$ (lookback = 96, horizon = 96, trên ETT, ECL, Traffic, Weather).
- Learning rate cần chọn cẩn thận khi số biến lớn (ECL, Traffic). Số block và chiều ẩn không nhất thiết càng lớn càng tốt.
- *[Hình 9: đường MSE theo learning rate, số block, hidden dimension.]*

### D. Hiệu quả mô hình (Model Efficiency)

- So sánh MSE, tốc độ huấn luyện và bộ nhớ của iTransformer, iTransformer + efficient training, iTransformer + Flowformer (iFlowformer), DLinear, TiDE, Transformer, PatchTST, Crossformer, với cùng batch size, trên Weather (21 biến) và Traffic (862 biến), input-96-predict-96.
- Weather: iTransformer hiệu quả hơn các Transformer khác (~0.88 GB, ~30 ms/iter).
- Traffic: bộ nhớ gần bằng các Transformer khác (~7.5 GB, ~265 ms/iter), nhưng huấn luyện nhanh hơn PatchTST (8.58 GB, 635 ms) và Crossformer (9.74 GB, 702 ms); Transformer gốc nhanh/nhẹ hơn vì chỉ có 96 temporal token so với 862 variate token (độ phức tạp $O(N^2)$ với $N$ là số token).
- Dùng linear attention (Flowformer) hoặc chiến lược huấn luyện 20% biến, iTransformer đạt tốc độ/bộ nhớ ngang mô hình tuyến tính (ví dụ ~1.28 GB, 91 ms/iter với efficient training); hai kỹ thuật có thể kết hợp.
- *[Hình 10: biểu đồ MSE theo thời gian huấn luyện (ms/iter) với nhãn bộ nhớ cho các mô hình trên Weather và Traffic.]*

### E. Minh hoạ (Showcases)

#### E.1 Trực quan hoá tương quan đa biến
- Dùng hệ số tương quan Pearson giữa các cặp biến:

$$\rho_{xy}=\frac{\sum_i (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_i (x_i-\bar{x})^2}\sqrt{\sum_i (y_i-\bar{y})^2}}$$

  ($x_i,y_i$: giá trị theo thời gian của hai biến; $\bar{x},\bar{y}$: trung bình.)
- Trên Solar-Energy (có sự thay đổi ngày–đêm rõ rệt nên tương quan lookback và future khác nhau): score map tầng nông giống tương quan lookback, tầng sâu giống tương quan future → cơ chế có tính diễn giải.
- Trên Market: các biến là giao diện dịch vụ thuộc các ứng dụng (App). Score map chia thành các khối rõ rệt tương ứng với nhóm ứng dụng; giá trị attention được đánh dấu phản ánh độ giống nhau của chuỗi thô (cùng App giống hơn khác App).
- *[Hình 11: 3 ví dụ Solar-Energy – tương quan thô lookback/future và score map tầng 1/tầng L.]*
- *[Hình 12: ví dụ Market – chuỗi của từng biến theo ứng dụng và bản đồ tương quan học được, có cấu trúc khối.]*

#### E.2 Trực quan hoá kết quả dự báo
- So sánh dự báo input-96-predict-96 của iTransformer, PatchTST, DLinear, Crossformer, Autoformer, Transformer trên Traffic, ECL, Weather, PEMS; iTransformer bám sát biến thiên tương lai nhất.
- *[Hình 13–16: đường dự báo so với ground truth của 6 mô hình trên Traffic, ECL, Weather, PEMS.]*

#### E.3 Rủi ro khi nhúng các điểm đa biến của một timestamp
- Minh hoạ bằng Traffic (cảm biến đường ở Los Angeles): các chuỗi tương quan mạnh nhưng có **lệch pha** (ví dụ cảm biến 1 và 2, 859 và 861), vì sự kiện như tắc đường ảnh hưởng độ chiếm dụng ở các vị trí với độ trễ khác nhau.
- Do đó attention trên temporal token (hàng 2–3 ở Bảng 6) cho kết quả kém trên Traffic: bản thân attention theo thời gian không sai, nhưng nó giả định các điểm cùng timestamp phản ánh cùng một sự kiện; khi có độ trễ nội tại, bản đồ attention trở nên vô nghĩa, trừ khi receptive field đủ lớn để học quá trình suy giảm/nhân quả.
- Rủi ro khác: các đại lượng đo khác bản chất (nhiệt độ và lượng mưa trong Weather; số lượng và tỉ lệ trong ILI).
- Variate token bền vững hơn với: sự kiện trễ pha, phép đo không nhất quán, chuỗi không đều (irregular/unevenly spaced), độ trễ hệ thống của thiết bị giám sát, và khoảng thời gian sinh/ghi nhận khác nhau giữa các chuỗi.
- *[Hình 17: các chuỗi độ chiếm dụng đường của Traffic cho thấy đồng bộ mạnh nhưng có sớm/trễ pha giữa các cảm biến.]*

### F. Kết quả đầy đủ (Full Results)

**F.1 Transformer vs iTransformer (Bảng 7, MSE trung bình):** ETT 2.750 → 0.383 (−86.1%), ECL 0.277 → 0.178 (−35.6%), PEMS 0.157 → 0.113 (−28.0%), Solar 0.256 → 0.233 (−9.0%), Traffic 0.665 → 0.428 (−35.6%), Weather 0.657 → 0.258 (−60.2%).

**F.2 Tính tổng quát của khung (Bảng 8):** Đảo chiều cải thiện nhất quán Transformer, Reformer, Informer, Flowformer, Flashformer ở mọi horizon trên ECL, Traffic, Weather (ví dụ Weather–Flowformer: 0.286 → 0.266 MSE; Transformer: 0.657 → 0.258).

**F.3 Tổng quát hoá sang biến chưa thấy (Hình 18):** kết quả đầy đủ trung bình trên 5 folder, xác nhận iTransformers huấn luyện trên 20% biến vẫn dự báo tốt toàn bộ biến so với CI-Transformers.

**F.4 Kết quả dự báo đầy đủ:**
- **PEMS (Bảng 9)**, MSE/MAE trung bình theo từng tập:

| Tập | iTransformer | PatchTST | Crossformer | TimesNet | SCINet |
|---|---|---|---|---|---|
| PEMS03 | 0.113 / 0.221 | 0.180 / 0.291 | 0.169 / 0.281 | 0.147 / 0.248 | 0.114 / 0.224 |
| PEMS04 | 0.111 / 0.221 | 0.195 / 0.307 | 0.209 / 0.314 | 0.129 / 0.241 | 0.092 / 0.202 |
| PEMS07 | 0.101 / 0.204 | 0.211 / 0.303 | 0.235 / 0.315 | 0.124 / 0.225 | 0.119 / 0.234 |
| PEMS08 | 0.150 / 0.226 | 0.280 / 0.321 | 0.268 / 0.307 | 0.193 / 0.271 | 0.158 / 0.244 |

  iTransformer đạt "1st count" 13/13 (MSE/MAE) so với 7/7 của SCINet, còn các mô hình khác gần như không thắng.
- **Long-term forecasting (Bảng 10):** iTransformer thắng nhiều nhất (1st count 16 MSE / 22 MAE) so với RLinear (6/12), PatchTST (12/11), TimesNet (3/0)… Mạnh nhất ở ECL, Traffic, Solar-Energy, Weather; trên ETT và Exchange các mô hình tuyến tính/PatchTST cạnh tranh sát hoặc nhỉnh hơn (ví dụ ETTh1 Avg: RLinear 0.446 vs iTransformer 0.454; ETTm1 Avg: PatchTST 0.387 vs iTransformer 0.407; Exchange: DLinear 0.354 vs 0.360).
- **Market (Bảng 11)**, MSE/MAE trung bình theo tập:

| Tập con | iTransformer | PatchTST | TimesNet | RLinear |
|---|---|---|---|---|
| Merchant | 0.072 / 0.147 | 0.084 / 0.171 | 0.093 / 0.184 | 0.152 / 0.247 |
| Wealth | 0.345 / 0.289 | 0.394 / 0.326 | 0.360 / 0.318 | 0.585 / 0.461 |
| Finance | 0.184 / 0.216 | 0.231 / 0.248 | 0.516 / 0.308 | 0.395 / 0.336 |
| Terminal | 0.065 / 0.150 | 0.077 / 0.179 | 0.080 / 0.179 | 0.180 / 0.286 |
| Payment | 0.072 / 0.144 | 0.084 / 0.174 | 0.105 / 0.182 | 0.143 / 0.245 |
| Customer | 0.094 / 0.150 | 0.118 / 0.180 | 0.142 / 0.191 | 0.214 / 0.261 |

  iTransformer đạt 28/27 lần xếp nhất (MSE/MAE).

### G. Thảo luận và hướng cải thiện (Discussions and Further Improvement)

**G.1 Phương pháp "architecture-free"**
- **Channel Independence (CI):** các nghiên cứu gần đây (Han et al., Li et al.) cho rằng CD (channel dependence) về lý thuyết có dung lượng cao hơn, nhưng CI cải thiện hiệu năng nhờ các benchmark nhỏ (khan hiếm mẫu). Tác giả cho rằng việc làm các biến độc lập là cần thiết, nhất là khi có rủi ro nhúng như ở E.3. Nhưng CI coi chuỗi đa biến như đơn biến → huấn luyện/suy luận chậm, khó mở rộng, và không tận dụng tương quan đa biến một cách tường minh. iTransformer giải quyết bằng kiến trúc với các module Transformer gốc.
- **RevIN và Stationarization:** các kỹ thuật chống distribution shift; trong iTransformer, vai trò tương tự do LayerNorm đảm nhiệm, và vẫn còn dư địa cải thiện đối với distribution shift.

**G.2 So với linear forecaster**
- Linear forecaster có lợi thế tự nhiên trong mô hình hoá phụ thuộc thời gian (trọng số dày cho quan hệ độc lập với phép đo giữa các điểm của cùng một biến). iTransformer mạnh ở chuỗi **nhiều chiều** với tương quan phức tạp, vì embedding giữ biến độc lập và attention khai thác tương quan. Trong trường hợp đơn biến, iTransformer thoái hoá thành một mô hình tuyến tính xếp chồng được (attention suy biến), nên còn chỗ để cải thiện mô hình hoá phụ thuộc thời gian.

**G.3 So với các Transformer khác**
- iTransformer đề xuất một góc nhìn mới về cách xem xét biến và tokenization (Hình 19: temporal token – variate token – patch token). Embedding theo timestamp có rủi ro với chuỗi đa chiều, có thể giảm nhẹ bằng cách mở rộng receptive field; patching (PatchTST/Crossformer) mịn hơn nhưng tốn tính toán hơn và có nhiễu tương tác giữa patch không căn thời gian.
- Nếu tăng cường inductive bias cho embedding (ví dụ TCN), mô hình variate token có thể xử lý ca khó mạnh hơn mà vẫn giữ số token linh hoạt.
- Hướng cải thiện dựa trên kiến trúc đảo: attention hiệu quả cho tương quan đa biến, mô hình hoá phụ thuộc thời gian có cấu trúc dưới distribution shift, variate tokenization tinh hơn, cơ chế embedding được thiết kế kỹ.
- *[Hình 19: so sánh tokenization – Transformer (temporal token), iTransformer (variate token), Crossformer/PatchTST (patch token).]*

---

## Ghi chú

- **Kiến trúc mục tiêu phổ biến:** iTransformer là một trong những backbone dự báo đa biến SOTA (cùng PatchTST, DLinear, TiDE…), nên rất có khả năng được dùng làm *target model* khi nghiên cứu membership inference attack (MIA) trên mô hình chuỗi thời gian.
- **Đơn vị "mẫu" khác với PatchTST:** iTransformer xử lý mỗi chuỗi như một token toàn cục, attention trộn thông tin *giữa các biến*; khi xây dựng membership signal cần cân nhắc rò rỉ có thể nằm ở mức cửa sổ (window) hay mức biến/token, và các biến khác trong cùng cửa sổ có thể ảnh hưởng điểm số (loss) của một biến.
- **Khả năng tổng quát sang biến chưa thấy:** kết quả huấn luyện chỉ trên 20% biến mà vẫn dự báo tốt toàn bộ biến gợi ý mô hình ít "ghi nhớ" riêng từng biến; đây là điểm đáng đối chiếu khi đánh giá mức rò rỉ (memorization) giữa biến đã thấy và chưa thấy.
- **Thiết lập huấn luyện nhỏ, dễ tái lập shadow model/reference model:** 10 epoch, batch 32, $L\in\{2,3,4\}$, $D\in\{256,512\}$, loss MSE, chia train/val/test theo thời gian – thuận lợi cho việc huấn luyện nhiều shadow model; loss MSE/MAE theo cửa sổ dự báo có thể dùng trực tiếp làm score cho các tấn công kiểu LiRA/RMIA.
- **Benchmark chuẩn:** các dataset (ECL, Traffic, Weather, ETT, Solar, PEMS) và lookback 96 là cấu hình chuẩn, giúp so sánh kết quả MIA với các nghiên cứu khác trên cùng mô hình và dữ liệu.
