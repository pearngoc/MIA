# Các cuộc tấn công suy luận thành viên (MIA) có hiệu quả trên mô hình ngôn ngữ lớn không?

*Do Membership Inference Attacks Work on Large Language Models?*

- **Tác giả:** Michael Duan\*, Anshuman Suri\* (đóng góp ngang nhau), Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, Hannaneh Hajishirzi (University of Washington, University of Virginia, Allen Institute for AI)
- **Hội nghị / năm:** COLM 2024 (arXiv:2402.07841v2)
- **Mã nguồn / dữ liệu:** thư viện MIMIR (github.com/iamgroot42/mimir)

> **Bản tóm tắt chi tiết bằng tiếng Việt (do AI viết, không phải bản dịch nguyên văn).**

---

## Tóm tắt (Abstract)

Bài báo đánh giá quy mô lớn các **membership inference attack (MIA – tấn công suy luận thành viên)** nhằm vào dữ liệu *pre-training* của các **LLM** (mô hình ngôn ngữ lớn) được huấn luyện trên bộ dữ liệu **the Pile**, kích thước từ 160M đến 12B tham số. Kết quả chính:

1. Hầu hết MIA chỉ nhỉnh hơn đoán ngẫu nhiên một chút, bất kể kích thước mô hình hay miền dữ liệu.
2. Hai nguyên nhân được đưa ra: (a) tập dữ liệu rất lớn kết hợp với số lần lặp huấn luyện (epoch) rất ít; (b) ranh giới giữa "thành viên" và "không phải thành viên" vốn mờ (fuzzy) trong ngôn ngữ tự nhiên.
3. Khi MIA có vẻ thành công ở các nghiên cứu trước, nguyên nhân thường là **distribution shift** (lệch phân phối) – ví dụ non-member lấy cùng miền nhưng khác khoảng thời gian.
4. MIA hiện có rất nhạy với thay đổi nhỏ: mẫu chỉ sửa vài token (hoặc paraphrase) bị phân loại là non-member, dù về mặt rò rỉ thông tin thì vẫn "gần như là member".
5. Nhóm công bố mã và dữ liệu thành một benchmark thống nhất chứa mọi MIA hiện có.

---

## 1. Giới thiệu (Introduction)

- **MIA** dự đoán một bản ghi có nằm trong tập huấn luyện của mô hình mục tiêu hay không. Ứng dụng: kiểm toán quyền riêng tư (privacy auditing), nghiên cứu *memorization* (ghi nhớ dữ liệu huấn luyện), phát hiện vi phạm bản quyền, phát hiện nhiễm bẩn tập kiểm thử (test-set contamination).
- Các nghiên cứu trước chủ yếu cho thấy MIA mạnh trên bộ phân loại hoặc trên mô hình ngôn ngữ đã **fine-tuning** (tinh chỉnh); hiệu quả trên dữ liệu pre-training của LLM còn ít được khảo sát.
- Đóng góp:
  - **MIMIR**: thư viện thống nhất để đánh giá MIA cho LM, cài đặt 5 tấn công phổ biến.
  - Thí nghiệm trên họ mô hình **Pythia** với dữ liệu **the Pile**: hiệu năng gần ngẫu nhiên ở đa số miền.
  - Phân tích nguyên nhân: dữ liệu lớn + gần 1 epoch làm giảm khả năng ghi nhớ; **n-gram overlap** (tỉ lệ chuỗi n từ trùng lặp) cao giữa member và non-member (ví dụ non-member của Wikipedia/ArXiv có overlap 7-gram trung bình trên 30%) khiến việc phân biệt khó. Non-member có overlap thấp hơn thì MIA phân biệt dễ hơn.
  - Hiệu năng cao ở công trình trước (Shi và cộng sự, 2023) có thể do non-member được chọn cùng miền nhưng lệch thời gian (§4).
  - Đề xuất xem lại định nghĩa "membership" cho mô hình sinh văn bản: các mẫu member bị sửa rất ít vẫn bị MIA coi là non-member với độ tin cậy cao (§5).

---

## 2. Bối cảnh (Background)

### 2.1 Bài toán MIA

- Mục tiêu: cho điểm $f(x; M)$ cho mẫu $x$ đối với mô hình $M$, rồi **thresholding** (đặt ngưỡng) để quyết định member/non-member.
- MIA là mô hình đe dọa (threat model) mặc định khi bàn về quyền riêng tư ML; có nhiều tấn công (Yeom, Carlini, Mireshghallah...) và phòng thủ (DP-SGD, v.v.).
- Cách mạnh hơn dùng **shadow model** (mô hình bóng, huấn luyện trên dữ liệu không giao với dữ liệu của mục tiêu, cùng phân phối). Tấn công như **LiRA** hiệu quả nhưng cần huấn luyện nhiều shadow model, không khả thi với LLM. Các giả định mạnh khác: truy cập white-box hoặc có tập con ground-truth để huấn luyện meta-classifier.
- Thiết lập của bài: $M$ là mô hình **auto-regressive** cho phân phối $P(x_t \mid x_1 \dots x_{t-1}; M)$; kẻ tấn công chỉ cần xác suất token (black-box).

### 2.2 Năm MIA được đánh giá

Ký hiệu $L(x;M)$ là **loss** (negative log-likelihood trung bình) của mẫu $x$ dưới mô hình $M$.

1. **LOSS** (Yeom và cộng sự, 2018): dùng thẳng loss.
$$f(x;M) = L(x;M)$$
2. **Reference-based** (tấn công dựa trên mô hình tham chiếu): hiệu chỉnh loss theo độ khó nội tại của mẫu bằng một mô hình tham chiếu $M_{ref}$.
$$f(x;M) = L(x;M) - L(x;M_{ref})$$
3. **Zlib entropy**: chia loss cho kích thước nén zlib của mẫu (xấp xỉ độ phức tạp của văn bản).
$$f(x;M) = \frac{L(x;M)}{\mathrm{zlib}(x)}$$
4. **Neighborhood attack** (Mattern và cộng sự, 2023): ước lượng "độ cong" của loss quanh $x$ bằng cách tạo $n$ mẫu lân cận $\tilde{x}_i$ (thay một phần token) rồi so sánh loss.
$$f(x;M) = L(x;M) - \frac{1}{n}\sum_{i=1}^{n} L(\tilde{x}_i;M)$$
5. **Min-k% Prob** (Shi và cộng sự, 2023): chỉ lấy trung bình trên $k\%$ token có xác suất thấp nhất.
$$f(x;M) = \frac{1}{|\text{min-}k(x)|}\sum_{x_i \in \text{min-}k(x)} -\log p(x_i \mid x_1,\dots,x_{i-1})$$

Giải thích ký hiệu: $x$ – mẫu văn bản; $M$ – mô hình mục tiêu; $M_{ref}$ – mô hình tham chiếu; $\tilde{x}_i$ – mẫu lân cận thứ $i$; $\text{min-}k(x)$ – tập $k\%$ token có likelihood thấp nhất; $p(x_i\mid\cdot)$ – xác suất token $x_i$ cho trước tiền tố.

### 2.3 MIA so với data extraction

- "Extractability" (khả năng trích xuất dữ liệu huấn luyện bằng prompt tiền tố rồi so khớp phần hậu tố) hiện gần như đồng nghĩa với memorization và hay được dùng để so sánh các mô hình.
- Hai kỹ thuật đo các loại rò rỉ khác nhau: MIA cần biết trước ứng viên và chỉ cho biết ứng viên có trong tập huấn luyện không; extraction cần tiền tố đủ dài và cần cách kiểm chứng văn bản trích ra có thật sự là dữ liệu huấn luyện.

---

## 3. Suy luận thành viên trên LLM là khó (Membership Inference on LLMs is Difficult)

### 3.0 Thiết lập thí nghiệm

- **Mô hình mục tiêu:** Pythia (160M, 1.4B, 2.8B, 6.9B/6.7B, 12B; huấn luyện trên the Pile gốc) và **Pythia-dedup** (cùng kích thước, huấn luyện trên the Pile đã khử trùng lặp – deduplicated). Kiểm tra thêm với họ **GPT-Neo** (Phụ lục A.6). Bảng 1 còn có bản 70M.
- **Mô hình tham chiếu** cho reference-based attack: StableLM-Base-Alpha-3B-v2 (chọn theo thực nghiệm, §3.1).
- **Dữ liệu:** 7 miền của the Pile – Pile-CC (web), Wikipedia, PubMed Central, ArXiv (bài báo khoa học), HackerNews (đối thoại), DM Math, GitHub – cộng thêm thí nghiệm trên toàn bộ the Pile. Member lấy từ tập train, non-member từ tập test của the Pile.
- **Tiền xử lý:** the Pile test đã được khử nhiễm ở mức tài liệu; nhóm còn khử trùng lặp bổ sung (theo Groeneveld và cộng sự, 2023); chỉ lấy tài liệu dài hơn 100 từ và cắt tối đa 200 từ.
- **Metric:** chủ yếu **AUC ROC**; thêm **TPR@low%FPR** để đánh giá ở vùng tin cậy cao; vẽ khoảng tin cậy 95%.

### 3.1 Kết quả chính

**MIA gần như ngẫu nhiên.** Ở hầu hết miền, không MIA nào vượt AUC 0,6 (ngoại lệ GitHub, xem Phụ lục B.3). Reference-based thường tốt nhất nhưng chênh lệch giữa các MIA nhỏ, khó chọn ra tấn công tốt nhất.

Bảng 1 (rút gọn) – AUC ROC trên **Pythia-dedup**, giá trị cao nhất theo từng miền/cỡ mô hình:

| Miền | 160M | 1.4B | 6.9B | 12B | MIA tốt nhất (12B) |
|---|---|---|---|---|---|
| Wikipedia | .515 | .544 | .571 | .579 | Ref |
| GitHub | .656 | .670 | .684 | .690 | zlib |
| Pile CC | .503 | .525 | .564 | .582 | Ref |
| PubMed Central | .516 | .530 | .552 | .559 | Ref |
| ArXiv | .507 | .513 | .538 | .555 | Ref |
| DM Math | .523 | .512 | .508 | .512 | Ref |
| HackerNews | .505 | .514 | .546 | .565 | Ref |
| The Pile (toàn bộ) | .511 | .521 | .549 | .558 | Ref |

(Bảng 1 gốc có đủ 5 MIA và 70M–12B; các số trên là giá trị tốt nhất mỗi hàng.)

*[Hình 1: AUC ROC của reference-based attack theo kích thước mô hình trên Wikipedia, ArXiv, HackerNews; mô hình lớn hơn cho AUC nhích lên, bản dedup thấp hơn bản thường.]*

- Hiệu năng tăng nhẹ theo kích thước mô hình (mô hình lớn dễ overfit hơn) và giảm khi dữ liệu được **deduplicate**. Tại TPR@1%FPR (Bảng 11, Phụ lục E), rò rỉ ở vùng tin cậy cao thấp (<3%, trừ GitHub) và xu hướng nhiễu hơn.
- **Khó chọn reference model:** hầu hết mô hình tham chiếu cho kết quả kém, StableLM-3B-v2 tốt nhất; việc gộp (aggregate) mọi reference model cũng không tốt. Mô hình tham chiếu lý tưởng phải cùng phân phối nhưng gần như không giao với dữ liệu huấn luyện của mục tiêu – điều khó đảm bảo khi các kho dữ liệu web chồng lấn lẫn nhau.

### 3.2 Vì sao MI khó trên LLM?

Có hai nhóm nguyên nhân: đặc thù huấn luyện LLM (§3.2.1) và sự mơ hồ vốn có của MIA (§3.2.2).

#### 3.2.1 Đặc điểm của LLM

**Kích thước dữ liệu huấn luyện.**
- Giả thuyết: kho pre-training càng lớn thì mô hình càng khái quát tốt hơn và MIA càng kém.
- Cách kiểm tra: dùng các **checkpoint** trung gian của Pythia-dedup (mỗi 5000 bước đến bước 95000, thêm bước 1000 và 99000 ≈ 1 epoch trên the Pile dedup). Non-member cố định (1000 mẫu từ test). Member của checkpoint tại bước $n$ được lấy từ các tài liệu thấy trong 100 bước gần nhất $[n-100, n]$ để loại ảnh hưởng của độ mới (recency).
- Kết quả: AUC bắt đầu gần ngẫu nhiên (mô hình mới khởi động), tăng vọt trong vài nghìn bước, rồi giảm dần khi thấy nhiều dữ liệu hơn (tỉ lệ dữ liệu/tham số tăng nên ít overfit hơn). Pythia-dedup-2.8B là ngoại lệ khi luôn gần ngẫu nhiên.

**Số epoch huấn luyện.**
- LLM thường huấn luyện ~1 epoch; trong khi các nghiên cứu MIA thành công trước đây dùng fine-tuning hoặc masked LM huấn luyện >10 epoch.
- Kiểm tra với bộ **Datablations** (các mô hình 2,8B huấn luyện trên tập con C4, thấy tổng cộng 55B token, số epoch từ 1 đến 14; bản 44 epoch bị loại).
- Kết quả: AUC tăng gần tuyến tính theo số epoch hiệu dụng. Hàm ý: huấn luyện đa epoch hoặc upsampling mạnh làm tăng rò rỉ dữ liệu.

*[Hình 2: (trái) AUC theo số bước huấn luyện trong 1 epoch: tăng vọt rồi giảm dần; (phải) AUC trên Datablations tăng khi số epoch tăng.]*

#### 3.2.2 Sự mơ hồ vốn có của MIA

Văn bản tự nhiên chứa nhiều đoạn lặp (cụm từ chung, trích dẫn, cấu trúc cú pháp đặc thù miền), dù đã khử nhiễm/khử trùng lặp. Giả thuyết: overlap giữa member và non-member càng cao thì MIA càng khó.

**Định nghĩa n-gram overlap** của non-member $x = x_1\dots x_m$ (gồm $m$ từ) đối với tập huấn luyện $D$:

$$\text{overlap}(x; D) = \frac{1}{m-n+1}\sum_{i=1}^{m-n+1}\mathbb{1}\{\exists y\in D:\; x_i\dots x_{i+n-1}\in y\}$$

trong đó $n$ là độ dài n-gram, $y$ là một tài liệu huấn luyện, $\mathbb{1}\{\cdot\}$ là hàm chỉ thị. Nói cách khác: tỉ lệ n-gram của non-member xuất hiện ở ít nhất một tài liệu huấn luyện. (Cài đặt bằng **bloom filter** chia thành 2 shard; xem Phụ lục B.1.)

**Quan sát:** overlap 7-gram trung bình của non-member so với toàn bộ the Pile train: Wikipedia 32,5%, ArXiv 39,3%, PubMed Central 41,0%, và còn cao hơn ở GitHub (76,9%), DM Math (72,8%), FreeLaw (62,3%).

*[Hình 3: phân phối overlap 7-gram của non-member ở Wikipedia, GitHub, ArXiv, HackerNews; GitHub lệch hẳn về overlap cao.]*

**Thí nghiệm kiểm chứng:** lấy lại mẫu non-member sao cho overlap 7-gram $\le 20\%$ rồi chạy lại MIA (Pythia-dedup-12B, AUC):

| MIA | Wiki gốc → ≤20% | GitHub | PubMed Central | Pile CC | ArXiv |
|---|---|---|---|---|---|
| LOSS | .516 → .666 | .678 → .878 | .506 → .780 | .516 → .574 | .527 → .787 |
| Ref | .579 → .677 | .559 → .615 | .559 → .595 | .582 → .644 | .555 → .715 |
| min-k | .517 → .644 | .683 → .890 | .512 → .792 | .521 → .578 | .530 → .734 |
| zlib | .524 → .631 | .690 → .908 | .506 → .772 | .517 → .560 | .521 → .780 |
| Ne | .520 → .612 | .660 → .877 | .497 → .737 | .514 → .566 | .519 → .773 |

**Diễn giải:**
- MIA cải thiện mạnh khi non-member "ít giống" dữ liệu huấn luyện hơn – điều này là hiển nhiên vì mô hình gán likelihood thấp hơn cho dữ liệu xa phân phối huấn luyện.
- Cách lọc này **không phải** khuyến nghị đổi benchmark: nó đưa vào độ lệch rõ rệt giữa member và non-member, đẩy bài toán sang **distribution inference** (suy luận phân phối) thay vì MI chuẩn.
- Overlap n-gram là tính chất tự nhiên của ngôn ngữ chứ không phải lỗi chia train/test của the Pile; phân phối overlap có thể dùng để đánh giá non-member ứng viên có đại diện cho miền hay không.
- Reference-based tăng ít hơn các MIA khác ở GitHub/PubMed (xem B.2): non-member overlap thấp thường là outlier của miền, nên cả mô hình tham chiếu cũng cho loss cao, triệt tiêu một phần tín hiệu.

---

## 4. Tầm quan trọng của việc chọn tập ứng viên (Importance of Candidate Set Selection)

Các công trình gần đây (Shi và cộng sự 2023; Meeus và cộng sự 2023) báo cáo AUC >0,7 trên LLM pre-trained. Nhóm xem xét việc chọn non-member theo mốc thời gian như một nguồn lệch phân phối.

**Thiết lập.** Member là mẫu trước mốc cutoff kiến thức của mô hình; non-member là mẫu sau mốc đó. Thử trên hai miền:
- **Wikipedia:** member từ dump Wikipedia trước tháng 3/2020 (the Pile); non-member là bài Wikipedia tạo từ 8/2023 trở đi (tập RealTimeData WikiText).
- **ArXiv:** member từ bài đăng trước 7/2020; non-member lấy từ các tháng 2020-08, 2021-01, 2021-06, 2022-01, 2022-06, 2023-01, 2023-06 (qua ArXiv API), xử lý giống the Pile (LaTeX → Markdown).

**Kết quả (Bảng 3, AUC trên Wikipedia lệch thời gian, Pythia-dedup):**

| Tham số | LOSS | Ref | min-k | zlib | Ne |
|---|---|---|---|---|---|
| 160M | .643 | .602 | .648 | .541 | .600 |
| 1.4B | .653 | .705 | .682 | .572 | .603 |
| 2.8B | .667 | .754 | .701 | .593 | .615 |
| 6.9B | .675 | .788 | .714 | .601 | .620 |
| 12B | .680 | .796 | .719 | .607 | .626 |

Cao hơn đáng kể so với non-member cùng khoảng thời gian (Bảng 1: ≤ .58). AUC còn tăng khi non-member lệch thời gian xa hơn (Hình 6, ArXiv; ví dụ ArXiv 2020-08 → 2023-06: .723 → .795).

*[Hình 4: phân phối overlap 7-gram của non-member Wikipedia gốc và lệch thời gian; bản lệch thời gian tập trung ở overlap thấp.]*
*[Hình 6: AUC của các MIA trên ArXiv khi non-member đến từ các tháng càng muộn; xu hướng tăng theo độ lệch thời gian.]*

**Giải thích.** Lệch thời gian có thể xem như thay đổi phân phối overlap: Wikipedia non-member gốc có overlap 7-gram TB 39,3% (như nêu trong bài) còn bản lệch thời gian chỉ 13,9%. Ở ArXiv, overlap giảm từ 39,3% xuống ~22,7% (2020-08) và ~20,5% (2023-06); tuy nhiên sự chênh nhỏ giữa hai tháng này không giải thích đủ mức tăng AUC, nên còn các yếu tố khác (đổi chủ đề, thuật ngữ mới như LLM, ngày tháng...).

**Ngưỡng bị đánh lừa (Bảng 5).** Khi chọn ngưỡng quyết định bằng non-member lệch thời gian (đạt 1/5/10% FPR), áp lên non-member *gốc* (không lệch) thì FPR thật cao hơn nhiều (ví dụ trên Wikipedia lệch thời gian, ngưỡng 5% FPR cho FPR 23,8% với LOSS; ngưỡng 1% cho 9,8%). Vậy ngưỡng đang kiểm tra "lệch thời gian" hơn là "membership".

**Khuyến nghị:** so sánh phân phối n-gram overlap của tập non-member ứng viên với một tập member giữ lại; nếu lệch về overlap thấp thì kết quả MIA cao cần được xem xét cẩn thận. Phân biệt member với non-member lệch thời gian vẫn là trò chơi suy luận thực tế, nhưng khác với MI cổ điển. Lưu ý nhóm **không** tái tạo được cấu hình của các công trình trước (thiếu dữ liệu non-member/pipeline hoặc mô hình như LLaMA không công bố dữ liệu huấn luyện), nên chưa kết luận chắc rằng khác biệt kết quả hoàn toàn do lệch thời gian.

---

## 5. Xem lại khái niệm "thành viên" (Revisiting Membership)

**Động cơ.** Trong MI chuẩn, chỉ chuỗi xuất hiện *chính xác* trong dữ liệu huấn luyện mới là member. Nhưng với mô hình sinh, một paraphrase vẫn giữ thông tin (ví dụ "Product launch in Q2, 2025" so với "Q2 2025 release") có thể quan trọng không kém đối với kẻ tấn công hoặc người kiểm toán. Ngay cả **Differential Privacy** chuẩn cũng không bảo vệ trực tiếp loại rò rỉ này vì bản ghi được kiểm tra không phải member theo nghĩa đen.

**Khoảng cách từ vựng (Lexical Distance).**
- Tạo "modified member" bằng cách thay ngẫu nhiên $n\in\{1,10,25\}$ token bằng token ngẫu nhiên từ từ vựng (20 lần thử mỗi $n$).
- Với LOSS, phân phối điểm của modified member tách rõ khỏi member/non-member (mô hình nhạy với token lạ). Với Reference-based, phân phối nằm gần cả hai (do hiệu chỉnh), nên khó phán xét.
- Lấy ngưỡng đạt 1/5/10% FPR trên member/non-member gốc rồi đếm tỉ lệ modified member (coi là non-member) bị phân loại thành member – tức FPR.

| Miền (Ref, ngưỡng 1% FPR) | n=1 | n=10 | n=25 |
|---|---|---|---|
| ArXiv | 0,1 | 0,0 | 0,0 |
| Wikipedia | 0,0 | 0,0 | 0,0 |

(ở ngưỡng 10%: ArXiv 0,7/0,3/0,2; Wikipedia 0,6/0,4/0,1. Với LOSS mọi FPR đều bằng 0.) Nghĩa là mẫu chỉ sửa 1 token đã gần như chắc chắn bị coi là non-member, dù về thông tin vẫn gần như là member.

**Khoảng cách ngữ nghĩa (Semantic Distance).**
- Dùng các "neighbor" sinh ra trong Neighborhood attack (BERT thay 5% token) làm modified member ngữ nghĩa.
- Phân phối điểm gần phân phối member hơn so với thay token ngẫu nhiên (đặc biệt với Ref), nhưng FPR vẫn thấp:

| Miền | LOSS 1% | LOSS 5% | LOSS 10% | Ref 1% | Ref 5% | Ref 10% |
|---|---|---|---|---|---|---|
| ArXiv | 0,0 | 0,8 | 2,5 | 0,7 | 1,9 | 4,0 |
| Wikipedia | 0,0 | 0,5 | 2,3 | 0,4 | 3,0 | 8,2 |

*[Hình 5: phân phối điểm LOSS/Ref của member, non-member, modified member trên ArXiv và Wikipedia; trên: thay token ngẫu nhiên (edit distance 25), dưới: neighbor ngữ nghĩa 5% token.]*

**Hướng đi:** hàm khoảng cách lý tưởng nên kết hợp cả từ vựng lẫn ngữ nghĩa; nên cân nhắc một "trò chơi MI ngữ nghĩa" trong đó láng giềng của member được định nghĩa bằng khoảng cách trong không gian embedding.

---

## 6. Kết luận (Conclusion)

- Hai khả năng giải thích MIA khó trên LLM: (1) dữ liệu để lại "dấu ấn" yếu do quy mô dữ liệu lớn và huấn luyện ~1 epoch; (2) sự giống nhau giữa dữ liệu trong và ngoài tập huấn luyện (thể hiện qua n-gram overlap) làm ranh giới mờ.
- Tấn công mạnh hơn có thể cải thiện kết quả (nhắc đến Casper và cộng sự, 2024), nhưng yếu tố thứ hai đòi hỏi phải **định nghĩa lại trò chơi membership** cho mô hình sinh, sát với rò rỉ mà kẻ tấn công/auditor quan tâm (user-level leakage, PII...).
- Trong khi chờ, cần tránh gây lệch phân phối ngoài ý muốn khi dựng tập non-member cho benchmark MIA.

---

## Phụ lục (Appendix)

### A. Chi tiết triển khai (Implementation details)

#### A.1 MIMIR
- Gói Python (kèm tài liệu và test) gồm: lớp tấn công cơ sở để dễ cài thêm MIA mới; tiện ích xử lý/lọc/cache dữ liệu; hỗ trợ nhiều mô hình làm target hoặc reference; cấu hình dạng module để chạy nhiều thí nghiệm song song không cần sửa mã. Dữ liệu có trên HuggingFace (iamgroot42/mimir). Môi trường: Python 3.9.7, PyTorch 2.0.1, GPU từ RTX6k đến A100.

#### A.2 Các mô hình mục tiêu bổ sung
- **Pythia / Pythia-dedup:** cả hai có checkpoint trung gian. Pythia-dedup huấn luyện >1 epoch trên the Pile dedup, nên dùng checkpoint `step99000` (gần mốc 1 epoch); Pythia thường dùng checkpoint cuối (≈0,9 epoch).
- **SILO (PDSW):** LM 1,3B kiến trúc LLaMA (OpenLM), huấn luyện nhiều epoch trên Open License Corpus (văn bản public domain và phần mềm giấy phép mở); miền chiếm <5% dữ liệu được upsample 3 lần (gồm HackerNews, DM Math).
- **Datablations:** tập mô hình khảo sát scaling law khi dữ liệu bị giới hạn; nhóm dùng nhóm 2,8B, thấy 55B token C4, 1–14 epoch.
- **GPT-Neo:** 125M, 1,3B, 2,7B; huấn luyện trên the Pile (~300B token), tiền thân của GPT-NeoX (kiến trúc của Pythia); khác ở tokenizer và mục tiêu TPU/GPU.
- **OLMo:** 1B (3T token) và 7B (2,5T token) huấn luyện trên Dolma; checkpoint và thứ tự dữ liệu công khai.

#### A.3 Chi tiết benchmark
- Lấy mẫu 1000 member và 1000 non-member mỗi miền (10 000 mỗi loại với toàn bộ the Pile); tài liệu >100 từ, cắt 200 từ đầu. Độ dài mẫu ảnh hưởng hiệu năng (Shi và cộng sự), nên được cố định.
- Với Datablations: member/non-member từ tập train/validation của C4.
- Khử nhiễm bổ sung: bloom filter với $n=13$, ngưỡng overlap $\le 80\%$ (cài đặt mặc định của Groeneveld và cộng sự, 2023).

**A.3.1 Benchmark kích thước dữ liệu huấn luyện.** Mỗi bước = 1024 mẫu × 2048 token (khoảng 2,1 triệu token). Non-member cố định (1000 mẫu từ the Pile test). Member của checkpoint ở bước $n$: 1000 mẫu từ tài liệu thấy trong khoảng bước $[n-100, n]$, xác định được nhờ thứ tự dữ liệu tất định của Pythia-dedup. Lý tưởng nhất là tập member cố định (nhiều lần huấn luyện), nhưng quá tốn kém.

**A.3.2 Benchmark lệch thời gian.** Non-member Wikipedia từ RealTimeData ("wikitext latest", bài tạo từ tuần 12/8/2023 đến 8/1/2024), thêm tiêu đề bài vào đầu văn bản giống định dạng the Pile. Với ArXiv: member cố định, non-member lấy theo từng tháng; chuyển LaTeX sang Markdown và lọc lỗi. Các tác giả nêu khó khăn khi tái tạo công trình trước (thiếu pipeline hoặc dữ liệu huấn luyện của mô hình như LLaMA).

*[Hình 6: xem ở §4.]*

**Bảng 5 (rút gọn) – FPR (%) trên non-member gốc khi dùng ngưỡng chọn từ non-member lệch thời gian (Pythia-dedup-12B).** Ví dụ ngưỡng 5% FPR trên "Temporal Wiki" cho FPR thật là 23,8% (LOSS), 22,8% (Ref), 24,3% (min-k), 17,6% (zlib); trên ArXiv 2023-06 với ngưỡng 10%: 26,3% (LOSS), 37,3% (Ref). FPR thực tế lớn hơn nhiều mức danh nghĩa.

#### A.4 Chi tiết các MIA
- **Công thức** giống mục 2.2. Điểm đáng chú ý:
  - **Reference-based** đi theo Watson và cộng sự (2022); tương tự bản offline của **LiRA** nhưng LiRA dùng nhiều shadow model; ở đây chỉ dùng một mô hình tham chiếu khác (giả định dữ liệu rời nhau là không thực tế).
  - **Zlib:** $\mathrm{zlib}(x)$ là số byte sau khi nén zlib.
  - **Neighborhood:** dùng **BERT** làm masking model, che 5% token (theo span), neighbor được chọn theo likelihood cao nhất; nếu loss mẫu thấp hơn rõ rệt so với neighbor thì coi là member.
  - **Min-k% Prob:** thử $k\in\{10,20,30,40,50\}$, chọn $k=20$.
- **Cách đánh giá:** 1000 lần bootstrap trên benchmark, báo cáo AUC ROC và TPR@low%FPR trung bình.
- **Meta-classifier:** không tập trung vì cần tập con member/non-member có nhãn, khó đảm bảo trong thực tế khi kho pre-training ngày càng mơ hồ.

#### A.5 Lựa chọn reference model
- Các mô hình thử: GPT-2-small (124M), DistilGPT2 (82M), OPT-1.3B, GPT-Neo-1.3B, SILO-PDSWBY (1,4B), LLaMA-7B, StableLM-Base-Alpha-3B-v2, và Pythia-dedup-1.4B (phiên bản nhỏ hơn của target, cùng kiến trúc và thứ tự dữ liệu). Aggregate = trung bình điểm các reference model (nhưng các tokenizer khác nhau nên điểm không hoàn toàn so sánh được).
- Kết quả (Bảng 6): StableLM-3B-v2 thường (không phải luôn) tốt nhất; ví dụ target Pythia-dedup-12B: Wikipedia .579 (GPT-2: .514), Pile CC .582 (GPT-2: .548), HackerNews .565; riêng ArXiv thì Pythia-1.4B làm reference cho .581, cao hơn StableLM (.555). Giả thuyết về lý do StableLM tốt: nó huấn luyện trên kho rất lớn nên khái quát tốt và cho loss gần với target trên non-member, trong khi target lớn (12B) có thể overfit một số member, làm khoảng cách loss giữa member lớn hơn khoảng cách ở non-member.

#### A.6 Kết quả với GPT-Neo
- Xu hướng tương tự Pythia (Bảng 7): AUC tăng theo kích thước mô hình, thường bằng hoặc thấp hơn Pythia-dedup cùng cỡ. Ví dụ GPT-Neo-2.7B: GitHub zlib .712, Wikipedia Ref .545, The Pile Ref .519; đa số miền <.55. Ở HackerNews MIA tốt nhất khác nhau (min-k cho GPT-Neo, Ref cho Pythia-dedup), nhưng chênh lệch nhỏ.

#### A.7 Kết quả sơ bộ với OLMo
- Benchmark dựng từ các miền của Dolma (Wikipedia, C4, Reddit, Common Crawl, peS2o gồm s2ag – abstract và s2orc – bài đầy đủ); non-member lấy từ dữ liệu giữ lại PALOMA.
- Bảng 8: AUC vẫn gần ngẫu nhiên cho cả 1B và 7B, hai cỡ mô hình hội tụ về cùng mức khi dữ liệu huấn luyện cực lớn (3T/2,5T token). Reference-based với StableLM kém hơn nhiều so với khi dùng cho Pythia (khó chọn reference phù hợp cho từng LLM). Nhiều trường hợp AUC <0,5 (ví dụ Common Crawl Ref .412/.410; s2ag Ref .376) nghĩa là MIA hay dự đoán ngược; cần nghiên cứu thêm.

---

### B. Chi tiết n-gram overlap và các bài học

#### B.1 Đo n-gram overlap
- Dùng bloom filter (Groeneveld và cộng sự, 2023). Do bộ nhớ có hạn, chia dữ liệu huấn luyện làm hai nửa, mỗi nửa một shard; một n-gram tính là "có mặt" nếu có ở ít nhất một shard. Tỉ lệ dương tính giả của mỗi shard <1% (0,6%). Tokenize ở mức từ, trượt cửa sổ n từ với bước 1; cách này dùng cho cả việc thêm dữ liệu huấn luyện lẫn kiểm tra non-member.

#### B.2 Hiệu năng của reference-based attack
- Với ngưỡng overlap, Ref tăng ít hơn MIA khác ở GitHub/PubMed Central vì non-member overlap thấp là outlier của miền, nên cũng là outlier với mô hình tham chiếu; hiệu loss của target và reference ở non-member vẫn lớn, thu hẹp khác biệt so với member. Với Pile CC thì ngưỡng 20% không đủ chọn ra outlier vì miền này vốn overlap thấp.
- Với Wikipedia/ArXiv lệch thời gian, Ref vẫn tốt vì StableLM được huấn luyện trên dữ liệu mới hơn (ví dụ RedPajama dùng dump Wikipedia 3/2023), khái quát tốt trên non-member lệch thời gian.
- Phân biệt hai họ MIA: *calibration-free* (LOSS, Min-k%) chỉ dựa vào mô hình mục tiêu; *calibration-based* (zlib, Ref) hiệu chỉnh độ khó. Khi non-member quá ngoài miền, tín hiệu mục tiêu và hiệu chỉnh có thể triệt tiêu nhau; ngược lại hiệu chỉnh khuếch đại tín hiệu khi member dễ bị ghi nhớ (ít dữ liệu, nhiều epoch). Do đó **baseline của MIA mới nên gồm cả hai họ**, để lộ ra các đặc điểm ẩn như lệch phân phối hoặc overfit.

#### B.3 GitHub là ngoại lệ
- GitHub có AUC cao (~0,70 với zlib) ngay cả khi không lọc overlap. Giả thuyết: không phải vì code dễ tấn công, mà vì ngưỡng khử nhiễm ≤80% overlap 13-gram chỉ loại một phần nhỏ non-member trong miền vốn overlap rất cao (code lặp: bản quyền, định nghĩa hàm, thẻ HTML), nên non-member còn lại đa phần là outlier của miền.

*[Hình 7: phân phối overlap 7-gram của GitHub non-member trước/sau khử nhiễm 13-gram ≤80%; sau lọc lệch về outlier.]*
*[Hình 15: ví dụ non-member GitHub outlier – một mẫu văn bản tiếng Miến Điện (tin tức) trong repo tài nguyên ngôn ngữ, rõ ràng không phải code.]*

Kết luận: ranh giới non-member bằng từ vựng gây khó diễn giải hiệu năng ở miền overlap cao; khác biệt ngữ nghĩa có thể phù hợp hơn để vạch ranh giới.

#### B.4 Lệch thời gian như thay đổi n-gram overlap
- Hình 8: non-member ArXiv lệch thời gian có phân phối overlap tập trung ở mức thấp hơn nhiều so với non-member gốc (TB 39,3% → 22,7% cho 2020-08 và 20,5% cho 2023-06). Nhưng chênh giữa 2020-08 và 2023-06 nhỏ, không giải thích đủ AUC .723 → .795; các yếu tố khác có thể là thay đổi chủ đề (LLM trở nên phổ biến) và token đặc trưng (ngày tháng, tham chiếu, thuật ngữ mới).

*[Hình 8: phân phối overlap 7-gram của ArXiv non-member gốc và hai tháng lệch thời gian.]*

---

### C. Đặc điểm huấn luyện LLM (bổ sung)

#### C.1 Độ mới của member (recency)
- Cố định mô hình (Pythia-dedup, checkpoint step99000) và đổi khoảng bước chứa member. Member thấy gần đây hơn cho AUC cao hơn một chút; AUC giảm dần rồi chạm plateau với member thấy sớm hơn; với mô hình nhỏ giảm nhanh hơn, mô hình lớn giữ được lâu hơn; các đường hội tụ ở dữ liệu rất sớm. Điều này khớp với nghiên cứu về forgetting (Jagielski và cộng sự, 2023).
- Hàm ý cho fine-tuning/continued pre-training: dữ liệu thấy gần đây dễ bị lộ hơn – quan trọng khi tinh chỉnh bằng dữ liệu nhạy cảm.

*[Hình 9: AUC theo bước thấy member (Pythia-dedup-12B step99000), cao hơn cho member gần đây, giảm dần đến plateau.]*

#### C.2 Số epoch (SILO-PDSW)
- Thiết lập thực tế hơn: dữ liệu tổng và số epoch hiệu dụng tăng cùng lúc.
- HackerNews: AUC lúc đầu tăng theo epoch rồi plateau hoặc giảm. Giải thích: miền chỉ chiếm 5,9% dữ liệu; vài epoch đầu mô hình ghi nhớ được, về sau (mô hình 1,4B nhỏ) ghi nhớ ít hơn các mẫu thuộc miền chiếm tỉ lệ nhỏ.
- DM Mathematics: AUC giảm khi số epoch tăng (miền chỉ 3,5%; văn bản thiên về ký hiệu/chữ số, khó ghi nhớ từng mẫu).

*[Hình 10: AUC trên SILO-PDSW theo số epoch hiệu dụng; HackerNews tăng rồi bão hòa, DM Math giảm.]*

---

### D. Xem lại membership qua tương đồng ngữ nghĩa

- Dùng **GPT-4** paraphrase member (5 lần mỗi mẫu) theo Yang và cộng sự (2023), trên ArXiv, Wikipedia, HackerNews, với prompt yêu cầu giữ chi tiết cụ thể (công thức, tiêu đề, liên kết...) và số từ tương đương; riêng HackerNews đổi cấu trúc bình luận nên khác từ vựng rõ.
- Kết quả: phân phối điểm của paraphrase tách khỏi member/non-member gốc nhưng chồng lấn đáng kể (giống neighbor dựa trên masking). Ở vùng tin cậy cao, paraphrase phần lớn bị coi là non-member (Bảng 10, FPR %):

| Miền | LOSS 1% | LOSS 5% | LOSS 10% | Ref 1% | Ref 5% | Ref 10% |
|---|---|---|---|---|---|---|
| ArXiv | 0,1 | 0,2 | 0,5 | 0,2 | 0,7 | 1,7 |
| Wikipedia | 0,0 | 0,0 | 0,2 | 0,0 | 0,2 | 0,7 |
| HackerNews | 0,1 | 1,1 | 1,7 | 0,0 | 0,1 | 0,2 |

*[Hình 11: phân phối điểm LOSS/Ref của member, non-member và member được GPT-4 paraphrase trên ba miền.]*

Cả LOSS và Ref đều kém nhạy với paraphrase.

---

### E. Bảng và hình bổ sung

- **Bảng 11 – %TPR@1%FPR trên Pythia-dedup:** hầu hết dưới 3% (ví dụ The Pile 12B: LOSS 2,7, Ref 1,8, min-k 2,6, zlib 2,6, Ne 2,2); GitHub là ngoại lệ (12B: 13,9/0,8/14,2/17,4/4,9). Xu hướng nhiễu hơn so với AUC (kích thước mô hình và MIA tốt nhất không còn nhất quán).
- **Bảng 12 – AUC trên Pythia (không dedup):** cùng xu hướng, nhìn chung cao hơn một chút so với dedup (ví dụ 12B: Wikipedia Ref .639, HackerNews Ref .606, GitHub zlib .748, The Pile Ref .569). Không chạy Neighborhood vì giới hạn tính toán.
- *[Hình 12: AUC theo kích thước mô hình cho LOSS, Min-k%, Zlib: cùng xu hướng như Hình 1 (lớn hơn → nhỉnh hơn; dedup → thấp hơn).]*
- *[Hình 13: AUC theo số bước huấn luyện cho các MIA còn lại: tăng vọt rồi giảm dần.]*
- *[Hình 14: phân phối overlap n-gram (n = 4, 7, 13) trên mọi miền đánh giá.]*

---

## Ghi chú

- **Cảnh báo về benchmark:** kết quả MIA cao trên mô hình ngôn ngữ có thể phản ánh lệch phân phối (thời gian, chủ đề, độ dài) giữa member và non-member hơn là ghi nhớ thật; với mô hình chuỗi thời gian, khi chia member/non-member theo thời gian cũng cần kiểm tra tương tự.
- **Yếu tố ảnh hưởng độ rò rỉ:** kích thước dữ liệu, số epoch, kích thước mô hình, deduplication và độ mới của dữ liệu – các biến này đáng được kiểm soát khi đánh giá MIA trên mô hình chuỗi thời gian, nhất là mô hình nhỏ huấn luyện nhiều epoch, nơi mức overfit có thể khác xa LLM pre-training.
- **Phương pháp đánh giá đáng tái sử dụng:** AUC kèm TPR@low%FPR, bootstrap, so sánh tấn công có/không hiệu chỉnh (calibration-based và calibration-free) và kiểm tra overlap để đánh giá non-member có đại diện không.
- **Khó chọn reference/shadow model:** việc LiRA/reference attack không thể mở rộng và khó chọn mô hình tham chiếu cùng phân phối nhưng rời dữ liệu là thách thức cũng áp dụng cho các mô hình chuỗi thời gian có dữ liệu chồng lấn (cửa sổ trượt, chuỗi liền kề).
- **Membership mờ:** các mẫu "gần member" (ví dụ cửa sổ chuỗi thời gian dịch vài bước, chuỗi bị nhiễu nhẹ) có thể bị MIA coi là non-member dù vẫn rò rỉ thông tin – gợi ý cần định nghĩa membership theo khoảng cách cho dữ liệu chuỗi thời gian.
