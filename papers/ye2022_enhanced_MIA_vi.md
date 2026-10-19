# Các tấn công suy luận thành viên được tăng cường nhằm vào mô hình học máy

*Enhanced Membership Inference Attacks against Machine Learning Models*

- **Tác giả:** Jiayuan Ye, Aadyaa Maddi, Sasi Kumar Murakonda, Vincent Bindschaedler, Reza Shokri
- **Đơn vị:** National University of Singapore, Privitar Labs, University of Florida
- **Venue/năm:** arXiv:2111.09679v4 (cs.LG), 13/09/2022 (bản được đăng ở ACM CCS 2022)
- **Mã nguồn:** công cụ Privacy Meter (`ml_privacy_meter/research/2022_enhanced_mia`)

> *Bản tóm tắt chi tiết bằng tiếng Việt (do AI viết, không phải bản dịch nguyên văn).*

---

## Tóm tắt (Abstract)

Bài báo hỏi: một thuật toán học máy làm lộ bao nhiêu thông tin về dữ liệu huấn luyện, và vì sao? Membership inference attack (MIA – tấn công suy luận thành viên) được dùng như công cụ kiểm toán (audit) để đo mức rò rỉ này. Nhóm tác giả đề xuất:

- Một **khung kiểm định giả thuyết (hypothesis testing)** thống nhất, giúp diễn đạt các tấn công trước đó theo cùng một ngôn ngữ và thiết kế tấn công mới dùng **reference model** (mô hình tham chiếu) để đạt TPR cao hơn tại mọi mức FPR.
- Một **mẫu trò chơi không phân biệt được (indistinguishability game)**, trong đó ta chỉ rõ nguồn ngẫu nhiên nào được cố định, từ đó giải thích ý nghĩa của "tỉ lệ thành công của tấn công".
- Phân tích các loại **bất định (uncertainty)** của kẻ tấn công và cách các tấn công mới giảm bất định đó xuống còn đúng 1 bit bí mật: điểm dữ liệu có nằm trong tập huấn luyện hay không.
- Phân tích vi sai (differential analysis) giữa các tấn công để giải thích vì sao có điểm dữ liệu dễ bị tấn công này nhưng không dễ bị tấn công khác (từ overfitting đến conditional memorization).

---

## 1. Giới thiệu (Introduction)

**Bối cảnh.** Các cơ quan quản lý (ICO của Anh, NIST của Mỹ) xem khả năng bị MIA là một nguy cơ vi phạm bảo mật dữ liệu huấn luyện. MIA cũng gắn chặt với differential privacy (DP – quyền riêng tư vi sai), nên được dùng cho privacy auditing và đánh giá tác động bảo vệ dữ liệu (DPIA).

**Ba hạn chế của các nghiên cứu trước:**

1. *Thiếu nhất quán/khả năng so sánh:* các tấn công được báo cáo bằng độ chính xác thực nghiệm, nhưng ý nghĩa của "thành công" phụ thuộc giả định riêng của từng bài (ví dụ kẻ tấn công biết gì về quần thể dữ liệu). Các cách chia thô thành "rò rỉ trung bình" và "rò rỉ xấu nhất" mà không chỉ rõ thí nghiệm ngẫu nhiên gây thêm nhầm lẫn.
2. *Thiếu giải thích và hướng dẫn thiết kế:* ít có lời giải thích hình thức vì sao một chiến lược tấn công thành công hay thất bại. Một số phân tích kết luận rò rỉ chỉ do overfitting, số khác lại cho rằng do ghi nhớ các điểm khó; bài báo cho rằng có nhiều nguyên nhân đồng thời, mỗi tấn công bắt được một phần.
3. *Thiếu báo cáo chi tiết:* chỉ báo accuracy là không đủ; cần đường cong power–error.

**Ý tưởng cốt lõi.** Với một điểm dữ liệu $x$, chia các "thế giới khả dĩ" thành thế giới **in** (x thuộc tập huấn luyện) và **out** (x không thuộc). Ta dựng một thí nghiệm ngẫu nhiên: chọn $x$, lấy mẫu một cặp thế giới in/out, đặt kẻ tấn công vào một thế giới ngẫu nhiên và thách đoán thế giới đó chỉ dựa vào $x$ và mô hình $\theta$ đã huấn luyện. Lặp lại nhiều lần, đo **power** (TPR – true positive rate) và **error** (FPR – false positive rate; "positive" ứng với thế giới in). **Mức rò rỉ** của thuật toán chính là đường cong power-so-với-error trên toàn bộ các kẻ tấn công.

**Ý nghĩa của rò rỉ phụ thuộc cách đặt thí nghiệm:**
- Cố định điểm đích $x$, tập huấn luyện hai thế giới chỉ khác $x$ ⇒ đo rò rỉ trung bình về $x$ (trung bình trên các điểm khác và ngẫu nhiên của thuật toán).
- Cố định cả các điểm còn lại (do kẻ tấn công chọn) ⇒ rò rỉ có điều kiện (conditional leakage) về $x$.
- Lấy mẫu mới mọi thứ ở mỗi lần ⇒ rò rỉ trung bình trên mọi điểm dữ liệu.

**Đóng góp chính:**
- Khung hypothesis testing + mẫu trò chơi; biểu diễn các tấn công trước đó trong khung này.
- **Attack S** (dựa trên shadow model, hình thức hóa [Shokri et al. 2017]).
- **Attack P** (population-based): công suất tương đương hoặc tốt hơn Attack S nhưng không cần huấn luyện shadow model, chi phí thấp hơn nhiều.
- **Attack R** (reference-model): vượt qua tấn công cùng lớp của Carlini et al. (LiRA) trên mọi chỉ số được so sánh (theo Hình 6).
- **Attack D** (self-distillation): tấn công mới phụ thuộc cả mô hình lẫn mẫu, nhằm loại bỏ bất định của kẻ tấn công về dữ liệu huấn luyện của mô hình đích; cho AUC cao hơn và FPR thấp hơn tại TPR cao.
- Phân tích vi sai: tìm các tập điểm dễ bị tổn thương với tấn công này mà không với tấn công kia; với tập điểm tìm được, AUC của tấn công mạnh nhất lên tới **0,984** (Hình 4). Giải thích bằng phân bố loss và ảnh hưởng của hàng xóm gần nhất (cosine distance trên đầu ra lớp áp chót).

---

## 2. Công trình liên quan (Related Work)

- **Phân tích rủi ro quyền riêng tư bằng MIA:** bắt đầu từ dữ liệu genome [Homer et al.], sau đó có phân tích hình thức khi công bố thống kê tổng hợp trên thuộc tính độc lập, rồi mạng Bayes với thuộc tính phụ thuộc, rồi thống kê có nhiễu khi kẻ tấn công chỉ có một mẫu tham chiếu. Các kết quả này cho biết rủi ro tăng theo số thống kê được công bố và số cá nhân trong tập dữ liệu.
- **Differential privacy và MIA:** DP, theo định nghĩa, chặn thành công của MIA giữa hai tập dữ liệu lân cận. Nhiều công trình đưa ra cận trên trung bình của MIA theo tham số DP; một số khác dùng thành công thực nghiệm của MIA để suy ra cận dưới cho đảm bảo DP. Điều này đòi hỏi MIA mạnh hơn, đo rò rỉ về một điểm cụ thể.
- **MIA cho mô hình ML:** Shokri et al. đưa ra shadow model trong thiết lập black-box; về sau mở rộng sang white-box, label-only, federated learning. Phần lớn dùng công thức trò chơi của Yeom et al., tập trung vào thành viên/không thành viên "trung bình", nên không phản ánh rủi ro của từng bản ghi.
- **Vượt xa hiệu năng trung bình:** một số công trình giới hạn vào tập con bản ghi (heuristic) hoặc tập dữ liệu bị đầu độc (poisoned neighbors) và thấy thành công cao hơn hẳn. Các tấn công theo từng mẫu có hiệu chỉnh độ khó (difficulty calibration, ví dụ Carlini et al., Watson et al.) rất gần Attack R, nhưng chưa giải thích hình thức vì sao các bản ghi "khó" lại bị lộ và vì sao tấn công của họ tốt hơn. Bài báo này điều tra vì sao các tấn công khác nhau tìm thấy các điểm dễ tổn thương khác nhau.

---

## 3. Khung tấn công (Attack Framework)

Mục tiêu: kiểm toán rò rỉ quyền riêng tư của mô hình về một bản ghi cụ thể, trong thiết lập **black-box** (chỉ thấy đầu ra). Khung gồm ba phần: (i) trò chơi suy luận làm thiết lập đánh giá; (ii) thước đo không phân biệt được (indistinguishability) để đo rủi ro; (iii) xây dựng MIA như một phép kiểm định giả thuyết.

### 3.1 Trò chơi suy luận (Inference Game)

Ký hiệu: $\pi$ là kho dữ liệu quần thể (population pool), $\mathcal{T}$ là thuật toán huấn luyện, $\theta$ là mô hình, $z=(x,y)$ là bản ghi, $b$ là bit bí mật, $\mathcal{A}$ là kẻ tấn công. Các seed $s_D, s_\theta, s_{z}$ điều khiển nguồn ngẫu nhiên nào bị cố định.

**Định nghĩa 3.1 – Mô hình và bản ghi trung bình (average model and record).** 
1. Challenger lấy $D \sim \pi^n$ (seed $s_D$ mới) và huấn luyện $\theta \leftarrow \mathcal{T}(D)$ (seed $s_\theta$ mới).
2. Lấy $z_0 \sim \pi$ (non-member, gần như chắc chắn $z_0\notin D$ khi $\pi$ lớn).
3. Lấy $z_1 \sim D$ (member).
4. Tung đồng xu $b \in \{0,1\}$ và gửi $(\theta, z_b)$ cho kẻ tấn công.
5. Kẻ tấn công có quyền truy cập $\pi$ và $\theta$, đưa ra $\hat b = \mathcal{A}(\theta, z_b)$.
6. Thành công nếu $\hat b = b$.

Trò chơi này tương tự Yeom et al., Sablayrolles et al., Carlini et al.: cả mô hình lẫn bản ghi đều ngẫu nhiên, nên chỉ đo rò rỉ trung bình. Ba biến thể sau thu hẹp "thế giới out":

| Định nghĩa | Cố định gì | Ý nghĩa |
|---|---|---|
| 3.2 – Mô hình cố định | $s_D, s_\theta$ cố định (một mô hình đích duy nhất trên một tập dữ liệu cố định); $z_0, z_1$ lấy mẫu mới | Rò rỉ của một mô hình cụ thể (phổ biến trong ML-as-a-service) |
| 3.3 – Bản ghi cố định | $s_z$ cố định; mô hình lấy mẫu mới: $\theta_0\leftarrow\mathcal T(D)$, $\theta_1\leftarrow\mathcal T(D\cup\{z\})$ | Phân biệt mô hình huấn luyện có/không có một bản ghi cụ thể (phần dữ liệu còn lại ngẫu nhiên) |
| 3.4 – Bản ghi và tập dữ liệu xấu nhất cố định | Cả $s_D$ và $s_z$ cố định | Giống rò rỉ trong định nghĩa DP; dùng để audit thuật toán DP |

### 3.2 Thước đo không phân biệt được (Indistinguishability Metric)

Rò rỉ thật của mô hình $\theta$ (huấn luyện trên $D$) về $z_b$ là mức độ mà một mô hình $\theta'$ huấn luyện trên cùng tập nhưng bỏ $z_b$ có thể phân biệt được với $\theta$. Việc đoán $b=0$ hay $1$ là một kiểm định giả thuyết; lỗi gồm false positive (nhận non-member là member) và false negative.

Sai số của kẻ tấn công trong thực tế phụ thuộc vào nhiều yếu tố:
- **Rò rỉ thật** của mô hình về $z_b$ (khi $b=1$).
- **Bất định về quần thể dữ liệu:** hiểu biết nền của kẻ tấn công về phân phối dữ liệu; yếu tố này không liên quan đến rò rỉ do công bố mô hình nhưng lại bị đo lẫn.
- **Bất định về thuật toán huấn luyện** $\mathcal T$ (ví dụ tấn công thiết kế cho cross-entropy có thể kém với 0-1 loss).
- **Sự phụ thuộc của cách xây dựng tấn công vào mục tiêu cụ thể** $(x_b,y_b)$ và $\theta$: dùng cùng một chiến lược cho mọi mục tiêu sẽ kém trên các mục tiêu không điển hình.
- **Bất định về toàn bộ dữ liệu huấn luyện còn lại:** trường hợp cực đoan, kẻ tấn công biết hết các bản ghi còn lại và thực hiện leave-one-out attack, về nguyên tắc là mạnh nhất (Mục 4.5).

**Nguyên tắc thiết kế:** lý tưởng nhất, sai số chỉ phụ thuộc vào rò rỉ thật. Muốn vậy, dữ liệu quần thể dùng để xây tấn công và đánh giá phải *cùng phân phối* với dữ liệu huấn luyện; nếu vi phạm có thể đánh giá quá cao (ví dụ mọi member màu đỏ, non-member màu xanh) hoặc quá thấp rủi ro.

**Yêu cầu báo cáo:** phải báo đường cong error-vs-power (TPR theo FPR) chứ không chỉ accuracy; diện tích dưới đường cong (AUC) phản ánh xác suất suy luận đúng tư cách thành viên của một điểm ngẫu nhiên.

---

## 4. Xây dựng tấn công suy luận thành viên black-box

**Giả định:**
1. *Black-box:* chỉ truy cập hàm đầu ra của mô hình.
2. *Biết quần thể:* có thể lấy mẫu từ kho dữ liệu quần thể.
3. *Biết thuật toán huấn luyện:* có thể lấy mẫu mô hình đã huấn luyện trên tập dữ liệu bất kỳ qua $\mathcal T$ ngẫu nhiên.

**Tín hiệu:** chỉ dùng giá trị **loss** $\ell(\theta,x_z,y_z)$, vì dưới công thức kiểm định giả thuyết nhị phân trên mục tiêu tổng quát, kiểm định tỉ số khả dĩ (LRT – likelihood ratio test) xấp xỉ so loss với một ngưỡng hằng.

**Lemma 4.1 (LRT xấp xỉ cho mục tiêu tổng quát).** Với hai giả thuyết

$$H_0:\ D\overset{\text{i.i.d.}}{\sim}\pi^n,\ \theta\sim\mathcal T(D),\ z\sim\pi \qquad H_1:\ D\overset{\text{i.i.d.}}{\sim}\pi^n,\ \theta\sim\mathcal T(D),\ z\sim D$$

dưới một số giả định (Phụ lục A), LRT xấp xỉ quy tắc:

$$\text{Nếu } \ell(\theta,x_z,y_z)\le c,\ \text{bác bỏ } H_0 \tag{3}$$

với $c$ là ngưỡng hằng tùy ý. Đây **không phải** bảo đảm tối ưu (giả định có thể không đúng, xấp xỉ chỉ chính xác khi tập dữ liệu đủ lớn), mà là lời giải thích vì sao tấn công theo ngưỡng loss mạnh về nguyên tắc.

**Mẫu tổng quát cho chiến lược phụ thuộc mục tiêu.** Vì ngưỡng hằng quá chung, tác giả cho ngưỡng phụ thuộc mô hình và/hoặc bản ghi:

$$\text{Nếu } \ell(\theta,x_z,y_z)\le c_\alpha(\theta,x_z,y_z),\ \text{bác bỏ } H_0 \tag{4}$$

$c_\alpha$ là hàm ngưỡng thỏa mức FPR $\alpha\in[0,1]$. Ngưỡng càng phụ thuộc mục tiêu thì "thế giới out" càng nhỏ (bất định của kẻ tấn công càng giảm).

**Bảng 1 – Các tấn công, ngưỡng và trò chơi tương ứng:**

| Tấn công | Hàm ngưỡng | Phụ thuộc vào | Trò chơi |
|---|---|---|---|
| S (shadow) | $c_\alpha(y_z)$ | Nhãn bản ghi | Định nghĩa 3.1 |
| P (population) | $c_\alpha(\theta)$ | Mô hình (tập dữ liệu) | Định nghĩa 3.2 |
| R (reference) | $c_\alpha(x_z,y_z)$ | Bản ghi | Định nghĩa 3.3 |
| D (distillation) | $c_\alpha(\theta,x_z,y_z)$ | Bản ghi và mô hình | Định nghĩa 3.4 |

**Phương pháp chung xác định ngưỡng.** Muốn ngưỡng (1) có mức phụ thuộc mục tiêu mong muốn, và (2) cho dự đoán tự tin bảo đảm FPR thấp, tác giả tính ngưỡng theo yêu cầu FPR $\alpha$ bất kỳ bằng cách lấy **percentile** của phân phối loss trong từng thế giới out:

$$\mathbb{E}_{P_{out}(D,\theta,z)}\Big[\mathbf 1_{\ell(\theta,x_z,y_z)\le c_\alpha(\theta,x_z,y_z)}\Big]=\alpha \tag{5}$$

Các bước: (1) dựng histogram loss thực nghiệm của thế giới out tương ứng; (2) làm mượt (smoothing, chi tiết ở Phụ lục B.3) để ước lượng CDF; (3) lấy $\alpha$-percentile làm ngưỡng. Nhờ vậy ngưỡng đảm bảo xấp xỉ FPR $\alpha$ *trước khi* thấy dữ liệu đánh giá. Đây là điểm khác biệt với các tấn công trước: Shokri et al. và các công trình shadow sau đó xây một "tấn công đơn" tối ưu TPR+TNR theo heuristic; Sablayrolles et al. làm tương tự nhưng chặt chẽ hơn (nghiệm Bayes tối ưu); Yeom và Carlini et al. chỉ có quy tắc phát hiện và quét ngưỡng để vẽ đường cong; kẻ tấn công thực tế không quét ngưỡng được, và ngưỡng không có quan hệ đơn giản với FPR.

### 4.1 Attack S: MIA qua shadow model

Đây là baseline mạnh hình thức hóa từ Shokri et al. Quy tắc: nếu $\ell(\theta,x_z,y_z)\le c_\alpha(y_z)$ thì bác bỏ $H_0$. Ngưỡng hằng trên mọi mô hình và đặc trưng $x_z$ (với nhãn cố định), nên ứng với trò chơi tổng quát (Định nghĩa 3.1). Thế giới out:

$$P_{out}(D,\theta,z):\ D\overset{\text{i.i.d.}}{\sim}\pi^n,\ \theta\leftarrow\mathcal T(D),\ z\sim\pi^{y_z=y_0} \tag{6}$$

Thực tế: xấp xỉ bằng tập các cặp (shadow model, điểm shadow) $S_{y}=\bigcup_i\{(\theta_i,z_{i1}),(\theta_i,z_{i2}),\dots\}$ với $\theta_i\leftarrow\mathcal T(D_i)$, $D_i\sim\text{pop}$, $z_{ij}\sim\pi^{y_z=y_0}$. Ngưỡng $c_\alpha(y_0)$ chọn sao cho đúng $\alpha$ tỉ lệ cặp trong $S_{y_0}$ có loss nhỏ hơn hoặc bằng ngưỡng (phương trình 7). Ngưỡng này **không phụ thuộc $x_z$ hay $\theta$**, chỉ giảm bất định về nhãn.

### 4.2 Attack P: MIA phụ thuộc mô hình qua dữ liệu quần thể

Ý tưởng: dùng ngưỡng riêng $c_\alpha(\theta)$ cho từng mô hình đích, tính trực tiếp trên mô hình đích (không cần shadow model) bằng cách đo loss của mô hình đích trên các điểm quần thể. Kỹ thuật tương tự dùng trong genomics nhưng theo tác giả chưa từng dùng cho MIA trên ML. Thế giới out cố định mô hình và lấy mẫu bản ghi từ quần thể:

$$P_{out}(D,\theta,z):\ D=D_0,\ \theta=\theta_0,\ z\sim\pi \tag{8}$$

Xấp xỉ bằng tập $P_{\theta_0}=\{(\theta_0,z_i)\}$ với $z_i\sim\pi$. Ngưỡng $c_\alpha(\theta_0)$ là $\alpha$-percentile của histogram loss của các điểm quần thể trên mô hình $\theta_0$ (phương trình 9). Ngưỡng không phụ thuộc $(x_z,y_z)$, nên Attack P chỉ giảm bất định về mô hình đích, không giảm bất định về bản ghi.

### 4.3 Attack R: MIA phụ thuộc mẫu qua reference model

Động cơ: rò rỉ của mô hình về bản ghi liên quan đến mức độ bản ghi dễ bị ghi nhớ (ví dụ là outlier). Attack R dùng ngưỡng $c_\alpha(x_z,y_z)$ riêng cho từng bản ghi, dựa trên nhiều **reference model** huấn luyện trên các tập quần thể không chứa bản ghi đích, rồi đo loss của chúng trên đúng bản ghi đó (tương tự Long et al. và các tấn công cho thống kê tóm tắt/mô hình đồ thị). Thế giới out:

$$P_{out}(D,\theta,z):\ D\overset{\text{i.i.d.}}{\sim}\pi^n,\ \theta\leftarrow\mathcal T(D),\ z=z_0=(x_{z_0},y_{z_0}) \tag{10}$$

Xấp xỉ bằng $R_{z_0}=\{(\theta_i,z_0)\}$ với $\theta_i\leftarrow\mathcal T(D_i)$, $D_i\sim\pi^n$. Ngưỡng $c_\alpha(x_z,y_z)$ là $\alpha$-percentile của loss của $z$ trên các reference model (phương trình 11). Thế giới out này **nhỏ hơn** của Attack S (đã biết cả đặc trưng $x_{z_0}$ chứ không chỉ nhãn) nhưng ngưỡng không phụ thuộc mô hình đích $\theta$.

### 4.4 Attack D: MIA phụ thuộc cả mô hình và mẫu qua distillation

Câu hỏi: có thể thiết kế tấn công tận dụng *mọi* thông tin có từ mô hình đích và bản ghi đích (dù chỉ black-box + dữ liệu quần thể) không? Attack D dùng ngưỡng

$$\text{Nếu } \ell(\theta,x_z,y_z)\le c_\alpha(D_\theta,x_z,y_z),\ \text{bác bỏ } H_0 \tag{12}$$

với $D_\theta=\mathcal T^{-1}(\theta)$ là tập huấn luyện (không biết) của mô hình đích. Giả định (đơn giản hóa) $\mathcal T$ khả nghịch tất định; khi không khả nghịch thì xấp xỉ $D_\theta$ bằng cách dán nhãn lại dữ liệu quần thể. Thế giới out chỉ còn bất định do ngẫu nhiên của thuật toán huấn luyện:

$$P_{out}(D,\theta,z):\ D=D_{\theta_0}\setminus z_0,\ \theta\leftarrow\mathcal T(D),\ z=z_0 \tag{13}$$

**Các bước xây dựng Attack D (self-distillation):**
1. Lấy các tập quần thể $D_i\sim\pi^n$.
2. Dán **nhãn mềm (soft label)** cho $D_i$ bằng chính mô hình đích $\theta_0$, được tập $D'_i$.
3. Huấn luyện mô hình chưng cất (distilled model) $\theta_i\leftarrow\mathcal T(D'_i)$.
4. Đo loss của bản ghi đích $z_0$ trên các $\theta_i$, tạo $M_{\theta_0,z_0}=\{(\theta_i,z_0)\}$.
5. Đặt $c_\alpha(\theta_0,x_{z_0},y_{z_0})$ là $\alpha$-percentile của các loss đó (phương trình 16).

Giả định: các distilled model xấp xỉ tốt mô hình huấn luyện lại trên tập dữ liệu của mô hình đích nhưng không chứa $z_0$. Hợp lý vì $D'_i$ được dán nhãn mềm bởi $\theta_0$ nên phần nào phục hồi tập huấn luyện của $\theta_0$, trừ thông tin về bản ghi đích (với quần thể lớn, $z_0$ rất khó rơi vào $D_i$). Ngưỡng phụ thuộc cả mô hình lẫn bản ghi.

### 4.5 Attack L lý tưởng: leave-one-out

Kẻ tấn công huấn luyện các reference model $\theta'$ trên $D\setminus\{(x_b,y_b)\}$, ngẫu nhiên chỉ đến từ $\mathcal T$. Cùng lớp với Attack D (phụ thuộc mô hình và dữ liệu) nhưng yêu cầu biết chính xác $n-1$ bản ghi còn lại, hợp lý trong audit nhưng quá mạnh trong thực tế. **Attack D nhằm tái tạo hiệu năng của Attack L mà không cần giả định đó.**

### 4.6 Tổng kết và so sánh các tấn công

Các câu hỏi mà mỗi tấn công đặt ra (loss $\ell(\theta,z)$ có giống mẫu từ phân phối nào?):
- **P:** loss của dữ liệu quần thể ngẫu nhiên trên *cùng mô hình*.
- **S:** loss trên các mô hình huấn luyện từ quần thể.
- **R:** loss của *đúng bản ghi $z$* trên các mô hình huấn luyện từ quần thể.
- **D:** loss của $z$ trên các mô hình gần với mô hình đích nhất có thể (qua distillation).
- **L:** loss của $z$ trên các mô hình huấn luyện với $n-1$ bản ghi của $D$ (trừ $z$).

Kỳ vọng: S và P cùng bản chất, nhưng S có thể sai số cao hơn do dùng mô hình khác để xấp xỉ phân phối loss. R và D cùng bản chất, nhưng D tự tin hơn nhờ giảm bất định về các dữ liệu huấn luyện khác ảnh hưởng đến loss; D gần Attack L nhất. Mục tiêu cuối: tiến gần hiệu năng leave-one-out mà không cần biết $D\setminus\{z\}$.

---

## 5. Đánh giá thực nghiệm (Empirical Evaluation)

**Mục tiêu:** không chỉ chứng minh mô hình dễ bị MIA hay tấn công mới mạnh hơn, mà còn hiểu: (1) giảm bất định cải thiện hiệu năng thế nào (cùng mức tin cậy) trên mục tiêu tổng quát và xấu nhất; (2) vì sao hai tấn công khác nhau cho dự đoán khác nhau trên cùng mô hình/bản ghi; (3) mức dễ tổn thương của các bản ghi mà các tấn công đồng ý/bất đồng; vì sao một số tấn công bỏ sót bản ghi rất dễ tổn thương. "Dễ tổn thương" (vulnerability) của bản ghi được đo bằng hiệu năng của leave-one-out attack trên các mô hình huấn luyện có/không có bản ghi đó (cố định phần còn lại).

Kết quả chính báo cáo trên **Purchase100** (setup II); thêm dataset và setup ở Phụ lục C. Cài đặt nằm trong Privacy Meter.

### 5.1 Minh họa ngưỡng tấn công

*[Hình 1: Histogram log(loss) của các thế giới out trong Attack S, P, R, D cho hai mô hình $\theta_1,\theta_2$ và hai bản ghi $z_1,z_2$ (Purchase100 II, $\alpha=0{,}1$). Ngưỡng S là hằng cho cả bốn mục tiêu; P đổi theo mô hình; R đổi theo bản ghi; D đổi theo cả hai. Phân phối loss của R và D tập trung hơn rõ rệt.]*

Nhận xét: ngưỡng của S dùng chung cho cả bốn mục tiêu nên sai cả bốn; P đổi ngưỡng theo mô hình nhưng vẫn dùng cùng ngưỡng cho $z_1,z_2$ nên cũng sai cả bốn; R tạo thế giới out tập trung hơn theo bản ghi; D còn tập trung hơn nữa và dự đoán đúng cả bốn mục tiêu là member với độ tin cậy cao (ngưỡng nằm xa trên loss của mục tiêu). Tín hiệu sắc nét hơn ⇒ bất định giảm.

### 5.2 Đánh giá hiệu năng tấn công

Hiệu năng trung bình đo bằng TPR/FPR trên member/non-member ngẫu nhiên của mô hình ngẫu nhiên, vẽ **ROC** khi thay đổi $\alpha$; **AUC** đo độ mạnh.

**Bảng – AUC trên Purchase100 setup II (Hình 2):**

| Tấn công | AUC |
|---|---|
| S | 0,754 |
| P | 0,756 |
| R | 0,808 |
| D | 0,831 |

*[Hình 2: Đường cong TPR–FPR trên Purchase100 II (S, R, D dùng 999 shadow/reference/distilled model; P dùng 1000 điểm quần thể mỗi lớp; trung bình trên 10 mô hình đích, riêng D trên 3 mô hình). (b) vùng FPR nhỏ: R và D có TPR cao hơn S và P hơn 10 lần, R nhỉnh hơn D một chút; (c) vùng TPR cao (>0,7): D có FPR thấp hơn khoảng 2 lần so với S, P, R.]*

Giải thích: R hơi tốt hơn D ở FPR rất nhỏ có thể do sai số xấp xỉ bổ sung khi D dùng distilled model để thay cho mô hình huấn luyện lại; cải thiện chất lượng xấp xỉ này là hướng tiếp theo. Xu hướng AUC tăng từ S, P đến R, D lặp lại trên các setup Purchase100 khác và trên CIFAR10, CIFAR100, MNIST (Phụ lục C). Với DP (giới hạn tổng sai số FPR và FNR = 1−TPR), cả TPR cao lẫn FPR thấp đều quan trọng.

### 5.3 So sánh chi tiết các tấn công

*[Hình 3: Biểu đồ phân tán so sánh độ tin cậy của hai tấn công trên từng bản ghi huấn luyện (FPR $\alpha=0{,}3$); tọa độ = ngưỡng − loss. (a) S và P tương quan mạnh; (b) S và R rất khác nhau, 19,84% bản ghi R đúng mà S sai; (c) D vượt R (dịch lên vùng tây bắc).]*

**Bảng 2 – Tỉ lệ đồng thuận (agreement) giữa các tấn công và ground truth (GT), Purchase100 II.** Tam giác trên: điểm train; tam giác dưới: điểm test. Một số giá trị chính:

| Cặp | FPR 0,05 (train) | FPR 0,3 (train) |
|---|---|---|
| S–P | 0,972 | 0,892 |
| S–R | 0,774 | 0,534 |
| R–D | 0,792 | 0,814 |
| D–L | 0,352 | 0,804 |
| R–GT | 0,204 | 0,730 |
| D–GT | 0,180 | 0,78 |
| L–GT | 0,816 | 0,968 |

Nhận xét: ở FPR 0,3, D đồng thuận với ground truth nhiều nhất trong các tấn công thực tế trên điểm train; cặp S–P đồng thuận cao (0,9), cho thấy bản chất tương tự; D đồng thuận với L cao nhất (0,804), phù hợp thiết kế (D xấp xỉ L bằng distillation).

### 5.4 Vì sao một số điểm dễ tổn thương không bị phát hiện

Tác giả chia các bản ghi huấn luyện (Purchase100 II) thành bốn loại dựa trên việc tấn công nào đoán đúng là member trên ≥80% mô hình đích (50 mô hình, $\alpha=0{,}3$):

| Loại | Định nghĩa | Số lượng xấp xỉ |
|---|---|---|
| All correct | S, P, R, D đều đúng | ≈500 |
| R correct | R đúng, S và P sai | ≈450 |
| SP correct | S và P đúng, R sai | ≈125 |
| Random (baseline) | Mẫu ngẫu nhiên trong tập 5000 điểm | – |

(Do hạn chế tính toán, việc D có bỏ sót hay không chưa được kiểm tra.)

- **Khả năng nhận diện bản ghi dễ tổn thương:** *[Hình 4: AUC của Attack L trên 50 mô hình có/không có từng bản ghi: R correct 0,984; All correct 0,899; Random 0,805; SP correct 0,741.]* Các bản ghi R-correct dễ tổn thương nhất, SP-correct ít nhất (thậm chí thấp hơn ngẫu nhiên), nghĩa là R/D xác định rủi ro cá nhân đáng tin hơn S/P.
- **Vì sao S và P bỏ sót:** *[Hình 5: Histogram loss; bản ghi R-correct có loss cao hơn SP-correct và All-correct.]* S và P (ngưỡng quá chung) thường bỏ sót các bản ghi dễ tổn thương nhưng vốn có loss cao ("hard examples"), phù hợp trực giác của các nghiên cứu về hiệu chỉnh độ khó, nhưng ở đây được nhận diện chặt chẽ hơn.
- **Vì sao điểm S/P phát hiện lại ít dễ tổn thương:** vì khi S/P phát hiện đúng một bản ghi thực sự dễ tổn thương thì R cũng hầu như luôn phát hiện; các điểm chỉ S/P bắt được là các điểm ít tổn thương, nên việc S/P phát hiện một bản ghi có thể là dấu hiệu sai về mức rủi ro.
- **Ảnh hưởng của phần dữ liệu huấn luyện còn lại:** *[Hình 7: Histogram loss của điểm xấu nhất, 1-NN, 3-NN và các điểm huấn luyện trung bình, tách theo mô hình mà Attack R ($\alpha=0{,}05$) đoán đúng/sai.]* Loss của điểm xấu nhất thấp hơn trên mô hình bị đoán đúng (theo thiết kế), loss trên điểm huấn luyện ngẫu nhiên gần như không đổi, nhưng loss của các hàng xóm gần nhất (1-NN, 3-NN; khoảng cách cosine trên lớp áp chót) khác biệt rõ giữa hai nhóm mô hình. Như vậy một bản ghi dễ tổn thương hơn khi tập huấn luyện chứa bản ghi tương tự nó; hàng xóm tiềm ẩn ảnh hưởng đến mức rò rỉ nhiều hơn cả tập dữ liệu. Câu hỏi mở: giảm bất định về hàng xóm của bản ghi có cho tấn công hiệu quả hơn Attack D (vốn "lãng phí" khi xấp xỉ cả tập dữ liệu) không.

### So sánh với công trình đồng thời (LiRA, Carlini et al.)

Thiết lập: CIFAR10, Wide ResNet (depth 28, width 2), 25000 điểm huấn luyện, 1 mô hình đích, 29 mô hình cho mỗi tấn công.

| Phương pháp | AUC |
|---|---|
| LiRA online (fixed σ²) | 0,664 |
| LiRA offline (fixed σ²) | 0,569 |
| Global threshold | 0,571 |
| Attack R | 0,697 |
| Attack D | 0,704 |

*[Hình 6/10: ROC của R và D so với các biến thể của Carlini et al. Attack R đạt TPR tương tự ở FPR nhỏ; D đạt FPR thấp hơn ở TPR cao.]*

Chi phí: LiRA cần $n$ mô hình OUT để tấn công mọi điểm và $n$ mô hình IN cho mỗi điểm; Attack R chỉ cần $2n$ mô hình OUT; để tấn công $k$ điểm, chi phí phân bổ của LiRA lớn hơn khoảng $(k+1)/2$ lần. Attack D còn nắm bắt ảnh hưởng của tập dữ liệu đích chưa biết, điều LiRA không xét.

---

## 6. Kết luận (Conclusions)

Bài báo cung cấp khung kiểm toán rủi ro quyền riêng tư của mô hình đối với từng bản ghi, thông qua các MIA bảo đảm tin cậy FPR trên các thế giới out chi tiết. Trong khung này, các tấn công mạnh dần (S, P, R, D) nêu bật các loại bất định của kẻ tấn công vốn giới hạn hiệu năng.

*(Phần Acknowledgments và Danh mục tài liệu tham khảo được lược bỏ.)*

---

## Phụ lục A. Suy diễn chi tiết LRT xấp xỉ cho MIA

Mục tiêu: chứng minh Lemma 4.1 (dạng Lemma A.1) rằng LRT xấp xỉ quy tắc ngưỡng loss.

**Hai bất đẳng thức/xấp xỉ về hậu nghiệm $P(\theta|D)$:**
1. Với mọi điểm $z=(z_x,z_y)$ và tập $D$:

$$P(\theta\mid D\cup z)\ \ge\ e^{-\ell(\theta,z_x,z_y)/T}\cdot P(\theta\mid D) \tag{17}$$

2. Với $z_1,\dots,z_n$ i.i.d. từ $\pi$ và $n$ đủ lớn:

$$\mathbb E_{z_{1..n}}\big[P(\theta\mid z_1..z_n)\big]\ \approx\ \mathbb E_{z_{1..n-1}}\big[P(\theta\mid z_1..z_{n-1})\big] \tag{18}$$

(đúng nhờ sự hội tụ của hậu nghiệm khi $n\to\infty$.) Chứng minh (1) dùng mô hình hậu nghiệm kiểu Gibbs

$$P(\theta\mid D)\approx\frac{e^{-\frac1T\sum_{(x,y)\in D}\ell(\theta,x,y)}}{\int e^{-\frac1T\sum_{(x,y)\in D}\ell(\theta,x,y)}d\theta} \tag{28}$$

với $T$ là hằng nhiệt độ của thuật toán huấn luyện, và dùng $\ell\ge0$. Xấp xỉ này đúng cho nhiều thuật toán học Bayes, như SGD; khi $T\to0$ nó tập trung quanh cực tiểu của loss (suy diễn MAP tất định), khi $T=1$ tương ứng lấy mẫu hậu nghiệm Bayes với prior đều.

**Các bước suy ra LRT:**
1. Likelihood dưới $H_0$: $L(H_0|\theta,z)=\pi(z)\cdot\mathbb E_{D\sim\pi^n}[P(\theta|D)]$.
2. Likelihood dưới $H_1$: $L(H_1|\theta,z)=\pi(z)\cdot\mathbb E_{D'\sim\pi^{n-1}}[P(\theta|D'\cup z)]$.
3. Áp dụng (17) và (18):

$$L(H_1|\theta,z)\ \gtrsim\ \pi(z)\,e^{-\ell(\theta,z_x,z_y)/T}\,\mathbb E_{D\sim\pi^n}[P(\theta|D)]$$

4. Do đó thống kê LRT:

$$LR(\theta,z)=\frac{L(H_0|\theta,z)}{L(H_1|\theta,z)}\ \lesssim\ e^{\ell(\theta,z_x,z_y)/T}\approx e^{\ell/T} \tag{31}$$

5. LRT bác bỏ $H_0$ khi $LR$ nhỏ, nên vùng bác bỏ xấp xỉ $\{(\theta,z):\ \ell(\theta,z_x,z_y)\le T\log c\}$ (phương trình 32), tức quy tắc ngưỡng loss hằng.

---

## Phụ lục B. Thiết lập thực nghiệm (Experimental Setup)

### B.1 Mô hình đích

Mọi mô hình huấn luyện bằng categorical cross-entropy; dữ liệu lấy không hoàn lại (trừ CIFAR10 IV dùng Poisson sampling).

| Dataset/setup | Kiến trúc và tối ưu | Số điểm huấn luyện |
|---|---|---|
| Purchase100 I | MLP 4 lớp [512,256,128,64], SGD | 2500 |
| Purchase100 II, III | như trên | 10000 (theo mô tả B.1; Bảng 3 ghi II = 5000, III = 10000) |
| Purchase100 IV | như trên + gradient clipping (ℓ2 = 2,0) | 10000 |
| CIFAR10 I | AlexNet, Adam | 2500 |
| CIFAR10 II | AlexNet, Adam | 5000 |
| CIFAR10 III | VGGNet 3 khối, SGD momentum, L2 λ=0,001 | 10000 |
| CIFAR10 IV | Wide ResNet (28, 2), Adam | 25000 (kỳ vọng, Poisson) |
| CIFAR100/MNIST I, II | CNN 2 lớp [32,64] + max pooling, SGD | 2500 / 5000 |

*(Lưu ý: bản gốc có mâu thuẫn nhỏ giữa mô tả B.1 và Bảng 3 về kích thước tập của Purchase100 II.)*

### B.2 Huấn luyện shadow/reference/distilled model và điểm quần thể

Các mô hình này dùng cùng kiến trúc, thuật toán và kích thước tập dữ liệu như mô hình đích, huấn luyện trên các phần chia i.i.d. (có thể chồng lấn). Số lượng:
- Purchase100 I, II, III: 999 mô hình; 1000 điểm quần thể mỗi lớp cho Attack P. Purchase100 IV: 29 mô hình.
- CIFAR10 I, II: 399 mô hình; 400 điểm mỗi lớp cho P. CIFAR10 III: 29 mô hình. CIFAR10 IV: 15 mô hình, không chạy Attack P.
- CIFAR100: 400 điểm mỗi lớp cho P; MNIST: 1000 điểm mỗi lớp.

### B.3 Làm mượt phân phối loss và tính percentile liên tục

Để tính ngưỡng cho $\alpha<1/(N+1)$ ($N+1$ là số mẫu loss), tác giả làm mượt thay vì dùng histogram rời rạc. Bốn kỹ thuật được thử, báo cáo kỹ thuật cho hiệu năng tốt nhất mỗi setup:

1. **Nội suy tuyến tính** percentile giữa các loss đã sắp xếp $l_0\le\dots\le l_N$.
2. **Logit rescaling** [Carlini et al.]: biến đổi mỗi loss thành $\log\frac{e^{-l}}{1-e^{-l}}$, fit Gaussian $\mathcal N(\mu,\sigma^2)$ lên các giá trị này, rồi giải $\frac{e^{-p(\alpha)}}{1-e^{-p(\alpha)}}=(1-\alpha)$-percentile của Gaussian để lấy ngưỡng $p(\alpha)$.
3. **Min của hai ngưỡng** (tuyến tính và logit): $p(\alpha)=\min(p_{linear}(\alpha),p_{logit}(\alpha))$, bảo thủ: chỉ kết luận member khi cả hai cùng cho FPR ở mức $\alpha$.
4. **Trung bình độ tin cậy** của hai CDF làm mượt: $F(\ell)=\tfrac12(F_{linear}(\ell)+F_{logit}(\ell))$, rồi nghịch đảo $F^{-1}$ để lấy ngưỡng.

---

## Phụ lục C. Kết quả thực nghiệm bổ sung

### C.1 Minh họa ngưỡng trên nhiều mục tiêu

*[Hình 8: Histogram ngưỡng loss của S, P, R, D trên 10 mô hình đích (3 cho D) và các bản ghi train/test cùng một lớp (Purchase100 II). S dùng cùng một ngưỡng cho mọi mục tiêu, P gần như vậy; R và D dùng ngưỡng đa dạng, thích nghi với từng mô hình và bản ghi.]*

### C.2 Hiệu năng tấn công bổ sung

**Bảng 3 – AUC trên Purchase100** (I: 2500 điểm, II: 5000 điểm, III, IV: 10000 điểm; IV có gradient clipping):

| Setup | Train acc. | Test acc. | S | P | R | D |
|---|---|---|---|---|---|---|
| I | 92,36 | 49,9 | 0,800 | 0,818 | 0,825 | 0,837 |
| II | 99,5 | 65,4 | 0,754 | 0,756 | 0,808 | 0,831 |
| III | 100,0 | 75,5 | 0,687 | 0,687 | 0,755 | 0,768 |
| IV | 95,74 | 71,71 | 0,649 | 0,656 | 0,701 | 0,717 |

Rủi ro giảm khi tập huấn luyện lớn hơn, nhưng xu hướng S, P < R < D vẫn giữ.

**Bảng 4 – AUC trên CIFAR10** (I, II: AlexNet; III: VGGNet; IV: Wide ResNet):

| Setup | S | P | R | D |
|---|---|---|---|---|
| I | 0,870 | 0,857 | 0,874 | 0,871 |
| II | 0,860 | 0,868 | 0,858 | 0,889 |
| III | 0,707 | 0,709 | 0,792 | 0,806 |
| IV | – | – | 0,697 | 0,704 |

*[Hình 9: ROC trên CIFAR10 III (29 mô hình mỗi loại; P dùng 400 điểm mỗi lớp); D cao nhất, R sát sau, xu hướng giống Purchase100 nhưng hiệu năng thấp hơn.]*

**Bảng 5 – CIFAR100:** S/P/R/D = 0,959/0,960/0,964/0,957 (setup I) và 0,944/0,945/0,945/0,936 (setup II): mọi tấn công đều thành công.
**Bảng 6 – MNIST:** S/P ≈ 0,50 và R/D ≈ 0,55: mọi tấn công đều yếu; R và D nhỉnh hơn S và P ở 3 trong 4 setup của CIFAR100/MNIST.

### C.3 Điểm xấu nhất bổ sung

*[Hình 12: Thí nghiệm giống Hình 7 trên một bản ghi dễ tổn thương khác; kết quả tương tự: loss của các hàng xóm gần nhất khác biệt rõ giữa mô hình đoán đúng và sai.]*

### C.4 So sánh chi tiết với LiRA

*[Hình 10: ROC của R, D so với ba biến thể của Carlini et al. trên CIFAR10 IV (29 mô hình mỗi loại, 1 mô hình đích).]* R và D có AUC cao hơn cả ba biến thể; R đạt TPR tương tự ở FPR nhỏ; D đạt FPR thấp hơn ở TPR cao.

### C.5 Ảnh hưởng của chi phí tính toán

*[Hình 11: AUC theo số mô hình (10–200) cho S, R, D và theo số điểm quần thể (100–1000 mỗi lớp) cho P, Purchase100 II.]* AUC của R và D tăng nhanh theo số reference/distilled model rồi ổn định; S chỉ tăng nhẹ theo số shadow model; P tăng nhẹ theo số điểm quần thể.

---

## Phụ lục D. Hiệu năng tấn công dưới thuật toán phòng thủ

Thử nghiệm nhỏ với gradient clipping như phòng thủ heuristic (Purchase100 IV): hiệu năng tấn công thấp hơn nói chung (so với không clipping), nhưng xu hướng mạnh dần S, P, R, D vẫn giữ nguyên. Tác giả cho rằng điều này vì khung của họ không giả định gì về thuật toán huấn luyện hay phòng thủ khi suy ra các tấn công.

---

## Ghi chú

- **Nền tảng phương pháp luận:** bài báo là nguồn gốc của ý tưởng ngưỡng loss hiệu chỉnh theo từng mẫu và thế giới "out" thu hẹp dần (S → P → R → D → L); đây là khuôn khổ tự nhiên để so sánh các tấn công trên mô hình chuỗi thời gian (time-series) và các dạng dữ liệu khác, ngoài ảnh/bảng.
- **Attack R ≈ cơ sở của reference-model/difficulty-calibration:** có thể làm nền để thích nghi sang forecasting, trong đó "loss" thay bằng sai số dự báo (MSE/MAE) và reference model là các mô hình huấn luyện trên cửa sổ quần thể khác.
- **Báo cáo đúng cách:** nhấn mạnh phải báo TPR ở FPR thấp và đường ROC, không chỉ accuracy hay AUC; quan trọng khi đánh giá MIA trên mô hình chuỗi thời gian vốn thường có tín hiệu yếu.
- **Cố định nguồn ngẫu nhiên:** các trò chơi 3.1–3.4 giúp xác định rõ đang audit mô hình, bản ghi hay cặp (bản ghi, tập dữ liệu), cần thiết khi mẫu (cửa sổ thời gian) của chuỗi tương quan mạnh và khái niệm "bản ghi" mơ hồ.
- **Ghi chú thận trọng:** bài báo chỉ thử trên dữ liệu ảnh và bảng (Purchase100, CIFAR, MNIST); các giả định (i.i.d., hậu nghiệm kiểu Gibbs, tập huấn luyện dán nhãn lại bằng distillation) có thể không đúng với dữ liệu chuỗi thời gian có phụ thuộc theo thời gian, nên cần kiểm chứng lại.
