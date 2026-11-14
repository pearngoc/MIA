# Tấn công suy luận thành viên từ các nguyên lý cơ bản

*Membership Inference Attacks From First Principles*

- **Tác giả:** Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramèr (Google Research; Milad Nasr còn thuộc UMass Amherst)
- **Hội nghị/năm:** IEEE Symposium on Security and Privacy (S&P) 2022 (bản arXiv:2112.03570v2, 04/2022)
- **Mã nguồn:** TensorFlow Privacy, thư mục `research/mi_lira_2021`

> *Bản tóm tắt chi tiết bằng tiếng Việt (do AI viết, không phải bản dịch nguyên văn).*

---

## Tóm lược (Abstract)

- **Membership inference attack (MIA – tấn công suy luận thành viên)** là tấn công trong đó kẻ tấn công truy vấn một mô hình đã huấn luyện để đoán xem một mẫu cụ thể có nằm trong tập huấn luyện hay không.
- Các bài trước đánh giá MIA bằng các thước đo "trung bình" (accuracy, AUC). Những thước đo này không cho biết tấn công có xác định được *chắc chắn* thành viên nào hay không. Tác giả đề xuất đánh giá bằng **TPR (true-positive rate) tại FPR (false-positive rate) rất thấp** (ví dụ ≤ 0,1%). Khi đo như vậy, phần lớn tấn công trước đây hoạt động rất kém.
- Họ xây dựng **LiRA (Likelihood Ratio Attack)**, kết hợp cẩn thận nhiều ý tưởng đã có. LiRA mạnh hơn khoảng 10 lần ở vùng FPR thấp và vẫn vượt các tấn công trước trên những thước đo cũ.

---

## I. Giới thiệu (Introduction)

**Bối cảnh và luận điểm chính**

- Mạng nơ-ron ngày càng được huấn luyện trên dữ liệu nhạy cảm, nên cần kiểm tra thực nghiệm tính riêng tư của mô hình. MIA là công cụ kiểm tra "mặc định" vì đơn giản.
- Nhiều tác phẩm trước dùng thước đo trung bình, gộp kết quả trên toàn bộ tập dữ liệu và mọi ngưỡng quyết định. Tác giả cho rằng **quyền riêng tư không phải là một đại lượng trung bình**: nếu tấn công xác định đáng tin cậy dù chỉ vài người dùng trong tập nhạy cảm thì tấn công đã thành công.
- Cách nghĩ này giống nhiều lĩnh vực an ninh khác (phát hiện mã độc, phát hiện xâm nhập...), nơi người ta thiết kế hệ thống quanh FPR thấp.

**Đóng góp**

1. Lập luận rằng MIA phải được đánh giá bằng TPR ở FPR thấp, kèm đường cong ROC thang log.
2. Chỉ ra rằng khi đánh giá như vậy, đa số tấn công trước đây gần như thất bại ở vùng FPR thấp. Accuracy hay AUC cao không đảm bảo điều gì ở đó. Ví dụ, tấn công của Yeom et al. có accuracy cao (59,5%) nhưng thất bại hoàn toàn ở FPR thấp. Tấn công của Long et al. có accuracy thấp hơn nhiều (53,5%) nhưng lại đạt TPR cao hơn ở FPR thấp.
3. Đề xuất LiRA: kết hợp **độ khó theo từng mẫu (per-example difficulty/hardness)** với một ước lượng likelihood Gaussian có hiệu chỉnh tốt.
4. Thực hiện đánh giá thực nghiệm rộng và công bố mã nguồn.

*[Hình 1: Đường cong ROC (thang log–log) trên mô hình CIFAR-10 không overfit (độ chính xác test 92%) cho thấy LiRA vượt xa các tấn công trước ở vùng FPR thấp, dù accuracy trung bình chỉ hơn vài điểm phần trăm.]*

**Hệ quả cho các nghiên cứu sau:** nhiều câu hỏi cũ cần xét lại với tấn công mạnh hơn: tấn công *label-only* (chỉ dùng nhãn) có thành công ở FPR thấp không, các cơ chế phòng thủ từng bị coi là "vô dụng" có chặn được LiRA không, và các thuật toán từng "an toàn" có thực sự an toàn không.

---

## II. Kiến thức nền (Background)

### A. Ký hiệu học máy

- Mạng phân loại $f_\theta : \mathcal{X} \to [0,1]^n$ ánh xạ đầu vào sang phân phối xác suất trên $n$ lớp. $f(x)_y$ là xác suất gán cho lớp $y$.
- $f_\theta \leftarrow \mathcal{T}(D)$: mô hình thu được khi chạy thuật toán huấn luyện $\mathcal{T}$ trên tập $D$.
- Cập nhật SGD (Eq. 1) và cross-entropy loss: $\ell(f_\theta(x), y) = -\log f_\theta(x)_y$.
- Có thể viết $f(x) = \sigma(z(x))$, trong đó $z(x)$ là các **logit** (đặc trưng trước softmax) và $\sigma$ là softmax.
- Đạt 100% độ chính xác trên tập train là dễ; khó là khái quát hóa tốt. Mọi mô hình trong bài đều dùng các kỹ thuật chống overfit hiện đại (augmentation, weight decay, early stopping). Vì vậy kết quả thực tế hơn so với các bài trước, vốn dùng mô hình có lỗi cao hơn 2–5 lần.

### B. Quyền riêng tư của dữ liệu huấn luyện

- **Các dạng tấn công riêng tư:** *training data extraction* (khôi phục từng mẫu huấn luyện), *model inversion* (khôi phục thông tin gộp theo lớp), *property inference* (suy ra thuộc tính của tập dữ liệu), và *membership inference / tracing attack*.
- Ngay cả việc xác định chính xác vài người trong một tập y tế cũng là vi phạm riêng tư, kể cả khi recall thấp. MIA cũng là nền tảng cho các tấn công trích xuất dữ liệu mạnh hơn, và các tấn công đó cần precision rất cao.
- **Lý thuyết về ghi nhớ (memorization):** khả năng bị MIA gắn trực tiếp với việc mô hình ghi nhớ từng mẫu. Mạng chuẩn có thể ghi nhớ cả tập nhãn ngẫu nhiên (Zhang et al.), và theo Feldman, một mức ghi nhớ nhất định có thể là cần thiết để khái quát hóa tối ưu.
- **Huấn luyện bảo vệ riêng tư:** chủ yếu là **differential privacy (DP – riêng tư vi phân)**, ví dụ DP-SGD hoặc PATE. Ngoài ra có các phòng thủ heuristic không có chứng minh hình thức, và nhiều phòng thủ loại này đã bị phá bởi các tấn công mạnh hơn.
- **Đo lường quyền riêng tư:** các kỹ thuật kiểm toán thường phải sửa quy trình huấn luyện (chèn canary, poisoning), nên tốn chi phí triển khai. Vì vậy cách phổ biến nhất vẫn là chạy MIA. Các thư viện MIA hiện có là nền tảng cho phần lớn phân tích riêng tư trong thực tế, nên chúng phải đánh giá chính xác.

---

## III. Tấn công suy luận thành viên (Membership Inference Attacks)

### A. Định nghĩa

**Định nghĩa 1 – Trò chơi bảo mật suy luận thành viên** (giữa challenger $\mathcal{C}$ và adversary $\mathcal{A}$):

1. $\mathcal{C}$ lấy mẫu tập huấn luyện $D \leftarrow \mathbb{D}$ và huấn luyện $f_\theta \leftarrow \mathcal{T}(D)$.
2. $\mathcal{C}$ tung một bit $b$. Nếu $b=0$, chọn một điểm mới $(x,y)$ từ phân phối (không thuộc $D$). Nếu $b=1$, chọn ngẫu nhiên một điểm trong $D$.
3. $\mathcal{C}$ gửi $(x,y)$ cho $\mathcal{A}$.
4. $\mathcal{A}$ có quyền truy vấn phân phối $\mathbb{D}$ và mô hình $f_\theta$, rồi xuất ra dự đoán $\hat b$.
5. Kết quả bằng 1 nếu $\hat b = b$, ngược lại bằng 0.

Giả định: kẻ tấn công biết phân phối dữ liệu (cần để huấn luyện **shadow model** – mô hình bóng) và biết cả mẫu lẫn nhãn thật. Mọi tấn công trong bài đều xuất ra điểm số liên tục $\mathcal{A}'(x,y)$ rồi so với ngưỡng $\tau$:

$$\mathcal{A}(x,y) = \mathbb{1}[\mathcal{A}'(x,y) > \tau]$$

**Tấn công ví dụ – LOSS attack (Yeom et al.):** mô hình được huấn luyện để giảm loss trên mẫu train, nên loss thấp thì khả năng là thành viên cao hơn:

$$\mathcal{A}_{\text{loss}}(x,y) = \mathbb{1}[-\ell(f(x),y) > \tau]$$

### B. Đánh giá MIA

Ví dụ chạy xuyên suốt: ResNet trên CIFAR-10, độ chính xác test 92%, huấn luyện trên 25.000 ảnh (một nửa tập) và 25.000 ảnh còn lại làm non-member. Khoảng cách train–test chỉ 8%.

**Balanced attack accuracy (Định nghĩa 2)** là $\Pr[\mathcal{A}(x,y)=b]$ trên tập cân bằng giữa member và non-member. Tác giả phê phán thước đo này vì hai lý do:

- **Đối xứng:** nó coi false positive và false negative có chi phí như nhau. Thực tế kẻ tấn công thường chỉ quan tâm một loại lỗi. Ví dụ khi MIA dùng để trích xuất dữ liệu, false negative vô hại còn false positive làm giảm hiệu quả của cả tấn công.
- **Trung bình:** hai tấn công có thể có accuracy trung bình y hệt nhau mà giá trị thực tế khác hẳn. Tấn công A đoán đúng tuyệt đối một nhóm 0,1% người dùng và ngẫu nhiên với phần còn lại, rất nguy hiểm. Tấn công B đúng 50,05% trên mọi người, vô dụng.

**Minh họa với LOSS attack:** accuracy 60% nghe có vẻ đáng lo. Nhưng trong 1% mẫu có loss thấp nhất (nơi tấn công "tự tin" nhất rằng đó là member), tấn công chỉ đúng 48%, kém ngẫu nhiên. Trong 1% mẫu có loss cao nhất (tự tin là non-member), nó đúng 100%. Như vậy LOSS thực chất là một tấn công *suy luận non-membership* mạnh, và gần như vô dụng để suy luận membership.

**ROC analysis:** nên xét đánh đổi TPR–FPR. Tác giả ưu tiên TPR/FPR hơn precision/recall vì TPR/FPR không phụ thuộc vào tỉ lệ member trong quần thể, vốn thường không biết. Đường ROC thể hiện TPR và FPR cho mọi ngưỡng $\tau$.

*[Hình 2: Đường ROC của LOSS attack ở thang tuyến tính (trái) và thang log–log (phải). Ở thang tuyến tính, tấn công không hơn ngẫu nhiên ở FPR dưới 20%, nhưng thang log làm lộ rõ chi tiết vùng FPR thấp.]*

**Vì sao không dùng AUC:** AUC lấy trung bình trên mọi FPR, kể cả vùng FPR cao vô nghĩa với tấn công thực tế (vùng FPR > 50% đã chiếm hơn một nửa diện tích). Tấn công A ở trên có AUC ≈ 51%, thấp hơn cả LOSS yếu.

**Thước đo đề xuất:** (1) vẽ ROC thang log; (2) tùy chọn tóm tắt bằng TPR tại một FPR thấp cố định (ví dụ 0,001% hoặc 0,1%). LOSS attack có TPR bằng 0% ở FPR 0,1%. Các bài trước nếu báo cáo TPR thì thường ở FPR 3–40%, quá cao để có ý nghĩa thực tế. Tác giả khuyến nghị luôn kèm đường ROC đầy đủ.

---

## IV. Likelihood Ratio Attack (LiRA)

### A. Suy luận thành viên như kiểm định giả thuyết

- Kẻ tấn công cần phân biệt hai "thế giới": mô hình được huấn luyện *có* chứa $(x,y)$ hoặc *không*.
- Định nghĩa hai phân phối trên mô hình:
  - $\mathcal{Q}_{\text{in}}(x,y) = \{ f \leftarrow \mathcal{T}(D \cup \{(x,y)\}) \mid D \leftarrow \mathbb{D}\}$
  - $\mathcal{Q}_{\text{out}}(x,y) = \{ f \leftarrow \mathcal{T}(D \setminus \{(x,y)\}) \mid D \leftarrow \mathbb{D}\}$
- Theo **bổ đề Neyman–Pearson**, kiểm định tối ưu ở một FPR cho trước là ngưỡng hóa **likelihood ratio**:

$$\Lambda(f;x,y) = \frac{p(f \mid \mathcal{Q}_{\text{in}}(x,y))}{p(f \mid \mathcal{Q}_{\text{out}}(x,y))} \quad (2)$$

  Ở đây $p(f\mid\mathcal{Q}_b)$ là mật độ xác suất của mô hình $f$ dưới phân phối tham số $\mathcal{Q}_b$.
- Test này không tính được vì không biết $\mathcal{Q}_{\text{in}}, \mathcal{Q}_{\text{out}}$ giải tích. Tác giả đơn giản hóa bằng cách xét phân phối của **loss trên đúng mẫu $(x,y)$**, ký hiệu $\tilde{\mathcal{Q}}_{\text{in}}, \tilde{\mathcal{Q}}_{\text{out}}$ (mô hình được hoặc không được huấn luyện trên mẫu). Khi đó chỉ cần tính đại lượng một chiều:

$$p\big(\ell(f(x),y) \mid \tilde{\mathcal{Q}}_{\text{in/out}}(x,y)\big) \quad (3)$$

- Cách thực hiện: huấn luyện nhiều shadow model để ước lượng trực tiếp $\tilde{\mathcal{Q}}_{\text{in/out}}$. Để tiết kiệm, giả sử chúng là Gaussian, nên chỉ cần ước lượng 4 tham số (trung bình và phương sai của mỗi phân phối). Sau đó tính likelihood của loss quan sát được dưới mỗi phân phối và chọn bên khả dĩ hơn.

### B. Ghi nhớ và độ khó theo từng mẫu

- Với LOSS attack, ngưỡng loss toàn cục ngầm giả định mọi mẫu có thang loss như nhau và việc thêm hay bỏ một mẫu tác động như nhau lên mô hình. Điều này sai: theo Feldman & Zhang, một số mẫu ("outlier") ảnh hưởng đến mô hình nhiều hơn hẳn các mẫu "inlier".
- Thí nghiệm: huấn luyện 1024 mô hình trên các tập con ngẫu nhiên của CIFAR-10, rồi vẽ histogram loss của 4 ảnh khi là member (đỏ) hoặc non-member (xanh).
- Hai trục biến thiên độc lập:
  - mẫu là *inlier hay outlier* (đo bằng loss khi không có mẫu trong tập train);
  - mẫu *dễ hay khó fit* (loss khi có mặt trong tập train thấp hay cao).
  Một mẫu có thể là outlier mà vẫn dễ fit, hoặc inlier mà khó fit.

*[Hình 3: Histogram loss của 4 ảnh CIFAR-10 cho trường hợp member và non-member. Hai phân phối tách biệt nhau theo những cách rất khác nhau giữa các mẫu, nên một ngưỡng loss toàn cục không thể phân biệt các tình huống này.]*

- Kết luận: một ngưỡng toàn cục chỉ có thể tự tin nói "loss cao thì là non-member". Likelihood-ratio test thì mô hình hóa riêng cặp $\tilde{\mathcal{Q}}_{\text{in}}, \tilde{\mathcal{Q}}_{\text{out}}$ cho từng mẫu, nên tính đến độ khó của mẫu.

### C. Ước lượng likelihood ratio bằng mô hình tham số

**Vì sao dùng Gaussian (tham số) thay vì phi tham số:**

- Cần ít shadow model hơn nhiều. Tác giả nói có thể bằng kết quả của Ye et al. với số mô hình ít hơn 400 lần.
- Có thể mở rộng sang Gaussian đa biến để truy vấn nhiều lần (xem phần VI-C).
- Ngoài ra, ước lượng phi tham số không ngoại suy tốt xuống FPR rất thấp.

**Logit scaling (co giãn logit):**

- Cross-entropy loss không phải Gaussian vì nằm trên thang log. Confidence $f(x)_y$ bị chặn trong $[0,1]$ nên cũng không Gaussian.
- Giải pháp: áp hàm logit lên confidence:

$$\phi(p) = \log\frac{p}{1-p}, \qquad p = f(x)_y$$

  Giá trị thu được nằm trong $(-\infty,\infty)$ và xấp xỉ chuẩn.

*[Hình 4: Phân phối của confidence, của cross-entropy loss, và của logit-scaled confidence. Chỉ dạng logit xấp xỉ tốt bằng hai Gaussian.]*

**Thuật toán 1 – LiRA online** (các bước):

1. Cho mô hình đích $f$, mẫu $(x,y)$ và phân phối dữ liệu $\mathbb{D}$.
2. Lặp $N$ lần: lấy mẫu một tập shadow $D_{\text{attack}} \leftarrow \mathbb{D}$.
3. Huấn luyện **IN model** trên $D_{\text{attack}} \cup \{(x,y)\}$ và ghi lại $\phi(f_{\text{in}}(x)_y)$.
4. Huấn luyện **OUT model** trên $D_{\text{attack}} \setminus \{(x,y)\}$ và ghi lại $\phi(f_{\text{out}}(x)_y)$.
5. Tính $\mu_{\text{in}}, \mu_{\text{out}}$ (trung bình) và $\sigma^2_{\text{in}}, \sigma^2_{\text{out}}$ (phương sai) của hai tập giá trị trên.
6. Truy vấn mô hình đích để được $\text{conf}_{\text{obs}} = \phi(f(x)_y)$.
7. Trả về

$$\Lambda = \frac{p(\text{conf}_{\text{obs}} \mid \mathcal{N}(\mu_{\text{in}}, \sigma^2_{\text{in}}))}{p(\text{conf}_{\text{obs}} \mid \mathcal{N}(\mu_{\text{out}}, \sigma^2_{\text{out}}))}$$

- **Song song hóa:** với tập $D$ cho trước, huấn luyện shadow model trên $N$ tập con của $D$ sao cho mỗi mẫu xuất hiện trong đúng $N/2$ tập. Cùng $N$ mô hình đó phục vụ test cho mọi mẫu.
- **Truy vấn nhiều lần:** có thể truy vấn trên nhiều phiên bản augmentation $x_1,\dots,x_m$ của $x$. Khi đó dùng Gaussian cầu $m$ chiều $\mathcal{N}(\mu_{\text{in}}, \sigma^2_{\text{in}} I)$ và $\mathcal{N}(\mu_{\text{out}}, \sigma^2_{\text{out}} I)$ với test likelihood-ratio chuẩn giữa hai Gaussian đa biến.

**Tấn công offline:** bản online phải huấn luyện mô hình mới cho mỗi (nhóm) mẫu cần suy luận, rất tốn kém. Bản offline huấn luyện trước shadow model trên các tập ngẫu nhiên và không bao giờ huấn luyện trên mẫu đích. Cụ thể bỏ các dòng huấn luyện/ghi IN model, chỉ ước lượng $\mu_{\text{out}}$ và $\sigma^2_{\text{out}}$, và đổi sang kiểm định một phía:

$$\Lambda = 1 - \Pr\big[Z > \phi(f(x)_y)\big], \qquad Z \sim \mathcal{N}(\mu_{\text{out}}, \sigma^2_{\text{out}}) \quad (4)$$

Confidence của mô hình đích càng vượt $\mu_{\text{out}}$ thì khả năng là member càng cao. Bản offline cũng có thể mở rộng sang truy vấn nhiều augmentation với Gaussian đa biến.

---

## V. Đánh giá tấn công (Attack Evaluation)

### Thiết lập thực nghiệm

- **Dataset:** CIFAR-10, CIFAR-100, ImageNet, WikiText-103. Hai bộ dữ liệu đồ chơi từng dùng nhiều (Purchase, Texas) bị tác giả cho là quá đơn giản nên chỉ đưa vào phụ lục.
- **Mô hình đích:**

| Dataset | Mô hình | Dữ liệu huấn luyện | Ghi chú |
|---|---|---|---|
| CIFAR-10 | ResNet | 25.000 ảnh (một nửa) | ~90–92% test acc |
| CIFAR-100 | Wide ResNet | 25.000 ảnh | ~60% test acc |
| ImageNet | ResNet-50 | ~một nửa tập (~0,5 triệu ảnh) | ~65% test acc |
| WikiText-103 | GPT-2 nhỏ (tokenizer GPT-2) | một nửa của ~1 triệu câu, 20 epoch | mô hình ngôn ngữ |

- **Shadow model:** $N = 256$ cho mọi dataset trừ ImageNet ($N = 64$). Mỗi mẫu nằm trong đúng một nửa tập train của các shadow model. Tập train của shadow và của mô hình đích có thể giao nhau, một giả định mạnh. Mục VI-D chứng minh tấn công vẫn hiệu quả khi hai tập rời nhau.
- Với các dataset không phải ImageNet, mỗi tấn công được lặp 10 lần.

### A. Kết quả tấn công online

- TPR của LiRA ở FPR 0,001% dao động từ 0,1% đến 10% tùy dataset.
- AUC trung bình tương quan với khoảng cách khái quát hóa (generalization gap). Cả ba mô hình ảnh đều đạt 100% accuracy train, nhưng accuracy test lần lượt là 90% (CIFAR-10), 65% (ImageNet), 60% (CIFAR-100).
- Tuy vậy ở FPR thấp, mô hình CIFAR-10 *dễ bị tấn công hơn* ImageNet dù khái quát hóa tốt hơn.

*[Hình 5: Đường ROC thang log của LiRA online trên CIFAR-10, CIFAR-100, ImageNet, WikiText (AUC lần lượt khoảng 0,720; 0,925; 0,765; 0,715).]*

### B. Kết quả tấn công offline

- LiRA offline chỉ kém online một chút: ở FPR 0,1%, TPR thấp hơn tối đa khoảng 20% (tương đối) so với bản online tốt nhất cùng số shadow model.

*[Hình 6: ROC của LiRA offline trên bốn dataset (AUC: CIFAR-100 0,859; CIFAR-10 0,674; ImageNet 0,728; WikiText 0,713), vẽ cạnh bản online để so sánh.]*

### C. Đánh giá lại các tấn công trước

Tác giả chạy lại 8 tấn công đại diện dưới cùng giao thức và với thước đo FPR thấp (Bảng I). Bảng dưới đây rút gọn:

| Phương pháp | TPR@0,001% FPR (C-10 / C-100 / WT103) | TPR@0,1% FPR (C-10 / C-100 / WT103) | Balanced acc. (C-10 / C-100 / WT103) |
|---|---|---|---|
| Yeom et al. (LOSS) | 0,0 / 0,0 / 0,00% | 0,0 / 0,0 / 0,1% | 59,4 / 78,0 / 50,0% |
| Shokri et al. | 0,0 / 0,0 / – | 0,3 / 1,6 / – | 59,6 / 74,5 / – |
| Jayaraman et al. (MERLIN) | 0,0 / 0,0 / – | 0,0 / 0,0 / – | 59,4 / 76,9 / – |
| Song & Mittal | 0,0 / 0,0 / – | 0,1 / 1,4 / – | 59,5 / 77,3 / – |
| Sablayrolles et al. | 0,1 / 0,8 / 0,01% | 1,7 / 7,4 / 1,0% | 56,3 / 69,1 / 65,7% |
| Long et al. | 0,0 / 0,0 / – | 2,2 / 4,7 / – | 53,5 / 54,5 / – |
| Watson et al. | 0,1 / 0,9 / 0,02% | 1,3 / 5,4 / 1,1% | 59,1 / 70,1 / 65,4% |
| Ye et al. | – | – | 60,3 / 76,9 / 65,5% |
| **LiRA (online)** | **2,2 / 11,2 / 0,09%** | **8,4 / 27,6 / 1,4%** | 63,8 / 82,6 / 65,6% |

Điều bất ngờ là tấn công của Sablayrolles et al. (2019) vượt các tấn công mới hơn ở FPR thấp, thường bằng cả một bậc độ lớn. Bảng I còn phân loại tấn công theo bốn thành phần: dùng shadow model, truy vấn nhiều lần, độ khó theo lớp, độ khó theo mẫu.

**Nhận xét từng nhóm tấn công:**

- **Shadow model (Shokri et al.):** tấn công này huấn luyện thêm một mạng $g$ trên đặc trưng $f_i(x)$ và nhãn $y$ để dự đoán membership. Tác giả cài đặt bản mạnh hơn (tập shadow chồng lấn tập đích), nhưng nó vẫn không tốt hơn LOSS nhiều ở FPR thấp dù đắt hơn hẳn.
- **Truy vấn nhiều lần:** MERLIN (Jayaraman et al.) truy vấn 100 lần quanh $x$ với nhiễu Gaussian nhưng không cải thiện ở FPR thấp. Ý tưởng dùng augmentation của Choquette-Choo et al. là động lực trực tiếp cho truy vấn nhiều lần của LiRA, và nó có ích đáng kể (mục VI-C).
- **Độ khó theo lớp (per-class):** dùng ngưỡng riêng cho từng lớp không cải thiện accuracy hay TPR ở FPR thấp, chỉ tăng AUC khoảng 2% trên CIFAR. Tấn công Song & Mittal kết hợp điểm theo lớp với một thước đo entropy đã sửa đổi nhưng cũng không vượt tấn công cơ sở của Shokri ở FPR thấp.
- **Độ khó theo mẫu (per-example):**
  - Sablayrolles et al. là ảnh hưởng trực tiếp nhất lên LiRA: chỉnh loss bằng ngưỡng riêng cho mỗi mẫu, đặt gần trung điểm $\tau_{x,y} = (\mu_{\text{in}} + \mu_{\text{out}})/2$ (phi tham số, chỉ dùng trung bình).
  - Watson et al. là biến thể offline: $\tau_{x,y} = \mu_{\text{out}}(x,y)$.
  - Cả hai chỉ đánh giá bằng thước đo trung bình và thấy cải thiện khiêm tốn. Dưới thước đo của tác giả, accuracy/AUC của chúng thực ra hơi thấp hơn các tấn công đơn giản, nhưng TPR ở FPR thấp tốt hơn hẳn.
  - Long et al. lọc ra 10% outlier (theo khoảng cách láng giềng gần nhất trong đồ thị cosine của đặc trưng), chỉ tấn công các outlier đó. Vì không đưa ra dự đoán tự tin nào cho phần còn lại nên accuracy thấp, nhưng TPR ở FPR thấp lại tốt hơn đa số tấn công trước.
  - Ye et al. (nghiên cứu đồng thời) giống bản offline của LiRA: huấn luyện nhiều OUT model và làm kiểm định chính xác một phía, đặt ngưỡng sao cho $\alpha$ phần OUT loss nằm dưới ngưỡng.
- **Khác biệt then chốt của LiRA:** dùng mô hình Gaussian tham số. Các tấn công chỉ đo trung bình (Sablayrolles, Watson) sẽ không tối ưu khi các mẫu có phân phối loss với độ phân tán rất khác nhau. Long et al. và Ye et al. dùng toàn bộ phân phối OUT nhưng không ngoại suy được xuống FPR thấp: test chính xác của Ye et al. chỉ đạt được FPR thấp nhất là $1/N$ (không áp dụng được với 256 mô hình và FPR 0,1%), còn Long et al. dùng cubic spline dễ overfit và phân kỳ ngoài miền dữ liệu.

### D. Suy luận thành viên và overfitting

- Vẽ TPR@0,1% FPR theo khoảng cách train–test cho nhiều mô hình CIFAR-10 (CNN và Wide ResNet nhiều cỡ, nhiều optimizer, nhiều augmentation).
- Có xu hướng chung: mô hình overfit hơn thì dễ bị tấn công hơn. Nhưng có những mô hình cùng train–test gap mà dễ bị tấn công hơn tới 100 lần.
- Khi vẽ theo accuracy test (Hình 16, phụ lục), xu hướng rõ rệt là mô hình *chính xác hơn thì dễ bị tấn công hơn*.

*[Hình 7: TPR@0,1% FPR so với khoảng cách train–test cho nhiều cấu hình mô hình CIFAR-10, cho thấy xu hướng tăng nhưng phân tán rất lớn.]*

---

## VI. Nghiên cứu loại bỏ thành phần (Ablation Study)

Mọi ablation dùng CIFAR-10 (dataset phổ biến nhất và khó nhất trong số đã xét). Bảng II tóm tắt cách cộng dần từng thành phần để đi từ LOSS đến LiRA:

| Cấu hình tấn công | TPR@0,1% FPR |
|---|---|
| LOSS attack | 0,0% |
| + Logit scaling | 0,1% |
| + Multiple queries | 0,1% |
| *Offline:* LOSS + per-example threshold (chỉ $\tilde{\mathcal{Q}}_{\text{out}}$, Watson) | 1,3% |
| + Logit scaling | 4,7% |
| + Gaussian likelihood | 4,7% |
| + Multiple queries (**LiRA offline**) | 7,1% |
| *Online:* LOSS + per-example threshold ($\tilde{\mathcal{Q}}_{\text{in}}$ & $\tilde{\mathcal{Q}}_{\text{out}}$, Sablayrolles) | 1,7% |
| + Logit scaling | 1,9% |
| + Gaussian likelihood | 5,6% |
| + Multiple queries (**LiRA online**) | 8,4% |

**Nhận xét:**

- Không có ngưỡng theo mẫu thì chỉ co giãn loss và truy vấn nhiều lần chỉ cải thiện rất ít.
- Ngưỡng theo mẫu cho TPR 1,7% (khoảng một bậc độ lớn trên ngẫu nhiên). Logit scaling cộng Gaussian tăng thêm 3,3 lần, và truy vấn trên augmentation gần như nhân đôi nữa.
- Bản offline đi từ Watson (1,3%) và cộng các thành phần để đạt 7,1%, gần bằng 8,4% của bản online.

### A. Logit scaling hàm loss

- Dùng confidence $f(x)_y$ hay log của nó (cross-entropy) cho kết quả kém vì không Gaussian. Logit scaling cho kết quả tốt nhất, nhưng cách tính số học quan trọng.
- Có hai công thức tương đương toán học:

$$\phi_{\text{unstable}} = \log f(x)_y - \log\big(1 - f(x)_y\big)$$

$$\phi_{\text{stable}} = \log f(x)_y - \log \sum_{y' \neq y} f(x)_{y'}$$

  Bản thứ hai ổn định hơn khi confidence rất gần 1, nhưng cần toàn bộ vector $f(x)$. (Tác giả tính log dưới dạng $\log(x+\epsilon)$ với $\epsilon>0$ nhỏ.)
- Nếu truy cập được logit $z(x)$ (đầu ra trước softmax), có thể dùng **hinge loss** tương đương và đơn giản hơn:

$$\ell_{\text{Hinge}}(x,y) = z(x)_y - \max_{y' \neq y} z(x)_{y'}$$

  Lý do: $\phi_{\text{stable}} = z(x)_y - \text{LogSumExp}_{y'\neq y}\, z(x)_{y'}$, và LogSumExp là xấp xỉ trơn của hàm max.
- Các biến thể thống kê khác nhau cho TPR chênh nhau nhiều bậc độ lớn ở FPR thấp, dù AUC đều quanh 68–72%. Điều này nhấn mạnh việc đo ở FPR thấp.
- Logit scaling cũng giúp các tấn công trước: ví dụ tấn công Watson tăng TPR gần gấp 4 lần.

*[Hình 8: ROC so sánh nhiều thống kê (confidence, CE loss, logit không ổn định, logit ổn định, logit đầu ra, hinge). Thống kê nào làm đầu ra gần Gaussian thì tốt nhất.]*

### B. Khớp phân phối Gaussian

- Cần đủ shadow model để ước lượng phân phối loss. Hình 9 thay đổi số shadow model (một nửa IN, một nửa OUT). Hơn 64 mô hình thì lợi ích giảm dần, còn ít hơn 64 thì tấn công giảm nhanh vì khó khớp Gaussian bằng ít điểm dữ liệu.
- **Phương sai toàn cục (global variance):** khi có ít mô hình, nên ước lượng trung bình $\mu_{\text{in}}, \mu_{\text{out}}$ riêng cho từng mẫu nhưng ước lượng $\sigma^2_{\text{in}}, \sigma^2_{\text{out}}$ chung trên mọi member/non-member. Cách này tốt hơn phương sai riêng từng mẫu khi dưới 64 mô hình, và gần bằng tấn công tốt nhất chỉ với khoảng 16 mô hình. Khi có nhiều mô hình, phương sai theo mẫu tốt hơn: với 1024 mô hình, dùng phương sai toàn cục khiến TPR giảm từ 8,4% xuống 7,9%.

*[Hình 9: TPR@0,1% FPR theo số shadow model (4 đến 512), so sánh Gaussian với phương sai theo mẫu, phương sai toàn cục, và phi tham số (Sablayrolles).]*

### C. Số lượng truy vấn

- Mô hình thường được huấn luyện trên cả các phiên bản augmentation của mẫu, nên hợp lý khi tấn công cũng truy vấn trên các augmentation đó. Với CIFAR-10 có 162 augmentation khả dĩ ($2 \times 9 \times 9$: lật ngang hay không, dịch tối đa ±4 pixel mỗi chiều).

| Số truy vấn | TPR@0,1% FPR | TPR@0,001% FPR |
|---|---|---|
| 1 (không augmentation) | 5,6% | 1,0% |
| 2 (mirror) | 7,5% | 1,8% |
| 18 (mirror + shift) | 8,4% | 2,2% |
| 162 (mirror + shift) | 8,4% | 2,2% |

  Hai truy vấn đã cho phần lớn lợi ích, và 18 truy vấn bằng 162.

### D. Tập dữ liệu rời nhau (Disjoint datasets)

- Thực tế kẻ tấn công có $D_{\text{attack}}$ rời với $D_{\text{train}}$. Thí nghiệm dùng **CINIC-10** (CIFAR-10 cộng 210 nghìn ảnh ImageNet cùng lớp): huấn luyện một mô hình đích và 128 shadow model (chỉ OUT) trên 50.000 điểm. So sánh ba thiết lập:
  1. Shadow lấy từ toàn bộ CINIC-10 (tập train nằm trong $D_{\text{attack}}$, như các thí nghiệm trước).
  2. Shadow lấy từ phần rời hoàn toàn với tập train của mô hình đích ($D_{\text{train}} \cap D_{\text{attack}} = \emptyset$).
  3. Mô hình đích huấn luyện trên CIFAR-10, shadow huấn luyện trên phần ImageNet của CINIC-10 (có **distribution shift**).
- Kết quả: chồng lấn tập dữ liệu không ảnh hưởng gì (thiết lập 1 và 2 như nhau). Distribution shift làm giảm TPR, nhưng AUC lại cao hơn nhiều (hiện tượng mà tác giả để lại cho nghiên cứu sau).

*[Hình 10: ROC trên CINIC-10 cho ba thiết lập; AUC lần lượt khoảng 0,692, 0,682 và 0,783.]*

### E. Quy trình huấn luyện không khớp

- Kẻ tấn công phải đoán kiến trúc, optimizer và augmentation của mô hình đích. Mỗi lần dùng 64 shadow model và LiRA online với phương sai toàn cục. Đo TPR@0,1% FPR.
- **Kiến trúc** (CNN 16/32/64 filter, WRN28-1/2/10): tấn công tốt nhất khi đoán đúng, nhưng dùng kiến trúc gần giống (ví dụ WRN28-1 thay vì WRN28-2) ít ảnh hưởng. Mô hình lớn hơn dễ bị tấn công hơn.
- **Optimizer** (SGD, SGDM, Adam, trên WRN28-10): ảnh hưởng tối thiểu.
- **Augmentation** (không, mirror, mirror + shift, + cutout): đoán đúng augmentation có tác động lớn nhất. Augmentation mạnh hơn làm mô hình ít overfit và khó tấn công hơn.

*[Hình 11: TPR@0,1% FPR khi thay đổi kiến trúc, optimizer, augmentation của mô hình đích và của shadow. Điểm tốt nhất (khung đen) luôn là khi đoán đúng, nhưng sai lệch chỉ làm giảm vừa phải.]*

---

## VII. Các khảo sát bổ sung (Additional Investigations)

### A. Tấn công mô hình thực tế

- Để chắc chắn không vô tình huấn luyện ra mô hình quá dễ/khó tấn công, tác giả tấn công các mô hình CIFAR-10 hiện đại đã huấn luyện sẵn của Phan (huấn luyện trên toàn bộ 50.000 ảnh train).
- Huấn luyện 256 shadow model bằng mã huấn luyện tương ứng, mỗi mô hình lấy ngẫu nhiên 50.000 điểm từ 60.000. Trung bình mỗi mẫu có 213 IN và 43 OUT model. Vì chỉ có 43 OUT model nên phương sai toàn cục cho kết quả tốt nhất.
- Hai biến thể: biết đúng kiến trúc đích, hoặc dùng kiến trúc khác.
- Kết luận giống mục VI-E: kiến trúc ít ảnh hưởng đến mức rò rỉ (tấn công ResNet-18 hiệu quả hơn MobileNet-v2); tấn công tốt nhất khi shadow cùng kiến trúc nhưng bền trước sai lệch (ví dụ tấn công ResNet-34 bằng shadow ResNet-18 hoặc ResNet-50, TPR giảm từ 5% xuống 4%).

*[Hình 12: TPR@0,1% FPR của LiRA trên các kiến trúc CIFAR-10 hiện đại (VGG16, ResNet18/34/50, DenseNet121, Inception-v3, MobileNet-v2), với shadow cùng hoặc khác kiến trúc.]*

### B. Vì sao một số mẫu kém riêng tư hơn?

- TPR ở FPR thấp có thể rất cao dù mức thành công trung bình khiêm tốn, nghĩa là có một nhóm mẫu dễ bị tấn công hơn hẳn. Tác giả đưa ra một yếu tố chính: mẫu **out-of-distribution (OOD – ngoài phân phối)**.
- Thí nghiệm: chèn 1.000 mẫu OOD từ nhiều nguồn vào tập 50.000 ảnh CIFAR-10 (thành 51.000), huấn luyện shadow model và đo độ tách biệt giữa phân phối loss của IN và OUT cho từng mẫu được chèn, qua "privacy score":

$$d = \frac{|\mu_{\text{in}} - \mu_{\text{out}}|}{\sigma_{\text{in}} + \sigma_{\text{out}}}$$

  với $\mu$ và $\sigma$ là trung bình và độ lệch chuẩn của logit-confidence trên các IN/OUT model.
- Kết quả: CIFAR-10 chuẩn có score tập trung quanh 0. Ảnh CINIC-10 (lấy từ ImageNet, hơi lệch phân phối) có score cao hơn. Ảnh CIFAR-10 test bị gán nhãn ngẫu nhiên (OOD cực đoan) có score cao nhất. Ảnh CIFAR-100 nhãn ngẫu nhiên nằm ở khoảng giữa.

*[Hình 13: Phân bố privacy score cho CIFAR-10 nhãn đúng, CINIC-10 nhãn đúng, CIFAR-100 nhãn ngẫu nhiên, CIFAR-10 nhãn ngẫu nhiên. Mẫu càng OOD thì càng dễ nhận diện.]*

---

## VIII. Kết luận (Conclusion)

Tác giả nhấn mạnh MIA nên hướng tới TPR cao ở FPR thấp, và LiRA là một cách đạt mục tiêu đó. Các hướng cho tương lai:

- **MIA như thước đo riêng tư:** các công cụ đo riêng tư nên dùng tấn công mạnh như LiRA và xét vùng FPR thấp, để biết riêng tư của dù chỉ vài người dùng có bị xâm phạm chắc chắn không.
- **Cải thiện tính khả dụng:** nhược điểm chính của tấn công theo từng mẫu là cần học các siêu tham số từ dữ liệu và tốn tính toán. Tác giả cho rằng tấn công mạnh (dù chậm) quan trọng hơn tấn công nhanh nhưng yếu, và hy vọng sau này sẽ có phiên bản hiệu quả hơn.
- **Cải thiện các tấn công riêng tư khác:** MIA là nền tảng của nhiều tấn công khác (trích xuất dữ liệu, property inference), nên phương pháp này về nguyên tắc có thể nâng cao chúng.
- **Xét lại hiểu biết hiện tại về MIA:** nhiều kết quả cũ dựa trên thước đo trung bình cần được kiểm tra lại. Các câu hỏi mở: (i) các phòng thủ từng bị coi là "bị phá" có chặn được LiRA không; (ii) differential privacy tương tác thế nào với các tấn công mạnh hơn (có bằng chứng sơ bộ rằng bảo đảm DP "rỗng" có thể ngăn tấn công FPR thấp); (iii) các tấn công yếu quyền hơn (label-only) có hoạt động ở FPR thấp không; (iv) các tấn công có thêm quyền (truy cập gradient, mô hình trung gian) có thực sự hiệu quả hơn ở FPR thấp không.

---

## Phụ lục A: Các thí nghiệm bổ sung (Additional Experiments)

### A. Tấn công DP-SGD

- **DP-SGD** đưa nhiễu và cắt (clip) gradient để đạt differential privacy, và DP cho một cận trên về thành công của mọi MIA. Tác giả thử nhiều tổ hợp **noise multiplier** $\sigma$ và **clipping norm** $C$ trên CNN CIFAR-10.
- Độ chính xác của mô hình DP-SGD (Bảng IV):

| Noise multiplier $\sigma$ | $C=10$ | $C=5$ | $C=1$ |
|---|---|---|---|
| 0,0 | 84,0% | 78,5% | 61,3% |
| 0,2 | 73,9% | 77,1% | 62,8% |
| 0,8 | 36,9% | 43,3% | 61,3% |

- Chỉ cần clip gradient (không thêm nhiễu) đã làm giảm đáng kể hiệu quả của LiRA, nhưng $C$ nhỏ làm giảm độ chính xác.
- Với $C$ lớn, thêm một lượng nhiễu rất nhỏ ($\sigma = 0{,}2$) đã đưa tấn công về mức ngẫu nhiên (AUC ≈ 0,52) mà mô hình vẫn có độ chính xác cao hơn. Kết quả này đạt được dù ràng buộc $\varepsilon$ chứng minh được là rất lớn (ví dụ $\varepsilon > 5000$).

*[Hình 14: ROC của LiRA trên mô hình DP-SGD với ba mức ngân sách riêng tư ($\varepsilon=\infty$, $\varepsilon>5000$, $\varepsilon=8$), kèm đường cận trên lý thuyết của DP với $\varepsilon=1$.]*

### B. Tấn công white-box

- Các công trình trước cho rằng truy cập white-box (ví dụ chuẩn gradient tại điểm đích) cải thiện MIA. Tác giả thử dùng chuẩn gradient trong khung LiRA.
- Kết quả: chuẩn gradient cải thiện AUC tổng thể (cả với LiRA online và với ngưỡng toàn cục), nhưng ở FPR thấp **không** cải thiện so với dùng confidence của mô hình.

*[Hình 15: So sánh LiRA dùng logit (AUC 0,711) và dùng chuẩn gradient (AUC 0,724), cũng như ngưỡng toàn cục dùng logit (0,576) và chuẩn gradient (0,601).]*

---

## Phụ lục B: Hình và bảng bổ sung (Additional Figures and Tables)

### A. Mức thành công tấn công theo độ chính xác mô hình

- Vẽ TPR@FPR thấp theo độ chính xác test của cùng các mô hình ở Hình 7. Có xu hướng rõ rằng mô hình tốt hơn thì dễ bị tấn công hơn. Hiện tượng tương tự đã được ghi nhận với tấn công trích xuất dữ liệu.

### B. ROC đầy đủ cho việc khớp Gaussian

- Hình 17 là ROC thang log của thí nghiệm thay đổi số shadow model từ 4 đến 256, với hai biến thể: phương sai theo mẫu và phương sai toàn cục.
- Luôn nên ước lượng trung bình theo mẫu. Nhưng khi chỉ có ít mô hình, dùng một phương sai chung hiệu quả hơn hàng bậc độ lớn, và đạt gần mức tối đa chỉ với khoảng 16 mô hình.

### C. ROC của các tấn công trước

- Hình 18–21 trình bày đường ROC đầy đủ của các tấn công trước trên các dataset ở Bảng I.

### D. Ablation trên các dataset bổ sung

Các bảng V–VIII lặp lại phân tích Bảng II cho CIFAR-100, WikiText-103, Texas và Purchase (với ba dataset sau không dùng augmentation nên cũng không truy vấn nhiều lần). Số liệu TPR@0,1% FPR chính:

| Dataset | Offline: ngưỡng $\tilde{\mathcal{Q}}_{\text{out}}$ → +logit → +Gaussian → +queries | Online: ngưỡng in/out → +logit → +Gaussian → +queries |
|---|---|---|
| CIFAR-100 (Bảng V) | 5,2% → 14,7% → 18,9% → 22,3% | 7,4% → 2,8% → 24,1% → 27,6% |
| WikiText-103 (Bảng VI) | 1,1% → 1,1% → 1,2% (offline) | 1,0% → 1,0% → 1,4% (online) |

Nhận xét chung:

- Kiểm định Gaussian sau khi logit scaling tăng mạnh hiệu quả các tấn công vốn dùng ngưỡng theo mẫu, cả offline lẫn online.
- Khác với CIFAR-10, logit scaling riêng lẻ nhiều khi *có hại* cho tấn công online của Sablayrolles. Tương tự, kiểm định Gaussian riêng lẻ (không có logit scaling) thường làm tấn công kém đi. Hai thành phần này phải được dùng cùng nhau.

### E. ROC đầy đủ cho quy trình huấn luyện không khớp

- Hình 22–24 cho ROC đầy đủ của thí nghiệm mục VI-E, thay đổi kiến trúc, optimizer và augmentation giữa mô hình đích và shadow.

---

## Ghi chú

- Bài báo thiết lập chuẩn đánh giá **TPR@FPR thấp trên thang log** thay cho accuracy/AUC. Chuẩn này hiện được dùng rộng rãi và rất đáng áp dụng khi đánh giá MIA trên mô hình chuỗi thời gian, vì quyền riêng tư cần được xét ở trường hợp xấu chứ không ở trung bình.
- **LiRA** là baseline mạnh nhất nên dùng. Nó dựa trên shadow model, kiểm định likelihood-ratio Gaussian và hiệu chỉnh theo từng mẫu. Cần thay phần "logit của confidence phân loại" bằng một thống kê phù hợp với mô hình chuỗi thời gian (ví dụ loss dự báo hay sai số tái dựng), miễn là thống kê này xấp xỉ chuẩn sau khi biến đổi. Đây là suy luận của người viết tóm tắt: bài báo không xét dữ liệu chuỗi thời gian.
- Các thí nghiệm của bài chỉ có ảnh (CIFAR, ImageNet) và văn bản (WikiText-103, GPT-2), không có chuỗi thời gian. Tuy vậy WikiText-103 cho thấy khung LiRA áp dụng được cho dữ liệu tuần tự (mô hình ngôn ngữ).
- Mô hình phản ánh *độ khó theo từng mẫu* gợi ý rằng với chuỗi thời gian, các chuỗi hiếm hoặc bất thường (outlier) sẽ dễ lộ membership hơn. Điều này nhất quán với kết quả OOD ở mục VII-B.
- Các ablation về số shadow model, phương sai toàn cục và việc kẻ tấn công đoán sai kiến trúc hay quy trình huấn luyện cho thấy hướng thiết kế thí nghiệm thực tế. Với chi phí huấn luyện cao, bản offline hoặc phương sai toàn cục (khoảng 16–64 shadow model) là lựa chọn thực dụng.
