# Về tầm quan trọng của hiệu chỉnh độ khó (difficulty calibration) trong các cuộc tấn công suy luận thành viên

*(On the Importance of Difficulty Calibration in Membership Inference Attacks)*

**Tác giả:** Lauren Watson (University of Edinburgh), Chuan Guo (Meta AI), Graham Cormode (Meta AI), Alexandre Sablayrolles (Meta AI)

**Hội nghị / năm:** ICLR 2022 (arXiv:2111.08440v2 [cs.CR])

> Bản dịch tiếng Việt của bài báo gốc (dịch bởi AI)

> **Lưu ý về phạm vi:** Tệp này dịch phần Tóm tắt, và trình bày lại bằng tiếng Việt nội dung từng mục, từng phần phụ lục của bài báo theo dạng cô đọng, giữ nguyên đánh số và cấu trúc tiêu đề. Đây **không** phải bản dịch từng câu của toàn bộ văn bản. Các công thức, số liệu chính và danh mục tài liệu tham khảo được giữ lại. Các bảng số liệu rất lớn được rút gọn (xem ghi chú tại từng bảng).

---

## Tóm tắt (Abstract)

Tính dễ bị tổn thương của các mô hình học máy trước các cuộc tấn công suy luận thành viên (membership inference attack) đã nhận được nhiều sự quan tâm trong những năm gần đây. Phần lớn các cuộc tấn công hiện có vẫn thiếu tính thực tiễn do có tỷ lệ dương tính giả (false positive rate) cao, tức là các mẫu không thuộc tập huấn luyện thường bị dự đoán nhầm là thành viên. Loại lỗi này khiến tín hiệu thành viên được dự đoán trở nên không đáng tin cậy, đặc biệt vì trong các ứng dụng thực tế đa số mẫu là không-thành-viên. Trong công trình này, nhóm tác giả lập luận rằng các cuộc tấn công suy luận thành viên có thể hưởng lợi rất lớn từ hiệu chỉnh độ khó (difficulty calibration), trong đó điểm thành viên mà cuộc tấn công dự đoán được điều chỉnh theo mức độ khó của việc phân loại đúng mẫu mục tiêu. Nhóm chứng minh rằng hiệu chỉnh độ khó có thể giảm đáng kể tỷ lệ dương tính giả của nhiều cuộc tấn công hiện có mà không làm giảm độ chính xác.

---

## 1. Giới thiệu (Introduction)

- Các mô hình học máy hiện đại thường được huấn luyện trên dữ liệu nhạy cảm (hồ sơ sức khỏe, thông tin cá nhân) và có thể ghi nhớ dữ liệu huấn luyện ở mức độ lớn, làm lộ quyền riêng tư của người tham gia (Fredrikson et al., 2014; 2015; Shokri et al., 2017; Carlini et al., 2019).
- **Suy luận thành viên (membership inference attack, MIA):** kẻ tấn công muốn suy ra một cá nhân có nằm trong tập huấn luyện của mô hình mục tiêu hay không (Shokri et al., 2017; Yeom et al., 2018). Khả năng này cũng là một cận dưới cho mức độ ghi nhớ của mô hình, và được dùng như thước đo thực nghiệm của tổn thất quyền riêng tư (Jagielski et al., 2020; Nasr et al., 2021).
- Độ chính xác (accuracy) của tấn công, thước đo phổ biến nhất, bỏ qua yếu tố then chốt là tỷ lệ dương tính giả (FPR) trên các mẫu không-thành-viên (Rezaei & Liu, 2021). Hầu hết các tấn công định nghĩa một **điểm thành viên** $s(h, z)$ cho mô hình $h$ và cặp đầu vào–nhãn $z=(x,y)$, chọn một ngưỡng $\tau$ và dự đoán $z$ là thành viên khi và chỉ khi $s(h,z) > \tau$. Phân phối điểm của thành viên và không-thành-viên thường chồng lấn mạnh (Hình 1a), nên FPR cao.
- **Quan sát chính:** một mẫu không-thành-viên có thể có điểm cao chỉ vì nó "dễ" hoặc được biểu diễn quá nhiều trong phân phối dữ liệu. Nếu đo điểm thành viên *so với* một mô hình "điển hình" được huấn luyện trên dữ liệu cùng phân phối, sự chênh lệch này là tín hiệu thành viên đáng tin cậy hơn nhiều. Hình 1b cho thấy sau hiệu chỉnh, điểm của thành viên và không-thành-viên tách biệt tốt hơn và có thể đạt FPR thấp.

*[Hình 1: Biểu đồ histogram của điểm negative loss (xem Phương trình 1) trước và sau khi hiệu chỉnh độ khó. Khi chưa hiệu chỉnh, điểm của thành viên và không-thành-viên chồng lấn đáng kể và không thể xác định ngưỡng cho FPR thấp. Sau hiệu chỉnh, các mẫu có điểm cao nhất chủ yếu thuộc lớp thành viên, cho phép tấn công có độ chính xác (precision) cao và FPR thấp. (a) Trước hiệu chỉnh; (b) Sau hiệu chỉnh.]*

- **Đóng góp:** đề xuất hiệu chỉnh độ khó như một kỹ thuật tổng quát cho các tấn công dựa trên điểm số; áp dụng cho các điểm confidence (Salem et al., 2018), loss (Yeom et al., 2018) và gradient norm (Nasr et al., 2019). Trên nhiều tập dữ liệu chuẩn, các tấn công đã hiệu chỉnh đạt FPR thấp hơn đáng kể; AUC tăng tới 0,10 trên các benchmark ML phổ biến; đồng thời cải thiện đánh đổi precision–recall mà accuracy vẫn tương đương hoặc tốt hơn.

---

## 2. Bối cảnh (Background)

- **Suy luận thành viên:** Homer et al. (2008) chỉ ra có thể suy ra sự hiện diện của một cá nhân trong hỗn hợp DNA phức tạp. Shokri et al. (2017) giới thiệu phương pháp **shadow model**: kẻ tấn công huấn luyện các mô hình thay thế bắt chước mô hình mục tiêu, quan sát hành vi của chúng trên mẫu thành viên/không-thành-viên, rồi huấn luyện một meta-model tấn công.
- **Tấn công dựa trên điểm số (score-based attacks):** Yeom et al. (2018) liên hệ MIA với overfitting. Điểm thành viên có thể tính từ loss, gradient norm (GN) hoặc confidence. Với hàm mất mát cross-entropy $\ell$:

$$s_{\text{loss}}(h,(x,y)) = -\ell(h(x),y) := \log\big(h(x)_y\big) \tag{1}$$

$$s_{\text{GN}}(h,(x,y)) = -\left\|\nabla \ell(h(x),y)\right\|_2 \tag{2}$$

$$s_{\text{confidence}}(h,(x,y)) = -\max_{y'} \ell(h(x),y') = \max_i \log\big(h(x)_i\big) \tag{3}$$

- **Tấn công chỉ dùng nhãn (label-only attacks):** khi mô hình chỉ trả về nhãn top-1, vẫn có thể định nghĩa điểm đạt độ chính xác gần state-of-the-art (Li & Zhang, 2020; Choquette-Choo et al., 2021).
- **Tấn công độ chính xác cao (high-precision attacks):** Long et al. (2018) chọn các mẫu có loss khác biệt nhiều nhất giữa mô hình mục tiêu và tập mô hình tham chiếu. Carlini et al. (2020) xếp hạng các câu sinh từ GPT-2 theo perplexity rồi chia cho entropy z-lib hoặc perplexity của một mô hình nhỏ hơn. Cả hai đều là dạng hiệu chỉnh độ khó (so sánh loss của mô hình mục tiêu với mô hình tham chiếu).
- **Quyền riêng tư vi phân (differential privacy, DP) như biện pháp giảm thiểu:** thuật toán ngẫu nhiên $\mathcal{M}$ thỏa $(\epsilon,\delta)$-DP nếu với mọi cặp tập dữ liệu $D, D'$ khác nhau tối đa một mẫu và mọi tập con $R$ của không gian đầu ra:

$$\mathbb{P}(\mathcal{M}(D)\in R) \le \exp(\epsilon)\,\mathbb{P}(\mathcal{M}(D')\in R) + \delta \tag{4}$$

  DP bảo vệ có thể chứng minh được trước MIA (Yeom et al., 2018; Sablayrolles et al., 2019). Abadi et al. (2016) đề xuất DP-SGD; phân tích được tinh chỉnh bởi Mironov (2017), Mironov et al. (2019) và được chứng minh chặt về mặt thực nghiệm (Nasr et al., 2021).

---

## 3. Hiệu chỉnh độ khó (Difficulty Calibration)

Các tấn công dựa trên điểm số trước đây rất kém tin cậy khi phân biệt mẫu không-thành-viên *dễ dự đoán* với mẫu thành viên *khó dự đoán*, vì cả hai đều có thể có điểm cao. Hiệu chỉnh độ khó giải quyết điều này.

**Ý tưởng.** Thay vì chỉ tính điểm trên mô hình mục tiêu $h$, lấy mẫu nhiều mô hình "điển hình" $g$ được huấn luyện trên cùng phân phối dữ liệu và tính điểm trên chúng. Nếu $(x,y)$ là một mẫu không-thành-viên dễ dự đoán thì $s(h,(x,y))$ cao nhưng $s(g,(x,y))$ cũng cao, khoảng cách nhỏ cho thấy nhiều khả năng đó là không-thành-viên.

Hình thức hóa: gọi $D_{\text{shadow}}$ là tập shadow rút từ cùng phân phối với tập huấn luyện của $h$, và $A$ là thuật toán huấn luyện ngẫu nhiên tạo ra phân phối trên các mô hình huấn luyện từ $D_{\text{shadow}}$. Điểm đã hiệu chỉnh:

$$s_{\text{cal}}(h,(x,y)) = s(h,(x,y)) - \mathbb{E}_{g\leftarrow A(D_{\text{shadow}})}\big[s(g,(x,y))\big] \tag{5}$$

Kỳ vọng được xấp xỉ bằng cách lấy mẫu một hoặc nhiều mô hình. Tấn công thực hiện bằng cách đặt ngưỡng trên $s_{\text{cal}}$.

**Hiệu chỉnh hiệu quả thông qua forgetting.** Thực hiện Phương trình 5 đòi hỏi huấn luyện nhiều mô hình, rất tốn kém với mạng lớn như transformer quy mô lớn (Brown et al., 2020). Nếu có quyền truy cập white-box vào $h$, có thể *warm-start* huấn luyện từ chính $h$ trên dữ liệu khác cùng phân phối. Cách làm này nối MIA với hiện tượng quên thảm khốc (catastrophic forgetting) (Goodfellow et al., 2013; Kirkpatrick et al., 2017). Toneva et al. (2018) quan sát rằng khi mô hình tiếp tục huấn luyện trên dữ liệu khác, nó dễ "quên" các mẫu huấn luyện ban đầu, đặc biệt các mẫu bất thường/đặc trưng, và tập này ổn định qua các lần huấn luyện.

**Liên hệ với suy luận hậu nghiệm (posterior inference).** Sablayrolles et al. (2019) giả định mật độ của $A(D)$ có dạng

$$p(h;D) \propto e^{-\sum_{(x,y)\in D}\ell(h(x),y)} \tag{6}$$

(có thể suy ra được cho tối ưu hóa bằng SGD trong một số điều kiện; Sato & Nakagawa, 2014). Với mô hình đe dọa của DP (kẻ tấn công biết tập huấn luyện là $D$ hoặc $D' = D\cup\{(x,y)\}$ và biết $D$), chiến lược tấn công tối ưu Bayes có thể được tiệm cận bằng cách đặt ngưỡng trên điểm hiệu chỉnh ở Phương trình 5 với $s = s_{\text{loss}}$ và $D = D_{\text{shadow}}$. Phân tích này mở rộng cho mọi hàm điểm $s$ như gradient norm hay confidence.

---

## 4. Thực nghiệm (Experiments)

Đánh giá hai biến thể: (i) hiệu chỉnh bằng mô hình tham chiếu huấn luyện từ đầu (Cal.), và (ii) hiệu chỉnh qua forgetting (Cal.-F). Mã nguồn: https://github.com/facebookresearch/calibration_membership.

### 4.1 Thiết lập thực nghiệm (Experimental setup)

**Tập dữ liệu.** German Credit, Hepatitis, Adult (UCI; Dua & Graff, 2017), MNIST (LeCun et al., 1998), CIFAR10/100 (Krizhevsky et al., 2009) và ImageNet (Deng et al., 2009); kích thước từ 155 đến 1.281.167 điểm, gồm cả bài toán ảnh và phi ảnh.

**Thiết lập tấn công.** Dữ liệu chia thành tập *private* (chỉ người huấn luyện biết) và tập *public* (dùng để huấn luyện mô hình tham chiếu và chọn ngưỡng $\tau$). Mô hình $h$ được huấn luyện trên một nửa tập private; nửa còn lại là không-thành-viên. Với ImageNet lấy mẫu 10.000 thành viên và 10.000 không-thành-viên; các tập khác dùng toàn bộ tập private. Do đó accuracy của tấn công ngẫu nhiên là 50%, cũng như precision ở mọi mức recall. Lặp lại $T$ lần với các phép chia ngẫu nhiên khác nhau ($T=5$; ImageNet $T=3$).

**Kiến trúc mô hình.**
- German Credit, Adult, Hepatitis: MLP một tầng ẩn $2m$ đơn vị (với $m$ đặc trưng), ReLU, softmax.
- MNIST: CNN nhỏ theo Leino & Fredrikson (2020): hai tầng tích chập (20 và 50 kênh, kernel 5), max pooling, một tầng fully connected 500 đơn vị ReLU; dropout 0,25 sau max pooling và 0,5 sau tầng fully connected.
- CIFAR10/100: CNN với 4 tầng tích chập và average pooling, ReLU, 32/64/64/128 kênh, kernel 3.
- ImageNet: ResNet18 (He et al., 2016).

**Huấn luyện.** 50–200 epoch, batch size từ 4 đến 1024. SGD với learning rate 0,1, Nesterov momentum 0,9, cosine schedule cho CIFAR10/100 và ImageNet; các tập nhỏ như German Credit dùng thêm weight decay $10^{-4}$. Huấn luyện riêng tư vi phân dùng thư viện Opacus (Yousefpour et al., 2021). Mô hình tham chiếu dùng cùng siêu tham số với mô hình mục tiêu; đa số thí nghiệm dùng một mô hình tham chiếu (ảnh hưởng của số lượng được xét ở mục 4.3). Với Cal.-F, các mô hình tham chiếu được tinh chỉnh siêu tham số để train/test accuracy tương tự mô hình mục tiêu. Với cả hai phương pháp, mô hình tham chiếu được huấn luyện đến khi validation accuracy đạt cực đại và dừng trước khi train accuracy hội tụ.

**Bảng 1: Train/test accuracy trung bình của các mô hình mục tiêu.**

| Tác vụ | Train Acc. | Test Acc. |
|---|---|---|
| Credit | 0,921 | 0,751 |
| Hep. | 0,998 | 0,871 |
| Adult | 0,904 | 0,842 |
| MNIST | 0,990 | 0,989 |
| CIFAR10 | 0,946 | 0,740 |
| CIFAR100 | 0,938 | 0,373 |
| ImageNet | 0,773 | 0,637 |

**Chọn ngưỡng.** Khi tính AUC và precision/recall, quét qua một dải giá trị $\tau$. Để tìm ngưỡng tối ưu accuracy, chia đôi tập public, coi một nửa là thành viên, nửa kia là không-thành-viên, rồi chọn $\tau$ tốt nhất trên thiết lập mô phỏng này. Với tấn công đã hiệu chỉnh, ngưỡng tối ưu thường chỉ lớn hơn 0 một chút (ví dụ 0,0001), có thể dùng làm ngưỡng mặc định.

### 4.2 Kết quả chính (Main results)

Đánh giá các tấn công dùng đầu ra liên tục: loss (Pt. 1), GN (Pt. 2), confidence (Pt. 3), với **gap attack** (Yeom et al., 2018) làm đường cơ sở. Phụ lục C có thêm thí nghiệm với nhiều mô hình hiệu chỉnh và các điểm khác.

**AUC và accuracy.** Các tấn công đã hiệu chỉnh thường tốt hơn hẳn bản chưa hiệu chỉnh; có trường hợp AUC tăng hơn 0,10 (German Credit, CIFAR10). Hiệu chỉnh nhìn chung không làm giảm accuracy, đôi khi còn cải thiện nhẹ. Tấn công confidence kém hơn các phương pháp khác vì mô hình dự đoán sai các điểm không-thành-viên với độ tin cậy cao; do đó hiệu chỉnh confidence không cải thiện trên nhiều tập dữ liệu.

**Bảng 2: AUC của các tấn công dựa trên điểm số trước và sau hiệu chỉnh độ khó** (Orig = chưa hiệu chỉnh; Cal. = hiệu chỉnh; Cal.-F = hiệu chỉnh qua forgetting). Độ lệch chuẩn tối đa 0,024 (Credit), 0,034 (Hep.), 0,003 (MNIST), 0,015 (các tập còn lại).

| Tập dữ liệu | Gap | Loss Orig | Loss Cal. | Loss Cal.-F | GN Orig | GN Cal. | GN Cal.-F | Conf. Orig | Conf. Cal. | Conf. Cal.-F |
|---|---|---|---|---|---|---|---|---|---|---|
| Credit | 0,549 | 0,542 | 0,648 | 0,594 | 0,513 | 0,579 | 0,559 | 0,501 | 0,553 | 0,543 |
| Hep. | 0,535 | 0,514 | 0,558 | 0,559 | 0,515 | 0,553 | 0,548 | 0,513 | 0,525 | 0,533 |
| Adult | 0,514 | 0,516 | 0,554 | 0,529 | 0,510 | 0,523 | 0,515 | 0,507 | 0,510 | 0,512 |
| MNIST | 0,506 | 0,505 | 0,519 | 0,509 | 0,504 | 0,514 | 0,509 | 0,503 | 0,510 | 0,505 |
| CIFAR10 | 0,663 | 0,676 | 0,731 | 0,708 | 0,678 | 0,787 | 0,762 | 0,629 | 0,635 | 0,631 |
| CIFAR100 | 0,854 | 0,911 | 0,903 | 0,886 | 0,912 | 0,924 | 0,912 | 0,852 | 0,707 | 0,760 |
| ImageNet | 0,536 | 0,547 | 0,557 | 0,551 | 0,540 | 0,511 | 0,520 | 0,544 | 0,515 | 0,512 |
| CIFAR10 (aug.) | 0,603 | 0,603 | 0,690 | 0,674 | 0,603 | 0,666 | 0,707 | 0,560 | 0,568 | 0,575 |
| CIFAR100 (aug.) | 0,784 | 0,812 | 0,844 | 0,831 | 0,811 | 0,856 | 0,843 | 0,679 | 0,672 | 0,647 |

**Bảng 3: Accuracy của các tấn công dựa trên điểm số trước và sau hiệu chỉnh độ khó.** Độ lệch chuẩn tối đa 0,018 (Credit), 0,026 (Hep.), 0,002 (MNIST), 0,005 (các tập còn lại). Trong hầu hết các trường hợp, tấn công đã hiệu chỉnh tương đương hoặc tốt hơn bản chưa hiệu chỉnh về accuracy.

| Tập dữ liệu | Gap | Loss Orig | Loss Cal. | Loss Cal.-F | GN Orig | GN Cal. | GN Cal.-F | Conf. Orig | Conf. Cal. | Conf. Cal.-F |
|---|---|---|---|---|---|---|---|---|---|---|
| Credit | 0,589 | 0,617 | 0,618 | 0,577 | 0,569 | 0,573 | 0,567 | 0,557 | 0,554 | 0,555 |
| Hep. | 0,561 | 0,574 | 0,575 | 0,591 | 0,574 | 0,593 | 0,585 | 0,574 | 0,577 | 0,576 |
| Adult | 0,534 | 0,536 | 0,534 | 0,522 | 0,518 | 0,518 | 0,515 | 0,512 | 0,511 | 0,511 |
| MNIST | 0,506 | 0,508 | 0,513 | 0,509 | 0,508 | 0,511 | 0,508 | 0,507 | 0,508 | 0,506 |
| CIFAR10 | 0,664 | 0,712 | 0,657 | 0,662 | 0,719 | 0,720 | 0,702 | 0,642 | 0,623 | 0,622 |
| CIFAR100 | 0,854 | 0,911 | 0,829 | 0,862 | 0,915 | 0,876 | 0,898 | 0,820 | 0,731 | 0,711 |
| ImageNet | 0,536 | 0,542 | 0,545 | 0,542 | 0,538 | 0,518 | 0,521 | 0,540 | 0,541 | 0,553 |
| CIFAR10 (aug.) | 0,602 | 0,609 | 0,626 | 0,625 | 0,610 | 0,610 | 0,651 | 0,562 | 0,542 | 0,558 |
| CIFAR100 (aug.) | 0,784 | 0,785 | 0,765 | 0,775 | 0,788 | 0,793 | 0,789 | 0,647 | 0,641 | 0,621 |

**Ảnh hưởng của tăng cường dữ liệu (data augmentation).** Hai hàng cuối của Bảng 2 và 3 cho thấy các tấn công kém hiệu quả hơn trên mô hình dùng data augmentation do giảm overfitting; tuy vậy hiệu chỉnh vẫn thường cải thiện AUC và accuracy của các tấn công dựa trên loss và gradient norm.

**Đường cong ROC và precision-recall.** Trên CIFAR10 (Hình 2), ở FPR thấp, tấn công chưa hiệu chỉnh gần như không có lợi thế về TPR (tương đương đoán ngẫu nhiên), trong khi tấn công đã hiệu chỉnh có khoảng cách TPR–FPR lớn; ở recall thấp, tấn công đã hiệu chỉnh phát hiện được nhóm thành viên với precision rất cao. Ngưỡng $\tau$ tối ưu accuracy luôn nằm ở phần đuôi của đường cong (FPR cao, recall cao, precision thấp), phù hợp với Rezaei & Liu (2021): các tấn công hiện có không dự đoán thành viên đáng tin cậy khi tối ưu accuracy.

*[Hình 2: Đường cong ROC (trái) và precision-recall (phải) của các tấn công điểm loss đã/chưa hiệu chỉnh trên CIFAR10. Ngưỡng $\tau$ tối ưu accuracy được đánh dấu bằng chấm đỏ. Hiệu chỉnh cho TPR cao hơn ở cùng FPR, hay tương đương precision cao hơn ở recall thấp. Đường cong precision-recall bộc lộ các hành vi rất khác nhau giữa hai phương pháp vốn có accuracy gần như nhau.]*

**Tấn công chỉ dùng nhãn (label-only).** Khi mô hình chỉ trả về nhãn rời rạc, vẫn có thể suy luận thành viên bằng khoảng cách tới biên quyết định bằng HopSkipJump (Choquette-Choo et al., 2021). Hiệu chỉnh độ khó cũng cải thiện các tấn công này (AUC tăng, accuracy tương đương hoặc tốt hơn), tuy mức tăng trên CIFAR10/100 nhỏ hơn rõ rệt. MNIST và ImageNet bị loại do HopSkipJump cần quá nhiều truy vấn.

**Bảng 4: Accuracy và AUC trung bình của tấn công label-only HopSkipJump**, dùng $\min(n,1000)$ điểm cho mỗi tập huấn luyện và held-out ($n$ là kích thước dữ liệu gốc). Mô hình CIFAR được huấn luyện không dùng data augmentation.

| Tập dữ liệu | AUC Orig. | AUC Cal. | AUC Cal.-F | Acc. Orig. | Acc. Cal. | Acc. Cal.-F |
|---|---|---|---|---|---|---|
| Credit | 0,552 | 0,640 | 0,599 | 0,616 | 0,620 | 0,638 |
| Hep. | 0,547 | 0,585 | 0,594 | 0,596 | 0,588 | 0,591 |
| Adult | 0,512 | 0,541 | 0,537 | 0,533 | 0,549 | 0,539 |
| CIFAR10 | 0,660 | 0,686 | 0,620 | 0,665 | 0,666 | 0,623 |
| CIFAR100 | 0,849 | 0,841 | 0,856 | 0,847 | 0,841 | 0,844 |

### 4.3 Nghiên cứu loại trừ (Ablation studies)

**Kích thước tập huấn luyện.** Vì độ dễ bị tấn công gắn với overfitting, AUC và accuracy tăng khi tập huấn luyện của mô hình mục tiêu nhỏ lại (Hình 3). Hiệu chỉnh cải thiện AUC và accuracy trên toàn bộ dải kích thước.

*[Hình 3: AUC (trái) và accuracy (phải) của gap attack và các tấn công điểm loss đã/chưa hiệu chỉnh, nhắm vào mô hình mục tiêu huấn luyện với các kích thước tập huấn luyện khác nhau trên CIFAR10. Cả AUC và accuracy tăng khi tập huấn luyện nhỏ đi do mô hình mục tiêu overfit nặng hơn. Hiệu chỉnh độ khó tận dụng điều này một cách hiệu quả để cải thiện AUC và accuracy ở mọi kích thước.]*

**Tỷ lệ thành viên / không-thành-viên.** Trong thực tế đa số mẫu là không-thành-viên. Khi giảm tỷ lệ thành viên (Hình 4), accuracy tăng mạnh vì ngưỡng tối ưu accuracy ưu tiên recall cao và precision thấp, về thực chất là "suy luận không-thành-viên". Ngược lại, đánh đổi precision-recall xấu đi khi số mẫu không-thành-viên tăng, phản ánh việc tấn công kém tin cậy trong thực tế. Dù vậy, tấn công đã hiệu chỉnh vẫn đạt precision cao ở recall thấp ngay cả với tỷ lệ thành viên thấp.

*[Hình 4: Accuracy (trái) và đường cong precision-recall (phải) của tấn công điểm loss khi thay đổi tỷ lệ thành viên : không-thành-viên (1:10, 1:7, 1:5, 1:3, 1:2, 1:1). Ở tỷ lệ thấp, tấn công đã hiệu chỉnh vẫn xác định được thành viên với precision cao, trong khi tấn công chưa hiệu chỉnh không làm được dù accuracy cao.]*

**Số lượng mô hình tham chiếu.** Các kết quả ở mục 4.2 dùng một mô hình tham chiếu. Hình 5 xét việc tăng số mô hình tham chiếu (mỗi mô hình dùng một tập con khác của tập public, lấy mẫu không hoàn lại), áp dụng calibration-via-forgetting cho điểm loss và thay đổi kích thước tập shadow. Tăng số mô hình và kích thước tập shadow đều cải thiện AUC và accuracy; chênh lệch giữa 1 và 10 mô hình tham chiếu tới 0,05 AUC và 0,03 accuracy.

*[Hình 5: AUC (trái) và accuracy (phải) của tấn công loss đã hiệu chỉnh với số lượng mô hình tham chiếu thay đổi trên CIFAR-10. Dùng nhiều mô hình tham chiếu hơn và huấn luyện chúng với tập shadow lớn hơn đều cải thiện hiệu năng, đổi lại chi phí tính toán lớn hơn.]*

**Quyền riêng tư vi phân.** DP là biện pháp phòng thủ phổ biến nhất và giới hạn có thể chứng minh được accuracy của MIA khi $\epsilon$ nhỏ (Yeom et al., 2018). Với $\epsilon$ lớn, DP vẫn bảo vệ trung bình trước tấn công chưa hiệu chỉnh, nhưng tấn công đã hiệu chỉnh có lợi thế đáng kể trên các điểm ngoại lai. Hình 6 cho thấy DP làm giảm hiệu quả tấn công như kỳ vọng, nhưng với $\epsilon = 100$ và $1000$ tấn công đã hiệu chỉnh vẫn đạt đánh đổi FPR–TPR và precision–recall không tầm thường. Chỉ khi $\epsilon \to 1$ thì AUC và accuracy mới tiệm cận mức của tấn công chưa hiệu chỉnh, và cả hai hội tụ về mức đoán ngẫu nhiên.

*[Hình 6: Đường cong ROC (trái) và precision-recall (phải) của tấn công điểm loss đã/chưa hiệu chỉnh trên CIFAR10 nhắm vào các mô hình riêng tư vi phân, với $\epsilon \in \{1, 10, 100, 1000\}$ và không có DP. Với giá trị $\epsilon$ lớn, tấn công đã hiệu chỉnh vẫn suy luận thành công các mẫu thành viên với precision cao ở recall thấp.]*

Kiểm tra định tính các ảnh thành viên của CIFAR10 bị suy luận với precision > 0,9 ở mô hình DP với $\epsilon = 1000$ nhưng không bị phát hiện với $\epsilon \le 10$ cho thấy các ảnh dễ tổn thương thường là mẫu không điển hình trong lớp của chúng, nhưng không phải ngoại lai cực đoan; kiểm tra thủ công thêm không cho thấy một quy luật chung nào cho các điểm dễ tổn thương nhất.

---

## 5. Kết luận (Conclusion)

Hiệu chỉnh độ khó là kỹ thuật hậu xử lý mạnh mẽ để cải thiện các tấn công suy luận thành viên hiện có. Hầu hết các tấn công dựa trên điểm số đều hưởng lợi lớn từ hiệu chỉnh, giúp tấn công đáng tin cậy hơn nhiều trong các chế độ FPR thấp và precision cao. Nghiên cứu cũng cho thấy các công trình trước đây tập trung đánh giá accuracy chưa phản ánh đầy đủ bức tranh, và phân tích chi tiết bằng đường cong ROC và precision-recall là quan trọng để thiết kế các tấn công suy luận thành viên đáng tin cậy.

Khi có quyền truy cập white-box vào mô hình mục tiêu, hiệu chỉnh qua forgetting là phương án thay thế hiệu quả cho việc huấn luyện các mô hình tham chiếu từ đầu, đồng thời tạo liên kết tường minh giữa suy luận thành viên và hiện tượng catastrophic forgetting đã được nghiên cứu kỹ. Nhóm tác giả hy vọng các công trình tương lai sẽ khai thác thêm mối liên hệ này để cải thiện state-of-the-art của MIA.

---

## Tài liệu tham khảo

Martin Abadi, Andy Chu, Ian Goodfellow, Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. Deep learning with differential privacy. In CCS, 2016.

Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020.

Nicholas Carlini, Chang Liu, Úlfar Erlingsson, Jernej Kos, and Dawn Song. The secret sharer: Evaluating and testing unintended memorization in neural networks. In 28th USENIX Security Symposium, pp. 267–284, 2019.

Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. Extracting training data from large language models. arXiv preprint arXiv:2012.07805, 2020.

Christopher A Choquette-Choo, Florian Tramer, Nicholas Carlini, and Nicolas Papernot. Label-only membership inference attacks. In International Conference on Machine Learning, pp. 1964–1974. PMLR, 2021.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Dheeru Dua and Casey Graff. UCI machine learning repository, 2017. URL http://archive.ics.uci.edu/ml.

Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Theory of cryptography conference, pp. 265–284. Springer, 2006.

Matt Fredrikson, Somesh Jha, and Thomas Ristenpart. Model inversion attacks that exploit confidence information and basic countermeasures. In Proceedings of the 22nd ACM SIGSAC conference on computer and communications security, pp. 1322–1333, 2015.

Matthew Fredrikson, Eric Lantz, Somesh Jha, Simon Lin, David Page, and Thomas Ristenpart. Privacy in pharmacogenetics: An end-to-end case study of personalized warfarin dosing. In 23rd USENIX Security Symposium, pp. 17–32, 2014.

Ian J Goodfellow, Mehdi Mirza, Da Xiao, Aaron Courville, and Yoshua Bengio. An empirical investigation of catastrophic forgetting in gradient-based neural networks. arXiv preprint arXiv:1312.6211, 2013.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

Nils Homer, Szabolcs Szelinger, Margot Redman, David Duggan, Waibhav Tembe, Jill Muehling, John V Pearson, Dietrich A Stephan, Stanley F Nelson, and David W Craig. Resolving individuals contributing trace amounts of DNA to highly complex mixtures using high-density SNP genotyping microarrays. PLoS genetics, 4(8):e1000167, 2008.

Matthew Jagielski, Jonathan Ullman, and Alina Oprea. Auditing differentially private machine learning: How private is private SGD? arXiv preprint arXiv:2006.07709, 2020.

Bargav Jayaraman, Lingxiao Wang, Katherine Knipmeyer, Quanquan Gu, and David Evans. Revisiting membership inference under realistic assumptions. Proceedings on Privacy Enhancing Technologies, 2021(2), 2021.

Jinyuan Jia, Ahmed Salem, Michael Backes, Yang Zhang, and Neil Zhenqiang Gong. Memguard: Defending against black-box membership inference attacks via adversarial examples. In Proceedings of the 2019 ACM SIGSAC conference on computer and communications security, pp. 259–274, 2019.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526, 2017.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

Klas Leino and Matt Fredrikson. Stolen memories: Leveraging model memorization for calibrated white-box membership inference. In 29th USENIX Security Symposium, pp. 1605–1622, 2020.

Zheng Li and Yang Zhang. Label-leaks: Membership inference attack with label. arXiv e-prints, pp. arXiv–2007, 2020.

Yunhui Long, Vincent Bindschaedler, Lei Wang, Diyue Bu, Xiaofeng Wang, Haixu Tang, Carl A Gunter, and Kai Chen. Understanding membership inferences on well-generalized learning models. arXiv preprint arXiv:1802.04889, 2018.

Ilya Mironov. Rényi differential privacy. In 2017 IEEE 30th Computer Security Foundations Symposium (CSF), pp. 263–275. IEEE, 2017.

Ilya Mironov, Kunal Talwar, and Li Zhang. Rényi differential privacy of the sampled gaussian mechanism. arXiv preprint arXiv:1908.10530, 2019.

Milad Nasr, Reza Shokri, and Amir Houmansadr. Comprehensive privacy analysis of deep learning: Passive and active white-box inference attacks against centralized and federated learning. In 2019 IEEE symposium on security and privacy (SP), pp. 739–753. IEEE, 2019.

Milad Nasr, Shuang Song, Abhradeep Thakurta, Nicolas Papernot, and Nicholas Carlini. Adversary instantiation: Lower bounds for differentially private machine learning. arXiv preprint arXiv:2101.04535, 2021.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Shahbaz Rezaei and Xin Liu. On the difficulty of membership inference attacks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7892–7900, 2021.

Alexandre Sablayrolles, Matthijs Douze, Cordelia Schmid, Yann Ollivier, and Hervé Jégou. White-box vs black-box: Bayes optimal strategies for membership inference. In International Conference on Machine Learning, pp. 5558–5567. PMLR, 2019.

Ahmed Salem, Yang Zhang, Mathias Humbert, Pascal Berrang, Mario Fritz, and Michael Backes. ML-leaks: Model and data independent membership inference attacks and defenses on machine learning models. arXiv preprint arXiv:1806.01246, 2018.

Issei Sato and Hiroshi Nakagawa. Approximation analysis of stochastic gradient Langevin dynamics by using Fokker-Planck equation and Ito process. In International Conference on Machine Learning, pp. 982–990. PMLR, 2014.

Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against machine learning models. In 2017 IEEE Symposium on Security and Privacy (SP), pp. 3–18. IEEE, 2017.

Liwei Song and Prateek Mittal. Systematic evaluation of privacy risks of machine learning models. In 30th USENIX Security Symposium (USENIX Security 21), pp. 2615–2632, 2021.

Mariya Toneva, Alessandro Sordoni, Remi Tachet des Combes, Adam Trischler, Yoshua Bengio, and Geoffrey J Gordon. An empirical study of example forgetting during deep neural network learning. arXiv preprint arXiv:1812.05159, 2018.

Samuel Yeom, Irene Giacomelli, Matt Fredrikson, and Somesh Jha. Privacy risk in machine learning: Analyzing the connection to overfitting. In 2018 IEEE 31st Computer Security Foundations Symposium (CSF), pp. 268–282. IEEE, 2018.

Ashkan Yousefpour, Igor Shilov, Alexandre Sablayrolles, Davide Testuggine, Karthik Prasad, Mani Malek, John Nguyen, Sayan Ghosh, Akash Bharadwaj, Jessica Zhao, et al. Opacus: User-friendly differential privacy library in pytorch. 2021.

Elias A Zerhouni and Elizabeth G Nabel. Protecting aggregate genomic data. Science, 2008.

---

## Phụ lục

### A. Tấn công gap đã hiệu chỉnh (Calibrated Gap Attack)

Phụ lục này khảo sát tác động của hiệu chỉnh độ khó lên gap attack (Yeom et al., 2018), nhằm giải thích trực giác vì sao hiệu chỉnh cải thiện AUC của nhiều tấn công nhưng vẫn giữ accuracy tương tự bản chưa hiệu chỉnh.

Giả sử có hai mô hình: *target* và *calib*. Gap attack đã hiệu chỉnh dự đoán một mẫu thuộc tập huấn luyện nếu mô hình target dự đoán đúng và mô hình calib dự đoán sai.

**Bảng 5: Hiệu năng của mô hình hiệu chỉnh và mô hình mục tiêu trên tập train (trái) và test (phải) theo phân tích đơn giản hóa.**

| Train: calib \ target | đúng | sai |
|---|---|---|
| **đúng** | $p_{\text{test}}$ | 0 |
| **sai** | $p_{\text{train}} - p_{\text{test}}$ | $1 - p_{\text{train}}$ |

| Test: calib \ target | đúng | sai |
|---|---|---|
| **đúng** | $p_{\text{test}}$ | 0 |
| **sai** | 0 | $1 - p_{\text{test}}$ |

**Phân tích đơn giản hóa.** Với hai giả thiết: (i) trên tập train, nếu mô hình target sai thì mô hình calib cũng sai; (ii) trên tập test, hai mô hình hoặc cùng đúng hoặc cùng sai trên mỗi mẫu. Khi đó toàn bộ tập test được dự đoán đúng và một phần $p_{\text{train}} - p_{\text{test}}$ của tập train được dự đoán đúng. Accuracy là:

$$\frac{1}{2}\left(1 + p_{\text{train}} - p_{\text{test}}\right) \tag{7}$$

**Bảng 6: Hiệu năng của mô hình hiệu chỉnh và mô hình mục tiêu trên tập train (trái) và test (phải) khi bỏ giả thiết.**

| Train: calib \ target | đúng | sai |
|---|---|---|
| **đúng** | $p_{\text{test}} - \epsilon_1$ | $\epsilon_1$ |
| **sai** | $p_{\text{train}} - p_{\text{test}} + \epsilon_1$ | $1 - p_{\text{train}} - \epsilon_1$ |

| Test: calib \ target | đúng | sai |
|---|---|---|
| **đúng** | $p_{\text{test}} - \epsilon_2$ | $\epsilon_2$ |
| **sai** | $\epsilon_2$ | $1 - p_{\text{test}} - \epsilon_2$ |

**Phân tích đầy đủ.** Bỏ các giả thiết trên, để điền Bảng 6 có 3 phương trình (tổng xác suất bằng 1; tổng hàng và cột "đúng" bằng $p_{\text{train}}$, tương ứng $p_{\text{test}}$); với bốn ẩn, còn lại một đại lượng chưa xác định cho mỗi bảng là $\epsilon_1$ và $\epsilon_2$. Accuracy của gap attack đã hiệu chỉnh là:

$$\frac{1}{2}\left(1 - \epsilon_2 + p_{\text{train}} - p_{\text{test}} + \epsilon_1\right) \tag{8}$$

Thực nghiệm trên CIFAR10 và CIFAR100 (có và không có data augmentation) cho thấy $\epsilon_1$ thường rất nhỏ (0%–2%), còn $\epsilon_2$ lớn hơn, khoảng 8%–12%.

- **Accuracy:** theo Phương trình 8, accuracy của gap attack đã hiệu chỉnh thấp hơn bản chưa hiệu chỉnh vì $\epsilon_2 > \epsilon_1$. Tuy nhiên accuracy của gap attack chưa hiệu chỉnh có thể phục hồi trivial bằng cách dự đoán mọi điểm mà mô hình target dự đoán đúng là thành viên, thay vì chỉ những điểm target đúng và calib sai.
- **AUC:** gap attack đã hiệu chỉnh cải thiện đánh đổi precision-recall: hiệu chỉnh cho phép xác định một tập con điểm thành viên với FPR thấp hơn đáng kể (Hình 7 và 8). Vì vậy AUC của gap attack trên CIFAR10 tăng từ trung bình 0,66 lên 0,71.

*[Hình 7: Phân phối điểm của Gap Attack so với Calibrated Gap Attack. "Target (Corr.)" nghĩa là lớp dự đoán đúng với mô hình target; "Calib (Incorr.)" nghĩa là lớp bị dự đoán sai trên mô hình hiệu chỉnh. Hai nhóm ngoài cùng bên trái ở biểu đồ đã hiệu chỉnh tương ứng với sự chia nhỏ nhóm bên trái ở biểu đồ gap attack gốc; quan hệ tương tự áp dụng cho hai nhóm ngoài cùng bên phải. Nhóm ngoài cùng bên trái của gap attack đã hiệu chỉnh cho thấy mức cải thiện TPR đạt được nhờ hiệu chỉnh, dẫn đến AUC cao hơn ở Hình 8.]*

*[Hình 8: Đường cong ROC của gap attack và calibrated gap attack. AUC của tấn công đã hiệu chỉnh cao hơn do TPR tăng ở FPR thấp.]*

### B. Tổng quát hóa và suy luận thành viên (Generalization and Membership Inference)

Suy luận thành viên đã hiệu chỉnh xét đại lượng $\ell(\theta', z) - \ell(\theta, z)$, trong đó $\theta$ là mô hình mục tiêu, $\theta'$ là mô hình hiệu chỉnh và $z$ là mẫu thuộc tập huấn luyện của $\theta$. Trong khi đó, tổng quát hóa (generalization) xét $\ell(\theta, z') - \ell(\theta, z)$ với $z'$ là mẫu không thuộc tập huấn luyện của $\theta$. Tác giả lập luận rằng $\ell(\theta', z)$ và $\ell(\theta, z')$ là các đại lượng khác nhau nhưng đều biểu diễn loss trên một mẫu nằm ngoài tập huấn luyện nên về mặt thống kê chúng nên hành xử tương tự nhau.

### C. Thí nghiệm bổ sung (Additional Experiments)

Bảng 7 và 8 cho thấy tác động của việc dùng nhiều mô hình hiệu chỉnh thay vì một. Bảng 9 và 10 cho thấy tác động của hiệu chỉnh độ khó lên các điểm thành viên khác không có trong phần chính: điểm entropy và modified entropy (Song & Mittal, 2021), cùng các tấn công Merlin và Morgan (Jayaraman et al., 2021). Theo Song & Mittal (2021), Hình 9 cho thấy các thước đo entropy liên quan chặt chẽ tới điểm confidence và loss tương ứng; điều này giải thích vì sao kết quả AUC và accuracy có hành vi giống điểm confidence và loss khi hiệu chỉnh.

**Merlin và Morgan.** Hai tấn công này nhắm tới giá trị dự đoán dương (PPV) cao thay vì accuracy, nên accuracy và AUC thấp hơn các tấn công khác. Ở các tấn công này, phân phối thành viên và không-thành-viên thường không thể phân biệt, ngoại trừ một số ít điểm được xác định với PPV rất cao. Cụ thể, tấn công Morgan xuất ra giá trị nhị phân được chọn để tối đa hóa PPV; khi thành công, phân phối rất lệch: một vài điểm có PPV gần 1 (giá trị 1) và mọi điểm còn lại nhận giá trị 0, nên accuracy và AUC xấp xỉ đoán ngẫu nhiên. Bảng 11 báo cáo PPV của các phiên bản Merlin và Morgan chưa hiệu chỉnh và đã hiệu chỉnh; trong nhiều trường hợp hiệu chỉnh cải thiện PPV tối đa đạt được.

**Bảng 7: AUC của các tấn công dựa trên điểm số trước và sau hiệu chỉnh độ khó với 10 mô hình hiệu chỉnh.** Hiệu chỉnh (Cal.) cải thiện AUC nhất quán và đáng kể; calibration-via-forgetting (Cal.-F) hy sinh một phần nhỏ mức cải thiện để đổi lấy hiệu quả tính toán.

| Tập dữ liệu | Gap | Loss Orig | Loss Cal. | Loss Cal.-F | GN Orig | GN Cal. | GN Cal.-F | Conf. Orig | Conf. Cal. | Conf. Cal.-F |
|---|---|---|---|---|---|---|---|---|---|---|
| Credit | 0,589 | 0,542 | 0,668 | 0,650 | 0,513 | 0,597 | 0,596 | 0,501 | 0,569 | 0,550 |
| Hep. | 0,561 | 0,514 | 0,562 | 0,534 | 0,515 | 0,557 | 0,568 | 0,513 | 0,582 | 0,573 |
| Adult | 0,534 | 0,516 | 0,559 | 0,539 | 0,510 | 0,527 | 0,520 | 0,507 | 0,518 | 0,513 |
| MNIST | 0,506 | 0,505 | 0,527 | 0,510 | 0,504 | 0,525 | 0,517 | 0,503 | 0,517 | 0,509 |
| CIFAR10 | 0,663 | 0,676 | 0,739 | 0,696 | 0,678 | 0,796 | 0,759 | 0,629 | 0,658 | 0,646 |
| CIFAR100 | 0,854 | 0,911 | 0,914 | 0,916 | 0,912 | 0,933 | 0,918 | 0,852 | 0,721 | 0,746 |
| CIFAR10 (aug.) | 0,603 | 0,603 | 0,704 | 0,693 | 0,603 | 0,746 | 0,725 | 0,560 | 0,581 | 0,578 |
| CIFAR100 (aug.) | 0,784 | 0,812 | 0,878 | 0,873 | 0,811 | 0,891 | 0,877 | 0,679 | 0,665 | 0,635 |

**Bảng 8: Accuracy của các tấn công dựa trên điểm số trước và sau hiệu chỉnh độ khó với 10 mô hình hiệu chỉnh.** Trong hầu hết các trường hợp, tấn công đã hiệu chỉnh tương đương hoặc tốt hơn bản chưa hiệu chỉnh về accuracy.

| Tập dữ liệu | Gap | Loss Orig | Loss Cal. | Loss Cal.-F | GN Orig | GN Cal. | GN Cal.-F | Conf. Orig | Conf. Cal. | Conf. Cal.-F |
|---|---|---|---|---|---|---|---|---|---|---|
| Credit | 0,589 | 0,617 | 0,624 | 0,605 | 0,569 | 0,586 | 0,584 | 0,557 | 0,563 | 0,553 |
| Hep. | 0,561 | 0,574 | 0,587 | 0,574 | 0,574 | 0,590 | 0,593 | 0,574 | 0,601 | 0,594 |
| Adult | 0,534 | 0,536 | 0,537 | 0,525 | 0,518 | 0,520 | 0,516 | 0,512 | 0,514 | 0,511 |
| MNIST | 0,506 | 0,508 | 0,517 | 0,510 | 0,508 | 0,516 | 0,512 | 0,507 | 0,513 | 0,508 |
| CIFAR10 | 0,664 | 0,712 | 0,657 | 0,663 | 0,719 | 0,707 | 0,703 | 0,642 | 0,614 | 0,623 |
| CIFAR100 | 0,854 | 0,911 | 0,832 | 0,846 | 0,915 | 0,885 | 0,896 | 0,820 | 0,675 | 0,691 |
| CIFAR10 (aug.) | 0,602 | 0,609 | 0,631 | 0,639 | 0,610 | 0,668 | 0,657 | 0,562 | 0,567 | 0,564 |
| CIFAR100 (aug.) | 0,784 | 0,785 | 0,791 | 0,801 | 0,788 | 0,831 | 0,822 | 0,647 | 0,616 | 0,606 |

**Bảng 9: Kết quả AUC cho các điểm Entropy, Modified Entropy, Merlin và Morgan** (cột Gap là đường cơ sở; mỗi điểm gồm Orig. / Cal. / Cal.-F). Các tấn công Merlin và Morgan được đề xuất nhằm tối đa hóa PPV thay vì accuracy, điều này phản ánh trong kết quả. Với tập Hepatitis, Merlin và Morgan có độ lệch chuẩn lớn tới 0,09 nên khó so sánh. Với Morgan, giá trị gần 0,500 là do tấn công xác định một số ít điểm với FPR = 0 và đạt mục tiêu tối đa hóa PPV; giá trị cao hơn cho thấy điều đó không thực hiện được, và hành vi mong muốn sau đó được cải thiện nhờ hiệu chỉnh nhờ tạo ra vùng precision cao (ví dụ CIFAR100 (aug.) và Bảng 11).

| Tập dữ liệu | Gap | Entropy (O / C / C-F) | Mod. Entropy (O / C / C-F) | Merlin (O / C / C-F) | Morgan (O / C / C-F) |
|---|---|---|---|---|---|
| Credit | 0,588 | 0,525 / 0,591 / 0,597 | 0,562 / 0,637 / 0,621 | 0,514 / 0,523 / 0,527 | 0,505 / 0,502 / 0,508 |
| Hep. | 0,544 | 0,516 / 0,520 / 0,534 | 0,534 / 0,581 / 0,587 | 0,472 / 0,476 / 0,484 | 0,524 / 0,544 / 0,526 |
| Adult | 0,514 | 0,509 / 0,512 / 0,513 | 0,519 / 0,557 / 0,530 | 0,500 / 0,502 / 0,507 | 0,500 / 0,500 / 0,500 |
| MNIST | 0,506 | 0,507 / 0,512 / 0,510 | 0,510 / 0,512 / 0,513 | 0,497 / 0,503 / 0,501 | 0,503 / 0,502 / 0,503 |
| CIFAR10 | 0,662 | 0,627 / 0,634 / 0,635 | 0,675 / 0,730 / 0,710 | 0,506 / 0,504 / 0,504 | 0,500 / 0,500 / 0,500 |
| CIFAR100 | 0,858 | 0,849 / 0,701 / 0,750 | 0,910 / 0,912 / 0,902 | 0,373 / 0,530 / 0,510 | 0,500 / 0,500 / 0,500 |
| CIFAR10 (aug.) | 0,602 | 0,563 / 0,598 / 0,581 | 0,603 / 0,691 / 0,688 | 0,500 / 0,516 / 0,514 | 0,555 / 0,500 / 0,500 |
| CIFAR100 (aug.) | 0,785 | 0,664 / 0,700 / 0,706 | 0,798 / 0,851 / 0,868 | 0,495 / 0,507 / 0,509 | 0,644 / 0,500 / 0,500 |

**Bảng 10: Kết quả accuracy cho các điểm Entropy, Modified Entropy, Merlin và Morgan** (định dạng như Bảng 9). Ghi chú về PPV và Morgan giống Bảng 9; độ lệch chuẩn của Hepatitis với Merlin và Morgan lên tới 0,07.

| Tập dữ liệu | Gap | Entropy (O / C / C-F) | Mod. Entropy (O / C / C-F) | Merlin (O / C / C-F) | Morgan (O / C / C-F) |
|---|---|---|---|---|---|
| Credit | 0,588 | 0,525 / 0,591 / 0,597 | 0,622 / 0,633 / 0,629 | 0,537 / 0,537 / 0,543 | 0,505 / 0,502 / 0,508 |
| Hep. | 0,544 | 0,516 / 0,520 / 0,534 | 0,597 / 0,597 / 0,591 | 0,544 / 0,571 / 0,558 | 0,524 / 0,544 / 0,526 |
| Adult | 0,514 | 0,509 / 0,512 / 0,513 | 0,537 / 0,535 / 0,533 | 0,505 / 0,505 / 0,507 | 0,500 / 0,500 / 0,500 |
| MNIST | 0,506 | 0,507 / 0,512 / 0,510 | 0,510 / 0,509 / 0,509 | 0,502 / 0,505 / 0,504 | 0,503 / 0,502 / 0,503 |
| CIFAR10 | 0,662 | 0,627 / 0,634 / 0,635 | 0,710 / 0,661 / 0,664 | 0,515 / 0,504 / 0,511 | 0,500 / 0,500 / 0,500 |
| CIFAR100 | 0,858 | 0,849 / 0,701 / 0,750 | 0,910 / 0,850 / 0,839 | 0,501 / 0,532 / 0,525 | 0,500 / 0,500 / 0,500 |
| CIFAR10 (aug.) | 0,602 | 0,563 / 0,598 / 0,581 | 0,608 / 0,632 / 0,634 | 0,501 / 0,516 / 0,513 | 0,555 / 0,500 / 0,500 |
| CIFAR100 (aug.) | 0,785 | 0,664 / 0,700 / 0,706 | 0,771 / 0,801 / 0,793 | 0,502 / 0,522 / 0,524 | 0,644 / 0,500 / 0,500 |

**Bảng 11: Kết quả giá trị dự đoán dương (PPV) và số điểm được xác định với FPR = 0** (định dạng: PPV (số điểm ± độ lệch chuẩn)). Trong nhiều trường hợp hiệu chỉnh cải thiện PPV. Kỹ thuật chọn ngưỡng của các tấn công nhằm tối đa hóa PPV, theo Jayaraman et al. (2021).

| Tập dữ liệu | Merlin Orig. | Merlin Cal. | Merlin Cal.-F | Morgan Orig. | Morgan Cal. | Morgan Cal.-F |
|---|---|---|---|---|---|---|
| Credit | 0,92 (2,80±2,04) | 0,71 (1,0±1,26) | 0,84 (2,0±2,60) | 0,64 (0,4±0,8) | 0,80 (0,8±0,75) | 0,92 (1,4±0,8) |
| Hep. | 0,76 (1,0±1,22) | 1,0 (1,0±0,00) | 0,81 (1,0±0,89) | 1,0 (1,5±0,87) | 1,0 (2,75±2,05) | 1,0 (1,0±0,0) |
| Adult | 0,72 (0,2±0,40) | 0,76 (1,0±1,55) | 0,77 (0,4±0,49) | 0,78 (0,2±0,4) | 0,80 (0,6±0,49) | 0,80 (1,2±1,16) |
| MNIST | 0,51 (0,0±0,0) | 0,64 (0,0±0,0) | 0,60 (0,0±0,0) | 0,52 (0,0±0,0) | 0,64 (0,0±0,0) | 0,66 (0,0±0,0) |
| CIFAR10 | 0,83 (1,2±1,60) | 0,79 (0,6±0,55) | 0,78 (0,4±0,55) | 0,83 (1,2±1,60) | 1,0 (1,4±0,8) | 1,0 (2,8±2,4) |
| CIFAR100 | 1,0 (2,4±1,14) | 1,0 (1,0±0,0) | 0,94 (0,6±0,55) | 1,0 (2,4±1,14) | 1,0 (4,0±1,67) | 1,0 (3,0±1,26) |
| CIFAR10 (aug.) | 0,50 (0,0±0,0) | 0,85 (0,0±0,0) | 0,84 (0,8±1,17) | 0,58 (0,0±0,0) | 1,0 (1,0±0,0) | 1,0 (1,2±0,4) |
| CIFAR100 (aug.) | 0,50 (0,0±0,0) | 1,0 (2,0±1,55) | 1,0 (1,4±0,49) | 0,72 (0,0±0,0) | 0,98 (1,2±0,98) | 1,0 (3,4±1,36) |

*[Hình 9: So sánh điểm entropy và modified entropy với điểm confidence và loss tương ứng trên CIFAR10 không dùng data augmentation. Song & Mittal (2021) lưu ý rằng entropy tương tự confidence, còn modified entropy tương tự cross-entropy loss.]*
