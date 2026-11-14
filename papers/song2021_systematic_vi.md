# Đánh giá có hệ thống các rủi ro quyền riêng tư của mô hình học máy

*Systematic Evaluation of Privacy Risks of Machine Learning Models*

**Tác giả:** Liwei Song, Prateek Mittal (Princeton University)
**Hội nghị/năm:** USENIX Security Symposium 2021 (bản arXiv:2003.10595v2)

> Bản tóm tắt chi tiết bằng tiếng Việt (do AI viết, không phải bản dịch nguyên văn)

---

## Tóm tắt (Abstract)

- Các công trình trước về **membership inference attack (MIA – tấn công suy luận thành viên)** có thể **đánh giá thấp** rủi ro quyền riêng tư vì hai lý do: (i) chỉ dùng bộ phân loại mạng nơ-ron (NN) tự huấn luyện làm kẻ tấn công, và (ii) chỉ báo cáo kết quả *tổng hợp* (aggregate) như attack accuracy.
- Đề xuất giải pháp thứ nhất: một **bộ benchmark** gồm các tấn công không dựa trên NN ("metric-based attacks"): cải tiến các tấn công sẵn có bằng **ngưỡng riêng theo từng lớp (class-dependent threshold)** và đề xuất tấn công mới dựa trên **modified prediction entropy**. Ngoài ra đề xuất hai khuyến nghị khi đánh giá phòng thủ: so sánh với **early stopping** và xét **adaptive adversary** (kẻ tấn công biết cơ chế phòng thủ).
- Với benchmark này, các phòng thủ hiện có (adversarial regularization, MemGuard) kém hiệu quả hơn báo cáo ban đầu.
- Giải pháp thứ hai: **privacy risk score** – chỉ số đo rủi ro *theo từng mẫu*, chính là xác suất hậu nghiệm mẫu đó thuộc tập huấn luyện. Bài chứng minh chỉ số này phản ánh đúng xác suất thành viên, rủi ro giữa các mẫu là không đồng đều, và phân tích mối liên hệ với model sensitivity, generalization error, feature embedding.
- Mã nguồn được công bố và đã được tích hợp vào thư viện TensorFlow Privacy của Google.

---

## 1. Giới thiệu (Introduction)

**Bối cảnh.** Mô hình ML có xu hướng ghi nhớ dữ liệu huấn luyện nhạy cảm. Bài tập trung vào MIA: kẻ tấn công đoán một mẫu đầu vào có nằm trong tập huấn luyện của mô hình mục tiêu hay không. Ví dụ: nếu một người nằm trong tập huấn luyện của mô hình phân tích y tế của bệnh viện thì suy ra người đó từng là bệnh nhân. MIA cũng dùng để lượng hóa mức bảo vệ của differential privacy (DP) và giúp chọn tham số riêng tư. Shokri và cộng sự (2017) mở đầu với black-box attack bằng **shadow model** + bộ phân loại NN; sau đó mở rộng sang mô hình sinh, mô hình DP, học phân tán (federated), và white-box.

**Các phòng thủ hiện có.**
- *Adversarial regularization* (Nasr 2018): đưa kẻ tấn công NN vào quá trình huấn luyện như một số hạng điều chuẩn đối kháng.
- *MemGuard* (Jia 2019): không huấn luyện lại, chỉ thêm nhiễu được thiết kế khéo vào vector dự đoán để đánh lừa bộ phân loại tấn công.
- Cả hai báo cáo attack accuracy gần 50% (đoán ngẫu nhiên).

**Hai hạn chế của đánh giá trước đây:**
1. Chỉ dùng NN attack classifier tự huấn luyện → phụ thuộc siêu tham số (số lớp ẩn, learning rate...) và có thể đánh giá thấp rủi ro.
2. Chỉ xét rủi ro tổng hợp → thiếu hiểu biết chi tiết về từng mẫu.

**Kết quả gây chú ý (Bảng 1 gốc):** với benchmark của bài, attack accuracy trên mô hình đã được phòng thủ cao hơn nhiều so với báo cáo.

| Phòng thủ | Dataset | Attack acc báo cáo | Attack acc (benchmark của bài) |
|---|---|---|---|
| Adversarial regularization | Purchase100 | 51.6% | 59.5% |
| Adversarial regularization | Texas100 | 51.0% | 58.6% |
| MemGuard | Location30 | 50.1% | 69.1% |
| MemGuard | Texas100 | 50.3% | 74.2% |

*[Hình 1: CDF của privacy risk score trên mô hình không phòng thủ (Purchase100, Location30, CIFAR100); phân bố trải rộng, cho thấy rủi ro giữa các mẫu không đồng nhất.]*

**Đóng góp chính:**
1. Bộ tấn công không dựa NN: cải tiến bằng ngưỡng theo lớp; tấn công mới dựa trên modified prediction entropy; khuyến nghị so sánh với early stopping và xét adaptive adversary.
2. Với benchmark này: adversarial regularization không tốt hơn early stopping; đánh giá MemGuard thiếu adversary thích nghi; white-box attack chỉ hơn black-box benchmark một chút; ngưỡng theo lớp luôn tốt hơn ngưỡng chung; modified entropy luôn tốt hơn entropy thường.
3. Đề xuất **privacy risk score** cho phân tích mịn theo từng mẫu.
4. Kiểm chứng thực nghiệm chỉ số này và phân tích sâu tương quan với sensitivity, generalization error, embedding.

---

## 2. Nền tảng và các công trình liên quan (Background and Related Work)

### 2.1 Kiến thức ML cơ bản và ký hiệu

Mô hình $F_\theta:\mathbb{R}^d\to\mathbb{R}^k$ có $d$ đặc trưng đầu vào, $k$ lớp, tham số $\theta$. Với mẫu $z=(x,y)$, mô hình trả về vector xác suất $F(x)$ (tổng bằng 1); nhãn dự đoán là $\hat y=\arg\max_i F(x)_i$. Huấn luyện bằng cực tiểu hóa loss trung bình:

$$\min_\theta \frac{1}{|D_{tr}|}\sum_{z\in D_{tr}} \ell(F_\theta, z)$$

- $D_{tr}$: tập huấn luyện; $\ell$: hàm loss (vd. cross-entropy).

### 2.2 Membership inference attacks

Mục tiêu: xác định mẫu cho trước có được dùng để huấn luyện hay không.

**2.2.1 Black-box attack.** Kẻ tấn công chỉ thấy đầu ra $F(x)$.
- Shokri et al.: **shadow training** gồm (1) huấn luyện nhiều shadow model mô phỏng mô hình mục tiêu; (2) lấy đầu ra của chúng trên dữ liệu train/test của chính shadow để tạo tập có nhãn member/non-member; (3) huấn luyện các bộ phân loại NN (mỗi lớp một bộ) làm attack model.
- Salem et al.: chỉ cần *một* shadow model vẫn thành công.
- Nếu kẻ tấn công biết một phần tập train/test của mục tiêu, có thể huấn luyện trực tiếp attack classifier trên đó (Nasr et al.); Nasr còn dùng one-hot nhãn làm đặc trưng đầu vào và chỉ một NN cho mọi lớp.
- **Tấn công không dựa NN (metric-based):** Leino et al. dùng *độ đúng* của dự đoán; Yeom et al. và Song et al. dùng ngưỡng trên *confidence* của nhãn đúng; Shokri et al. chỉ ra phân bố *entropy* khác biệt giữa train và test (nhưng chưa dùng để tấn công).
- Tuy vậy, nhiều bài báo vẫn chỉ dùng NN attack → đánh giá thấp. Bài này cải tiến bằng ngưỡng theo lớp và đề xuất metric mới có xét nhãn thật.

**2.2.2 White-box attack.** Nasr et al.: kẻ tấn công biết kiến trúc và trọng số. Chỉ gộp đầu ra trung gian không giúp hơn black-box; nhưng thêm **gradient của loss theo tham số** $\partial\ell(F_\theta,z)/\partial\theta$ làm đặc trưng thì tăng accuracy. Bài này cho thấy khoảng cách white-box/black-box nhỏ hơn nhiều so với ước tính trước đó.

### 2.3 Phòng thủ chống MIA

- L2 regularization và dropout giảm overfitting nên giảm phần nào rủi ro nhưng mô hình vẫn dễ bị tấn công. **Differential privacy** có bảo đảm lý thuyết nhưng làm giảm accuracy mạnh ở các tham số riêng tư mong muốn.
- Hai phòng thủ chuyên dụng được xét:

**2.3.1 Adversarial regularization (AdvReg) [Nasr 2018].** Huấn luyện luân phiên ở mỗi bước:
1. Cập nhật attack classifier $I$ (tham số $\vartheta$, đầu vào là $F(x)$ và nhãn $y$, đầu ra trong $[0,1]$, >0.5 là member) để cực đại hóa "membership inference gain" trên tập train $D_{tr}$ và tập validation $D_{val}$:

$$\arg\max_\vartheta \frac{\sum_{z\in D_{tr}}\log I(F(x),y)}{2|D_{tr}|}+\frac{\sum_{z\in D_{val}}\log(1-I(F(x),y))}{2|D_{val}|}$$

2. Cập nhật mô hình $F$ để vừa cực tiểu hóa loss vừa làm gain của $I$ giảm:

$$\arg\min_\theta \frac{1}{|D_{tr}|}\sum_{z\in D_{tr}}\ell(F(x),y)+\lambda\log I(F(x),y)$$

- $\lambda$: hệ số phạt rủi ro quyền riêng tư.

**2.3.2 MemGuard [Jia 2019].** Không sửa quá trình huấn luyện; với mô hình đã huấn luyện, thêm nhiễu $n$ vào $F(x)$ sao cho attack classifier $I$ bị nhầm nhưng nhãn dự đoán không đổi:

$$\min_n d(F(x)+n,F(x))\ \text{s.t.}\ \arg\max_i(F(x)_i+n_i)=\arg\max_iF(x)_i,\ I(F(x)+n)=0.5,\ F(x)_i+n_i\ge0,\ \textstyle\sum_i n_i=0$$

- $d$: khoảng cách giữa dự đoán gốc và dự đoán nhiễu; các ràng buộc đảm bảo: nhãn không đổi, $I$ không phân biệt được (đầu ra 0.5), vector hợp lệ.

Cả hai bài đều đánh giá bằng NN attack; phần 3 chỉ ra điều này đánh giá thấp rủi ro.

---

## 3. Đánh giá có hệ thống rủi ro MIA (Systematically Evaluating MIA Privacy Risks)

### 3.1 Bộ benchmark tấn công (metric-based)

Các tấn công này chỉ cần quan sát đầu ra (black-box), tính một metric trên dự đoán rồi so sánh với ngưỡng. Ưu điểm: dễ cài đặt, không cần tinh chỉnh siêu tham số; chỉ cần đặt ngưỡng bằng shadow training. Ký hiệu $I(\cdot)$ trả về 1 nếu suy ra là member.

**3.1.1 Tấn công có sẵn – dựa trên độ đúng (correctness).** Suy luận member nếu mẫu được dự đoán đúng (vì mô hình khớp tốt dữ liệu train nhưng tổng quát kém hơn trên test):

$$I_{corr}(F,(x,y))=\mathbb{1}\{\arg\max_i F(x)_i=y\}$$

**3.1.2 Cải tiến bằng ngưỡng theo lớp (class-dependent thresholds).**

*Confidence-based:* member nếu độ tin cậy nhãn đúng đủ lớn:

$$I_{conf}(F,(x,y))=\mathbb{1}\{F(x)_y\ge\tau_y\}$$

- $\tau_y$: ngưỡng riêng cho lớp $y$ (các công trình trước dùng một ngưỡng chung). Lý do: dữ liệu mất cân bằng, mô hình có mức tự tin khác nhau giữa các lớp.

Cách học $\tau_y$ bằng shadow training (3 bước):
1. Huấn luyện shadow model mô phỏng mô hình mục tiêu.
2. Lấy confidence của shadow model trên dữ liệu shadow-train và shadow-test.
3. Với từng lớp $y$, chọn ngưỡng cho accuracy cao nhất khi phân biệt shadow-train và shadow-test.

*Entropy-based* (chưa có công trình dùng để tấn công): member nếu entropy dự đoán nhỏ (mẫu train có đầu ra gần one-hot):

$$I_{entr}(F,(x,y))=\mathbb{1}\Big\{-\sum_iF(x)_i\log F(x)_i\le\hat\tau_y\Big\}$$

- $\hat\tau_y$ cũng theo lớp và học bằng shadow training.

**3.1.3 Tấn công mới: modified prediction entropy.** Vấn đề của entropy thường: không chứa nhãn thật – dự đoán đúng với xác suất 1 và dự đoán sai hoàn toàn với xác suất 1 đều cho entropy 0. Cần metric (1) giảm đơn điệu theo xác suất nhãn đúng $F(x)_y$; (2) tăng đơn điệu theo xác suất mỗi nhãn sai. Vì $-x\log x$ không đơn điệu, còn $-(1-x)\log x$ giảm đơn điệu và $-x\log(1-x)$ tăng đơn điệu, định nghĩa:

$$\mathrm{Mentr}(F(x),y)=-(1-F(x)_y)\log F(x)_y-\sum_{i\ne y}F(x)_i\log(1-F(x)_i)$$

- Dự đoán đúng với xác suất 1 → 0; dự đoán sai chắc chắn → vô cùng.

Tấn công:

$$I_{Mentr}(F,(x,y))=\mathbb{1}\{\mathrm{Mentr}(F(x),y)\le\check\tau_y\}$$

- $\check\tau_y$: ngưỡng theo lớp, học bằng shadow training. Thực nghiệm: luôn vượt trội entropy thường.

### 3.2 Đánh giá phòng thủ một cách nghiêm ngặt

Ngoài benchmark tấn công, bài đưa ra hai khuyến nghị.

**3.2.1 So sánh với early stopping.** Khi số epoch tăng, train/test error giảm nhưng mô hình cũng ghi nhớ nhiều hơn nên dễ bị MIA hơn. Early stopping (dừng sớm, chấp nhận giảm nhẹ accuracy) là baseline để đánh đổi accuracy–privacy. Mọi phòng thủ làm giảm accuracy để giảm rủi ro nên được so với mô hình không phòng thủ dừng sớm ở epoch có accuracy tương đương.

*[Hình 2: test accuracy theo epoch của classifier Purchase100 không phòng thủ và có AdvReg; mô hình AdvReg nên được so với mô hình không phòng thủ ở epoch sớm có accuracy tương đương.]*

**3.2.2 Adaptive attack.** Phải đặt adversary ở bước cuối của "cuộc chạy đua": kẻ tấn công biết cơ chế phòng thủ, huấn luyện shadow model *có áp dụng phòng thủ đó*, rồi học attack classifier hoặc ngưỡng từ các shadow model đã được bảo vệ, sau đó tấn công mô hình mục tiêu đã phòng thủ. Phòng thủ chỉ tốt trước tấn công không thích nghi thì chưa chứng tỏ hiệu quả.

### 3.3 Kết quả thực nghiệm

Mẫu được lấy từ tập train hoặc test với xác suất 0.5 mỗi bên, nên đoán ngẫu nhiên cho 50% accuracy. Bài đánh giá lại hai phòng thủ và white-box attack.

#### 3.3.1 Datasets

| Dataset | Mô tả | Quy mô / cách dùng |
|---|---|---|
| Purchase100 | Hồ sơ mua sắm (Kaggle), 600 đặc trưng nhị phân, 100 lớp "phong cách mua" | 197,324 mẫu; 19,732 mẫu (10%) để train |
| Texas100 | Hồ sơ xuất viện Texas, 6,170 đặc trưng nhị phân, 100 thủ thuật phổ biến nhất | 67,330 mẫu; 10,000 mẫu để train |
| Location30 | Check-in Foursquare, 446 đặc trưng nhị phân (bài ghi nhầm 466 ở một chỗ), 30 lớp geosocial | 5,010 mẫu; 1,000 mẫu để train |
| CIFAR100 | Ảnh 32×32, 100 lớp | 500 ảnh/lớp để train, 100/lớp để test |

Chọn các dataset này để so sánh công bằng với các bài trước. Do ba dataset đầu là dữ liệu nhị phân, phụ lục A thêm dữ liệu ảnh phức tạp hơn.

#### 3.3.2 Đánh giá lại adversarial regularization

- Mô hình: mạng fully-connected 4 lớp ẩn (1024, 512, 256, 128), kích hoạt Tanh; huấn luyện cả mô hình có và không có phòng thủ trên Purchase100 và Texas100.
- AdvReg làm giảm accuracy: Purchase100 test 80.9% → 76.6%; Texas100 52.3% → 46.4%.
- Mô hình **early stopping** được lấy bằng cách lưu mô hình không phòng thủ ở mọi epoch và chọn epoch có accuracy gần với mô hình có AdvReg.

**Bảng 2 (rút gọn): attack accuracy (%)**

| Dataset | Phòng thủ | Train acc | Test acc | NN attack [31] | $I_{corr}$ | $I_{conf}$ | $I_{entr}$ | $I_{Mentr}$ |
|---|---|---|---|---|---|---|---|---|
| Purchase100 | Không | 99.8 | 80.9 | 67.6 | 59.5 | 67.1 | 65.7 | 67.1 |
| Purchase100 | AdvReg | 92.7 | 76.6 | 51.6 | 58.1 | 59.4 | 55.8 | 59.5 |
| Purchase100 | Early stopping | 92.9 | 76.4 | N.A. | 58.2 | 59.2 | 55.9 | 59.1 |
| Texas100 | Không | 81.0 | 52.3 | 63.0 | 64.4 | 67.8 | 60.2 | 67.7 |
| Texas100 | AdvReg | 56.6 | 46.4 | 51.0 | 55.1 | 58.6 | 53.5 | 58.6 |
| Texas100 | Early stopping | 59.3 | 47.9 | N.A. | 55.7 | 59.4 | 54.0 | 59.5 |

Nhận xét:
- Mô hình có phòng thủ vẫn dễ bị tấn công: 59.5% và 58.6% so với 51.6% và 51.0% mà Nasr báo cáo; phòng thủ chỉ hạ accuracy tấn công từ khoảng 67% xuống khoảng 59%.
- Trên mọi mô hình trừ Purchase100 không phòng thủ, tấn công tốt nhất của benchmark vượt NN attack của Nasr.
- **AdvReg không tốt hơn early stopping** (attack accuracy gần như bằng nhau, 59.2%/59.5% với early stopping). Early stopping hiệu chỉnh được để khớp accuracy mô hình phòng thủ, còn AdvReg thì accuracy khó kiểm soát nên so sánh không công bằng.
- $I_{Mentr}$ luôn hơn $I_{entr}$ và thuộc nhóm tốt nhất.

**Bảng 3 (rút gọn): ngưỡng theo lớp vs ngưỡng chung, Texas100 (attack acc %)**

| Tấn công | Không phòng thủ | AdvReg | Early stopping |
|---|---|---|---|
| $I_{conf}$ (chung) | 64.7 | 55.5 | 55.8 |
| $I_{conf}$ (theo lớp) | 67.8 | 58.6 | 59.4 |
| $I_{entr}$ (chung) | 58.3 | 52.9 | 53.2 |
| $I_{entr}$ (theo lớp) | 60.2 | 53.5 | 54.0 |
| $I_{Mentr}$ (chung) | 64.8 | 55.4 | 55.9 |
| $I_{Mentr}$ (theo lớp) | 67.7 | 58.6 | 59.5 |

Ngưỡng theo lớp tăng attack accuracy 1–4%.

#### 3.3.3 Đánh giá lại MemGuard

- Mô hình: FC 4 lớp ẩn (1024, 512, 256, 128), ReLU, trên Location30 và Texas100. MemGuard không đổi accuracy nên không cần so với early stopping.
- Jia et al. dùng 6 NN attack; bài lấy accuracy cao nhất trong 6 cái.

**Bảng 4 (rút gọn): attack accuracy (%)**

| Dataset | Phòng thủ | Train / Test acc | NN attack [20] | $I_{corr}$ | $I_{conf}$ | $I_{entr}$ | $I_{Mentr}$ |
|---|---|---|---|---|---|---|---|
| Location30 | Không | 100 / 60.7 | 81.1 | 68.7 | 76.3 | 61.6 | 78.1 |
| Location30 | MemGuard | 100 / 60.7 | 50.1 | 68.7 | 69.1 | 52.1 | 68.8 |
| Texas100 | Không | 99.95 / 51.77 | 74.0 | 74.2 | 79.0 | 66.6 | 79.4 |
| Texas100 | MemGuard | 99.95 / 51.77 | 50.3 | 74.2 | 74.1 | 54.6 | 74.0 |

- Mô hình có MemGuard vẫn bị tấn công với 69.1% (Location30) và 74.2% (Texas100). Phòng thủ chỉ giảm accuracy tấn công khoảng 12% (Location30) và 5% (Texas100).
- $I_{Mentr}$ vẫn vượt $I_{entr}$; ở Location30 và Texas100 không phòng thủ, $I_{Mentr}$ vượt hẳn $I_{conf}$.
- **Vì sao Jia et al. thấy thất bại của tấn công?** 4/6 tấn công của họ là *không thích nghi* (kẻ tấn công không biết phòng thủ). Hai tấn công thích nghi còn lại chưa đặt adversary ở bước cuối: (a) làm tròn dự đoán đến một chữ số thập phân để loại nhiễu – hiệu quả giảm khi nhiễu lớn; (b) huấn luyện attack classifier bằng robust training (Madry) – nhưng cách này tổng quát kém, nhiễu vẫn có thể làm sai dự đoán trên test. Kẻ tấn công thực sự phải biết thuật toán sinh nhiễu và huấn luyện trên shadow model đã được phòng thủ.

#### 3.3.4 Đánh giá lại white-box attack [Nasr 2019]

- Mô hình: Purchase100, Texas100 (như 3.3.2, không phòng thủ) và CIFAR100 (DenseNet pretrained công khai).

**Bảng 5 (rút gọn): attack accuracy (%)**

| Dataset | White-box [32] | Black-box NN [32] | $I_{corr}$ | $I_{conf}$ | $I_{entr}$ | $I_{Mentr}$ |
|---|---|---|---|---|---|---|
| Purchase100 | 73.4 | 67.6 | 59.5 | 67.1 | 65.7 | 67.1 |
| Texas100 | 68.3 | 63.0 | 64.4 | 67.8 | 60.2 | 67.7 |
| CIFAR100 | 74.3 | 67.7 | 58.5 | 73.7 | 73.3 | 73.6 |

- White-box chỉ hơn tấn công black-box tốt nhất của benchmark 0.5% (Texas100) và 0.6% (CIFAR100); chênh 5.8% ở Purchase100. Kết quả tương đồng với Shejwalkar & Houmansadr.

---

## 4. Phân tích mịn về rủi ro quyền riêng tư (Fine-Grained Analysis on Privacy Risks)

Các công trình trước chỉ báo cáo kết quả tổng hợp (attack accuracy, hoặc cặp precision–recall) tính trung bình trên mọi mẫu; trong khi hiệu năng mô hình khác nhau giữa các mẫu nên rủi ro cũng không đồng đều. Cần phân tích theo từng mẫu để biết phân bố rủi ro và nhận diện mẫu rủi ro cao.

### 4.1 Định nghĩa privacy risk score

**Định nghĩa 1.** Privacy risk score của mẫu $z=(x,y)$ đối với mô hình $F$ là xác suất hậu nghiệm mẫu thuộc tập train sau khi quan sát hành vi $O(F,z)$ của mô hình trên mẫu đó:

$$r(z)=P\big(z\in D_{tr}\mid O(F,z)\big)$$

Theo định lý Bayes:

$$r(z)=\frac{P(z\in D_{tr})\,P(O(F,z)\mid z\in D_{tr})}{P(z\in D_{tr})\,P(O(F,z)\mid z\in D_{tr})+P(z\in D_{te})\,P(O(F,z)\mid z\in D_{te})}$$

- $D_{te}$: tập test (non-member); $O(F,z)$: quan sát của kẻ tấn công – black-box là $F(x)$; white-box còn gồm đầu ra tầng trung gian và gradient. Bài tập trung black-box (white-box để dành cho tương lai).

Với giả thiết tiên nghiệm 0.5/0.5 (bất định cực đại; các prior khác ở Phụ lục B):

$$r(z)=\frac{P(F(x)\mid z\in D_{tr})}{P(F(x)\mid z\in D_{tr})+P(F(x)\mid z\in D_{te})}$$

**Cách ước lượng các phân bố điều kiện** bằng shadow training:
1. Huấn luyện shadow model mô phỏng mô hình mục tiêu.
2. Lấy đầu ra của shadow model trên dữ liệu shadow-train và shadow-test.
3. Tính thực nghiệm các phân bố điều kiện trên hai tập đó.

Theo hướng ngưỡng theo lớp (đã chứng tỏ cải thiện tấn công), phân bố được tính riêng cho từng lớp: $P(F(x)\mid z\in D_{tr})=P(F(x)\mid z\in D_{tr},y=y_n)$ khi $y=y_n$ (tương tự với test).

**Xấp xỉ vì thiếu dữ liệu.** Chất lượng ước lượng phụ thuộc độ giống nhau giữa shadow và mục tiêu và kích thước dữ liệu shadow; phân bố đa chiều theo từng lớp cần nhiều mẫu (trung bình chỉ 197 mẫu/lớp ở Purchase100, 100 ở Texas100, 33 ở Location30, 500 ở CIFAR100). Vì các metric một chiều (confidence, modified entropy) đã cho tấn công tốt bằng hoặc hơn NN dùng cả vector, bài xấp xỉ phân bố đa chiều bằng phân bố của modified entropy theo lớp:

$$P(F(x)\mid z\in D_{tr})\approx P\big(\mathrm{Mentr}(F(x),y)\mid z\in D_{tr},y=y_n\big)\ \text{khi }y=y_n$$

(tương tự cho test), rồi thay vào công thức $r(z)$ ở trên.

### 4.2 Kết quả thực nghiệm

Bài kiểm chứng: (i) chỉ số phản ánh đúng xác suất thành viên; (ii) so sánh phân bố điểm trước/sau phòng thủ; (iii) dùng điểm để tấn công độ tin cậy cao; (iv) phân tích tương quan với tính chất mô hình. Dùng 3 classifier Purchase100 (không phòng thủ, AdvReg, early stopping) và 2 classifier Texas100 (không phòng thủ, MemGuard); mạng FC 4 lớp ẩn (1024/512/256/128), Tanh cho Purchase100 và ReLU cho Texas100.

#### 4.2.1 Kiểm chứng privacy risk score

**Quy trình:** tính điểm cho mọi mẫu train và test; chia khoảng điểm thành các bin; trong mỗi bin tính tỷ lệ mẫu train $n_{tr}/(n_{tr}+n_{te})$ (xác suất thành viên *thực*). Nếu chỉ số đúng thì điểm trung bình và tỷ lệ thực phải bám sát nhau (đường lý tưởng $y=x$).

**Baseline:** đầu ra (một giá trị trong [0,1]) của NN attack classifier theo kiểu Nasr (đầu vào là dự đoán + one-hot nhãn), vì các bài trước đề xuất dùng nó làm proxy xác suất thành viên.

*[Hình 3: (a) phân bố privacy risk score của mẫu train trên 3 mô hình Purchase100 và đường so sánh với xác suất thực – RMSE 0.05, 0.09, 0.06; (b) cùng phân tích với đầu ra NN attack classifier – RMSE 0.26, 0.26, 0.25, lệch xa đường lý tưởng.]*

- Privacy risk score bám sát xác suất thành viên thực; NN attack classifier thì không (do được huấn luyện để đẩy member về 1 và non-member về 0, không hiệu chuẩn xác suất).
- Kết quả tương tự trên Texas100 (Phụ lục C): RMSE của risk score là 0.08 và 0.05 so với 0.13 và 0.21 của đầu ra NN (không phòng thủ / MemGuard).
- Kiểm chứng với nhiều kiến trúc: với Purchase100/Texas100 thay đổi độ sâu (3 hoặc 5 lớp ẩn), độ rộng (×0.5, ×2) và hàm kích hoạt (ReLU/Tanh/Sigmoid; Sigmoid trên Texas100 chỉ đạt 4% accuracy nên bị loại); với CIFAR100 dùng DenseNet, AlexNet, VGG, ResNet, Wide ResNet. RMSE đều nhỏ hơn 0.09.

*[Hình 4: đường điểm rủi ro so với xác suất thành viên thực cho nhiều kiến trúc trên Purchase100, Texas100, CIFAR100; mọi đường sát đường lý tưởng.]*

#### 4.2.2 Tính không đồng nhất (heterogeneity) của điểm rủi ro

Vẽ CDF điểm rủi ro của mẫu train, so sánh mô hình có và không phòng thủ:
- *Purchase100 (Hình 5):* AdvReg giảm điểm trung bình, nhưng lại có một nhóm nhỏ mẫu có điểm cao hơn mô hình không phòng thủ (mô hình không phòng thủ có mọi điểm <0.8, mô hình AdvReg có vài mẫu >0.8). Early stopping cho phân bố tương tự AdvReg.
- *Texas100 (Hình 6):* MemGuard giảm điểm rủi ro nhưng vẫn rất dễ bị tổn thương: 70% mẫu train có điểm >0.6.

*[Hình 5: CDF điểm rủi ro, các classifier Purchase100 của Nasr. Hình 6: CDF điểm rủi ro, các classifier Texas100 của Jia.]*

#### 4.2.3 Cách dùng privacy risk score

Kẻ tấn công có thể không chạy theo accuracy trung bình mà chọn các mẫu rủi ro cao: suy ra member khi và chỉ khi điểm vượt ngưỡng xác suất. Ngưỡng cao hơn → precision cao hơn, recall thấp hơn.

**Bảng 6 (rút gọn): (precision, recall) theo ngưỡng của điểm rủi ro**

| Dataset | Phòng thủ | 1 | 0.9 | 0.8 | 0.7 | 0.6 | 0.5 |
|---|---|---|---|---|---|---|---|
| Texas100 | Không | (85.4, 21.2) | (83.4, 29.1) | (81.2, 45.3) | (77.0, 66.1) | (72.8, 85.4) | (70.6, 94.3) |
| Texas100 | MemGuard | (88.2, 1.4) | (84.5, 7.6) | (82.6, 18.7) | (77.0, 43.7) | (71.3, 70.5) | (66.0, 99.9) |
| Purchase100 | Không | (N.A., 0) | (N.A., 0) | (N.A., 0) | (63.4, 7.8) | (62.6, 55.1) | (61.6, 90.9) |
| Purchase100 | Early stopping | (N.A., 0) | (N.A., 0) | (N.A., 0) | (60.4, 1.3) | (57.1, 20.2) | (56.6, 75.2) |
| Purchase100 | AdvReg | (83.3, 0.2) | (83.3, 0.2) | (65.9, 0.5) | (63.9, 1.7) | (58.9, 24.6) | (56.5, 76.3) |

(đơn vị %)

- MemGuard: cùng ngưỡng thì precision tương tự mô hình không phòng thủ nhưng recall thấp hơn; vẫn suy ra đúng 70.5% member với precision 71.3%, và 1.4% member với precision 88.2%.
- AdvReg giảm rủi ro trung bình nhưng *tăng* rủi ro cho một số mẫu (0.2% member suy ra đúng với precision 83.3%, trong khi mô hình không phòng thủ không có mẫu nào đạt mức này). Do đó người thiết kế phòng thủ nên xét toàn bộ phân bố rủi ro.

#### 4.2.4 Ảnh hưởng của tính chất mô hình lên risk score

Dùng classifier Texas100 không phòng thủ của Jia et al.

**Với model sensitivity.** Sensitivity của mẫu train $z$ = logarit của tỷ số modified entropy của mô hình huấn luyện lại sau khi bỏ $z$ so với mô hình gốc:

$$\log\frac{\mathrm{Mentr}(\tilde F_z(x),y)}{\mathrm{Mentr}(F(x),y)}$$

- $\tilde F_z$: mô hình huấn luyện lại sau khi loại $z$ khỏi tập train.

*[Hình 7: quan hệ giữa điểm rủi ro và sensitivity (tứ phân vị 1, trung bình, tứ phân vị 3); mẫu điểm cao có ảnh hưởng lớn hơn tới mô hình.]*

**Với generalization error.** Mẫu rủi ro cao tập trung ở một số lớp. Điểm rủi ro trung bình theo lớp tương quan rất mạnh với generalization error của lớp (Pearson 0.94) – hợp lý vì generalization error ảnh hưởng mạnh tới thành công của MIA.

*[Hình 8: điểm rủi ro trung bình và generalization error theo từng lớp trong 100 lớp, sắp theo generalization error; tương quan Pearson 0.94.]*

**Với feature embedding.** Dùng đầu ra tầng ẩn cuối làm embedding, chọn 5 lớp có điểm trung bình thấp nhất (30, 93, 97, 18, 98; 0.50–0.55) và 5 lớp cao nhất (72, 49, 45, 51, 78; 0.82–0.90), mỗi lớp ≥100 mẫu train; hiển thị bằng t-SNE.
- Mẫu train tách rõ theo lớp (train accuracy ~100%).
- Mẫu test của lớp rủi ro thấp có embedding giống mẫu train và vẫn tách tốt; mẫu test của lớp rủi ro cao có biểu diễn khác mẫu train tương ứng.
- Thấy tính không đồng nhất cả ở mức mẫu (trong lớp 78) lẫn mức lớp (lớp 30 so với 78), củng cố sự cần thiết của phân tích mịn và của ngưỡng theo lớp. Phòng thủ tốt nên làm embedding của train và validation giống nhau và xét tính không đồng nhất của rủi ro.

*[Hình 9: t-SNE embedding của mẫu train (a) và test (b) cho 10 lớp; 5 lớp đầu rủi ro thấp, 5 lớp sau rủi ro cao.]*

---

## 5. Kết luận (Conclusions)

- Đánh giá rủi ro MIA chỉ bằng NN attack là không đủ; cần bổ sung bộ metric-based attacks (ngưỡng theo lớp + modified entropy).
- Khi đánh giá phòng thủ có đánh đổi accuracy–privacy, phải so với early stopping và xét adaptive attacker. Kết quả: (1) AdvReg chỉ giảm rủi ro hạn chế và không tốt hơn early stopping; (2) hiệu quả MemGuard giảm mạnh khi gặp adaptive attack.
- Privacy risk score ước lượng đúng xác suất thành viên của từng mẫu; rủi ro không đồng nhất giữa các mẫu; liên hệ chặt với sensitivity, generalization error và embedding.
- Kêu gọi cộng đồng đánh giá rủi ro quyền riêng tư một cách có hệ thống và nghiêm ngặt.

---

## Phụ lục (Appendix)

### A. MIA trên các dataset khác

Thêm hai bộ ảnh: **CH-MNIST** (mô bệnh học ung thư đại trực tràng, ảnh 64×64 xám, 8 lớp, 5,000 mẫu; dùng 2,000 mẫu để train CNN 2 khối conv với 32 và 64 kênh; train 99.0%, test 71.7%) và **Car196** (196 lớp xe; 8,144 train/8,041 test; fine-tune ResNet50 pretrain ImageNet; train 99.3%, test 87.5%). Tấn công NN theo Nasr et al. [32].

| Dataset | NN-based | $I_{corr}$ | $I_{conf}$ | $I_{entr}$ | $I_{Mentr}$ |
|---|---|---|---|---|---|
| CH-MNIST | 70.5% | 63.7% | 72.6% | 69.6% | 72.6% |
| Car196 | 63.1% | 55.9% | 63.7% | 62.9% | 63.7% |

Tấn công tốt nhất của benchmark hơn NN attack 2.1% và 0.6%.

### B. Privacy risk score với xác suất tiên nghiệm khác nhau

Trên Purchase100 không phòng thủ, thay đổi $P(z\in D_{tr})\in\{0.1,0.3,0.7,0.9\}$ (dùng công thức đầy đủ có prior). Phần lớn mẫu train có điểm cao hơn prior (đường đoán ngẫu nhiên). Độ rò rỉ định nghĩa bằng khoảng cách trung bình $\frac{1}{|D_{tr}|}\sum_{z\in D_{tr}}(r(z)-P(z\in D_{tr}))$ là 0.05, 0.09, 0.07, 0.02 tương ứng; so với 0.1 khi prior = 0.5. Prior càng gần 0.5 thì bất định của MIA càng lớn, khoảng cách càng lớn.

*[Hình 10: phân bố điểm rủi ro của mẫu train ở các prior khác nhau, kèm đường prior.]*

### C. Kiểm chứng risk score trên Texas100

RMSE của privacy risk score là 0.08 (không phòng thủ) và 0.05 (MemGuard), trong khi đầu ra NN attack là 0.13 và 0.21; risk score phản ánh xác suất thành viên thực tốt hơn.

*[Hình 11: ước lượng xác suất thành viên bằng risk score (trái) và đầu ra NN attack classifier (phải) trên Texas100.]*

### D. Kiểm chứng risk score trên nhiều kiến trúc (có phòng thủ)

Với classifier Purchase100 có AdvReg và Texas100 có MemGuard, trên các kiến trúc khác nhau (activation, width, depth), điểm rủi ro vẫn sát xác suất thành viên thực, RMSE < 0.10.

*[Hình 12: kiểm chứng trên các mô hình có phòng thủ với nhiều kiến trúc.]*

---

## Ghi chú

- **Cảnh báo về đánh giá yếu:** bài chỉ ra rằng kết luận "phòng thủ hiệu quả" dựa trên một kẻ tấn công NN yếu hoặc không thích nghi là không đáng tin; đây là động lực cho việc dùng tấn công mạnh, hiệu chuẩn tốt khi đo MIA trên mô hình chuỗi thời gian (time-series).
- **Metric-based attack đơn giản làm baseline:** các tấn công dựa ngưỡng trên loss/confidence/modified entropy với ngưỡng theo lớp là baseline rẻ, không cần siêu tham số; có thể chuyển sang mô hình forecasting bằng cách thay metric bằng sai số dự báo/loss theo mẫu (ở đây chỉ là suy luận của người tóm tắt, bài gốc chỉ xét classifier).
- **Ngưỡng theo nhóm/lớp, rủi ro không đồng nhất:** ý tưởng hiệu chuẩn ngưỡng theo nhóm (ở đây là lớp) và privacy risk score theo từng mẫu gợi ý hiệu chuẩn theo độ khó của mẫu/chuỗi (difficulty calibration) – liên quan trực tiếp tới các công trình sau như LiRA/RMIA.
- **Cách báo cáo kết quả:** nhấn mạnh phân tích theo mẫu và precision–recall ở ngưỡng cao thay vì chỉ accuracy trung bình, tương thích với yêu cầu báo cáo TPR@low FPR trong đánh giá MIA hiện đại.
- **Early stopping là baseline bắt buộc:** hữu ích khi so sánh phòng thủ trên mô hình chuỗi thời gian (regularization, DP...) để tránh kết luận sai về hiệu quả bảo vệ.
