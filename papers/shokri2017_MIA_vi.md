# Tấn công suy luận tư cách thành viên (membership inference) nhằm vào các mô hình học máy

*Membership Inference Attacks Against Machine Learning Models*

- **Tác giả:** Reza Shokri (Cornell Tech), Marco Stronati (INRIA), Congzheng Song (Cornell), Vitaly Shmatikov (Cornell Tech)
- **Nơi công bố / năm:** IEEE Symposium on Security and Privacy (S&P) 2017; bản arXiv:1610.05820v2

> **Bản tóm tắt chi tiết bằng tiếng Việt (do AI viết, không phải bản dịch nguyên văn).**

---

## Tóm tắt (Abstract)

Bài báo định lượng mức độ mà các mô hình học máy làm rò rỉ thông tin về từng bản ghi dữ liệu đã dùng để huấn luyện chúng. Tác giả tập trung vào dạng tấn công cơ bản nhất, **membership inference attack (MIA – tấn công suy luận tư cách thành viên)**: cho một bản ghi và quyền truy cập black-box (hộp đen, chỉ gửi truy vấn và nhận đầu ra) vào một mô hình, hãy xác định bản ghi đó có nằm trong tập huấn luyện của mô hình hay không.

Ý tưởng chính: dùng chính học máy để tấn công học máy. Kẻ tấn công huấn luyện một **attack model** (mô hình tấn công) để nhận ra sự khác biệt giữa cách mô hình mục tiêu dự đoán trên dữ liệu đã huấn luyện và trên dữ liệu chưa từng thấy. Các tác giả đánh giá trên các dịch vụ "machine learning as a service" thương mại (Google Prediction API, Amazon ML) với các bộ dữ liệu thực tế, trong đó có dữ liệu xuất viện bệnh viện (việc là thành viên của tập này là thông tin nhạy cảm). Sau đó họ phân tích các yếu tố ảnh hưởng đến mức rò rỉ và đánh giá các biện pháp giảm thiểu.

---

## I. Giới thiệu (Introduction)

- Học máy là nền tảng của nhiều dịch vụ Internet (nhận dạng ảnh/giọng nói, dịch máy, quảng cáo, gợi ý...). Dữ liệu huấn luyện thường là hoạt động cá nhân: mua sắm, sức khỏe, vị trí, ảnh, giọng nói.
- Các hãng như Google và Amazon cung cấp **machine learning as a service (MLaaS – học máy như một dịch vụ)**: khách hàng tải dữ liệu lên, trả tiền để dịch vụ dựng mô hình, rồi truy cập mô hình dưới dạng API hộp đen (đôi khi còn chia sẻ hoặc bán lại cho bên thứ ba).
- **Câu hỏi nghiên cứu:** cho một mô hình và một bản ghi, bản ghi đó có thuộc tập huấn luyện không? Đặt trong bối cảnh khó nhất: kẻ tấn công chỉ có truy vấn black-box, và nhận về vector xác suất dự đoán.
- **Cách tiếp cận:** biến bài toán suy luận thành bài toán phân loại nhị phân. Do kẻ tấn công không biết cấu trúc/tham số mô hình đích, tác giả đề xuất **shadow training (huấn luyện mô hình bóng)**:
  1. Tạo nhiều **shadow model** bắt chước hành vi mô hình đích nhưng biết rõ tập huấn luyện của chúng, tức biết chân trị (ground truth) về tư cách thành viên.
  2. Huấn luyện attack model trên cặp (đầu vào, đầu ra) có nhãn "in/out" của các shadow model.
- **Ba cách sinh dữ liệu cho shadow model:** (a) tổng hợp dữ liệu bằng chính mô hình đích (không cần biết gì về phân phối dữ liệu); (b) dùng thống kê của quần thể (phân phối biên của từng đặc trưng); (c) dùng dữ liệu thật nhưng có nhiễu. Hai cách (b), (c) chỉ cần truy vấn mô hình đích một lần cho mỗi bản ghi cần kiểm tra.
- Phương pháp mang tính tổng quát, không phụ thuộc loại mô hình hay bộ dữ liệu; đã thử với mạng nơ-ron, Amazon ML, Google Prediction API (không biết thuật toán hay kiến trúc bên trong), trên dữ liệu ảnh, giao dịch mua sắm, vết vị trí, hồ sơ nhập viện.
- **Kết quả nổi bật:**
  - Với mô hình nhiều lớp huấn luyện trên 10.000 bản ghi mua sắm: độ chính xác trung vị của MIA là 94% (Google) và 74% (Amazon) ở cấu hình mặc định.
  - Dùng hoàn toàn dữ liệu tổng hợp cho shadow model, độ chính xác với mô hình Google vẫn đạt 90%.
  - Dữ liệu bệnh viện Texas: trên 70% độ chính xác.
- Bài cũng bàn về nguyên nhân gốc và so sánh các biện pháp giảm thiểu: giới hạn đầu ra top-k lớp, giảm độ chính xác của vector dự đoán, tăng entropy, và regularization (chính quy hóa).
- **Phân biệt với model inversion:** model inversion dùng đầu ra của mô hình để suy ra thuộc tính của một đầu vào ẩn hoặc dựng đặc trưng đại diện cho một lớp; nó không tạo ra một thành viên thực sự của tập huấn luyện và không trả lời được "bản ghi này có là thành viên không". Bài toán MIA về bản chất giống bài toán xác định sự hiện diện của một cá nhân trong một nhóm trộn dựa trên thống kê (kiểu nghiên cứu bộ gen), nhưng ở đây thông tin có được qua API hộp đen của mô hình thay vì thống kê tường minh.

---

## II. Kiến thức nền về học máy (Machine Learning Background)

- **Học không giám sát** trích xuất đặc trưng/cấu trúc ẩn từ dữ liệu không nhãn; **học có giám sát** (trọng tâm của bài) học quan hệ giữa dữ liệu và nhãn để tổng quát hóa ra ngoài tập huấn luyện.
- Thuật toán huấn luyện cực tiểu hóa sai số trên tập huấn luyện, nên có thể **overfitting (quá khớp)**: mô hình tốt hơn trên đầu vào đã học so với đầu vào cùng quần thể nhưng chưa thấy. Có nhiều kỹ thuật **regularization** để hạn chế điều này.
- **MLaaS:** Google Prediction API, Amazon ML, Microsoft Azure ML, BigML. Các dịch vụ này cung cấp API đơn giản để tải dữ liệu, huấn luyện, truy vấn. Chi tiết mô hình và thuật toán bị ẩn: Google không cho chỉnh gì; Amazon chỉ cho vài tùy chọn (L1/L2 regularization, số lượt duyệt dữ liệu). Mô hình không tải xuống được và doanh thu chủ yếu đến từ phí truy vấn, nên bài coi MLaaS như hộp đen hoàn toàn và mọi tấn công đều chỉ qua API chuẩn.

---

## III. Quyền riêng tư trong học máy (Privacy in Machine Learning)

### A. Suy luận về các thành viên của quần thể

- Định nghĩa "mô hình không tiết lộ thêm gì về đầu vào so với khi không có mô hình" (**Dalenius desideratum**) không thể đạt được bởi bất kỳ mô hình hữu ích nào.
- Khái niệm trong model inversion (suy ra thuộc tính nhạy cảm từ đầu ra) cũng khó ngăn chặn nếu mô hình dựa trên các sự thật thống kê của cả quần thể. Ví dụ: tương quan giữa kiểu hình bên ngoài và nguy cơ di truyền là một tri thức khoa học công khai, áp dụng cho mọi người trong quần thể, kể cả người không có dữ liệu trong tập huấn luyện ("tác động từ xa"). Mô hình tổng quát hóa tốt thì người tạo ra nó không thể làm gì để chặn kiểu "rò rỉ" này.

### B. Suy luận về các thành viên của tập huấn luyện

- Để tránh khó khăn trên, bài tập trung bảo vệ những cá nhân có dữ liệu được dùng để huấn luyện, gần với động cơ ban đầu của differential privacy. Mục tiêu: đo rủi ro về tư cách thành viên mà một người chịu khi cho phép dữ liệu của mình huấn luyện mô hình, so với một thành viên bất kỳ của quần thể.
- Tấn công cơ bản là membership inference. Khi bản ghi đã hoàn toàn được kẻ tấn công biết, việc biết nó nằm trong tập huấn luyện của một mô hình gắn với một bệnh (ví dụ liều thuốc, cơ sở di truyền) có thể tiết lộ người đó mắc bệnh.
- Kịch bản nghiên cứu: black-box, kẻ tấn công chỉ đưa đầu vào và nhận đầu ra. Có thể mô hình chỉ được truy cập gián tiếp qua một ứng dụng; để đơn giản, bài giả định kẻ tấn công truy vấn thẳng mô hình.

---

## IV. Phát biểu bài toán (Problem Statement)

**Thiết lập**
- Có tập bản ghi có nhãn lấy từ một quần thể, chia thành các lớp; một thuật toán học máy huấn luyện mô hình phân loại. Với mỗi đầu vào, mô hình trả về **prediction vector** (vector xác suất, mỗi lớp một giá trị, còn gọi là confidence values); nhãn dự đoán là lớp có xác suất cao nhất.
- Kẻ tấn công có quyền truy vấn mô hình và nhận vector dự đoán cho bản ghi bất kỳ; biết định dạng và miền giá trị của đầu vào/đầu ra. Về mô hình, hoặc (1) biết loại, kiến trúc, thuật toán huấn luyện, hoặc (2) chỉ có truy cập hộp đen tới một "oracle học máy" (như MLaaS) đã huấn luyện mô hình, không biết cấu trúc hay siêu tham số.
- Kẻ tấn công có thể có tri thức nền về quần thể: các mẫu độc lập tách rời khỏi tập huấn luyện đích, hoặc chỉ các thống kê chung (ví dụ phân phối biên của các đặc trưng).

**Mục tiêu và thước đo**
- Cho một bản ghi và truy cập hộp đen, kẻ tấn công phải xác định bản ghi có trong tập huấn luyện hay không.
- Thước đo: **precision** (trong số bản ghi bị suy là thành viên, bao nhiêu thực sự là thành viên) và **recall** (trong số thành viên thực sự, bao nhiêu bị phát hiện).

---

## V. Suy luận tư cách thành viên (Membership Inference)

### A. Tổng quan về cuộc tấn công

*[Hình 1: Quy trình tấn công hộp đen. Kẻ tấn công truy vấn mô hình đích bằng bản ghi (x, y), nhận vector dự đoán, rồi đưa vector này cùng nhãn thật vào attack model để dự đoán "in/out".]*

*[Hình 2: Huấn luyện k shadow model bằng cùng nền tảng ML như mô hình đích; tập huấn luyện của các shadow có cùng định dạng, tách rời với tập đích, nhưng có thể chồng lấn nhau.]*

- Cơ sở: mô hình thường cư xử khác nhau trên dữ liệu đã huấn luyện và dữ liệu mới. Overfitting là lý do phổ biến nhưng không phải duy nhất (xem mục VII).
- **Attack model là một tập mô hình, mỗi lớp đầu ra của mô hình đích có một attack model riêng**, vì phân phối đầu ra phụ thuộc mạnh vào lớp thật của đầu vào.
- Ký hiệu: $f_{target}()$ là mô hình đích, $D^{train}_{target}$ là tập huấn luyện riêng tư; $c_{target}$ là số lớp. Đầu ra $\mathbf{y}=f_{target}(x)$ là vector kích thước $c_{target}$, các phần tử thuộc $[0,1]$ và tổng bằng 1. Attack model $f_{attack}()$ nhận (nhãn thật $y$, vector dự đoán $\mathbf{y}$) và là bộ phân loại nhị phân "in"/"out".
- Công thức mà attack model ước lượng:

$$\Pr\{(x,y)\in D^{train}_{target}\}$$

  đây là xác suất bản ghi $(x,y)$ thuộc lớp "in", tức nằm trong tập huấn luyện của mô hình đích.
- Khó khăn chính: không biết tham số bên trong của mô hình đích. Giải pháp: shadow training, huấn luyện attack model trên các mô hình thay thế có biết tập huấn luyện để áp dụng học có giám sát.

### B. Shadow models

- Kẻ tấn công tạo $k$ shadow model $f^i_{shadow}()$, mỗi cái được huấn luyện trên tập $D^{train}_{shadow^i}$ có cùng định dạng và phân phối tương tự tập đích (sinh theo các cách ở mục V-C).
- Giả định các tập shadow tách rời tập huấn luyện đích: $\forall i,\ D^{train}_{shadow^i}\cap D^{train}_{target}=\emptyset$. Đây là trường hợp tệ nhất cho kẻ tấn công; nếu có chồng lấn, tấn công còn tốt hơn.
- Shadow model phải được huấn luyện giống mô hình đích. Nếu biết thuật toán và cấu trúc (mạng nơ-ron, SVM, hồi quy logistic...) thì dễ. Với MLaaS, kẻ tấn công dùng đúng dịch vụ đó (ví dụ Google Prediction API) để huấn luyện shadow model (Hình 2).
- Càng nhiều shadow model, attack model càng chính xác (nhưng chi phí tăng).

### C. Sinh dữ liệu huấn luyện cho shadow model

Bốn cách, từ ít đến nhiều giả định:

1. **Model-based synthesis (tổng hợp dựa trên mô hình).** Khi không có dữ liệu thật hay thống kê, dùng chính mô hình đích để tạo dữ liệu: bản ghi mà mô hình đích phân loại với độ tin cậy cao nên giống dữ liệu huấn luyện. Gồm hai pha: (1) tìm kiếm leo đồi (hill climbing) trong không gian bản ghi để tìm đầu vào được phân loại tự tin vào lớp $c$; (2) lấy mẫu từ các bản ghi này. Thuật toán 1 (Synthesize) cho một lớp $c$:
   1. Khởi tạo ngẫu nhiên bản ghi $x$ (mỗi đặc trưng lấy ngẫu nhiên đều trong miền giá trị); đặt $y^*_c=0$, bộ đếm từ chối $j=0$, độ lớn bước $k=k_{max}$.
   2. Mỗi vòng lặp (tối đa $iter_{max}$), truy vấn mô hình đích để được $\mathbf{y}$.
   3. Nếu xác suất lớp $c$, $y_c$, không nhỏ hơn $y^*_c$ (cải thiện mục tiêu leo đồi) thì chấp nhận bản ghi: nếu đồng thời $y_c>conf_{min}$ và $c$ là lớp có xác suất cao nhất thì với xác suất $y_c$ trả về $x$ làm mẫu tổng hợp; nếu không, cập nhật $x^*\leftarrow x$, $y^*_c\leftarrow y_c$, $j\leftarrow 0$.
   4. Nếu bị từ chối: tăng $j$; khi $j>rej_{max}$ (quá nhiều lần từ chối liên tiếp) thì giảm bước $k\leftarrow\max(k_{min},\lceil k/2\rceil)$ và đặt lại $j=0$.
   5. Đề xuất bản ghi mới bằng cách ngẫu nhiên hóa $k$ đặc trưng của bản ghi chấp nhận gần nhất $x^*$ (lật bit với đặc trưng nhị phân, lấy mẫu lại giá trị với loại khác); lặp lại.
   6. Nếu hết vòng lặp mà không được chọn, trả về thất bại.

   Cách này chỉ hiệu quả khi kẻ tấn công khám phá hiệu quả không gian đầu vào; có thể không dùng được với ảnh độ phân giải cao và tác vụ phức tạp.
2. **Statistics-based synthesis (tổng hợp dựa trên thống kê).** Lấy mẫu độc lập giá trị từng đặc trưng từ phân phối biên của nó. Các attack model thu được rất hiệu quả.
3. **Noisy real data (dữ liệu thật có nhiễu).** Dữ liệu tương tự nhưng bị "bẩn"; trong thí nghiệm với dữ liệu vị trí, lật giá trị nhị phân của 10% hoặc 20% đặc trưng ngẫu nhiên. Mô phỏng trường hợp dữ liệu đích và dữ liệu shadow không lấy từ cùng quần thể hoàn toàn.
4. **Dữ liệu thật** (kịch bản lạc quan nhất, dùng làm chuẩn so sánh).

### D. Huấn luyện attack model

*[Hình 3: Từ mỗi shadow model, truy vấn trên tập huấn luyện của nó (nhãn "in") và trên một tập test tách rời (nhãn "out"), gộp thành tập huấn luyện của attack model, sau đó huấn luyện $c_{target}$ attack model, mỗi lớp một cái.]*

- Ý tưởng cốt lõi: các mô hình tương tự, huấn luyện trên dữ liệu tương tự, cùng một dịch vụ, cư xử giống nhau. Học cách suy luận tư cách thành viên trên shadow model (có chân trị) sẽ chuyển sang suy luận trên mô hình đích.
- **Các bước:**
  1. Với mỗi $(x,y)\in D^{train}_{shadow^i}$, tính $\mathbf{y}=f^i_{shadow}(x)$ và thêm $(y,\mathbf{y},\text{in})$ vào tập huấn luyện của attack model $D^{train}_{attack}$.
  2. Với tập $D^{test}_{shadow^i}$ tách rời tập huấn luyện của shadow thứ $i$ (cùng kích thước), với mỗi $(x,y)$ tính $\mathbf{y}$ và thêm $(y,\mathbf{y},\text{out})$.
  3. Chia $D^{train}_{attack}$ thành $c_{target}$ phần theo nhãn lớp.
  4. Với mỗi nhãn $y$, huấn luyện một mô hình riêng dự đoán "in"/"out" từ vector $\mathbf{y}$.
- **Lưu ý với model-based synthesis:** mọi dữ liệu thô của attack model đều là bản ghi mà mô hình đích tự tin; điều này đúng cho cả bản ghi huấn luyện và bản ghi test của shadow. Do đó attack model không chỉ học "độ tin cậy cao", mà học việc tinh tế hơn: phân biệt bản ghi huấn luyện có độ tin cậy cao với bản ghi không huấn luyện cũng có độ tin cậy cao.
- Nhờ chuyển về phân loại nhị phân chuẩn, có thể dùng bất kỳ framework hay dịch vụ nào để huấn luyện attack model (trong bài: mạng nơ-ron, hoặc chính Google Prediction API dù không kiểm soát cấu trúc).

---

## VI. Đánh giá (Evaluation)

### A. Dữ liệu (Data)

| Bộ dữ liệu | Mô tả | Cấu hình dùng trong thí nghiệm |
|---|---|---|
| **CIFAR-10 / CIFAR-100** | Ảnh màu 32×32; 10 lớp (6.000 ảnh/lớp, 50.000 train + 10.000 test) hoặc 100 lớp (600 ảnh/lớp) | Dùng các phần khác nhau để xem ảnh hưởng của kích thước tập huấn luyện |
| **Purchases** | Từ cuộc thi Kaggle "acquire valued shoppers"; 197.324 bản ghi, 600 đặc trưng nhị phân (đã mua sản phẩm hay chưa); gom cụm thành các lớp "phong cách mua sắm" với 2, 10, 20, 50, 100 lớp | 10.000 bản ghi huấn luyện đích |
| **Locations** | Check-in Foursquare ở Bangkok (4/2012–9/2013); 5.010 hồ sơ người dùng, 446 đặc trưng nhị phân (khu vực 0,5×0,5 km: 318 vùng, và 128 loại địa điểm); gom thành 30 lớp "geosocial type" | 1.600 bản ghi huấn luyện đích trong mô tả dữ liệu, nhưng phần cấu hình thí nghiệm dùng 1.200 |
| **Texas hospital stays** | Dữ liệu xuất viện công khai Texas 2006–2009; 67.330 bản ghi, 6.170 đặc trưng nhị phân (nguyên nhân chấn thương, chẩn đoán, thủ thuật, giới tính, tuổi, chủng tộc...); dự đoán thủ thuật chính trong 100 thủ thuật phổ biến nhất | 10.000 bản ghi huấn luyện đích |
| **MNIST** | 70.000 chữ số viết tay 32×32 | 10.000 bản ghi huấn luyện đích |
| **UCI Adult** | 48.842 bản ghi, 14 thuộc tính; phân loại nhị phân thu nhập >50K | 10.000 bản ghi huấn luyện đích |

Thí nghiệm không tái định danh cá nhân và tuân thủ thỏa thuận sử dụng dữ liệu.

### B. Mô hình đích (Target models)

- **MLaaS:** Google Prediction API (không có tham số tùy chỉnh) và Amazon ML (không chọn được loại mô hình, nhưng chỉnh được số lượt duyệt dữ liệu tối đa và mức L2). Dùng hai cấu hình: mặc định (10, 1e-6) và (100, 1e-4).
- **Mạng nơ-ron cục bộ** (Torch7): 
  - CIFAR: CNN chuẩn với 2 lớp tích chập + max pooling, 1 lớp fully connected 128 đơn vị, lớp SoftMax; kích hoạt Tanh; learning rate 0,001, decay 1e-7, tối đa 100 epoch.
  - Purchase: 1 lớp ẩn 128 đơn vị, SoftMax, Tanh; tối đa 200 epoch.
- Với dịch vụ đám mây, tấn công coi mô hình hoàn toàn là hộp đen.

### C. Thiết lập thí nghiệm

- Tập huấn luyện và tập test của mỗi mô hình đích và shadow được chọn ngẫu nhiên, cùng kích thước, rời nhau. Tập đích và tập shadow không chồng lấn; các tập của các shadow khác nhau có thể chồng lấn.
- Kích thước tập huấn luyện: 10.000 cho Purchase, Texas, Adult, MNIST; 1.200 cho Location; CIFAR-10: 2.500, 5.000, 10.000, 15.000; CIFAR-100: 4.600, 10.520, 19.920, 29.540.
- CIFAR được chạy trên mô hình cục bộ để biến đổi cấu hình; các bộ còn lại chạy trên Google/Amazon. Bộ Purchase được huấn luyện trên mọi nền tảng để so sánh rò rỉ.
- Attack model: mạng fully connected 1 lớp ẩn 64 đơn vị ReLU + SoftMax, hoặc mô hình Google black-box.
- **Số shadow model:** 100 (CIFAR), 20 (Purchase), 10 (Texas), 60 (Location), 50 (MNIST), 20 (Adult).

### D. Độ chính xác của tấn công

- Đánh giá trên các bản ghi huấn luyện/test của mô hình đích được xáo trộn, với số thành viên bằng số không thành viên, nên **baseline = 0,5** (đoán ngẫu nhiên). Kết quả báo cáo theo từng lớp vì độ chính xác khác nhau nhiều giữa các lớp (kích thước và thành phần dữ liệu từng lớp).
- Mô hình CIFAR có tập lớn nhất có độ chính xác test thấp (0,6 cho CIFAR-10, 0,2 cho CIFAR-100), tức overfit mạnh.

*[Hình 4: Precision của MIA trên mạng CNN huấn luyện bằng CIFAR; trái: theo từng lớp CIFAR-10; giữa/phải: theo kích thước tập huấn luyện của CIFAR-10 và CIFAR-100. Recall gần 1.]*

| Dataset | Median precision khi tăng dần kích thước tập huấn luyện |
|---|---|
| CIFAR-10 | 0,78 → 0,74 → 0,72 → 0,71 |
| CIFAR-100 | 1 → 1 → 0,98 → 0,97 |

*[Hình 5: CDF thực nghiệm của precision/recall trên các lớp của mô hình Purchase (10.000 bản ghi) huấn luyện bằng Amazon ML (2 cấu hình) và Google.]*

| Nền tảng (Purchase, 100 lớp) | Percentile 50/75/90 của precision | Recall |
|---|---|---|
| Amazon (10, 1e-6) | 0,74 / 0,79 / 0,84 | ≈ 1 |
| Amazon (100, 1e-4) | 0,84 / 0,88 / 0,91 | ≈ 1 |
| Google | 0,94 / 0,97 / 1 | ≈ 1 |

**Bảng I – Độ chính xác của mô hình trên Purchase (100 lớp)**

| Nền tảng | Train | Test |
|---|---|---|
| Google | 0,999 | 0,656 |
| Amazon (10, 1e-6) | 0,941 | 0,468 |
| Amazon (100, 1e-4) | 1,00 | 0,504 |
| Neural network | 0,830 | 0,670 |

*[Hình 7: So sánh precision của tấn công trên các nền tảng khác nhau với cùng dữ liệu; mô hình Google rò rỉ nhiều nhất.]*

- **Texas hospital stays (Google):** train 0,66, test 0,51; precision phần lớn trên 0,6, một nửa số lớp trên 0,7, hơn 20 lớp trên 0,85 (Hình 6).
- **Location (Google):** train 1,0, test 0,66; precision 0,6–0,8, recall gần 1.

### E. Ảnh hưởng của dữ liệu huấn luyện shadow

*[Hình 8: CDF precision trên mô hình Google cho dữ liệu vị trí khi shadow dùng dữ liệu thật hoặc dữ liệu nhiễu 10%/20%.]*

| Nguồn dữ liệu shadow (Location, Google) | Precision tổng | Recall |
|---|---|---|
| Dữ liệu thật | 0,678 | 0,98 |
| Nhiễu 10% | 0,666 | 0,99 |
| Nhiễu 20% | 0,613 | 1,00 |

- Precision giảm khi nhiễu tăng nhưng vẫn vượt baseline, và với 10% nhiễu gần ngang tấn công gốc: tấn công khá bền vững trước sai lệch giả định về phân phối dữ liệu.

*[Hình 9: CDF precision với mô hình Google trên Purchase theo cách sinh dữ liệu shadow (thật, tổng hợp từ thống kê biên, tổng hợp từ mô hình).]*

| Dữ liệu shadow (Purchase, Google) | Precision tổng | Recall |
|---|---|---|
| Dữ liệu thật | 0,935 | 0,994 |
| Tổng hợp theo phân phối biên (187.300 bản ghi, 20 shadow) | 0,795 | 0,991 |
| Tổng hợp dựa trên mô hình (30.000 bản ghi, 8 shadow) | 0,896 | 0,526 |

- Chi tiết model-based synthesis: $k_{max}=128$, giảm một nửa sau $rej_{max}=10$ lần từ chối liên tiếp, $k_{min}=4$, $conf_{min}=0{,}2$. Độ tin cậy trung bình của các bản ghi được chọn là 0,24; mỗi bản ghi cần trung bình 156 truy vấn.
- Với dữ liệu tổng hợp từ mô hình, precision cao ở đa số lớp (gần mức dùng dữ liệu thật) nhưng rất thấp (dưới 0,1) ở một vài lớp. Nguyên nhân: các lớp đó ít đại diện trong tập huấn luyện đích (dưới 0,6% dữ liệu, có lớp chưa tới 30 bản ghi trên 10.000), nên mô hình đích không mô hình hóa tự tin được và thuật toán khó tổng hợp mẫu đại diện trong không gian nhiều chiều.
- Kết luận: có thể huấn luyện MIA chỉ với truy cập hộp đen và không biết trước phân phối dữ liệu, miễn là kẻ tấn công sinh được các đầu vào được mô hình phân loại với độ tin cậy cao.

### F. Ảnh hưởng của số lớp và lượng dữ liệu mỗi lớp

- Nhiều lớp hơn nghĩa là nhiều tín hiệu hơn về trạng thái bên trong của mô hình cho kẻ tấn công; mô hình nhiều lớp phải "nhớ" nhiều hơn về dữ liệu huấn luyện. Vì vậy CIFAR-100 rò rỉ nhiều hơn CIFAR-10; ngay cả khi số bản ghi trung bình mỗi lớp bằng nhau (200), tấn công CIFAR-100 gần như tuyệt đối còn CIFAR-10 thì kém hơn.
- Trên Purchase/Google với 2, 10, 20, 50, 100 lớp: càng ít lớp càng ít rò rỉ (Hình 10).
- Nhìn chung, lớp có càng nhiều dữ liệu trong tập huấn luyện thì precision của tấn công trên lớp đó càng thấp (Hình 11).

**Bảng II – Độ chính xác mô hình Google và precision của tấn công**

| Dataset | Train acc. | Test acc. | Attack precision |
|---|---|---|---|
| Adult | 0,848 | 0,842 | 0,503 |
| MNIST | 0,984 | 0,928 | 0,517 |
| Location | 1,000 | 0,673 | 0,678 |
| Purchase (2) | 0,999 | 0,984 | 0,505 |
| Purchase (10) | 0,999 | 0,866 | 0,550 |
| Purchase (20) | 1,000 | 0,781 | 0,590 |
| Purchase (50) | 1,000 | 0,693 | 0,860 |
| Purchase (100) | 0,999 | 0,659 | 0,935 |
| TX hospital stays | 0,668 | 0,517 | 0,657 |

- **Trường hợp tấn công thất bại:** MNIST (precision 0,517) do dữ liệu mỗi lớp thiếu tính ngẫu nhiên (các chữ số cùng lớp rất giống nhau) và ít lớp; Adult (0,503) vì mô hình không overfit (train ≈ test) và là bộ phân loại nhị phân, nên kẻ tấn công chỉ có thực chất một tín hiệu (hai đầu ra bù nhau).

### G. Ảnh hưởng của overfitting

- Mô hình càng overfit thì càng rò rỉ, nhưng chỉ khi so sánh các mô hình cùng loại. Ví dụ Amazon (100, 1e-4) overfit hơn và rò rỉ hơn Amazon (10, 1e-6), nhưng cả hai đều rò rỉ ít hơn Google dù mô hình Google ít overfit hơn một trong hai mô hình Amazon và có khả năng dự đoán tốt hơn. Vậy **overfitting không phải yếu tố duy nhất**; cấu trúc và loại mô hình cũng đóng góp.
- Hình 11: khoảng cách độ chính xác train–test (generalization gap) tính theo từng lớp; gap lớn đi kèm precision tấn công cao.

*[Hình 10: Phân phối precision theo số lớp của bài toán Purchase trên Google.]*

*[Hình 11: Quan hệ giữa precision của tấn công theo lớp với tỉ lệ dữ liệu của lớp trong tập huấn luyện và với gap train–test của mô hình đích; mỗi điểm là một lớp.]*

---

## VII. Vì sao tấn công hiệu quả (Why Our Attacks Work)

*[Hình 12: Phân phối độ bất định dự đoán (hàng trên) và độ chính xác dự đoán (hàng dưới) của mô hình đích giữa thành viên và không thành viên cho vài lớp mẫu; hai phân phối càng tách biệt thì tấn công càng tốt (xem Bảng II).]*

- Hai đại lượng được xét cho mỗi lớp: độ chính xác (xác suất mô hình phân loại đúng đầu vào lớp $i$ vào lớp $i$) và **prediction uncertainty** (độ bất định dự đoán), là entropy chuẩn hóa của vector dự đoán:

$$-\frac{1}{\log n}\sum_i p_i \log p_i$$

  với $p_i$ là xác suất đầu vào thuộc lớp $i$ và $n$ là số lớp. Ở những trường hợp tấn công thành công, hai đại lượng này khác biệt rõ giữa thành viên và không thành viên.
- Thành công của MIA phụ thuộc trực tiếp vào (1) khả năng tổng quát hóa của mô hình đích và (2) độ đa dạng của dữ liệu huấn luyện. Nếu mô hình overfit hoặc dữ liệu không đại diện, mô hình rò rỉ thông tin về đầu vào huấn luyện.
- Overfitting gây hại thêm một mặt: không chỉ giảm khả năng dự đoán mà còn rò rỉ dữ liệu nhạy cảm. Tuy nhiên, các cấu trúc mô hình khác nhau "nhớ" lượng thông tin khác nhau, nên rò rỉ có thể khác nhau ngay cả khi mức overfit như nhau (Bảng I).

---

## VIII. Biện pháp giảm thiểu (Mitigation)

- Vì overfitting là nguyên nhân quan trọng, ở đây mục tiêu của nghiên cứu học máy và nghiên cứu quyền riêng tư tương đồng (không có đánh đổi tiện ích–riêng tư thông thường). Các kỹ thuật như **dropout** có thể chống overfitting đồng thời tăng bảo đảm riêng tư; regularization cũng được dùng trong objective perturbation của học máy riêng tư vi phân.
- Mô hình được regularize tốt (lý tưởng) sẽ ít rò rỉ, và MIA có thể làm thước đo. Mô hình có cấu trúc đơn giản (ví dụ XOR của một số đặc trưng) tổng quát hóa toàn bộ không gian nên không rò rỉ.
- **Differential privacy (DP – riêng tư vi phân):** nếu huấn luyện thỏa DP, xác suất sinh ra một mô hình khi có hay không có một bản ghi là gần nhau, nên mô hình DP về mặt cấu trúc an toàn trước các MIA chỉ dựa vào đầu ra mô hình như trong bài. Trở ngại: với $\epsilon$ nhỏ, độ chính xác dự đoán có thể giảm đáng kể.
- **Trách nhiệm của nhà cung cấp MLaaS:** hiện các dịch vụ chỉ nhận dữ liệu và trả API mờ đục, không cảnh báo về overfitting hay rủi ro khi dữ liệu kém đại diện/ít mẫu cho một số lớp. Khi chọn mô hình thích nghi, nên cân nhắc cả rủi ro rò rỉ thông tin lẫn độ chính xác, cảnh báo khách hàng và cung cấp thêm khả năng quan sát mô hình. MIA có thể dùng làm thước đo rò rỉ của một mô hình và đo hiệu quả các kỹ thuật bảo vệ trong tương lai.

### A. Các chiến lược giảm thiểu

1. **Restrict to top-k classes (giới hạn top-k lớp):** chỉ trả về xác suất của $k$ lớp có khả năng cao nhất; $k$ càng nhỏ càng ít rò rỉ; cực đoan nhất chỉ trả về nhãn.
2. **Coarsen precision (làm thô độ chính xác):** làm tròn xác suất xuống $d$ chữ số thập phân; $d$ càng nhỏ càng ít rò rỉ.
3. **Increase entropy (tăng entropy):** dùng softmax có nhiệt độ $t>0$:

$$\mathrm{softmax}_t(z)_i=\frac{e^{z_i/t}}{\sum_j e^{z_j/t}}$$

   với $z$ là vector logit, $t$ là nhiệt độ (như trong knowledge distillation); $t$ lớn thì đầu ra gần đồng đều nên không rò rỉ thông tin.
4. **L2 regularization:** thêm phạt $\lambda\sum_i\theta_i^2$ vào hàm mất mát ($\theta_i$ là các tham số mô hình); $\lambda$ càng lớn càng chính quy hóa mạnh.

### B. Đánh giá các biện pháp

Cài đặt trên mô hình cục bộ (mạng 1 lớp ẩn: 256 đơn vị cho Purchase, 1.000 đơn vị cho Texas, kích hoạt Tanh); kẻ tấn công vẫn chỉ có hộp đen.

**Bảng III (rút gọn) – Độ chính xác mô hình đích và hiệu quả MIA**

| Purchase (100 lớp) | Test acc. | Attack total acc. | Precision | Recall |
|---|---|---|---|---|
| Không giảm thiểu | 0,66 | 0,92 | 0,87 | 1,00 |
| Top k = 3 | 0,66 | 0,92 | 0,87 | 0,99 |
| Top k = 1 | 0,66 | 0,89 | 0,83 | 1,00 |
| Chỉ trả nhãn (top-1 label) | 0,66 | 0,66 | 0,60 | 0,99 |
| Làm tròn d = 3 | 0,66 | 0,92 | 0,87 | 0,99 |
| Làm tròn d = 1 | 0,66 | 0,89 | 0,83 | 1,00 |
| Temperature t = 5 | 0,66 | 0,88 | 0,86 | 0,93 |
| Temperature t = 20 | 0,66 | 0,84 | 0,83 | 0,86 |
| L2 λ = 1e-4 | 0,68 | 0,87 | 0,81 | 0,96 |
| L2 λ = 1e-3 | 0,72 | 0,77 | 0,73 | 0,86 |
| L2 λ = 1e-2 | 0,63 | 0,53 | 0,54 | 0,52 |

| Texas (100 lớp) | Test acc. | Attack total acc. | Precision | Recall |
|---|---|---|---|---|
| Không giảm thiểu | 0,55 | 0,83 | 0,77 | 0,95 |
| Top k = 3 | 0,55 | 0,83 | 0,77 | 0,95 |
| Top k = 1 | 0,55 | 0,82 | 0,76 | 0,95 |
| Chỉ trả nhãn | 0,55 | 0,73 | 0,67 | 0,93 |
| Làm tròn d = 3 | 0,55 | 0,83 | 0,77 | 0,95 |
| Làm tròn d = 1 | 0,55 | 0,81 | 0,75 | 0,96 |
| Temperature t = 5 | 0,55 | 0,79 | 0,77 | 0,83 |
| Temperature t = 20 | 0,55 | 0,76 | 0,76 | 0,76 |
| L2 λ = 1e-4 | 0,56 | 0,80 | 0,74 | 0,92 |
| L2 λ = 5e-4 | 0,57 | 0,73 | 0,69 | 0,86 |
| L2 λ = 1e-3 | 0,56 | 0,66 | 0,64 | 0,73 |
| L2 λ = 5e-3 | 0,35 | 0,52 | 0,52 | 0,53 |

**Nhận xét:**
- Các biện pháp triển khai nhìn chung không làm giảm độ chính xác dự đoán; với regularization, độ chính xác test thậm chí tăng (trừ khi $\lambda$ quá lớn: λ = 1e-2 trên Purchase và λ = 5e-3 trên Texas làm test accuracy sụt, dù chặn được MIA).
- Tấn công **khá bền vững**: lọc top-1 hoặc top-3 không đủ chặn. Ngay cả chỉ trả nhãn vẫn bị khai thác, vì thành viên và không thành viên bị gán nhãn sai theo kiểu khác nhau (vào các lớp sai khác nhau). Nếu có thêm xác suất, rò rỉ còn nhiều hơn.
- Một số biện pháp không phù hợp cho API MLaaS đa dụng; **regularization** thì có vẻ cần thiết và hữu ích (cải thiện tổng quát hóa và giảm rò rỉ), nhưng phải áp dụng cẩn thận để không hại hiệu năng trên tập test.

---

## IX. Các công trình liên quan (Related Work)

- **Tấn công lên mô hình thống kê/học máy:** Ateniese et al. dùng tham số của SVM/HMM để suy ra thống kê chung của tập huấn luyện (ví dụ có dữ liệu của một chủng tộc hay không); bài này là black-box và suy ra thông tin về bản ghi cụ thể. Homer et al. và các nghiên cứu sau suy ra sự hiện diện của một bộ gen trong tập dữ liệu dựa trên thống kê công bố (tần suất alen hiếm) so với quần thể; bài này nhắm vào mô hình học máy đã huấn luyện. Calandrino et al. suy ra đầu vào từ thay đổi đầu ra của hệ gợi ý cộng tác, khai thác hành vi theo thời gian đặc thù của hệ này.
- **Model inversion:** suy ra đặc trưng của đầu vào ẩn từ đầu ra mô hình. Nó không xác định được bản ghi có nằm trong tập huấn luyện hay không, vì cách hoạt động hoàn toàn giống nhau cho thành viên và không thành viên (ví dụ ca pharmacogenetics cho kết quả gần như giống hệt; chỉ chính xác hơn khoảng 4% cho thành viên và chỉ đo được sau khi đã biết chân trị). Với nhận dạng khuôn mặt, kết quả inversion là ảnh "trung bình" của một lớp (một người); chỉ trong trường hợp đặc thù này mới nhận ra được. Khi lớp đa dạng (như CIFAR-10), ảnh dựng lại vô nghĩa (Hình 13).
- **Model extraction** (Tramèr et al.): trích xuất tham số mô hình; có thể làm rò rỉ dữ liệu trong các mô hình lưu trực tiếp dữ liệu huấn luyện (như kernel logistic regression với các "import points"), nhưng đây là kết quả đặc thù và không được định lượng.
- **Học máy bảo vệ quyền riêng tư:** SMC, tính toán trên dữ liệu mã hóa chỉ hạn chế rò rỉ trong lúc huấn luyện; mô hình thu được vẫn dễ bị tấn công suy luận như thường. **Differential privacy** (nhiều biến thể cho hồi quy, SVM, ERM, học sâu...) giới hạn xác suất thành công của MIA chỉ dựa vào mô hình, bao gồm các tấn công trong bài.

*[Hình 13: Ảnh do model inversion dựng từ mô hình CIFAR-10, không nhận diện được và không tương ứng với ảnh huấn luyện cụ thể nào.]*

---

## X. Kết luận (Conclusions)

- Đây là cuộc tấn công membership inference đầu tiên nhằm vào mô hình học máy, đặc biệt các mô hình hộp đen huấn luyện trên đám mây (Google Prediction API, Amazon ML). Đây là cách tiếp cận định lượng, tổng quát để hiểu mô hình rò rỉ thông tin về tập huấn luyện thế nào; có thể dùng như một tiêu chí khi chọn loại mô hình hay dịch vụ học máy.
- Đóng góp kỹ thuật chính: **shadow training**, huấn luyện attack model phân biệt đầu ra của mô hình đích trên thành viên và không thành viên; shadow model có thể xây bằng dữ liệu tổng hợp hoặc nhiễu. Với dữ liệu tổng hợp sinh từ mô hình đích, không cần biết gì trước về phân phối dữ liệu huấn luyện.
- Tư cách thành viên trong các tập dữ liệu bệnh viện/y tế là nhạy cảm, nên kết quả có hàm ý thực tiễn đáng kể về quyền riêng tư.

---

## Ghi chú

- **Bài báo nền tảng của MIA** hiện đại: định nghĩa bài toán, mô hình đe dọa (black-box, có tri thức nền về quần thể) và cách đo (precision/recall, baseline 0,5 với tập cân bằng) mà nhiều nghiên cứu sau kế thừa, kể cả khi chuyển sang mô hình chuỗi thời gian.
- **Shadow training** là khung chuẩn để tạo chân trị "in/out" khi không có thông tin về mô hình đích; có thể áp dụng cho mô hình dự báo chuỗi thời gian bằng cách thay vector xác suất bằng sai số dự báo/loss trên các cửa sổ dữ liệu.
- Bài chỉ xét phân loại, và tín hiệu chính là vector xác suất + nhãn thật. Với dự báo chuỗi thời gian (đầu ra liên tục, không có "độ tin cậy lớp") cần tìm tín hiệu tương đương, như độ lệch dự báo hoặc loss; cần lưu ý điểm yếu khi mô hình ít lớp/ít tín hiệu (trường hợp Adult, Purchase 2 lớp cho thấy tấn công gần ngẫu nhiên).
- Kết luận "overfitting không phải nguyên nhân duy nhất" và các biện pháp (top-k, làm tròn, nhiệt độ, L2) là cơ sở để bàn về phòng thủ cho mô hình dự báo; lưu ý chỉ regularization có tác dụng rõ, còn việc ẩn bớt đầu ra thì gần như không đủ.
- **Hạn chế về đánh giá:** bài dùng precision/recall trên tập cân bằng và chưa dùng TPR@FPR thấp hay hiệu chỉnh độ khó của mẫu (các công trình sau như LiRA, RMIA, difficulty calibration sẽ cải tiến phần này), nên đây là mốc cơ sở chứ không phải phương pháp mạnh nhất.
