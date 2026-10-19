# Tấn công suy luận thành viên chi phí thấp, hiệu năng cao

*(Low-Cost High-Power Membership Inference Attacks)*

**Tác giả:** Sajjad Zarifzadeh, Philippe Liu, Reza Shokri — Khoa Khoa học Máy tính, Đại học Quốc gia Singapore (NUS)

**Hội nghị/năm:** Proceedings of the 41st International Conference on Machine Learning (ICML), Vienna, Áo. PMLR 235, 2024 (arXiv:2312.03262v3)

> Bản dịch tiếng Việt của bài báo gốc (dịch bởi AI)
>
> **Lưu ý về phạm vi:** Đây là bản dịch **rút gọn** (dịch sát ý nhưng không chép lại toàn văn từng câu). Phần tóm tắt, các định nghĩa, công thức cốt lõi, các kết quả chính và cấu trúc đề mục được giữ theo bài gốc; các đoạn lập luận dài, các bảng/biểu đồ phụ trợ ở phần phụ lục được cô đọng và có ghi chú tại chỗ. Danh mục tài liệu tham khảo giữ nguyên tiếng Anh.

---

## Tóm tắt (Abstract)

Tấn công suy luận thành viên (membership inference attack – MIA) nhằm phát hiện xem một điểm dữ liệu cụ thể có được dùng để huấn luyện một mô hình hay không. Nhóm tác giả thiết kế một phép kiểm định thống kê mới để thực hiện tấn công suy luận thành viên bền vững (robust membership inference attacks – **RMIA**) với chi phí tính toán thấp. Điều này đạt được nhờ mô hình hóa chi tiết hơn giả thuyết không (null hypothesis) trong các phép kiểm định tỉ số hợp lý (likelihood ratio), đồng thời khai thác hiệu quả cả các mô hình tham chiếu (reference models) lẫn các mẫu dữ liệu tham chiếu thuộc quần thể (population). RMIA có công suất kiểm định (test power) vượt trội so với các phương pháp trước đó trên toàn bộ đường cong TPR–FPR (kể cả ở FPR cực thấp, thậm chí bằng 0). Trong điều kiện bị giới hạn về tính toán — chỉ có một số ít mô hình tham chiếu được huấn luyện sẵn (ít nhất là 1) — và cả khi các yếu tố khác của tấn công (ví dụ phân phối dữ liệu) thay đổi, phương pháp này vẫn hoạt động rất tốt, trong khi các tấn công trước đó tiệm cận mức đoán ngẫu nhiên. RMIA đặt nền móng cho việc đánh giá rủi ro quyền riêng tư dữ liệu trong học máy một cách thực tiễn nhưng vẫn chính xác.

---

## 1. Giới thiệu (Introduction)

MIA được dùng để định lượng mức rò rỉ thông tin của các thuật toán học máy về dữ liệu huấn luyện (Shokri et al., 2017). MIA khởi nguồn từ lĩnh vực thống kê tổng hợp trên dữ liệu chiều cao (Homer et al., 2008), nơi nhiều phương pháp kiểm định giả thuyết được thiết kế để tối ưu đánh đổi giữa công suất kiểm định và sai số (Sankararaman et al., 2009; Visscher & Hill, 2009; Dwork et al., 2015; Murakonda et al., 2021). Với các thuật toán học sâu, các phép kiểm định này tiến hóa từ việc dùng chính ML để thực hiện MIA (Shokri et al., 2017) sang các phép xấp xỉ khác nhau của kiểm định thống kê gốc (Sablayrolles et al., 2019; Ye et al., 2022; Carlini et al., 2022; Watson et al., 2022a; Bertran et al., 2023). Các tấn công cũng khác nhau ở mô hình đe dọa và lượng tính toán cần để "may đo" tấn công cho từng điểm dữ liệu và mô hình cụ thể (ví dụ tấn công toàn cục (Shokri et al., 2017; Yeom et al., 2018) so với tấn công được tùy biến theo từng mẫu (Ye et al., 2022; Carlini et al., 2022; Sablayrolles et al., 2019; Watson et al., 2022a)); loại thứ hai đều đòi hỏi huấn luyện rất nhiều mô hình tham chiếu.

Dù hiệu quả tấn công đã được cải thiện, chi phí tính toán khiến chúng khó dùng cho kiểm toán quyền riêng tư thực tế. Ngoài ra, các nghiên cứu trước (Carlini et al., 2022; Ye et al., 2022) cho thấy các tấn công mạnh "thống trị lẫn nhau" tùy kịch bản kiểm thử: với ngân sách tính toán thực tế, (Carlini et al., 2022) gần như đoán ngẫu nhiên; còn khi ngân sách dồi dào, (Ye et al., 2022) lại có công suất thấp ở FPR thấp. Qua phân tích thực nghiệm mở rộng, nhóm tác giả quan sát thêm sự thiếu ổn định của các tấn công trước đó khi thay đổi: số lượng mô hình tham chiếu, số truy vấn suy luận cần thiết, mức tương đồng giữa mô hình tham chiếu và mô hình đích, sự lệch phân phối giữa dữ liệu đích và dữ liệu quần thể, và hiệu năng trên dữ liệu ngoài phân phối (out-of-distribution – OOD). Những hạn chế này cho thấy cần có các tấn công suy luận thành viên vừa bền vững vừa hiệu quả.

MIA là một bài toán kiểm định giả thuyết, và các tấn công được đánh giá dựa trên đường cong đánh đổi TPR–FPR. Nhóm tác giả thiết kế một phép kiểm định thống kê mới bằng cách liệt kê các "thế giới khả dĩ" chi tiết gắn với giả thuyết không, trong đó điểm dữ liệu đích có thể bị thay thế bởi một mẫu ngẫu nhiên bất kỳ từ quần thể. Tấn công được thực hiện bằng cách tổng hợp các phép kiểm định tỉ số hợp lý (LR) của các trường hợp này. Phép kiểm định giúp phân biệt rõ hơn giữa điểm thành viên và không thành viên, cho phép ước lượng thống kê kiểm định chính xác hơn. Cách tính thống kê LR đề xuất cũng cực kỳ hiệu quả (cần rất ít mô hình tham chiếu) và vẫn mạnh khi có bất định về phân phối dữ liệu. RMIA vượt các công trình trước trong mọi kịch bản kiểm thử, và luôn đạt TPR cao ở mọi FPR (kể cả thấp đến 0) với bất kỳ ngân sách tính toán nào. Một khía cạnh quan trọng khác là nhiều tấn công trước đó có thể được xem như các dạng đơn giản hóa của khung đề xuất, từ đó làm rõ nguyên nhân gây bất ổn và hiệu năng thấp của chúng.

*[Hình 1: RMIA so với các tấn công trước đó là Attack-P và Attack-R (Ye et al., 2022) và LiRA (Carlini et al., 2022), trên các mô hình CIFAR-100, với ràng buộc chỉ dùng 1 mô hình tham chiếu (ở chế độ offline). RMIA vượt các tấn công khác trên toàn bộ đường cong đánh đổi TPR–FPR (ví dụ AUC cao hơn ít nhất 25% và TPR tại FPR bằng 0 cao hơn một bậc độ lớn so với LiRA). Các giá trị AUC trong chú giải: RMIA 0,872; LiRA 0,69; Attack-R 0,814; Attack-P 0,76.]*

Nhóm tác giả cho thấy RMIA vượt các tấn công trước đó¹ trên các bộ dữ liệu chuẩn², đạt AUC cao hơn đáng kể (tức TPR cao hơn trên mọi giá trị FPR), và TPR cao hơn 2× đến 4× ở các FPR thấp khi chỉ dùng 1 hoặc 2 mô hình tham chiếu (xem Hình 1). Lợi thế của RMIA đặc biệt rõ khi đối thủ chỉ dùng các mô hình tham chiếu huấn luyện sẵn (tức huấn luyện độc lập với dữ liệu đích). Nhóm cũng kiểm tra tác động của việc dịch chuyển phân phối giữa dữ liệu huấn luyện và dữ liệu quần thể (dùng dữ liệu nhiễu và OOD) và của việc thay đổi kiến trúc mô hình. Khi được dùng như một "oracle" trong các tấn công tái tạo dữ liệu (Carlini et al., 2021), MIA cần hoạt động chính xác ở vùng FPR thấp để lọc bỏ số lượng khổng lồ các điểm không thuộc tập huấn luyện nhằm tìm ra thành viên trong không gian chiều cao. Phần lớn các điểm không phải thành viên được kiểm tra trong ứng dụng này là dữ liệu OOD; do đó lợi thế của việc có TPR cao ở FPR thấp chủ yếu thể hiện khi tấn công được đánh giá với lượng lớn dữ liệu không-thành-viên (có thể là OOD). Hơn nữa, trong bối cảnh này, các phương pháp MIA online³ gần như vô dụng vì phải huấn luyện rất nhiều mô hình cho mỗi truy vấn MIA. Vì vậy MIA cần vừa hiệu quả vừa bền vững trước các mẫu OOD. Nhóm thực hiện các thử nghiệm rộng để phân tích độ bền vững của MIA, và ngay cả trong các kịch bản xấu nhất, RMIA vẫn luôn vượt các tấn công khác.

> ¹ Tập trung vào Ye et al. (2022); Carlini et al. (2022); Bertran et al. (2023) — đại diện cho các phương pháp MIA mạnh trước đó.
> ² CIFAR10/100, CINIC10, ImageNet và Purchase100.
> ³ Bài báo có phân tích các tấn công online, nhưng chỉ xem chúng như các tấn công minh chứng khái niệm (proof-of-concept).

---

## 2. Thực hiện tấn công suy luận thành viên (Performing Membership Inference Attacks)

MIA xác định liệu một điểm dữ liệu $x$ có được dùng để huấn luyện một mô hình học máy $\theta$ cho trước hay không. MIA được định nghĩa bởi một trò chơi không thể phân biệt (indistinguishability game) giữa một bên thách thức (challenger) và một đối thủ (adversary, tức người kiểm toán quyền riêng tư); xem (Ye et al., 2022) để có trình bày đầy đủ về các trò chơi MIA. Bài báo dùng mẫu trò chơi/tấn công phổ biến (Homer et al., 2008; Sankararaman et al., 2009; Shokri et al., 2017; Ye et al., 2022; Carlini et al., 2022; Bertran et al., 2023). Trò chơi mô hình hóa các thí nghiệm ngẫu nhiên liên quan đến hai thế giới/giả thuyết: $H_{in}$: mô hình $\theta$ được huấn luyện trên $x$; và $H_{out}$: $x$ không nằm trong tập huấn luyện của $\theta$ (giả thuyết không). Đối thủ được đặt ngẫu nhiên vào một trong hai thế giới và phải suy ra mình đang ở đâu, chỉ dùng điểm dữ liệu $x$, mô hình đã huấn luyện $\theta$, và kiến thức nền về thuật toán huấn luyện và phân phối dữ liệu quần thể.

**Định nghĩa 2.1 (Trò chơi suy luận thành viên).** Gọi $\pi$ là phân phối dữ liệu và $A$ là thuật toán huấn luyện.

- **i –** Bên thách thức lấy mẫu tập huấn luyện $S \sim \pi$ và huấn luyện mô hình $\theta \sim A(S)$.
- **ii –** Bên thách thức tung một đồng xu công bằng $b$. Nếu $b = 1$, chọn ngẫu nhiên điểm dữ liệu $x$ từ $S$. Ngược lại, lấy mẫu $x \sim \pi$ sao cho $x \notin S$. Bên thách thức gửi mô hình đích $\theta$ và điểm dữ liệu đích $x$ cho đối thủ.
- **iii –** Đối thủ, có quyền truy cập phân phối dữ liệu quần thể $\pi$, tính $\mathrm{Score}_{MIA}(x;\theta)$ và dùng nó để đưa ra bit dự đoán thành viên $\hat b \leftarrow \mathrm{MIA}(x;\theta)$.

Một tấn công suy luận thành viên gán điểm thành viên $\mathrm{Score}_{MIA}(x;\theta)$ cho mỗi cặp $(x,\theta)$ và thực hiện kiểm định giả thuyết bằng cách xuất bit thành viên thông qua so sánh điểm số với ngưỡng $\beta$:

$$\mathrm{MIA}(x;\theta) = \mathbf{1}_{\mathrm{Score}_{MIA}(x;\theta) \ge \beta} \tag{1}$$

Công suất (tỉ lệ dương thật – TPR) và sai số (tỉ lệ dương giả – FPR) của đối thủ được định lượng qua rất nhiều lần lặp lại thí nghiệm trò chơi MIA. Ngưỡng $\beta$ điều khiển mức sai số dương giả mà đối thủ sẵn sàng chấp nhận (Sankararaman et al., 2009; Murakonda et al., 2021; Ye et al., 2022; Bertran et al., 2023).

$\mathrm{Score}_{MIA}(x;\theta)$ và phép kiểm định ở phương trình (1) được thiết kế để tối đa hóa hiệu năng kiểm định MIA, tức công suất (TPR) ở mọi FPR. (Cận dưới của) mức rò rỉ của thuật toán ML được định nghĩa là đường cong đánh đổi công suất–sai số (đường cong ROC), thu được từ kết quả các thí nghiệm trò chơi với mọi giá trị $\beta$. Bài báo chủ yếu so sánh các tấn công dựa trên đường cong TPR–FPR, đồng thời phân tích hiệu quả tính toán và độ ổn định của chúng (tức công suất thay đổi bao nhiêu khi phân phối dữ liệu và ngân sách tính toán của kẻ tấn công thay đổi).

---

## 3. Thiết kế RMIA (Designing RMIA)

Nhóm tác giả đề xuất một phép kiểm định thống kê mới cho MIA. Giả thuyết không (khi $x$ không thuộc tập huấn luyện của $\theta$) được mô hình hóa như một tổ hợp các thế giới trong đó điểm dữ liệu đích $x$ bị thay bằng một điểm dữ liệu ngẫu nhiên $z$ lấy từ quần thể. Sau đó, nhiều phép kiểm định tỉ số hợp lý theo cặp được tổng hợp lại, mỗi phép kiểm tra tư cách thành viên của $x$ so với một điểm $z$ khác. Để bác bỏ giả thuyết không, cần thu thập đủ bằng chứng (tức một phần lớn các điểm $z$ của quần thể) rằng xác suất quan sát $\theta$ dưới giả thuyết "$x$ nằm trong tập huấn luyện" lớn hơn xác suất quan sát $\theta$ khi, thay vì $x$, một điểm $z$ ngẫu nhiên nằm trong tập huấn luyện. Cách tiếp cận này cho phép phân tích rò rỉ chi tiết hơn nhiều, và phân biệt giữa các thế giới mà $x$ không phải thành viên (thay vì dựa vào khả năng trung bình của giả thuyết không). Tỉ số hợp lý ứng với cặp $x$ và $z$ được định nghĩa là:

$$\mathrm{LR}_\theta(x,z) = \frac{\Pr(\theta\mid x)}{\Pr(\theta\mid z)}, \tag{2}$$

trong đó $\Pr(\theta\mid\cdot)$ được tính trên tính ngẫu nhiên của thuật toán huấn luyện (ví dụ SGD). Số hạng $\Pr(\theta\mid x)$ là xác suất thuật toán sinh ra mô hình $\theta$ khi $x$ nằm trong tập huấn luyện, còn phần còn lại của tập huấn luyện được lấy mẫu ngẫu nhiên từ phân phối quần thể $\pi$.

**Tính tỉ số hợp lý theo cặp.** Để tính hiệu quả các giá trị LR theo cặp trong thiết lập hộp đen (đối thủ chỉ quan sát được đầu ra mô hình), áp dụng quy tắc Bayes cho phương trình (2):⁴

$$\mathrm{LR}_\theta(x,z) = \left(\frac{\Pr(x\mid\theta)}{\Pr(x)}\right)\cdot\left(\frac{\Pr(z\mid\theta)}{\Pr(z)}\right)^{-1} \tag{3}$$

Ở đây $\Pr(x\mid\theta)$ là hàm hợp lý của mô hình $\theta$ tính tại điểm dữ liệu $x$. Với các mô hình phân loại trong thiết lập MIA hộp đen, $\Pr(x\mid\theta)$ là điểm dự đoán (SoftMax) của đầu ra mô hình $f_\theta(x_{features})$ cho lớp $x_{label}$ (MacKay, 2003; Blundell et al., 2015).⁵

Cần lưu ý rằng $\Pr(x)$ không giống $\pi(x)$ (vốn là phân phối tiên nghiệm của $x$). Số hạng $\Pr(x)$ là hằng số chuẩn hóa trong quy tắc Bayes, và được tính bằng cách lấy tích phân trên mọi mô hình $\theta'$ có cùng cấu trúc và phân phối dữ liệu huấn luyện như $\theta$:

$$\Pr(x) = \sum_{\theta'} \Pr(x\mid\theta')\Pr(\theta') = \sum_{D,\theta'} \Pr(x\mid\theta')\Pr(\theta'\mid D)\Pr(D) \tag{4}$$

Trong thực tế, $\Pr(x)$ được tính bằng trung bình thực nghiệm của $\Pr(x\mid\theta')$ trên các mô hình tham chiếu $\theta'$ được lấy mẫu, mỗi mô hình huấn luyện trên một tập $D$ ngẫu nhiên rút từ phân phối quần thể $\pi$. Vì chỉ dùng một số ít cặp $D,\theta'$, cần đảm bảo các mô hình tham chiếu được lấy mẫu không thiên lệch, đặc biệt đối với việc $x$ có nằm trong dữ liệu huấn luyện $D$ của chúng hay không. Do đó $x$ cần nằm trong tập huấn luyện của một nửa số mô hình tham chiếu (mô hình IN) và không nằm trong tập huấn luyện của nửa còn lại (mô hình OUT). Cách này (tấn công online) tốn kém vì phải huấn luyện mô hình tham chiếu riêng cho từng truy vấn MIA. Để tránh chi phí đó, thuật toán offline chỉ tính $\Pr_{OUT}(x)$ bằng cách lấy trung bình $\Pr(x\mid\theta')$ trên các mô hình OUT ($x \notin D$). Để xấp xỉ nửa còn lại, $\Pr_{IN}(x)$, ta phóng đại $\Pr_{OUT}(x)$ lên, vì việc đưa một điểm dữ liệu vào tập huấn luyện thường làm tăng xác suất của nó. Xem Phụ lục B.2.2 để biết chi tiết cách tính $\Pr(x)$ không thiên lệch từ các mô hình tham chiếu ở cả hai thiết lập online và offline. Cách tính tương tự áp dụng cho $\Pr(z)$.

**Xây dựng RMIA bằng cách tổng hợp các LR theo cặp.** Với $\mathrm{LR}_\theta(x,z)$, nhóm tác giả xây dựng phép kiểm định giả thuyết cho tấn công suy luận thành viên mới RMIA như sau:

$$\mathrm{Score}_{MIA}(x;\theta) = \Pr_{z\sim\pi}\big(\mathrm{LR}_\theta(x,z)\ge\gamma\big) \tag{5}$$

Đây là xác suất để $x$ "$\gamma$-thống trị" (γ-dominate) một mẫu $z$ ngẫu nhiên từ quần thể, với ngưỡng $\gamma \ge 1$. Ngưỡng $\gamma \ge 1$ cho phép điều chỉnh xác suất học được $\theta$ khi có $x$ trong tập huấn luyện phải lớn hơn bao nhiêu so với một điểm thay thế ngẫu nhiên $z$ để vượt qua phép kiểm định.⁶ Trong thiết lập đơn giản nhất $\gamma = 1$, điểm MIA phản ánh phân vị ứng với $\Pr(x\mid\theta)/\Pr(x)$ trong phân phối của $\Pr(z\mid\theta)/\Pr(z)$ trên các mẫu $z$ ngẫu nhiên.

Ta bác bỏ giả thuyết không nếu tìm được đủ tỉ lệ các mẫu $z$ mà với chúng, khoảng cách giữa xác suất của $x$ trên mô hình đích và xác suất của $x$ trên các mô hình tham chiếu lớn hơn khoảng cách tương ứng của $z$ trong quần thể tham chiếu (các điểm không nằm trong tập huấn luyện của $\theta$). Sau đây là quy trình tấn công (mã giả chi tiết ở Phụ lục B.1).

**Định nghĩa 3.1 (Tấn công suy luận thành viên bền vững).** Gọi $\theta$ là mô hình đích và $x$ là điểm dữ liệu đích. Gọi $\gamma$ và $\beta$ là các tham số của phép kiểm định MIA. RMIA xác định $x$ có nằm trong tập huấn luyện của $\theta$ hay không bằng các bước:

- **i –** Lấy mẫu nhiều $z \sim \pi$ và tính $\mathrm{Score}_{MIA}(x;\theta)$ bằng tỉ lệ các mẫu $z$ vượt qua phép kiểm định tỉ số hợp lý theo cặp $\mathrm{LR}_\theta(x,z)\ge\gamma$. Xem phương trình (5).
- **ii –** Trả về MEMBER nếu $\mathrm{Score}_{MIA}(x;\theta)\ge\beta$, ngược lại trả về NON-MEMBER. Xem phương trình (1).

Bằng cách thực hiện phép kiểm định trên mọi giá trị $\beta\in[0,1]$, ta tính được đường cong ROC đánh đổi công suất–sai số. Như Hình 8 cho thấy, phép kiểm định được hiệu chuẩn (calibrated) theo nghĩa khi $\gamma=1$ thì FPR kỳ vọng của tấn công là $1-\beta$. Khả năng điều chỉnh tấn công để đạt một FPR cụ thể là lợi thế đáng kể khi kiểm toán thực tế để đánh giá rủi ro quyền riêng tư của mô hình.

> ⁴ $\Pr(\theta)$ triệt tiêu giữa tử số và mẫu số.
> ⁵ Xem Phụ lục B.2.1 về các cách thay thế để tính $\Pr(x\mid\theta)$.
> ⁶ Hình 7 cho thấy phép kiểm định MIA không quá nhạy với các biến thiên nhỏ của $\gamma$.

---

## 4. Vì sao RMIA là phép kiểm định mạnh hơn các tấn công trước? (Why is RMIA a More Powerful Test Compared with Prior Attacks?)

Các MIA, khi được xem như kiểm định giả thuyết, về bản chất tính khả năng tương đối của việc quan sát $\theta$ khi $x$ thuộc tập huấn luyện của $\theta$ so với khi $x$ không thuộc (giả thuyết không). Chìa khóa của một phép kiểm định mạnh là tính đến mọi nguồn thông tin giúp phân biệt các thế giới khả dĩ này. Các MIA dùng tham chiếu từ các thế giới giả thuyết, so sánh cặp $(x,\theta)$ với chúng. Thiết kế phép kiểm định hiệu quả đòi hỏi tận dụng mọi tham chiếu có thông tin, có thể là dữ liệu quần thể hoặc các mô hình huấn luyện trên chúng. Các phương pháp MIA chủ yếu tập trung vào mô hình tham chiếu, và cách dùng các mô hình tham chiếu này rất quan trọng. Như thực nghiệm cho thấy, các tấn công trước đó (Carlini et al., 2022; Ye et al., 2022) có hành vi khác nhau tùy mô hình tham chiếu (trong các kịch bản khác nhau, chúng thống trị lẫn nhau theo chiều ngược nhau). Dù vượt xa các tấn công dựa trên dữ liệu quần thể, chúng không thống trị tuyệt đối trên mọi truy vấn suy luận thành viên (Ye et al., 2022), do bỏ sót một số loại tín hiệu phân biệt.

Bảng 1 tóm tắt điểm MIA của các tấn công khác nhau. Phương pháp của nhóm đưa ra một góc nhìn mới về bài toán: tận dụng cả dữ liệu quần thể lẫn mô hình tham chiếu, qua đó tăng công suất và độ bền vững trước thay đổi trong kiến thức nền của đối thủ (ví dụ về phân phối dữ liệu huấn luyện và quần thể cũng như cấu trúc mô hình). Phép kiểm định tỉ số hợp lý trong phương trình (5) và (3) đo mức khả phân biệt giữa $x$ và bất kỳ $z$ nào dựa trên sự dịch chuyển xác suất của chúng khi điều kiện theo $\theta$, thông qua đối chiếu $\Pr(x\mid\theta)/\Pr(x)$ với $\Pr(z\mid\theta)/\Pr(z)$. Các công trình trước có thể được xem là phiên bản trung bình (average-case) và chưa hiệu chuẩn của phép kiểm định này:

- **Attack-P** (Ye et al., 2022) và các phương pháp liên quan (Shokri et al., 2017; Chang & Shokri, 2021; Bertran et al., 2023) chủ yếu dựa vào việc khả năng của mô hình đích trên $x$ và $z$ thay đổi ra sao, và bỏ qua hoặc không nắm bắt chính xác số hạng $\Pr(x)/\Pr(z)$ (tức ngầm giả định $\Pr(x)=\Pr(z)$). Sức mạnh của phép kiểm định nằm ở khả năng phát hiện khác biệt tinh tế trong các tỉ số $\Pr(x\mid\theta)/\Pr(z\mid\theta)$ và $\Pr(z)/\Pr(x)$, phản ánh sự thay đổi LR đáng chú ý do việc đưa $x$ vào tập huấn luyện đích. Trong phép kiểm định này, một điểm tham chiếu $z$ chỉ "bỏ phiếu" cho tư cách thành viên của $x$ nếu tỉ số $\Pr(x)/\Pr(z)$ được phóng đại khi các xác suất điểm được tính trên mô hình đích $\theta$ (cho thấy $\theta$ khớp $x$ tốt hơn).
- Các tấn công mạnh trước đó dùng mô hình tham chiếu, đặc biệt trong (Ye et al., 2022) và các tấn công tương tự (Watson et al., 2022b), bỏ qua thành phần $\Pr(z\mid\theta)/\Pr(z)$. Việc hiệu chuẩn bằng $z$ cho biết độ lớn của $\Pr(x\mid\theta)/\Pr(x)$ có đáng kể hay không (so với các điểm không thành viên); thiếu nó, các tấn công kém hiệu quả trên toàn đường cong TPR–FPR.
- **LiRA** (Carlini et al., 2022) cũng thất bại theo cách tương tự — thiếu hiệu chuẩn thiết yếu bằng dữ liệu quần thể.⁷ Nhưng sự thiếu tin cậy của LiRA không chỉ vì lý do này. Để giải thích rõ hơn lợi thế của mình, nhóm tác giả trình bày một cách tính khác cho phương trình LR (2), qua đó cho thấy LiRA là một trường hợp trung bình của RMIA.

Trong thiết lập hộp đen, độ phân kỳ giữa phân phối đầu ra của một mô hình huấn luyện trên $x$ và mô hình leave-one-out tương ứng (không huấn luyện trên $x$), khi phân phối được tính trên tính ngẫu nhiên của thuật toán huấn luyện, đạt cực đại khi các mô hình được truy vấn tại điểm khác biệt $x$ (Ye et al., 2024). Do đó, xấp xỉ tốt cho tỉ số hợp lý trong thiết lập hộp đen là đánh giá xác suất của $f_\theta(x_{features})$ và $f_\theta(z_{features})$, với $f_\theta(\cdot)$ là mô hình phân loại tham số $\theta$. Một cách tính LR trực tiếp ở phương trình (2) là:

$$\mathrm{LR}_\theta(x,z)=\frac{\Pr(\theta\mid x)}{\Pr(\theta\mid z)}\approx\frac{\Pr\big(f_\theta(x),f_\theta(z)\mid x\big)}{\Pr\big(f_\theta(x),f_\theta(z)\mid z\big)}, \tag{6}$$

trong đó các số hạng có thể tính như ở Phụ lục B.3 và (Carlini et al., 2022). LiRA (online) tính trung bình của LR này trên mọi $z$, làm giảm công suất kiểm định. Ngoài ra, việc tính LR trực tiếp đòi hỏi nhiều mô hình tham chiếu để xây dựng phép kiểm định ổn định. Phụ lục B.3 so sánh thực nghiệm giữa cách tính chính (phương trình 3, dùng quy tắc Bayes) và cách tính trực tiếp. Kết quả cho thấy cấu trúc RMIA dùng quy tắc Bayes (phương trình 3) rất bền vững và thống trị cách tính LR trực tiếp (phương trình 6) khi chỉ dùng vài mô hình tham chiếu (Hình 20), và hai cách tương đương khi dùng nhiều mô hình tham chiếu (Hình 19). Vì vậy tấn công của nhóm thống trị tuyệt đối LiRA trên toàn đường cong công suất–sai số (TPR–FPR), và khoảng cách tăng đáng kể khi giới hạn ngân sách mô hình tham chiếu (xem Hình 25). Sự kết hợp giữa LR theo cặp và cách tính theo hướng Bayes tạo nên một tấn công bền vững, công suất cao và chi phí thấp, mà ở chế độ offline chỉ cần huấn luyện các mô hình OUT.

> ⁷ Áp dụng quy tắc Bayes lên tử số của LR trong LiRA và lưu ý rằng mẫu số bằng $\Pr(\theta)$ khi các mô hình tham chiếu OUT được lấy mẫu từ quần thể (chứ không phải một tập hữu hạn nhỏ), thì $\mathrm{Score}_{MIA}(x;\theta)$ của LiRA bằng $\Pr(x\mid\theta)/\Pr(x)$.

**Bảng 1: Cách tính $\mathrm{Score}_{MIA}(x;\theta)$ trong các tấn công suy luận thành viên khác nhau**, trong đó ký hiệu $\bar x$ (đối với LiRA) biểu thị trường hợp $x$ không nằm trong tập huấn luyện. Tấn công là $\mathrm{MIA}(x;\theta)=\mathbf 1_{\mathrm{Score}_{MIA}(x;\theta)\ge\beta}$ theo Định nghĩa 2.1.

| Phương pháp | RMIA (bài báo này) | LiRA (Carlini et al., 2022) | Attack-R (Ye et al., 2022) | Attack-P (Ye et al., 2022) | Global (Yeom et al., 2018) |
|---|---|---|---|---|---|
| Điểm MIA | $\Pr_z\!\left(\dfrac{\Pr(\theta\mid x)}{\Pr(\theta\mid z)}\ge\gamma\right)$ | $\dfrac{\Pr(\theta\mid x)}{\Pr(\theta\mid\bar x)}$ | $\Pr_{\theta'}\!\left(\dfrac{\Pr(x\mid\theta)}{\Pr(x\mid\theta')}\ge1\right)$ | $\Pr_z\!\left(\dfrac{\Pr(x\mid\theta)}{\Pr(z\mid\theta)}\ge1\right)$ | $\Pr(x\mid\theta)$ |

---

## 5. Đánh giá thực nghiệm (Empirical Evaluation)

*(Từ phần này trở đi, nội dung được trình bày dưới dạng tóm lược theo sát bài gốc; các số liệu chính được giữ nguyên.)*

Mục tiêu của phần đánh giá là phân tích:

1. Hiệu năng của các tấn công khi ngân sách huấn luyện mô hình tham chiếu bị hạn chế, kể cả khi giới hạn ở chế độ offline với các mô hình tham chiếu huấn luyện sẵn.
2. Công suất tối đa của các tấn công khi có thể huấn luyện không giới hạn số mô hình tham chiếu (chế độ online).
3. Khả năng phân biệt thành viên với không thành viên khi các điểm dữ liệu đích là OOD — phản ánh độ bền vững và tính hữu dụng của tấn công khi làm oracle để chia toàn bộ không gian dữ liệu thành thành viên và không thành viên.
4. Tác động của kiến thức đối thủ lên hiệu năng (đặc biệt là dịch chuyển phân phối của dữ liệu quần thể và sai khác kiến trúc giữa mô hình đích và mô hình tham chiếu).

**Thiết lập (Setup).** Thí nghiệm trên CIFAR-10, CIFAR-100, CINIC-10, ImageNet và Purchase-100 — các bộ dữ liệu chuẩn thường dùng để đánh giá MIA. Các chỉ số chuẩn gồm đường cong FPR–TPR và diện tích dưới đường cong ROC (AUC). Chi tiết (bộ dữ liệu, mô hình, siêu tham số, chỉ số) ở Phụ lục A.

**Chế độ tấn công (offline và online).** Mô hình tham chiếu được huấn luyện trên các mẫu quần thể. Đối thủ cũng có thể mô phỏng kịch bản leave-one-out (Ye et al., 2024) và thực hiện tấn công online bằng cách huấn luyện một số mô hình tham chiếu trên chính dữ liệu đích (truy vấn MIA) (Carlini et al., 2022). Tác giả xem các tấn công online chỉ là minh chứng khái niệm do chi phí cao. Mô hình huấn luyện trên dữ liệu đích gọi là mô hình **IN**, mô hình không huấn luyện trên đó gọi là mô hình **OUT**. Ở chế độ offline, mọi mô hình tham chiếu đều là OUT.

**Các tấn công cơ sở (Baseline).** RMIA được so sánh chủ yếu với các tấn công tiên tiến nhất (Ye et al., 2022; Carlini et al., 2022), vốn đã vượt các phương pháp trước đó. Cụ thể: **Attack-P** (không dùng mô hình tham chiếu; tương đương tấn công LOSS của Yeom et al., 2018 — đặt ngưỡng lên tín hiệu loss của mô hình đích), **Attack-R** (dùng mô hình tham chiếu ở chế độ offline), **LiRA** (chế độ online và offline), và **Quantile Regression** (Bertran et al., 2023). Quantile Regression vượt LiRA trong một số kịch bản trên ImageNet do độ bất định cao của tín hiệu LiRA trên mô hình lớn, nhưng tỏ ra yếu hơn trên các bộ dữ liệu khác, kể cả khi LiRA chỉ dùng vài mô hình tham chiếu.

**Tái hiện kết quả.** Với các công trình trước, nhóm chỉ dùng mã nguồn do chính tác giả của chúng cung cấp.⁸ ⁹ ¹⁰ Nhóm nêu rõ sự sai lệch trong kết quả của phiên bản offline của LiRA (dùng mã của tác giả), cũng đã được ghi nhận bởi (Ye et al., 2022; Wen et al., 2023): công suất đạt được thấp hơn nhiều so với báo cáo trong (Carlini et al., 2022). RMIA hoạt động ở cả hai chế độ offline và online (Phụ lục B.2.2).

> ⁸ https://github.com/privacytrustlab/ml_privacy_meter/tree/master/research/ (Ye et al., 2022 và RMIA)
> ⁹ https://github.com/tensorflow/privacy/tree/master/research/mi_lira_2021 (Carlini et al., 2022)
> ¹⁰ https://github.com/amazon-science/quantile-mia (Bertran et al., 2023)

**Bảng 2: Hiệu năng của các tấn công khi dùng một vài mô hình tham chiếu.** Mọi tấn công, trừ LiRA (Online), đều offline (không cần huấn luyện mô hình tham chiếu mới cho mỗi truy vấn). Quantile Regression huấn luyện các mô hình tấn công dựa trên hồi quy (thay vì mô hình tham chiếu) để tính ngưỡng cho một giá trị FPR nhất định; việc điều chỉnh siêu tham số đòi hỏi huấn luyện nhiều mô hình (Bertran et al., 2023). Kết quả là trung bình trên 10 mô hình đích ngẫu nhiên (độ lệch chuẩn ở Bảng 9, Phụ lục C.5). Kết quả Purchase-100 ở Bảng 10 (Phụ lục C.8). Các cột TPR@FPR ghi theo FPR 0,01% / 0,0%.

| # Ref | Tấn công | CIFAR-10 AUC | TPR@0,01% | TPR@0,0% | CIFAR-100 AUC | TPR@0,01% | TPR@0,0% | CINIC-10 AUC | TPR@0,01% | TPR@0,0% | ImageNet AUC | TPR@0,01% | TPR@0,0% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Attack-P | 58,19 | 0,01 | 0,0 | 75,91 | 0,01 | 0,0 | 66,91 | 0,01 | 0,0 | 64,48 | 0,01 | 0,0 |
| * | Quantile-Reg. | 61,45 | 0,08 | 0,03 | 83,32 | 0,26 | 0,04 | 73,48 | 0,35 | 0,12 | 70,8 | 0,06 | 0,0 |
| 1 | Attack-R | 63,65 | 0,07 | 0,02 | 81,61 | 0,06 | 0,02 | 72,04 | 0,07 | 0,02 | 70,91 | 0,02 | 0,0 |
| 1 | LiRA | 53,2 | 0,48 | 0,25 | 68,95 | 0,54 | 0,27 | 59,93 | 0,32 | 0,07 | 58,66 | 0,01 | 0,0 |
| 1 | RMIA | 68,64 | 1,19 | 0,51 | 87,18 | 2,06 | 0,77 | 79 | 0,86 | 0,32 | 72,21 | 0,06 | 0,01 |
| 2 | Attack-R | 63,35 | 0,32 | 0,08 | 81,52 | 0,31 | 0,06 | 72,02 | 0,21 | 0,07 | 71,03 | 0,04 | 0,01 |
| 2 | LiRA | 54,42 | 0,67 | 0,27 | 72,21 | 1,52 | 0,76 | 62,18 | 0,57 | 0,26 | 61,05 | 0,01 | 0,0 |
| 2 | LiRA (Online) | 63,97 | 0,76 | 0,43 | 84,55 | 1,15 | 0,55 | 73,17 | 0,53 | 0,12 | 67,55 | 0,01 | 0,01 |
| 2 | RMIA | 70,13 | 1,71 | 0,91 | 88,92 | 4,9 | 1,73 | 80,56 | 2,14 | 0,98 | 73,95 | 0,16 | 0,02 |
| 4 | Attack-R | 63,52 | 0,65 | 0,21 | 81,78 | 0,63 | 0,19 | 72,18 | 0,4 | 0,14 | 71,11 | 0,07 | 0,02 |
| 4 | LiRA | 54,6 | 0,97 | 0,57 | 73,57 | 2,26 | 1,14 | 63,07 | 1,03 | 0,45 | 62,69 | 0,01 | 0,0 |
| 4 | LiRA (Online) | 67 | 1,38 | 0,51 | 87,82 | 3,64 | 2,19 | 77,06 | 1,34 | 0,51 | 71,44 | 0,01 | 0,0 |
| 4 | RMIA | 71,02 | 2,91 | 2,13 | 89,81 | 7,05 | 3,5 | 81,46 | 3,2 | 1,39 | 74,98 | 0,45 | 0,06 |

### 5.1. Tấn công suy luận với ngân sách tính toán thấp (Inference Attack under Low Computation Budget)

**Dùng một vài mô hình tham chiếu.** Hình 1 cho thấy đường cong TPR–FPR của mọi tấn công khi giới hạn 1 mô hình tham chiếu: RMIA vượt Attack-R và LiRA trên mọi giá trị FPR. Bảng 2 so sánh kết quả trong kịch bản chi phí thấp. Trọng tâm là các tấn công offline với số mô hình tham chiếu cố định, huấn luyện sẵn, dùng chung cho mọi truy vấn. LiRA (Online) cũng được đưa vào làm chuẩn so sánh dù chi phí cao (một nửa số mô hình tham chiếu phải huấn luyện trên dữ liệu đích). Bảng cũng gồm kết quả của tấn công Quantile Regression (Bertran et al., 2023): phương pháp này không dùng mô hình tham chiếu nhưng huấn luyện nhiều mô hình hồi quy để xác định ngưỡng tấn công cho từng mẫu (đạt một giá trị FPR nhất định), và cần một mô hình tấn công cho mỗi FPR, nên được xem là tốn kém.

RMIA thống trị tuyệt đối các công trình trước trên mọi bộ dữ liệu. Ví dụ, chỉ với 2 mô hình tham chiếu CIFAR-10, RMIA đạt AUC cao hơn khoảng 10% so với Attack-R và LiRA và vẫn có TPR tại FPR bằng 0 tốt hơn. Đáng chú ý, RMIA (offline) cũng thống trị LiRA (Online): với 4 mô hình CIFAR-10, RMIA có AUC cao hơn ít nhất 6% và TPR tại FPR bằng 0 cải thiện đáng kể 3× so với LiRA (Online). Trong trường hợp cực đoan chỉ dùng một mô hình tham chiếu, RMIA có AUC cao hơn LiRA ít nhất 24% và TPR tốt hơn ở FPR thấp trên mọi bộ dữ liệu; tấn công này cũng vượt Quantile Regression về cả AUC lẫn TPR ở FPR thấp. Hơn nữa, hiệu năng RMIA còn có thể cải thiện thêm khi thêm mô hình tham chiếu, trong khi Quantile Regression không còn dư địa như vậy. Attack-R vượt LiRA (Offline) và đạt AUC tương đối cao, nhưng kém ở vùng FPR thấp hơn. Attack-P không dùng mô hình tham chiếu nhưng vẫn vượt LiRA ở chế độ offline.

**Dùng dữ liệu quần thể.** Khi số mô hình tham chiếu thấp, mọi tấn công phải dùng thông tin từ dữ liệu quần thể. Cụ thể, LiRA cần tính trung bình và phương sai của rescaled-logit trên các mô hình tham chiếu; khi có ít mô hình (dưới 64 mô hình, theo Carlini et al., 2022 [Hình 9]), dùng phương sai toàn cục trên dữ liệu quần thể/kiểm tra có lợi. Tương tự Attack-R dùng các kỹ thuật làm trơn để xấp xỉ CDF của phân phối loss trên dữ liệu quần thể/kiểm tra nhằm tính phân vị (Ye et al., 2022); Quantile Regression cũng dùng một tập lớn dữ liệu quần thể để huấn luyện mô hình tấn công. Do đó không có khác biệt đáng kể giữa các tấn công về nhu cầu dữ liệu quần thể bổ sung.

*[Hình 2: ROC của các tấn công trên mô hình ImageNet. Kết quả lấy trên một mô hình đích ngẫu nhiên, dùng 1 mô hình tham chiếu (OUT). Bảng 2 báo cáo AUC của các tấn công.]*

*[Hình 3: Số lượng mô hình tham chiếu so với AUC của các tấn công trên CIFAR-10 (trục hoành: 2⁰ … 2⁸ mô hình tham chiếu; các đường: RMIA (Online), RMIA (Offline), LiRA (Online), LiRA (Offline), Attack-R, Attack-P). Ở các tấn công online, một nửa số mô hình tham chiếu phải được huấn luyện cho mỗi truy vấn MIA.]*

**Tấn công mô hình lớn với tập huấn luyện lớn.** Như Bertran et al. (2023) lưu ý, MIA có thể suy giảm hiệu năng trên mô hình có tập huấn luyện lớn hơn. Bảng 2 trình bày kết quả trên ImageNet (lớn hơn CIFAR khoảng 20 lần). Quantile Regression vượt LiRA, kể cả LiRA online với vài mô hình tham chiếu. Tuy nhiên RMIA offline chỉ với một mô hình tham chiếu vẫn vượt mọi tấn công khác cả về AUC lẫn TPR ở FPR thấp, ví dụ AUC cao hơn Quantile Regression khoảng 2%; khoảng cách này mở rộng đáng kể khi thêm mô hình tham chiếu. Hình 2 cho thấy RMIA luôn đạt TPR vượt trội trên mọi giá trị FPR.

**Dùng nhiều mô hình tham chiếu hơn ở chế độ offline.** Bảng 3 so sánh các tấn công offline khi dùng nhiều mô hình OUT hơn (127 mô hình). RMIA luôn vượt các tấn công offline khác trên mọi bộ dữ liệu, với công suất lớn hơn nhiều so với Attack-R (tấn công offline mạnh nhất trước đó). Attack-R có thể bác bỏ nhầm một thành viên điển hình thành không thành viên chỉ vì nó có xác suất cao hơn trên các mô hình tham chiếu; kết quả là AUC thấp hơn RMIA 5%–10% trên mọi bộ dữ liệu. RMIA được thiết kế để khắc phục hạn chế này bằng cách xét cả đặc trưng của mẫu đích trong các mô hình tham chiếu lẫn xác suất tương đối của nó so với các bản ghi quần thể khác. So sánh Bảng 2 và Bảng 3 cho thấy AUC của RMIA với vài mô hình tham chiếu gần như bằng khi dùng số lượng lớn (127), tức công suất tổng thể không phụ thuộc đáng kể vào việc có nhiều mô hình tham chiếu; tuy vậy TPR ở vùng FPR thấp tăng khi thêm mô hình. Để phân tích sâu sự khác biệt giữa các tấn công, Phụ lục C.10 so sánh sự biến thiên điểm MIA của mẫu thành viên và không thành viên.

### 5.2. Công suất tối đa của các tấn công suy luận (online) (Ultimate Power of (Online) Inference Attacks)

Bảng 3 trình bày hiệu năng của mọi tấn công trên các mô hình huấn luyện với các bộ dữ liệu khác nhau khi đủ tài nguyên để huấn luyện nhiều mô hình tham chiếu (254 mô hình IN và OUT). Ở thiết lập này, RMIA (Online) tốt hơn LiRA (Online) (vốn hưởng lợi đáng kể từ nhiều mô hình tham chiếu) về AUC. Khi tiến gần mức rò rỉ tối đa của thuật toán huấn luyện, ngay cả cải thiện AUC nhỏ cũng quan trọng. Điều quan trọng hơn là TPR tại FPR bằng 0 của RMIA (Online) tốt hơn LiRA (Online) đáng kể (tới 50%). Tấn công offline của nhóm đạt hiệu năng tương đương các tấn công online, điều đáng chú ý khi xét khoảng cách chi phí lớn giữa hai chế độ.

Có thể cải thiện thêm bằng cách truy vấn mô hình đích với nhiều biến đổi tăng cường (augmentation) của truy vấn đầu vào, thu được qua các phép lật gương và dịch chuyển đơn giản trên dữ liệu ảnh. Trong thiết lập đa truy vấn, nhóm dùng biểu quyết đa số trên phép kiểm định giả thuyết ở phương trình (2): truy vấn $x$ được coi là thống trị bản ghi quần thể $z$ nếu hơn một nửa số biến đổi của $x$ thống trị $z$ (xem Phụ lục B.5). RMIA tận dụng kỹ thuật này tốt hơn nhiều, và có thể đạt TPR tại FPR bằng 0 cao hơn 4× và AUC cao hơn khoảng 4,6% so với LiRA. Kết quả dùng 18 truy vấn tăng cường.

**Bảng 3: Hiệu năng của các tấn công khi dùng số lượng lớn mô hình tham chiếu, trung bình trên 10 mô hình đích ngẫu nhiên (trích: CIFAR-10).** Bốn hàng đầu là các tấn công offline dùng 127 mô hình tham chiếu OUT (trừ Attack-P không dùng mô hình tham chiếu). Hai hàng cuối là tấn công online, dùng 127 mô hình OUT và 127 mô hình IN. (Bài báo gốc còn báo cáo CIFAR-100 và CINIC-10 với cùng xu hướng; ví dụ AUC của RMIA lần lượt là 90,57 và 82,33 ở chế độ offline.)

| Tấn công | AUC | TPR@FPR 0,01% | TPR@FPR 0,0% |
|---|---|---|---|
| Attack-P | 58,19 ± 0,33 | 0,01 ± 0,01 | 0,00 ± 0,01 |
| Attack-R | 64,41 ± 0,41 | 1,52 ± 0,33 | 0,80 ± 0,43 |
| LiRA | 55,18 ± 0,37 | 1,37 ± 0,32 | 0,72 ± 0,31 |
| RMIA | 71,71 ± 0,43 | 4,18 ± 0,61 | 3,14 ± 0,87 |
| LiRA (online) | 72,04 ± 0,47 | 3,39 ± 0,86 | 2,01 ± 0,78 |
| RMIA (online) | 72,25 ± 0,46 | 4,31 ± 0,47 | 3,15 ± 0,61 |

### 5.3. Sự phụ thuộc vào sự sẵn có của mô hình tham chiếu (Dependency on Availability of Reference Models)

Các tấn công suy luận phải bền vững trước thay đổi trong giả định về tri thức nền của đối thủ, ví dụ dữ liệu quần thể dùng để huấn luyện mô hình tham chiếu. Các phương pháp MIA chi phí thấp cũng nên ít phụ thuộc vào số lượng lớn mô hình tham chiếu.

Hình 3 trình bày số mô hình tham chiếu cần thiết để đạt hiệu năng tổng thể (tính bằng AUC). Các tấn công online cần ít nhất 2 mô hình (1 IN, 1 OUT). Khác với các tấn công mạnh khác, RMIA cho kết quả ổn định, không thay đổi nhiều khi giảm số mô hình tham chiếu: RMIA vẫn hưởng lợi từ việc tăng mô hình, nhưng ngay cả với số ít mô hình đã rất gần hiệu năng tối đa. Ngược lại LiRA (Online) rất nhạy với số mô hình (khoảng cách AUC khoảng 8% trên CIFAR-10 giữa 2 và 254 mô hình), cho thấy cần số lượng lớn mô hình để tấn công này hoạt động hiệu quả. Đáng chú ý, RMIA (Offline) mạnh hơn LiRA (Online) trừ khi dùng hàng trăm mô hình tham chiếu. Phụ lục C.7 trình bày kết quả bổ sung với số mô hình khác nhau trên các bộ dữ liệu khác, cho thấy xu hướng vượt trội tương tự CIFAR-10.

### 5.4. Độ bền vững của các tấn công trước các điểm không-thành-viên OOD (Robustness of Attacks against OOD Non-members)

Nhóm thử thách các MIA bằng dữ liệu không-thành-viên ngoài phân phối (OOD). Một MIA mạnh phải loại bỏ được mọi không-thành-viên dù chúng cùng phân phối với dữ liệu huấn luyện hay không; điều này đặc biệt quan trọng khi dùng oracle MIA cho các ứng dụng như tấn công trích xuất dữ liệu (Carlini et al., 2021). Trong các trường hợp đó, tấn công phải giữ độ chính xác (TPR cao ở mọi FPR) trên dữ liệu không-thành-viên OOD đồng thời phát hiện thành viên. Lưu ý rằng các kỹ thuật lọc đơn giản không giải quyết được vấn đề này: dù mẫu OOD thường có độ tin cậy thấp hơn mẫu trong phân phối, chỉ lọc theo độ tin cậy dẫn đến TPR thấp vì loại bỏ các mẫu thành viên khó.

Để kiểm tra, mô hình được huấn luyện trên CIFAR-10 và dùng mẫu từ CINIC-10 làm truy vấn kiểm tra không-thành-viên OOD. Thiết lập trò chơi MIA này khác Định nghĩa 2.1 do $x\not\sim\pi$, nên kết quả không thể so trực tiếp với trò chơi gốc. Nhóm tập trung vào các tấn công offline vì không khả thi khi huấn luyện mô hình IN cho từng mẫu kiểm tra OOD với lượng truy vấn lớn.

Hình 4 cho thấy các đường ROC. RMIA có lợi thế hiệu năng lớn (AUC cao hơn ít nhất 21%) so với Attack-R, còn LiRA chỉ hơn đoán ngẫu nhiên không đáng kể. Attack-P đạt AUC mạnh nhờ khai thác khoảng cách độ tin cậy/loss giữa mẫu OOD và mẫu trong phân phối, nhưng như đã giải thích, gặp khó khăn để đạt TPR tốt ở FPR thấp, và cũng không mạnh trong việc phát hiện không-thành-viên trong phân phối. Việc kiểm định bằng LR theo cặp giữa mẫu đích $x$ và nhiều mẫu $z$ trong phân phối cho phép nắm bắt khác biệt đáng kể giữa $\Pr(x\mid\theta)/\Pr(x)$ và $\Pr(z\mid\theta)/\Pr(z)$ trong phương trình (3) để phân biệt hiệu quả các truy vấn OOD với thành viên. Kết quả khi dùng nhiễu thuần túy làm truy vấn không-thành-viên ở Hình 9 (Phụ lục C.1) cho thấy khoảng cách giữa RMIA và các phương pháp khác còn lớn hơn so với mẫu OOD.

*[Hình 4: ROC của các tấn công offline dùng mô hình huấn luyện trên CIFAR-10, trong khi các truy vấn không-thành-viên là mẫu OOD từ CINIC-10. Dùng 127 mô hình tham chiếu. AUC: RMIA 0,768; Attack-P 0,75; Attack-R 0,632; LiRA 0,509.]*

### 5.5. Độ bền vững trước dịch chuyển mô hình và phân phối dữ liệu (Robustness to Model and Data Distribution Shift)

Khi đánh giá MIA, cần xem hiệu năng chịu ảnh hưởng thế nào nếu đối thủ không biết chính xác phân phối dữ liệu huấn luyện và cấu trúc mô hình đích. Nhóm so sánh kết quả khi mô hình tham chiếu được huấn luyện trên bộ dữ liệu khác với mô hình đích: mô hình đích huấn luyện trên CIFAR-10, mô hình tham chiếu trên CINIC-10. Hiệu năng của mọi tấn công đều bị ảnh hưởng khi có dịch chuyển phân phối giữa tập huấn luyện của mô hình đích và mô hình tham chiếu. Tuy nhiên RMIA luôn có AUC cao hơn (ví dụ tới 25% so với LiRA khi dùng 2 mô hình tham chiếu) và TPR tốt hơn ở FPR thấp. Điều này xác nhận tấn công vẫn hiệu quả khi dùng các mô hình huấn luyện sẵn trên bộ dữ liệu khác làm mô hình tham chiếu, với chi phí huấn luyện bằng không. Bảng 8 ở Phụ lục C.2 trình bày chi tiết.

Nhóm cũng nghiên cứu tác động của việc thay đổi kiến trúc mạng giữa mô hình đích và mô hình tham chiếu. Hiệu năng tối ưu của mọi tấn công đạt được khi cả hai chia sẻ kiến trúc tương tự, nhưng ưu thế của RMIA rõ rệt hơn khi có dịch chuyển kiến trúc (khoảng cách AUC giữa RMIA và các tấn công khác tăng tới 3%). Xem chi tiết ở Phụ lục C.3.

### 5.6. Phân tích các tham số của RMIA (Analyzing RMIA Parameters)

**Số mẫu $z$.** Tấn công đánh giá tỉ số hợp lý của mô hình đích trên điểm dữ liệu đích $x$ so với các mẫu quần thể $z$ khác. Việc tính LR so với các mẫu quần thể cho phép tính thống kê kiểm định đáng tin cậy. Bảng 4 cho thấy kết quả của RMIA thay đổi thế nào theo số mẫu tham chiếu. AUC tăng khi tăng số mẫu $z$, nhưng đáng chú ý là dùng 2500 mẫu quần thể, tương đương 10% kích thước tập huấn luyện của mô hình, cho kết quả tương đương với tập quần thể lớn gấp 10 lần. Hơn nữa TPR ở FPR thấp vẫn cao ngay cả khi chỉ dùng 250 mẫu tham chiếu. Xu hướng này nhất quán khi dùng ít mô hình tham chiếu hơn (xem Bảng 11, Phụ lục C.9). Trong thiết lập mặc định, khi tấn công mô hình đích với mẫu $x$, tất cả các mẫu không-thành-viên của mô hình đích được dùng làm tập $z$; truy vấn $x$ không được dùng làm $z$.

**Bảng 4: Hiệu năng của RMIA (Online) với số mẫu $z$ ngẫu nhiên khác nhau.** Số mô hình tham chiếu huấn luyện trên CIFAR-10 là 254 (127 IN, 127 OUT). Không dùng truy vấn tăng cường. Trung bình trên 10 mô hình đích ngẫu nhiên.

| # mẫu quần thể $z$ | AUC | TPR@FPR 0,01% | TPR@FPR 0,0% |
|---|---|---|---|
| 25 | 58,71 ± 0,26 | 1,32 ± 0,15 | 1,15 ± 0,34 |
| 250 | 64,88 ± 0,24 | 2,23 ± 0,38 | 1,65 ± 0,46 |
| 1250 | 67,57 ± 0,30 | 2,22 ± 0,41 | 1,73 ± 0,49 |
| 2500 | 68,25 ± 0,27 | 2,26 ± 0,45 | 1,72 ± 0,49 |
| 6250 | 68,78 ± 0,31 | 2,28 ± 0,44 | 1,75 ± 0,51 |
| 12500 | 69,03 ± 0,33 | 2,28 ± 0,43 | 1,77 ± 0,52 |
| 25000 | 69,15 ± 0,35 | 2,26 ± 0,43 | 1,80 ± 0,55 |

**Độ nhạy với ngưỡng γ của kiểm định LR theo cặp.** Kết quả ở Phụ lục B.4 cho thấy hiệu năng của tấn công nhất quán trước thay đổi của $\gamma$. Cả AUC lẫn đường cong FPR–TPR tương đối ổn định khi $\gamma$ thay đổi nhỏ (trừ khi $\gamma$ rất cao). Thực tế, bằng cách điều chỉnh ngưỡng $\beta$, ta đạt gần như cùng kết quả trên các giá trị $\gamma$ khác nhau.

### 5.7. Áp dụng RMIA cho các thuật toán ML khác (Applying RMIA to Other ML Algorithms)

Để đánh giá hiệu năng các tấn công trên các thuật toán ML khác, nhóm xem xét rủi ro quyền riêng tư của các thuật toán Gradient Boosting Decision Tree (GBDT). Ở đây TPR của RMIA luôn vượt mọi tấn công khác, đặc biệt hơn một bậc độ lớn tại FPR bằng 0, như Hình 15 (Phụ lục C.6).

---

## 6. Kết luận (Conclusions)

Nhóm tác giả cho rằng các phép kiểm định MIA, với vai trò công cụ kiểm toán quyền riêng tư, phải được thử nghiệm áp lực (stress-test) với ngân sách tính toán thấp, ít mô hình tham chiếu, và thay đổi về phân phối dữ liệu và mô hình. Một phép kiểm định mạnh là phép kiểm định vượt các phương pháp khác trong những kịch bản đó chứ không chỉ trong các kịch bản điển hình. Bài báo đề xuất một phép kiểm định thống kê mới cho MIA và một chuỗi các kịch bản đánh giá để so sánh các MIA về hiệu quả và độ bền vững. RMIA có thể được dùng một cách đáng tin cậy để kiểm toán rủi ro quyền riêng tư trong ML dưới các giả định thực tiễn.

## Lời cảm ơn (Acknowledgements)

Nhóm tác giả cảm ơn Vincent Bindschaedler, Jiayuan Ye và Hongyan Chang vì các nhận xét về các phiên bản trước của công trình, cũng như các phản biện viên vì những góp ý quý giá và cuộc thảo luận sôi nổi trong quá trình phản hồi. Công trình của Reza Shokri được hỗ trợ bởi Asian Young Scientist Fellowship 2023 và Quỹ Nghiên cứu Học thuật (AcRF) Tier 1 của Bộ Giáo dục Singapore.

---

## Phụ lục (Appendix)

*(Các phụ lục dưới đây được tóm lược; các bảng kết quả chi tiết trải dài nhiều trang được rút gọn và có ghi chú tại chỗ.)*

### A. Thiết lập thí nghiệm (Experimental Setup)

Để thực hiện tấn công, trước hết phải huấn luyện mô hình. Nhóm dùng cùng thiết lập huấn luyện với (Carlini et al., 2022): với mỗi bộ dữ liệu, các mô hình được huấn luyện trên các tập huấn luyện được chọn ngẫu nhiên, mỗi tập chứa một nửa bộ dữ liệu; mỗi mẫu của bộ dữ liệu nằm trong tập huấn luyện của đúng một nửa số mô hình tham chiếu. Các mô hình đích được chọn ngẫu nhiên từ tập mô hình đã huấn luyện. Trong thiết lập này, tập huấn luyện của mô hình đích có thể trùng 50% với mỗi mô hình tham chiếu.

**Bộ dữ liệu.** Kết quả tấn công được báo cáo trên mô hình huấn luyện với năm bộ dữ liệu truyền thống dùng để đánh giá MIA:

- **CIFAR-10** (He et al., 2016): huấn luyện Wide ResNet (độ sâu 28, độ rộng 2) trong 100 epoch trên một nửa bộ dữ liệu chọn ngẫu nhiên.
- **CIFAR-100** và **CINIC-10**: quy trình tương tự CIFAR-10, huấn luyện wide ResNet trên một nửa bộ dữ liệu (với CINIC-10, chọn ngẫu nhiên 50k mẫu trong 270k để huấn luyện); batch size 256.
- **ImageNet** (~1,2 triệu ảnh, 1000 lớp): huấn luyện ResNet-50 trên một nửa bộ dữ liệu trong 100 epoch, batch size 256, learning rate 0,1, weight decay 1e-4.
- **Purchase-100** (dữ liệu bảng về lịch sử mua sắm; Shokri et al., 2017): MLP 4 lớp với số đơn vị [512, 256, 128, 64], huấn luyện trên 25k mẫu trong 50 epoch.

**Bảng 5: Độ chính xác trung bình của mô hình huấn luyện trên các bộ dữ liệu khác nhau.**

| Bộ dữ liệu | Độ chính xác huấn luyện | Độ chính xác kiểm tra |
|---|---|---|
| CIFAR-10 | 99,9% | 92,4% |
| CIFAR-100 | 99,9% | 67,5% |
| CINIC-10 | 99,5% | 77,2% |
| ImageNet | 90,2% | 58,6% |
| Purchase-100 | 100% | 83,4% |

**Các chỉ số đánh giá.** Hiệu năng mỗi tấn công được đo bằng TPR và FPR trên mọi bản ghi thành viên và không-thành-viên của các mô hình đích ngẫu nhiên. Đường cong ROC phản ánh đánh đổi TPR–FPR khi quét mọi giá trị ngưỡng $\beta$. AUC cho biết mức thành công trung bình trên mọi mẫu đích và đo sức mạnh tổng thể của tấn công. Lấy cảm hứng từ (Carlini et al., 2022), nhóm cũng xét TPR ở FPR rất thấp, cụ thể là TPR tại FPR 0% — chỉ số ít được dùng trong các tài liệu trước. Khi tấn công một mô hình đích, mọi mẫu trong dữ liệu quần thể đều được dùng làm truy vấn đầu vào, do đó với mỗi mô hình đích một nửa truy vấn là thành viên và nửa còn lại là không-thành-viên.

### B. Chi tiết về RMIA và đánh giá (Details of RMIA and its Evaluation)

#### B.1. Mã giả của RMIA (Pseudo-code of RMIA)

MIA cần huấn luyện các mô hình tham chiếu (hay shadow model) để phân biệt thành viên với không-thành-viên của một mô hình đích. Nhóm huấn luyện $k$ mô hình tham chiếu trên các mẫu rút ngẫu nhiên từ dữ liệu quần thể. Có hai phiên bản RMIA:

- **RMIA (Online)** (tương tự Carlini et al., 2022) huấn luyện mô hình tham chiếu riêng cho từng dữ liệu đích (truy vấn kiểm tra MIA): khi nhận truy vấn $x$, huấn luyện $k$ mô hình IN có chứa $x$ trong tập huấn luyện. Cách này tốn kém và không thực tế.
- **RMIA (Offline)** chỉ dùng $k$ mô hình tham chiếu huấn luyện sẵn trên các tập dữ liệu lấy mẫu ngẫu nhiên, không huấn luyện gì trên truy vấn kiểm tra (chỉ dùng mô hình OUT).

Hai phiên bản khác nhau ở cách tính số hạng chuẩn hóa $\Pr(x)$ trong phương trình (3): bản online lấy trung bình xác suất dự đoán trên cả mô hình IN lẫn OUT một cách không thiên lệch (phương trình 9), còn bản offline chỉ tính với mô hình OUT dựa trên phương trình (10) (xấp xỉ $\Pr(x)$ như thể có mô hình IN; xem Phụ lục B.2.2).

**Thuật toán 1: Tính điểm MIA với RMIA.** Đầu vào: $k$ mô hình tham chiếu $\Theta$, mô hình đích $\theta$, mẫu đích $x$, tham số $\gamma$, hệ số tỉ lệ $a$ (Phụ lục B.2.2), và cờ `online`. Giả định các mô hình tham chiếu $\Theta$ được huấn luyện sẵn trên các mẫu ngẫu nhiên từ tập quần thể; mỗi mẫu quần thể nằm trong tập huấn luyện của một nửa số mô hình tham chiếu.

1. Chọn ngẫu nhiên tập con $Z$ từ tập dữ liệu quần thể.
2. $C \leftarrow 0$.
3. Nếu `online`:
   - $\Theta_{in}\leftarrow\emptyset$; lặp $k$ lần: lấy mẫu ngẫu nhiên tập $D_i$ từ $\pi$, huấn luyện $\theta_x \leftarrow T(D_i\cup x)$, thêm $\theta_x$ vào $\Theta_{in}$.
   - $\Pr(x)\leftarrow\frac{1}{2}\left(\frac1k\sum_{\theta'\in\Theta}\Pr(x\mid\theta')+\frac1k\sum_{\theta'\in\Theta_{in}}\Pr(x\mid\theta')\right)$ (phương trình 9).
4. Ngược lại: $\Pr(x)_{OUT}\leftarrow\frac1k\sum_{\theta'\in\Theta}\Pr(x\mid\theta')$; $\Pr(x)\leftarrow\frac12\big((1+a)\Pr(x)_{OUT}+(1-a)\big)$.
5. $\mathrm{Ratio}_x\leftarrow\Pr(x\mid\theta)/\Pr(x)$.
6. Với mỗi mẫu $z\in Z$: $\Pr(z)\leftarrow\frac1k\sum_{\theta'\in\Theta}\Pr(z\mid\theta')$; $\mathrm{Ratio}_z\leftarrow\Pr(z\mid\theta)/\Pr(z)$; nếu $\mathrm{Ratio}_x/\mathrm{Ratio}_z>\gamma$ thì $C\leftarrow C+1$.
7. $\mathrm{Score}_{MIA}(x;\theta)\leftarrow C/|Z|$ (phương trình 5).

Đầu ra của thuật toán là $\mathrm{Score}_{MIA}(x;\theta)$, sau đó dùng theo phương trình (1) để suy luận tư cách thành viên của $x$. Phép chia tỉ số của $x$ cho tỉ số của $z$ xác định $\mathrm{LR}_\theta(x,z)$ để đánh giá xem $z$ có bị $x$ $\gamma$-thống trị hay không; tỉ lệ các mẫu tham chiếu bị $\gamma$-thống trị tạo thành điểm MIA của $x$.

#### B.2. Tính tỉ số hợp lý trong phương trình (3) (Likelihood Ratio Computation in equation 3)

##### B.2.1. Tính $\Pr(x\mid\theta)$

Gọi $\theta$ là mô hình phân loại (mạng nơ-ron) ánh xạ mỗi đầu vào $x$ sang phân phối xác suất trên $d$ lớp. Giả sử $x$ thuộc lớp $y$; $c(x)=\langle c_1,\dots,c_d\rangle$ là vector đầu ra (logit) trước khi chuẩn hóa cuối. Xác suất dự đoán chuẩn hóa của lớp $y$ là $f_\theta(x)_y$. Với hàm Softmax:

$$f_\theta(x)_y=\frac{e^{c_y/T}}{\sum_{i=1}^{d}e^{c_i/T}},$$

trong đó $T$ là hằng số nhiệt độ (temperature). Ngoài Softmax, còn nhiều lựa chọn thay thế trong tài liệu (khai triển Taylor của hàm mũ và các heuristic tinh chỉnh quan hệ giữa các lớp; De Brebisson & Vincent, 2016; Liu et al., 2016; Liang et al., 2017; Banerjee et al., 2021). Phương pháp gần đây của Banerjee et al. (2021) kết hợp khai triển Taylor với kỹ thuật soft-margin:

$$f_\theta(x)_y=\frac{\mathrm{apx}(c_y-m)}{\mathrm{apx}(c_y-m)+\sum_{i\ne y}\mathrm{apx}(c_i)}, \tag{7}$$

trong đó $\mathrm{apx}(a)=\sum_{i=0}^{n}\frac{a^i}{i!}$ là xấp xỉ Taylor bậc $n$ của $e^a$, và $m$ là siêu tham số điều khiển độ tách giữa xác suất của các lớp.

Bảng 6 so sánh kết quả RMIA với bốn hàm tin cậy: Softmax, Taylor-Softmax, Soft-Margin Softmax (SM-Softmax), và tổ hợp của hai hàm sau (SM-Taylor-Softmax, phương trình 7). Soft-margin $m=0{,}6$, bậc $n=4$, nhiệt độ $T=2$ cho mô hình CIFAR-10. Các hàm cho kết quả gần tương đương, SM-Taylor-Softmax nhỉnh hơn một chút về AUC và TPR ở FPR thấp, nên được dùng cho CIFAR và CINIC-10. Hình 5 cho thấy AUC của RMIA khá bền vững với $n$, $m$ và $T$ (kết quả nhất quán khi $n\ge3$). Với ImageNet và Purchase-100 dùng Softmax (nhiệt độ 1). Nhóm không áp dụng các hàm tin cậy cho các tấn công khác vì chúng không dùng độ tin cậy làm tín hiệu (LiRA dùng rescaled logit; Attack-P và Attack-R dùng tín hiệu loss) và tái hiện theo thuật toán và mã gốc.

**Bảng 6: Hiệu năng RMIA với các hàm tin cậy khác nhau** (254 mô hình tham chiếu huấn luyện trên CIFAR-10).

| Hàm tin cậy | AUC | TPR@FPR 0,01% | TPR@FPR 0,0% |
|---|---|---|---|
| Softmax | 68,96 ± 0,31 | 2,25 ± 0,44 | 1,69 ± 0,37 |
| SM-Softmax | 69,02 ± 0,34 | 2,27 ± 0,43 | 1,67 ± 0,44 |
| Taylor-Softmax | 68,57 ± 0,29 | 2,06 ± 0,49 | 1,43 ± 0,49 |
| SM-Taylor-Softmax | 69,15 ± 0,35 | 2,26 ± 0,43 | 1,80 ± 0,55 |

*[Hình 5: AUC của RMIA với các giá trị khác nhau của $n$ (bậc Taylor), $m$ (soft-margin) và $T$ (nhiệt độ) trong hàm SM-Taylor-Softmax. Khi thay đổi một tham số, hai tham số còn lại giữ ở giá trị tối ưu. Dùng 254 mô hình tham chiếu huấn luyện trên CIFAR-10, trung bình trên 10 mô hình đích. Các điểm đỏ là giá trị mặc định.]*

##### B.2.2. Tính $\Pr(x)$

$\Pr(x)$ là hằng số chuẩn hóa của quy tắc Bayes khi tính LR (phương trình 4). Nó được tính bằng trung bình thực nghiệm của $\Pr(x\mid\theta')$ trên các mô hình tham chiếu $\theta'$ huấn luyện trên các tập $D$ ngẫu nhiên từ $\pi$. Các mô hình tham chiếu phải được lấy mẫu không thiên lệch đối với việc $x$ có thuộc dữ liệu huấn luyện hay không, vì tổng ở phương trình (4) chạy trên mọi $\theta'$, vốn có thể chia thành tập mô hình huấn luyện trên $x$ (IN) và tập không (OUT). Ký hiệu $\bar x$ là tập huấn luyện không chứa $x$; $\theta'_x$ là mô hình IN huấn luyện trên $D_x$ ($x\in D_x$) và $\theta'_{\bar x}$ là mô hình OUT huấn luyện trên $D_{\bar x}$ ($x\notin D_{\bar x}$). Từ phương trình (4):

$$\Pr(x)=\sum_{\theta'_x,D_x}\Pr(x\mid\theta'_x)\Pr(\theta'_x\mid D_x)\Pr(D_x)+\sum_{\theta'_{\bar x},D_{\bar x}}\Pr(x\mid\theta'_{\bar x})\Pr(\theta'\mid D_{\bar x})\Pr(D_{\bar x}) \tag{8}$$

Hai tổng này tính được bằng phương pháp lấy mẫu: $D_{\bar x}$ lấy từ phân phối dữ liệu $\pi$; $D_x$ lấy từ $\pi$ rồi thêm $x$; các mô hình được huấn luyện trên các tập đó. Ở thiết lập online, $\Pr(x)$ được ước lượng bằng trung bình $\Pr(x\mid\theta')$ trên 50% mô hình IN và 50% mô hình OUT (tổng cộng $2k$ mô hình):

$$\Pr(x)\approx\frac12\Big(\underbrace{\frac1k\sum_{\theta'_x}\Pr(x\mid\theta'_x)}_{\Pr(x)_{IN}}+\underbrace{\frac1k\sum_{\theta'_{\bar x}}\Pr(x\mid\theta'_{\bar x})}_{\Pr(x)_{OUT}}\Big) \tag{9}$$

Ở thiết lập offline không có mô hình IN, nên chỉ dùng $\Pr(x\mid\theta'_{\bar x})$. Nhóm đưa ra một heuristic đơn giản để có ước lượng ít thiên lệch hơn của $\Pr(x)$ mà không cần mô hình IN, thông qua tiền tính toán offline nhằm xấp xỉ sự dịch chuyển xác suất giữa mô hình IN và OUT (độ nhạy của mô hình; xem Hình 1 trong Zhang et al., 2021). Dùng các mô hình tham chiếu sẵn có, nhóm tính tốc độ thay đổi của $\Pr(x)$ giữa các mô hình có $x$ và không có $x$, xấp xỉ khoảng cách bằng một hàm tuyến tính: $\Pr(x)_{IN}=a\cdot\Pr(x)_{OUT}+b$. Vì cả hai giá trị nằm trong $[0,1]$ nên $a+b=1$, do đó $\Pr(x)_{IN}=a(\Pr(x)_{OUT}-1)+1$ và:

$$\Pr(x)\approx\frac12\big((1+a)\Pr(x)_{OUT}+(1-a)\big) \tag{10}$$

Hình 6 cho thấy AUC của RMIA offline trên CIFAR-10 khi $a$ biến thiên từ 0 đến 1. Khi $a\to1$, AUC giảm (tối đa 5,5%) vì phụ thuộc quá nhiều vào $\Pr(x)_{OUT}$ để xấp xỉ $\Pr(x)_{IN}$. Giá trị $a$ thấp hơn (nhất là dưới 0,5) cho kết quả bền vững hơn; ngay cả với $a=0$ (tức $\Pr(x)=(\Pr(x)_{OUT}+1)/2$) kết quả cải thiện đáng kể do giảm ảnh hưởng của $\Pr(x)_{OUT}$ rất thấp ở các mẫu không điển hình/khó.

Để chọn $a$ tốt nhất cho mô hình của từng bộ dữ liệu, nhóm dùng quy trình chỉ thực hiện một lần, không phụ thuộc truy vấn kiểm tra và không cần huấn luyện thêm mô hình: chọn hai mô hình sẵn có, dùng một mô hình làm mô hình đích tạm thời để bị tấn công từ mô hình kia với các giá trị $a$ khác nhau, rồi chọn $a$ cho AUC cao nhất. Nếu chỉ có một mô hình tham chiếu, mô phỏng tấn công vào mô hình tham chiếu và dùng mô hình đích gốc làm mô hình tham chiếu cho tấn công mô phỏng. Giá trị $a$ tối ưu khá nhất quán qua các lựa chọn ngẫu nhiên mô hình tham chiếu. Các giá trị thực nghiệm: $a=0{,}3$ cho CIFAR-10 và CINIC-10; $a=0{,}6$ cho CIFAR-100; $a=1$ cho ImageNet; $a=0{,}2$ cho Purchase-100.

*[Hình 6: AUC của RMIA offline với các giá trị $a$ khác nhau trong hàm xấp xỉ tuyến tính. Dùng 127 mô hình tham chiếu (OUT) huấn luyện trên CIFAR-10; hai đường tham chiếu: $P_{IN}=1$ và $P_{IN}=P_{OUT}$.]*

#### B.3. Tính trực tiếp tỉ số hợp lý trong phương trình (2) (Direct Computation of Likelihood Ratio in equation 2)

Mục 3 giới thiệu hai cách tính LR cơ bản trong phương trình (2): (I) phương pháp Bayes (phương trình 3) và (II) phương pháp trực tiếp (phương trình 6). Kết quả chính của bài báo dựa trên phương pháp Bayes; nhóm kỳ vọng hai cách hội tụ khi đủ mô hình tham chiếu nhưng có thể khác nhau khi số mô hình ít, và thử kiểm chứng điều này.

Phương pháp trực tiếp dùng mô hình hóa Gaussian trên logit: ước lượng trung bình và phương sai logit của $x$ và $z$ trong hai phân phối của các mô hình tham chiếu — một gồm các mô hình huấn luyện có $x$ nhưng không có $z$ ($\theta'_{x,\bar z}$), một gồm các mô hình có $z$ nhưng không có $x$ ($\theta'_{\bar x,z}$). Gọi $\mu_{x,\bar z}(x)$ và $\sigma^2_{x,\bar z}(x)$ là trung bình và phương sai của $f_{\theta'}(x)$ trong phân phối mô hình $\theta'_{x,\bar z}$ (ký hiệu tương tự cho $\theta'_{\bar x,z}$). LR trong phương trình (6) xấp xỉ:

$$\mathrm{LR}_\theta(x,z)\approx\frac{\Pr\big(f_\theta(x),f_\theta(z)\mid x\big)}{\Pr\big(f_\theta(x),f_\theta(z)\mid z\big)}\approx\frac{\Pr\big(f_\theta(x)\mid\mathcal N(\mu_{x,\bar z}(x),\sigma^2_{x,\bar z}(x))\big)}{\Pr\big(f_\theta(x)\mid\mathcal N(\mu_{\bar x,z}(x),\sigma^2_{\bar x,z}(x))\big)}\times\frac{\Pr\big(f_\theta(z)\mid\mathcal N(\mu_{x,\bar z}(z),\sigma^2_{x,\bar z}(z))\big)}{\Pr\big(f_\theta(z)\mid\mathcal N(\mu_{\bar x,z}(z),\sigma^2_{\bar x,z}(z))\big)} \tag{11}$$

trong đó $f_\theta(x)$ là đầu ra (logit) của mô hình đích $\theta$ trên mẫu $x$. Phương pháp trực tiếp tưởng như đơn giản và chính xác hơn nhưng chi phí tính toán cao hơn nhiều vì phải huấn luyện các mô hình tham chiếu online $\theta'_{x,\bar z}$.

Nhóm so sánh RMIA khi LR tính theo phương trình (3) (RMIA-Bayes) với LR theo phương trình (11) (RMIA-direct). Hình 19 (64 mô hình tham chiếu): RMIA-direct nhỉnh hơn một chút về AUC và TPR ở FPR thấp, nhưng RMIA-Bayes bám sát kể cả ở FPR thấp. Hình 20 (4 mô hình tham chiếu): RMIA-Bayes tốt hơn trên mọi bộ dữ liệu (ví dụ AUC cao hơn 6,6% trên CIFAR-10); RMIA-direct khó ước lượng chính xác tham số các mô hình Gaussian chỉ với bốn mô hình. Do chi phí xử lý lớn của RMIA-direct (huấn luyện mô hình online theo từng cặp mẫu), RMIA-Bayes là lựa chọn hợp lý hơn nhờ chạy được với ít mô hình hơn trong bối cảnh offline.

*[Hình 19: ROC của RMIA khi dùng hai cách xấp xỉ tỉ số hợp lý — RMIA-direct (phương trình 11) và RMIA-Bayes (phương trình 3), hiển thị ở thang log và thang thường, dùng 64 mô hình tham chiếu. Ví dụ CIFAR-10: RMIA-Bayes AUC = 0,684, RMIA-direct AUC = 0,689.]*

*[Hình 20: Tương tự Hình 19 nhưng với 4 mô hình tham chiếu. Ví dụ CIFAR-10: RMIA-Bayes AUC = 0,672, RMIA-direct AUC = 0,63. Các biểu đồ chi tiết cho CIFAR-100, CINIC-10 và Purchase-100 được lược bỏ ở bản dịch này.]*

#### B.4. Phân tích các tham số γ và β và mối quan hệ giữa chúng (Analyzing γ and β Parameters and their Relation)

Nhóm khảo sát tác động của việc chọn $\gamma$ lên hiệu quả tấn công. Hình 7 minh họa độ nhạy của tấn công (đo bằng AUC và TPR@FPR) với $\gamma$. Không xét $\gamma<1$ vì khi đó mẫu đích $x$ được phép có khả năng là thành viên thấp hơn các mẫu tham chiếu mà vẫn qua phép kiểm định LR theo cặp, khiến nhiều không-thành-viên bị suy luận sai là thành viên. Hiệu năng ổn định theo $\gamma$: cả AUC và FPR@TPR tương đối ổn định khi $\gamma$ tăng (trừ khi $\gamma$ rất cao). Khi $\gamma$ tăng cần giảm $\beta$ trong phương trình (1) để cân bằng giữa công suất và sai số; bằng cách điều chỉnh $\beta$ phù hợp, đạt kết quả gần như nhau với các $\gamma$ khác nhau. Trong trường hợp cực đoan $\gamma$ rất cao, việc phát hiện các mẫu tham chiếu bị $\gamma$-thống trị trong tập $z$ hữu hạn trở nên rất khó. Xu hướng này nhất quán trên mọi bộ dữ liệu và số mô hình tham chiếu. Xuyên suốt thí nghiệm, đặt $\gamma=2$ vì cho TPR@FPR nhỉnh hơn.

*[Hình 7: Độ nhạy hiệu năng của RMIA với tham số $\gamma$ ($\gamma$ từ $2^0$ đến $2^6$). Dùng 254 mô hình tham chiếu huấn luyện trên CIFAR-10. Biểu đồ trái: AUC theo $\gamma$; biểu đồ phải: TPR tại FPR 0,01% và 0% theo $\gamma$.]*

$\beta$ là ngưỡng chung của mọi tấn công, dùng để tạo đường cong ROC. Ở phương pháp của nhóm (khác với Carlini et al., 2022), giá trị $\beta$ có thể diễn giải được: khi $\gamma=1$, $\beta$ tương quan chặt với $1-\mathrm{FPR}$. Hình 8 minh họa tác động của việc chọn $\beta\in[0,1]$ lên TPR và FPR của RMIA với $\gamma=1$ (trái) và $\gamma=2$ (phải). Cả TPR và FPR tiến về 0 khi $\beta\to1$ (ở cả hai $\gamma$) vì một mẫu đích khó có thể thống trị phần lớn bản ghi tham chiếu. Tuy nhiên FPR giảm nhanh hơn TPR ở cả hai biểu đồ, đặc biệt với $\gamma=2$, cho phép luôn có lợi ích công suất lớn hơn sai số. Khi $\gamma=1$, FPR giảm theo kiểu được hiệu chuẩn, tỉ lệ với $1-\beta$.

*[Hình 8: TPR và FPR đạt được bởi RMIA với các giá trị $\beta$ khác nhau. Biểu đồ trái ứng với $\gamma=1$, phải ứng với $\gamma=2$. Số mô hình tham chiếu huấn luyện trên CIFAR-10 là 254.]*

#### B.5. Tăng cường RMIA bằng các truy vấn tăng cường (Boosting RMIA with Augmented Queries)

Hiệu quả tấn công có thể cải thiện thêm bằng cách tăng cường truy vấn đầu vào $x$ với nhiều mẫu dữ liệu tương tự $x$ (Carlini et al., 2022; Choquette-Choo et al., 2021), chẳng hạn các phép biến đổi đơn giản như dịch chuyển hoặc xoay với dữ liệu ảnh. Để tổng hợp kết quả trong thiết lập đa truy vấn, nhóm dùng biểu quyết đa số trên phép kiểm định giả thuyết ở phương trình (2): $x$ được coi là thống trị $z$ nếu hơn một nửa số biến đổi của $x$ thống trị $z$. (Carlini et al., 2022) đã thảo luận cách mở rộng LiRA cho nhiều truy vấn; ở đây so sánh hiệu năng tấn công online và offline khi số truy vấn tăng cường từ 1 đến 50. Attack-P và Attack-R không hỗ trợ tăng cường truy vấn nên không có kết quả đa truy vấn. Dùng 254 mô hình tham chiếu (với tấn công offline chỉ dùng 127 mô hình OUT).

**Bảng 7: Hiệu năng các tấn công với số truy vấn tăng cường khác nhau cho LiRA và RMIA** (CIFAR-10; trung bình trên 10 mô hình đích; online dùng 254 mô hình IN/OUT, offline dùng 127 mô hình OUT; Attack-P không dùng mô hình tham chiếu nên xem như offline).

| # truy vấn | Tấn công | Online AUC | Online TPR@0,01% | Online TPR@0,0% | Offline AUC | Offline TPR@0,01% | Offline TPR@0,0% |
|---|---|---|---|---|---|---|---|
| 1 | Attack-R | - | - | - | 64,41 ± 0,41 | 1,52 ± 0,33 | 0,80 ± 0,43 |
| 1 | Attack-P | - | - | - | 58,19 ± 0,33 | 0,01 ± 0,01 | 0,00 ± 0,01 |
| 1 | LiRA | 68,92 ± 0,42 | 1,78 ± 0,70 | 0,92 ± 0,48 | 56,12 ± 0,41 | 0,46 ± 0,18 | 0,28 ± 0,17 |
| 1 | RMIA | 69,15 ± 0,35 | 2,26 ± 0,43 | 1,80 ± 0,55 | 68,74 ± 0,34 | 2,42 ± 0,57 | 1,73 ± 0,61 |
| 2 | LiRA | 71,28 ± 0,46 | 2,83 ± 0,49 | 1,73 ± 0,70 | 55,77 ± 0,46 | 1,16 ± 0,41 | 0,59 ± 0,29 |
| 2 | RMIA | 71,46 ± 0,43 | 3,69 ± 0,36 | 2,55 ± 0,71 | 71,06 ± 0,39 | 3,64 ± 0,61 | 2,46 ± 0,98 |
| 18 | LiRA | 72,04 ± 0,47 | 3,39 ± 0,86 | 2,01 ± 0,78 | 55,18 ± 0,37 | 1,37 ± 0,32 | 0,72 ± 0,31 |
| 18 | RMIA | 72,25 ± 0,46 | 4,31 ± 0,47 | 3,15 ± 0,61 | 71,71 ± 0,43 | 4,18 ± 0,61 | 3,14 ± 0,87 |
| 50 | LiRA | 72,26 ± 0,47 | 3,54 ± 0,50 | 2,19 ± 0,83 | 55,00 ± 0,36 | 1,52 ± 0,34 | 0,75 ± 0,36 |
| 50 | RMIA | 72,51 ± 0,46 | 4,47 ± 0,44 | 3,25 ± 0,36 | 71,95 ± 0,44 | 4,39 ± 0,54 | 3,22 ± 0,81 |

Với các tấn công offline (cột Offline của Bảng 7), khi không có truy vấn tăng cường RMIA có lợi thế rõ rệt so với Attack-R (AUC cao hơn 6,7% và TPR tại FPR bằng 0 cao hơn 116%), và khoảng cách mở rộng khi dùng nhiều truy vấn hơn. Khi tăng số truy vấn, RMIA cải thiện (ví dụ TPR tại FPR bằng 0 tăng 4× và AUC tăng khoảng 4,6% khi truy vấn tăng từ 1 lên 50), trong khi LiRA không hưởng lợi để cải thiện AUC. Cùng kỹ thuật tạo tăng cường như (Carlini et al., 2022). Ở thiết lập online (254 mô hình IN và OUT), RMIA luôn có AUC nhỉnh hơn LiRA và TPR tại FPR bằng 0 tốt hơn ít nhất 48%; cả hai đều cải thiện khi tăng số truy vấn (khoảng 2× TPR@FPR từ 1 lên 50 truy vấn). Trừ khi nêu khác đi, mọi kết quả thực nghiệm trong bài dùng 18 truy vấn tăng cường cho cả LiRA và RMIA.

### C. Kết quả thực nghiệm bổ sung (Supplementary Empirical Results)

#### C.1. Độ bền vững trước truy vấn nhiễu và không-thành-viên OOD (Robustness of Attacks against Noise and OOD Non-member Queries)

Để đánh giá các tấn công trước không-thành-viên OOD, nhóm xét hai chiến lược tạo truy vấn kiểm tra: dùng mẫu từ bộ dữ liệu khác bộ huấn luyện, và dùng nhiễu thuần túy. Hình 9 minh họa ROC của ba tấn công offline trên mô hình CIFAR-10 (mọi truy vấn thành viên đều trong phân phối), với 127 mô hình tham chiếu (OUT), hiển thị ở thang thường. Mẫu OOD lấy từ CINIC-10 có cùng nhãn với CIFAR-10. Với mẫu OOD (biểu đồ trái), RMIA hơn các tấn công khác một khoảng đáng kể (AUC cao hơn ít nhất 21%), LiRA chỉ nhỉnh hơn đoán ngẫu nhiên. Với nhiễu làm truy vấn không-thành-viên (biểu đồ phải), RMIA hơn hai tấn công còn lại ít nhất 61% về AUC, trong khi LiRA thấp hơn đáng kể so với đoán ngẫu nhiên (FPR rất lớn).

*[Hình 9: ROC của ba tấn công offline dùng mô hình huấn luyện trên CIFAR-10 khi truy vấn kiểm tra không-thành-viên đến từ phân phối khác với phân phối quần thể $\pi$. Kết quả trên một mô hình đích ngẫu nhiên. Biểu đồ trái dùng mẫu OOD từ CINIC-10 (AUC: RMIA 0,768; Attack-R 0,632; LiRA 0,509); biểu đồ phải dùng nhiễu ngẫu nhiên (AUC: RMIA 0,967; Attack-R 0,597; LiRA 0,415). Dùng 127 mô hình tham chiếu (OUT). Với RMIA dùng $a=0$ và $\gamma=2$.]*

Hình 10 minh họa hiệu năng khi dùng các tỉ lệ OOD khác nhau trong các truy vấn không-thành-viên (tỉ lệ 0: toàn bộ trong phân phối; tỉ lệ 1: toàn bộ OOD). Khi tăng tỉ lệ OOD, AUC của RMIA và Attack-P tăng, còn hiệu năng hai tấn công kia giảm — do hai tấn công đầu có thể lọc mẫu OOD bằng cách so sánh điểm MIA của truy vấn với dữ liệu quần thể trong phân phối. Xét TPR tại FPR bằng 0, tấn công dựa trên quần thể (Attack-P) cho TPR gần bằng 0, còn RMIA vượt mọi tấn công khác; xu hướng giảm TPR tại FPR bằng 0 ở mọi tấn công cho thấy có những trường hợp OOD khó phát hiện khi so điểm MIA với dữ liệu quần thể.

*[Hình 10: Hiệu năng của các tấn công khi kiểm tra với các tỉ lệ khác nhau của truy vấn không-thành-viên OOD (tỉ lệ = 0: không có mẫu OOD; tỉ lệ = 1: toàn bộ OOD). Biểu đồ trái: AUC; biểu đồ phải: TPR tại FPR bằng 0. Mọi tấn công đều offline với 127 mô hình tham chiếu; mô hình huấn luyện trên CIFAR-10, truy vấn OOD từ CINIC-10.]*

Hình 11 so sánh FPR của RMIA khi dùng hai tập không-thành-viên: OOD và trong phân phối (ID). Khi $\beta$ tăng từ 0, FPR của mẫu OOD giảm nhanh hơn nhiều so với FPR và TPR của mẫu ID, cho thấy RMIA nhận ra phần lớn các truy vấn OOD là không-thành-viên. Biểu đồ phải cho thấy sự chênh lệch rõ: FPR của mẫu ID có thể vượt 0,8 trong khi FPR của mẫu OOD (cùng $\beta$) gần bằng 0 — RMIA phát hiện không-thành-viên OOD dễ hơn không-thành-viên ID.

*[Hình 11: So sánh FPR của RMIA (Offline) khi dùng hai tập truy vấn không-thành-viên riêng biệt — trong phân phối (ID) và ngoài phân phối (OOD). 127 mô hình tham chiếu, $\gamma=2$. Mô hình huấn luyện trên CIFAR-10, OOD từ CINIC-10. Biểu đồ trái: TPR và FPR theo $\beta$ cho từng tập; biểu đồ phải: so sánh FPR của OOD với ID.]*

#### C.2. Dịch chuyển phân phối dữ liệu (Data Distribution Shift)

Bảng 8 so sánh kết quả khi mô hình đích huấn luyện trên bộ dữ liệu khác với mô hình tham chiếu: mô hình đích huấn luyện trên CIFAR-10, mô hình tham chiếu trên CINIC-10 (mọi truy vấn kiểm tra từ CIFAR-10); dùng các ảnh có nhãn chung giữa hai bộ dữ liệu. Chỉ xét các tấn công offline. Dịch chuyển phân phối dữ liệu huấn luyện giữa mô hình đích và mô hình tham chiếu ảnh hưởng đến hiệu năng của mọi tấn công. So với hai tấn công kia, RMIA luôn có AUC cao hơn (tới 25% so với LiRA) và TPR tốt hơn ở FPR thấp.

**Bảng 8: Hiệu năng các tấn công offline khi huấn luyện mô hình đích và mô hình tham chiếu trên các bộ dữ liệu khác nhau** (đích: CIFAR-10; tham chiếu: CINIC-10; 1, 2, 4 mô hình OUT; trung bình trên 10 mô hình đích ngẫu nhiên).

| # mô hình tham chiếu | Tấn công | AUC | TPR@FPR 0,01% | TPR@FPR 0,0% |
|---|---|---|---|---|
| 1 | Attack-R (Ye et al., 2022) | 61,41 ± 0,23 | 0,02 ± 0,01 | 0,01 ± 0,01 |
| 1 | LiRA (Offline) (Carlini et al., 2022) | 54,48 ± 0,21 | 0,06 ± 0,03 | 0,01 ± 0,01 |
| 1 | RMIA (Offline) | 64,75 ± 0,27 | 0,06 ± 0,02 | 0,02 ± 0,01 |
| 2 | Attack-R | 61,45 ± 0,3 | 0,03 ± 0,01 | 0,01 ± 0,01 |
| 2 | LiRA (Offline) | 52,58 ± 0,18 | 0,01 ± 0,01 | 0 ± 0 |
| 2 | RMIA (Offline) | 66,05 ± 0,29 | 0,13 ± 0,07 | 0,04 ± 0,04 |
| 4 | Attack-R | 61,54 ± 0,3 | 0,04 ± 0,02 | 0,02 ± 0,01 |
| 4 | LiRA (Offline) | 55,15 ± 0,25 | 0,03 ± 0,02 | 0,01 ± 0,01 |
| 4 | RMIA (Offline) | 66,78 ± 0,32 | 0,22 ± 0,06 | 0,09 ± 0,09 |

#### C.3. Các biến thể kiến trúc mạng nơ-ron (Variations in Neural Network Architectures)

Hình 12 minh họa hiệu năng các tấn công khi mô hình huấn luyện với các kiến trúc khác nhau — CNN và Wide ResNet (WRN) với nhiều kích cỡ — trên CIFAR-10, khi mô hình đích và mô hình tham chiếu cùng kiến trúc. RMIA luôn vượt các tấn công khác trên mọi kiến trúc (ví dụ AUC cao hơn LiRA 7,5%–16,8%). Kiến trúc mạng nhiều tham số hơn có thể dẫn đến rò rỉ nhiều hơn (Carlini et al., 2019). Hình 13 trình bày hiệu năng khi mô hình tham chiếu dùng kiến trúc khác nhau còn mô hình đích cố định là WRN28-2. Hiệu năng tối ưu đạt được khi kiến trúc hai bên tương tự nhau; RMIA vẫn vượt các tấn công khác và khoảng cách mở rộng khi có dịch chuyển kiến trúc.

*[Hình 12: ROC của các tấn công với các kiến trúc mạng khác nhau (CNN-16, CNN-32, CNN-64, WRN28-1, WRN28-2, WRN28-10) khi huấn luyện mô hình trên CIFAR-10. Mô hình đích và mô hình tham chiếu cùng kiến trúc; dùng 2 mô hình tham chiếu (1 IN, 1 OUT). AUC của RMIA lần lượt: 0,659; 0,693; 0,713; 0,675; 0,692; 0,711; của LiRA lần lượt: 0,559; 0,607; 0,642; 0,608; 0,633; 0,661.]*

*[Hình 13: ROC của các tấn công khi mô hình tham chiếu dùng kiến trúc khác nhau (CNN-16, CNN-32, CNN-64, WRN28-1, WRN28-2, WRN28-10) trên CIFAR-10; mô hình đích luôn là WRN28-2; dùng 2 mô hình tham chiếu (1 IN, 1 OUT). AUC của RMIA lần lượt: 0,633; 0,663; 0,676; 0,683; 0,692; 0,694; của LiRA lần lượt: 0,563; 0,593; 0,602; 0,612; 0,633; 0,636.]*

#### C.4. Hiệu năng trên mô hình huấn luyện với tập con lớn hơn của CINIC-10 (Performance of Attacks on Models Trained with a Larger Subset of CINIC-10)

Phân tích hiệu quả tấn công trên mô hình có tập huấn luyện lớn hơn là quan trọng, vì chúng có thể suy giảm hiệu năng (Bertran et al., 2023). Nhóm đánh giá trên mô hình huấn luyện với tập con ngẫu nhiên 90k mẫu của CINIC-10 (lớn hơn 25k ở các thí nghiệm trước): huấn luyện Wide ResNet 200 epoch trên một nửa bộ dữ liệu chọn ngẫu nhiên (batch size 64, learning rate 0,1, weight decay 0,0005). Độ chính xác kiểm tra của mô hình CINIC-10 là 79%. Với tấn công offline, dùng Softmax gốc với nhiệt độ 1 làm hàm tin cậy; giá trị tốt nhất của $a$ trong phương trình (10) là 1. Hình 14 cho thấy ROC của các tấn công trên CINIC-10 chỉ dùng 1 mô hình tham chiếu (OUT), kèm kết quả Quantile Regression (Bertran et al., 2023). RMIA vượt các tấn công khác về cả AUC lẫn TPR ở FPR thấp: AUC cao hơn LiRA khoảng 15% và TPR tại FPR 0,01% cao hơn Quantile Regression một bậc độ lớn. Dù AUC của Quantile Regression cao hơn LiRA đáng kể (với chỉ 1 mô hình tham chiếu), nó không đạt được dương thật nào tại FPR bằng 0.

*[Hình 14: ROC của các tấn công với mô hình huấn luyện trên tập con lớn hơn của CINIC-10 (90k mẫu). Kết quả trên một mô hình đích ngẫu nhiên; 1 mô hình tham chiếu (OUT); không dùng truy vấn tăng cường. AUC: RMIA 0,714; LiRA 0,621; Attack-R 0,679; Quantile-Reg. 0,671; Attack-P 0,656.]*

#### C.5. Hiệu năng của các tấn công khi dùng ít mô hình tham chiếu hơn (Performance of Attacks using Fewer Reference Models)

Bảng 9 trình bày kết quả trung bình kèm độ lệch chuẩn của các tấn công trên CIFAR-10, CIFAR-100 và CINIC-10 dưới ngân sách tính toán thấp (một vài mô hình tham chiếu). *[Bảng 9: Các giá trị trung bình đã được báo cáo đầy đủ ở Bảng 2 ở trên; bản dịch này không chép lại các độ lệch chuẩn của từng ô (ví dụ RMIA với 1 mô hình trên CIFAR-10: AUC 68,64 ± 0,43; TPR@0,01% 1,19 ± 0,28; TPR@0,0% 0,51 ± 0,33).]*

#### C.6. Hiệu năng của các tấn công trên mô hình huấn luyện bằng thuật toán ML khác (Performance of Attacks on Models Trained with other ML Algorithms)

Dù đa số mô hình ML hiện nay dựa trên mạng nơ-ron, việc tìm hiểu cách các tấn công tổng quát hóa với thuật toán ML khác cũng đáng quan tâm. Ngoài phạm vi bài báo để phân tích toàn diện, nhóm thực hiện một thí nghiệm đơn giản huấn luyện mô hình bằng thuật toán Gradient Boosting Decision Tree (GBDT) trên bộ dữ liệu phi ảnh Purchase-100. Siêu tham số GBDT: `n_estimators=250`, `lr=0,1`, `subsample=0,2`, độ chính xác kiểm tra khoảng 53%, với ba giá trị độ sâu tối đa. Dùng hai mô hình tham chiếu (1 IN, 1 OUT). Vì đầu ra GBDT là xác suất dự đoán (không phải logit), xác suất này được dùng làm tín hiệu đầu vào cho mọi tấn công. TPR của RMIA luôn vượt các tấn công khác, đặc biệt hơn một bậc độ lớn tại FPR bằng 0.

*[Hình 15: ROC của các tấn công trên mô hình huấn luyện bằng GBDT trên Purchase-100 với ba giá trị độ sâu tối đa (3, 5, 7). Độ chính xác kiểm tra khoảng 53%. Dùng 2 mô hình tham chiếu (1 IN, 1 OUT). Với LiRA, dùng xác suất đầu ra $p$ để tính tín hiệu rescaled-logit $\log\frac{p}{1-p}$; với các tấn công khác dùng trực tiếp xác suất đầu ra. AUC của RMIA lần lượt: 0,913; 0,973; 0,98; của LiRA: 0,863; 0,956; 0,968.]*

#### C.7. Hiệu năng của các tấn công trên các bộ dữ liệu khác nhau (Performance of Attacks Obtained on Different Datasets)

Nhóm so sánh ROC của các tấn công với mô hình tham chiếu huấn luyện trên bốn bộ dữ liệu — CIFAR-10, CIFAR-100, CINIC-10 và Purchase-100 — với số lượng mô hình tham chiếu khác nhau, ở cả thang log và thang thường.

- **1 mô hình tham chiếu (Hình 21):** chỉ có 1 mô hình OUT nên báo cáo kết quả offline. RMIA tốt hơn rõ rệt ba tấn công kia trên mọi bộ dữ liệu. Dù LiRA có TPR ở FPR thấp tương đối tương đương trên CIFAR-10 và Purchase-100, AUC thấp hơn nhiều (ví dụ thấp hơn 22% trên CIFAR-10). Ở hai bộ còn lại, RMIA cho TPR tại FPR bằng 0 tốt hơn khoảng 3 lần so với đối thủ gần nhất.
- **2 mô hình tham chiếu — 1 IN, 1 OUT (Hình 22):** RMIA lại tốt hơn nhiều (cả AUC lẫn TPR@FPR) trên mọi bộ dữ liệu; ví dụ AUC cao hơn 10% trên CIFAR-10 và TPR tại FPR bằng 0 tốt hơn ít nhất 3 lần trên CIFAR-100 và CINIC-10.
- **127 mô hình OUT, offline (Hình 23):** RMIA tốt hơn nhiều, ví dụ AUC cao hơn LiRA ít nhất 20%.
- **Toàn bộ 254 mô hình — 127 IN, 127 OUT (Hình 24):** nhờ huấn luyện hàng trăm mô hình IN và OUT, LiRA có thể tiến gần RMIA về AUC, nhưng TPR tại FPR bằng 0 của RMIA cao hơn đáng kể (ít nhất 50% trên CIFAR-10, CINIC-10 và Purchase-100).

Hình 25 trình bày AUC của các tấn công offline và online theo số mô hình tham chiếu (từ 1 đến 254). Số mô hình tham chiếu tăng thì AUC của mọi tấn công đều cải thiện; ở cả hai chế độ và mọi bộ dữ liệu RMIA luôn vượt các tấn công khác, đặc biệt khi chỉ dùng ít mô hình.

*[Hình 21–24: ROC của các tấn công trên các bộ dữ liệu khác nhau (thang log và thang thường) lần lượt với 1 mô hình tham chiếu, 2 mô hình tham chiếu (1 IN, 1 OUT), 127 mô hình OUT (chỉ tấn công offline) và 254 mô hình (127 IN, 127 OUT). Các biểu đồ không được tái tạo trong bản dịch.]*

*[Hình 25: AUC của các tấn công khi dùng số mô hình tham chiếu khác nhau, cho (a) CIFAR-10, (b) CIFAR-100, (c) CINIC-10, (d) Purchase-100. Biểu đồ trái: tấn công offline; biểu đồ phải: tấn công online (một nửa mô hình tham chiếu là OUT, một nửa là IN).]*

#### C.8. Hiệu năng của các tấn công trên mô hình Purchase-100 (Performance of Attacks on Purchase-100 Models)

Bảng 10 trình bày kết quả khi đánh giá mô hình huấn luyện trên Purchase-100. Ưu thế quan sát được nhất quán với các bộ dữ liệu khác: dùng nhiều mô hình tham chiếu hơn giúp cải thiện mọi tấn công, nhưng ở mọi kịch bản RMIA luôn vượt các tấn công khác về cả AUC lẫn TPR ở FPR thấp. Đáng chú ý, chỉ với 1 mô hình tham chiếu (OUT), RMIA có AUC cao hơn LiRA hơn 30%. RMIA offline với 2 hoặc 4 mô hình tham chiếu có thể vượt LiRA online (cùng số mô hình), và RMIA giữ ưu thế ngay cả khi có hàng trăm mô hình tham chiếu.

**Bảng 10: Hiệu năng các tấn công khi tấn công mô hình huấn luyện trên Purchase-100 với số mô hình tham chiếu khác nhau.** Mọi tấn công mặc định đều chi phí thấp và offline; kết quả online của LiRA và RMIA được cung cấp để cho thấy hiệu năng cao nhất. Không dùng truy vấn tăng cường. Trung bình trên 10 mô hình đích ngẫu nhiên.

| # Ref | Tấn công | AUC | TPR@FPR 0,01% | TPR@FPR 0,0% |
|---|---|---|---|---|
| 0 | Attack-P | 66,62 ± 0,27 | 0 ± 0,01 | 0 ± 0 |
| 0 | Attack-R | 74,24 ± 0,28 | 0 ± 0,01 | 0 ± 0 |
| 1 | LiRA | 59,3 ± 0,35 | 0,2 ± 0,09 | 0,1 ± 0,06 |
| 1 | RMIA | 77,8 ± 0,2 | 0,33 ± 0,2 | 0,14 ± 0,09 |
| 2 | Attack-R | 74,8 ± 0,26 | 0,05 ± 0,05 | 0,01 ± 0,02 |
| 2 | LiRA (Online) | 72,97 ± 0,46 | 0,37 ± 0,09 | 0,2 ± 0,07 |
| 2 | RMIA | 80,44 ± 0,24 | 0,89 ± 0,38 | 0,23 ± 0,19 |
| 4 | Attack-R | 75,68 ± 0,36 | 0,09 ± 0,07 | 0,01 ± 0,02 |
| 4 | LiRA (Online) | 77,57 ± 0,52 | 0,63 ± 0,1 | 0,27 ± 0,07 |
| 4 | RMIA | 81,94 ± 0,29 | 1,58 ± 0,67 | 0,41 ± 0,4 |
| 127 | Attack-R | 77,80 ± 0,41 | 1,02 ± 0,35 | 0,42 ± 0,36 |
| 127 | LiRA | 65,82 ± 0,58 | 0,31 ± 0,13 | 0,11 ± 0,07 |
| 127 | RMIA | 83,21 ± 0,33 | 2,35 ± 0,92 | 0,69 ± 0,67 |
| 254 | LiRA (Online) | 83,23 ± 0,37 | 1,99 ± 0,88 | 0,82 ± 0,59 |
| 254 | RMIA (Online) | 83,90 ± 0,36 | 2,56 ± 0,79 | 1,29 ± 0,91 |

#### C.9. Hiệu năng của RMIA với số lượng mẫu $z$ khác nhau (Performance of RMIA with Different Number of z Samples)

Bảng 11 trình bày hiệu năng RMIA với 1 mô hình tham chiếu (OUT) khi thay đổi số mẫu tham chiếu, cho xu hướng gần giống Bảng 4 (254 mô hình): AUC tăng khi dùng nhiều mẫu tham chiếu hơn, nhưng dùng 2500 mẫu quần thể (10% kích thước tập huấn luyện) cho kết quả tương đương tập quần thể lớn gấp 10 lần.

**Bảng 11: Hiệu năng RMIA (Offline) với số mẫu $z$ ngẫu nhiên khác nhau.** Chỉ dùng 1 mô hình tham chiếu (OUT) huấn luyện trên CIFAR-10; không dùng truy vấn tăng cường; trung bình trên 10 mô hình đích ngẫu nhiên.

| # mẫu quần thể $z$ | AUC | TPR@FPR 0,01% | TPR@FPR 0,0% |
|---|---|---|---|
| 25 | 58,92 ± 0,22 | 0 ± 0 | 0 ± 0 |
| 250 | 63,28 ± 0,2 | 0,29 ± 0,16 | 0,12 ± 0,09 |
| 1250 | 64,75 ± 0,27 | 0,31 ± 0,15 | 0,17 ± 0,1 |
| 2500 | 65,08 ± 0,22 | 0,31 ± 0,16 | 0,19 ± 0,14 |
| 6250 | 65,35 ± 0,24 | 0,34 ± 0,18 | 0,19 ± 0,13 |
| 12500 | 65,45 ± 0,25 | 0,32 ± 0,17 | 0,2 ± 0,16 |
| 25000 | 65,46 ± 0,16 | 0,61 ± 0,16 | 0,47 ± 0,1 |

#### C.10. So sánh điểm MIA giữa các tấn công (MIA Score Comparison between Attacks)

Để hiểu rõ hơn khác biệt hiệu năng, nhóm so sánh điểm MIA của mẫu thành viên và không-thành-viên giữa RMIA và các tấn công khác. Hình 16 so sánh điểm RMIA và LiRA trong hai kịch bản: 1 mô hình tham chiếu (trái) và 4 mô hình tham chiếu (phải). Với 1 mô hình, RMIA phân biệt rõ hơn nhiều giữa mẫu thành viên và không-thành-viên khi gán điểm MIA phân biệt trong khoảng [0, 1], tách nhiều thành viên về bên phải và không-thành-viên về bên trái; ngược lại điểm LiRA tập trung ở phía trên biểu đồ, thiếu sự tách biệt rõ giữa điểm thành viên và không-thành-viên. Khi dùng nhiều mô hình tham chiếu hơn, mức tương quan giữa điểm của hai tấn công cao hơn. Tương tự, Hình 17 so sánh RMIA và Attack-R: RMIA tách tốt hơn thành viên và không-thành-viên (điểm thành viên có xu hướng lớn hơn điểm không-thành-viên với nhiều mẫu), trong khi ở Attack-R không có sự tách biệt rõ như vậy. Hình 18 so sánh RMIA với Attack-P (RMIA chỉ dùng 1 mô hình OUT vì Attack-P không dùng mô hình tham chiếu): khác RMIA, Attack-P rõ ràng không phân tách tốt điểm thành viên và không-thành-viên.

Các tấn công suy luận mạnh tính $\mathrm{Score}_{MIA}(x;\theta)$ sao cho phản ánh chính xác khả năng phân biệt giữa các mô hình được huấn luyện trên điểm dữ liệu đích và các mô hình không. Hình 26 minh họa phân phối điểm MIA của các tấn công trên một tập mẫu kiểm tra: tính $\mathrm{Score}_{MIA}(x;\theta)$ trên 254 mô hình đích, mẫu là thành viên của một nửa và không-thành-viên của nửa còn lại; tấn công dùng 127 mô hình OUT. Điểm MIA của RMIA cho sự tách biệt rõ hơn giữa thành viên và không-thành-viên (điểm thành viên tập trung ở bên phải, khác với các tấn công khác).

*[Hình 16: So sánh điểm MIA giữa RMIA và LiRA (Carlini et al., 2022) trên mọi mẫu thành viên và không-thành-viên của một mô hình đích ngẫu nhiên. (a) 1 mô hình OUT; (b) 4 mô hình tham chiếu (2 IN, 2 OUT). Trục x: điểm RMIA; trục y: điểm LiRA.]*

*[Hình 17: So sánh điểm MIA giữa RMIA và Attack-R (Ye et al., 2022). (a) 1 mô hình tham chiếu (OUT); (b) 4 mô hình (OUT). Trục x: điểm RMIA; trục y: điểm Attack-R.]*

*[Hình 18: So sánh điểm MIA giữa RMIA và Attack-P (Ye et al., 2022) trên một mô hình đích ngẫu nhiên. Trục x: điểm RMIA; trục y: điểm Attack-P. RMIA chỉ dùng 1 mô hình OUT.]*

*[Hình 26: Phân phối điểm MIA của các tấn công khác nhau cho một mẫu kiểm tra. Điểm MIA của mỗi mẫu được tính trên 254 mô hình đích, một nửa là mô hình IN và một nửa là OUT. Tấn công dùng 127 mô hình tham chiếu (OUT) huấn luyện trên CIFAR-10.]*

### D. Các công trình liên quan (Related Work)

Mạng nơ-ron, đặc biệt khi huấn luyện trên dữ liệu nhạy cảm về quyền riêng tư, đã được chứng minh dễ rò rỉ thông tin về dữ liệu huấn luyện. Nhiều loại tấn công được thiết kế để đo mức rò rỉ và rủi ro quyền riêng tư: tấn công trích xuất dữ liệu tái tạo từng mẫu huấn luyện (Carlini et al., 2021); tấn công đảo ngược mô hình (model inversion) trích xuất thông tin tổng hợp về các lớp con (Fredrikson et al., 2015); tấn công suy luận thuộc tính (property inference) suy ra các thuộc tính không tầm thường của tập huấn luyện (Ganju et al., 2018). Bài báo này tập trung vào MIA, dự đoán một mẫu cụ thể có được dùng để huấn luyện mô hình hay không; do tính đơn giản, MIA thường được dùng làm công cụ kiểm toán để định lượng rò rỉ dữ liệu.

MIA lần đầu được dùng trong dữ liệu bộ gen để xác định sự hiện diện của bộ gen một cá nhân trong một hỗn hợp bộ gen (Homer et al., 2008). (Backes et al., 2016) hình thức hóa phân tích rủi ro nhận diện bộ gen cá nhân từ thống kê tổng hợp trên các thuộc tính độc lập, sau đó được mở rộng cho dữ liệu có thuộc tính phụ thuộc (Murakonda et al., 2021). Các thuật toán có quyền riêng tư vi phân (differential privacy – DP) được thiết kế để giới hạn tỉ lệ thành công của tấn công quyền riêng tư khi phân biệt hai tập dữ liệu lân cận (Nasr et al., 2021); một số nghiên cứu như (Thudi et al., 2022) đưa ra cận trên cho mức thành công trung bình của MIA trên các đích tổng quát; số khác đánh giá hiệu quả MIA trên mô hình ML huấn luyện bằng thuật toán DP (Rahman et al., 2018).

(Shokri et al., 2017) giới thiệu MIA cho thuật toán học máy, chứng minh hiệu quả trong thiết lập đối thủ chỉ có quyền truy vấn mô hình đích, dựa trên việc huấn luyện các mô hình tham chiếu (còn gọi là shadow model) với tập dữ liệu cùng phân phối với dữ liệu huấn luyện. Các công trình sau mở rộng ý tưởng shadow model sang các kịch bản khác: hộp trắng (Leino & Fredrikson, 2020; Nasr et al., 2019; Sablayrolles et al., 2019), hộp đen (Song & Mittal, 2021; Hisamoto et al., 2020; Chen et al., 2021), chỉ truy cập nhãn (Choquette-Choo et al., 2021; Li & Zhang, 2021), và các bộ dữ liệu đa dạng (Salem et al., 2019). Tuy nhiên các phương pháp này thường đòi hỏi huấn luyện rất nhiều mô hình khi nhận mỗi truy vấn, gây tốn kém xử lý, lưu trữ, thời gian phản hồi và lượng dữ liệu. MIA cũng được áp dụng trong học liên kết (federated learning) (Nasr et al., 2019; Melis et al., 2019; Truex et al., 2019) và mạng đa lối thoát (multi-exit networks) (Li et al., 2022).

Nhiều cơ chế phòng thủ trước MIA đã được đề xuất, dù nhiều chiến lược kém hiệu quả hơn báo cáo ban đầu (Song & Mittal, 2021). Do overfitting là yếu tố quan trọng gây rò rỉ thành viên, nhiều kỹ thuật chính quy hóa được dùng để phòng thủ, như chính quy hóa L2, dropout và label smoothing (Shokri et al., 2017; Salem et al., 2019; Liu et al., 2022a). Một số nghiên cứu giảm khoảng cách tổng quát hóa của mô hình đích (Li et al., 2021; Chen et al., 2022) hoặc tự chưng cất tập huấn luyện (Tang et al., 2022). (Abadi et al., 2016) đề xuất DP-SGD thêm quyền riêng tư vi phân (Dwork, 2006) vào SGD; sau đó các công trình tập trung giảm chi phí quyền riêng tư của DP-SGD qua cắt gradient thích ứng hoặc tốc độ học thích ứng (Yu et al., 2019; Xu et al., 2020). Ngoài ra còn có AdvReg (Nasr et al., 2018) và MemGuard (Jia et al., 2019), được thiết kế để che giấu sự khác biệt giữa posterior đầu ra của thành viên và không-thành-viên.

Nghiên cứu gần đây nhấn mạnh việc đánh giá tấn công bằng TPR ở FPR rất thấp (Carlini et al., 2022; Ye et al., 2022; Liu et al., 2022b; Long et al., 2020; Watson et al., 2022b). Ví dụ (Carlini et al., 2022) nhận thấy nhiều tấn công trước kém hiệu quả theo mô hình đánh giá này và tạo ra tấn công hiệu quả dựa trên kiểm định tỉ số hợp lý giữa phân phối của các mô hình dùng mẫu đích để huấn luyện (mô hình IN) và không dùng (mô hình OUT), nhưng cần huấn luyện nhiều mô hình tham chiếu để đạt hiệu năng cao. (Watson et al., 2022b) xây dựng MIA tích hợp độ khó của mẫu, dùng ngưỡng độ khó từng mẫu để hiệu chuẩn loss của mô hình đích. (Ye et al., 2022) đề xuất khuôn mẫu định nghĩa các trò chơi MIA và khung kiểm định giả thuyết toàn diện để thiết kế tấn công mạnh dùng mô hình tham chiếu nhằm tăng đáng kể TPR ở mỗi FPR. (Liu et al., 2022b) đề xuất tấn công dùng tín hiệu theo quỹ đạo sinh ra trong quá trình huấn luyện các mô hình chưng cất để tăng khả năng phân biệt thành viên. (Wen et al., 2023) cải thiện các tấn công dựa trên kiểm định tỉ số hợp lý bằng cách ước lượng một biến thể của mẫu đích, thu được qua nhiễu loạn tối thiểu mẫu gốc nhằm cực tiểu hóa loss khớp của các shadow model IN và OUT. (Leemann et al., 2023) giới thiệu khái niệm quyền riêng tư mới f-MIP, cho phép chặn đánh đổi công suất–sai số của tấn công bằng hàm $f$, đặc biệt áp dụng khi mô hình huấn luyện bằng cập nhật gradient; tác giả chứng minh dùng DP-SGD để đạt cận f-MIP và đề xuất tấn công chi phí thấp để kiểm toán rò rỉ quyền riêng tư với giả định đối thủ có quyền truy cập hộp trắng vào gradient của mô hình đích.

Một số công trình liên quan điều chỉnh ngưỡng $\beta$ của MIA theo từng điểm dữ liệu: Chang & Shokri (2021) cải thiện tấn công bằng cách xác định ngưỡng cho một điểm dữ liệu dựa trên thuộc tính của nó (ví dụ các điểm cùng giới tính được dùng làm quần thể tham chiếu để xác định ngưỡng ứng với một FPR cho trước). Khi khó xác định quần thể tham chiếu, Bertran et al. (2023) huấn luyện một mô hình tấn công cho từng ngưỡng FPR để xác định ngưỡng gắn với quần thể tham chiếu.

Một số nghiên cứu gần đây tập trung vào kiểm toán chi phí thấp cho các thuật toán học sâu có quyền riêng tư vi phân — bài toán khác nhưng cùng tinh thần giảm chi phí kiểm toán. Steinke et al. (2023) trình bày phương pháp hiệu quả để kiểm toán thuật toán ML riêng tư vi phân chỉ với một lần huấn luyện, tận dụng tính song song khi thêm/bớt nhiều mẫu huấn luyện một cách độc lập.

Phù hợp với các nghiên cứu MIA (Sankararaman et al., 2009; Murakonda et al., 2021; Ye et al., 2022; Carlini et al., 2022), nhóm nhấn mạnh công suất (TPR) của MIA ở sai số (FPR) cực thấp. Nhóm đề xuất phép kiểm định khác cho phép thiết kế tấn công mới tận dụng ưu điểm của nhiều tấn công mạnh. Tấn công đề xuất có hiệu năng tổng thể vượt trội và TPR tại FPR bằng 0 cao hơn các tấn công của (Carlini et al., 2022; Ye et al., 2022), đặc biệt khi chỉ huấn luyện số ít mô hình tham chiếu; đồng thời duy trì hiệu năng cao ở thiết lập offline, nơi không có mô hình tham chiếu nào được huấn luyện với mẫu đích. Đặc tính này khiến tấn công phù hợp với các kịch bản thực tế bị hạn chế về tài nguyên, thời gian và dữ liệu.

---

## Tài liệu tham khảo

Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., and Zhang, L. Deep learning with differential privacy. In Proceedings of the 23rd ACM SIGSAC Conference on Computer and Communications Security (CCS’16), pp. 308–318, 2016.

Backes, M., Berrang, P., Humbert, M., and Manoharan, P. Membership privacy in microrna-based studies. In Proceedings of the 23rd ACM SIGSAC Conference on Computer and Communications Security (CCS’16), pp. 319–330, 2016.

Banerjee, K., Prasad, V. C., Raj Gupta, R., Vyas, K., H, A., and Mishra, B. Exploring alternatives to softmax function. In Proceedings of the 2nd International Conference on Deep Learning Theory and Applications (DeLTA’21), pp. 81–86, 2021.

Bertran, M., Tang, S., Kearns, M., Morgenstern, J., Roth, A., and Wu, Z. S. Scalable membership inference attacks via quantile regression. In Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS’23), 2023.

Blundell, C., Cornebise, J., Kavukcuoglu, K., and Wierstra, D. Weight uncertainty in neural network. In International conference on machine learning, pp. 1613–1622. PMLR, 2015.

Carlini, N., Liu, C., Erlingsson, U., Kos, J., and Song, D. The secret sharer: Evaluating and testing unintended memorization in neural networks. In 28th USENIX Security Symposium (USENIX Security’19), pp. 267–284, 2019.

Carlini, N., Tramer, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K., Roberts, A., Brown, T., Song, D., Erlingsson, U., and et al. Extracting training data from large language models. In 30th USENIX Security Symposium (USENIX Security’21), 2021.

Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., and Tramer, F. Membership inference attacks from first principles. In IEEE Symposium on Security and Privacy (S&P’22), pp. 1897–1914, 2022.

Chang, H. and Shokri, R. On the privacy risks of algorithmic fairness. In 2021 IEEE European Symposium on Security and Privacy (EuroS&P), pp. 292–303. IEEE, 2021.

Chen, D., Yu, N., and Fritz, M. Relaxloss: Defending membership inference attacks without losing utility. In Proceedings of the 10th International Conference on Learning Representations (ICLR’22), 2022.

Chen, M., Zhang, Z., Wang, T., Backes, M., Humbert, M., and Zhang, Y. When machine unlearning jeopardizes privacy. In Proceedings of the 28th ACM SIGSAC Conference on Computer and Communications Security (CCS ’21), pp. 896–911, 2021.

Choquette-Choo, C. A., Tramer, F., Carlini, N., and Papernot, N. Label-only membership inference attacks. In Proceedings of the 38th International Conference on Machine Learning (ICML’21), pp. 1964–1974, 2021.

De Brebisson, A. and Vincent, P. An exploration of softmax alternatives belonging to the spherical loss family. In Proceedings of the 4th International Conference on Learning Representations (ICLR’16), 2016.

Dwork, C. Differential privacy. In Proceedings of 33rd International Colloquium in Automata, Languages and Programming (ICALP’06), pp. 1–12, 2006.

Dwork, C., Smith, A., Steinke, T., Ullman, J., and Vadhan, S. Robust traceability from trace amounts. In 2015 IEEE 56th Annual Symposium on Foundations of Computer Science, pp. 650–669. IEEE, 2015.

Fredrikson, M., Jha, S., and Ristenpart, T. Model inversion attacks that exploit confidence information and basic countermeasures. In Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security (CCS’15), pp. 1322–1333, 2015.

Ganju, K., Wang, Q., Yang, W., Gunter, C. A., and Borisov, N. Property inference attacks on fully connected neural networks using permutation invariant representations. In Proceedings of the 25th ACM SIGSAC Conference on Computer and Communications Security (CCS’18), pp. 619–633, 2018.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR’16), pp. 1, 2016.

Hisamoto, S., Post, M., and Duh, K. Membership inference attacks on sequence-to-sequence models: Is my data in your machine translation system? Transactions of the Association for Computational Linguistics, 8:49–63, 2020.

Homer, N., Szelinger, S., Redman, M., Duggan, D., Tembe, W., Muehling, J., Pearson, J. V., Stephan, D. A., Nelson, S. F., and Craig, D. W. Resolving individuals contributing trace amounts of dna to highly complex mixtures using high-density snp genotyping microarrays. PLoS Genetics, 4(8), 2008.

Jayaraman, B., Wang, L., Knipmeyer, K., Gu, Q., and Evans, D. Revisiting membership inference under realistic assumptions. arXiv preprint arXiv:2005.10881, 2020.

Jia, J., Salem, A., Backes, M., Zhang, Y., and Gong, N. Z. Memguard: Defending against black-box membership inference attacks via adversarial examples. In Proceedings of the 26th ACM SIGSAC Conference on Computer and Communications Security (CCS’19), pp. 259–274, 2019.

Leemann, T., Pawelczyk, M., and Kasneci, G. Gaussian membership inference privacy. In Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS’23), 2023.

Leino, K. and Fredrikson, M. Stolen memories: Leveraging model memorization for calibrated white-box membership inference. In 29th USENIX Security Symposium (USENIX Security’20), pp. 1605–1622, 2020.

Li, J., Li, N., and Ribeiro, B. Membership inference attacks and defenses in classification models. In Proceedings of the 11th ACM Conference on Data and Application Security and Privacy (CODASPY’21), pp. 5–16, 2021.

Li, Z. and Zhang, Y. Membership leakage in label-only exposures. In Proceedings of the 28th ACM SIGSAC Conference on Computer and Communications Security (CCS’21), pp. 880–895, 2021.

Li, Z., Liu, Y., He, X., Yu, N., Backes, M., and Zhang, Y. Auditing membership leakages of multi-exit networks. In Proceedings of the 29th ACM SIGSAC Conference on Computer and Communications Security (CCS’22), pp. 1917–1931, 2022.

Liang, X., Wang, X., Lei, Z., Liao, S., and Z., L. S. Soft-margin softmax for deep classification. In Proceedings of the 24th International Conference on Neural Information Processing (ICONIP’17), pp. 413–421, 2017.

Liu, W., Wen, Y., Yu, Z., and Yang, M. Large-margin softmax loss for convolutional neural networks. In Proceedings of the 33rd International Conference on Machine Learning (ICML’16), 2016.

Liu, Y., Wen, R., He, X., Salem, A., Zhang, Z., Backes, M., Cristofaro, E. D., Fritz, M., and Zhang, Y. Ml-doctor: Holistic risk assessment of inference attacks against machine learning models. In 31st USENIX Security Symposium (USENIX Security’22), pp. 4525–4542, 2022a.

Liu, Y., Zhao, Z., Backes, M., and Yang, Z. Membership inference attacks by exploiting loss trajectory. In Proceedings of the 29th ACM SIGSAC Conference on Computer and Communications Security (CCS’22), pp. 2085–2098, 2022b.

Long, Y., Wang, L., Bu, D., Bindschaedler, V., Wang, X., Tang, H., Gunter, C. A., and Chen, K. A pragmatic approach to membership inferences on machine learning models. In IEEE European Symposium on Security and Privacy (EuroS&P’20), pp. 521–534, 2020.

MacKay, D. J. Information theory, inference and learning algorithms. Cambridge university press, 2003.

Melis, L., Song, C., De Cristofaro, E., and Shmatikov, V. Exploiting unintended feature leakage in collaborative learning. In IEEE Symposium on Security and Privacy (S&P’19), pp. 691–706, 2019.

Murakonda, S. K., Shokri, R., and Theodorakopoulos, G. Quantifying the privacy risks of learning high-dimensional graphical models. In Proceedings of the International Conference on Artificial Intelligence and Statistics, pp. 2287–2295, 2021.

Nasr, M., Shokri, R., and Houmansadr, A. Machine learning with membership privacy using adversarial regularization. In Proceedings of the 25th ACM SIGSAC Conference on Computer and Communications Security (CCS’18), pp. 634–646, 2018.

Nasr, M., Shokri, R., and Houmansadr, A. Comprehensive privacy analysis of deep learning: Passive and active white-box inference attacks against centralized and federated learning. In IEEE Symposium on Security and Privacy (S&P’19), pp. 1022–1036, 2019.

Nasr, M., Song, S., Thakurta, A., Papernot, N., and Carlini, N. Adversary instantiation: Lower bounds for differentially private machine learning. In IEEE Symposium on Security and Privacy (S&P’21), pp. 866–882, 2021.

Rahman, M. A., Rahman, T., Laganiere, R., Mohammed, N., and Wang, Y. Membership inference attack against differentially private deep learning model. Transactions on Data Privacy, 11(1):61–79, 2018.

Sablayrolles, A., Douze, M., Schmid, C., Ollivier, Y., and Jégou, H. White-box vs black-box: Bayes optimal strategies for membership inference. In Proceedings of the 36th International Conference on Machine Learning (ICML’19), pp. 5558–5567, 2019.

Salem, A., Zhang, Y., Humbert, M., Fritz, M., and Backes, M. Ml-leaks: Model and data independent membership inference attacks and defenses on machine learning models. In Network and Distributed Systems Security Symposium (NDSS’19), 2019.

Sankararaman, S., Obozinski, G., Jordan, M. I., and Halperin, E. Genomic privacy and limits of individual detection in a pool. Nature genetics, 41(9):965–967, 2009.

Shokri, R., Stronati, M., Song, C., and Shmatikov, V. Membership inference attacks against machine learning models. In IEEE Symposium on Security and Privacy (S&P’17), pp. 3–18, 2017.

Song, L. and Mittal, P. Systematic evaluation of privacy risks of machine learning models. In 30th USENIX Security Symposium (USENIX Security’21), pp. 2615–2632, 2021.

Steinke, T., Nasr, M., and Jagielski, M. Privacy auditing with one (1) training run. Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS’23), 2023.

Tang, X., Mahloujifar, S., Song, L., Shejwalkar, V., Nasr, M., Houmansadr, A., and Mittal, P. Mitigating membership inference attacks by self-distillation through a novel ensemble architecture. In 31st USENIX Security Symposium (USENIX Security’22), pp. 1433–1450, 2022.

Thudi, A., Shumailov, I., Boenisch, F., and Papernot, N. Bounding membership inference. In arXiv preprint, pp. arXiv:2202.12232, 2022.

Truex, S., Liu, L., Gursoy, M. E., Yu, L., and Wei, W. Demystifying membership inference attacks in machine learning as a service. IEEE Transactions on Services Computing, 14(6):2073–2089, 2019.

Visscher, P. M. and Hill, W. G. The limits of individual identification from sample allele frequencies: theory and statistical analysis. PLoS genetics, 5(10):e1000628, 2009.

Watson, L., Guo, C., Cormode, G., and Sablayrolles, A. On the importance of difficulty calibration in membership inference attacks. In Proceedings of International Conference on Learning Representations (ICLR’22), 2022a.

Watson, L., Guo, C., Cormode, G., and Sablayrolles, A. On the importance of difficulty calibration in membership inference attacks. In Proceedings of the 10th International Conference on Learning Representations (ICLR’22), 2022b.

Wen, Y., Bansal, A., Kazemi, H., Borgnia, E., Goldblum, M., Geiping, J., and Goldstein, T. Canary in a coalmine: Better membership inference with ensembled adversarial queries. In Proceedings of the 11th International Conference on Learning Representations (ICLR’23), 2023.

Xu, Z., Shi, S., Liu, A. X., Zhao, J., and Chen, L. An adaptive and fast convergent approach to differentially private deep learning. In Proceedings of 39th IEEE International Conference on Computer Communications (Infocom’20), pp. 1867–1876, 2020.

Ye, J., Maddi, A., Murakonda, S. K., Bindschaedler, V., and Shokri, R. Enhanced membership inference attacks against machine learning models. In Proceedings of the 29th ACM SIGSAC Conference on Computer and Communications Security (CCS’22), pp. 3093–3106, 2022.

Ye, J., Borovykh, A., Hayou, S., and Shokri, R. Leave-one-out distinguishability in machine learning. 12th International Conference on Learning Representations (ICLR’24), 2024.

Yeom, S., Giacomelli, I., Fredrikson, M., and Jha, S. Privacy risk in machine learning: Analyzing the connection to overfitting. In 2018 IEEE 31st Computer Security Foundations Symposium (CSF’18), pp. 268–282. IEEE, 2018.

Yu, L., Liu, L., Pu, C., Gursoy, M. E., and Truex, S. Differentially private model publishing for deep learning. In IEEE Symposium on Security and Privacy (S&P’19), pp. 332–349, 2019.

Zhang, C., Ippolito, D., Lee, K., Jagielski, M., Tramèr, F., and Carlini, N. Counterfactual memorization in neural language models. arXiv preprint arXiv:2112.12938, 2021.
