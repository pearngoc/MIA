# Một chuỗi thời gian đáng giá 64 từ: Dự báo dài hạn với Transformer

*(A Time Series Is Worth 64 Words: Long-term Forecasting with Transformers)*

**Tác giả:** Yuqi Nie (Princeton University), Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam (IBM Research)
**Hội nghị/năm:** ICLR 2023 (arXiv:2211.14730)

> Tóm tắt bằng tiếng Việt của bài báo gốc (do AI viết lại bằng lời văn riêng, **không phải bản dịch đầy đủ**). Các con số lấy từ bài báo; vui lòng đối chiếu với bản gốc khi trích dẫn.

---

## Tóm tắt (Abstract)

Bài báo đề xuất **PatchTST**, một thiết kế Transformer hiệu quả cho dự báo chuỗi thời gian đa biến và học biểu diễn tự giám sát (self-supervised representation learning). Mô hình dựa trên hai thành phần chính:

1. **Patching**: chia chuỗi thời gian thành các patch ở mức chuỗi con (subseries) và dùng chúng làm token đầu vào cho Transformer.
2. **Channel-independence**: mỗi kênh là một chuỗi đơn biến, các kênh dùng chung embedding và trọng số Transformer.

Patching mang lại ba lợi ích: giữ được thông tin ngữ nghĩa cục bộ, giảm theo bậc hai chi phí tính toán/bộ nhớ của attention map với cùng look-back window, và cho phép mô hình nhìn được lịch sử dài hơn. PatchTST cải thiện đáng kể độ chính xác so với các mô hình Transformer SOTA, đạt kết quả fine-tuning tốt khi tiền huấn luyện tự giám sát (vượt huấn luyện có giám sát trên các tập dữ liệu lớn), và việc chuyển giao biểu diễn masked pre-trained giữa các tập dữ liệu cũng cho độ chính xác SOTA.

## 1. Giới thiệu

- Dự báo là tác vụ trọng tâm của phân tích chuỗi thời gian. Transformer (Informer, Autoformer, FEDformer…) thành công ở nhiều lĩnh vực, nhưng Zeng et al. (2022) cho thấy một mô hình tuyến tính rất đơn giản (DLinear) vượt các Transformer trước đó, đặt ra nghi vấn về tính hữu ích của Transformer cho dự báo chuỗi thời gian.
- Các tác giả trả lời bằng hai thiết kế:
  - **Patching**: một bước thời gian đơn lẻ không có ý nghĩa ngữ nghĩa như một từ trong câu, nên gom các bước thời gian thành patch để nắm thông tin cục bộ.
  - **Channel-independence**: mỗi token chỉ chứa thông tin của một kênh (trái với channel-mixing, trong đó token là vector của mọi kênh). Cách này đã hiệu quả với CNN và mô hình tuyến tính nhưng chưa được áp dụng cho Transformer.
- Nghiên cứu tình huống trên tập Traffic (862 chuỗi), Bảng 1, cho thấy ba ưu điểm:
  1. **Giảm độ phức tạp thời gian/bộ nhớ**: số token giảm từ khoảng L xuống khoảng L/S; với P = 16, S = 8, L = 336, thời gian huấn luyện giảm tới khoảng 22 lần trên tập lớn.
  2. **Học từ look-back window dài hơn**: tăng L từ 96 lên 336 giảm MSE từ 0,518 xuống 0,397; thêm patching giảm tiếp xuống 0,367. Down-sampling với L = 380 (N = 96) cho 0,447, tốt hơn 0,518, chứng tỏ lịch sử dài mang thêm thông tin.
  3. **Khả năng học biểu diễn**: tiền huấn luyện tự giám sát đạt MSE 0,349 (tốt nhất trong Bảng 1).

## 2. Công trình liên quan

- **Patch trong các mô hình Transformer**: NLP (tokenization theo subword trong BERT), thị giác máy tính (ViT chia ảnh thành patch 16×16; BEiT, masked autoencoder), và xử lý tiếng nói (tích chập trích đặc trưng mức chuỗi con).
- **Dự báo dài hạn bằng Transformer**: LogTrans, Informer (ProbSparse), Autoformer (decomposition, auto-correlation), FEDformer (cấu trúc tăng cường Fourier), Pyraformer (attention dạng kim tự tháp). Hầu hết tập trung giảm độ phức tạp của attention và dùng attention theo từng điểm (point-wise), bỏ qua vai trò của patch; Triformer có patch attention nhưng chỉ nhằm giảm độ phức tạp.
- **Học biểu diễn chuỗi thời gian**: nhiều phương pháp không dựa trên Transformer (contrastive, v.v.); các nỗ lực dựa trên Transformer như TST và TS-TCC chưa khai thác hết tiềm năng.

## 3. Phương pháp đề xuất

### 3.1 Cấu trúc mô hình

- **Bài toán**: cho chuỗi đa biến với look-back window L, mỗi $x_t$ có số chiều $M$; dự báo $T$ giá trị tương lai.
- **Quy trình xuôi (forward)**: chuỗi đầu vào được tách thành $M$ chuỗi đơn biến $x^{(i)} \in \mathbb{R}^{1\times L}$, mỗi chuỗi đi độc lập qua cùng một Transformer backbone, cho ra dự báo $\hat{x}^{(i)} \in \mathbb{R}^{1\times T}$.
- **Patching**: với độ dài patch $P$ và stride $S$, số patch là $N = \lfloor (L-P)/S \rfloor + 2$ (đệm $S$ giá trị cuối vào cuối chuỗi). Số token giảm từ $L$ còn xấp xỉ $L/S$, nên bộ nhớ và độ phức tạp attention giảm khoảng $S^2$ lần.
- **Transformer Encoder**: dùng encoder Transformer chuẩn; các patch được chiếu tuyến tính vào không gian ẩn $D$ chiều và cộng positional encoding học được; multi-head attention dạng scaled dot-product, kèm BatchNorm và mạng feed-forward có residual. Cuối cùng là lớp flatten + linear head để ra dự báo.
- **Hàm mất mát**: MSE, tính trung bình trên $M$ kênh.
- **Instance normalization**: chuẩn hóa từng chuỗi (trung bình 0, độ lệch chuẩn 1) trước khi patching, rồi cộng lại trung bình/độ lệch chuẩn vào đầu ra, nhằm giảm ảnh hưởng của distribution shift.

### 3.2 Học biểu diễn

- Dùng masked autoencoder: che ngẫu nhiên một phần đầu vào và huấn luyện mô hình khôi phục.
- Hạn chế của cách che ở mức từng bước thời gian (như TST): giá trị bị che dễ suy ra bằng nội suy lân cận, không buộc mô hình hiểu cấu trúc tổng thể; và lớp đầu ra cho dự báo có ma trận tham số rất lớn $(L\cdot D)\times(M\cdot T)$, dễ overfit.
- PatchTST: dùng cùng encoder, thay head bằng lớp tuyến tính $D\times P$; patch không chồng lấn; chọn ngẫu nhiên một tập chỉ số patch để che bằng 0; huấn luyện bằng MSE để tái tạo patch bị che. Do dùng chung trọng số giữa các kênh, dữ liệu tiền huấn luyện có thể có số chuỗi khác với dữ liệu downstream.

## 4. Thực nghiệm

### 4.1 Dự báo chuỗi thời gian dài hạn

- **Dữ liệu**: 8 tập phổ biến — Weather, Traffic, Electricity, ILI và 4 tập ETT (ETTh1, ETTh2, ETTm1, ETTm2). Weather, Traffic, Electricity có nhiều chuỗi hơn nên kết quả ổn định hơn.
- **Baseline**: FEDformer, Autoformer, Informer, Pyraformer, LogTrans (Transformer) và DLinear (phi-Transformer). Các tác giả chạy lại FEDformer, Autoformer, Informer với 6 giá trị look-back khác nhau và lấy kết quả tốt nhất để tạo baseline mạnh. Thước đo: MSE, MAE.
- **Biến thể**: PatchTST/64 (64 patch, L = 512) và PatchTST/42 (42 patch, L = 336), cùng P = 16, S = 8.
- **Kết quả (Bảng 3)**: PatchTST vượt các baseline. So với kết quả tốt nhất của Transformer, PatchTST/64 giảm MSE 21,0% và MAE 16,7%; PatchTST/42 giảm MSE 20,2% và MAE 16,4%. So với DLinear, PatchTST nhìn chung tốt hơn, đặc biệt ở các tập lớn và ILI. Kết quả đơn biến nằm ở Phụ lục A.3.

### 4.2 Học biểu diễn

- **Thiết lập**: độ dài chuỗi 512, kích thước patch 12 (42 patch, không chồng lấn), che 40% patch; tiền huấn luyện 100 epoch. Đánh giá bằng (a) linear probing (chỉ huấn luyện head 20 epoch) hoặc (b) linear probing 10 epoch rồi fine-tuning toàn mạng 20 epoch.
- **So với học có giám sát (Bảng 4)**: trên tập lớn, tiền huấn luyện cải thiện rõ so với huấn luyện từ đầu; linear probing đã tương đương huấn luyện từ đầu và tốt hơn DLinear; fine-tuning end-to-end cho kết quả tốt nhất.
- **Transfer learning (Bảng 5)**: tiền huấn luyện trên Electricity rồi fine-tune sang tập khác; MSE kém hơn một chút so với tiền huấn luyện cùng tập (hợp lý) và đôi khi kém hơn huấn luyện có giám sát, nhưng vẫn tốt hơn các mô hình khác, đồng thời tốn ít thời gian hơn vì chỉ huấn luyện lại head hoặc toàn mô hình trong ít epoch.
- **So với phương pháp tự giám sát khác (Bảng 6)**: so với BTSF, TS2Vec, TNC, TS-TCC trên ETTh1 (chỉ linear probing), PatchTST vượt trội, cải thiện từ 34,5% đến 48,8% tùy độ dài dự báo, cả khi tiền huấn luyện trên chính ETTh1 lẫn khi chuyển từ Traffic.

### 4.3 Nghiên cứu loại trừ (Ablation)

- **Patching và channel-independence (Bảng 7)**: so sánh bốn trường hợp (P+CI, CI, P, mô hình TST gốc); cả hai thành phần đều góp phần cải thiện độ chính xác. Patching còn cải thiện thời gian chạy và bộ nhớ. Phân tích sâu về channel-independence nằm ở Phụ lục A.7.
- **Thay đổi look-back window (Hình 2)**: các baseline Transformer phần lớn không hưởng lợi khi tăng L, trong khi MSE của PatchTST giảm đều khi receptive field tăng.

## 5. Kết luận và hướng phát triển

PatchTST giới thiệu patching và cấu trúc channel-independent cho Transformer, giúp nắm thông tin ngữ nghĩa cục bộ và tận dụng look-back window dài. Mô hình vượt các baseline trong học có giám sát và cho thấy tiềm năng trong học biểu diễn tự giám sát và transfer learning; có thể là nền tảng cho các mô hình Transformer dự báo và foundation model chuỗi thời gian. Hướng tương lai: mô hình hóa đúng đắn sự phụ thuộc giữa các kênh.

## Tài liệu tham khảo

Danh sách tài liệu tham khảo đầy đủ xem trong bài báo gốc (`nie2023_PatchTST.pdf`, từ trang 10).

## Phụ lục (tóm lược)

- **A.1 Chi tiết thực nghiệm**: mô tả các tập dữ liệu (Weather ở Đức, Traffic từ cảm biến xa lộ San Francisco, Electricity của 321 khách hàng, ILI hàng tuần, các tập ETT); lý do không đưa Exchange-rate vào benchmark (thị trường hiệu quả, dự báo bằng giá trị cuối đã rất mạnh); cài đặt baseline (look-back mặc định, và tập ILI dùng bộ tham số riêng); lý do không so sánh với mô hình truyền thống (ARIMA, LSTM, TCN, DeepAR); tham số mô hình mặc định (3 lớp encoder, H = 16 head, D = 128, F = 256, GELU, dropout 0,2; tập nhỏ ILI/ETTh1/ETTh2 dùng H = 4, D = 16, F = 128); cách cài đặt hiệu quả bằng cách reshape tensor $B\times M\times P\times N$ thành $(B\cdot M)\times P\times N$.
- **A.2 Trực quan hóa**: PatchTST bám sát xu hướng và giá trị thực nhất (Hình 3).
- **A.3 Dự báo đơn biến**: kết quả trên các tập ETT (Bảng 8), baseline lấy từ Zeng et al. (2022).
- **A.4 Thêm kết quả ablation**: ảnh hưởng của độ dài patch, look-back window, patching và channel-independence, instance normalization.
- **A.5 Thêm kết quả học biểu diễn tự giám sát**: benchmark đầy đủ cho dự báo đa biến và transfer learning.
- **A.6 Phân tích độ bền vững**: kết quả với nhiều random seed khác nhau và nhiều cấu hình tham số mô hình.
- **A.7 Phân tích channel-independence**:
  - *So với channel-mixing*, ba yếu tố khiến channel-independence tốt hơn: (i) tính thích ứng — mỗi chuỗi có attention map riêng, các chuỗi tương tự cho map tương tự; (ii) mô hình channel-mixing cần nhiều dữ liệu huấn luyện hơn để đạt hiệu năng tương đương; (iii) ít overfit hơn trong quá trình huấn luyện. Ngoài ra còn các lợi thế: khả năng mở rộng để học tương quan giữa các kênh (ví dụ bằng GNN), học đa nhiệm, và bền vững hơn với nhiễu.
  - *Trên các mô hình khác*: áp dụng channel-independence cho Informer, Autoformer, FEDformer (Bảng 15) nhìn chung cải thiện hiệu năng, nhưng vẫn chưa vượt PatchTST.

---

*Ghi chú: Các bảng số liệu chi tiết (Bảng 1–8, 15 và các bảng phụ lục), công thức đầy đủ và hình minh họa không được sao chép lại ở đây; xem bản gốc.*
