# Script thuyết trình — MIA trên mô hình chuỗi thời gian

Đề tài: *Tấn công suy luận tư cách thành viên trên các mô hình chuỗi thời gian hiện đại: đánh giá hệ thống trên PTB-XL ECG.*

Tổng thời lượng nội dung: **khoảng 15 phút**. Để slide 15 trên màn hình khi hỏi đáp.

Cách dùng: đọc phần **Lời thoại**. Chữ trong ngoặc vuông là việc làm trên slide, không đọc thành tiếng. Mỗi slide kết bằng một câu nhớ; dừng một nhịp rồi mới chuyển.

Lưu ý khi nói: tiêu đề slide 1 đang ghi *Membership Interference Attack*. Thuật ngữ đúng là **Membership Inference Attack** — nói “inference”, nghĩa là suy luận, không nói “interference”.

Gợi ý một người nói xuyên suốt cho mạch liền. Nếu chia vai: người mở (slide 1–6 và 10–15) giữ nhịp; ba thành viên còn lại chỉ vào ở slide 13 và phần hỏi đáp đúng mô hình mình làm.

| Slide | Nội dung | Phút |
|------:|----------|-----:|
| 1 | Mở đầu | 0:30 |
| 2 | Chuyển phần | 0:10 |
| 3 | Vấn đề và câu hỏi | 1:30 |
| 4 | Dữ liệu và thiết lập | 1:30 |
| 5 | Bốn họ tấn công | 1:30 |
| 6 | Chuyển phần | 0:10 |
| 7 | Tổng quan kết quả | 1:30 |
| 8 | RQ1 | 2:00 |
| 9 | RQ2 | 2:00 |
| 10 | Chuyển phần | 0:10 |
| 11 | Bài học kiểm toán | 2:00 |
| 12 | Hạn chế và kết luận | 1:30 |
| 13 | Phân công | 0:30 |
| 14 | Tài liệu | lướt |
| 15 | Cảm ơn và hỏi đáp | — |

---

## Slide 1 — Trang bìa (0:30)

[Đứng yên, nhìn hội đồng, không đọc danh sách giảng viên.]

Kính chào thầy cô và các bạn. Nhóm em xin trình bày báo cáo cuối kỳ môn An toàn và Bảo mật Hệ thống Thông tin.

Đề tài của nhóm là tấn công suy luận tư cách thành viên — Membership Inference Attack — trên các mô hình chuỗi thời gian. Câu hỏi xuyên suốt của bài là: một mô hình học sâu huấn luyện trên điện tâm đồ có làm lộ bệnh nhân nào đã nằm trong tập huấn luyện hay không.

Nhóm được hướng dẫn bởi PGS.TS. Đặng Trần Khánh, TS. Phạm Thị Bạch Huệ và ThS. Lương Vĩ Minh. Em là [tên người đang nói], trình bày thay cho nhóm 8.

---

## Slide 2 — Chuyển phần: bối cảnh (0:10)

Phần đầu, em xin nói ngắn về vấn đề, cách nhóm thiết lập thực nghiệm, và bốn họ tấn công đã dùng.

---

## Slide 3 — Vấn đề và câu hỏi nghiên cứu (1:30)

[Chỉ cột “Vấn đề”.]

Kẻ tấn công ở đây không cần xem trọng số mô hình. Chỉ cần gửi một bản ghi ECG vào và nhận đầu ra — tức là truy cập hộp đen — là có thể đoán bản ghi đó đã được dùng để huấn luyện hay chưa. Đó là tấn công suy luận tư cách thành viên, MIA, được Shokri và cộng sự nêu từ năm 2017.

Với dữ liệu y tế, hệ quả không dừng ở một nhãn kỹ thuật. Nếu mô hình gắn với một bệnh, việc biết ECG của một người nằm trong tập huấn luyện có thể tiết lộ thông tin sức khỏe, và đụng tới các quy định như GDPR hay HIPAA.

[Chỉ “Khoảng trống”.]

MIA đã được nghiên cứu nhiều trên ảnh và dữ liệu bảng. Chuỗi thời gian thì ít hơn nhiều. Nhóm thấy hai khoảng trống. Một là chưa có đánh giá hệ thống trên các kiến trúc hiện đại — PatchTST, iTransformer — và trên foundation model như MOMENT. Hai là chưa có so sánh, trên cùng một tập dữ liệu, giữa bài toán phân loại và bài toán dự báo, theo tiêu chuẩn low-FPR của Carlini.

[Chỉ RQ1 rồi RQ2.]

Từ đó nhóm đặt hai câu hỏi. RQ1: kiến trúc hiện đại có rò rỉ nhiều hơn baseline LSTM và TCN hay không. RQ2: trên cùng dữ liệu, phân loại hay dự báo dễ bị tấn công hơn. Em sẽ quay lại trả lời từng câu ở phần kết quả.

**Câu nhớ:** vấn đề không phải mô hình có chính xác hay không, mà là đầu ra hộp đen có tiết lộ tư cách thành viên hay không.

---

## Slide 4 — Dữ liệu và thiết lập (1:30)

[Chỉ số 16.272.]

Nhóm dùng PTB-XL, tập ECG công khai. Sau khi giữ các bản ghi có đúng một siêu lớp chẩn đoán, còn 16.272 bản ghi của 14.641 bệnh nhân.

[Chỉ 8.136 | 7.636.]

Chia ở mức bản ghi, seed 42: 8.136 member, tức là nằm trong tập huấn luyện, và 7.636 non-member. Đây là điểm em muốn nói trước: chia theo bản ghi, không theo bệnh nhân. Ít nhất 266 bệnh nhân xuất hiện ở cả hai phía. Em sẽ nhắc lại ở phần hạn chế.

[Chỉ “5 × 2”.]

Năm mô hình, mỗi mô hình hai bài toán. Baseline là LSTM và TCN. Kiến trúc hiện đại là PatchTST và iTransformer. Foundation model là MOMENT, tinh chỉnh toàn bộ trọng số. Phân loại là 5 lớp chẩn đoán. Dự báo lấy ngữ cảnh 512 điểm và horizon 96.

[Chỉ “Mô hình đe dọa”.]

Mô hình đe dọa chính là hộp đen: gửi x, nhận f(x). LiRA và RMIA cần thêm shadow model. Temporal Feature thì mạnh hơn về giả định: cần nhãn member, non-member và checkpoint theo epoch, nên là gray-box. Em không xếp Temporal ngang hàng với các tấn công hộp đen.

[Chỉ “Chỉ số chính”.]

Chỉ số chính là TPR tại 0,1% FPR, kèm khoảng tin cậy bootstrap 95%. Lý do nhóm không lấy AUC làm chỉ số chính: AUC cao có thể đến từ vùng FPR cao, nơi kẻ tấn công báo nhầm rất nhiều. Trong thực tế, kẻ tấn công cần nhận đúng member trong khi gần như không nhận nhầm người ngoài. Với 7.636 non-member, ngưỡng 0,1% chỉ cho phép khoảng 7 trường hợp nhận nhầm.

**Câu nhớ:** nhóm đo “nhận đúng khi hầu như không báo động nhầm”, không đo AUC chung chung.

---

## Slide 5 — Bốn họ tấn công (1:30)

[Đi từ trái sang phải, mỗi ô một câu.]

Nhóm xếp tấn công từ đơn giản đến mạnh.

Họ thứ nhất là Loss, Confidence và Entropy. Chỉ nhìn đầu ra của mô hình mục tiêu, không cần shadow model. Đây là kiểm tra mà một hệ thống rất dễ dừng lại.

Họ thứ hai là LiRA, Likelihood Ratio Attack của Carlini. Ý tưởng là hỏi: bản ghi này giống những gì mô hình thường làm khi nó đã học bản ghi đó, hay giống khi nó chưa từng thấy. Để trả lời, cần các shadow model huấn luyện trên tập có và không có bản ghi mục tiêu.

Họ thứ ba là RMIA. Cũng dùng mô hình tham chiếu, nhưng đối chiếu bản ghi với một tập tham chiếu và hiệu chỉnh theo độ khó của mẫu. Mẫu vốn đã khó thì loss cao không có nghĩa là nó là member.

Họ thứ tư là Temporal Feature. Bộ phân loại học trên đặc trưng theo các epoch, gồm cả tín hiệu tần số. Mạnh, nhưng gray-box, như em vừa nói.

Điểm em muốn hội đồng giữ lại: không có họ nào thắng ở mọi mô hình và mọi bài toán. Kết quả phía sau sẽ cho thấy điều đó rất rõ.

**Câu nhớ:** kiểm tra quyền riêng tư bằng một điểm số loss là chưa đủ.

---

## Slide 6 — Chuyển phần: kết quả (0:10)

Phần hai là kết quả. Em đi từ bức tranh chung, rồi trả lời RQ1 và RQ2.

---

## Slide 7 — Tổng quan (1:30)

[Chỉ dải 0,6% → 9,9%.]

Ở bài toán phân loại, tấn công tốt nhất tách năm mô hình thành hai nhóm. LSTM khoảng 0,6% và PatchTST khoảng 1,8% — nhóm thấp. TCN, iTransformer và MOMENT nằm khoảng 8 đến 10% — cao hơn từ 4 đến 16 lần. Khoảng tin cậy của ba mô hình sau chồng lên nhau, nên em không xếp hạng giữa TCN, iTransformer và MOMENT. Chỉ tách được nhóm thấp với nhóm cao.

[Chỉ 0,76–0,78.]

Trong khi đó, độ chính xác kiểm tra của LSTM, TCN, PatchTST và MOMENT nằm sát nhau, khoảng 0,76 đến 0,78.

Đây là thông điệp của slide: utility ngang nhau không kéo theo quyền riêng tư ngang nhau. Không thể nhìn accuracy rồi kết luận mô hình an toàn như nhau.

**Câu nhớ:** cùng độ chính xác, mức rò rỉ vẫn có thể chênh tới khoảng 16 lần.

---

## Slide 8 — RQ1 (2:00)

[Đọc câu trả lời trên slide, rồi mới giải thích.]

Trả lời RQ1: không có ưu thế hệ thống của kiến trúc hiện đại so với baseline.

Ở phân loại, PatchTST rò rỉ hơn LSTM, nhưng ít hơn TCN. Tức là một Transformer theo patch không tự động nguy hiểm hơn, cũng không tự động an toàn hơn, một mạng tích chập thời gian.

Ở dự báo, LSTM, TCN và iTransformer tụ quanh 13 đến 14%. iTransformer ngang TCN. PatchTST cao hơn, khoảng 28,5%. MOMENT là điểm ngoại lệ, em sẽ nói ở slide sau.

[Chỉ cột generalization gap.]

Về generalization gap. Ở phân loại, thứ tự gap khớp khá với thứ tự rò rỉ của các tấn công chính: LSTM nhỏ nhất, rồi PatchTST, rồi MOMENT và TCN. Điều này ủng hộ kết quả cổ điển của Yeom: overfit là điều kiện cần cho tấn công loss.

Nhưng gap không giải thích hết. iTransformer có gap lớn nhất, khoảng 0,48 — train gần như hoàn hảo, test chỉ khoảng 0,52 — mà LiRA và RMIA chỉ quanh 1%. Nguyên nhân là điểm số bị bão hòa và mô hình gần như không tổng quát hóa. Ngược lại, TCN trên dự báo có gap MSE nhỏ nhất, khoảng 0,08, mà vẫn rò rỉ 13,8%.

Kết luận thận trọng của nhóm: gap là điều kiện cần cho tấn công điểm số đơn giản. Tấn công có hiệu chỉnh theo độ khó vẫn phát hiện được rò rỉ khi gap trung bình nhỏ. Và nhóm chưa so các mô hình ở cùng một mức gap, vì epoch và learning rate được chọn theo utility trên tập validation.

**Câu nhớ:** kiến trúc không quyết định mức rò rỉ; overfit cũng không phải lời giải thích duy nhất.

---

## Slide 9 — RQ2 (2:00)

[Chỉ câu trả lời trên slide.]

Trả lời RQ2: trên cả năm mô hình, dự báo rò rỉ nhiều hơn phân loại. Hệ số từ 1,7 lần ở TCN đến 21 lần ở LSTM. PatchTST khoảng 15 lần. MOMENT khoảng 10 lần, nhưng mức tăng tuyệt đối là lớn nhất: từ 9,9% lên 97,7%, tức thêm khoảng 88 điểm phần trăm.

Em muốn tách hai cách đọc. Hệ số nhân dễ gây hiểu nhầm. LSTM nhân 21 chỉ vì mẫu số phân loại rất nhỏ, 0,6%; mức tăng tuyệt đối chỉ khoảng 12,5 điểm. MOMENT nhân khoảng 10, nhưng gần như mọi bản ghi huấn luyện đều bị nhận ra ở ngưỡng rất chặt.

Mức chắc chắn cũng không đều. Kết luận chắc với LSTM, PatchTST và MOMENT. Với TCN, khoảng tin cậy của hai bài toán tách nhau, khác biệt có ý nghĩa nhưng nhỏ. Với iTransformer, hai khoảng tin cậy chồng lấn, nên hệ số 1,8 lần chưa được chứng minh.

Vì sao dự báo dễ lộ hơn? Nhóm coi đây là giả thuyết, chưa phải định lý. Dự báo trả về 12 nhân 96, tức hơn một nghìn giá trị liên tục; mỗi giá trị là một phép đo ghi nhớ. Phân loại chỉ trả về 5 xác suất, và các xác suất này dễ bão hòa. Hồi quy trên cả dạng sóng khuyến khích khớp chi tiết từng bản ghi; cross-entropy chỉ cần tách 5 lớp.

Có một yếu tố gây nhiễu cần nói thẳng: LSTM, TCN và PatchTST huấn luyện dự báo 50 epoch, phân loại 20 epoch. MOMENT và iTransformer thì cùng số epoch cho cả hai bài toán — MOMENT 10 và 10, iTransformer 50 và 50 — nên phép so sánh của hai mô hình này không bị lệch epoch. MOMENT vẫn cho khoảng 10 lần.

Một lưu ý nữa: con số “tấn công tốt nhất” gồm cả biến thể online, là cận trên. Nếu chỉ lấy tấn công chính theo protocol, MOMENT dự báo vẫn ở 94,4% với LiRA. Kết luận định tính không đổi.

**Câu nhớ:** cùng một ECG, bài toán dự báo để lộ tư cách thành viên nhiều hơn phân loại; với MOMENT, mức lộ là gần như hoàn toàn.

---

## Slide 10 — Chuyển phần: bài học (0:10)

Phần cuối là những gì nhóm rút ra cho việc kiểm toán quyền riêng tư, rồi hạn chế và kết luận.

---

## Slide 11 — Bài học (2:00)

[Chỉ ô 1.]

Bài học một: đừng chỉ dùng Loss, Confidence hay Entropy. Ví dụ rõ nhất là MOMENT trên dự báo. Loss chỉ đạt 2,2% tại ngưỡng 0,1% FPR. Tấn công tốt nhất đạt 97,7%. Một hệ thống chỉ kiểm bằng loss sẽ tuyên bố mô hình gần như an toàn, trong khi gần như mọi bản ghi huấn luyện đều bị nhận diện.

[Chỉ ô 2.]

Bài học hai: phải chạy nhiều họ. LiRA hoặc RMIA thắng ở 8 trên 10 cấu hình. Temporal thắng ở cả hai bài toán của iTransformer. Không có một tấn công dominant.

[Chỉ ô 3.]

Bài học ba: TPR phải đi kèm kiểm tra phân phối điểm số. Với iTransformer phân loại, 97,2% member có điểm loss bão hòa đúng cực trị, và một phần non-member cũng nằm ở đúng giá trị đó. Hạ ngưỡng một chút là FPR nhảy lên hơn 20%. Vì vậy AUC vẫn cao, khoảng 0,90, nhưng TPR tại 1% FPR bằng 0. AUC ở đây không đo được sức tấn công ở vùng an toàn.

Trường hợp tương tự là TCN dự báo với RMIA-online: điểm số suy biến, TPR không phản ánh khả năng bảo vệ của TCN. Ở cấu hình đó nhóm đọc qua LiRA, 13,8%.

[Chỉ ô 4.]

Bài học bốn: mô hình lớn và bài toán dự báo trên dữ liệu y tế cần được bảo vệ rồi đo lại. Hướng nhóm nêu là DP-SGD, nhiễu đầu ra, giới hạn số truy vấn, và dừng huấn luyện sớm. Nhóm chưa chạy các biện pháp này; đây là việc cần làm tiếp, không phải kết quả đã có.

**Câu nhớ:** một phép kiểm tra đơn lẻ có thể kết luận “an toàn” một cách sai.

---

## Slide 12 — Hạn chế và kết luận (1:30)

[Cột trái, không đọc hết từng gạch đầu dòng.]

Bốn hạn chế em muốn hội đồng nghe.

Mỗi mô hình chỉ huấn luyện một lần, seed 42. Bootstrap chỉ lấy lại trên tập đánh giá, không phản ánh phương sai do huấn luyện. Những khác biệt nhỏ giữa mô hình, ví dụ TCN và MOMENT ở phân loại, chưa kết luận được.

Chia theo bản ghi, có bệnh nhân trùng hai phía. Kết quả không áp nguyên cho mức người dùng. Tài liệu gần đây cho thấy mức người dùng thường còn dễ tấn công hơn.

Một tập dữ liệu, PTB-XL, và chưa thử phòng thủ.

Công thức huấn luyện chưa đồng nhất: số epoch, learning rate và utility khác nhau. RQ1 theo nghĩa “cùng generalization gap” chưa được thỏa đúng như câu hỏi đã đặt.

[Cột phải.]

Với các hạn chế đó, bốn kết luận nhóm vẫn giữ.

Một: kiến trúc không quyết định mức rò rỉ.

Hai: dự báo rò rỉ nhiều hơn phân loại trên cả năm mô hình. Chắc với LSTM, PatchTST và MOMENT; yếu hơn với TCN và iTransformer.

Ba: MOMENT tinh chỉnh toàn bộ là mô hình rò rỉ nhiều nhất. Ở dự báo, 97,7% tại 0,1% FPR. Ở phân loại, khoảng 9,9%, chỉ ngang nhóm TCN và iTransformer. Phần chênh này còn đan với kích thước mô hình và gap, nên em không quy riêng cho việc tinh chỉnh.

Bốn: cần nhiều họ tấn công. Loss một mình là không đủ.

Việc tiếp theo: nhiều seed, đối chứng cho MOMENT — huấn luyện từ đầu và đóng băng encoder — chia theo bệnh nhân, thêm tập khác như UCI HAR và ETTh1, rồi đánh giá DP-SGD.

**Câu nhớ:** kết luận chính đứng được; độ lớn của một số chênh lệch thì chưa.

---

## Slide 13 — Phân công (0:30)

[Mỗi người gật hoặc nói một câu khi tới tên. Không đọc lại pipeline.]

Bốn thành viên chia theo họ mô hình, mỗi người làm trọn pipeline: huấn luyện, tấn công và phân tích.

Bùi Lê Hoài An phụ trách LSTM và TCN. Trần Trọng Hoàng phụ trách PatchTST. Cao Anh Tuấn phụ trách iTransformer. Lê Thành Ngọc phụ trách MOMENT.

Cả nhóm dùng chung PTB-XL, cùng cách chia seed 42 và cùng bộ đánh giá, nên các con số so sánh được với nhau.

---

## Slide 14 — Tài liệu

[Không đọc. Chỉ nếu được hỏi nguồn.]

Tài liệu đánh số theo thứ tự xuất hiện trên slide. [1] là Shokri, định nghĩa MIA. [2] là Carlini, LiRA và tiêu chuẩn low-FPR. [3] là PTB-XL. [12] là RMIA. [13] là MIA trên chuỗi thời gian, nguồn của Temporal Feature. Danh mục đầy đủ nằm trong báo cáo.

---

## Slide 15 — Cảm ơn và hỏi đáp

Nhóm em xin cảm ơn thầy cô đã nghe. Em dừng ở slide kết luận để tiện đối chiếu khi thầy cô hỏi. Nhóm sẵn sàng trả lời.

[Nếu bị hỏi chi tiết một mô hình, người phụ trách mô hình đó trả lời. Câu dưới đây là câu ngắn, khoảng 20–40 giây.]

---

## Hỏi đáp — câu ngắn

**Vì sao không dùng AUC?**
AUC cộng mọi ngưỡng, kể cả vùng báo nhầm rất nhiều. Hai tấn công có AUC gần bằng nhau vẫn có thể khác xa ở 0,1% FPR. Ví dụ PatchTST dự báo: LiRA và RMIA-online có AUC 0,963 và 0,966, nhưng TPR là 17,2% và 28,5%. iTransformer còn rõ hơn: AUC khoảng 0,90 mà TPR tại 1% FPR bằng 0 vì điểm số bão hòa.

**Shadow model là gì, kẻ tấn công có thực tế không?**
Shadow model là mô hình cùng họ, huấn luyện trên dữ liệu cùng phân phối, để ước lượng đầu ra khi một bản ghi có hoặc không nằm trong tập học. Nhóm dùng 8 shadow. Đây là giả định mạnh hơn hộp đen thuần, nhưng là giả định chuẩn của LiRA và RMIA. Loss, Confidence, Entropy không cần shadow — và chính chúng gần như mù ở phân loại.

**Online và offline khác gì?**
Biến thể online giả định kẻ tấn công biết bản ghi mục tiêu có nằm trong nhóm IN của shadow hay không. Vì vậy các số “tấn công tốt nhất” có RMIA-online là cận trên. Tấn công chính theo protocol, MOMENT dự báo vẫn 94,4% với LiRA.

**97,7% có phải do fine-tune MOMENT?**
Chưa kết luận được. MOMENT là mô hình lớn nhất và được tinh chỉnh toàn bộ. Nhóm không có đối chứng huấn luyện từ đầu hay đóng băng encoder. Ở phân loại, MOMENT chỉ khoảng 9,9%, khoảng tin cậy chồng với TCN. Rò rỉ cực đoan nằm ở bài toán dự báo.

**266 bệnh nhân trùng hai phía thì kết quả còn đúng không?**
Đúng trong protocol record-level mà nhóm đã chạy. Không suy ra nguyên mức một bệnh nhân. Johansson và cộng sự ghi nhận mức người dùng thường dễ tấn công hơn, nên con số này có thể là chặn dưới theo hướng đó, chứ không phải chặn trên.

**Vì sao iTransformer gap lớn mà LiRA yếu?**
Train accuracy gần 1, test khoảng 0,52. 97,2% member có loss bão hòa. Tấn công hộp đen trên xác suất gần như không còn ngưỡng nào vừa chặt vừa bắt được member. Temporal, vì nhìn quỹ đạo theo epoch, vẫn đạt khoảng 7,7% ở phân loại và 14% ở dự báo. Do utility kém, em xem iTransformer là chặn dưới cho các tấn công hộp đen, không phải đại diện cho một Transformer đã hội tụ tốt.

**Dự báo hơn phân loại có phải chỉ vì 50 epoch so với 20?**
Có thể góp một phần với LSTM, TCN và PatchTST. MOMENT cùng 10 epoch và iTransformer cùng 50 epoch cho cả hai bài toán, mà chiều “dự báo cao hơn” vẫn còn. Nên epoch không phải lời giải thích duy nhất.

**DP-SGD có làm mất độ chính xác nhiều không?**
Nhóm chưa đo. Em chỉ nói hướng: với MOMENT dự báo, rủi ro đủ lớn để việc đo lại sau DP-SGD, nhiễu đầu ra hoặc giới hạn truy vấn là cần thiết. Dừng sớm cũng đáng thử, vì với MOMENT rò rỉ tăng mạnh ở các epoch cuối, khi learning rate của OneCycleLR hạ dần.

**TPR 0,1% có ổn định không?**
Độ phân giải thấp. 0,1% của 7.636 non-member chỉ khoảng 7 mẫu nhận nhầm, nên khoảng tin cậy rộng. Một giá trị mà khoảng tin cậy còn chứa mức ngẫu nhiên 0,1% thì nhóm không coi là đã chứng minh rò rỉ.

**Temporal có công bằng không?**
Không, nếu mục tiêu là so kẻ tấn công hộp đen. Temporal cần nhãn và checkpoint. Nhóm báo cáo nó như một họ riêng, và không dùng nó để tuyên bố một mô hình “dễ tấn công hơn” khi các tấn công hộp đen thất bại — trường hợp iTransformer là ví dụ.
