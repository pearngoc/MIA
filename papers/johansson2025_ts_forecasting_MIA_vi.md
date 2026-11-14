# Rủi ro quyền riêng tư trong dự báo chuỗi thời gian: Suy luận thành viên ở cấp người dùng và cấp bản ghi

*(Privacy Risks in Time Series Forecasting: User- and Record-Level Membership Inference)*

**Tác giả:** Nicolas Johansson‡∗, Tobias Olsson‡∗, Daniel Nilsson†, Johan Östman†, Fazeleh Hoseini†
∗ Chalmers University of Technology; † AI Sweden
‡ Các tác giả này đóng góp ngang nhau.

**Nguồn:** arXiv:2509.04169v2 [cs.LG], ngày 12 tháng 2 năm 2026 (bài báo gốc được phát hành theo giấy phép CC BY 4.0)

*Bản dịch tiếng Việt của bài báo gốc (dịch bởi AI)*

> Công trình này được tài trợ bởi Vinnova, cơ quan đổi mới sáng tạo của Thụy Điển, theo các khoản tài trợ 2023-03000 và 2025-03066.

---

**Tóm tắt —** Các tấn công suy luận thành viên (membership inference attacks – MIA) nhằm xác định liệu một dữ liệu cụ thể có được dùng để huấn luyện một mô hình hay không. Mặc dù đã được nghiên cứu rộng rãi trên các mô hình phân loại, tác động của chúng đối với bài toán dự báo chuỗi thời gian (time series forecasting) vẫn gần như chưa được khám phá. Chúng tôi giải quyết khoảng trống này bằng cách giới thiệu hai tấn công mới: (i) một sự điều chỉnh của LiRA đa biến (multivariate LiRA) – một MIA tiên tiến nhất (state-of-the-art) ban đầu được phát triển cho các mô hình phân loại – sang bối cảnh dự báo chuỗi thời gian, và (ii) một phương pháp học đầu-cuối (end-to-end) mới có tên là tấn công Deep Time Series (DTS). Chúng tôi đánh giá các tấn công của mình trong các bối cảnh thực tế trên hai tập dữ liệu TUH-EEG và ELD, nhắm vào hai kiến trúc dự báo mạnh là LSTM và N-HiTS (hiện đại nhất), dưới cả mô hình đe dọa (threat model) cấp bản ghi và cấp người dùng. Kết quả cho thấy các mô hình dự báo dễ bị tổn thương, trong đó các tấn công cấp người dùng thường đạt được khả năng phát hiện hoàn hảo. Các phương pháp được đề xuất đạt hiệu năng mạnh nhất trong nhiều bối cảnh, thiết lập các đường cơ sở (baseline) mới cho việc đánh giá rủi ro quyền riêng tư trong dự báo chuỗi thời gian. Hơn nữa, mức độ dễ bị tổn thương tăng lên khi tầm dự báo (prediction horizon) dài hơn và quần thể huấn luyện nhỏ hơn, phản ánh các xu hướng đã quan sát được ở các mô hình ngôn ngữ lớn.

**Từ khóa —** Tấn công suy luận thành viên (Membership Inference Attacks), Chuỗi thời gian (Time Series), Kiểm toán quyền riêng tư (Privacy Auditing), Mô hình dự báo (Forecasting Models), Dữ liệu thời gian (Temporal Data)

---

## I. Giới thiệu

Việc tích hợp học máy (machine learning – ML) vào các ứng dụng thực tế đã mở rộng đáng kể cách các ngành công nghiệp sử dụng dữ liệu để ra quyết định. Đặc biệt, dự báo chuỗi thời gian được ứng dụng rộng rãi trong các lĩnh vực như y tế, tài chính và hạ tầng thông minh. Tuy nhiên, khi được triển khai trong các bối cảnh liên quan đến thông tin nhạy cảm, các mô hình này làm nảy sinh những mối lo ngại quan trọng về quyền riêng tư dữ liệu.

Các mô hình ML hiện đại đã được chứng minh là có khả năng ghi nhớ thông tin từ dữ liệu huấn luyện [1]. Các công trình gần đây cho thấy thông tin được ghi nhớ này, trong một số trường hợp, có thể bị kẻ tấn công trích xuất thông qua việc tương tác với mô hình [2], [3]. Những phát hiện này đã thúc đẩy sự quan tâm đến việc phát triển các phương pháp thực nghiệm để đánh giá mức độ thông tin nhạy cảm mà một mô hình có thể vô tình làm lộ ra, thường được gọi là kiểm toán quyền riêng tư (privacy auditing) [4].

Các tấn công suy luận thành viên (MIA) là khối xây dựng cơ bản nhất của kiểm toán quyền riêng tư. Mục tiêu của một MIA là suy luận xem một điểm dữ liệu cụ thể có thuộc tập huấn luyện của mô hình hay không. Mặc dù việc chỉ suy luận tư cách thành viên có thể không nhất thiết vi phạm quyền riêng tư, MIA có thể được dùng như một thành phần trong các tấn công tinh vi hơn, chẳng hạn tái tạo dữ liệu (data reconstruction) [5], [6]. Hơn nữa, một MIA thành công hàm ý một mức độ ghi nhớ (memorization) hoặc quá khớp (overfitting) [7], có thể làm lộ các điểm yếu trước những loại tấn công khác [8]. Ngoài ra, MIA đã tỏ ra hữu ích trong việc đánh giá các phương pháp xóa học máy (machine unlearning) [9], thiết lập các cận dưới cho đảm bảo quyền riêng tư của các thuật toán riêng tư vi phân (differentially-private) [10], và trong việc phát hiện nhiễm bẩn tập dữ liệu (dataset contamination) [11].

Mặc dù MIA đã được nghiên cứu rộng rãi trên nhiều lĩnh vực như phân loại ảnh [3], [12], [13], sinh văn bản [14]–[16], và mạng nơ-ron đồ thị [17], [18], các hệ quả của chúng đối với các mô hình dự báo chuỗi thời gian vẫn gần như chưa được khám phá. Một vài nghiên cứu đã khảo sát suy luận thành viên trong bối cảnh dữ liệu chuỗi thời gian tổng hợp (aggregated), trong đó kẻ tấn công cố gắng xác định liệu dữ liệu của một cá nhân có được đưa vào bản tóm tắt cấp nhóm đã công bố hay không [19], [20]. Ngược lại, công trình của chúng tôi tập trung vào các mô hình dự đoán được huấn luyện bằng dữ liệu chuỗi thời gian, trong đó câu hỏi đặt ra là một điểm dữ liệu cho trước có được dùng để huấn luyện mô hình hay không.

Trong các tập dữ liệu chuỗi thời gian, dữ liệu thường được thu thập liên tục từ từng người dùng hoặc thiết bị trong một khoảng thời gian. Để làm cho dữ liệu này phù hợp với các tác vụ học máy khác nhau, nó thường được phân đoạn thành các chuỗi có độ dài cố định. Quá trình này tạo ra nhiều đoạn chồng lấn hoặc kề nhau, tức là các bản ghi huấn luyện, xuất phát từ cùng một người dùng. Do đó, mỗi người dùng đóng góp nhiều điểm dữ liệu vào tập huấn luyện. Điều này tương tự trường hợp của một số tác vụ phân tích văn bản, trong đó một tài liệu đơn lẻ được chia thành các đơn vị nhỏ hơn như đoạn văn hoặc câu [21], [22]. Trong công trình này, chúng tôi phân biệt các MIA nhắm vào tư cách thành viên của một bản ghi đơn lẻ với các MIA nhắm vào tư cách thành viên ở cấp người dùng (nhiều bản ghi).

Ở mức sâu hơn, vẫn chưa rõ đơn vị phân tích phù hợp trong MIA nên là gì. Liệu việc đánh giá rủi ro quyền riêng tư ở cấp bản ghi riêng lẻ có đủ hay không, hay trọng tâm nên chuyển sang mức độ phơi lộ ở cấp người dùng, do nhiều bản ghi có thể cùng nhau tiết lộ sự hiện diện của một người dùng trong dữ liệu huấn luyện? Những mối quan ngại tương tự đã được nêu ra trong bối cảnh các mô hình ngôn ngữ lớn (large language models – LLM), nơi đại lượng quan tâm có thể thay đổi từ cấp chuỗi đến toàn bộ tập dữ liệu [23].

Các MIA nhắm vào dự báo chuỗi thời gian gần đây đã được nghiên cứu ở cấp bản ghi [24], cho thấy rằng các đặc trưng riêng của chuỗi thời gian, ví dụ tính mùa vụ (seasonality) và xu hướng (trend), có thể bị MIA khai thác. Mặc dù nghiên cứu này cung cấp những hiểu biết có giá trị, nó chưa phân tích toàn diện cách các mô hình dự báo khác nhau phản ứng với MIA, và bỏ qua các chi tiết quan trọng về hiệu năng của các mô hình này, ví dụ mức độ quá khớp và việc xử lý dữ liệu. Hơn nữa, đối với các mô hình phân loại, phương pháp ensemble của họ không đạt hiệu năng ngang bằng các MIA tiên tiến nhất gần đây như Likelihood Ratio Attack (LiRA) [12] hay Robust Membership Inference Attack (RMIA) [3]. Những tấn công này cũng có thể hiệu quả đối với dự báo chuỗi thời gian: phần mở rộng gần đây của LiRA sang LLM cho thấy cách các tương quan trong quỹ đạo mất mát (loss trajectory) tuần tự có thể được khai thác [25].

Bài báo này có các đóng góp sau:¹

- Chúng tôi đề xuất hai tấn công mới: một tấn công dựa trên thống kê (statistics-based) xây dựng trên LiRA đa biến, và một tấn công dựa trên bộ phân loại (classifier-based) mới, có tên Deep Time Series (DTS), đạt hiệu năng tiên tiến nhất trong bối cảnh cấp bản ghi.
- Chúng tôi cung cấp nghiên cứu có hệ thống đầu tiên về các MIA tiên tiến nhất trong bối cảnh dự báo chuỗi thời gian. Cụ thể, chúng tôi điều chỉnh hai tấn công suy luận thành viên tiên tiến nhất, LiRA và RMIA, vốn được phát triển cho các tác vụ phân loại, sang bối cảnh dự báo bằng cách kết hợp các tín hiệu đặc thù của chuỗi thời gian.
- Chúng tôi tiến hành các thí nghiệm có hệ thống trên các kiến trúc dự báo thực tế, các tập dữ liệu chuỗi thời gian thực tế, và hai mô hình đối thủ, nhắm vào cả MIA cấp bản ghi lẫn cấp người dùng.

¹ Mã nguồn có tại https://github.com/aidotse/LeakPro/tree/SaTML-time-series-mia.

---

## II. Các công trình liên quan

### A. Các tấn công suy luận thành viên

Các tấn công suy luận thành viên lần đầu được giới thiệu trong công trình đầu tiên [1] và từ đó đã được nghiên cứu rộng rãi [2], [3], [12], [13], [18], [26]. Chúng thường được chia thành các tiếp cận dựa trên thống kê và dựa trên bộ phân loại. Các phương pháp dựa trên thống kê [2], [3], [12], [18] tính các độ đo từ đầu ra của mô hình mục tiêu (target model) và các mô hình bóng phụ trợ (shadow model), vốn được huấn luyện để mô phỏng hành vi của mô hình mục tiêu. Dựa trên ý tưởng này, các biến thể tiên tiến nhất tinh chỉnh tấn công bằng cách hiệu chỉnh các ngưỡng theo độ khó của từng mẫu thông qua các mô hình bóng [3], [12], [18]. Tuy nhiên, một hạn chế then chốt là chúng thường không phân biệt được thành viên với không phải thành viên khi đầu ra của chúng quá giống nhau [13].

Ngược lại, các tấn công dựa trên bộ phân loại [1] huấn luyện một bộ phân loại riêng để phân biệt hành vi của mô hình mục tiêu trên các đầu vào thuộc tập huấn luyện và không thuộc tập huấn luyện. Để cải thiện khả năng phân biệt, một số tiếp cận huấn luyện bộ phân loại này trên các đặc trưng thu được bằng cách mô phỏng quá trình huấn luyện của mô hình mục tiêu ở các giai đoạn khác nhau [13], [26], cho phép tấn công phân biệt được các điểm dữ liệu có đầu ra mô hình mục tiêu tương tự nhau.

### B. Đại lượng quan tâm trong MIA

Phần lớn các tài liệu về MIA tập trung vào các bản ghi riêng lẻ. Tuy nhiên, trong một số bối cảnh, chẳng hạn tranh chấp bản quyền hoặc nhiễm bẩn tập dữ liệu, mục tiêu có thể là suy luận tư cách thành viên của một tập hợp (collection) các bản ghi [11], [27]. Chẳng hạn trong bối cảnh văn bản, các đại lượng quan tâm có thể bao gồm chuỗi, đoạn văn, tài liệu, hoặc thậm chí toàn bộ tập dữ liệu [23]. Các công trình trước đây thường áp dụng MIA cấp bản ghi rồi tổng hợp kết quả trên nhiều bản ghi để tạo ra một điểm số thành viên cho tập hợp đó. Cách tiếp cận này đã được chứng minh là hiệu quả hơn so với việc suy luận tư cách thành viên cho từng bản ghi riêng lẻ [21], [22].

Hầu hết các nghiên cứu cấp tập hợp hiện có đều giả định rằng kẻ tấn công có quyền truy cập trực tiếp vào một số bản ghi huấn luyện, một bối cảnh được gọi là suy luận thành viên cấp người dùng (user-level membership inference). Ngược lại, Kandpal và cộng sự [28] xem xét một bối cảnh suy luận người dùng (user inference), trong đó kẻ tấn công không có quyền truy cập vào các mẫu huấn luyện chính xác mà chỉ có các điểm dữ liệu khác được rút từ cùng phân phối. Trong công trình của chúng tôi, chúng tôi tập trung vào bối cảnh trước, tức suy luận thành viên cấp người dùng.

### C. MIA trên các mô hình dự báo chuỗi thời gian

MIA chủ yếu được nghiên cứu trong bối cảnh phân loại ảnh và sinh văn bản, trong khi việc áp dụng chúng cho các mô hình chuỗi thời gian vẫn chưa được khám phá đầy đủ. Tuy nhiên, các công trình gần đây đã tận dụng các thuộc tính đặc thù của chuỗi thời gian, như quỹ đạo mất mát, để thiết kế các tấn công dựa trên bộ phân loại [13], [26]. Chẳng hạn, Li và cộng sự [26] trích xuất các đặc trưng như độ dao động (fluctuation), tương quan (correlation) và tốc độ suy giảm (decline rate) từ quỹ đạo mất mát, sau đó được dùng để huấn luyện một bộ phân loại phân biệt giữa các mẫu thành viên và không phải thành viên. Tương tự, quỹ đạo mất mát của các chuỗi văn bản đã được dùng cùng với LiRA để tấn công các LLM, sử dụng hơn 400 mô hình bóng [25].

Công trình gần đây đã khảo sát các tấn công nhắm vào các mô hình dự báo chuỗi thời gian [24]. Đóng góp chính của họ là giới thiệu hai đặc trưng tấn công đặc thù của chuỗi thời gian: xu hướng và tính mùa vụ. Tính mùa vụ được nắm bắt bằng Biến đổi Fourier rời rạc 2D (2D Discrete Fourier Transform – 2D-DFT), trích xuất các thành phần tần số theo thời gian và theo các biến. Các đặc trưng bao gồm các hệ số Fourier của chuỗi dự đoán và chuỗi thực, cùng với hiệu L2 giữa chúng. Đối với xu hướng, một đa thức bậc thấp được khớp vào từng chuỗi, và các đặc trưng thu được là các hệ số đa thức cùng hiệu L2 giữa chúng.

---

## III. Kiến thức nền tảng

### A. Các tín hiệu dùng làm đặc trưng tấn công

Trong bối cảnh suy luận thành viên đối với các mô hình chuỗi thời gian, nhiều tín hiệu khác nhau có thể được trích xuất từ đầu ra của mô hình để phân biệt giữa các mẫu thuộc tập huấn luyện và không thuộc tập huấn luyện. Chúng tôi phác thảo các tín hiệu cốt lõi được khảo sát trong các công trình trước và trong nghiên cứu của chúng tôi.

#### 1) Các tín hiệu dựa trên sai số (Error-Based Signals)

Các độ đo dự báo tiêu chuẩn như Sai số bình phương trung bình (Mean Squared Error – MSE), Sai số tuyệt đối trung bình (Mean Absolute Error – MAE), và Sai số phần trăm tuyệt đối trung bình đối xứng (Symmetric Mean Absolute Percentage Error – SMAPE) định lượng sự chênh lệch giữa các chuỗi tương lai dự đoán và thực tế [29]. Trong khi MSE và MAE phụ thuộc thang đo (scale-dependent), SMAPE bất biến theo thang đo và bị chặn trong khoảng từ 0 đến 1.

#### 2) Các tín hiệu cấu trúc (Structural Signals)

Xu hướng và tính mùa vụ thường được dùng để đặc trưng hóa cấu trúc thời gian ở mức cao của dữ liệu chuỗi thời gian. Koren và cộng sự [24] tận dụng các thành phần này để nắm bắt các mẫu thời gian rộng hơn cho các tấn công suy luận thành viên, như đã thảo luận ở Mục II-C.

#### 3) Các tín hiệu dựa trên biểu diễn (Representation-Based Signals)

TS2Vec [30] cung cấp một khung học tương phản (contrastive learning) tạo ra các embedding đa thang đo phong phú của chuỗi thời gian. Sử dụng một bộ mã hóa (encoder) dùng chung và một hàm mất mát tương phản phân cấp (hierarchical contrastive loss), TS2Vec học các biểu diễn có thể khái quát hóa qua nhiều tác vụ. Trong bối cảnh tấn công của chúng tôi, khoảng cách L2 giữa các embedding TS2Vec của chuỗi dự đoán và chuỗi sự thật nền (ground-truth) đóng vai trò là một đặc trưng phân biệt.

### B. Dự báo chuỗi thời gian

Chúng tôi xét một tập dữ liệu gồm các mẫu chuỗi thời gian được thu thập từ $U$ thực thể (entity) khác nhau, mỗi thực thể có thể có độ dài khác nhau. Gọi $T_u$ là chuỗi thời gian gắn với thực thể $u \in [U]$. Từ $T_u$, chúng tôi xây dựng $n_u$ bản ghi huấn luyện

$$D_u = \{(X_{u,1}, Y_{u,1}), \ldots, (X_{u,n_u}, Y_{u,n_u})\}.$$

Mỗi bản ghi $j \in [n_u]$ gồm một chuỗi đầu vào

$$X_{u,j} = [x_{t_j-L+1}, \ldots, x_{t_j}] \in \mathbb{R}^{M \times L},$$

trong đó $L$ là độ dài cửa sổ nhìn lại (look-back window), $M$ là số biến, và $t_j$ ký hiệu chỉ số của quan sát cuối cùng trong chuỗi đầu vào, tức bước thời gian tại điểm tách huấn luyện–nhãn, và một chuỗi đầu ra (nhãn đích)

$$Y_{u,j} = [y_{t_j+1}, \ldots, y_{t_j+H}] \in \mathbb{R}^{M \times H},$$

trong đó $H$ là tầm dự báo. Toàn bộ tập dữ liệu được cho bởi

$$D = \bigcup_{u \in [U]} D_u.$$

Mục tiêu là huấn luyện một mô hình $f_\theta$, tham số hóa bởi $\theta$, để dự đoán $H$ bước thời gian tương lai từ một chuỗi $X \in \mathbb{R}^{M \times L}$, tức là

$$\hat{Y} = f_\theta(X) \in \mathbb{R}^{M \times H}.$$

Chúng tôi gọi quá trình huấn luyện (ngẫu nhiên) là $\mathcal{T}$, sao cho $\theta = \mathcal{T}(D_{\text{train}})$, trong đó $D_{\text{train}} \subseteq D$.

---

## IV. Mô hình đe dọa và thiết lập

Chúng tôi áp dụng cách phát biểu MIA dựa trên trò chơi (game-based) [2], [8], mô hình hóa tấn công như một trò chơi đối kháng giữa một bên thách thức (challenger) và một kẻ tấn công (attacker). Chúng tôi giả định rằng mỗi thực thể $u \in [U]$ lấy mẫu dữ liệu của mình từ một phân phối $\pi_u$. Do đó, tập dữ liệu người dùng $D_u$ chứa các bản ghi $X_{u,j} \sim \pi_u$ với $j \in [n_u]$. Chúng tôi ký hiệu một tập dữ liệu $D_u$, kích thước $n_u$, được lấy mẫu i.i.d. từ $\pi_u$ là $D_u \sim \pi_u^{n_u}$. Hơn nữa, chúng tôi ký hiệu $\pi$ là phân phối trên các bản ghi của tất cả người dùng, gọi là quần thể bản ghi (record population), mà kẻ tấn công có quyền truy cập lấy mẫu.

Để khởi tạo trò chơi MIA, bên thách thức lấy mẫu đồng đều ngẫu nhiên một tập $I$ người dùng $I_{\text{train}} \subset [U]$. Với mỗi $u \in I_{\text{train}}$, bên thách thức lấy mẫu một tập dữ liệu $D_u \sim \pi_u^{n_u}$ và nối chúng lại để tạo thành tập huấn luyện

$$D_{\text{train}} = \bigcup_{u \in I_{\text{train}}} D_u.$$

Ở đây, chúng tôi đặt $n_u = n$ cho mọi $u$. Tương tự, một tập kiểm định (validation) $D_{\text{val}}$ được xây dựng từ các người dùng trong $I_{\text{val}} \subset [U] \setminus I_{\text{train}}$. Các người dùng còn lại được gom vào $I_{\text{test}} = [U] \setminus (I_{\text{train}} \cup I_{\text{val}})$, và một tập kiểm tra (test) tương ứng $D_{\text{test}}$ được tạo theo cách tương tự. Cuối cùng, bên thách thức huấn luyện một mô hình $f_\theta$, với $\theta = \mathcal{T}(D_{\text{train}})$.

Kẻ tấn công có quyền truy cập hộp đen (black-box) vào mô hình $f_\theta$, tức là có thể thu được dự báo cho các truy vấn đầu vào tùy ý nhưng không có quyền truy cập vào các trọng số $\theta$. Hơn nữa, kẻ tấn công được giả định biết quy trình huấn luyện, bao gồm các siêu tham số và kiến trúc mô hình của mô hình mục tiêu.

Chúng tôi phân biệt MIA cấp bản ghi và cấp người dùng. Trong MIA cấp bản ghi, mục tiêu của kẻ tấn công là thiết kế một tấn công $\mathcal{A}$ để suy luận xem một bản ghi dữ liệu cho trước $(X, Y)$ có thuộc tập huấn luyện $D_{\text{train}}$ của $f_\theta$ hay không. Trường hợp cụ thể của trò chơi quyền riêng tư được minh họa ở Trò chơi 1. Đáng chú ý, kẻ tấn công không có quyền truy cập vào các trọng số $\theta$ mà chỉ có thể truy vấn mô hình $f_\theta$, và có thể sử dụng quần thể bản ghi $\pi$ trong tấn công.

**Trò chơi 1: Trò chơi suy luận thành viên cấp bản ghi**

*Đầu vào:* Phân phối quần thể $\pi$, số người dùng trong quần thể $U$, các phân phối quần thể người dùng $\{\pi_u\}_{u \in [U]}$, thuật toán huấn luyện $\mathcal{T}$, số người dùng dùng để huấn luyện $I$, kích thước tập dữ liệu $n$.
*Đầu ra:* Dự đoán $\tilde{b}$ của kẻ tấn công.

1. Lấy mẫu đồng đều ngẫu nhiên $I$ người dùng vào $I_{\text{train}}$ từ $[U]$
2. Tạo tập huấn luyện $D_{\text{train}} \leftarrow \bigcup_{u \in I_{\text{train}}} D_u$, $D_u \sim \pi_u^n$
3. Huấn luyện các tham số mô hình: $\theta \leftarrow \mathcal{T}(D_{\text{train}})$
4. Lấy mẫu một bản ghi thành viên $(X_1, Y_1) \sim D_{\text{train}}$
5. Lấy mẫu một bản ghi không phải thành viên $(X_0, Y_0) \sim D_{\text{test}}$
6. Lấy mẫu một bit ngẫu nhiên $b \sim \text{Bernoulli}(0.5)$
7. Bên thách thức gửi $(\mathcal{T}, f_\theta, (X_b, Y_b))$ cho kẻ tấn công
8. Kẻ tấn công suy luận tư cách thành viên $\tilde{b} \leftarrow \mathcal{A}(\mathcal{T}, \pi, f_\theta, (X_b, Y_b))$

Trong MIA cấp người dùng, được hình thức hóa ở Trò chơi 2, kẻ tấn công cố gắng xác định xem một tập hợp các bản ghi thuộc về một người dùng duy nhất có nằm trong tập huấn luyện hay không, qua đó suy luận tình trạng thành viên của người dùng đó. Lưu ý rằng, để thiết lập tấn công, kẻ tấn công có quyền truy cập vào quần thể bản ghi $\pi$ thay vì các quần thể người dùng $\{\pi_u\}_{u \in [U]}$, tức là không thể lấy mẫu từ phân phối của một người dùng cho trước.

**Trò chơi 2: Trò chơi suy luận thành viên cấp người dùng**

*Đầu vào:* Phân phối quần thể $\pi$, số người dùng trong quần thể $U$, các phân phối quần thể người dùng $\{\pi_u\}_{u \in [U]}$, thuật toán huấn luyện $\mathcal{T}$, số người dùng dùng để huấn luyện $I$, kích thước tập dữ liệu $n$, kích thước tập dữ liệu thử thách $n_c$.
*Đầu ra:* Dự đoán $\tilde{b}$ của kẻ tấn công.

1. Lấy mẫu đồng đều ngẫu nhiên $I$ người dùng vào $I_{\text{train}}$ từ $[U]$
2. Tạo tập huấn luyện $D_{\text{train}} \leftarrow \bigcup_{u \in I_{\text{train}}} D_u$, $D_u \sim \pi_u^n$
3. Huấn luyện các tham số mô hình: $\theta \leftarrow \mathcal{T}(D_{\text{train}})$
4. Lấy mẫu một người dùng thành viên $u_1 \sim I_{\text{train}}$
5. Lấy mẫu một người dùng không phải thành viên $u_0 \sim I_{\text{test}}$
6. Lấy mẫu một bit ngẫu nhiên $b \sim \text{Bernoulli}(0.5)$
7. Lấy mẫu một tập dữ liệu thử thách $\tilde{D}_{u_b} \sim \pi_{u_b}^{n_c}$ có kích thước $n_c \le n$
8. Bên thách thức gửi $(\mathcal{T}, f_\theta, \tilde{D}_{u_b})$ cho kẻ tấn công
9. Kẻ tấn công suy luận tư cách thành viên $\tilde{b} \leftarrow \mathcal{A}(\mathcal{T}, \pi, f_\theta, \tilde{D}_{u_b})$

---

## V. Phương pháp

Chúng tôi đề xuất hai tấn công cho các mô hình dự báo chuỗi thời gian: một tấn công dựa trên thống kê và một tấn công dựa trên bộ phân loại. Các tấn công được mô tả từ góc nhìn của các tấn công cấp bản ghi, và phần mở rộng của chúng sang các tấn công cấp người dùng được thảo luận ở Mục V-C. Cả hai tấn công đều dựa trên các mô hình bóng được huấn luyện trên các tập dữ liệu được tạo theo cùng cách với tập dữ liệu mục tiêu nhưng sử dụng các phép chia người dùng khác nhau. Chúng tôi ký hiệu mô hình bóng thứ $i$ là $\phi_i$.

### A. LiRA đa tín hiệu (Multi-Signal LiRA)

Như đã thảo luận ở Mục III-A, nhiều tín hiệu thống kê khác nhau có thể được trích xuất từ các dự báo chuỗi thời gian để dùng làm đặc trưng cho suy luận thành viên. Dựa trên điều này, chúng tôi đề xuất một biến thể đa tín hiệu của LiRA, lấy cảm hứng từ công trình trước sử dụng các điểm khả dĩ (likelihood score) từ nhiều phép tăng cường (augmentation) của cùng một bản ghi để cải thiện hiệu năng suy luận [12, Mục IV.C].

Gọi $S$ là tập các tín hiệu tấn công khả dụng. Với một bản ghi $(X, Y)$ cho trước, mô hình dự đoán $\hat{Y} = f_\theta(X)$, từ đó một tín hiệu được tính theo $s(Y, \hat{Y})$, $s \in S$. Lưu ý rằng một số tín hiệu trong $S$ có thể không sử dụng $\hat{Y}$, trong trường hợp đó nó đơn giản bị bỏ qua.

Đối với tấn công, chúng tôi huấn luyện $K$ mô hình bóng để mô phỏng mô hình mục tiêu $f_\theta$. Kẻ tấn công cho từng bản ghi đi qua cả mô hình mục tiêu $f_\theta$ lẫn từng mô hình bóng $f_{\phi_i}$, $i \in [K]$, để thu được $K+1$ dự đoán. Gọi $s(Y, f_{\phi_i}(X))$ là giá trị của tín hiệu $s$ được đánh giá trên đầu ra của mô hình bóng $i$. Chúng tôi tính trung bình và độ lệch chuẩn cho từng loại tín hiệu trên các mô hình bóng. Các mô hình bóng được chia thành mô hình trong (in-model), tức các mô hình có $(X, Y)$ trong dữ liệu huấn luyện của chúng, và mô hình ngoài (out-model), tức các mô hình không có. Với mỗi nhóm, chúng tôi tính trung bình và độ lệch chuẩn của các giá trị tín hiệu, thu được $(\mu_v^s, \sigma_v^s)$ với $v \in \{\text{in}, \text{out}\}$ và $s \in S$ [12].

Chúng tôi mô hình hóa các tín hiệu cho mỗi nhóm $v$ theo phân phối Gauss đa biến, $s \sim \mathcal{N}(\mu_v, \Sigma_v)$, trong đó $\mu_v = (\mu_v^s)_{s \in S}$ và $\Sigma_v$ là ma trận hiệp phương sai thực nghiệm của các tín hiệu trong nhóm $v$. Một cách tiếp cận có nguyên tắc là ước lượng toàn bộ ma trận hiệp phương sai $\Sigma_v$, qua đó nắm bắt được sự phụ thuộc giữa các tín hiệu. Tuy nhiên, việc ước lượng ổn định các phần tử ngoài đường chéo của $\Sigma_v$ thường đòi hỏi số lượng lớn mô hình bóng; công trình trước trên các mô hình văn bản đã sử dụng hơn 400 mô hình để đạt được các ước lượng đáng tin cậy [25]. Do ràng buộc thực tế là chúng tôi chỉ có thể huấn luyện một số lượng hạn chế các mô hình bóng, chúng tôi xấp xỉ $\Sigma_v$ bằng một ma trận đường chéo chỉ chứa các phương sai $(\sigma_v^s)^2$ cho mỗi $s \in S$.

Đối với tấn công trực tuyến (online), trong đó các mô hình bóng có thể bao gồm dữ liệu mục tiêu, điểm số tỷ số khả dĩ (likelihood-ratio) thu được là

$$\text{score}(X, Y; f_\theta) = \frac{p\big(\mathbf{s}(X, Y; f_\theta) \mid \mathcal{N}(\mu_{\text{in}}, \Sigma_{\text{in}})\big)}{p\big(\mathbf{s}(X, Y; f_\theta) \mid \mathcal{N}(\mu_{\text{out}}, \Sigma_{\text{out}})\big)}, \tag{1}$$

trong đó $\mathbf{s}(X, Y; f_\theta) = \big(s(Y, f_\theta(X))\big)_{s \in S}$ là vectơ các giá trị tín hiệu cho bản ghi $(X, Y)$.

Đối với tấn công ngoại tuyến (offline), điểm số được định nghĩa là xác suất đuôi Gauss đa biến dưới phân phối của mô hình ngoài:

$$\text{score}(X, Y; f_\theta) = \Pr\big(Z \preceq \mathbf{s}(X, Y; f_\theta)\big), \quad Z \sim \mathcal{N}(\mu_{\text{out}}, \Sigma_{\text{out}}), \tag{2}$$

trong đó $\preceq$ ký hiệu bất đẳng thức theo từng phần tử. Khi $\Sigma_{\text{out}}$ là ma trận đường chéo, biểu thức này rút gọn thành tích của các xác suất đuôi đơn biến của từng tín hiệu.

### B. Tấn công Deep Time Series

Các MIA cho dự báo chuỗi thời gian thường dựa vào các đặc trưng thống kê được thiết kế thủ công, trích xuất từ đầu ra của mô hình, bao gồm cả tấn công đề xuất của chúng tôi ở Mục V-A. Các tiếp cận này đòi hỏi việc lựa chọn đặc trưng đặc thù miền và có thể không nắm bắt được các mẫu tinh tế hiện diện trong cấu trúc thời gian thô của các dự đoán và các sai số tương ứng. Để giải quyết hạn chế này, chúng tôi đề xuất tấn công DTS, một tiếp cận dựa trên bộ phân loại bỏ qua việc trích xuất tín hiệu thủ công. DTS học trực tiếp việc suy luận tư cách thành viên từ đầu ra của mô hình bằng cách huấn luyện một bộ phân loại học sâu. Quy trình được hình thức hóa trong Thuật toán 1.

**Thuật toán 1: MIA Deep Time Series (DTS)**

*Đầu vào:* Mô hình mục tiêu $f_\theta$, tập dữ liệu phụ trợ $D_{\text{aux}}$, tỷ lệ lấy mẫu $f$, thuật toán huấn luyện $\mathcal{T}$, số mô hình bóng $K$, dữ liệu mục tiêu $(X, Y)$.
*Đầu ra:* Điểm số thành viên cho dữ liệu mục tiêu $(X, Y)$.

*Bước 1: Huấn luyện các mô hình bóng*
1. for $i = 1 \ldots K$ do
2.   Lấy mẫu một tập con huấn luyện $D_i$ từ $D_{\text{aux}}$
3.   Huấn luyện mô hình bóng $\phi_i \leftarrow \mathcal{T}(D_i)$
4. end for

*Bước 2: Xây dựng tập huấn luyện DTS $D_{\text{DTS}}$*

5. $D_{\text{DTS}} = \emptyset$
6. for $i = 1 \ldots K$ do
7.   Lấy mẫu một tỷ lệ $f$ từ $D_{\text{aux}}$ vào $\tilde{D}$
8.   for each $(\tilde{X}, \tilde{Y}) \in \tilde{D}$ do
9.     Tính dự đoán $\hat{Y} = f_{\phi_i}(\tilde{X})$
10.    Gán $\ell_i = 1$ nếu $\tilde{X} \in D_i$, ngược lại $\ell_i = 0$
11.    Thêm bộ $(\tilde{Y}, \hat{Y}, \ell_i)$ vào $D_{\text{DTS}}$
12.  end for
13. end for

*Bước 3: Huấn luyện bộ phân loại*

14. Huấn luyện $\theta_{\text{DTS}}$ bằng $D_{\text{DTS}}$

*Bước 4: Chấm điểm bản ghi mục tiêu*

15. Tính dự đoán mục tiêu $\hat{Y} = f_\theta(X)$
16. Thu điểm số: $\text{score}(X, Y; f_\theta) = \theta_{\text{DTS}}(Y, \hat{Y})$

Tấn công dựa trên việc huấn luyện có giám sát một bộ phân loại học cách phân biệt thành viên với không phải thành viên dựa trên đầu ra của một mô hình dự báo. Để xây dựng tập huấn luyện, chúng tôi giả định kẻ tấn công có quyền truy cập vào một tập dữ liệu phụ trợ $D_{\text{aux}}$. Trong bối cảnh trực tuyến, $D_{\text{aux}}$ gồm toàn bộ $D$, trong khi ở bối cảnh ngoại tuyến, $D_{\text{aux}}$ không chứa bản ghi nào từ $D_{\text{train}} \cup D_{\text{val}} \cup D_{\text{test}}$. Từ $D_{\text{aux}}$, $K$ tập dữ liệu $D_i$ (có thể chồng lấn) được lấy mẫu và dùng để huấn luyện các mô hình bóng $\{\phi_i\}_{i=1}^K$ (Bước 1 trong Thuật toán 1).

Để xây dựng tập huấn luyện $D_{\text{DTS}}$, với mỗi mô hình bóng $\phi_i$, $i \in [K]$, chúng tôi lấy mẫu ngẫu nhiên một tỷ lệ $f$ các bản ghi từ $D_{\text{aux}}$. Mỗi bản ghi được lấy mẫu $(X, Y)$ được đánh giá bởi $\phi_i$ để tạo ra bộ ba $(Y, f_{\phi_i}(X), \ell_i)$, trong đó $\ell_i = \mathbb{1}\{(X, Y) \in D_i\} \in \{0, 1\}$ cho biết $(X, Y)$ có được dùng để huấn luyện $\phi_i$ hay không. Các bộ ba này được thêm vào $D_{\text{DTS}}$, mỗi phần tử thuộc $(\mathbb{R}^{M \times H} \times \mathbb{R}^{M \times H} \times \{0, 1\})$, tạo ra một tập dữ liệu có kích thước $Kf|D_{\text{aux}}|$.

Bộ phân loại thành viên $\theta_{\text{DTS}}$ sau đó được huấn luyện trên $D_{\text{DTS}}$ để ánh xạ các cặp $(Y, \hat{Y})$ thành một xác suất thành viên,

$$\theta_{\text{DTS}} : \mathbb{R}^{M \times H} \times \mathbb{R}^{M \times H} \longrightarrow [0, 1],$$

như thể hiện ở các Bước 2–3 của Thuật toán 1. Thông qua quá trình huấn luyện này, bộ phân loại học cách phân biệt giữa các bản ghi thuộc và không thuộc tập huấn luyện bằng cách tận dụng các phụ thuộc thời gian và các quy luật thống kê trong phân phối đồng thời của các dự đoán mô hình và kết quả thực.

Sau khi huấn luyện, $\theta_{\text{DTS}}$ có thể được dùng để tấn công mô hình mục tiêu $f_\theta$. Với một bản ghi mục tiêu $(X, Y)$, dự đoán của mô hình mục tiêu $\hat{Y} = f_\theta(X)$ được tính, sau đó $(Y, \hat{Y})$ được đưa vào $\theta_{\text{DTS}}$ để thu điểm số suy luận thành viên

$$\text{score}(X, Y; f_\theta) = \theta_{\text{DTS}}(Y, \hat{Y}), \tag{3}$$

biểu diễn xác suất ước lượng rằng $(X, Y)$ thuộc tập huấn luyện của mô hình mục tiêu. Kẻ tấn công sau đó so sánh xác suất này với một ngưỡng quyết định để xác định tư cách thành viên.

### C. Từ MIA cấp bản ghi sang MIA cấp người dùng

Hai tấn công được mô tả trong các mục trước đưa ra các điểm số suy luận thành viên cấp bản ghi, tức là xác suất ước lượng rằng một bản ghi cho trước từ tập kiểm toán (audit dataset) đã thuộc tập huấn luyện của mô hình mục tiêu. Việc mở rộng cách phát biểu này lên cấp người dùng đặt ra một thách thức bổ sung, vì mỗi người dùng thường đóng góp nhiều bản ghi. Do đó, các quyết định thành viên cấp người dùng phải xét đến nhiều điểm số cấp bản ghi từ cùng một người dùng, có thể được kết hợp theo nhiều cách khác nhau.

Có thể sử dụng nhiều chiến lược kết hợp, bao gồm bỏ phiếu đa số đơn giản, lấy trung bình hoặc trung vị của các điểm số cấp bản ghi, hoặc nhấn mạnh các ngoại lệ cực trị. Trong công trình này, chúng tôi áp dụng một cách tiếp cận dựa trên khả dĩ (likelihood-based), tính tích của các xác suất thành viên cấp bản ghi cho một người dùng cho trước. Dưới giả định đơn giản hóa rằng các dự đoán cấp bản ghi là độc lập, tích này tương ứng với khả dĩ tổng thể rằng tập dữ liệu của người dùng đã được đưa vào huấn luyện.

Một cách hình thức, một người dùng $u \in I_{\text{train}}$ đóng góp $D_u = \{(X_{u,1}, Y_{u,1}), \ldots, (X_{u,n}, Y_{u,n})\}$, gồm $n$ bản ghi huấn luyện. Trong một cuộc tấn công nhắm vào người dùng $u$, kẻ tấn công quan sát một tập hợp $n_c$ bản ghi $\tilde{D}_u \sim \pi_u^{n_c}$ mà các phần tử của nó có thể trùng với các phần tử trong $D$. Điểm số thành viên cấp người dùng cho người dùng $u$ khi đó được cho bởi

$$\text{score}_{\text{user}}(\tilde{D}_u; f_\theta) = \prod_{(X, Y) \in \tilde{D}_u} \text{score}(X, Y; f_\theta), \tag{4}$$

trong đó $\text{score}(X, Y; f_\theta)$ là xác suất thành viên cấp bản ghi cho một bản ghi dự báo đơn lẻ $(X, Y)$.

---

## VI. Thiết lập thực nghiệm

### A. Thiết lập dữ liệu và tiền xử lý

Chúng tôi đánh giá các tấn công của mình trên hai tập dữ liệu chuỗi thời gian thực tế từ các lĩnh vực nhạy cảm về quyền riêng tư, mỗi tập đòi hỏi tiền xử lý đặc thù theo miền. Tập thứ nhất gồm các tín hiệu điện não đồ (electroencephalogram – EEG), vốn mang tính sinh trắc học và có tính đặc trưng cao giữa các cá nhân, khiến chúng vốn dĩ có thể nhận dạng được [31]. Tập thứ hai, dữ liệu phụ tải điện, mặc dù không mang tính sinh trắc học, vẫn có thể tiết lộ nhiều thông tin riêng tư: công trình trước đã chỉ ra rằng các dấu vết tiêu thụ chi tiết có thể bị khai thác để suy ra các hoạt động của hộ gia đình, việc sử dụng thiết bị, các mẫu có người ở, và thậm chí tình trạng kinh tế–xã hội hoặc việc làm [32]. Chúng tôi mô tả chi tiết cả hai tập dữ liệu dưới đây.

#### 1) Temple University Hospital EEG Data Corpus (TUH-EEG v2.0.1)

Kho dữ liệu TUH-EEG là một tập dữ liệu lâm sàng chứa 26846 bản ghi EEG từ khoảng 1500 cá nhân, được thu thập trong giai đoạn từ năm 2002 đến 2017 [33]. Phần lớn các bản ghi EEG có 31 kênh, tuy nhiên một số chỉ có ít nhất 20 kênh. Hầu hết các tín hiệu được lấy mẫu ở 250 Hz (87%), với các phần nhỏ hơn được ghi ở 256 Hz (8,3%), 400 Hz (3,7%) và 512 Hz (1%).

Để đảm bảo chất lượng dữ liệu, chúng tôi chọn một tập con gồm 100 bản ghi từ 300 cá nhân đầu tiên có các bản ghi được lấy mẫu ở 250 Hz. Chúng tôi loại bỏ 60 giây đầu tiên của mỗi bản ghi vì thường chứa các nhiễu (artifact) như sóng hiệu chuẩn, và loại trừ các phiên có nhiễu dai dẳng hoặc các kênh phẳng. Theo công trình trước [24], chúng tôi chọn $M = 3$ đạo trình EEG vùng trán (FP1, FP2, F3) và cắt ngắn các bản ghi còn 15000 bước thời gian. Các bản ghi (record) được tạo bằng cửa sổ trượt với cửa sổ nhìn lại $L = 100$, bước nhảy (stride) bằng 1, và tầm dự báo $H = 20$, xem Mục III-B.

Tất cả các chuỗi thời gian được chuẩn hóa bằng phép co giãn theo khoảng tứ phân vị (interquartile range – IQR) thông qua RobustScaler của scikit-learn [34], mà chúng tôi thấy cho kết quả thực nghiệm tốt nhất nhờ tính bền vững trước các giá trị ngoại lai và sự biến thiên giữa các đối tượng.

#### 2) Tập dữ liệu Electricity Load Diagrams (ELD)

Tập dữ liệu ELD [35] là một benchmark được sử dụng rộng rãi cho dự báo chuỗi thời gian. Nó dựa trên các ghi nhận tiêu thụ điện của 370 hộ gia đình Bồ Đào Nha trong khoảng thời gian bốn năm (2011–2014), với độ phân giải lấy mẫu 15 phút. Các số đo được biểu diễn bằng kilowatt. Do không phải hộ gia đình nào cũng được theo dõi trong suốt toàn bộ giai đoạn, một số bản ghi chứa phần đệm số không (zero-padding) để làm cho mọi chuỗi thời gian có độ dài bằng nhau.

Để chuẩn bị dữ liệu, chúng tôi loại bỏ phần đệm số không và gộp độ phân giải thời gian lên 1 giờ bằng cách cộng bốn phép đo liên tiếp. Các hộ gia đình có ít hơn 15000 bước thời gian hợp lệ, hoặc có mức sử dụng trung bình dưới 200 kW/h hoặc trên 2000 kW/h, bị loại để tránh các tín hiệu không đáng tin cậy và các giá trị ngoại lai cực đoan. Sau đó chúng tôi chọn ngẫu nhiên 100 hộ gia đình cho các thí nghiệm. Mỗi chuỗi được cắt ngắn còn 15000 bước thời gian và chuẩn hóa bằng IQR scaling, theo cùng quy trình đã dùng cho dữ liệu EEG.

Tập dữ liệu này là một chuỗi thời gian đơn biến, tức $M = 1$, và các bản ghi được tạo bằng cửa sổ trượt, với bước nhảy bằng 1, cửa sổ nhìn lại $L = 100$ và tầm dự báo $H = 20$, như mô tả ở Mục III-B.

### B. Phân chia dữ liệu

Sau tiền xử lý, mỗi tập dữ liệu chứa dữ liệu chuỗi thời gian từ $I = 100$ cá nhân. Với mỗi cá nhân, thao tác cửa sổ cho ra 14881 điểm dữ liệu. Chúng tôi áp dụng chiến lược phân chia theo cá nhân để đảm bảo dữ liệu của không cá nhân nào xuất hiện đồng thời trong cả tập huấn luyện và tập kiểm tra. Cụ thể, chúng tôi chia 100 cá nhân thành tập huấn luyện $D_{\text{train}}$, tập kiểm định $D_{\text{val}}$ và tập kiểm tra $D_{\text{test}}$, mỗi tập gồm 20 cá nhân. Chúng tôi nhấn mạnh rằng chúng tôi sử dụng tập kiểm định để dừng sớm (early stopping) nhằm hạn chế mức độ quá khớp, do đó làm cho bối cảnh tấn công khó khăn hơn. 40 cá nhân còn lại tạo thành tập dữ liệu phụ trợ $D_{\text{aux}}$ và được kẻ tấn công dùng để huấn luyện các mô hình bóng. Trong quá trình tấn công, chúng tôi sử dụng $D_{\text{train}} \cup D_{\text{test}}$ để lấy mẫu các bản ghi cần suy luận tư cách thành viên. Trong bối cảnh trực tuyến, mỗi mô hình bóng được huấn luyện trên một tập con 50% cá nhân được chọn ngẫu nhiên từ $D_{\text{train}} \cup D_{\text{test}}$ thay vì lấy mẫu từ $D_{\text{aux}}$ như trong bối cảnh ngoại tuyến. Đối với các tấn công cấp người dùng, chúng tôi đặt $\tilde{D}_u = D_u$, tức kẻ tấn công quan sát toàn bộ các bản ghi dữ liệu của người dùng trong quá trình tấn công. Chúng tôi xem xét các quan sát một phần trong các nghiên cứu cắt bỏ (ablation).

### C. Các kiến trúc mô hình mục tiêu

Chúng tôi đánh giá rủi ro suy luận thành viên trong dự báo chuỗi thời gian bằng hai kiến trúc học sâu tiêu biểu: Long Short-Term Memory (LSTM) [36] và Neural Hierarchical Interpolation for Time Series (N-HiTS) [37]. N-HiTS là một mô hình dự báo phân cấp, đa độ phân giải, mở rộng N-BEATS [38] bằng lấy mẫu đầu vào đa tốc độ (multi-rate input sampling) và nội suy đa thang đo (multi-scale interpolation), đạt độ chính xác và hiệu quả cao hơn các phương pháp trước đó trong dự báo tầm xa trong khi vẫn hỗ trợ đầu vào đa biến và đầu vào ngoại sinh (exogenous).

Bằng việc chọn hai mô hình này, chúng tôi bao quát được cả một đường cơ sở đã được thiết lập và một tiếp cận tiên tiến nhất trong mô hình hóa chuỗi thời gian. Cả hai đều được cài đặt bằng PyTorch và được cấu hình cho các chuỗi đa biến với độ dài đầu vào và tầm dự báo biến thiên. Các cấu hình và siêu tham số cụ thể được trình bày chi tiết dưới đây.

#### 1) LSTM

Mô hình LSTM là một mô hình chuỗi-sang-chuỗi đa biến tiêu chuẩn. Nó gồm hai tầng LSTM một chiều, mỗi tầng có 64 đơn vị ẩn. Mô hình nhận các chuỗi có độ dài $L$ với $M$ biến làm đầu vào (tức $X \in \mathbb{R}^{M \times L}$), trong đó số đặc trưng đầu vào được đặt bằng $M$. Trạng thái ẩn cuối cùng từ tầng LSTM thứ hai được đưa vào một tầng kết nối đầy đủ (fully connected) với $H \times M$ nơ-ron đầu ra, biểu diễn dự báo cho $H$ bước thời gian tiếp theo.

#### 2) N-HiTS

Với mô hình N-HiTS, chúng tôi xây dựng dựa trên cài đặt chính thức [39], áp dụng các sửa đổi tối thiểu để phù hợp với thiết lập thực nghiệm của mình. Cụ thể, chúng tôi loại bỏ các phụ thuộc vào biến ngoại sinh và điều chỉnh việc xử lý chiều đầu vào để tương thích với các tập dữ liệu đã tiền xử lý. Kiến trúc gồm ba ngăn xếp (stack), mỗi ngăn chứa một khối duy nhất với hai tầng kết nối đầy đủ gồm 512 đơn vị. Mỗi khối sử dụng hàm IdentityBasis để ánh xạ trực tiếp đầu ra thành các thành phần backcast và forecast, mà không mô hình hóa tường minh xu hướng hay tính mùa vụ. Chúng tôi áp dụng các siêu tham số mặc định do các tác giả gốc khuyến nghị: khởi tạo Xavier normal [40], không chia sẻ trọng số giữa các khối, không dropout hay batch normalization, và các hàm kích hoạt ReLU. Nội suy tuyến tính được dùng làm phương pháp pooling giữa các mức độ phân giải.

### D. Quy trình huấn luyện

Chúng tôi huấn luyện các mô hình mục tiêu bằng bộ tối ưu Adam với tốc độ học mặc định 0,001 và hàm mất mát MAE làm tiêu chí tối ưu hóa. Việc huấn luyện được thực hiện tối đa 50 epoch với dừng sớm dựa trên mất mát kiểm định, sử dụng độ kiên nhẫn (patience) 3 epoch. Kích thước batch 1024 được dùng cho mọi lần huấn luyện.

### E. Đánh giá mô hình mục tiêu

Để đánh giá hiệu năng của các mô hình mục tiêu, chúng tôi báo cáo bốn độ đo dự báo: MSE, MAE, SMAPE, và độ lệch chuẩn hóa (normalized deviation – ND) [38]. MSE và MAE là các độ đo theo điểm tiêu chuẩn, trong khi SMAPE và ND đo sai số tương đối so với độ lớn của các giá trị thực. Với phân chia dữ liệu theo cá nhân và phương sai đáng kể giữa các đối tượng, các độ đo nhạy với thang đo này cung cấp các chỉ báo có nhiều thông tin hơn về quá khớp và khả năng khái quát hóa. Một cách hình thức, với tầm dự báo $H$ và $M$ biến mục tiêu, SMAPE và ND được định nghĩa như sau:

$$\text{SMAPE}(Y, \hat{Y}) = \frac{1}{HM} \sum_{i=1}^{H} \sum_{j=1}^{M} \frac{|Y_{i,j} - \hat{Y}_{i,j}|}{|Y_{i,j}| + |\hat{Y}_{i,j}|} \tag{5}$$

và

$$\text{ND}(Y, \hat{Y}) = \frac{\sum_{i=1}^{H} \sum_{j=1}^{M} |Y_{i,j} - \hat{Y}_{i,j}|}{\sum_{i=1}^{H} \sum_{j=1}^{M} |Y_{i,j}|}. \tag{6}$$

### F. Cài đặt các tấn công

Chúng tôi đánh giá hiệu năng tấn công bằng ba đường cơ sở tiên tiến nhất, cùng với các tấn công đề xuất của chúng tôi ở Mục V, sử dụng 64 mô hình bóng cho mọi cấu hình. Các cách cài đặt cụ thể của chúng trong thiết lập thực nghiệm của chúng tôi được mô tả dưới đây.

Tất cả các tấn công đều tận dụng các tín hiệu đã giới thiệu ở Mục III-A. Đối với một số tấn công, tín hiệu nền phải không bị chặn (unbounded); để đáp ứng điều này, chúng tôi định nghĩa một phiên bản SMAPE được co giãn lại (rescaled):

$$\text{rSMAPE} = \log\left(\frac{\text{SMAPE}(Y, \hat{Y})}{1 - \text{SMAPE}(Y, \hat{Y})}\right). \tag{7}$$

#### 1) Baseline 1 (LiRA)

Chúng tôi thực hiện nhiều phiên bản của tấn công LiRA [12], thay thế logit được co giãn lại (rescaled logit) trong công thức gốc bằng từng tín hiệu đã giới thiệu ở Mục III-A và trong (7). Các biến thể này được đánh giá trên mọi tập dữ liệu và mô hình mục tiêu ở cả bối cảnh trực tuyến lẫn ngoại tuyến. Phương sai cho các phân phối trong (in) và ngoài (out) được ước lượng theo phương pháp phương sai theo từng mẫu (per-example variance) [12, Mục VI.B].

#### 2) Baseline 2 (RMIA)

RMIA [3] so sánh tỷ số khả dĩ giữa một mô hình được huấn luyện với bản ghi mục tiêu và một mô hình trong đó bản ghi đó được thay bằng một bản ghi quần thể ngẫu nhiên, sử dụng xác suất đuôi của nó làm điểm số kiểm định. Các cách cài đặt tấn công sử dụng sáu tín hiệu: MSE, MAE, SMAPE, Trend, Seasonality và TS2Vec, được đánh giá ở cả bối cảnh trực tuyến lẫn ngoại tuyến trên mọi tập dữ liệu và mô hình mục tiêu. Vì rSMAPE không bị chặn nên nó bị loại khỏi RMIA. Đối với RMIA, vốn hoạt động trên các xác suất, chúng tôi áp dụng một ánh xạ giảm đơn điệu

$$g\big(s(X, Y; f_\theta)\big) = \frac{1}{1 + s(X, Y; f_\theta)}$$

để chuyển các giá trị tín hiệu thành các điểm số xác suất. RMIA cũng đòi hỏi lấy mẫu từ quần thể để ước lượng đầu ra trung bình của mô hình bóng; vì vậy, chúng tôi rút các bản ghi từ tập dữ liệu sau khi loại trừ tất cả các mẫu huấn luyện, kiểm tra và kiểm định. Chúng tôi đặt $\gamma = 1{,}0$ và $\alpha = 1/3$ cho bối cảnh ngoại tuyến [3, Phụ lục B.2.2].

#### 3) Baseline 3 (Tấn công Ensemble)

Shachor và cộng sự [41] giới thiệu một tấn công Ensemble, giả định rằng kẻ tấn công có quyền truy cập vào một tập dữ liệu gồm các thành viên trong (in-member; các điểm dữ liệu huấn luyện của mục tiêu) cùng với các không phải thành viên. Phương pháp huấn luyện nhiều bộ phân loại nhỏ, chuyên biệt trên các tập con dữ liệu rời nhau, mỗi tập con cân bằng với 50% thành viên và 50% không phải thành viên, nhằm cải thiện suy luận thành viên. Với mỗi tập con, nhiều bộ phân loại được huấn luyện trên đầu ra của mô hình mục tiêu, hoặc các đặc trưng dẫn xuất, được đánh giá, và bộ tốt nhất được giữ lại; việc lặp lại điều này trên nhiều tập con cho ra một ensemble mà điểm số của nó cho một điểm dữ liệu là tỷ lệ các mô hình dự đoán điểm đó là thành viên. Cách tiếp cận này được Koren và cộng sự [24] áp dụng, những người đầu tiên nghiên cứu dự báo chuỗi thời gian trong bối cảnh này. Trong công trình của chúng tôi, chúng tôi sử dụng các tham số tấn công giống [24], áp dụng phép tổng hợp trung bình trong ensemble như ở [41], và chọn mô hình tốt nhất dựa trên hiệu năng ROC-AUC. Cụ thể, chúng tôi đặt số lần thực thi hoàn chỉnh của tấn công trên các bản ghi ngẫu nhiên khác nhau là năm, số lần lặp huấn luyện/kiểm tra cho mỗi cặp tập con là ba, kích thước tập con là 50, và số tổ hợp tập con thành viên–không phải thành viên cố định cho mỗi thực thể là chín.

Tiếp theo, chúng tôi xét các tấn công được đề xuất ở Mục V.

#### 4) Multi-Signal LiRA

Chúng tôi áp dụng tấn công Multi-Signal LiRA (Mục V-A) cho mọi tập dữ liệu và mô hình mục tiêu ở cả hai bối cảnh đánh giá trực tuyến và ngoại tuyến. Vectơ tín hiệu gồm MSE, MAE, rSMAPE, Trend, Seasonality và TS2Vec. Chúng tôi giả định ma trận hiệp phương sai đường chéo và ước lượng từng phần tử bằng phương pháp phương sai theo từng mẫu của [12, Mục VI.B].²

² Chúng tôi cũng đã thử nghiệm với ma trận hiệp phương sai đầy đủ, nhưng số lượng mô hình bóng hạn chế trong thiết lập của chúng tôi khiến việc ước lượng không ổn định. Trong khi [25] áp dụng một ý tưởng tương tự cho các LLM, ổn định việc ước lượng hiệp phương sai thông qua co rút (shrinkage) với hơn 400 mô hình bóng, chúng tôi cố ý giới hạn thiết lập của mình ở quy mô còn khả thi đối với nhiều tập dữ liệu, mô hình và biến thể tấn công được xét trong nghiên cứu này.

#### 5) Tấn công Deep Time Series

Tấn công DTS được áp dụng cho mọi tập dữ liệu và mô hình mục tiêu ở cả hai bối cảnh đánh giá trực tuyến và ngoại tuyến. Với mỗi cấu hình, chúng tôi xây dựng $D_{\text{DTS}}$ bằng cách lấy mẫu $f = 10\%$ từ tập dữ liệu dùng để tạo các tập dữ liệu bóng. Chúng tôi chọn $f$ nhỏ để chứng minh rằng phương pháp hoạt động mạnh ngay cả với các giá trị thấp; hiệu năng được kỳ vọng sẽ cải thiện khi $f$ tăng. Mỗi bộ phân loại DTS được huấn luyện bằng bộ tối ưu Adam, với dừng sớm (patience = 3) và tối đa 64 epoch.

Chúng tôi đánh giá hai kiến trúc bộ phân loại: một mô hình đơn giản dựa trên LSTM và InceptionTime [42], một kiến trúc dựa trên tích chập mà cùng với các biến thể của nó được xem là tiên tiến nhất cho phân loại chuỗi thời gian trên kho UCR [43], vốn là một tập hợp nhiều tập dữ liệu chuỗi thời gian [44]. Trái với công trình trước [42], vốn dựa vào một ensemble các bộ dự đoán, chúng tôi đơn giản hóa cách tiếp cận bằng cách sử dụng một bộ duy nhất.

---

## VII. Kết quả thực nghiệm

### A. Các mô hình mục tiêu

Bảng I báo cáo hiệu năng của các mô hình mục tiêu trên các tập dữ liệu TUH-EEG và ELD qua các phần chia huấn luyện, kiểm định và kiểm tra, với kết quả được lấy trung bình trên năm lần chạy để tính đến sự biến thiên do khởi tạo ngẫu nhiên và phân chia dữ liệu. Tính ngẫu nhiên bao gồm cả phép chia dữ liệu của mô hình mục tiêu lẫn việc chọn người dùng; trong mỗi lần chạy, một tập con người dùng mới được lấy mẫu (100 trong số 300 với TUH-EEG và 100 trong số 370 với ELD). Các khoảng chênh lệch hiệu năng tương tự giữa các phần chia ở cả hai kiến trúc cho thấy dừng sớm ngăn chặn quá khớp một cách hiệu quả. Trên tất cả các độ đo (MSE, MAE, SMAPE và ND), N-HiTS vượt trội hơn LSTM. Trên TUH-EEG, mức cải thiện lớn nhất của N-HiTS được quan sát ở MSE (giảm từ $4{,}15 \times 10^{-9}$ xuống $8{,}43 \times 10^{-10}$) và ND (từ 0,319 xuống 0,223) trên tập kiểm tra. Trên ELD, các cải thiện rõ rệt nhất ở MSE (từ $1{,}85 \times 10^{4}$ xuống $1{,}37 \times 10^{4}$) và SMAPE (từ 0,0754 xuống 0,0645). MSE và MAE có phương sai cao hơn giữa các lần chạy, trong khi SMAPE và ND, bất biến theo thang đo, ổn định hơn, khiến chúng phù hợp hơn cho việc so sánh giữa các tập dữ liệu.

**Bảng I:** Hiệu năng của các mô hình mục tiêu trên các tập dữ liệu TUH-EEG và ELD. Mỗi mục báo cáo giá trị trung bình ± độ lệch chuẩn qua năm lần chạy với các hạt giống ngẫu nhiên khác nhau cho các phần chia Huấn luyện, Kiểm định và Kiểm tra (các độ đo chưa co giãn).

| Tập dữ liệu | Mô hình | Độ đo | Huấn luyện | Kiểm định | Kiểm tra |
|---|---|---|---|---|---|
| TUH-EEG | LSTM | MSE | 6.68 × 10⁻¹⁰ ± 4.46 × 10⁻¹⁰ | 4.42 × 10⁻⁹ ± 4.87 × 10⁻⁹ | 4.15 × 10⁻⁹ ± 2.20 × 10⁻⁹ |
| TUH-EEG | LSTM | MAE | 1.28 × 10⁻⁵ ± 2.77 × 10⁻⁶ | 2.05 × 10⁻⁵ ± 5.29 × 10⁻⁶ | 2.30 × 10⁻⁵ ± 4.45 × 10⁻⁶ |
| TUH-EEG | LSTM | SMAPE | 0.247 ± 0.0285 | 0.271 ± 0.0186 | 0.285 ± 0.0111 |
| TUH-EEG | LSTM | ND | 0.198 ± 0.0323 | 0.296 ± 0.0590 | 0.319 ± 0.0063 |
| TUH-EEG | N-HiTS | MSE | 2.60 × 10⁻¹⁰ ± 1.47 × 10⁻¹⁰ | 6.97 × 10⁻¹⁰ ± 3.04 × 10⁻¹⁰ | 8.43 × 10⁻¹⁰ ± 5.10 × 10⁻¹⁰ |
| TUH-EEG | N-HiTS | MAE | 8.92 × 10⁻⁶ ± 1.43 × 10⁻⁶ | 1.49 × 10⁻⁵ ± 2.45 × 10⁻⁶ | 1.60 × 10⁻⁵ ± 2.46 × 10⁻⁶ |
| TUH-EEG | N-HiTS | SMAPE | 0.208 ± 0.0301 | 0.258 ± 0.0195 | 0.267 ± 0.0048 |
| TUH-EEG | N-HiTS | ND | 0.139 ± 0.0196 | 0.216 ± 0.0294 | 0.223 ± 0.0201 |
| ELD | LSTM | MSE | 1.47 × 10⁴ ± 6.00 × 10³ | 2.77 × 10⁴ ± 1.90 × 10⁴ | 1.85 × 10⁴ ± 6.65 × 10³ |
| ELD | LSTM | MAE | 65.8 ± 12.9 | 88.0 ± 25.0 | 73.8 ± 6.9 |
| ELD | LSTM | SMAPE | 0.0607 ± 0.0178 | 0.0668 ± 0.0226 | 0.0754 ± 0.0056 |
| ELD | LSTM | ND | 0.0984 ± 0.0177 | 0.116 ± 0.0315 | 0.123 ± 0.0089 |
| ELD | N-HiTS | MSE | 7.92 × 10³ ± 2.80 × 10³ | 1.95 × 10⁴ ± 1.29 × 10⁴ | 1.37 × 10⁴ ± 3.45 × 10³ |
| ELD | N-HiTS | MAE | 49.1 ± 9.6 | 69.5 ± 19.6 | 61.1 ± 10.9 |
| ELD | N-HiTS | SMAPE | 0.0471 ± 0.0164 | 0.0556 ± 0.0250 | 0.0645 ± 0.0088 |
| ELD | N-HiTS | ND | 0.0731 ± 0.0126 | 0.0926 ± 0.0306 | 0.102 ± 0.0143 |

### B. Kết quả tấn công

Hiệu năng của các tấn công khác nhau được trình bày ở Bảng II cho tập dữ liệu TUH-EEG và Bảng III cho tập dữ liệu ELD, cũng như ở Bảng IV. Theo các công trình trước [3], [12], [18], chúng tôi báo cáo tỷ lệ dương tính thật (true positive rate – TPR) tại một FPR cho trước (Bảng II và Bảng III) và điểm ROC-AUC ở Bảng IV. Để xem các đường cong ROC đầy đủ, chúng tôi tham khảo Phụ lục A. Cụ thể, với các tấn công cấp bản ghi, chúng tôi báo cáo TPR tại FPR bằng 0,1% và 0,01%, trong khi với các tấn công cấp người dùng, chúng tôi báo cáo TPR tại 0%. Tất cả các kết quả cấp người dùng tương ứng với tỷ lệ truy cập $g = 1$. Tất cả kết quả được lấy trung bình trên năm lần chạy độc lập, với cả giá trị trung bình và độ lệch chuẩn được báo cáo. Với mỗi loại tấn công, chúng tôi minh họa cấu hình mạnh nhất bằng chữ in đậm, và tấn công tốt nhất tổng thể cho từng độ đo được báo cáo được làm nổi bật bằng các ký hiệu màu tương ứng cho bối cảnh trực tuyến và ngoại tuyến.

#### 1) Độ nhạy của dữ liệu

Các tấn công nhìn chung hiệu quả hơn trên TUH-EEG so với ELD, một sự khác biệt có khả năng do nhiều yếu tố. Thứ nhất, các thí nghiệm EEG của chúng tôi dùng ba kênh đầu vào so với một kênh cho ELD, nên với cửa sổ nhìn lại và tầm dự báo cố định, MIA có nhiều thông tin đầu vào hơn. Thứ hai, như đã thảo luận ở Mục VI-A, các tín hiệu EEG mang tính đặc thù cá nhân cao, trong khi mức sử dụng điện, mặc dù cũng mang các mẫu có thể nhận dạng, ít đặc trưng hơn do các thói quen hằng ngày chồng lấn và hành vi dùng chung của các hộ gia đình. Cuối cùng, dữ liệu EEG có độ phân giải thời gian mịn hơn nhiều (4ms) so với ELD (1 giờ), cung cấp cho kẻ tấn công đầu vào chi tiết hơn nhiều. Cùng nhau, các yếu tố này giúp giải thích hiệu năng tấn công mạnh hơn được quan sát trên các mô hình huấn luyện bằng EEG, vì các mô hình dự báo vốn thích nghi với cấu trúc thời gian ổn định đặc thù của người dùng khi nó hiện diện.

#### 2) So sánh các chiến lược tấn công

Tấn công Ensemble [41] hoạt động kém dưới các độ đo nghiêm ngặt được xét, xếp gần cuối trên các tập dữ liệu và bối cảnh. Dữ liệu huấn luyện trên mỗi mô hình hạn chế và thiếu khả năng thích nghi của nó có lẽ giải thích cho hiệu năng yếu này.

RMIA [3] nổi bật trong một số kịch bản. Nó đạt hiệu năng cấp người dùng cao nhất trên TUH-EEG ở cả hai mô hình và là phương pháp mạnh nhất trong bối cảnh trực tuyến cho N-HiTS trên ELD. Ở cấp bản ghi, RMIA chi phối các tấn công ngoại tuyến cho N-HiTS trên TUH-EEG và cho mọi mô hình trên ELD. Tuy nhiên, trong bối cảnh cấp bản ghi trực tuyến, RMIA thường bị LiRA, Multi-Signal LiRA và DTS vượt qua. Xét ROC-AUC trong Bảng IV, RMIA nổi lên là người chiến thắng rõ ràng trong bối cảnh ngoại tuyến.

Với cả hai tập dữ liệu, LiRA hoạt động cạnh tranh ở cấp người dùng, đôi khi sánh ngang các kết quả tốt nhất, nhưng nhìn chung tụt lại sau các phương pháp mạnh nhất ở cấp bản ghi, đặc biệt trong bối cảnh ngoại tuyến. Trong bối cảnh trực tuyến, TS2Vec thường cho kết quả cấp bản ghi tốt nhất của LiRA, trong khi rSMAPE có xu hướng hoạt động tốt hơn ở chế độ ngoại tuyến. Không tín hiệu đơn lẻ nào chi phối trong mọi trường hợp, và Trend cùng SMAPE thường là yếu nhất. Một ví dụ đáng chú ý là TUH-EEG với N-HiTS trong bối cảnh trực tuyến, nơi mọi tín hiệu ngoại trừ Trend đều đạt TPR cấp người dùng hoàn hảo 100% tại FPR 0% trên mọi hạt giống. Trên ELD, các điểm số cấp bản ghi của LiRA thường bị DTS hoặc RMIA vượt qua, nhưng nó vẫn cạnh tranh ở cấp người dùng.

Multi-Signal LiRA do chúng tôi đề xuất cải thiện so với LiRA đơn tín hiệu tốt nhất trong nhiều bối cảnh trực tuyến cấp bản ghi nhưng hiếm khi vượt qua các phương pháp không thuộc họ LiRA có hiệu năng cao nhất. Tuy nhiên, xét về ROC-AUC, Bảng IV, nó đạt hiệu năng tốt nhất đối với mô hình mục tiêu LSTM. Trong bối cảnh ngoại tuyến, nó có thể kém hơn các tín hiệu đơn lẻ mạnh nhất, như rSMAPE và TS2Vec. Chúng tôi dự đoán rằng một LiRA đa biến đầy đủ, có khả năng mô hình hóa các tương quan giữa các tín hiệu, có thể cải thiện hiệu năng hơn nữa, vì các tín hiệu được dự đoán là có tương quan. Tuy nhiên, vì cài đặt hiện tại của chúng tôi mô hình hóa ma trận hiệp phương sai dạng đường chéo, do số lượng mô hình bóng hạn chế, các tương quan này không thể được khai thác đầy đủ.

Tấn công DTS đưa ra một giải pháp thay thế mạnh dựa trên học sâu, đạt một số TPR cấp bản ghi trực tuyến cao nhất trên cả hai tập dữ liệu, đặc biệt với kiến trúc InceptionTime. Chẳng hạn, trên TUH-EEG với N-HiTS, DTS đạt TPR 23,32% tại FPR 0,1% trong bối cảnh trực tuyến, vượt trội hơn hẳn mọi phương pháp khác với biên độ lớn. Xét ROC-AUC, nó đạt hiệu năng mạnh nhất đối với kiến trúc N-HiTS, lên tới 0,884. DTS kém cạnh tranh hơn trong bối cảnh ngoại tuyến, nơi RMIA thường chiếm ưu thế, nhưng vẫn đạt các kết quả đáng kể. Hiệu năng của nó phụ thuộc nhiều vào kiến trúc bộ phân loại suy luận thành viên, với InceptionTime thường vượt trội so với LSTM.

Trong Phụ lục B, chúng tôi trình bày các kết quả về hiệu năng tấn công khi các mô hình dự báo được huấn luyện bằng riêng tư vi phân.

#### 3) Tấn công ngoại tuyến so với trực tuyến

Trong mọi đánh giá, các tấn công trực tuyến luôn vượt trội so với các tấn công ngoại tuyến, với mức cải thiện đặc biệt lớn ở cấp người dùng. Khoảng cách này lớn hơn đáng kể so với những gì được báo cáo cho các phương thức dữ liệu khác [3], [12], [18], gợi ý rằng suy luận thành viên trong dự báo chuỗi thời gian có thể đặc biệt nhạy với các dịch chuyển phân phối. Một yếu tố có khả năng là cách phân chia theo cá nhân của chúng tôi. Với chỉ 100 người dùng đa dạng, các mẫu thời gian và mẫu đặc thù người dùng có thể tạo ra những khác biệt phân phối đáng kể giữa tập huấn luyện/kiểm tra và tập phụ trợ, trái với các phép chia cấp bản ghi thường được dùng trong công trình trước. Chúng tôi khảo sát thêm giả thuyết này ở Mục VII-C.

Với cả hai tập dữ liệu, các kết quả cấp bản ghi mạnh nhất trong bối cảnh trực tuyến thu được bởi LiRA và DTS, trong khi RMIA và LiRA chi phối ở cấp người dùng. Độ lớn của khoảng cách trực tuyến–ngoại tuyến có thể rất đáng kinh ngạc. Chẳng hạn, trên TUH-EEG với N-HiTS tại FPR = 0,01%, LiRA đạt TPR 2,64% trực tuyến so với chỉ 0,13% ngoại tuyến. Sự khác biệt còn rõ rệt hơn đối với DTS, đạt 15,81% trực tuyến so với 0,67% ngoại tuyến.

#### 4) Tấn công cấp người dùng so với cấp bản ghi

Kết quả của chúng tôi cho thấy một khoảng cách nhất quán giữa MIA cấp bản ghi và cấp người dùng; việc suy luận tình trạng thành viên của một người dùng dễ hơn đáng kể so với một bản ghi riêng lẻ. Điều này phù hợp với các công trình trước [21], [22] chủ trương các tấn công cấp người dùng, vốn tổng hợp các dự đoán qua nhiều bản ghi, cung cấp tín hiệu mạnh hơn cho suy luận thành viên. Quan trọng là, sự chênh lệch này làm nổi bật một cạm bẫy tiềm tàng trong đánh giá, vì chỉ xét các độ đo cấp bản ghi có thể đánh giá thấp đáng kể rủi ro quyền riêng tư tiềm tàng. Chẳng hạn, trong bối cảnh trực tuyến đối với mô hình N-HiTS trên tập dữ liệu TUH-EEG, LiRA chỉ đạt TPR 7,41% tại FPR 0,1% ở cấp bản ghi, nhưng 100% TPR tại FPR 0% ở cấp người dùng, cho thấy mức độ dễ bị tổn thương nghiêm trọng hơn nhiều khi xét tư cách thành viên ở cấp người dùng.

Một quan sát khác là phương sai lớn được thấy ở một số kết quả cấp người dùng. Như thể hiện ở Mục VII-C, phương sai này giảm khi số lượng cá nhân tăng, gợi ý sự ổn định được cải thiện trong các quần thể lớn hơn. Cần lưu ý rằng hiệu năng cấp người dùng được báo cáo tại FPR 0%, một ngưỡng nhạy cảm về mặt thống kê. Vì không cho phép dương tính giả nào, ngay cả những dao động nhỏ trong dự đoán của mô hình cũng có thể dẫn đến những thay đổi đáng kể trong TPR đo được. Bất chấp sự bất ổn này, chúng tôi vẫn báo cáo kết quả tại FPR 0% để làm nổi bật rằng một số tấn công đạt khả năng phát hiện cấp người dùng hoàn hảo (100%) trong bối cảnh nghiêm ngặt này.

**Bảng II:** Hiệu năng tấn công suy luận thành viên trên tập dữ liệu TUH-EEG. Chúng tôi báo cáo TPR cấp bản ghi tại các FPR cố định (0,1% và 0,01%) và TPR cấp người dùng tại FPR 0%. Kết quả được lấy trung bình trên năm lần chạy, hiển thị giá trị trung bình và độ lệch chuẩn. Với các tấn công có nhiều cấu hình, cấu hình tốt nhất được in đậm. Các ký hiệu màu chỉ ra tấn công tốt nhất trên từng độ đo trong bối cảnh trực tuyến và ngoại tuyến tương ứng. *(Lưu ý: định dạng in đậm và tô màu của bản gốc không được giữ lại trong bản Markdown này; mỗi ô gồm giá trị trung bình ± độ lệch chuẩn, đơn vị %.)*

| Tấn công | Tín hiệu/Mô hình | N-HiTS: Cấp bản ghi 0.1% FPR | N-HiTS: Cấp bản ghi 0.01% FPR | N-HiTS: Cấp người dùng 0% FPR | LSTM: Cấp bản ghi 0.1% FPR | LSTM: Cấp bản ghi 0.01% FPR | LSTM: Cấp người dùng 0% FPR |
|---|---|---|---|---|---|---|---|
| Ensemble | MSE | 0.00 ±0.00 | 0.00 ±0.00 | 6.00 ±7.35 | 0.00 ±0.00 | 0.00 ±0.00 | 2.00 ±2.45 |
| Ensemble | MAE | 0.00 ±0.00 | 0.00 ±0.00 | 8.00 ±8.12 | 0.00 ±0.00 | 0.00 ±0.00 | 7.00 ±5.10 |
| Ensemble | SMAPE | 0.00 ±0.00 | 0.00 ±0.00 | 6.00 ±2.00 | 0.00 ±0.00 | 0.00 ±0.00 | 5.00 ±5.48 |
| Ensemble | rSMAPE | 0.00 ±0.00 | 0.00 ±0.00 | 8.00 ±4.00 | 0.00 ±0.00 | 0.00 ±0.00 | 10.00 ±7.07 |
| Ensemble | Trend | 0.03 ±0.05 | 0.00 ±0.00 | 4.00 ±5.83 | 0.01 ±0.01 | 0.00 ±0.00 | 5.00 ±4.47 |
| Ensemble | Seasonality | 0.03 ±0.03 | 0.00 ±0.00 | 9.00 ±8.00 | 0.00 ±0.00 | 0.00 ±0.00 | 4.00 ±4.90 |
| Ensemble | TS2Vec | 0.00 ±0.00 | 0.00 ±0.00 | 15.00 ±14.83 | 0.00 ±0.00 | 0.00 ±0.00 | 1.00 ±2.00 |
| RMIA-Online | MSE | 3.28 ±2.02 | 1.38 ±1.98 | 100.0 ±0.00 | 1.90 ±1.92 | 0.16 ±0.24 | 32.00 ±26.94 |
| RMIA-Online | MAE | 3.40 ±1.91 | 1.48 ±1.93 | 100.0 ±0.00 | 2.02 ±2.12 | 0.36 ±0.37 | 41.00 ±25.38 |
| RMIA-Online | SMAPE | 1.70 ±1.57 | 0.46 ±0.53 | 100.0 ±0.00 | 1.40 ±2.27 | 1.06 ±2.11 | 36.00 ±23.54 |
| RMIA-Online | Trend | 1.01 ±1.43 | 0.38 ±0.68 | 58.00 ±21.12 | 1.19 ±1.90 | 0.58 ±1.11 | 10.00 ±7.07 |
| RMIA-Online | Seasonality | 2.08 ±1.56 | 0.74 ±1.09 | 100.0 ±0.00 | 1.49 ±1.84 | 0.56 ±0.90 | 30.00 ±21.68 |
| RMIA-Online | TS2Vec | 2.61 ±1.80 | 1.31 ±1.98 | 100.0 ±0.00 | 1.84 ±2.30 | 1.28 ±2.23 | 37.00 ±23.37 |
| RMIA-Offline | MSE | 1.24 ±2.25 | 0.00 ±0.00 | 42.00 ±30.59 | 1.51 ±1.24 | 0.66 ±1.32 | 17.00 ±14.35 |
| RMIA-Offline | MAE | 1.10 ±2.06 | 1.04 ±2.09 | 40.00 ±29.50 | 1.67 ±1.17 | 0.60 ±1.21 | 15.00 ±10.49 |
| RMIA-Offline | SMAPE | 2.20 ±1.58 | 0.00 ±0.00 | 27.00 ±18.60 | 1.24 ±1.39 | 0.01 ±0.01 | 12.00 ±6.78 |
| RMIA-Offline | Trend | 0.46 ±0.28 | 0.03 ±0.05 | 9.00 ±4.90 | 0.38 ±0.41 | 0.02 ±0.03 | 10.00 ±5.48 |
| RMIA-Offline | Seasonality | 2.37 ±2.19 | 0.97 ±1.45 | 14.00 ±10.68 | 1.87 ±1.90 | 0.73 ±1.39 | 8.00 ±8.12 |
| RMIA-Offline | TS2Vec | 0.29 ±0.59 | 0.00 ±0.00 | 41.00 ±25.38 | 1.94 ±1.27 | 0.00 ±0.00 | 19.00 ±12.41 |
| LiRA-Online | MSE | 4.02 ±2.81 | 0.19 ±0.24 | 100.0 ±0.00 | 3.05 ±2.35 | 2.27 ±2.03 | 22.00 ±10.77 |
| LiRA-Online | MAE | 6.73 ±2.60 | 1.35 ±1.79 | 100.0 ±0.00 | 3.43 ±2.47 | 2.58 ±2.19 | 28.00 ±15.03 |
| LiRA-Online | SMAPE | 5.04 ±2.20 | 1.11 ±1.10 | 100.0 ±0.00 | 3.19 ±2.37 | 1.93 ±2.08 | 34.00 ±12.41 |
| LiRA-Online | rSMAPE | 6.65 ±2.39 | 2.26 ±1.83 | 100.0 ±0.00 | 3.50 ±2.40 | 2.23 ±2.13 | 35.00 ±13.42 |
| LiRA-Online | Trend | 5.19 ±4.12 | 2.42 ±2.05 | 94.00 ±4.90 | 2.26 ±2.51 | 1.99 ±2.39 | 15.00 ±5.48 |
| LiRA-Online | Seasonality | 7.28 ±2.50 | 1.77 ±2.12 | 100.0 ±0.00 | 3.42 ±2.41 | 2.61 ±2.13 | 23.00 ±12.08 |
| LiRA-Online | TS2Vec | 7.41 ±0.97 | 2.64 ±1.49 | 100.0 ±0.00 | 3.57 ±2.19 | 2.57 ±2.19 | 31.00 ±9.70 |
| LiRA-Offline | MSE | 0.01 ±0.02 | 0.00 ±0.00 | 2.00 ±2.45 | 1.66 ±2.16 | 1.35 ±2.13 | 4.00 ±4.90 |
| LiRA-Offline | MAE | 0.07 ±0.10 | 0.00 ±0.00 | 6.00 ±8.00 | 1.97 ±2.35 | 1.80 ±2.29 | 6.00 ±3.74 |
| LiRA-Offline | SMAPE | 0.20 ±0.11 | 0.05 ±0.03 | 16.00 ±5.83 | 1.88 ±2.25 | 1.54 ±1.90 | 8.00 ±6.00 |
| LiRA-Offline | rSMAPE | 1.04 ±0.93 | 0.12 ±0.08 | 22.00 ±9.80 | 2.12 ±2.38 | 1.69 ±1.98 | 8.00 ±4.00 |
| LiRA-Offline | Trend | 0.55 ±0.57 | 0.13 ±0.14 | 12.00 ±8.72 | 1.84 ±2.14 | 1.46 ±1.98 | 3.00 ±4.00 |
| LiRA-Offline | Seasonality | 0.08 ±0.11 | 0.01 ±0.01 | 10.00 ±7.07 | 1.97 ±2.34 | 1.71 ±2.22 | 6.00 ±3.74 |
| LiRA-Offline | TS2Vec | 0.33 ±0.34 | 0.05 ±0.03 | 8.00 ±6.78 | 1.90 ±2.24 | 1.71 ±2.22 | 5.00 ±3.16 |
| Multi-Signal LiRA-Online | — | 9.11 ±2.86 | 2.71 ±2.88 | 100.0 ±0.00 | 3.79 ±2.51 | 2.99 ±2.35 | 32.00 ±8.72 |
| Multi-Signal LiRA-Offline | — | 0.12 ±0.16 | 0.01 ±0.00 | 10.00 ±8.94 | 2.12 ±2.46 | 1.88 ±2.35 | 5.00 ±3.16 |
| DTS-Online | LSTM | 23.32 ±6.65 | 15.62 ±7.22 | 96.00 ±4.90 | 2.97 ±3.75 | 2.15 ±2.69 | 30.00 ±23.45 |
| DTS-Online | InceptionTime | 22.39 ±7.80 | 15.81 ±7.35 | 90.00 ±6.32 | 2.49 ±4.99 | 1.70 ±3.39 | 20.00 ±19.24 |
| DTS-Offline | LSTM | 2.09 ±3.13 | 0.67 ±1.01 | 27.00 ±16.61 | 0.17 ±0.16 | 0.00 ±0.01 | 9.00 ±15.62 |
| DTS-Offline | InceptionTime | 1.12 ±1.45 | 0.64 ±1.28 | 22.00 ±23.79 | 0.23 ±0.29 | 0.05 ±0.10 | 5.00 ±7.75 |

**Bảng III:** Hiệu năng tấn công suy luận thành viên trên tập dữ liệu ELD. Chúng tôi báo cáo TPR cấp bản ghi tại các FPR cố định (0,1% và 0,01%) và TPR cấp người dùng tại FPR 0%. Kết quả được lấy trung bình trên năm lần chạy, hiển thị giá trị trung bình và độ lệch chuẩn. Với các tấn công có nhiều cấu hình, cấu hình tốt nhất được in đậm. Các ký hiệu màu chỉ ra tấn công tốt nhất trên từng độ đo trong bối cảnh trực tuyến và ngoại tuyến tương ứng. *(Lưu ý: định dạng in đậm và tô màu của bản gốc không được giữ lại trong bản Markdown này.)*

| Tấn công | Tín hiệu/Mô hình | N-HiTS: Cấp bản ghi 0.1% FPR | N-HiTS: Cấp bản ghi 0.01% FPR | N-HiTS: Cấp người dùng 0% FPR | LSTM: Cấp bản ghi 0.1% FPR | LSTM: Cấp bản ghi 0.01% FPR | LSTM: Cấp người dùng 0% FPR |
|---|---|---|---|---|---|---|---|
| Ensemble | MSE | 0.01 ±0.02 | 0.00 ±0.00 | 8.00 ±6.78 | 0.00 ±0.00 | 0.00 ±0.00 | 4.00 ±4.90 |
| Ensemble | MAE | 0.00 ±0.00 | 0.00 ±0.00 | 6.00 ±7.35 | 0.00 ±0.00 | 0.00 ±0.00 | 5.00 ±5.48 |
| Ensemble | SMAPE | 0.02 ±0.02 | 0.00 ±0.00 | 2.00 ±4.00 | 0.02 ±0.05 | 0.00 ±0.00 | 3.00 ±6.00 |
| Ensemble | rSMAPE | 0.00 ±0.00 | 0.00 ±0.00 | 3.00 ±4.00 | 0.00 ±0.00 | 0.00 ±0.00 | 0.00 ±0.00 |
| Ensemble | Trend | 0.01 ±0.02 | 0.00 ±0.00 | 5.00 ±4.47 | 0.00 ±0.00 | 0.00 ±0.00 | 2.00 ±2.45 |
| Ensemble | Seasonality | 0.00 ±0.00 | 0.00 ±0.00 | 9.00 ±9.70 | 0.00 ±0.00 | 0.00 ±0.00 | 6.00 ±3.74 |
| Ensemble | TS2Vec | 0.00 ±0.00 | 0.00 ±0.00 | 2.00 ±4.00 | 0.00 ±0.00 | 0.00 ±0.00 | 3.00 ±4.00 |
| RMIA-Online | MSE | 1.12 ±1.26 | 0.31 ±0.37 | 44.00 ±20.35 | 1.16 ±0.58 | 0.15 ±0.14 | 19.00 ±17.44 |
| RMIA-Online | MAE | 1.19 ±1.20 | 0.26 ±0.40 | 59.00 ±19.60 | 1.00 ±0.60 | 0.13 ±0.15 | 16.00 ±12.41 |
| RMIA-Online | SMAPE | 0.67 ±0.44 | 0.09 ±0.12 | 47.00 ±21.59 | 0.24 ±0.21 | 0.04 ±0.05 | 16.00 ±11.58 |
| RMIA-Online | Trend | 0.17 ±0.12 | 0.02 ±0.04 | 21.00 ±7.35 | 0.09 ±0.06 | 0.01 ±0.01 | 13.00 ±11.66 |
| RMIA-Online | Seasonality | 0.97 ±0.69 | 0.04 ±0.05 | 73.00 ±19.65 | 0.46 ±0.37 | 0.08 ±0.10 | 19.00 ±14.97 |
| RMIA-Online | TS2Vec | 0.79 ±0.50 | 0.03 ±0.05 | 64.00 ±30.23 | 0.40 ±0.47 | 0.10 ±0.19 | 18.00 ±11.66 |
| RMIA-Offline | MSE | 0.06 ±0.13 | 0.00 ±0.00 | 7.00 ±6.00 | 1.28 ±1.33 | 0.26 ±0.23 | 6.00 ±5.83 |
| RMIA-Offline | MAE | 0.50 ±0.66 | 0.00 ±0.01 | 8.00 ±6.78 | 1.29 ±1.44 | 0.03 ±0.06 | 5.00 ±6.32 |
| RMIA-Offline | SMAPE | 0.02 ±0.04 | 0.00 ±0.00 | 8.00 ±2.45 | 0.61 ±0.45 | 0.00 ±0.00 | 4.00 ±4.90 |
| RMIA-Offline | Trend | 0.34 ±0.15 | 0.03 ±0.06 | 6.00 ±5.83 | 0.16 ±0.09 | 0.02 ±0.03 | 6.00 ±5.83 |
| RMIA-Offline | Seasonality | 0.89 ±0.86 | 0.31 ±0.50 | 7.00 ±6.00 | 0.68 ±1.02 | 0.42 ±0.84 | 5.00 ±6.32 |
| RMIA-Offline | TS2Vec | 0.61 ±0.90 | 0.03 ±0.04 | 7.00 ±5.10 | 0.77 ±1.03 | 0.36 ±0.73 | 7.00 ±7.48 |
| LiRA-Online | MSE | 1.06 ±0.57 | 0.28 ±0.21 | 43.00 ±28.21 | 2.26 ±3.18 | 0.84 ±1.19 | 29.00 ±20.10 |
| LiRA-Online | MAE | 1.99 ±1.00 | 0.65 ±0.36 | 49.00 ±26.72 | 3.29 ±2.64 | 1.47 ±0.88 | 35.00 ±15.49 |
| LiRA-Online | SMAPE | 1.50 ±0.61 | 0.46 ±0.23 | 43.00 ±26.76 | 2.18 ±1.46 | 0.78 ±0.29 | 29.00 ±11.58 |
| LiRA-Online | rSMAPE | 1.94 ±0.79 | 0.59 ±0.26 | 53.00 ±29.43 | 3.10 ±1.75 | 0.93 ±0.51 | 38.00 ±12.08 |
| LiRA-Online | Trend | 1.34 ±0.29 | 0.45 ±0.20 | 50.00 ±19.49 | 1.17 ±0.72 | 0.31 ±0.32 | 27.00 ±19.65 |
| LiRA-Online | Seasonality | 2.02 ±0.85 | 0.76 ±0.37 | 49.00 ±26.72 | 3.46 ±3.07 | 1.43 ±1.14 | 34.00 ±13.56 |
| LiRA-Online | TS2Vec | 2.16 ±1.08 | 0.88 ±0.45 | 48.00 ±25.42 | 3.50 ±2.86 | 1.68 ±1.25 | 33.00 ±12.08 |
| LiRA-Offline | MSE | 0.07 ±0.07 | 0.01 ±0.01 | 4.00 ±5.83 | 0.03 ±0.03 | 0.00 ±0.00 | 3.00 ±4.00 |
| LiRA-Offline | MAE | 0.26 ±0.17 | 0.07 ±0.07 | 9.00 ±7.35 | 0.21 ±0.24 | 0.02 ±0.02 | 7.00 ±4.00 |
| LiRA-Offline | SMAPE | 0.35 ±0.23 | 0.09 ±0.06 | 10.00 ±7.07 | 0.18 ±0.14 | 0.05 ±0.06 | 8.00 ±7.48 |
| LiRA-Offline | rSMAPE | 0.59 ±0.27 | 0.13 ±0.10 | 12.00 ±9.27 | 0.34 ±0.26 | 0.07 ±0.04 | 9.00 ±4.90 |
| LiRA-Offline | Trend | 0.22 ±0.10 | 0.05 ±0.03 | 6.00 ±3.74 | 0.06 ±0.08 | 0.01 ±0.02 | 5.00 ±4.47 |
| LiRA-Offline | Seasonality | 0.30 ±0.18 | 0.08 ±0.09 | 10.00 ±8.37 | 0.24 ±0.18 | 0.07 ±0.09 | 8.00 ±5.10 |
| LiRA-Offline | TS2Vec | 0.52 ±0.36 | 0.26 ±0.23 | 9.00 ±5.83 | 0.41 ±0.32 | 0.10 ±0.06 | 7.00 ±5.10 |
| Multi-Signal LiRA-Online | — | 2.32 ±1.01 | 0.85 ±0.41 | 48.00 ±26.76 | 4.15 ±4.13 | 2.29 ±2.27 | 34.00 ±12.81 |
| Multi-Signal LiRA-Offline | — | 0.32 ±0.20 | 0.08 ±0.08 | 10.00 ±6.32 | 0.37 ±0.42 | 0.06 ±0.06 | 8.00 ±4.00 |
| DTS-Online | LSTM | 6.34 ±3.91 | 2.77 ±1.87 | 56.00 ±22.45 | 4.78 ±4.56 | 2.57 ±2.42 | 34.00 ±21.54 |
| DTS-Online | InceptionTime | 6.66 ±3.48 | 2.94 ±1.69 | 65.00 ±21.21 | 4.49 ±4.83 | 2.40 ±3.20 | 24.00 ±25.77 |
| DTS-Offline | LSTM | 0.57 ±0.39 | 0.15 ±0.18 | 13.00 ±12.88 | 0.76 ±0.85 | 0.26 ±0.39 | 9.00 ±5.83 |
| DTS-Offline | InceptionTime | 0.68 ±0.71 | 0.21 ±0.26 | 11.00 ±13.19 | 0.78 ±0.86 | 0.30 ±0.47 | 6.00 ±4.90 |

**Bảng IV:** Hiệu năng tấn công suy luận thành viên trên các tập dữ liệu TUH-EEG và ELD. Chúng tôi báo cáo ROC-AUC được lấy trung bình trên năm lần chạy, hiển thị giá trị trung bình và độ lệch chuẩn. Với các tấn công có nhiều cấu hình, cấu hình tốt nhất được in đậm. Các ký hiệu màu chỉ ra tấn công tốt nhất trong bối cảnh trực tuyến và ngoại tuyến tương ứng. *(Lưu ý: định dạng in đậm và tô màu của bản gốc không được giữ lại trong bản Markdown này.)*

| Tấn công | Tín hiệu/Mô hình | TUH-EEG N-HiTS | TUH-EEG LSTM | ELD N-HiTS | ELD LSTM |
|---|---|---|---|---|---|
| Ensemble | MSE | 0.546 ±0.0823 | 0.530 ±0.0420 | 0.504 ±0.0908 | 0.509 ±0.0764 |
| Ensemble | MAE | 0.532 ±0.0909 | 0.532 ±0.0525 | 0.503 ±0.0709 | 0.500 ±0.0767 |
| Ensemble | SMAPE | 0.491 ±0.0476 | 0.495 ±0.0312 | 0.496 ±0.0377 | 0.482 ±0.0711 |
| Ensemble | rSMAPE | 0.494 ±0.0454 | 0.499 ±0.0392 | 0.493 ±0.0385 | 0.484 ±0.0648 |
| Ensemble | Trend | 0.537 ±0.0291 | 0.506 ±0.0300 | 0.509 ±0.0622 | 0.496 ±0.0599 |
| Ensemble | Seasonality | 0.538 ±0.0911 | 0.529 ±0.0483 | 0.484 ±0.0905 | 0.502 ±0.0719 |
| Ensemble | TS2Vec | 0.555 ±0.1110 | 0.507 ±0.0795 | 0.477 ±0.0599 | 0.499 ±0.0750 |
| RMIA-Online | MSE | 0.744 ±0.0201 | 0.602 ±0.0302 | 0.646 ±0.0454 | 0.658 ±0.0362 |
| RMIA-Online | MAE | 0.775 ±0.0142 | 0.614 ±0.0384 | 0.665 ±0.0470 | 0.669 ±0.0322 |
| RMIA-Online | SMAPE | 0.747 ±0.0089 | 0.614 ±0.0233 | 0.628 ±0.0554 | 0.636 ±0.0207 |
| RMIA-Online | Trend | 0.610 ±0.0121 | 0.527 ±0.0122 | 0.581 ±0.0190 | 0.559 ±0.00678 |
| RMIA-Online | Seasonality | 0.778 ±0.0282 | 0.594 ±0.0593 | 0.660 ±0.0483 | 0.666 ±0.0312 |
| RMIA-Online | TS2Vec | 0.782 ±0.0173 | 0.609 ±0.0515 | 0.648 ±0.0414 | 0.654 ±0.0316 |
| RMIA-Offline | MSE | 0.724 ±0.0392 | 0.631 ±0.0395 | 0.592 ±0.0326 | 0.593 ±0.0466 |
| RMIA-Offline | MAE | 0.744 ±0.0443 | 0.637 ±0.0406 | 0.593 ±0.0515 | 0.580 ±0.0498 |
| RMIA-Offline | SMAPE | 0.665 ±0.0334 | 0.588 ±0.0265 | 0.600 ±0.0135 | 0.593 ±0.0358 |
| RMIA-Offline | Trend | 0.588 ±0.0203 | 0.551 ±0.0267 | 0.550 ±0.0534 | 0.535 ±0.0394 |
| RMIA-Offline | Seasonality | 0.706 ±0.0423 | 0.622 ±0.0262 | 0.558 ±0.0767 | 0.537 ±0.0699 |
| RMIA-Offline | TS2Vec | 0.748 ±0.0488 | 0.634 ±0.0416 | 0.607 ±0.0223 | 0.599 ±0.0360 |
| LiRA-Online | MSE | 0.811 ±0.0099 | 0.640 ±0.0333 | 0.672 ±0.0445 | 0.672 ±0.0519 |
| LiRA-Online | MAE | 0.816 ±0.0102 | 0.646 ±0.0306 | 0.682 ±0.0433 | 0.681 ±0.0444 |
| LiRA-Online | SMAPE | 0.796 ±0.0089 | 0.647 ±0.0186 | 0.660 ±0.0366 | 0.661 ±0.0366 |
| LiRA-Online | rSMAPE | 0.801 ±0.0100 | 0.650 ±0.0202 | 0.668 ±0.0361 | 0.666 ±0.0297 |
| LiRA-Online | Trend | 0.697 ±0.0204 | 0.552 ±0.0159 | 0.646 ±0.0169 | 0.602 ±0.0397 |
| LiRA-Online | Seasonality | 0.821 ±0.0112 | 0.646 ±0.0315 | 0.682 ±0.0437 | 0.680 ±0.0437 |
| LiRA-Online | TS2Vec | 0.822 ±0.0147 | 0.655 ±0.0374 | 0.669 ±0.0434 | 0.672 ±0.0495 |
| LiRA-Offline | MSE | 0.440 ±0.0284 | 0.445 ±0.0321 | 0.510 ±0.0689 | 0.461 ±0.0840 |
| LiRA-Offline | MAE | 0.495 ±0.0288 | 0.474 ±0.0355 | 0.523 ±0.0558 | 0.469 ±0.0780 |
| LiRA-Offline | SMAPE | 0.521 ±0.0196 | 0.486 ±0.0257 | 0.495 ±0.0146 | 0.461 ±0.0398 |
| LiRA-Offline | rSMAPE | 0.563 ±0.0213 | 0.518 ±0.0304 | 0.532 ±0.0149 | 0.488 ±0.0414 |
| LiRA-Offline | Trend | 0.470 ±0.0364 | 0.459 ±0.0548 | 0.492 ±0.0654 | 0.477 ±0.0632 |
| LiRA-Offline | Seasonality | 0.495 ±0.0305 | 0.476 ±0.0381 | 0.521 ±0.0568 | 0.469 ±0.0833 |
| LiRA-Offline | TS2Vec | 0.506 ±0.0182 | 0.480 ±0.0231 | 0.500 ±0.0331 | 0.464 ±0.0405 |
| Multi-Signal LiRA-Online | — | 0.847 ±0.0119 | 0.661 ±0.0312 | 0.703 ±0.0439 | 0.700 ±0.0464 |
| Multi-Signal LiRA-Offline | — | 0.492 ±0.0278 | 0.469 ±0.0378 | 0.517 ±0.0567 | 0.459 ±0.0812 |
| DTS-Online | LSTM | 0.884 ±0.0303 | 0.637 ±0.0378 | 0.754 ±0.0131 | 0.667 ±0.069 |
| DTS-Online | InceptionTime | 0.866 ±0.0389 | 0.597 ±0.0540 | 0.758 ±0.0137 | 0.658 ±0.081 |
| DTS-Offline | LSTM | 0.653 ±0.0288 | 0.529 ±0.0355 | 0.574 ±0.0395 | 0.549 ±0.0312 |
| DTS-Offline | InceptionTime | 0.660 ±0.0503 | 0.504 ±0.0507 | 0.56 ±0.0218 | 0.55 ±0.0265 |

### C. Nghiên cứu cắt bỏ (Ablation Study)

Chúng tôi tiến hành ba nghiên cứu cắt bỏ để xem xét cách tầm dự báo, kích thước quần thể và tỷ lệ truy cập ảnh hưởng đến hiệu năng tấn công. Hai nghiên cứu đầu sử dụng mô hình mục tiêu N-HiTS trên tập dữ liệu ELD, trong khi nghiên cứu thứ ba được thực hiện trên tập dữ liệu EEG, nơi các tấn công cấp người dùng đạt hiệu năng mạnh nhất. Trong mọi nghiên cứu, chúng tôi đánh giá ba tấn công trực tuyến tiêu biểu: LiRA với TS2Vec, Multi-Signal LiRA và DTS với InceptionTime.

#### 1) Tác động của tầm dự báo

Hình 1 báo cáo các TPR cấp bản ghi và cấp người dùng tại các FPR cố định cho các tầm dự báo khác nhau $H \in \{5, 10, 20, 40, 80\}$. Các tầm dự báo dài hơn nhìn chung cho TPR cấp bản ghi cao hơn, cho thấy sự rò rỉ thông tin tăng lên với các cửa sổ dự đoán lớn hơn. Xu hướng này nhất quán trên mọi tấn công. Ở cấp người dùng, DTS tiếp tục cải thiện với các tầm dự báo dài hơn, trong khi các tấn công dựa trên LiRA đạt đỉnh ở khoảng tầm dự báo 20 rồi giảm, có thể do suy giảm tín hiệu hoặc dịch chuyển phân phối dữ liệu nền. Việc hiệu năng tấn công tăng theo độ dài chuỗi trước đây đã được ghi nhận trong bối cảnh LLM [14], [45].

*[Hình 1: Ảnh hưởng của tầm dự báo lên các TPR cấp bản ghi (trái và giữa) và cấp người dùng (phải) cho ba cấu hình tấn công (LiRA-Online (TS2Vec), Multi-Signal LiRA-Online, DTS (InceptionTime)) đối với mô hình N-HiTS trên tập dữ liệu ELD.]*

#### 2) Tác động của số lượng cá nhân

Để xem xét ảnh hưởng của kích thước quần thể, chúng tôi thay đổi tổng số người dùng $I \in \{50, 100, 200\}$, giữ nguyên mọi tham số khác. Mỗi cấu hình được đánh giá trên năm hạt giống ngẫu nhiên để tính đến sự biến thiên do phân chia dữ liệu. Các tổng số này tương ứng với $D_{\text{train}} \cup D_{\text{test}}$ có lần lượt 20, 40 và 80 cá nhân. Như thể hiện ở Hình 2, cả TPR cấp bản ghi và cấp người dùng đều giảm khi số cá nhân tăng, gợi ý rằng các quần thể lớn hơn làm giảm tính dễ bị tổn thương trước MIA dưới cách phân chia theo cá nhân.

*[Hình 2: Ảnh hưởng của việc thay đổi số lượng cá nhân lên các TPR cấp bản ghi (trái và giữa) và cấp người dùng (phải) cho LiRA-Online, Multi-Signal LiRA-Online và DTS với InceptionTime; được vẽ cùng khoảng tin cậy qua năm lần chạy.]*

#### 3) Tác động của dữ liệu người dùng được quan sát một phần

Chúng tôi nghiên cứu ảnh hưởng của kích thước quan sát của kẻ tấn công bằng cách cho $\tilde{D}_u$ gồm các tập con của $D_u$ và thay đổi $n_c$ với $n_c \in \{0, 0{,}05n, 0{,}10n, \ldots, n\}$. Ở đây, $n_c = 1$ phản ánh suy luận cấp bản ghi không có ngữ cảnh cấp người dùng, trong khi $n_c = n$ tương ứng với bối cảnh quan sát đầy đủ đã được phân tích. Cả hai bối cảnh này đều được nghiên cứu ở Mục VII-B.

Trong Hình 3, chúng tôi báo cáo hiệu năng cấp người dùng, được lấy trung bình trên năm tập con ngẫu nhiên độc lập sử dụng cùng các mô hình mục tiêu và mô hình bóng. Đúng như kỳ vọng, hiệu năng tấn công tăng cùng $n_c$; tuy nhiên, đáng ngạc nhiên hơn là các kết quả mạnh xuất hiện ngay cả ở những phần nhỏ trong các bản ghi của một người dùng, cho thấy mức độ dễ bị tổn thương cao trước suy luận thành viên cấp người dùng.

*[Hình 3: Ảnh hưởng của kích thước tập dữ liệu quan sát được lên MIA cấp người dùng cho LiRA-Online, Multi-Signal LiRA-Online và DTS-Online.]*

---

## VIII. Kết luận

Tóm lại, chúng tôi đưa các tấn công suy luận thành viên vào dự báo chuỗi thời gian dựa trên học sâu, điều chỉnh LiRA và RMIA, giới thiệu một LiRA đa tín hiệu, và đề xuất tấn công DTS dựa trên học sâu. Các đánh giá trên các tập dữ liệu thực (EEG và ELD) với LSTM và N-HiTS cho thấy các mô hình này dễ bị tổn thương, đặc biệt dưới các mô hình đe dọa cấp người dùng, bất chấp việc dừng sớm và khả năng khái quát hóa tốt. Các tấn công trực tuyến vượt trội đáng kể so với các tấn công ngoại tuyến, và các tấn công cấp người dùng có thể phơi bày những điểm yếu nghiêm trọng bị bỏ sót bởi đánh giá cấp bản ghi. Các nghiên cứu cắt bỏ còn cho thấy rằng các tầm dự báo dài hơn làm tăng rò rỉ, trong khi các quần thể lớn hơn làm giảm nó. Kết quả của chúng tôi thiết lập các đường cơ sở mạnh và làm nổi bật các yếu tố then chốt ảnh hưởng đến rủi ro quyền riêng tư trong dự báo chuỗi thời gian, cung cấp thông tin cho việc thiết kế cả các tấn công lẫn các biện pháp phòng thủ trong tương lai.

Mặc dù nghiên cứu của chúng tôi cung cấp một điểm khởi đầu, nó bị giới hạn ở hai tập dữ liệu và hai kiến trúc mô hình. Việc mở rộng các đánh giá sang các lĩnh vực khác (ví dụ tín hiệu ECG, giao dịch tài chính, cảm biến IoT, dấu vết GPS), các kiến trúc thay thế như transformer và mạng tích chập thời gian (temporal convolutional network), và các bối cảnh có biến ngoại sinh hoặc dự báo xác suất có thể cho một bức tranh đầy đủ hơn về các điểm dễ bị tổn thương. Một hạn chế quan trọng khác là số lượng cá nhân tương đối nhỏ trong các tập dữ liệu của chúng tôi, có thể gây ra sự bất ổn trong kết quả; việc mở rộng các đánh giá sang các nhóm lớn hơn sẽ giúp đánh giá độ vững chắc và khả năng tổng quát hóa của các phát hiện. Hơn nữa, việc tinh chỉnh suy luận cấp người dùng, vượt ra ngoài phép tổng hợp điểm số độc lập để hướng tới các tiếp cận đầu-cuối, vẫn là một thách thức mở quan trọng. Giải quyết các hướng này là then chốt để hiểu rõ các rủi ro quyền riêng tư trong dự báo chuỗi thời gian.

---

## Tuyên bố về đạo đức (Ethics Statement)

Công trình này nhằm cải thiện việc đánh giá rủi ro đối với dữ liệu chuỗi thời gian nhạy cảm bằng cách xác định các điểm dễ bị tổn thương trong các mô hình dự báo. Công trình này dựa trên các tập dữ liệu công khai và mã nguồn được mở để hỗ trợ nghiên cứu có thể tái lập.

---

## Tài liệu tham khảo

[1] R. Shokri, M. Stronati, C. Song, and V. Shmatikov, "Membership inference attacks against machine learning models," in IEEE symposium on security and privacy (SP), 2017.

[2] J. Ye, A. Maddi, S. K. Murakonda, V. Bindschaedler, and R. Shokri, "Enhanced membership inference attacks against machine learning models," in ACM SIGSAC Conference on Computer and Communications Security (CCS), 2022.

[3] S. Zarifzadeh, P. Liu, and R. Shokri, "Low-cost high-power membership inference attacks," in International Conference on Machine Learning (ICML), 2024.

[4] S. K. Murakonda and R. Shokri, "Ml privacy meter: Aiding regulatory compliance by quantifying the privacy risks of machine learning," in Workshop on Hot Topics in Privacy Enhancing Technologies (HotPETs), 2020.

[5] N. Carlini, F. Tramer, E. Wallace, M. Jagielski, A. Herbert-Voss, K. Lee, A. Roberts, T. Brown, D. Song, U. Erlingsson et al., "Extracting training data from large language models," in USENIX security symposium (USENIX Security), 2021.

[6] J. Zhang, D. Das, G. Kamath, and F. Tramèr, "Position: Membership inference attacks cannot prove that a model was trained on your data," in IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), 2025.

[7] N. Carlini, C. Liu, Ú. Erlingsson, J. Kos, and D. Song, "The secret sharer: Evaluating and testing unintended memorization in neural networks," in USENIX security symposium (USENIX security), 2019.

[8] A. Salem, G. Cherubin, D. Evans, B. Köpf, A. Paverd, A. Suri, S. Tople, and S. Zanella-Béguelin, "Sok: Let the privacy games begin! a unified treatment of data inference privacy in machine learning," in IEEE Symposium on Security and Privacy (SP), 2023.

[9] Z. Di, S. Yu, Y. Vorobeychik, and Y. Liu, "Adversarial machine unlearning," in International Conference on Learning Representations (ICLR), 2025.

[10] M. Nasr, S. Songi, A. Thakurta, N. Papernot, and N. Carlin, "Adversary instantiation: Lower bounds for differentially private machine learning," in IEEE Symposium on security and privacy (SP), 2021.

[11] Y. Oren, N. Meister, N. S. Chatterji, F. Ladhak, and T. Hashimoto, "Proving test set contamination in black-box language models," in International Conference on Learning Representations (ICLR), 2023.

[12] N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis, and F. Tramér, "Membership inference attacks from first principles," in IEEE Symposium on Security and Privacy (SP), 2022.

[13] Y. Liu, Z. Zhao, M. Backes, and Y. Zhang, "Membership inference attacks by exploiting loss trajectory," in ACM SIGSAC Conference on Computer and Communications Security (CCS), 2022.

[14] W. Shi, A. Ajith, M. Xia, Y. Huang, D. Liu, T. Blevins, D. Chen, and L. Zettlemoyer, "Detecting pretraining data from large language models," in International Conference on Learning Representations (ICLR), 2024.

[15] J. Mattern, F. Mireshghallah, Z. Jin, B. Schoelkopf, M. Sachan, and T. Berg-Kirkpatrick, "Membership inference attacks against language models via neighbourhood comparison," in Findings of the Association for Computational Linguistics (ACL), 2023.

[16] M. Duan, A. Suri, N. Mireshghallah, S. Min, W. Shi, L. Zettlemoyer, Y. Tsvetkov, Y. Choi, D. Evans, and H. Hajishirzi, "Do membership inference attacks work on large language models?" in Conference on Language Modeling (CoLM), 2024.

[17] F. P. Krüger, J. Östman, L. Mervin, I. V. Tetko, and O. Engkvist, "Publishing neural networks in drug discovery might compromise training data privacy," Journal of Cheminformatics, 2025.

[18] M. Lassila, J. Östman, K.-H. Ngo et al., "Practical Bayes-optimal membership inference attacks," in Annual Conference on Advances in neural information processing systems (NeurIPS), 2025.

[19] A. Pyrgelis, C. Troncoso, and E. De Cristofaro, "Knock knock, who's there? membership inference on aggregate location data," in Network and Distributed Systems Security (NDSS) Symposium, 2018.

[20] A. Voyez, T. Allard, G. Avoine, P. Cauchois, E. Fromont, and M. Simonin, "Membership inference attacks on aggregated time series with linear programming," in International Conference on Security and Cryptography (SECRYPT), 2022.

[21] F. Mireshghallah, K. Goyal, A. Uniyal, T. Berg-Kirkpatrick, and R. Shokri, "Quantifying privacy risks of masked language models using membership inference attacks," in Conference on Empirical Methods in Natural Language Processing (EMNLP), 2022.

[22] M. Meeus, S. Jain, M. Rei, and Y.-A. de Montjoye, "Did the neurons read your book? document-level membership inference for large language models," in USENIX Security Symposium (USENIX Security), 2024.

[23] M. Meeus, I. Shilov, S. Jain, M. Faysse, M. Rei, and Y.-A. de Montjoye, "SoK: Membership Inference Attacks on LLMs are Rushing Nowhere (and How to Fix It)," in IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), 2025.

[24] N. Koren, A. Goldsteen, G. Amit, and A. Farkash, "Membership inference attacks against time-series models," in Proceedings of the Asian Conference on Machine Learning (ACML), 2024.

[25] L. Rossi, M. Aerni, J. Zhang, and F. Tramèr, "Membership inference attacks on sequence models," in IEEE Security and Privacy Workshops (SPW), 2025.

[26] H. Li, Z. Li, S. Wu, C. Hu, Y. Ye, M. Zhang, D. Feng, and Y. Zhang, "Seqmia: Sequential-metric based membership inference attack," in ACM SIGSAC Conference on Computer and Communications Security (CCS), 2024.

[27] T. Chen, A. Asai, N. Mireshghallah, S. Min, J. Grimmelmann, Y. Choi, H. Hajishirzi, L. Zettlemoyer, and P. W. Koh, "CopyBench: Measuring literal and non-literal reproduction of copyright-protected text in language model generation," in Conference on Empirical Methods in Natural Language Processing (EMNLP), 2024.

[28] N. Kandpal, K. Pillutla, A. Oprea, P. Kairouz, C. A. Choquette-Choo, and Z. Xu, "User inference attacks on large language models," in Conference on Empirical Methods in Natural Language Processing (EMNLP), 2024.

[29] S. Makridakis, "Accuracy measures: theoretical and practical concerns," International Journal of Forecasting, vol. 9, no. 4, 1993.

[30] Z. Yue, Y. Wang, J. Duan, T. Yang, C. Huang, Y. Tong, and B. Xu, "Ts2vec: Towards universal representation of time series," in AAAI conference on artificial intelligence, vol. 36, no. 8, 2022.

[31] A. Jalaly Bidgoly, H. Jalaly Bidgoly, and Z. Arezoumand, "A survey on methods and challenges in eeg based authentication," Computers & Security, vol. 93, 2020.

[32] A. Voyez, T. Allard, G. Avoine, P. Cauchois, E. Fromont, and M. Simonin, "Unique in the smart grid -the privacy cost of fine-grained electrical consumption data," Scientific Reports, vol. 15, no. 17391, 2025.

[33] I. Obeid and J. Picone, "The temple university hospital eeg data corpus," Frontiers in Neuroscience, vol. 10, 2016.

[34] scikit learn, "sklearn.preprocessing.robustscaler." [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html

[35] A. Trindade, "ElectricityLoadDiagrams20112014," UCI Machine Learning Repository, 2015, DOI: https://doi.org/10.24432/C58C86.

[36] S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Comput., vol. 9, no. 8, 1997.

[37] C. Challu, K. G. Olivares, B. N. Oreshkin, F. Garza Ramirez, M. Mergenthaler Canseco, and A. Dubrawski, "Nhits: Neural hierarchical interpolation for time series forecasting," AAAI Conference on Artificial Intelligence, vol. 37, no. 6, 2023.

[38] B. N. Oreshkin, D. Carpov, N. Chapados, and Y. Bengio, "N-beats: Neural basis expansion analysis for interpretable time series forecasting," in International Conference on Learning Representations (ICLR), 2020.

[39] C. Challu, K. G. Olivares, B. N. Oreshkin, F. Garza, M. Mergenthaler-Canseco, and A. Dubrawski, "N-HiTS: Neural hierarchical interpolation for time-series forecasting," https://github.com/cchallu/n-hits, 2023, gitHub repository.

[40] X. Glorot and Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks," in International Conference on Artificial Intelligence and Statistics (AISTATS), 2010.

[41] S. Shachor, N. Razinkov, and A. Goldsteen, "Improved membership inference attacks against language classification models," arXiv 2310.07219 [cs.LG], 2024.

[42] H. Ismail Fawaz, B. Lucas, G. Forestier, C. Pelletier, D. F. Schmidt, J. Weber, G. I. Webb, L. Idoumghar, P.-A. Muller, and F. Petitjean, "Inceptiontime: Finding alexnet for time series classification," Data Mining and Knowledge Discovery, vol. 34, no. 6, 2020.

[43] A. Ismail-Fawaz, M. Devanne, S. Berretti, J. Weber, and G. Forestier, "Look into the lite in deep learning for time series classification," International Journal of Data Science and Analytics, 2025.

[44] H. A. Dau, A. Bagnall, K. Kamgar, C.-C. M. Yeh, Y. Zhu, S. Gharghabi, C. A. Ratanamahatana, and E. Keogh, "The ucr time series archive," IEEE/CAA Journal of Automatica Sinica, vol. 6, no. 6, 2019.

[45] J. Hayes, I. Shumailov, C. A. Choquette-Choo, M. Jagielski, G. Kaissis, K. Lee, M. Nasr, S. Ghalebikesabi, N. Mireshghallah, M. S. M. S. Annamalai, I. Shilov, M. Meeus, Y.-A. de Montjoye, F. Boenisch, A. Dziedzic, and A. F. Cooper, "Exploring the limits of strong membership inference attacks on large language models," in Annual Conference on Advances in neural information processing systems (NeurIPS), 2025.

[46] M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, and L. Zhang, "Deep learning with differential privacy," in ACM SIGSAC Conference on Computer and Communications Security (CCS), 2016.

---

## Phụ lục

### A. Các đường cong ROC

Trong phụ lục này, chúng tôi báo cáo các đường cong ROC tương ứng với các kết quả được trình bày ở Mục VII cho cả bối cảnh trực tuyến và ngoại tuyến.

*[Hình 4: Các đường cong ROC với thang logarit (TPR trên trục tung và FPR trên trục hoành) cho năm tấn công suy luận thành viên: Ensemble, RMIA, LiRA, Multi-Signal LiRA và DTS, được sắp xếp theo hàng. Các cột tương ứng với các cặp tập dữ liệu–kiến trúc: EEG–LSTM, EEG–N-HiTS, ELD–LSTM và ELD–N-HiTS. Tất cả các kết quả đều thuộc bối cảnh trực tuyến, với tấn công Ensemble được đánh giá ở chế độ kiểm toán (audit mode). Trong mỗi biểu đồ con, các đường cong tương ứng với từng tín hiệu tấn công riêng lẻ đối với LiRA, RMIA và Ensemble, tất cả các tín hiệu kết hợp đối với Multi-Signal LiRA, và các kiến trúc bộ phân loại khác nhau đối với DTS. Mỗi đường cong biểu diễn giá trị trung bình qua năm lần chạy độc lập.]*

*[Hình 5: Các đường cong ROC với thang logarit (TPR trên trục tung và FPR trên trục hoành) cho bốn tấn công suy luận thành viên: RMIA, LiRA, Multi-Signal LiRA và DTS, được sắp xếp theo cột. Các hàng tương ứng với các cặp tập dữ liệu–kiến trúc: EEG–LSTM, EEG–N-HiTS, ELD–LSTM và ELD–N-HiTS. Tất cả các kết quả đều thuộc bối cảnh ngoại tuyến. Trong mỗi biểu đồ con, các đường cong tương ứng với từng tín hiệu tấn công riêng lẻ đối với LiRA và RMIA, tất cả các tín hiệu kết hợp đối với Multi-Signal LiRA, và các kiến trúc bộ phân loại khác nhau đối với DTS. Mỗi đường cong biểu diễn giá trị trung bình qua năm lần chạy độc lập.]*

### B. Giảm thiểu bằng DP-SGD

Suy luận thành viên về cơ bản gắn liền với sự ghi nhớ, dẫn đến những khác biệt có hệ thống giữa các dự đoán trên thành viên và không phải thành viên [7], [12]. Do đó, một số chiến lược giảm thiểu nhắm đến việc giảm sự ghi nhớ này, chẳng hạn thông qua các kỹ thuật chính quy hóa như dropout, weight decay và dừng sớm. Trong tất cả các thí nghiệm chính ở Mục VII, chúng tôi sử dụng dừng sớm như một phần của quy trình huấn luyện tiêu chuẩn.

Ngoài các phương pháp dựa trên chính quy hóa, riêng tư vi phân (differential privacy) thường được áp dụng để cung cấp thêm khả năng chống chịu. Cụ thể, DP-SGD [46] chèn nhiễu được hiệu chỉnh trong quá trình huấn luyện và cung cấp các đảm bảo quyền riêng tư hình thức.

Để xem xét hiệu ứng này trong các mô hình dự báo, chúng tôi thực hiện một thí nghiệm trong đó một mô hình mục tiêu N-HiTS được huấn luyện trên tập dữ liệu ELD bằng DP-SGD với chuẩn cắt (clipping norm) cố định $C = 10$, kích thước batch 1024, và hệ số nhiễu (noise multiplier) thay đổi theo $\sigma \in \{0{,}0;\ 0{,}2;\ 0{,}8;\ 1{,}0\}$. Sau đó chúng tôi tấn công mô hình mục tiêu dưới mô hình đe dọa cấp bản ghi bằng DTS và Multi-Signal LiRA trong bối cảnh trực tuyến.

Bảng V báo cáo hiệu năng tấn công và các mất mát của mô hình mục tiêu (MSE, MAE, SMAPE, ND) ở các mức nhiễu khác nhau. Kết quả không dùng DP-SGD ($\sigma = 0$ và $C = \text{None}$) cũng được đưa vào. Đúng như kỳ vọng, nhiễu càng nhiều thì MIA càng yếu đi một cách nhất quán trong khi các mất mát xấu đi. Cụ thể, hiệu năng của tấn công DTS trong thiết lập riêng tư nhất ($\sigma = 1{,}0$) cho TPR 3,26% tại FPR 0,1%. Con số này thấp hơn một nửa so với TPR quan sát được khi không dùng DP-SGD.

**Bảng V:** Các mất mát của mô hình mục tiêu đối với N-HiTS được huấn luyện trên ELD bằng DP-SGD với chuẩn cắt $C = 10$, kích thước batch 1024, và các hệ số nhiễu $\sigma \in \{0; 0{,}2; 0{,}8; 1\}$, cùng với hiệu năng tấn công DTS-online và Multi-Signal LiRA-Online trên các mô hình nêu trên tại FPR 0,1%. Các mất mát của mô hình và hiệu năng tấn công đối với đường cơ sở không dùng DP-SGD ($\sigma = 0$ và $C = \text{None}$) cũng được báo cáo để tham chiếu.

| $\sigma$ | $C$ | DTS (InceptionTime) 0.1% FPR | Multi-Signal LiRA 0.1% FPR | MSE | MAE | SMAPE | ND |
|---|---|---|---|---|---|---|---|
| 0.0 | None | 7.12 | 3.32 | 0.034314 | 0.088645 | 0.166489 | 0.163490 |
| 0.0 | 10 | 5.69 | 3.02 | 0.344160 | 0.091787 | 0.178334 | 0.177658 |
| 0.2 | 10 | 5.71 | 2.85 | 0.034814 | 0.094275 | 0.175760 | 0.173873 |
| 0.8 | 10 | 4.91 | 1.91 | 0.037147 | 0.098007 | 0.181079 | 0.180756 |
| 1.0 | 10 | 3.26 | 1.88 | 0.039224 | 0.105430 | 0.195587 | 0.195476 |
