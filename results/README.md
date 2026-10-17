# results/

Cấu trúc: `results/<model>/<task>/<attack>.json`

- model: lstm, tcn (TV1), patchtst (TV2), itransformer (TV3), foundation (TV4)
- task: cls (PTB-XL superclass), fcst (L = 512, H = 96)
- attack: loss, conf, entropy (chỉ cls), lira, rmia, temporal
- level: chỉ `record` (ghi vào trường `level`)

Trường `null` là chưa chạy. Người sở hữu model điền vào; không sửa file của người khác.
Metric chính: TPR @ 0,1% và 1% FPR kèm bootstrap CI; AUC chỉ đối chiếu.

## File mẫu
Xem `_sample.json`. Mọi người giữ đúng tên trường, đơn vị (tỉ lệ 0–1, không dùng %), và quy ước điểm cao = nhiều khả năng là member. Trường không áp dụng thì để `null`, thông tin thêm đặt vào `extra`.
