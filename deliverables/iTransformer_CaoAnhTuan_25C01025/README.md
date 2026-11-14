# Deliverable iTransformer — Cao Anh Tuấn (MSHV: 25C01025)

**Đề tài:** Membership Inference trên mô hình chuỗi thời gian hiện đại  
**Model:** **iTransformer**; phụ trách RMIA + Temporal Feature; phân tích **RQ3**

Sau khi review OK, zip cả thư mục này để gửi (không cần kèm raw PTB-XL ~3 GB).

```bash
cd deliverables
zip -r iTransformer_CaoAnhTuan_25C01025.zip iTransformer_CaoAnhTuan_25C01025
```

---

## Cấu trúc thư mục

```
iTransformer_CaoAnhTuan_25C01025/
├── README.md                          ← file này
├── report/
│   └── iTransformer_RQ3.md        ← section báo cáo + phân tích RQ3 (đọc trước)
├── results/
│   ├── cls/                           ← MIA classification (đủ attack)
│   ├── fcst/                          ← MIA forecasting (đủ attack)
│   ├── README_schema.md               ← quy ước JSON nhóm
│   └── _sample.json
├── code/
│   ├── setup.py
│   ├── itransformer/                  ← model, data loader, train, attack scripts
│   └── k8s/                           ← Job/PVC chạy trên GPU k3s (tham khảo)
├── protocol/
│   ├── Research_v2.md                 ← đề cương nhóm
│   ├── dataset_protocol.md            ← protocol dữ liệu/train/attack
│   └── ptbxl_record_s42.meta.json     ← meta split dùng chung (sha256)
└── meta/
    └── manifest.json                  ← danh sách file + hash rút gọn
```

**Không có trong gói:** dataset PTB-XL thô, checkpoint `.pt` (dung lượng lớn, tái tạo được từ code + split).

---

## Checklist đã nộp (iTransformer)

| Hạng mục | Trạng thái |
|---|---|
| Train iTransformer **cls** + Loss/Conf/Entropy | Done |
| LiRA / RMIA / Temporal **cls** | Done |
| Train iTransformer **fcst** + Loss | Done |
| LiRA / RMIA / Temporal **fcst** | Done |
| JSON đúng schema `results/<model>/<task>/<attack>.json` | Done |
| Phân tích **RQ3** + section Model iTransformer | Done (`report/`) |
| DLinear | Bỏ (tuỳ chọn, không bắt buộc) |
| Merge RMIA/Temporal vào attack lib TV2 | Code sẵn trong `code/`; phối hợp TV2 khi cần |

### File kết quả cls
`loss`, `conf`, `entropy`, `lira`, `lira_offline`, `rmia`, `rmia_online`, `temporal`

### File kết quả fcst
`loss`, `lira`, `lira_offline`, `rmia`, `rmia_online`, `temporal`

---

## Tóm tắt số chính (RQ3)

Cùng split PTB-XL record-level (8136 member / 7636 non-member), gap ≈ 0,48 cả hai task.

| Attack | Cls TPR@1% | Fcst TPR@1% | Cls AUC | Fcst AUC |
|---|---|---|---|---|
| Loss | 0,000 | **0,164** | 0,897 | 0,911 |
| LiRA | 0,060 | 0,065 | 0,584 | **0,806** |
| RMIA | 0,096 | 0,094 | 0,650 | **0,872** |
| Temporal | 0,271 | **0,459** | 0,945 | **0,963** |

**Polarity fix (cls LiRA/RMIA):** bản thô có AUC &lt; 0,5 (score lệch chiều protocol). Đã đảo dấu score và tính lại metric; ghi trong `extra.score_polarity_fix` của từng JSON.

**Kết luận ngắn:** Forecasting lộ membership mạnh hơn classification trên iTransformer — rõ nhất ở Temporal và Loss (chi tiết trong `report/iTransformer_RQ3.md`).

---

## Setup thí nghiệm (tóm tắt)

- Split: `ptbxl_record_s42` (seed 42), sha256 trong `protocol/ptbxl_record_s42.meta.json`
- iTransformer: d_model=128, 50 epoch, batch=16, max_lr=1e-3, AdamW + OneCycleLR
- Shadow LiRA/RMIA: K=8
- GPU chạy thực tế: RTX 2060 (k3s node `tuan-cao`)

---

## Liên hệ

- **Cao Anh Tuấn** — MSHV **25C01025** — iTransformer
