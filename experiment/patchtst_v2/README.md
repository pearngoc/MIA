# PatchTST v2 (TV2): notebook tách từng bước

| | v1 (`experiment/patchtst/`) | v2 (folder này) |
|---|---|---|
| Cấu trúc | 1 cell `%%writefile` chứa cả `patchtst_mia.py` (720 dòng), chạy ngầm bằng `subprocess` | 15 bước, mỗi bước một cell có giải thích, biến nằm trong notebook |
| Xem / dừng giữa chừng | Chỉ đọc được log | Chạy từng cell, in / vẽ được bất kỳ biến nào (`target`, `PHI_S`, `scores`…) |
| Split chung | `git clone` repo, kiểm sha256 (nằm trong file `.py`) | Cũng `git clone` repo (vào `/tmp/MIA`) + kiểm sha256, nhưng là một cell riêng, in `OK` từng file |
| Lấy PTB-XL | Add Input | Add Input, hoặc `kagglehub.dataset_download` tự tải nếu chưa add |
| Task | cls và fcst song song trên 2 GPU | Một task mỗi lần chạy: đổi `TASK` ở bước 2 |
| Resume | Checkpoint từng epoch | Bỏ qua từng cấu hình tune / target / shadow đã xong |
| Số liệu | | **Giống v1**: cùng split, recipe, seed, công thức attack, schema JSON |

Đã kiểm tra trên dữ liệu PTB-XL giả (CPU): v2 cho ra **cùng điểm số với v1** ở mọi attack của cả cls và fcst.

## Chạy trên Kaggle

1. *File → Import Notebook* → chọn `patchtst_v2_kaggle.ipynb`.
2. *Settings*: **GPU T4 x2**, **Internet = On** (để clone repo). *Add Input* → dataset `khyeh0719/ptb-xl-dataset` (không add thì cell `kagglehub` tự tải, chậm hơn).
3. Ở bước 2 đặt `TASK = 'cls'` (lần sau `'fcst'`).
4. Học / thử: chạy từng cell. Chạy thật: **Save Version → Save & Run All (Commit)**.
   Thời gian trên T4: cls ≈ 1 giờ 45, fcst ≈ 2 giờ 40. `RUN_TUNE = False` bỏ qua tune (~50 phút), dùng cấu hình đã chọn ngày 18/10 (cls 20 epoch / 5e-4, fcst 50 epoch / 1e-3).
5. Tải `patchtst_v2_results_<task>.zip` ở tab *Output*, chép vào **gốc repo MIA** → **Extract Here**.

Muốn chạy lại phần attack mà không train lại: *Add Input → Notebook Output* của version trước (v2, hoặc v1 vì cùng tên file và key) rồi chạy lại. Cell "Chạy tiếp" chép checkpoint vào, các bước 6–8 tự bỏ qua.

## Các bước trong notebook

| Bước | Nội dung | File tạo ra (trong `/kaggle/working`) |
|---|---|---|
| 0 | Cài `wfdb`, `kagglehub` lấy PTB-XL, kiểm GPU + vị trí `ptbxl_database.csv` | |
| 1 | Clone repo nhóm, tìm PTB-XL, kiểm sha256 của split + shadow mask | `/tmp/MIA/` |
| 2 | Cấu hình (`TASK`, `RUN_TUNE`, lưới tune) + chạy tiếp | |
| 3 | Đọc 16.272 bản ghi, nội suy 1000 → 608, z-score; `idx_m` / `idx_v` / `idx_n`, `masks_pool` | `/tmp/cache_patchtst/` |
| 4 | Model PatchTST (HF `transformers`), chạy thử 1 batch | |
| 5 | `train()`, `predict()`, `utility()` dùng chung | |
| 6 | Tune ≤ 6 cấu hình trên `val` | `ckpt/patchtst_<task>/tuning.json` |
| 7 | Target trên member, lưu output ở 4 epoch | `target_ep*.pt`, `target.done.json` |
| 8 | 8 shadow | `shadow_k0.pt` … `shadow_k7.pt` |
| 9 | Metric (TPR @ 0,1% / 1% FPR, AUC, bootstrap CI) và hàm ghi JSON | |
| 10 | Loss / Conf / Entropy | `results/patchtst/<task>/{loss,conf,entropy}.json` |
| 11 | LiRA online / offline, RMIA offline / online | `{lira,lira_offline,rmia,rmia_online}.json` |
| 12 | Temporal (gray-box) + bảng ablation | `temporal.json` |
| 13 | Bảng tóm tắt + ROC log–log | `patchtst_<task>_roc.png` |
| 14 | Hình demo: member lộ nhất vs non-member điển hình | `demo_lira_example.png` |
| 15 | Đóng gói | `patchtst_v2_results_<task>.zip` |

Các khác biệt so với `dataset/README.md` mục 2–3 giữ nguyên như v1 (xem `experiment/patchtst/README.md` và `extra.deviations` trong mỗi JSON).
