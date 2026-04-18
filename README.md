# DATATHON 2026 – The Gridbreakers · Đội Attention Seekers

Repo nộp bài Vòng 1, cuộc thi **DATATHON 2026 – The Gridbreakers** do
VinTelligence (VinUni DS&AI Club) tổ chức.

- Kaggle: <https://www.kaggle.com/competitions/datathon-2026-round-1>
- Timeline: 19/04/2026 – 29/04/2026
- Finalist on-site: 23/05/2026 tại VinUniversity, Hà Nội

## 1. Mục tiêu

| Phần | Nội dung | Điểm |
|---|---|---|
| 1 | 10 câu trắc nghiệm (Q1–Q10) | 20 |
| 2 | EDA & Visualization (Descriptive → Prescriptive) | 60 |
| 3 | Sales Forecasting 2023-01-01 → 2024-07-01 (MAE / RMSE / R²) | 20 |

Xem chi tiết phân công trong [`PHAN-CONG-CONG-VIEC.md`](./PHAN-CONG-CONG-VIEC.md).

## 2. Cấu trúc thư mục

```
.
├── data/                 # 15 CSV từ Kaggle (không commit)
│   └── README.md         # Hướng dẫn tải data
├── notebooks/            # EDA, MCQ, modeling notebooks
├── src/                  # Code tái dùng (data_loader, features, models)
├── experiments/          # Log thí nghiệm (.csv)
├── models/               # Mô hình đã train (.pkl)
├── submissions/          # File submission.csv cho Kaggle
├── reports/
│   ├── figures/          # Figure xuất bản (PNG/PDF)
│   └── paper/            # Báo cáo NeurIPS LaTeX
├── requirements.txt
├── PHAN-CONG-CONG-VIEC.md
└── README.md
```

## 3. Cách chạy lại

### 3.1. Chuẩn bị môi trường

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3.2. Tải dữ liệu

Tải 15 file CSV từ [Kaggle competition page](https://www.kaggle.com/competitions/datathon-2026-round-1/data)
và đặt vào thư mục `data/`. Xem [`data/README.md`](./data/README.md) để biết danh sách file.

### 3.3. Kiểm tra dữ liệu

```bash
python src/check_data.py
```

Script sẽ in MD5 và row count của từng CSV, đồng thời cảnh báo nếu thiếu file.

### 3.4. Pipeline end-to-end (sẽ bổ sung dần)

```bash
# 1. EDA & MCQ
jupyter lab notebooks/

# 2. Train forecasting model
python src/train.py              # sẽ thêm ở D5

# 3. Generate submission
python src/predict.py            # sẽ thêm ở D6
# → submissions/submission.csv
```

## 4. Thành viên

| Vai trò | Thành viên | Phụ trách chính |
|---|---|---|
| TV1 | (điền tên) | Pipeline, Feature Engineering, Forecasting, SHAP |
| TV2 | (điền tên) | EDA 4 cấp độ, Visualization, Report NeurIPS |

## 5. License

Code: MIT. Dữ liệu thuộc về VinTelligence / VinUniversity.
