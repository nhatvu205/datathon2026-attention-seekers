# DATATHON 2026 – The Gridbreakers · Đội Attention Seekers

Repo nộp bài Vòng 1, cuộc thi **DATATHON 2026 – The Gridbreakers** do
VinTelligence (VinUni DS&AI Club) tổ chức.

- Kaggle: <https://www.kaggle.com/competitions/datathon-2026-round-1>
- Timeline: 19/04/2026 – 29/04/2026
- Finalist on-site: 23/05/2026 tại VinUniversity, Hà Nội

## 1. Kết quả

| Metric | Giá trị |
|---|---|
| Public test MAE | **800,466** |
| OOF MAE (walk-forward CV 4 folds) | 882,374 |
| Baseline (sample submission) | 1,331,000 |
| Cải thiện so với baseline | 39.8% |

Pipeline: Prophet trend decomposition + XGBoost / LightGBM / CatBoost Huber residual stack + HuberRegressor meta-learner + per-(month, day) median shrinkage.

## 2. Cấu trúc thư mục

```
.
├── data/
│   ├── data_cleaned/          # 15 CSV đã qua preprocessing
│   ├── features/
│   │   ├── train_features.parquet   # ~190 features, train 2012-2022
│   │   └── test_features.parquet    # ~190 features, test 2023-2024
│   └── README.md
│
├── notebooks/
│   ├── 00-schema-overview.ipynb         # Cấu trúc, dtype, null, cardinality
│   ├── 05-preprocessing.ipynb           # Làm sạch dữ liệu gốc
│   ├── 10_mcq_all.ipynb                 # Phần 1: 10 câu trắc nghiệm
│   ├── 20-descriptive-diagnostic-eda.ipynb  # Phần 2: EDA Descriptive + Diagnostic
│   ├── 30-feature-engineering.ipynb     # Build feature matrix → parquet
│   ├── 40-predictive-prescriptive-eda.ipynb # Phần 2: EDA Predictive + Prescriptive
│   ├── 50-modeling.ipynb                # Phần 3: Training + submission
│   └── baseline.ipynb                   # Seasonal naive baseline
│
├── src/
│   ├── data_loader.py         # Load 15 CSV với type casting chuẩn
│   ├── forecast_features.py   # Utility functions cho feature engineering
│   └── check_data.py          # Validate MD5 + row count của 15 CSV
│
├── submissions/
│   └── attention_seekers_final_submission.csv
│
├── reports/
│   ├── modeling_report.docx   # Báo cáo Feature Engineering + Modeling
│   ├── generate_report.py     # Script tái tạo file .docx
│   ├── charts/                # Biểu đồ EDA, feature importance, SHAP, forecast
│   ├── figures/               # Figures xuất bản
│   └── paper/                 # Báo cáo NeurIPS LaTeX
│
├── models/                    # Model artifacts (.pkl) — gitignored
├── requirements.txt
└── README.md
```

## 3. Cách chạy pipeline

### 3.1. Chuẩn bị môi trường

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### 3.2. Tải và kiểm tra dữ liệu

Tải 15 CSV từ [Kaggle](https://www.kaggle.com/competitions/datathon-2026-round-1/data),
đặt vào `data/data_cleaned/`. Xem `data/README.md` để biết danh sách file và MD5.

```bash
python src/check_data.py   # kiểm tra MD5 + row count, cảnh báo nếu thiếu file
```

### 3.3. Feature Engineering

Chạy toàn bộ `notebooks/30-feature-engineering.ipynb` theo thứ tự cell.

**Output:** `data/features/train_features.parquet` và `data/features/test_features.parquet`

Notebook thực hiện theo trình tự:
1. Calendar, Fourier harmonics (monthly / weekly / yearly)
2. Lag + rolling features (short-range, year-over-year, same-weekday)
3. Promotion, web traffic, order-level, inventory, returns/reviews
4. Per-(month, day) historical statistics — feature quan trọng nhất
5. **Feature importance pre-screen** (LightGBM nhanh, val = 2022) → tự động drop zero-gain features → save parquet

Điều chỉnh trong Cell 8b nếu cần:
```python
FI_AUTO_FILTER  = True    # False để xem FI mà không drop
FI_MIN_GAIN_PCT = 0.005   # drop thêm bottom 0.5% gain tích lũy
```

### 3.4. Training và Submission

Chạy toàn bộ `notebooks/50-modeling.ipynb` theo thứ tự cell.

**Output:**
- `submissions/submission_modeling_v2.csv` — pure ML prediction
- `submissions/submission_v2_blended.csv` — ML + median shrinkage (recommended)

Điều chỉnh trong notebook nếu cần:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `RUN_OPTUNA` | `True` | Chạy Optuna (~60-90 phút); `False` dùng params đã lưu |
| `N_OPTUNA_TRIALS` | `30` | Số trials mỗi model; giảm xuống 15 để nhanh hơn |
| `BLEND_W` | `0.5` | Weight của ML trong blend (0 = pure median, 1 = pure ML) |
| `SHRINK` | `0.07` | Mức shrink về overall median |

Pipeline:
1. Load parquet (tự động fallback về CSV nếu không có parquet)
2. Optuna hyperparameter tuning cho XGBoost / LightGBM / CatBoost
3. Walk-forward OOF stacking (4 folds: 2019 / 2020 / 2021 / 2022)
4. Prophet trend + Huber GBM residual + HuberRegressor meta
5. Post-processing: per-(month, day) median shrinkage 7%


## 4. Thành viên

| Vai trò | Phụ trách chính |
|---|---|
| Vũ Đình Nhật | Pipeline, Feature Engineering, Forecasting |
| Hồ Huỳnh Thư Nhi | EDA 4 cấp độ và phân tích, Visualization, Report|

