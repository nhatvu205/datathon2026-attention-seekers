# Data

Thư mục này chứa 15 file CSV từ Kaggle. **Không commit lên Git** (đã gitignore).

## Tải dữ liệu

1. Truy cập <https://www.kaggle.com/competitions/datathon-2026-round-1/data>.
2. Download toàn bộ dataset (`Download All`).
3. Giải nén vào thư mục `data/` này.

Hoặc dùng Kaggle CLI:

```bash
pip install kaggle
# Đặt ~/.kaggle/kaggle.json với API key (Kaggle → Account → Create New Token)
kaggle competitions download -c datathon-2026-round-1 -p data/
cd data/ && unzip -o datathon-2026-round-1.zip && rm datathon-2026-round-1.zip
```

## Danh sách file (15 file)

### Master
- `products.csv` — Danh mục sản phẩm
- `customers.csv` — Khách hàng
- `promotions.csv` — Khuyến mãi
- `geography.csv` — Địa lý (mã bưu chính)

### Transaction
- `orders.csv` — Đơn hàng
- `order_items.csv` — Chi tiết dòng sản phẩm
- `payments.csv` — Thanh toán (1:1 với orders)
- `shipments.csv` — Vận chuyển
- `returns.csv` — Trả hàng
- `reviews.csv` — Đánh giá

### Analytical
- `sales.csv` — Doanh thu train (04/07/2012 – 31/12/2022)
- `sample_submission.csv` — Định dạng file nộp bài

### Operational
- `inventory.csv` — Tồn kho cuối tháng
- `web_traffic.csv` — Lưu lượng website hàng ngày

## Kiểm tra

Sau khi tải về, chạy:

```bash
python src/check_data.py
```

Script sẽ in MD5 + row count từng file và báo nếu thiếu.
