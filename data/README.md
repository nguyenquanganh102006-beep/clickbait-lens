# Dữ liệu dùng chung

manifests/dataset_v1.0 đóng băng manifest và split; không có toàn văn, ảnh hay nhãn từng bài. Lấy snapshot ngoài Git rồi chạy scripts/validate_research_dataset.py. Không tạo split riêng trong script task. Chỉ đổi manifest khi có release dataset mới được nhóm thống nhất.

input/ và output/ có sẵn thuộc tool annotation cũ, không dùng làm split train/test.
