# BTL-CV-Master-HUST

--- 

## Yêu cầu bài toán:
1) Dữ liệu đầu vào:
Giả sử bạn có một tập hợp các ảnh của một ứng dụng, trong đó chứa các hạt gạo. Các ảnh này
có thể bị nhiễu bởi nhiều loại nhiễu khác nhau (Ảnh trong folder [data](BTL-1/data))
2) Dữ liệu đầu ra:
Chương trình của bạn phải trả về số lượng hạt gạo trong mỗi ảnh.


## Giải pháp

### 1. Tiền xử lý ảnh
Xử lý các nhiễu và làm mịn ảnh hưởng
- Xử lý nhiễu muối tiêu

### 2. Đếm số hạt gạo trong ảnh
Có 3 giải pháp đếm số hạt gạo trong ảnh bao gồm:
- Đếm số hạt gạo dựa trên đường bao của hạt gạo
- Dếm số hạt gạo dựa trên vùng liên thông
- Watershed: open/close ảnh trước khi đến, sau đón sử dụng watershed để tách các hạt dính vào nhau