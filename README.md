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
1. Remove sinus trước. 
   - Lý do xử lý sinus đầu tiên là do nếu thực hiện các xử lý khác trước có thể làm mất tính nhiễu sin dẫn tới lọc tín hiệu sinus khó hơn.
   - Chuyển về miền tần số, tìm đó tìm tần số bị nhiễu từ đó loại bỏ các tần số nhiễu này đi và ngịch đảo lại về ảnh đa mức sáng
2. Lọa bỏ nhiễu muối tiêu
   - sử dụng medianBlur để loại bỏ nhiễu muối tiêu
3. Tăng độ tương phản: 
   - Sử dụng hiệu chỉnh gamma. Gamma sẽ được lựa chọn tinh chỉnh theo std và mean của ảnh.

### 2. Đếm số hạt gạo trong ảnh
Ý tưởng: Chuyển ảnh về dạng nhị phân, sau đó sử dụng thuật toán tìm biên (contours) của hạt gạo từ đó đếm số hạt gạo dựa trên số biên
1. Chuyển ảnh về ảnh nhị phân: Sử dụng otsu thesh
2. xử lý hình thái học theo phương pháp open để xóa bỏ các vùng trắng nhiễu nhỏ
3. tìm kiếm các vùng biên hạt gạo
4. xử lý wateshed local trên từng vùng biên hạt gạo to bất thường để cố gắng tách các hạt gạo dính vào nhau
5. loại bỏ các vùng biên hạt gạo nhiễu dựa trên diện tích trong contours
6. Trả về kết quả số hạt gạo tương ứng với số contours


## 3. Hướng dẫn chạy 
- Install requirements:
```commandline
pip install -r requirements.txt
```

- Chạy đếm số hạt gạo
```commandline
python main.py -f "<img_path>"
```

Ví dụ:
```commandline
python main.py -f data/1_wIXlvBeAFtNVgJd49VObgQ.png
```