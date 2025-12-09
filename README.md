# checktime mini app
I. Mục đích hệ thống

Mô tả ngắn gọn mục tiêu:

Trực quan hóa lịch thực hiện thủ thuật / CLS theo timeline.

Phát hiện xung đột vận hành (trùng BS / BN / máy / đa trùng).

Hỗ trợ điều phối giảm chồng chéo ca, ngăn lỗi vận hành.

Tự động chia lane tránh chồng khối.

Sinh báo cáo conflict, heatmap, thống kê sử dụng máy.

II. Cấu trúc dữ liệu đầu vào
1. Các cột yêu cầu

Procedure

Start datetime

End datetime

Doctor

Patient

Machine

Dept

2. Chuẩn hóa dữ liệu

Loại bỏ .0 trong mã BN

Uppercase để so khớp

“Không dùng máy” quy về 1 nhãn thống nhất

Parse thời gian với hơn 6 format fallback

III. Luồng xử lý tổng thể


Load file

Detect dạng PTTT/CLS

Ánh xạ cột mặc định

Parse thời gian

Lọc theo ngày & khung giờ

Chuẩn hóa BS/BN/máy

Tính lane theo từng dịch vụ

Phát hiện xung đột

Sinh màu

Gom nhóm xung đột (cluster)

Tạo tooltip

Vẽ timeline

Sinh thống kê

Xuất CSV

IV. Logic tính lane (không chồng khối)
Thuật toán "interval scheduling lane assignment":

Với mỗi thủ thuật trong cùng một Procedure:

Sắp theo thời gian

Với mỗi record:

Gán vào lane sớm nhất có end_time <= start_time

Nếu không có lane phù hợp → tạo lane mới

→ Kết quả: mỗi thủ thuật được “đẩy xuống” 1 lane, không chồng lên nhau.

V. Thuật toán phát hiện xung đột
1. Kiểm tra overlap thời gian
overlap = (start1 < end2) AND (end1 > start2)

2. Loại xung đột

BS: Doctor_norm trùng

BN: Patient_norm trùng

M: Machine_norm trùng & != “KHÔNG DÙNG MÁY”

3. Quy tắc xác định ConflictTypes

Danh sách đầy đủ: [BS, BN, M]

Nếu rỗng → OK

Nếu 1 loại → conflict đơn

Nếu ≥2 loại → conflict đa

VI. Quy tắc tô màu
1. Không xung đột → Xanh lá (#2ECC71)
2. Xung đột 1 loại

BS → Đỏ (#FF4D4D)

BN → Vàng (#FFA500)

M → Tím (#9D4EDD)

3. Xung đột 2 loại → Xanh đậm (#0A4BFF)
4. Xung đột 3 loại → Nâu sáng (#C68E17)
VII. Gom nhóm xung đột (cluster)

Dùng graph theory:

Mỗi ca = 1 node

Có edge khi overlap + trùng bất kỳ yếu tố

Duyệt connected components để tạo cluster

“Dominant reason” = hợp lý (BS+BN+M) theo full ConflictTypes

VIII. Tooltip

Hiển thị:

Thời gian có giây

Tên dịch vụ

BS / BN / Máy

Danh sách đầy đủ loại xung đột (dịch sang TV)

IX. Thống kê máy

busy_minutes = tổng phút máy hoạt động

avg_gap_min = khoảng trống trung bình giữa hai ca

util_pct = phần trăm sử dụng trong khung giờ làm việc

downtime_pct = 100 - util_pct

X. Outputs

Giao diện timeline

Heatmap từng giờ

Báo cáo CSV

Danh sách cluster

Bộ lọc theo khoa/ngày/giờ
