# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU (CUDA khả dụng)

**Python / PyTorch / Ultralytics:** Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không ghi nhận thay đổi; sử dụng đúng checkpoint, phiên bản Ultralytics và ngưỡng mặc định của notebook.

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=`468` ; `class_name`=`cab` ; `rank`=1 ; `score`=`0.510915` ; `taxonomy_name`=`ImageNet-1K`
2. Record này nói về **cả bức ảnh** (không phải một vật thể riêng) vì mô hình `yolo11n-cls.pt` thực hiện image classification và trả về danh sách các lớp xếp hạng cho toàn bộ bức ảnh, không cung cấp bounding box để định vị từng vật thể.
3. Danh sách tên lớp (class list) do **taxonomy đi kèm checkpoint/dữ liệu huấn luyện** quy định (ImageNet-1K); model không tự nghĩ ra tên lớp mới khi dự đoán.
4. Cần ghi cả ba trường vì:
   - `class_id`: mã số định danh lớp, thuận tiện cho máy xử lý và đối chiếu.
   - `class_name`: tên dễ đọc để con người hiểu kết quả.
   - `taxonomy_name`: cho biết mã và tên lớp thuộc bộ danh mục nào, tránh nhầm lẫn giữa các taxonomy khác nhau.
5. Guideline cần quy định rõ tiêu chí chọn lớp đại diện cho toàn ảnh (ví dụ: ưu tiên chủ thể chính/nổi bật nhất hay chủ đề tổng thể), cách xử lý khi có nhiều vật thể hoặc nhiều chủ đề, và quy trình xử lý (đưa đi review/escalation) khi không đủ căn cứ để chọn một lớp.
6. Không được lấy `score` làm ground truth vì score chỉ thể hiện mức độ mô hình tự tin với dự đoán chứ không chứng minh dự đoán đúng theo thực tế hay guideline; ground truth phải do annotator xác định.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record mẫu từ kết quả `kitchen` trong JSON: `class_name`=`cup/bowl (hoặc vật thể thực tế theo JSON)` ; `score`=`[Giá trị score từ file JSON]` ; `bbox_xyxy`=`[x_min, y_min, x_max, y_max]` ; `bbox_width`=`[Chiều rộng pixel]` ; `bbox_height`=`[Chiều cao pixel]`
2. Giải thích box bằng lời thường: `bbox_xyxy = [x_min, y_min, x_max, y_max]` trong đó `(x_min, y_min)` là góc trên-trái, `(x_max, y_max)` là góc dưới-phải tính theo pixel; chiều rộng bằng `x_max - x_min` và chiều cao bằng `y_max - y_min`.
3. Số lượng prediction theo các ngưỡng (`0.20`, `0.35`, `0.60`): Lấy trực tiếp từ 3 dòng log in ra ở ô so sánh threshold trong notebook của ảnh kitchen.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), reviewer phải kiểm tra thêm các hộp nhiễu, hộp trùng, phân loại sai, vật thể bị bỏ sót và độ chính xác của vị trí box để quyết định giữ, sửa hoặc loại bỏ prediction theo guideline.
5. Quy tắc box đề xuất: Vẽ một bounding box bao sát phần nhìn thấy của từng vật thể thuộc lớp mục tiêu sao cho các cạnh chạm sát biên ngoài của vật thể, không cố đoán phần bị che khuất và không lấy quá nhiều nền xung quanh.
6. Trường hợp trên ảnh `kitchen` có vật bị che khuất hoặc cắt mép: Cần đối chiếu trực tiếp `detection_predictions.png` và ảnh gốc; nếu guideline chưa quy định rõ hoặc không xác định chắc chắn ranh giới/lớp, annotator cần **escalation lên Lab Coach** thay vì tự suy diễn.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record mẫu từ JSON segmentation của ảnh `kitchen`: `instance_id`=`kitchen-001` ; `class_name`=`[Tên lớp trong JSON]` ; `score`=`[Điểm số score]` ; số điểm polygon=`polygon_point_count`=`[Số lượng điểm]` ; 3–5 điểm đầu của `polygon_xy`=`[[x1, y1], [x2, y2], [x3, y3], ...]`
2. So với box chữ nhật, polygon chứa các điểm tọa độ bám theo đường biên thực tế của vật thể, giúp thể hiện chính xác hình dạng, phần lồi/lõm và vùng chiếm chỗ thay vì khoanh vùng chữ nhật chứa nhiều pixel nền.
3. `instance_id` dùng làm mã định danh riêng cho từng đối tượng cụ thể trong output (ví dụ: `kitchen-001`) để phân biệt các instance ngay cả khi cùng class_name; nó không phải `class_id` (mã loại lớp) cũng không phải tracking ID (mã theo dõi qua các khung hình/video).
4. Quy tắc vẽ biên mask đề xuất: Vẽ polygon theo đường biên của phần vật thể nhìn thấy, đặt các điểm đủ sát để mô tả hình dạng nhưng không lấn sang nền hoặc vật thể bên cạnh; mỗi vật thể là một instance riêng biệt kể cả khi cùng lớp.
5. Với chỗ biên mờ, vật chạm nhau hay bị che khuất, nếu guideline chưa quy định rõ cách xử lý và không thể xác định ranh giới khách quan, annotator cần thực hiện **escalation** lên Lab Coach chứ không tự nối hai vật thành một mask hay tự đoán phần bị che.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho toàn bộ ảnh, gắn với taxonomy đã thống nhất. | Ảnh có nhiều vật thể/chủ đề; dễ chọn lớp không đại diện (như ảnh traffic top-1 là cab với score 0.510915 nhưng chỉ là prediction). | Đọc toàn ảnh, áp dụng tiêu chí chọn lớp trong guideline; không dùng score làm nhãn chuẩn. | Kiểm tra lớp có phù hợp nội dung ảnh và đúng taxonomy/guideline; xem xét các trường hợp nhiều chủ đề hoặc không rõ lớp. |
| Phát hiện vật thể | Mỗi vật thể mục tiêu có một class label và một bounding box xyxy theo pixel. | Box có thể quá rộng/hẹp, bỏ sót vật thể, nhận nhầm lớp, hoặc khó xác định biên khi vật bị che/cắt mép. | Vẽ box sát phần nhìn thấy theo quy tắc; gán đúng lớp; đưa trường hợp không rõ ranh giới lên escalation. | Kiểm tra class, số lượng vật thể, box có bao sát vật thể không, có bỏ sót/trùng lặp hoặc box sai không. |
| Instance segmentation | Mỗi instance có class label và một polygon/mask riêng theo từng pixel/đường biên. | Biên mờ, vật thể chạm nhau, bị che khuất; polygon có thể lấn nền hoặc thiếu vùng vật thể. | Vẽ polygon theo biên phần nhìn thấy; tách riêng từng instance, kể cả cùng lớp; escalation khi không rõ biên. | Kiểm tra mask có bám biên, đúng instance, không nhập nhằng hai vật thể và không bỏ sót vùng nhìn thấy không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là Tuyệt đối không đưa thông tin định danh cá nhân (họ tên, MSSV, email, số điện thoại) vào báo cáo hay tệp output cấu trúc, chỉ đặt tên trong repository.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cáo trực tiếp cho Lab Coach hoặc thầy cô.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

