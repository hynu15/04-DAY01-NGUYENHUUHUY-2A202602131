# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

Ngày chạy:

Runtime Colab: CPU/GPU

Python / PyTorch / Ultralytics:

Checkpoint: yolo11n-cls.pt, yolo11n.pt, yolo11n-seg.pt

Thay đổi so với notebook nguồn: [...]

> ZIP do notebook tạo có tên <KHOA>-DAY01-report.zip (ví dụ: K4-DAY01-report.zip). Giải nén rồi đặt trực tiếp REPORT.md và day1_lab_outputs/ vào thư mục report/ của repository tạo từ template. Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác định người nộp.

## Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: classification_predictions.json, sample traffic.

- Record hạng 1 (class_id, class_name, rank, score, taxonomy_name):
  rank: 1, class_id: 468, class_name: "cab", score: 0.510915, taxonomy_name: "ImageNet-1K"
- Record này mô tả toàn ảnh như thế nào?
  Toàn ảnh là các phương tiện giao thông.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
  Người hoặc nhóm nghiên cứu tạo ra tập dữ liệu huấn luyện (ví dụ nhóm phát triển tập dữ liệu ImageNet).
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
  Định danh ID để máy tính xử lý, tên lớp để con người đọc hiểu, và tên taxonomy để biết phân loại đó thuộc chuẩn từ điển nào (tránh xung đột dữ liệu khi trộn nhiều tập dữ liệu khác nhau).
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
  Cần quy định tiêu chí ưu tiên để chọn ra một chủ thể đại diện duy nhất (ví dụ dựa vào diện tích chiếm chỗ lớn nhất hoặc đối tượng nằm ở vị trí trung tâm).
- Vì sao model score không phải ground truth?
  Điểm số (score) chỉ là mức độ tự tin mang tính xác suất của mô hình, trong khi ground truth là sự thật tuyệt đối do con người (annotator) gắn nhãn và xác nhận.

## Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: detection_predictions.json và visuals/detection_predictions.png, sample kitchen.

- Một record (class_name, score, bbox_xyxy, bbox_width, bbox_height):
  "class_name": "bus",
    "score": 0.912558,
    "bbox_xyxy": [
      93.17,
      187.95,
      223.01,
      320.91
    ],
    "bbox_width": 129.84,
    "bbox_height": 132.96
  },
- Diễn giải vị trí box bằng lời:
  bao quanh ra phía bên ngoài vật thể xác định được.
- So sánh số prediction ở hai threshold:
  Khi threshold tăng lên thì vật thể xác định được cũng ít đi vì ngưỡng xác định là cao hơn, sẽ xác định các vật thể chuẩn xác hơn chứ không bắt nhầm nhiều như khi để threshold thấp.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
  Khi hạ thấp threshold, độ bao phủ tăng lên do mô hình trả về nhiều hộp (box) hơn, dẫn đến khối lượng công việc của reviewer tăng mạnh vì phải kiểm tra và loại bỏ nhiều hộp dự đoán sai (false positives).
- Đề xuất một quy tắc box chặt:
  Đường viền của hộp dự đoán (bounding box) phải bao trọn toàn bộ phần nhìn thấy được của vật thể, khoảng cách từ mép ngoài cùng của vật thể đến cạnh của hộp không được vượt quá giới hạn sai số cho phép (ví dụ 2 đến 3 pixel).
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
  Cần quy định ngưỡng che khuất tối đa để quyết định dán nhãn hay bỏ qua (ví dụ che khuất trên 50% thì bỏ qua), và quy định cách vẽ hộp (vẽ bao gồm cả phần bị che hay chỉ vẽ sát phần có thể nhìn thấy).

## Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: segmentation_predictions.json và visuals/segmentation_prediction.png, sample kitchen.

- Một record (instance_id, class_name, score, số điểm và một phần polygon_xy):
   "instance_id": "traffic-001",
    "class_id": 5,
    "class_name": "bus",
    "score": 0.925745,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      95.3,
      188.72,
      224.08,
      319.96
    ],
    "polygon_point_count": 120,
    "polygon_xy": [
      [
        148.0,
        189.0
      ],
      [
        147.0,
        190.0
      ],
- Polygon bổ sung chi tiết gì so với box?
  Cung cấp ranh giới hình học chính xác của vật thể ở cấp độ điểm ảnh (pixel), giúp loại bỏ hoàn toàn các phần diện tích thuộc về nền (background) hoặc vật thể khác mà bounding box vô tình bao gồm.
- instance_id dùng để làm gì và không phải loại ID nào?
  Được dùng để phân biệt các cá thể độc lập của cùng một loại vật thể trong bức ảnh (ví dụ cái cốc số 1, cái cốc số 2). Nó không dùng để định danh loại đối tượng (không phải class_id).
- Đề xuất một quy tắc biên mask:
  Đường đa giác (polygon) phải bám sát chính xác vào viền thực tế của đối tượng, không được lẹm ra ngoài nền và không được bỏ sót các phần diện tích nhỏ nối liền của vật thể.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
  Cần quy định cách xác định đường ranh giới chia cắt giữa hai vật thể đặt sát nhau, và quyết định có bao gồm vùng rìa bị mờ do chuyển động (motion blur) vào trong mặt nạ (mask) hay không.

## Vòng đời và kiểm tra chất lượng

ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Định danh class_id cấp độ toàn bộ ảnh | Ảnh chứa nhiều chủ thể cùng lúc, gây khó khăn khi chọn một nhãn duy nhất | Lựa chọn một nhãn đại diện chính xác nhất tuân theo quy tắc ưu tiên của guideline | Kiểm tra xem nhãn được chọn có phản ánh đúng đối tượng quan trọng nhất trong ảnh theo chuẩn quy định chưa |
| Phát hiện vật thể | Tọa độ giới hạn hộp bounding box và định danh class_id | Hộp dự đoán quá rộng, quá hẹp hoặc bao quanh gộp nhiều chủ thể cùng lúc | Điều chỉnh hoặc vẽ lại hộp sao cho ôm sát nhất vào các cạnh ngoài cùng của vật thể | Kiểm tra độ khít của hộp và tính chính xác của nhãn phân loại đi kèm |
| Instance segmentation | Tập hợp tọa độ điểm kèm định danh class_id và instance_id | Đường viền lẹm vào nền hoặc dính liền hai cá thể khác nhau thành một vùng | Đánh dấu chi tiết đường biên để bóc tách chính xác từng cá thể vật thể khỏi nền | Kiểm tra độ sắc nét, chính xác của đường ranh giới và đảm bảo các cá thể tách biệt không bị gộp chung |

## An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
  Không sao chép, chia sẻ, hoặc lưu trữ dữ liệu của dự án ra các thiết bị cá nhân hoặc dịch vụ đám mây bên ngoài không được cấp phép.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
  Quản lý dự án, bộ phận hỗ trợ kỹ thuật hoặc cá nhân phụ trách an toàn bảo mật dữ liệu của tổ chức.

## Danh sách bằng chứng

- [ ] classification_predictions.json
- [ ] detection_predictions.json
- [ ] segmentation_predictions.json
- [ ] IMAGE_ATTRIBUTION.md
- [ ] visuals/classification_top5.png
- [ ] visuals/detection_predictions.png
- [ ] visuals/segmentation_prediction.png
- [ ] Ô validation cuối notebook báo PASS.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
