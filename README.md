# yolo_v8
Object detection model with yolo



Link Document 

https://yolo-docs.readthedocs.io/en/latest/

https://www.youtube.com/watch?v=PI7Osnbhzo4

## 🧠 YOLO v1 Architecture (2016)

YOLO v1 (You Only Look Once) được giới thiệu năm **2016** bởi **Joseph Redmon**.  
Đây là mô hình phát hiện vật thể **một giai đoạn (single-stage detector)**,  
chỉ cần **một mạng duy nhất** để xác định vị trí và phân loại đối tượng trong ảnh.

### ⚙️ Kiến trúc

- Gồm **24 lớp Convolutional** xen kẽ **Max Pooling** để trích xuất đặc trưng.
- Tiếp theo là **2 lớp Fully Connected** để dự đoán bounding boxes và classes.
- Ảnh đầu vào được resize thành **448×448 pixels**.
- Ảnh được chia thành **lưới 7×7 (Grid cells)**.
- Mỗi cell dự đoán:
  - **B = 2 bounding boxes** (tọa độ x, y, w, h)
  - **Confidence score**
  - **Class probabilities**

> **Tóm lại:**  
> YOLO v1 = Convolution + Max Pooling + Fully Connected → Grid 7×7 vị trí vật thể.

---

### ✅ Ưu điểm

- **Tốc độ rất nhanh:** chỉ cần một lần forward qua mạng (45 FPS, Fast YOLO đạt 155 FPS).  
- **End-to-End Training:** học trực tiếp từ ảnh đến kết quả detection.  
- **Nhận biết ngữ cảnh toàn ảnh tốt**, ít nhầm lẫn vật thể với nền.

---

### ⚠️ Nhược điểm

- **Khó phát hiện vật thể nhỏ hoặc gần nhau**, do chia lưới 7×7 cố định.  
- **Độ chính xác định vị (localization) thấp** hơn Faster R-CNN.  
- **Giới hạn số lượng vật thể mỗi cell (tối đa 2)**.  
- **Kém linh hoạt** với vật thể có kích thước và tỉ lệ khác biệt lớn.

---
  
<img width="1880" height="833" alt="image" src="https://github.com/user-attachments/assets/d5d7161c-1e58-4ac6-99b2-ee7c6e273f87" />
