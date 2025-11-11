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
 
## 🎯 Confidence Score & Class Probabilities trong YOLO v1

Trong YOLO v1, mỗi **grid cell** không chỉ dự đoán vị trí của vật thể  
mà còn đưa ra hai thông tin quan trọng để xác định **độ chắc chắn và loại vật thể**.

---

### 🧠 1. Confidence Score

**Confidence Score** biểu thị mức độ tin cậy của bounding box mà mô hình dự đoán.

Công thức:
\[
\text{Confidence} = P(\text{Object}) \times IOU_{\text{pred,true}}
\]

Trong đó:
- `P(Object)`: Xác suất có vật thể trong cell đó.  
- `IOU_pred,true`: Mức độ trùng khớp (Intersection over Union) giữa box dự đoán và box thật.

✅ **Ý nghĩa:**
- Nếu **không có vật thể** → `P(Object) = 0` → Confidence = 0.  
- Nếu **có vật thể và box khớp tốt** → Confidence cao (gần 1).

📘 **Ví dụ:**
| Thông số | Giá trị |
|-----------|----------|
| P(Object) | 0.9 |
| IOU | 0.8 |
| 👉 Confidence | 0.9 × 0.8 = **0.72** |

---

### 🧩 2. Class Probabilities

Mỗi grid cell dự đoán xác suất vật thể đó thuộc **từng lớp (class)**:
\[
P(\text{class}_i | \text{Object})
\]

📘 **Ví dụ:**
| Lớp | Xác suất |
|------|-----------|
| Person | 0.8 |
| Car | 0.1 |
| Dog | 0.1 |

→ Nghĩa là: nếu có vật thể, **80% khả năng đó là “Person”**.

---

### 🔗 3. Kết hợp để ra kết quả cuối cùng

YOLO nhân hai phần này để ra **điểm số dự đoán cuối cùng cho từng lớp:**

\[
P(\text{class}_i) = P(\text{Object}) \times IOU_{\text{pred,true}} \times P(\text{class}_i | \text{Object})
\]

📘 **Ví dụ tổng hợp:**
| Thành phần | Giá trị |
|-------------|----------|
| P(Object) | 0.9 |
| IOU | 0.8 |
| P(Person|Object) | 0.8 |
| → Confidence | 0.72 |
| → Final Score (Person) | 0.72 × 0.8 = **0.576** |

➡️ Box này có **57.6% khả năng là “Person”**, và sẽ được hiển thị nếu vượt ngưỡng (threshold).

---

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
