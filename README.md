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

## 🧠 YOLO v3 Architecture (2018)

YOLO v3 được giới thiệu năm **2018** bởi **Joseph Redmon và Ali Farhadi**.  
Đây là phiên bản cải tiến mạnh mẽ so với YOLO v1/v2,  
tập trung nâng cao **độ chính xác**, đặc biệt với **vật thể nhỏ**,  
nhưng vẫn giữ được **tốc độ cao** – phù hợp cho ứng dụng real-time.

---

### ⚙️ Kiến trúc tổng quan

YOLO v3 sử dụng backbone **Darknet-53**, gồm:
- **53 lớp Convolutional** (thay vì 24 lớp như YOLO v1)  
- Không còn dùng **Fully Connected**, toàn bộ là CNN thuần túy.
- Sử dụng **Residual Connections** (giống ResNet) để tránh mất mát thông tin khi mạng sâu hơn.
- **Batch Normalization** và **Leaky ReLU** được dùng để ổn định quá trình học.

Ảnh đầu vào được chia thành **lưới (grid)**, nhưng YOLO v3 dự đoán ở **3 cấp độ độ phân giải khác nhau**:
1. **Scale 1:** 13×13 – phát hiện vật thể lớn  
2. **Scale 2:** 26×26 – phát hiện vật thể trung bình  
3. **Scale 3:** 52×52 – phát hiện vật thể nhỏ  

Mỗi cell trong mỗi scale sẽ dự đoán:
- **3 bounding boxes (anchor boxes)**  
- Với mỗi box, mô hình xuất ra **(x, y, w, h, objectness score, class scores)**  

---

### 🎯 Objectness Score & Class Prediction

- **Objectness Score**: cho biết mức độ tin cậy rằng bounding box chứa vật thể.  
- **Class Prediction:** YOLO v3 thay softmax bằng **sigmoid activation**  
  → cho phép mô hình dự đoán **đa lớp (multi-label)** (một vật thể có thể thuộc nhiều lớp).

📘 **Công thức đầu ra tổng quát:**

\[
\text{Output} = S \times S \times (B \times (5 + C))
\]

Trong đó:
- `S`: Kích thước grid (13, 26, 52)  
- `B`: Số anchor boxes (3 mỗi scale)  
- `5`: (x, y, w, h, objectness)  
- `C`: Số lớp (classes)

---

### 🧩 Ưu điểm

✅ **Phát hiện vật thể nhỏ tốt hơn:**  
→ Nhờ cơ chế multi-scale (13×13, 26×26, 52×52).  

✅ **Chính xác hơn YOLO v1/v2:**  
→ Do backbone Darknet-53 mạnh mẽ và có residual connections.

✅ **Không dùng fully connected:**  
→ Giảm tham số, tốc độ nhanh, dễ huấn luyện.  

✅ **Hỗ trợ multi-label classification:**  
→ Một vật thể có thể thuộc nhiều lớp cùng lúc.

---

### ⚠️ Nhược điểm

❌ **Kích thước mô hình lớn hơn**, tốc độ giảm nhẹ so với YOLO v2.  
❌ **Không dùng FPN đầy đủ** (Feature Pyramid Network), nên với vật thể cực nhỏ vẫn chưa tối ưu.  
❌ **Huấn luyện phức tạp hơn**, cần chọn anchor boxes phù hợp.

---



<img width="1753" height="850" alt="image" src="https://github.com/user-attachments/assets/c551fc32-6e3c-4a3d-938e-b88ef74de0be" />



## 🧠 YOLO v5 Architecture (2020)

YOLO v5 được phát hành năm **2020** bởi **Ultralytics** (không phải nhóm gốc của Joseph Redmon).  
Đây là phiên bản YOLO phổ biến nhất trên GitHub nhờ:
- Dễ huấn luyện, dễ triển khai (Python + PyTorch)
- Hiệu năng cao
- Tương thích với nhiều thiết bị (GPU, CPU, Edge)

---

### ⚙️ Kiến trúc tổng quan

YOLO v5 vẫn giữ triết lý “You Only Look Once”,  
nhưng được **viết lại hoàn toàn bằng PyTorch**,  
và cải tiến ở 3 phần chính: **Backbone – Neck – Head**.

---

#### 🧱 1. Backbone: CSPDarknet53
- Dựa trên Darknet-53 của YOLOv3, nhưng thêm **Cross Stage Partial Connections (CSP)**  
  → giúp giảm số lượng tham số mà vẫn giữ được độ chính xác.  
- Sử dụng **Focus layer** để giảm kích thước ảnh đầu vào mà vẫn giữ thông tin quan trọng.  
- Kết hợp **Conv + BatchNorm + SiLU (Swish)** activation cho hiệu năng cao.

---

#### 🪜 2. Neck: PANet (Path Aggregation Network)
- Truyền đặc trưng từ nhiều tầng của backbone.  
- Tăng khả năng nhận diện vật thể ở nhiều kích thước (multi-scale feature fusion).  
- Sử dụng **FPN + PAN** giúp kết hợp đặc trưng từ thấp → cao và ngược lại.

---

#### 🧩 3. Head: YOLO Layer
- Mỗi head dự đoán ở **3 tỉ lệ (scales)**:  
  - 13×13 → vật thể lớn  
  - 26×26 → vật thể trung bình  
  - 52×52 → vật thể nhỏ  
- Mỗi ô lưới (grid cell) dự đoán:
  - (x, y, w, h)
  - Objectness score
  - Class probabilities  
- Hàm kích hoạt: **sigmoid** (cho objectness và class score)

---

### 🧮 Công thức đầu ra

\[
\text{Output} = S \times S \times (3 \times (5 + C))
\]
Trong đó:
- `S`: Kích thước grid (13, 26, 52)  
- `3`: Số anchor boxes mỗi scale  
- `5`: (x, y, w, h, objectness)  
- `C`: Số lớp cần phát hiện

---

### ⚡ Các phiên bản của YOLOv5
Ultralytics cung cấp nhiều kích thước mô hình tùy theo nhu cầu:
| Phiên bản | Đặc điểm | Tốc độ (FPS) | Độ chính xác (mAP) |
|------------|-----------|---------------|--------------------|
| YOLOv5n (Nano) | Nhỏ gọn, dùng cho thiết bị edge | 🚀 Rất nhanh | Trung bình |
| YOLOv5s (Small) | Cân bằng tốc độ và độ chính xác | Nhanh | Tốt |
| YOLOv5m (Medium) | Trung bình | Trung bình | Cao |
| YOLOv5l (Large) | Mạnh hơn | Trung bình | Rất cao |
| YOLOv5x (X-Large) | Mô hình lớn nhất | Chậm hơn | 🔥 Chính xác nhất |

---

### ✅ Ưu điểm

- **Hiệu năng cao, dễ dùng:** Huấn luyện bằng PyTorch chỉ với vài dòng lệnh.  
- **Triển khai dễ dàng:** Hỗ trợ ONNX, TensorRT, CoreML, OpenVINO, TFLite.  
- **Cấu trúc linh hoạt:** Tách rõ backbone – neck – head.  
- **Phát hiện vật thể nhỏ tốt hơn** nhờ PANet.  
- **Tốc độ nhanh, độ chính xác cao** hơn YOLOv3/YOLOv4.  

---

### ⚠️ Nhược điểm

- **Không phải phiên bản chính thức** từ tác giả YOLO gốc.  
- **Cần GPU để huấn luyện hiệu quả.**  
- **Tài nguyên lớn hơn YOLOv3**, đặc biệt bản `l` và `x`.  

---

<img width="1768" height="911" alt="image" src="https://github.com/user-attachments/assets/ff919800-7069-423a-9a39-00898f72f438" />


## 🧠 YOLO v7 Architecture (2022)

YOLO v7 được giới thiệu năm **2022** bởi **Chien-Yao Wang và cộng sự** (nhóm nghiên cứu Vision).  
Phiên bản này tập trung **tối ưu tốc độ và độ chính xác**, đồng thời **giảm tài nguyên tính toán** so với YOLOv5 và YOLOv6.

---

### ⚙️ Kiến trúc tổng quan

YOLO v7 tiếp tục sử dụng triết lý **Backbone – Neck – Head**,  
nhưng có những cải tiến nổi bật:

#### 🧱 1. Backbone: E-ELAN
- **Extended Efficient Layer Aggregation Network (E-ELAN)**.  
- Giúp mạng **học được nhiều đặc trưng hơn** mà không tăng độ sâu hay tham số.  
- Kết hợp các **Residual + Concatenation Blocks** để giữ thông tin chi tiết.

#### 🪜 2. Neck: PAN + BiFPN
- Sử dụng **Path Aggregation Network (PAN)** kết hợp **Bi-directional Feature Pyramid Network (BiFPN)**  
- Cho phép **fusion đặc trưng multi-scale tốt hơn** → phát hiện vật thể nhỏ và trung bình mạnh mẽ.

#### 🧩 3. Head: YOLO Detection Layer
- Multi-scale detection giống YOLO v5: **3 tỉ lệ (13×13, 26×26, 52×52)**  
- Objectness score + Class probabilities dùng **Sigmoid activation**  
- Hỗ trợ **multi-label classification**  

---

### ⚡ Các điểm nổi bật của YOLO v7

1. **Trainable bag-of-freebies (BoF)**: tăng độ chính xác mà không tăng inference cost.  
2. **Trainable bag-of-specials (BoS)**: tối ưu backbone, head và các layer convolution.  
3. **Nhẹ hơn và nhanh hơn YOLOv5/YOLOv6** nhưng **độ chính xác cao hơn**.  
4. **Hỗ trợ các biến thể**: YOLOv7-tiny, YOLOv7-x, YOLOv7-w6… để cân bằng tốc độ và chính xác.

---

### 🧩 Ưu điểm

✅ **Chính xác hơn YOLOv5** trên benchmark COCO.  
✅ **Tốc độ inference nhanh**, vẫn suitable cho real-time.  
✅ **Học đặc trưng hiệu quả** nhờ E-ELAN và BiFPN.  
✅ **Hỗ trợ multi-label**, phát hiện vật thể nhỏ tốt.  

---

### ⚠️ Nhược điểm

❌ **Phức tạp hơn YOLOv5** → cài đặt và huấn luyện khó hơn.  
❌ **Yêu cầu GPU tốt** để huấn luyện và deploy hiệu quả.  
❌ **Chưa phổ biến bằng YOLOv5**, ít tài liệu, community nhỏ hơn.

---


<img width="1513" height="869" alt="image" src="https://github.com/user-attachments/assets/8e5b4930-6857-48d0-8b67-894e3e43a88d" />



### Thành Tựu và Ứng Dụng

<img width="984" height="524" alt="image" src="https://github.com/user-attachments/assets/000f1757-de62-4696-9a85-45412b47c0cd" />


