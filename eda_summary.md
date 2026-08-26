# 📊 Exploratory Data Analysis (EDA) Summary Report

## 1. ข้อมูลเชิงปริมาณ (Quantitative Summary)
* **จำนวนรูปภาพทั้งหมด (Total Images):** 530 ไฟล์
* **ไฟล์ที่ชำรุด/เปิดไม่ได้ (Corrupted):** 0 ไฟล์
* **รูปภาพที่มีความซ้ำซ้อน (Duplicate Hashes):** 0 รูป
* **รูป Grayscale ที่ปนใน RGB:** 3 รูป

### สรุปจำนวนข้อมูลแยกตาม Class (Class Distribution)
| class   |   count |
|:--------|--------:|
| cats    |     267 |
| dogs    |     263 |

### สถิติขนาดภาพและความสว่าง
| ตัวชี้วัด | ค่าเฉลี่ย (Mean) | ค่าน้อยสุด (Min) | ค่ามากสุด (Max) |
| :--- | :--- | :--- | :--- |
| **Width (px)** | 961.1 | 133 | 4272 |
| **Height (px)** | 664.2 | 133 | 4272 |
| **Aspect Ratio (W/H)** | 1.52 | 0.57 | 3.59 |
| **File Size (KB)** | 96.7 | 2.9 | 1697.6 |
| **Blur Score (Laplacian)** | 854.1 | 6.5 | 10569.0 |

---

## 2. ข้อค้นพบเชิงคุณภาพ (Qualitative Observations)
* **ตัวอย่างรูปภาพและ Histogram:** บันทึกไว้ที่โฟลเดอร์ `reports/figures/`
* **การตรวจสอบด้วยสายตา (Visual Check):**
  * ควรเปิดดูรูปภาพใน `sample_class_*.png` เพื่อตรวจว่ามีลายน้ำ (Watermark), ป้ายข้อความ, มุมกล้องที่กลับหัว หรือภาพเบลอจนมองไม่เห็นวัตถุหรือไม่

---

## 3. สรุปผลกระทบต่อโมเดลและแนวทางแก้ไข (Insights & Actionable Plan)
| ปัญหาที่อาจพบ | ผลกระทบต่อโมเดล (Impact) | แนวทางแก้ไขก่อนเทรน (Action Plan) |
| :--- | :--- | :--- |
| **Class Imbalance** | โมเดลจะ Bias ไปทางคลาสที่มีจำนวนเยอะ | ใช้ Weighted Loss, Data Augmentation |
| **Aspect Ratio ต่างกันมาก** | หาก Resize ดื้อๆ ภาพจะยืดหดจนเสียรูปทรง | ใช้ Padding (Letterbox) ก่อน Resize หรือทำ Random Cropping |
| **รูปซ้ำ (Duplicate Hashes)** | เกิด Data Leakage หากรูปซ้ำหลุดไปที่ Train และ Test | ทำ Data Deduplication โดยลบไฟล์ที่มี Hash ซ้ำออก |
| **ภาพเบลอ / นอยส์เยอะ** | Feature Extractor สับสนกับขอบภาพ | กำหนด Threshold ค่า Blur Score เพื่อคัดกรองภาพก่อนเทรน |
