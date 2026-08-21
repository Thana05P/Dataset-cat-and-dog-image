Markdown# Image Dataset Cleaning & Preprocessing Pipeline

## 📝 1. ภาพรวมโปรเจกต์ (Project Description)
โปรเจกต์นี้คือไพพ์ไลน์ (Pipeline) แบบครบวงจรสำหรับการเตรียมข้อมูลรูปภาพ (Image Data Preparation) เพื่อนำไปใช้ฝึกสอนโมเดล Machine Learning/Deep Learning โดยครอบคลุมตั้งแต่กระบวนการดึงข้อมูลจาก Kaggle, การทำความเข้าใจข้อมูล (EDA), การทำความสะอาดและปรับแต่งภาพ (Preprocessing & Augmentation) ไปจนถึงการแบ่งสัดส่วนข้อมูล (Train/Val/Test Split) อย่างถูกต้องตามหลักการ เพื่อป้องกัน Data Leakage

## 📊 2. ที่มาของ Dataset และการตั้งค่า Kaggle API
**Dataset:** [Cats and Dogs image classification บน Kaggle](https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification/data)

**วิธี Setup Kaggle API Key:**
1. ไปที่เว็บไซต์ [Kaggle](https://www.kaggle.com/) และเข้าสู่ระบบ
2. ไปที่รูปโปรไฟล์มุมขวาบน เลือก **Settings** (หรือ Account)
3. เลื่อนลงมาที่หัวข้อ **API** แล้วกดปุ่ม **Create New Token**
4. ไฟล์ `kaggle.json` จะถูกดาวน์โหลดลงเครื่องอัตโนมัติ
5. นำไฟล์ `kaggle.json` ไปวางไว้ในโฟลเดอร์ตามระบบปฏิบัติการของคุณ:
   - **Windows:** `C:\Users\<username>\.kaggle\kaggle.json`
   - **Mac/Linux:** `~/.kaggle/kaggle.json`
6. *(เฉพาะ Mac/Linux)* ตั้งค่าสิทธิ์การเข้าถึงไฟล์เพื่อความปลอดภัยด้วยคำสั่ง:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json

## ⚙️ 3. วิธีการติดตั้งและรัน Code (Installation & Usage)วิธีติดตั้ง (Installation):โคลน Repository นี้ลงเครื่องของคุณ:Bashgit clone [https://github.com/](https://github.com/)<your-username>/<repo-name>.git
cd <repo-name>
ติดตั้งไลบรารีที่จำเป็นผ่าน requirements.txt:Bashpip install -r requirements.txt
วิธีรัน Code ทีละขั้นตอน:เพื่อให้ไพพ์ไลน์ทำงานสมบูรณ์ ให้รันสคริปต์ตามลำดับดังนี้:ดึงข้อมูล (Data Collection):Bashpython src/data_collection.py
สำรวจข้อมูล (EDA):Bashpython src/eda.py
ปรับแต่งและทำความสะอาดภาพ (Preprocessing):Bashpython src/preprocessing.py
หรือ python src/image_processing.py (ตามที่กำหนดในโค้ด)
แบ่งสัดส่วนข้อมูลและสร้าง Manifest (Data Split):Bashpython src/data_split.py

## 📂 4. โครงสร้างโฟลเดอร์ (Repository Structure)Plaintextrepo-name/
├── README.md               # เอกสารอธิบายโปรเจกต์ (ไฟล์นี้)
├── requirements.txt        # ไฟล์ระบุ dependencies/libraries ที่ต้องใช้
├── .gitignore              # ไฟล์ละเว้นการ push ข้อมูล/ไฟล์ขยะขึ้น GitHub
├── src/                    # โฟลเดอร์เก็บ Source Code หลัก
│   ├── data_collection.py  # สคริปต์ดึง Dataset จาก Kaggle
│   ├── eda.py              # สคริปต์ทำ Exploratory Data Analysis
│   ├── preprocessing.py    # สคริปต์ทำ Data Cleaning / ภาพรวม
│   ├── image_processing.py # สคริปต์จัดการรูปภาพ (resize, denoise, augment)
│   └── data_split.py       # สคริปต์แบ่ง Train/Val/Test Split
├── notebooks/              # โฟลเดอร์สำหรับ Jupyter Notebook (Interactive EDA)
├── reports/                # โฟลเดอร์เก็บรายงาน
│   ├── eda_summary.md      # ไฟล์สรุปผลการทำ EDA
│   └── figures/            # โฟลเดอร์เก็บกราฟและภาพสรุปผล (ไม่เก็บภาพ Dataset จริง)
└── slides/                 # โฟลเดอร์สำหรับเก็บไฟล์ Slide นำเสนอ หรือ Export PDF
## 👥 5. รายชื่อสมาชิกกลุ่มและหน้าที่รับผิดชอบ รหัสนักศึกษาชื่อ-นามสกุล Branch ที่รับผิดชอบขอบเขตงาน (Role)

นายธนวัฒน์ ริ้วผดุงพันธ์ 077 feature/collect/splitคนที่ 1 & 4:
- Data Collection: เขียน Script ดึง Dataset จาก Kaggle API + จัดโครงสร้างโฟลเดอร์ข้อมูล
- Data Split: ทำ Train/Val/Test Split ตามหลักการ (Stratified, กัน Data Leak) + สร้าง Report สรุปทั้งหมด (README/Slide/ดูแล Repo)
นายธนาวัช ภักดี 078 feature/preprocessingคนที่ 3:
- Preprocessing: ทำ Data Cleaning, Image Processing (เช่น Resize ให้เท่ากัน, Denoise ลดสัญญาณรบกวน, Data Augmentation เพื่อเพิ่มความหลากหลาย)
นายนนทพันธ์ สุขกำนิด 079 feature/eda คนที่ 2:
- EDA: ทำ EDA เชิงปริมาณและเชิงคุณภาพ วิเคราะห์การกระจายตัวของคลาส ขนาดภาพ พร้อมเขียนสรุปผล
