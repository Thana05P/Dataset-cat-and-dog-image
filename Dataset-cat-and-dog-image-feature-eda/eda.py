import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
from glob import glob

# ==========================================
# 1. การจัดการ Path และโฟลเดอร์ผลลัพธ์
# ==========================================
current_path = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else current_path

possible_paths = [
    os.path.join(current_path, "Dataset-cat-and-dog-image", "Dataset-cat-and-dog-image", "train"),
    os.path.join(script_dir, "Dataset-cat-and-dog-image", "Dataset-cat-and-dog-image", "train"),
    os.path.join(current_path, "Dataset-cat-and-dog-image", "train"),
    os.path.join(script_dir, "Dataset-cat-and-dog-image", "train"),
    os.path.join(current_path, "train"),
    os.path.join(script_dir, "train"),
    os.path.join(current_path, "data", "train"),
    os.path.join(script_dir, "data", "train")
]

DATASET_DIR = None
for p in possible_paths:
    if os.path.exists(p):
        DATASET_DIR = p
        break

REPORTS_DIR = os.path.join(current_path, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
OUTLIERS_DIR = os.path.join(FIGURES_DIR, "outliers")
SUMMARY_FILE = os.path.join(REPORTS_DIR, "eda_summary.md")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTLIERS_DIR, exist_ok=True)

print(f"Current Working Directory: {current_path}")
if DATASET_DIR:
    print(f"✅ Target Dataset Path Found: {DATASET_DIR}")
else:
    print(f"❌ Error: ไม่พบโฟลเดอร์ train ในโครงสร้างปัจจุบัน กรุณานำโฟลเดอร์รูปภาพมาวาง")
print(f"Reports will be saved to: {REPORTS_DIR}\n")

# ==========================================
# 2. ฟังก์ชันตรวจสอบเชิงปริมาณ (Quantitative & Performance Optimized)
# ==========================================
def calculate_md5(image_path):
    """หาค่า Hash ตรวจจับรูปซ้ำ"""
    with open(image_path, "rb") as f:
        file_hash = hashlib.md5()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)
    return file_hash.hexdigest()

def extract_image_metadata(dataset_dir):
    """ดึง Metadata, Brightness, Contrast, Blur Score และตรวจไฟล์เสีย (Optimized Read)"""
    data = []
    
    if not dataset_dir or not os.path.exists(dataset_dir):
        print(f"Error: ไม่พบโฟลเดอร์ {dataset_dir}")
        return pd.DataFrame()
        
    classes = os.listdir(dataset_dir)
    
    for cls in classes:
        class_path = os.path.join(dataset_dir, cls)
        if not os.path.isdir(class_path): 
            continue
            
        for img_path in glob(os.path.join(class_path, "*.*")):
            file_size_kb = os.path.getsize(img_path) / 1024
            
            try:
                # อ่านไฟล์ด้วย OpenCV โดยตรง (ลด Overhead จากการเปิดด้วย PIL ซ้ำ)
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError("Cannot read image or file is corrupted")
                
                h, w, c = img.shape if len(img.shape) == 3 else (img.shape[0], img.shape[1], 1)
                aspect_ratio = w / h
                
                # แปลงเป็น Gray เพื่อคำนวณสถิติ
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if c == 3 else img
                
                # เช็ก Grayscale แบบ Fast Downsampling
                small_img = img[::10, ::10] if c == 3 else img
                is_grayscale = (c == 1) or (
                    np.array_equal(small_img[:,:,0], small_img[:,:,1]) and 
                    np.array_equal(small_img[:,:,1], small_img[:,:,2])
                )
                
                # คำนวณ Brightness, Contrast และ Blur Score
                brightness = float(np.mean(gray))
                contrast = float(np.std(gray))
                blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                img_hash = calculate_md5(img_path)
                
                data.append({
                    "path": img_path,
                    "class": cls,
                    "width": w,
                    "height": h,
                    "channels": c,
                    "aspect_ratio": aspect_ratio,
                    "size_kb": file_size_kb,
                    "is_grayscale": is_grayscale,
                    "brightness": brightness,
                    "contrast": contrast,
                    "blur_score": blur_score,
                    "hash": img_hash,
                    "corrupted": False
                })
                
            except Exception as e:
                data.append({
                    "path": img_path, 
                    "class": cls, 
                    "corrupted": True, 
                    "error": str(e)
                })
                
    return pd.DataFrame(data)

# ==========================================
# 3. ฟังก์ชัน Audit และสุ่มตรวจเชิงคุณภาพ (Qualitative)
# ==========================================
def save_outlier_samples(df, top_n=3):
    """ตรวจจับและเซฟรูป Outliers (เบลอที่สุด, มืดที่สุด, สว่างที่สุด) ออกมาตรวจแบบ Visual Audit"""
    df_valid = df[df["corrupted"] == False]
    if df_valid.empty:
        return

    outliers = {
        "Most_Blurry": df_valid.sort_values(by="blur_score").head(top_n),
        "Darkest": df_valid.sort_values(by="brightness").head(top_n),
        "Brightest": df_valid.sort_values(by="brightness", ascending=False).head(top_n)
    }

    for category_name, outlier_df in outliers.items():
        fig, axes = plt.subplots(1, len(outlier_df), figsize=(4 * len(outlier_df), 4))
        fig.suptitle(f"Outlier Audit: {category_name.replace('_', ' ')}", fontsize=14)
        
        if len(outlier_df) == 1:
            axes = [axes]

        for i, (_, row) in enumerate(outlier_df.iterrows()):
            img = cv2.imread(row["path"])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img_rgb)
            axes[i].set_title(
                f"Class: {row['class']}\n"
                f"Blur: {row['blur_score']:.1f} | Bright: {row['brightness']:.1f}"
            )
            axes[i].axis('off')

        plt.tight_layout()
        outlier_fig_path = os.path.join(OUTLIERS_DIR, f"outlier_{category_name.lower()}.png")
        plt.savefig(outlier_fig_path, dpi=200, bbox_inches='tight')
        plt.close()

def plot_and_save_samples(df, n_samples=3):
    """สุ่มตัวอย่างภาพและบันทึกกราฟ Histogram แยกราย Class"""
    classes = df["class"].unique()
    
    for cls in classes:
        sample_df = df[(df["class"] == cls) & (df["corrupted"] == False)]
        if sample_df.empty: 
            continue
            
        sample_df = sample_df.sample(min(n_samples, len(sample_df)))
        
        fig, axes = plt.subplots(n_samples, 2, figsize=(10, 3 * n_samples))
        fig.suptitle(f"Qualitative Inspection & Intensity Histogram: Class '{cls}'", fontsize=14)
        
        if n_samples == 1:
            axes = np.array([axes])
            
        for i, (_, row) in enumerate(sample_df.iterrows()):
            img = cv2.imread(row["path"])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i, 0].imshow(img_rgb)
            axes[i, 0].set_title(
                f"Blur: {row['blur_score']:.1f} | Brightness: {row['brightness']:.1f}"
            )
            axes[i, 0].axis('off')
            
            colors = ('r', 'g', 'b')
            for j, color in enumerate(colors):
                hist = cv2.calcHist([img_rgb], [j], None, [256], [0, 256])
                axes[i, 1].plot(hist, color=color)
                axes[i, 1].set_xlim([0, 256])
            axes[i, 1].set_title("Color Channel Distribution")
            
        plt.tight_layout()
        sample_fig_path = os.path.join(FIGURES_DIR, f"sample_class_{cls}.png")
        plt.savefig(sample_fig_path, dpi=200, bbox_inches='tight')
        plt.close()

# ==========================================
# 4. ฟังก์ชันเขียนรายงานสรุปผล (Markdown Report)
# ==========================================
def generate_summary_report(df):
    df_valid = df[df["corrupted"] == False]
    class_counts = df_valid['class'].value_counts().to_markdown()
    
    report_md = f"""# 📊 Exploratory Data Analysis (EDA) Summary Report

## 1. ข้อมูลเชิงปริมาณ (Quantitative Summary)
* **จำนวนรูปภาพทั้งหมด (Total Images):** {len(df)} ไฟล์
* **ไฟล์ที่ชำรุด/เปิดไม่ได้ (Corrupted):** {df['corrupted'].sum()} ไฟล์
* **รูปภาพที่มีความซ้ำซ้อน (Duplicate Hashes):** {df_valid['hash'].duplicated().sum()} รูป
* **รูป Grayscale ที่ปนใน RGB:** {df_valid['is_grayscale'].sum()} รูป

### สรุปจำนวนข้อมูลแยกตาม Class (Class Distribution)
{class_counts}

### สถิติขนาดภาพ ความสว่าง และคุณภาพเชิงลึก
| ตัวชี้วัด | ค่าเฉลี่ย (Mean) | ค่าน้อยสุด (Min) | ค่ามากสุด (Max) |
| :--- | :--- | :--- | :--- |
| **Width (px)** | {df_valid['width'].mean():.1f} | {df_valid['width'].min()} | {df_valid['width'].max()} |
| **Height (px)** | {df_valid['height'].mean():.1f} | {df_valid['height'].min()} | {df_valid['height'].max()} |
| **Aspect Ratio (W/H)** | {df_valid['aspect_ratio'].mean():.2f} | {df_valid['aspect_ratio'].min():.2f} | {df_valid['aspect_ratio'].max():.2f} |
| **File Size (KB)** | {df_valid['size_kb'].mean():.1f} | {df_valid['size_kb'].min():.1f} | {df_valid['size_kb'].max():.1f} |
| **Brightness (0-255)** | {df_valid['brightness'].mean():.1f} | {df_valid['brightness'].min():.1f} | {df_valid['brightness'].max():.1f} |
| **Contrast (Std Dev)** | {df_valid['contrast'].mean():.1f} | {df_valid['contrast'].min():.1f} | {df_valid['contrast'].max():.1f} |
| **Blur Score (Laplacian)** | {df_valid['blur_score'].mean():.1f} | {df_valid['blur_score'].min():.1f} | {df_valid['blur_score'].max():.1f} |

---

## 2. ข้อค้นพบเชิงคุณภาพและภาพแปลกปลอม (Qualitative & Outlier Observations)
* **ตัวอย่างรูปภาพและ Histogram:** บันทึกไว้ที่โฟลเดอร์ `reports/figures/`
* **ภาพ Outliers ที่ต้องตรวจสอบ (Visual Audit):** บันทึกรูปภาพกลุ่มเสี่ยง (เบลอมาก, มืดจัด, สว่างจัด) ไว้ที่ `reports/figures/outliers/`

---

## 3. สรุปผลกระทบต่อโมเดลและแนวทางแก้ไข (Insights & Actionable Plan)
| ปัญหาที่อาจพบ | ผลกระทบต่อโมเดล (Impact) | แนวทางแก้ไขก่อนเทรน (Action Plan) |
| :--- | :--- | :--- |
| **Class Imbalance** | โมเดลจะ Bias ไปทางคลาสที่มีจำนวนเยอะ | ใช้ Weighted Loss หรือ Class-balanced Augmentation |
| **Aspect Ratio ต่างกันมาก** | หาก Resize ดื้อๆ ภาพจะยืดหดจนเสียรูปทรง | ใช้ Letterbox Padding ก่อน Resize หรือใช้ Random Crop |
| **รูปซ้ำ (Duplicate Hashes)** | เกิด Data Leakage หากรูปซ้ำหลุดไปที่ Train และ Test | ทำ Deduplication ลบไฟล์ที่มี Hash ซ้ำออก |
| **ภาพเบลอ / มืดจัด / สว่างจัด** | Feature Extractor สับสน และเรียนรู้ Noise | กรองภาพออกด้วย Threshold (Blur Score, Brightness) |
"""
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(report_md)

# ==========================================
# 5. การรันการทำงานหลัก (Execution)
# ==========================================
if __name__ == "__main__":
    print("⏳ กำลังเริ่มวิเคราะห์ข้อมูลรูปภาพ...")
    
    if not DATASET_DIR:
        print("❌ ไม่พบโฟลเดอร์ train ในโครงสร้างปัจจุบัน กรุณาตรวจสอบตำแหน่งโฟลเดอร์รูปภาพ")
    else:
        df = extract_image_metadata(DATASET_DIR)
        
        if not df.empty:
            df_valid = df[df["corrupted"] == False]
            
            # 1. วาดกราฟสรุปเชิงปริมาณ (ขยายเป็น 6 กราฟ 2x3)
            plt.figure(figsize=(16, 10))
            
            plt.subplot(2, 3, 1)
            sns.countplot(data=df_valid, x="class")
            plt.title("Class Distribution")
            
            plt.subplot(2, 3, 2)
            sns.scatterplot(data=df_valid, x="width", y="height", hue="class", alpha=0.5)
            plt.title("Dimensions (Width vs Height)")
            
            plt.subplot(2, 3, 3)
            sns.histplot(data=df_valid, x="aspect_ratio", bins=30, kde=True)
            plt.title("Aspect Ratio Distribution")
            
            plt.subplot(2, 3, 4)
            sns.histplot(data=df_valid, x="size_kb", bins=30, kde=True)
            plt.title("File Size Distribution (KB)")
            
            plt.subplot(2, 3, 5)
            sns.histplot(data=df_valid, x="brightness", bins=30, kde=True, color="orange")
            plt.title("Brightness Distribution")
            
            plt.subplot(2, 3, 6)
            sns.histplot(data=df_valid, x="blur_score", bins=30, kde=True, color="purple")
            plt.title("Blur Score Distribution")
            
            plt.tight_layout()
            summary_plot_path = os.path.join(FIGURES_DIR, "01_eda_summary_plots.png")
            plt.savefig(summary_plot_path, dpi=200)
            plt.close()
            
            # 2. สุ่มวาดภาพตัวอย่างและ Histogram เชิงคุณภาพ
            print("🖼️ กำลังสุ่มตรวจคุณภาพรูปภาพและสร้าง Histogram...")
            plot_and_save_samples(df_valid, n_samples=3)
            
            # 3. คัดกรองและบันทึกรูป Outlier เพื่อการ Audit
            print("🔍 กำลังค้นหาและบันทึกภาพ Outliers...")
            save_outlier_samples(df_valid, top_n=3)
            
            # 4. สร้าง Markdown Report
            print("📝 กำลังเขียนรายงานสรุปผลลง eda_summary.md...")
            generate_summary_report(df)
            
            print("\n" + "="*40)
            print("🎉 การทำ EDA เสร็จสมบูรณ์แล้ว!")
            print(f"📁 ดูกราฟทั้งหมดได้ที่: {FIGURES_DIR}")
            print(f"🔍 ดูภาพ Outliers ได้ที่: {OUTLIERS_DIR}")
            print(f"📄 ดูรายงานสรุปผลได้ที่: {SUMMARY_FILE}")
            print("="*40)
        else:
            print("❌ ไม่พบข้อมูลรูปภาพ กรุณาตรวจสอบ Path โฟลเดอร์อีกครั้ง")